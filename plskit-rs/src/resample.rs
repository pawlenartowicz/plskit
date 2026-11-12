//! Resampling engines used by signal_test, sequential, and find_k.
//! Crate-internal — public surface is the callers.

use faer::{Col, Mat, MatMut, MatRef};
use rand::seq::SliceRandom;

use crate::rng::{Rng, child_rng, child_seeds};

/// Compute `(n_train, n_test)` for a 50/50 split-half given `(n, k)`.
/// Split fraction is hardcoded — NB calibration assumes balanced halves.
///
/// Precondition (caller's responsibility): `n ≥ k + 5`. The clamp
/// `n_train = min(max(n/2, k+2), n−3)` can produce `n_train < k+2`
/// when `n−3 < k+2` (i.e. `n < k+5`, e.g. `n=7, k=4 → n_train=4 < 6`).
/// Below that threshold the per-half fit degrades to a silent r=0.
/// `draw_splits` and `run_e` enforce this floor; subsamplers
/// enforce a stronger `m ≥ k+2` check after resolving `m`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn split_sizes(n: usize, k: usize) -> (usize, usize) {
    let want = n / 2;
    let mut n_train = want.max(k + 2);
    if n_train > n.saturating_sub(3) {
        n_train = n.saturating_sub(3);
    }
    let n_test = n - n_train;
    (n_train, n_test)
}

/// Generate one `(train_idx, test_idx)` split given a child RNG.
pub(crate) fn one_split(n: usize, n_train: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    let train = perm[..n_train].to_vec();
    let test = perm[n_train..].to_vec();
    (train, test)
}

/// Return a random permutation of `0..n` via a child RNG. The caller applies it
/// to reorder y rows; this function mutates nothing.
pub(crate) fn permute_indices(n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    perm
}

/// Permutation `b` of `0..n` as the replicate loops draw it:
/// `permute_indices(n, &mut child_rng(seed))`. A loop that holds the child
/// seeds can regenerate a permutation inside the unit that needs it instead
/// of holding `B` of them.
pub(crate) fn permutation_from_seed(n: usize, seed: u64) -> Vec<usize> {
    permute_indices(n, &mut child_rng(seed))
}

/// Map `0..n` in parallel, collecting in index order. No randomness:
/// element `i` is `f(i)` whichever worker computes it.
pub(crate) fn map_indexed<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    use rayon::prelude::*;
    (0..n).into_par_iter().map(&f).collect()
}

/// Bytes of an owned `f64` matrix of `rows × cols`, saturating: faer stores
/// each column of an owned `Mat<f64>` in a multiple of 8 rows (64-byte
/// aligned columns), so this is `⌈rows / 8⌉·8 · cols · 8`
/// (`copy_bytes_is_what_an_owned_mat_allocates` checks it against faer).
/// The usual term of a `unit_bytes` estimate. A value held as two matrices,
/// such as the halves of a split, is two calls: the padding is per matrix.
/// Mirrored by `copy` in `tests/memory_budget_parity.rs` (change together).
pub(crate) fn copy_bytes(rows: usize, cols: usize) -> usize {
    rows.checked_next_multiple_of(8)
        .unwrap_or(usize::MAX)
        .saturating_mul(cols)
        .saturating_mul(8)
}

/// Bytes of the two blocks a loop holds for one unit, `cols` columns each,
/// the most over `units`, which gives each unit's row counts `(n_a, n_b)`
/// (a split's training and test rows, a fold's training and validation
/// rows). They are two matrices, each padded as [`copy_bytes`] counts, so
/// the sum is taken per unit.
pub(crate) fn pair_blocks_bytes(units: impl Iterator<Item = (usize, usize)>, cols: usize) -> usize {
    units
        .map(|(n_a, n_b)| copy_bytes(n_a, cols).saturating_add(copy_bytes(n_b, cols)))
        .max()
        .unwrap_or(0)
}

/// [`pair_blocks_bytes`] for the folds of a fold loop (`folds` a partition
/// of `0..n`): one block on the training rows and one on the fold's rows.
/// Mirrored by the `fold_blocks` closure in `tests/memory_budget_parity.rs`
/// (change together).
pub(crate) fn fold_blocks_bytes(n: usize, folds: &[Vec<usize>], cols: usize) -> usize {
    pair_blocks_bytes(folds.iter().map(|f| (n - f.len(), f.len())), cols)
}

/// How many of `n_units` units a loop runs at once: `budget / unit_bytes`
/// (the budget is [`crate::fit::memory_budget`]), at least 1 and at most
/// `n_units`. `unit_bytes = 0` gives `n_units`. One unit always runs, even
/// when its buffers alone exceed the budget; `n_units = 0` gives 0.
pub(crate) fn units_at_once(n_units: usize, unit_bytes: usize) -> usize {
    if n_units == 0 {
        return 0;
    }
    if unit_bytes == 0 {
        return n_units;
    }
    let per_budget = crate::fit::memory_budget() / unit_bytes as u64;
    usize::try_from(per_budget).map_or(n_units, |k| k.clamp(1, n_units))
}

/// Map `0..n_units` in batches of [`units_at_once`]`(n_units, unit_bytes)`
/// through [`map_batches`]. `unit_bytes` is the estimated size of the large
/// buffers one unit holds.
pub(crate) fn map_units<T: Send>(
    n_units: usize,
    unit_bytes: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let k = units_at_once(n_units, unit_bytes);
    #[cfg(test)]
    BATCH_LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push((n_units, k));
        }
    });
    map_batches(n_units, k, f)
}

/// Map `0..n` in batches of `k`: in parallel inside a batch, batch after
/// batch, collected in index order. The next batch starts only when the
/// current one has ended, so at most `k` calls of `f` are live even when
/// `f` uses the pool itself (where a worker waiting inside `f` would
/// otherwise start another index). A batch of 1 or less is a sequential
/// loop. Element `i` is `f(i)` whatever `k` is, so results do not depend
/// on it.
pub(crate) fn map_batches<T: Send>(n: usize, k: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if k <= 1 {
        return (0..n).map(&f).collect();
    }
    let mut out = Vec::with_capacity(n);
    for start in (0..n).step_by(k) {
        let len = k.min(n - start);
        out.extend(map_indexed(len, |i| f(start + i)));
    }
    out
}

#[cfg(test)]
thread_local! {
    /// Set by `with_batch_log`; [`map_units`] appends to it while it is `Some`.
    static BATCH_LOG: std::cell::RefCell<Option<Vec<(usize, usize)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `f` and return its result with `(n_units, batch)` for every
/// [`map_units`] call made on the calling thread while `f` runs, in call
/// order, `batch` being [`units_at_once`]`(n_units, unit_bytes)` (test
/// builds). Calls on other threads are not recorded: a loop that a test
/// reaches on its own thread (as `fit::with_budget` needs) is, the calls
/// inside units that run on Rayon workers are not. The previous log is
/// restored afterwards, also when `f` panics.
#[cfg(test)]
pub(crate) fn with_batch_log<T>(f: impl FnOnce() -> T) -> (T, Vec<(usize, usize)>) {
    struct Restore(Option<Vec<(usize, usize)>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            BATCH_LOG.with(|log| *log.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(BATCH_LOG.with(|log| log.replace(Some(Vec::new()))));
    let out = f();
    let log = BATCH_LOG
        .with(|log| log.borrow_mut().take())
        .unwrap_or_default();
    (out, log)
}

/// Assert that a [`with_batch_log`] log holds exactly `calls` calls of
/// [`map_units`] over `n_units` units in batches of `batch`, and, when
/// `batch` is 1, that no call in it ran a larger batch (test builds).
#[cfg(test)]
#[track_caller]
pub(crate) fn assert_batches(log: &[(usize, usize)], n_units: usize, batch: usize, calls: usize) {
    let found = log.iter().filter(|&&e| e == (n_units, batch)).count();
    assert_eq!(
        found, calls,
        "map_units calls over {n_units} units in batches of {batch}; log {log:?}"
    );
    if batch == 1 {
        assert!(
            log.iter().all(|&(_, b)| b <= 1),
            "a batch larger than 1; log {log:?}"
        );
    }
}

/// [`map_units`] with a child generator per unit: `child_seeds(parent,
/// n_units)` is drawn up front (the same parent consumption as
/// [`parallel_for_each_seeded`]), then unit `u` runs
/// `f(u, &mut child_rng(seeds[u]))`. Results do not depend on the batch
/// size or on which worker runs a unit.
pub(crate) fn map_units_seeded<T: Send>(
    parent: &mut Rng,
    n_units: usize,
    unit_bytes: usize,
    f: impl Fn(usize, &mut Rng) -> T + Sync,
) -> Vec<T> {
    let seeds = child_seeds(parent, n_units);
    map_units(n_units, unit_bytes, |u| f(u, &mut child_rng(seeds[u])))
}

/// Second pass of a two-pass loop over units (splits or folds) and columns.
/// `rows[u][c]` is `None` where the first pass left column `c` of unit `u`
/// undecided. For each unit that has such columns, `fill(u, &pending)`
/// returns their values, `pending` being the undecided column indices
/// ascending. The units with pending columns run through [`map_units`] with
/// `unit_bytes` (the large buffers `fill` builds for one unit, such as a
/// standardized copy of its rows of X), so at most
/// [`units_at_once`] of them exist at once; `fill` may map its columns in
/// parallel itself. A unit with nothing pending is not handed to `fill`.
/// The result does not depend on the batch size.
///
/// # Panics
/// When `fill` returns fewer values than it was given columns.
pub(crate) fn fill_pending<T: Send>(
    rows: Vec<Vec<Option<T>>>,
    unit_bytes: usize,
    fill: impl Fn(usize, &[usize]) -> Vec<T> + Sync,
) -> Vec<Vec<T>> {
    let pending: Vec<(usize, Vec<usize>)> = rows
        .iter()
        .enumerate()
        .map(|(u, row)| (u, (0..row.len()).filter(|&c| row[c].is_none()).collect()))
        .filter(|(_, p): &(usize, Vec<usize>)| !p.is_empty())
        .collect();
    let mut filled = map_units(pending.len(), unit_bytes, |i| {
        let (u, cols) = &pending[i];
        fill(*u, cols)
    })
    .into_iter();
    let mut next_unit = pending.iter().map(|(u, _)| *u).peekable();
    rows.into_iter()
        .enumerate()
        .map(|(u, row)| {
            let mut vals = if next_unit.peek() == Some(&u) {
                next_unit.next();
                filled.next().unwrap_or_default().into_iter()
            } else {
                Vec::new().into_iter()
            };
            row.into_iter()
                .map(|v| {
                    v.unwrap_or_else(|| {
                        vals.next()
                            .expect("fill returns one value per pending column")
                    })
                })
                .collect()
        })
        .collect()
}

/// The per-column hook of [`Columns`]: `f(c, block)` rewrites null column
/// `c` (an `n × q` block) in place after its rows are permuted.
pub(crate) type NullMap<'a> = &'a (dyn Fn(usize, MatMut<'_, f64>) + Sync);

/// The outcome columns of a replicate loop, the one place every route takes
/// its null columns from: column 0 is `y`, column `c ≥ 1` is `y` with its
/// rows permuted by `permutation_from_seed(n, seeds[c - 1])`, then passed
/// through `null_map(c, ..)` when there is one. The runner draws `seeds`
/// once, before it chooses a route, so every route sees the same null
/// columns at a given seed, and a unit builds the column it needs with
/// [`Columns::column`] or [`Columns::block`] instead of the loop holding a
/// `B·n·q` buffer.
pub(crate) struct Columns<'a> {
    /// The observed outcome, raw or standardized, as the runner fits it,
    /// `n × q` (`q = 1` for PLS1).
    pub(crate) y: MatRef<'a, f64>,
    /// One child seed per null column.
    pub(crate) seeds: &'a [u64],
    /// Applied to every null column, never to column 0.
    pub(crate) null_map: Option<NullMap<'a>>,
}

impl Columns<'_> {
    /// `seeds.len() + 1`: the observed column plus one per seed.
    pub(crate) fn len(&self) -> usize {
        self.seeds.len() + 1
    }

    /// Column `c`, `n × q`: `y` for `c = 0`, else `y[perm[i], j]` with
    /// `perm = permutation_from_seed(n, seeds[c - 1])`, then `null_map`.
    pub(crate) fn block(&self, c: usize) -> Mat<f64> {
        let y = self.y;
        if c == 0 {
            return y.to_owned();
        }
        let perm = permutation_from_seed(y.nrows(), self.seeds[c - 1]);
        let mut m = Mat::<f64>::from_fn(y.nrows(), y.ncols(), |i, j| y[(perm[i], j)]);
        if let Some(f) = self.null_map {
            f(c, m.as_mut());
        }
        m
    }

    /// Column `c` of a `q = 1` outcome, length n: [`Columns::block`]'s
    /// only column, built without the `n × 1` matrix.
    pub(crate) fn column(&self, c: usize) -> Col<f64> {
        assert_eq!(self.y.ncols(), 1, "Columns::column needs a q = 1 outcome");
        let y = self.y.col(0);
        if c == 0 {
            return y.to_owned();
        }
        let perm = permutation_from_seed(y.nrows(), self.seeds[c - 1]);
        let mut col = Col::<f64>::from_fn(y.nrows(), |i| y[perm[i]]);
        if let Some(f) = self.null_map {
            f(c, col.as_mat_mut());
        }
        col
    }

    /// Columns `c0..c1` of a `q = 1` outcome side by side, `n × (c1 − c0)`.
    pub(crate) fn matrix(&self, c0: usize, c1: usize) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(self.y.nrows(), c1 - c0);
        for c in c0..c1 {
            m.col_mut(c - c0).copy_from(self.column(c));
        }
        m
    }
}

/// Sequentially compute J child seeds, then run `f(j, &mut child_rng)` in
/// parallel. The pre-computed seeds make the result independent of which
/// worker runs which `j`.
pub(crate) fn parallel_for_each_seeded<T: Send>(
    parent: &mut Rng,
    n_iterations: usize,
    f: impl Fn(usize, &mut Rng) -> T + Sync,
) -> Vec<T> {
    use rayon::prelude::*;
    child_seeds(parent, n_iterations)
        .into_par_iter()
        .enumerate()
        .map(|(i, s)| f(i, &mut child_rng(s)))
        .collect()
}

/// A row-major `n_rows × row_len` buffer allocated once and filled in
/// place, row `i` by `f(i, row)`, in parallel chunks. No randomness: row
/// `i` is `f(i, ..)`'s whichever worker runs it, so the buffer is
/// byte-identical across thread counts, and no per-row allocation outlives
/// its row. `f` must write every entry of its row.
/// With `row_len == 0` the buffer is empty and `f` is not called.
pub(crate) fn parallel_fill_rows(
    n_rows: usize,
    row_len: usize,
    f: impl Fn(usize, &mut [f64]) + Sync,
) -> Vec<f64> {
    use rayon::prelude::*;
    let mut buf = vec![0.0_f64; n_rows * row_len];
    if row_len == 0 {
        return buf;
    }
    buf.par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(i, row)| f(i, row));
    buf
}

/// H, the number of cells in the balanced split design: the smallest power of
/// two, at least 2, with `H/2 ≥ n_splits`, so that there are `n_splits`
/// distinct odd-weight masks to draw.
pub(crate) fn design_cells(n_splits: usize) -> usize {
    2 * n_splits.next_power_of_two()
}

/// Balanced splits over random cells of rows (balanced repeated replication).
/// `H = design_cells(J)` cells of `m` rows each; cell `c` is
/// `cell_rows[c*m .. (c+1)*m]`. The other rows are `leftovers`, in increasing
/// order. Split `j` puts cell `c` on its training side when
/// `((c & masks[j]).count_ones() + flips[j]) % 2 == 0`, which is exactly
/// `H/2` cells, and adds its own random pick of leftover rows.
pub(crate) struct SplitDesign {
    /// Rows per cell.
    pub(crate) m: usize,
    /// The `H·m` cell rows, cell by cell.
    pub(crate) cell_rows: Vec<usize>,
    /// The rows in no cell, in increasing order.
    pub(crate) leftovers: Vec<usize>,
    /// One distinct odd-weight mask in `1..H` per split.
    pub(crate) masks: Vec<u32>,
    /// One flip bit per split; without it cell 0 would train in every split.
    pub(crate) flips: Vec<bool>,
    /// The training and test row indices per split.
    pub(crate) splits: Vec<SplitIdx>,
}

impl SplitDesign {
    /// H, the number of cells ([`design_cells`] of the split count).
    pub(crate) fn cells(&self) -> usize {
        design_cells(self.masks.len())
    }
}

/// One split-half partition of the row indices `0..n`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SplitIdx {
    pub(crate) tr: Vec<usize>,
    pub(crate) te: Vec<usize>,
}

/// Draw `n_splits` balanced splits of `0..n` with `n_train` training rows
/// each. With `H = design_cells(n_splits)` and `n_B = n − n_train`,
/// `m = min(⌊n/H⌋, ⌊2·n_B/H⌋)`; each split takes `H/2` cells and
/// `n_train − H·m/2` of the `n − H·m` leftover rows.
///
/// Draw order from `rng`: the per-split child seeds as
/// `parallel_for_each_seeded` draws them, one shuffle of `0..n` whose first
/// `H·m` entries are the cell rows, the masks (drawn without replacement from
/// the odd-weight values in `1..H`), then the flip bits. Split `j` picks its
/// leftover training rows with `one_split(r, n_train − H·m/2, child_j)` over
/// the leftovers, so when `m = 0` the splits are J independent random halves,
/// exactly those `parallel_for_each_seeded(rng, n_splits, |_, c| one_split(n, n_train, c))`
/// draws.
///
/// Training indices are the training cells' rows in cell order, then the
/// leftover picks; test indices are the rest in the same order.
///
/// # Panics
/// Unless `n/2 ≤ n_train ≤ n` (the cells would put more rows on the
/// training side than the split takes), or when `H` exceeds `u32`.
pub(crate) fn balanced_splits(
    n: usize,
    n_train: usize,
    n_splits: usize,
    rng: &mut Rng,
) -> SplitDesign {
    use rand::RngExt as _;
    assert!(
        n / 2 <= n_train && n_train <= n,
        "balanced_splits needs n/2 <= n_train <= n"
    );
    let h = design_cells(n_splits);
    let h32 = u32::try_from(h).expect("design_cells(n_splits) fits in u32");
    let m = (n / h).min(2 * (n - n_train) / h);
    let leftover_train = n_train - h * m / 2;
    let picks = parallel_for_each_seeded(rng, n_splits, |_, c| {
        one_split(n - h * m, leftover_train, c)
    });
    let mut cell_rows = permute_indices(n, rng);
    let mut leftovers = cell_rows.split_off(h * m);
    leftovers.sort_unstable();
    let mut masks: Vec<u32> = (1..h32).filter(|s| s.count_ones() % 2 == 1).collect();
    masks.shuffle(rng);
    masks.truncate(n_splits);
    let flips: Vec<bool> = (0..n_splits).map(|_| rng.random()).collect();
    let splits = picks
        .into_iter()
        .enumerate()
        .map(|(j, (pick_tr, pick_te))| {
            let mut tr = Vec::with_capacity(n_train);
            let mut te = Vec::with_capacity(n - n_train);
            // With m = 0 every cell is empty; skip the J·H loop.
            let cells = if m == 0 { 0 } else { h32 };
            for c in 0..cells {
                let side = if ((c & masks[j]).count_ones() + u32::from(flips[j])) % 2 == 0 {
                    &mut tr
                } else {
                    &mut te
                };
                let c = c as usize;
                side.extend_from_slice(&cell_rows[c * m..(c + 1) * m]);
            }
            tr.extend(pick_tr.iter().map(|&i| leftovers[i]));
            te.extend(pick_te.iter().map(|&i| leftovers[i]));
            SplitIdx { tr, te }
        })
        .collect();
    SplitDesign {
        m,
        cell_rows,
        leftovers,
        masks,
        flips,
        splits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::resolve_seed;

    #[test]
    fn split_sizes_halves_n_and_bumps_for_small_train() {
        for (n, k, want) in [
            // n=10, k=1: n/2 = 5 and k+2 = 3 does not bump
            (10, 1, (5, 5)),
            // n=10, k=5: want=5, max(5, 5+2=7)=7
            (10, 5, (7, 3)),
        ] {
            assert_eq!(split_sizes(n, k), want, "n={n} k={k}");
        }
    }

    #[test]
    fn one_split_partitions_indices() {
        let (_, mut rng) = resolve_seed(Some(11)).unwrap();
        let (tr, te) = one_split(10, 7, &mut rng);
        assert_eq!(tr.len(), 7);
        assert_eq!(te.len(), 3);
        let mut all: Vec<usize> = tr.iter().chain(te.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn permute_indices_returns_permutation() {
        let (_, mut rng) = resolve_seed(Some(3)).unwrap();
        let p = permute_indices(20, &mut rng);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn permutation_from_seed_is_the_replicate_loops_draw() {
        let (_, mut a) = resolve_seed(Some(12)).unwrap();
        let (_, mut b) = resolve_seed(Some(12)).unwrap();
        let drawn = parallel_for_each_seeded(&mut a, 9, |_, rng| permute_indices(31, rng));
        let seeds = child_seeds(&mut b, 9);
        let regenerated: Vec<Vec<usize>> = seeds
            .iter()
            .map(|&s| permutation_from_seed(31, s))
            .collect();
        assert_eq!(drawn, regenerated);
    }

    #[test]
    fn map_indexed_collects_in_index_order() {
        let f = |i: usize| (i * 7919) % 101;
        assert_eq!(map_indexed(500, f), (0..500).map(f).collect::<Vec<_>>());
        assert_eq!(map_indexed(0, f), Vec::<usize>::new());
    }

    #[test]
    fn fill_pending_fills_each_unit_in_order_and_skips_complete_ones() {
        let rows: Vec<Vec<Option<usize>>> = vec![
            vec![Some(1), None, Some(3), None],
            vec![Some(5), Some(6)],
            vec![None, None, None],
            vec![],
        ];
        let calls = std::sync::Mutex::new(Vec::<(usize, Vec<usize>)>::new());
        let out = fill_pending(rows.clone(), 1, |u, pending| {
            calls.lock().unwrap().push((u, pending.to_vec()));
            pending.iter().map(|&c| 100 * u + c).collect()
        });
        let mut calls = calls.into_inner().unwrap();
        calls.sort();
        assert_eq!(calls, vec![(0, vec![1, 3]), (2, vec![0, 1, 2])]);
        // One unit at a time: the calls come in unit order.
        let ordered = std::sync::Mutex::new(Vec::<(usize, Vec<usize>)>::new());
        crate::fit::with_budget(0, || {
            fill_pending(rows.clone(), 1, |u, pending| {
                ordered.lock().unwrap().push((u, pending.to_vec()));
                pending.iter().map(|&c| 100 * u + c).collect()
            })
        });
        assert_eq!(
            ordered.into_inner().unwrap(),
            vec![(0, vec![1, 3]), (2, vec![0, 1, 2])]
        );
        let expected = vec![vec![1, 1, 3, 3], vec![5, 6], vec![200, 201, 202], vec![]];
        assert_eq!(out, expected);
        // The batch size does not change the output.
        for budget in [0, 1, u64::MAX] {
            let out = crate::fit::with_budget(budget, || {
                fill_pending(rows.clone(), 1, |u, pending| {
                    pending.iter().map(|&c| 100 * u + c).collect()
                })
            });
            assert_eq!(out, expected, "budget {budget}");
        }
    }

    #[test]
    fn units_at_once_clamps_to_one_and_to_the_unit_count() {
        crate::fit::with_budget(1000, || {
            assert_eq!(units_at_once(10, 1001), 1); // one unit over budget still runs
            assert_eq!(units_at_once(10, 500), 2);
            assert_eq!(units_at_once(10, 499), 2);
            assert_eq!(units_at_once(3, 1), 3);
            assert_eq!(units_at_once(10, 0), 10);
            assert_eq!(units_at_once(0, 10), 0);
        });
        crate::fit::with_budget(0, || assert_eq!(units_at_once(10, 1), 1));
        assert_eq!(copy_bytes(usize::MAX, 2), usize::MAX);
        assert_eq!(copy_bytes(3, 4), 256);
    }

    /// `copy_bytes` is what an owned faer `Mat<f64>` of that shape
    /// allocates: faer pads each column to a multiple of 8 rows.
    #[test]
    fn copy_bytes_is_what_an_owned_mat_allocates() {
        for rows in [0, 1, 7, 8, 9, 20, 41, 48] {
            for cols in [0, 1, 3] {
                let m = faer::Mat::<f64>::zeros(rows, cols);
                assert_eq!(
                    copy_bytes(rows, cols),
                    m.col_stride().unsigned_abs() * cols * 8,
                    "{rows} x {cols}"
                );
            }
        }
    }

    #[test]
    fn map_units_keeps_order_and_bounds_live_units_with_nested_columns() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        let (live, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let out = pool.install(|| {
            crate::fit::with_budget(300, || {
                map_units(10, 100, |u| {
                    peak.fetch_max(live.fetch_add(1, SeqCst) + 1, SeqCst);
                    // Hold the unit open so that units of one batch overlap.
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    let s: usize = map_indexed(64, |c| c * u).into_iter().sum();
                    live.fetch_sub(1, SeqCst);
                    s
                })
            })
        });
        assert_eq!(
            out,
            (0..10)
                .map(|u| (0..64).map(|c| c * u).sum())
                .collect::<Vec<usize>>()
        );
        assert!(peak.load(SeqCst) <= 3, "peak {}", peak.load(SeqCst));
        assert!(peak.load(SeqCst) >= 2, "peak {}", peak.load(SeqCst));
    }

    #[test]
    fn map_units_seeded_is_parallel_for_each_seeded_at_any_budget() {
        use rand::Rng as _;
        let (_, mut a) = resolve_seed(Some(21)).unwrap();
        let want = parallel_for_each_seeded(&mut a, 23, demo_row);
        for budget in [0, 200, u64::MAX] {
            let (_, mut b) = resolve_seed(Some(21)).unwrap();
            let got =
                crate::fit::with_budget(budget, || map_units_seeded(&mut b, 23, 100, demo_row));
            assert_eq!(got, want, "budget {budget}");
            let (_, mut a2) = resolve_seed(Some(21)).unwrap();
            let _ = parallel_for_each_seeded(&mut a2, 23, demo_row);
            assert_eq!(a2.next_u64(), b.next_u64(), "budget {budget}");
        }
    }

    #[test]
    fn columns_are_y_then_its_seeded_permutations() {
        let y = faer::Col::<f64>::from_fn(13, |i| i as f64 * 0.5 - 2.0);
        let seeds = [3_u64, 99, 7];
        let cols = Columns {
            y: y.as_ref().as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        assert_eq!(cols.len(), 4);
        let c0 = cols.column(0);
        assert!((0..13).all(|i| c0[i].to_bits() == y[i].to_bits()));
        let all = cols.matrix(1, 4);
        assert_eq!(all.ncols(), 3);
        for (c, s) in seeds.iter().enumerate() {
            let perm = permutation_from_seed(13, *s);
            let col = cols.column(c + 1);
            assert!(
                (0..13).all(|i| col[i].to_bits() == y[perm[i]].to_bits()
                    && all[(i, c)].to_bits() == col[i].to_bits()),
                "column {}",
                c + 1
            );
        }
    }

    #[test]
    fn columns_apply_the_null_map_to_null_columns_only() {
        let y = faer::Mat::<f64>::from_fn(13, 2, |i, j| i as f64 * 0.5 - 2.0 + 10.0 * j as f64);
        let seeds = [3_u64, 99, 7];
        // Adds 1000·c to every entry of null column c.
        let shift = |c: usize, mut m: faer::MatMut<'_, f64>| {
            for j in 0..m.ncols() {
                for i in 0..m.nrows() {
                    m[(i, j)] += 1000.0 * c as f64;
                }
            }
        };
        let cols = Columns {
            y: y.as_ref(),
            seeds: &seeds,
            null_map: Some(&shift),
        };
        assert!(cols.block(0) == y, "column 0 stays y");
        for (c, s) in seeds.iter().enumerate().map(|(c, s)| (c + 1, s)) {
            let perm = permutation_from_seed(13, *s);
            let want = faer::Mat::<f64>::from_fn(13, 2, |i, j| y[(perm[i], j)] + 1000.0 * c as f64);
            assert!(
                cols.block(c) == want,
                "null column {c}: permuted, then mapped"
            );
        }
        // The q = 1 accessors take the same path.
        let y1 = faer::Col::<f64>::from_fn(13, |i| i as f64);
        let cols1 = Columns {
            y: y1.as_ref().as_mat(),
            seeds: &seeds,
            null_map: Some(&shift),
        };
        let m = cols1.matrix(0, 4);
        for c in 0..4 {
            assert!(
                cols1.column(c).as_ref() == cols1.block(c).col(0),
                "column {c}"
            );
            assert!(m.col(c) == cols1.block(c).col(0), "matrix column {c}");
        }
        assert!(cols1.column(0) == y1, "column 0 stays y");
    }

    fn demo_row(i: usize, rng: &mut Rng) -> Vec<f64> {
        use rand::Rng as _;
        (0..5)
            .map(|j| (rng.next_u64() % 1000) as f64 + (i * 10 + j) as f64)
            .collect()
    }

    #[test]
    fn parallel_fill_rows_puts_row_i_at_offset_i() {
        let flat = parallel_fill_rows(37, 5, |i, out| {
            for (j, v) in out.iter_mut().enumerate() {
                *v = (i * 10 + j) as f64;
            }
        });
        let expected: Vec<f64> = (0..37_usize)
            .flat_map(|i| (0..5_usize).map(move |j| (i * 10 + j) as f64))
            .collect();
        assert_eq!(flat, expected);
    }

    #[test]
    fn parallel_fill_rows_handles_empty_shapes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let zero_rows = parallel_fill_rows(0, 4, |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(zero_rows, Vec::<f64>::new());
        let zero_cols = parallel_fill_rows(3, 0, |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(zero_cols, Vec::<f64>::new());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    fn design_at(n: usize, n_train: usize, n_splits: usize, seed: u64) -> SplitDesign {
        let (_, mut rng) = resolve_seed(Some(seed)).unwrap();
        balanced_splits(n, n_train, n_splits, &mut rng)
    }

    fn trains(d: &SplitDesign, c: usize, j: usize) -> bool {
        let c = u32::try_from(c).unwrap();
        ((c & d.masks[j]).count_ones() + u32::from(d.flips[j])) % 2 == 0
    }

    #[test]
    fn design_cells_is_the_smallest_power_of_two_with_enough_odd_masks() {
        assert_eq!(
            [1, 2, 3, 50, 64, 65].map(design_cells),
            [2, 4, 8, 128, 128, 256]
        );
    }

    /// At the default J the design has H = 64 cells: from n = 64 each cell
    /// holds a row, below that the splits are independent halves (m = 0).
    #[test]
    fn design_cells_at_default_j_is_64() {
        use crate::signal_test::{ConfirmatoryArgs, ConfirmatoryMethod};
        let ConfirmatoryArgs::SplitExact { n_splits, .. } =
            ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::SplitExact)
        else {
            unreachable!()
        };
        assert_eq!(design_cells(n_splits), 64);
        for (n, m) in [(63, 0), (64, 1)] {
            let d = design_at(n, split_sizes(n, 1).0, n_splits, 3);
            assert_eq!(d.m, m, "n={n}");
            assert_eq!(d.cell_rows.len(), 64 * m, "n={n}");
        }
    }

    #[test]
    fn every_split_is_a_partition_of_the_planned_size() {
        let shapes = [6, 127, 128, 129, 255, 256, 1000, 1001].map(|n| (n, n / 2));
        for (n, n_train) in shapes.into_iter().chain([(300, 202)]) {
            for n_splits in [1, 50, 64, 65] {
                for seed in 0..5 {
                    let d = design_at(n, n_train, n_splits, seed);
                    let at = format!("n={n} n_train={n_train} J={n_splits} seed={seed}");
                    assert_eq!(d.cell_rows.len(), design_cells(n_splits) * d.m, "{at}");
                    assert!(d.leftovers.is_sorted(), "{at}");
                    let mut rows: Vec<usize> =
                        d.cell_rows.iter().chain(&d.leftovers).copied().collect();
                    rows.sort_unstable();
                    assert_eq!(rows, (0..n).collect::<Vec<_>>(), "{at}");
                    assert_eq!(d.splits.len(), n_splits, "{at}");
                    for SplitIdx { tr, te } in &d.splits {
                        assert_eq!(tr.len(), n_train, "{at}");
                        assert_eq!(te.len(), n - n_train, "{at}");
                        let mut all: Vec<usize> = tr.iter().chain(te).copied().collect();
                        all.sort_unstable();
                        assert_eq!(all, (0..n).collect::<Vec<_>>(), "{at}");
                    }
                }
            }
        }
    }

    #[test]
    fn masks_are_distinct_odd_weight_and_cells_split_in_half() {
        for n_splits in [1, 50, 64, 65] {
            let h = design_cells(n_splits);
            for seed in 0..5 {
                let d = design_at(1000, 500, n_splits, seed);
                let mut distinct = d.masks.clone();
                distinct.sort_unstable();
                distinct.dedup();
                assert_eq!(distinct.len(), n_splits);
                assert!(
                    d.masks
                        .iter()
                        .all(|&s| s.count_ones() % 2 == 1 && (s as usize) < h)
                );
                assert_eq!(d.flips.len(), n_splits);
                let rows_of = |cells: &[usize]| -> Vec<usize> {
                    cells
                        .iter()
                        .flat_map(|&c| d.cell_rows[c * d.m..(c + 1) * d.m].iter().copied())
                        .collect()
                };
                for (j, SplitIdx { tr, te }) in d.splits.iter().enumerate() {
                    let at = format!("J={n_splits} seed={seed} split={j}");
                    let (cells, test_cells): (Vec<usize>, Vec<usize>) =
                        (0..h).partition(|&c| trains(&d, c, j));
                    assert_eq!(cells.len(), h / 2, "{at}");
                    // Index order: each side's cells' rows in cell order, then its leftovers.
                    let (rows, test_rows) = (rows_of(&cells), rows_of(&test_cells));
                    assert_eq!(tr[..rows.len()], rows[..], "{at}");
                    assert_eq!(te[..test_rows.len()], test_rows[..], "{at}");
                    let mut tails: Vec<usize> = tr[rows.len()..]
                        .iter()
                        .chain(&te[test_rows.len()..])
                        .copied()
                        .collect();
                    tails.sort_unstable();
                    assert_eq!(tails, d.leftovers, "{at}");
                }
            }
        }
    }

    #[test]
    fn pairs_of_splits_overlap_in_exact_quarters_of_the_cell_rows() {
        use std::collections::HashSet;
        let d = design_at(1000, 500, 50, 3);
        assert_eq!(d.m, 7);
        let cell: HashSet<usize> = d.cell_rows.iter().copied().collect();
        let trained: Vec<HashSet<usize>> = d
            .splits
            .iter()
            .map(|sp| sp.tr.iter().copied().filter(|r| cell.contains(r)).collect())
            .collect();
        for i in 0..trained.len() {
            for j in i + 1..trained.len() {
                assert_eq!(
                    trained[i].intersection(&trained[j]).count(),
                    128 * 7 / 4,
                    "splits {i}, {j}"
                );
            }
        }
    }

    #[test]
    fn no_cell_trains_in_every_split() {
        for seed in 0..20 {
            let d = design_at(1000, 500, 50, seed);
            let mut tested = vec![false; 1000];
            for &r in d.splits.iter().flat_map(|sp| &sp.te) {
                tested[r] = true;
            }
            assert!(d.cell_rows.iter().all(|&r| tested[r]), "seed={seed}");
        }
    }

    #[test]
    fn shrinks_m_when_the_training_side_is_large() {
        for (n, n_train, m) in [(300, 202, 1), (1000, 500, 7), (70, 42, 0)] {
            assert_eq!(design_at(n, n_train, 50, 0).m, m, "n={n} n_train={n_train}");
        }
    }

    #[test]
    fn below_h_rows_the_splits_are_independent_halves() {
        let (_, mut rng) = resolve_seed(Some(17)).unwrap();
        let mut rng_copy = rng.clone();
        let d = balanced_splits(100, 50, 50, &mut rng);
        assert_eq!(d.m, 0);
        let halves = parallel_for_each_seeded(&mut rng_copy, 50, |_, c| one_split(100, 50, c));
        let halves: Vec<SplitIdx> = halves
            .into_iter()
            .map(|(tr, te)| SplitIdx { tr, te })
            .collect();
        assert_eq!(d.splits, halves);
    }

    #[test]
    fn the_design_draws_seeds_then_cells_then_masks_then_flips() {
        use rand::{Rng as _, RngExt as _};
        let (n, n_train, n_splits, cell_rows) = (1000, 500, 50, 128 * 7);
        let (_, mut rng) = resolve_seed(Some(23)).unwrap();
        let mut replay = rng.clone();
        let d = balanced_splits(n, n_train, n_splits, &mut rng);
        let seeds = child_seeds(&mut replay, n_splits);
        let mut rows = permute_indices(n, &mut replay);
        let mut leftovers = rows.split_off(cell_rows);
        leftovers.sort_unstable();
        let mut masks: Vec<u32> = (1..128_u32).filter(|s| s.count_ones() % 2 == 1).collect();
        masks.shuffle(&mut replay);
        masks.truncate(n_splits);
        let flips: Vec<bool> = (0..n_splits).map(|_| replay.random()).collect();
        assert_eq!(d.cell_rows, rows);
        assert_eq!(d.masks, masks);
        assert_eq!(d.flips, flips);
        assert_eq!(
            rng.next_u64(),
            replay.next_u64(),
            "nothing is drawn after the flips"
        );
        // Each split's leftover picks come from its own child seed, after its cell rows.
        for (j, SplitIdx { tr, te }) in d.splits.iter().enumerate() {
            let (train_picks, test_picks) = one_split(
                n - cell_rows,
                n_train - cell_rows / 2,
                &mut child_rng(seeds[j]),
            );
            let mapped = |p: &[usize]| p.iter().map(|&i| leftovers[i]).collect::<Vec<_>>();
            assert_eq!(tr[cell_rows / 2..], mapped(&train_picks)[..], "split {j}");
            assert_eq!(te[cell_rows / 2..], mapped(&test_picks)[..], "split {j}");
        }
    }
}
