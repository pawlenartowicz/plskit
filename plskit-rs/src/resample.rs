//! Resampling engines used by signal_test, sequential, and find_k.
//! Crate-internal — public surface is the callers.

use commonstats::CommonStatsRng;
use faer::{Col, ColRef, Mat, MatMut, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::rng::{TAG_DSIGN, TAG_FOLDS, TAG_PICKS, shuffle};

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

/// One `(train_idx, test_idx)` split: `0..n` shuffled on `rng`, the first
/// `n_train` entries training.
pub(crate) fn one_split(
    n: usize,
    n_train: usize,
    rng: &mut CommonStatsRng,
) -> (Vec<usize>, Vec<usize>) {
    let mut perm: Vec<usize> = (0..n).collect();
    shuffle(rng, &mut perm);
    let test = perm.split_off(n_train);
    (perm, test)
}

/// `n_folds` folds of `0..n`: `0..n` shuffled on stream `(seed, id)` of
/// `TAG_FOLDS`, then cut by `linalg::fold_split`.
pub(crate) fn shuffled_folds(n: usize, n_folds: usize, seed: u64, id: u64) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(
        &mut CommonStatsRng::new_tagged(seed, id, TAG_FOLDS),
        &mut indices,
    );
    crate::linalg::fold_split(&indices, n_folds)
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

/// Bytes of faer's matrix-multiply buffer on one thread: on `x86_64`,
/// `private_gemm_x86` keeps a thread-local buffer of twice the L3 cache from
/// the thread's first large product until the thread exits; 0 elsewhere.
// shortcut: measured on the development machine (twice its L3 cache) and taken as 0 off x86_64 (aarch64, wasm), where faer packs into other buffers; read the cache size, or measure those targets, if a capped run there is killed by it.
pub(crate) const GEMM_THREAD_BYTES: usize = if cfg!(target_arch = "x86_64") {
    57_524_224
} else {
    0
};

/// faer's multiply buffers ([`GEMM_THREAD_BYTES`]) of a call run on this
/// thread: one per thread of the pool, and one more on a calling thread
/// outside any pool, which runs the serial parts of the call.
pub(crate) fn gemm_buffer_bytes() -> usize {
    let threads =
        rayon::current_num_threads() + usize::from(rayon::current_thread_index().is_none());
    GEMM_THREAD_BYTES.saturating_mul(threads)
}

/// [`units_fitting`] with the call's faer multiply buffers
/// ([`crate::fit::gemm_bytes`]) added to `call_bytes`.
///
/// # Errors
/// [`units_fitting`]'s, with those buffers in `needed`.
pub(crate) fn units_at_once(
    n_units: usize,
    call_bytes: usize,
    unit_bytes: usize,
) -> PlsKitResult<usize> {
    units_fitting(
        n_units,
        call_bytes.saturating_add(crate::fit::gemm_bytes()),
        unit_bytes,
    )
}

/// How many of `n_units` units a loop runs at once, `call_bytes` being the
/// bytes the loop holds across all its units (to which the bytes the call
/// holds beside the loop, [`crate::fit::held_bytes`], are added) and
/// `unit_bytes` those of one unit: `⌊(budget − call_bytes) / unit_bytes⌋` (the budget is
/// [`crate::fit::memory_budget`]), at most `n_units`. `unit_bytes = 0`
/// gives `n_units` at every budget (a unit of no bytes costs no memory);
/// otherwise budget 0 gives 1, one unit at a time, whatever the bytes.
/// `n_units = 0` gives 0. It leaves out faer's multiply buffers, which
/// depend on the pool, so the first pass of `route_select::select` compares
/// routes by it.
///
/// # Errors
/// `MemoryBudget` at a nonzero budget below `call_bytes + unit_bytes`,
/// with that sum as `needed` (`u64::MAX` when it overflows).
pub(crate) fn units_fitting(
    n_units: usize,
    call_bytes: usize,
    unit_bytes: usize,
) -> PlsKitResult<usize> {
    if n_units == 0 {
        return Ok(0);
    }
    let call_bytes = call_bytes.saturating_add(crate::fit::held_bytes());
    let budget = crate::fit::memory_budget();
    let needed = call_bytes
        .checked_add(unit_bytes)
        .and_then(|b| u64::try_from(b).ok());
    // A `call + unit` past `usize` is an error at any nonzero budget,
    // `u64::MAX` included; a call sum that its parts saturated at
    // `usize::MAX` (only past 16 EB) equals `u64::MAX` and fits that budget.
    if budget > 0 && needed.is_none_or(|b| b > budget) {
        return Err(PlsKitError::MemoryBudget {
            budget,
            needed: needed.unwrap_or(u64::MAX),
        });
    }
    if unit_bytes == 0 {
        return Ok(n_units);
    }
    if budget == 0 {
        return Ok(1);
    }
    let lanes = (budget - call_bytes as u64) / unit_bytes as u64;
    Ok(usize::try_from(lanes).map_or(n_units, |k| k.min(n_units)))
}

/// Map `0..n_units` in batches of
/// [`units_at_once`]`(n_units, call_bytes, unit_bytes)` through
/// [`map_batches`]. `call_bytes` is the estimated size of the large buffers
/// the loop holds across its units, `unit_bytes` that of one unit's.
///
/// # Errors
/// [`units_at_once`]'s, before `f` runs.
pub(crate) fn map_units<T: Send>(
    n_units: usize,
    call_bytes: usize,
    unit_bytes: usize,
    f: impl Fn(usize) -> T + Sync,
) -> PlsKitResult<Vec<T>> {
    let k = units_at_once(n_units, call_bytes, unit_bytes)?;
    #[cfg(test)]
    BATCH_LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push((n_units, k));
        }
    });
    Ok(map_batches(n_units, k, f))
}

/// Map `0..n` in batches of `k`: in parallel inside a batch, batch after
/// batch, collected in index order. The next batch starts only when the
/// current one has ended, so at most `k` calls of `f` are live even when
/// `f` uses the pool itself (where a worker waiting inside `f` would
/// otherwise start another index). A batch of 1 or less is a sequential
/// loop. Element `i` is `f(i)` whatever `k` is, so results do not depend
/// on it. Every call of `f` runs under the calling thread's
/// [`crate::fit::memory_budget`], whichever worker runs it.
pub(crate) fn map_batches<T: Send>(n: usize, k: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    // shortcut: a raw par_iter outside these helpers runs its closures under the worker thread's own budget (the environment's), held bytes and multiply buffers (0 on a worker), not the call's; wrap it if a loop inside it ever reads them
    let budget = crate::fit::memory_budget();
    let f = |i| crate::fit::with_budget(budget, || f(i));
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
/// order, `batch` being [`units_at_once`]`(n_units, call_bytes, unit_bytes)` (test
/// builds). Calls on other threads are not recorded: a loop that a test
/// reaches on its own thread is, the calls
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

/// Second pass of a two-pass loop over units (splits or folds) and columns.
/// `rows[u][c]` is `None` where the first pass left column `c` of unit `u`
/// undecided. For each unit that has such columns, `fill(u, &pending)`
/// returns their values, `pending` being the undecided column indices
/// ascending. The units with pending columns run through [`map_units`] with
/// `call_bytes` (the loop's buffers held across units) and `unit_bytes`
/// (the large buffers `fill` builds for one unit, such as a standardized
/// copy of its rows of X), so at most
/// [`units_at_once`] of them exist at once; `fill` may map its columns in
/// parallel itself. A unit with nothing pending is not handed to `fill`.
/// The result does not depend on the batch size.
///
/// # Errors
/// [`map_units`]'s, before `fill` runs.
///
/// # Panics
/// When `fill` returns fewer values than it was given columns.
pub(crate) fn fill_pending<T: Send>(
    rows: Vec<Vec<Option<T>>>,
    call_bytes: usize,
    unit_bytes: usize,
    fill: impl Fn(usize, &[usize]) -> Vec<T> + Sync,
) -> PlsKitResult<Vec<Vec<T>>> {
    let pending: Vec<(usize, Vec<usize>)> = rows
        .iter()
        .enumerate()
        .map(|(u, row)| (u, (0..row.len()).filter(|&c| row[c].is_none()).collect()))
        .filter(|(_, p): &(usize, Vec<usize>)| !p.is_empty())
        .collect();
    let mut filled = map_units(pending.len(), call_bytes, unit_bytes, |i| {
        let (u, cols) = &pending[i];
        fill(*u, cols)
    })?
    .into_iter();
    let mut next_unit = pending.iter().map(|(u, _)| *u).peekable();
    Ok(rows
        .into_iter()
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
        .collect())
}

/// The per-column hook of [`Columns`]: `f(c, block)` rewrites null column
/// `c` (an `n × q` block) in place after its rows are permuted.
pub(crate) type NullMap<'a> = &'a (dyn Fn(usize, MatMut<'_, f64>) + Sync);

/// The outcome columns of a replicate loop, the one place every route takes
/// its null columns from: column 0 is `y`, column `c ≥ 1` is `y` with its
/// rows permuted by [`Columns::permutation`], then passed through
/// `null_map(c, ..)` when there is one. Column `c` depends only on
/// `(seed, c)` and the weights, so every route and every loop order sees
/// the same null columns at a given seed, and a unit builds the column it
/// needs with [`Columns::column`] or [`Columns::block`] instead of the loop
/// holding a `B·n·q` buffer.
pub(crate) struct Columns<'a> {
    /// The observed outcome, raw or standardized, as the runner fits it,
    /// `n × q` (`q = 1` for PLS1).
    pub(crate) y: MatRef<'a, f64>,
    /// The run seed, the one recorded as `result.seed`. Unread when
    /// `n_perm` is 0.
    pub(crate) seed: u64,
    /// The number of null columns.
    pub(crate) n_perm: usize,
    /// Applied to every null column, never to column 0.
    pub(crate) null_map: Option<NullMap<'a>>,
    /// The call's weights, length n, or `None` unweighted. The null columns
    /// permute only the rows of positive weight and leave the others in
    /// place. Under the null `y` is exchangeable among the positive rows, so
    /// the p-value stays exact; `raw_perm`, which scores zero-weight
    /// validation rows unweighted, keeps their `y` in place in every column.
    pub(crate) w: Option<ColRef<'a, f64>>,
}

impl Columns<'_> {
    /// `n_perm + 1`: the observed column plus the null columns.
    pub(crate) fn len(&self) -> usize {
        self.n_perm + 1
    }

    /// Column `c`, `n × q`: `y` for `c = 0`, else `y[perm[i], j]` with
    /// `perm` [`Columns::permutation`]`(c)`, then `null_map`.
    pub(crate) fn block(&self, c: usize) -> Mat<f64> {
        let y = self.y;
        if c == 0 {
            return y.to_owned();
        }
        let perm = self.permutation(c);
        let mut m = Mat::<f64>::from_fn(y.nrows(), y.ncols(), |i, j| y[(perm[i] as usize, j)]);
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
        let perm = self.permutation(c);
        let mut col = Col::<f64>::from_fn(y.nrows(), |i| y[perm[i] as usize]);
        if let Some(f) = self.null_map {
            f(c, col.as_mat_mut());
        }
        col
    }

    /// The row permutation of null column `c`: `rng::null_permutation(n, c,
    /// seed)`. Under weights with `m < n` rows of positive weight it is
    /// `rng::null_permutation(m, c, seed)` applied to those rows in
    /// increasing order, every row of weight 0 mapped to itself, so with no
    /// zero weight it is the unweighted permutation.
    fn permutation(&self, c: usize) -> Vec<u32> {
        assert!(c <= self.n_perm, "null column {c} of {}", self.n_perm);
        let n = self.y.nrows();
        let Some(w) = self.w.filter(|w| w.iter().any(|&v| v <= 0.0)) else {
            return crate::rng::null_permutation(n, c, self.seed);
        };
        let n = u32::try_from(n).expect("resolve_seed's row check keeps n <= u32::MAX");
        let rows: Vec<u32> = (0..n).filter(|&i| w[i as usize] > 0.0).collect();
        let mut perm: Vec<u32> = (0..n).collect();
        for (i, &p) in crate::rng::null_permutation(rows.len(), c, self.seed)
            .iter()
            .enumerate()
        {
            perm[rows[i] as usize] = rows[p as usize];
        }
        perm
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

/// Rows `0..n_rows` of `row_len` f64s, row `i` written in place by
/// `f(i, row)`, in batches of `batch` rows (at least one), the rows of a
/// batch in parallel chunks; `fold` then reads each batch's rows
/// (row-major), batch after batch, so it sees every row in row order. With
/// `keep` every row stays in the returned row-major `n_rows × row_len`
/// buffer; otherwise only one batch's buffer exists and the returned one is
/// empty. No randomness: row `i` is `f(i, ..)`'s whichever worker runs it,
/// so the rows, and what `fold` sees, are byte-identical across thread
/// counts and batch sizes. `f` must write every entry of its row. With
/// `n_rows == 0` or `row_len == 0`, neither `f` nor `fold` is called.
/// Every call of `f` runs under the calling thread's
/// [`crate::fit::memory_budget`], as in [`map_batches`].
pub(crate) fn fill_rows_in_batches(
    n_rows: usize,
    row_len: usize,
    batch: usize,
    keep: bool,
    f: impl Fn(usize, &mut [f64]) + Sync,
    mut fold: impl FnMut(&[f64]),
) -> Vec<f64> {
    use rayon::prelude::*;
    if n_rows == 0 || row_len == 0 {
        return Vec::new();
    }
    let batch = batch.clamp(1, n_rows);
    let mut buf = vec![0.0_f64; if keep { n_rows } else { batch } * row_len];
    let budget = crate::fit::memory_budget();
    for start in (0..n_rows).step_by(batch) {
        let len = batch.min(n_rows - start);
        let off = if keep { start * row_len } else { 0 };
        let rows = &mut buf[off..off + len * row_len];
        rows.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, row)| crate::fit::with_budget(budget, || f(start + i, row)));
        fold(rows);
    }
    if keep { buf } else { Vec::new() }
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
    /// H, the number of cells: [`design_cells`] of the number of splits
    /// drawn. Removing splits ([`SplitDesign::retain`]) keeps it.
    pub(crate) h: usize,
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
    /// H, the number of cells ([`design_cells`] of the number of splits
    /// drawn).
    pub(crate) fn cells(&self) -> usize {
        self.h
    }

    /// Removes the splits `keep` refuses from `splits`, `masks` and `flips`
    /// together, keeping the order of the rest, and returns the number
    /// removed. The cells (`h`, `m`, `cell_rows`, `leftovers`) stay as
    /// drawn, so each kept split's sums over the cells are unchanged.
    pub(crate) fn retain(&mut self, keep: impl Fn(&SplitIdx) -> bool) -> usize {
        let drawn = self.splits.len();
        let mut kept = 0;
        for j in 0..drawn {
            if keep(&self.splits[j]) {
                self.splits.swap(kept, j);
                self.masks.swap(kept, j);
                self.flips.swap(kept, j);
                kept += 1;
            }
        }
        self.splits.truncate(kept);
        self.masks.truncate(kept);
        self.flips.truncate(kept);
        drawn - kept
    }
}

/// The bytes of a [`SplitDesign`] of `n_splits` splits on `n` rows with
/// halves of `n_tr` and `n_te` rows: every split's `tr` and `te` index
/// vectors and the `n` cell and leftover rows, 8 bytes per index. The
/// masks and flips are not counted.
#[allow(clippy::similar_names)] // n_tr, n_te
pub(crate) fn design_bytes(n: usize, n_tr: usize, n_te: usize, n_splits: usize) -> usize {
    n_splits
        .saturating_mul(n_tr.saturating_add(n_te))
        .saturating_add(n)
        .saturating_mul(8)
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
/// Draws: on stream `(seed, 0)` of `TAG_DSIGN`, one shuffle of `0..n` whose
/// first `H·m` entries are the cell rows, then one shuffle of the odd-weight
/// values in `1..H` whose first `n_splits` entries are the masks, then one
/// flip bit per split (`bounded(2) == 1`). Split `j` picks its leftover
/// training rows with `one_split(n − H·m, n_train − H·m/2, ..)` on stream
/// `(seed, j)` of `TAG_PICKS`, over the leftovers, so when `m = 0` the
/// splits are J independent random halves, split `j` being
/// `one_split(n, n_train, ..)` on that stream.
///
/// Training indices are the training cells' rows in cell order, then the
/// leftover picks; test indices are the rest in the same order.
///
/// # Panics
/// Unless `n/2 ≤ n_train ≤ n` (the cells would put more rows on the
/// training side than the split takes), or when `H` exceeds `u32`.
pub(crate) fn balanced_splits(n: usize, n_train: usize, n_splits: usize, seed: u64) -> SplitDesign {
    assert!(
        n / 2 <= n_train && n_train <= n,
        "balanced_splits needs n/2 <= n_train <= n"
    );
    let h = design_cells(n_splits);
    let h32 = u32::try_from(h).expect("design_cells(n_splits) fits in u32");
    let m = (n / h).min(2 * (n - n_train) / h);
    let leftover_train = n_train - h * m / 2;
    let picks = map_indexed(n_splits, |j| {
        let mut rng = CommonStatsRng::new_tagged(seed, j as u64, TAG_PICKS);
        one_split(n - h * m, leftover_train, &mut rng)
    });
    let mut rng = CommonStatsRng::new_tagged(seed, 0, TAG_DSIGN);
    let mut cell_rows: Vec<usize> = (0..n).collect();
    shuffle(&mut rng, &mut cell_rows);
    let mut leftovers = cell_rows.split_off(h * m);
    leftovers.sort_unstable();
    let mut masks: Vec<u32> = (1..h32).filter(|s| s.count_ones() % 2 == 1).collect();
    shuffle(&mut rng, &mut masks);
    masks.truncate(n_splits);
    let flips: Vec<bool> = (0..n_splits).map(|_| rng.bounded(2) == 1).collect();
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
        h,
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
    use crate::rng::TAG_SPLIT;

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
        let (tr, te) = one_split(10, 7, &mut CommonStatsRng::new_tagged(11, 0, TAG_SPLIT));
        assert_eq!(tr.len(), 7);
        assert_eq!(te.len(), 3);
        let mut all: Vec<usize> = tr.iter().chain(te.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
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
        let out = fill_pending(rows.clone(), 0, 1, |u, pending| {
            calls.lock().unwrap().push((u, pending.to_vec()));
            pending.iter().map(|&c| 100 * u + c).collect()
        })
        .unwrap();
        let mut calls = calls.into_inner().unwrap();
        calls.sort();
        assert_eq!(calls, vec![(0, vec![1, 3]), (2, vec![0, 1, 2])]);
        // One unit at a time: the calls come in unit order.
        let ordered = std::sync::Mutex::new(Vec::<(usize, Vec<usize>)>::new());
        crate::fit::with_budget(0, || {
            fill_pending(rows.clone(), 0, 1, |u, pending| {
                ordered.lock().unwrap().push((u, pending.to_vec()));
                pending.iter().map(|&c| 100 * u + c).collect()
            })
        })
        .unwrap();
        assert_eq!(
            ordered.into_inner().unwrap(),
            vec![(0, vec![1, 3]), (2, vec![0, 1, 2])]
        );
        let expected = vec![vec![1, 1, 3, 3], vec![5, 6], vec![200, 201, 202], vec![]];
        assert_eq!(out, expected);
        // The batch size does not change the output.
        for budget in [0, 1, u64::MAX] {
            let out = crate::fit::with_budget(budget, || {
                fill_pending(rows.clone(), 0, 1, |u, pending| {
                    pending.iter().map(|&c| 100 * u + c).collect()
                })
            })
            .unwrap();
            assert_eq!(out, expected, "budget {budget}");
        }
    }

    #[test]
    fn units_fitting_is_strict_above_zero() {
        use crate::error::PlsKitError;
        crate::fit::with_budget(1000, || {
            assert_eq!(units_fitting(10, 0, 500).unwrap(), 2);
            assert_eq!(units_fitting(10, 400, 300).unwrap(), 2);
            for (call, unit) in [(0, 1001), (900, 101)] {
                let r = units_fitting(10, call, unit);
                assert!(
                    matches!(
                        r,
                        Err(PlsKitError::MemoryBudget {
                            budget: 1000,
                            needed: 1001
                        })
                    ),
                    "call {call}, unit {unit}: {r:?}"
                );
            }
            assert_eq!(units_fitting(0, 0, 5000).unwrap(), 0);
            assert_eq!(units_fitting(10, 0, 0).unwrap(), 10);
            // A call-held buffer over the budget errors even with free units.
            let r = units_fitting(10, 1001, 0);
            assert!(
                matches!(
                    r,
                    Err(PlsKitError::MemoryBudget {
                        budget: 1000,
                        needed: 1001
                    })
                ),
                "call 1001, unit 0: {r:?}"
            );
        });
    }

    #[test]
    fn units_fitting_zero_is_one_at_a_time() {
        crate::fit::with_budget(0, || {
            assert_eq!(units_fitting(10, 5000, 5000).unwrap(), 1);
            // A unit of no bytes costs no memory: every unit at once.
            assert_eq!(units_fitting(10, 0, 0).unwrap(), 10);
            assert_eq!(units_fitting(10, usize::MAX, 1).unwrap(), 1);
        });
    }

    /// faer's buffers are one per thread of the pool and one more on a
    /// calling thread outside any pool; the real check adds the call's term
    /// and [`crate::fit::with_budget`] (a call made inside another, a loop's
    /// units) clears it.
    #[test]
    fn units_at_once_adds_the_calls_multiply_buffers() {
        use crate::error::PlsKitError;
        let three = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        assert_eq!(three.install(gemm_buffer_bytes), 3 * GEMM_THREAD_BYTES);
        assert_eq!(
            gemm_buffer_bytes(),
            (rayon::current_num_threads() + 1) * GEMM_THREAD_BYTES
        );
        crate::fit::with_budget(1000, || {
            let _gemm = crate::fit::hold_gemm_bytes(500);
            assert_eq!(units_at_once(10, 0, 100).unwrap(), 5);
            assert_eq!(units_fitting(10, 0, 100).unwrap(), 10);
            let r = units_at_once(10, 0, 501);
            assert!(
                matches!(r, Err(PlsKitError::MemoryBudget { needed: 1001, .. })),
                "{r:?}"
            );
            assert_eq!(crate::fit::with_budget(1000, crate::fit::gemm_bytes), 0);
        });
    }

    /// A call that reads its budget sets faer's buffers for its loops, on
    /// the thread that runs them; a call made inside another, which runs
    /// under the outer call's budget, sets none.
    #[test]
    fn with_thread_limit_sets_the_gemm_term_outermost_only() {
        let (term, there) =
            crate::fit::with_thread_limit(|| Ok((crate::fit::gemm_bytes(), gemm_buffer_bytes())))
                .unwrap();
        assert_eq!(term, there);
        let nested = crate::fit::with_budget(1000, || {
            crate::fit::with_thread_limit(|| Ok(crate::fit::gemm_bytes()))
        })
        .unwrap();
        assert_eq!(nested, 0);
    }

    #[test]
    fn units_at_once_saturates() {
        use crate::error::PlsKitError;
        // At the largest budget too: an overflowing need never wraps into a fit.
        for budget in [1000, u64::MAX] {
            let r = crate::fit::with_budget(budget, || units_at_once(10, usize::MAX, 1));
            assert!(
                matches!(r, Err(PlsKitError::MemoryBudget { budget: b, needed: u64::MAX }) if b == budget),
                "budget {budget}: {r:?}"
            );
        }
        assert_eq!(copy_bytes(usize::MAX, 2), usize::MAX);
        assert_eq!(copy_bytes(3, 4), 256);
    }

    #[test]
    fn workers_see_the_call_budget() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let seen = pool.install(|| {
            crate::fit::with_budget(123, || {
                map_units(8, 0, 1, |_| {
                    // Hold the unit open so that the other workers take units.
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    crate::fit::memory_budget()
                })
            })
        });
        assert_eq!(seen.unwrap(), vec![123; 8]);
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
        let out = pool
            .install(|| {
                crate::fit::with_budget(300, || {
                    map_units(10, 0, 100, |u| {
                        peak.fetch_max(live.fetch_add(1, SeqCst) + 1, SeqCst);
                        // Hold the unit open so that units of one batch overlap.
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        let s: usize = map_indexed(64, |c| c * u).into_iter().sum();
                        live.fetch_sub(1, SeqCst);
                        s
                    })
                })
            })
            .unwrap();
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
    fn columns_are_y_then_its_seeded_permutations() {
        let y = faer::Col::<f64>::from_fn(13, |i| i as f64 * 0.5 - 2.0);
        let cols = Columns {
            y: y.as_ref().as_mat(),
            seed: 2809,
            n_perm: 3,
            null_map: None,
            w: None,
        };
        assert_eq!(cols.len(), 4);
        let c0 = cols.column(0);
        assert!((0..13).all(|i| c0[i].to_bits() == y[i].to_bits()));
        let all = cols.matrix(1, 4);
        assert_eq!(all.ncols(), 3);
        for c in 0..3 {
            let perm = crate::rng::null_permutation(13, c + 1, 2809);
            let col = cols.column(c + 1);
            assert!(
                (0..13).all(|i| col[i].to_bits() == y[perm[i] as usize].to_bits()
                    && all[(i, c)].to_bits() == col[i].to_bits()),
                "column {}",
                c + 1
            );
        }
    }

    /// Under weights, null column `c` permutes the rows of positive weight
    /// by `null_permutation(m, c, seed)` and leaves the rows of weight 0 in
    /// place; with every weight positive it is the unweighted column.
    #[test]
    fn weighted_columns_permute_the_rows_of_positive_weight() {
        fn cols<'a>(y: &'a faer::Col<f64>, w: Option<&'a faer::Col<f64>>) -> Columns<'a> {
            Columns {
                y: y.as_ref().as_mat(),
                seed: 2809,
                n_perm: 3,
                null_map: None,
                w: w.map(faer::Col::as_ref),
            }
        }
        let y = faer::Col::<f64>::from_fn(13, |i| i as f64 * 0.5 - 2.0);
        let w = faer::Col::<f64>::from_fn(13, |i| if i % 4 == 1 { 0.0 } else { 1.5 });
        let positive = faer::Col::<f64>::from_fn(13, |i| 0.5 + i as f64);
        let rows: Vec<usize> = (0..13).filter(|i| i % 4 != 1).collect();
        for c in 1..=3 {
            let col = cols(&y, Some(&w)).column(c);
            let perm = crate::rng::null_permutation(rows.len(), c, 2809);
            for i in 0..13 {
                let want = match rows.iter().position(|&r| r == i) {
                    Some(j) => y[rows[perm[j] as usize]],
                    None => y[i],
                };
                assert_eq!(col[i].to_bits(), want.to_bits(), "column {c}, row {i}");
            }
            assert_eq!(
                cols(&y, Some(&positive)).column(c),
                cols(&y, None).column(c)
            );
        }
    }

    #[test]
    fn columns_apply_the_null_map_to_null_columns_only() {
        let y = faer::Mat::<f64>::from_fn(13, 2, |i, j| i as f64 * 0.5 - 2.0 + 10.0 * j as f64);
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
            seed: 2809,
            n_perm: 3,
            null_map: Some(&shift),
            w: None,
        };
        assert!(cols.block(0) == y, "column 0 stays y");
        for c in 1..=3 {
            let perm = crate::rng::null_permutation(13, c, 2809);
            let want = faer::Mat::<f64>::from_fn(13, 2, |i, j| {
                y[(perm[i] as usize, j)] + 1000.0 * c as f64
            });
            assert!(
                cols.block(c) == want,
                "null column {c}: permuted, then mapped"
            );
        }
        // The q = 1 accessors take the same path.
        let y1 = faer::Col::<f64>::from_fn(13, |i| i as f64);
        let cols1 = Columns {
            y: y1.as_ref().as_mat(),
            seed: 2809,
            n_perm: 3,
            null_map: Some(&shift),
            w: None,
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

    /// A null column depends only on its index and the seed: built in any
    /// order, the columns are the same bits.
    #[test]
    fn columns_do_not_depend_on_order() {
        let y = faer::Col::<f64>::from_fn(17, |i| (i as f64 * 0.37).sin());
        let cols = Columns {
            y: y.as_ref().as_mat(),
            seed: 2809,
            n_perm: 3,
            null_map: None,
            w: None,
        };
        let ascending: Vec<faer::Col<f64>> = (1..=3).map(|c| cols.column(c)).collect();
        let mut shuffled: Vec<(usize, faer::Col<f64>)> =
            [3, 1, 2].into_iter().map(|c| (c, cols.column(c))).collect();
        shuffled.sort_by_key(|(c, _)| *c);
        let m = cols.matrix(1, 4);
        for (i, (c, col)) in shuffled.iter().enumerate() {
            assert!(
                (0..17).all(|r| col[r].to_bits() == ascending[i][r].to_bits()
                    && m[(r, i)].to_bits() == col[r].to_bits()),
                "column {c}"
            );
        }
    }

    /// With covariates, null column `c` is the residual on `[1, Z]` of `ỹ`
    /// gathered by commonstats' permutation `c`, and column 0 is `ỹ`.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn freedman_lane_column_is_the_residual_of_the_permuted_residual() {
        use crate::covariates::CovariateBasis;
        use commonstats::{Scheme, gen_resample_indices};
        let n = 12;
        let z = crate::test_support::unif(n, 2, 5);
        let y = crate::test_support::unif_col(n, 6);
        let basis = CovariateBasis::new(z.as_ref(), None).unwrap();
        let mut y_res = faer::Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
        basis.residualize(y_res.as_mut());
        let fl = |_: usize, m: MatMut<'_, f64>| basis.residualize(m);
        let cols = Columns {
            y: y_res.as_ref(),
            seed: 2809,
            n_perm: 3,
            null_map: Some(&fl),
            w: None,
        };
        // The residual on [1, Z] from the normal equations.
        let a = crate::test_support::with_intercept(z.as_ref());
        let residual = |v: &faer::Mat<f64>| {
            let coef = crate::test_support::wls_normal_equations(a.as_ref(), None, v.as_ref());
            v - &a * &coef
        };
        let norm = y_res.norm_l2();
        let c0 = cols.column(0);
        assert!((0..n).all(|i| c0[i].to_bits() == y_res[(i, 0)].to_bits()));
        for c in 1..=3 {
            let mut perm = vec![0u32; n];
            gen_resample_indices(Scheme::Permutation, n, c as u64, 2809, &mut perm).unwrap();
            let gathered = faer::Mat::<f64>::from_fn(n, 1, |i, _| y_res[(perm[i] as usize, 0)]);
            let want = residual(&gathered);
            let got = cols.column(c);
            let diff = (0..n).fold(0.0_f64, |m, i| m.max((got[i] - want[(i, 0)]).abs()));
            assert!(diff <= 1e-12 * norm, "column {c}: {diff:e}");
        }
    }

    /// Row `i` is `f(i, ..)` and `fold` sees the rows in row order, kept
    /// or not, whatever the batch size; with `keep` the buffer holds them.
    #[test]
    fn fill_rows_in_batches_folds_rows_in_order() {
        let row = |i: usize, out: &mut [f64]| {
            for (j, v) in out.iter_mut().enumerate() {
                *v = (i * 10 + j) as f64;
            }
        };
        let expected: Vec<f64> = (0..37_usize)
            .flat_map(|i| (0..5_usize).map(move |j| (i * 10 + j) as f64))
            .collect();
        for (batch, keep) in [(1, false), (4, false), (37, false), (4, true), (100, true)] {
            let mut seen = Vec::new();
            let kept = fill_rows_in_batches(37, 5, batch, keep, row, |r| seen.extend_from_slice(r));
            assert_eq!(seen, expected, "batch {batch} keep {keep}");
            assert_eq!(kept, if keep { expected.clone() } else { Vec::new() });
        }
    }

    #[test]
    fn fill_rows_in_batches_handles_empty_shapes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let count = |_: usize, _: &mut [f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
        };
        let mut folds = 0;
        let empty = Vec::<f64>::new();
        assert_eq!(
            fill_rows_in_batches(0, 4, 2, true, count, |_| folds += 1),
            empty
        );
        assert_eq!(
            fill_rows_in_batches(3, 0, 2, true, count, |_| folds += 1),
            empty
        );
        assert_eq!((calls.load(Ordering::Relaxed), folds), (0, 0));
    }

    fn design_at(n: usize, n_train: usize, n_splits: usize, seed: u64) -> SplitDesign {
        balanced_splits(n, n_train, n_splits, seed)
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
        let d = balanced_splits(100, 50, 50, 17);
        assert_eq!(d.m, 0);
        let halves: Vec<SplitIdx> = (0..50_u64)
            .map(|j| {
                let (tr, te) =
                    one_split(100, 50, &mut CommonStatsRng::new_tagged(17, j, TAG_PICKS));
                SplitIdx { tr, te }
            })
            .collect();
        assert_eq!(d.splits, halves);
    }

    #[test]
    fn the_design_draws_cells_then_masks_then_flips() {
        let (n, n_train, n_splits, cell_rows) = (1000, 500, 50, 128 * 7);
        let d = balanced_splits(n, n_train, n_splits, 23);
        let mut replay = CommonStatsRng::new_tagged(23, 0, TAG_DSIGN);
        let mut rows: Vec<usize> = (0..n).collect();
        shuffle(&mut replay, &mut rows);
        let mut leftovers = rows.split_off(cell_rows);
        leftovers.sort_unstable();
        let mut masks: Vec<u32> = (1..128_u32).filter(|s| s.count_ones() % 2 == 1).collect();
        shuffle(&mut replay, &mut masks);
        masks.truncate(n_splits);
        let flips: Vec<bool> = (0..n_splits).map(|_| replay.bounded(2) == 1).collect();
        assert_eq!(d.cell_rows, rows);
        assert_eq!(d.masks, masks);
        assert_eq!(d.flips, flips);
        // Split j's leftover picks come from stream j of `TAG_PICKS`, after its cell rows.
        for (j, SplitIdx { tr, te }) in d.splits.iter().enumerate() {
            let (train_picks, test_picks) = one_split(
                n - cell_rows,
                n_train - cell_rows / 2,
                &mut CommonStatsRng::new_tagged(23, j as u64, TAG_PICKS),
            );
            let mapped = |p: &[usize]| p.iter().map(|&i| leftovers[i]).collect::<Vec<_>>();
            assert_eq!(tr[cell_rows / 2..], mapped(&train_picks)[..], "split {j}");
            assert_eq!(te[cell_rows / 2..], mapped(&test_picks)[..], "split {j}");
        }
    }

    /// Keeping the splits that train on row 0 keeps H = 64 and the cells,
    /// so each kept split reads the same Walsh row as in the full design:
    /// its mask, flip, rows and cell sums are the full design's at its
    /// position, bit for bit.
    #[test]
    fn retain_keeps_the_cells_of_the_design() {
        let draw = || balanced_splits(200, 100, 32, 9);
        let full = draw();
        assert_eq!((full.cells(), full.m), (64, 3), "premise: H = 64, m = 3");
        let mut kept = draw();
        let removed = kept.retain(|sp| sp.tr.contains(&0));
        let at: Vec<usize> = (0..32)
            .filter(|&j| full.splits[j].tr.contains(&0))
            .collect();
        assert!(removed > 0 && !at.is_empty(), "premise: {removed} removed");
        assert_eq!(removed + at.len(), 32);
        assert_eq!(kept.cells(), 64);
        assert_eq!(kept.splits.len(), at.len());
        let xs = crate::test_support::unif(200, 4, 31);
        let cols = crate::test_support::unif(200, 2, 32);
        let sums_full = crate::cell_sums::cell_sums(xs.as_ref(), &full, cols.as_ref(), 1);
        let sums_kept = crate::cell_sums::cell_sums(xs.as_ref(), &kept, cols.as_ref(), 1);
        for (u, &j) in at.iter().enumerate() {
            assert_eq!(kept.masks[u], full.masks[j], "split {j}");
            assert_eq!(kept.flips[u], full.flips[j], "split {j}");
            assert_eq!(kept.splits[u], full.splits[j], "split {j}");
            for c in 0..2 {
                for f in 0..4 {
                    assert_eq!(
                        sums_kept.xa_y[(f, u * 2 + c)].to_bits(),
                        sums_full.xa_y[(f, j * 2 + c)].to_bits(),
                        "split {j} xa_y"
                    );
                }
                assert_eq!(
                    sums_kept.ya_sum[(u, c)].to_bits(),
                    sums_full.ya_sum[(j, c)].to_bits(),
                    "split {j} ya_sum"
                );
                assert_eq!(
                    sums_kept.ya_ss[(u, c)].to_bits(),
                    sums_full.ya_ss[(j, c)].to_bits(),
                    "split {j} ya_ss"
                );
            }
        }
    }
}
