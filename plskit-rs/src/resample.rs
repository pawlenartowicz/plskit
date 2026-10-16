//! Resampling engines used by signal_test, sequential, and find_k.
//! Crate-internal — public surface is the callers.

use faer::{Col, ColRef};
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

/// The outcome columns of a replicate loop: column 0 is `y`, column
/// `c ≥ 1` is `y` permuted by `permutation_from_seed(n, seeds[c - 1])`.
/// The runner draws `seeds` once, before it chooses a route, so every route
/// sees the same permutations at a given seed, and a unit builds the column
/// it needs with [`Columns::column`] instead of the loop holding a `B·n`
/// buffer.
pub(crate) struct Columns<'a> {
    /// The observed outcome, raw.
    pub(crate) y: ColRef<'a, f64>,
    /// One child seed per null column.
    pub(crate) seeds: &'a [u64],
}

impl Columns<'_> {
    /// `seeds.len() + 1`: the observed column plus one per seed.
    pub(crate) fn len(&self) -> usize {
        self.seeds.len() + 1
    }

    /// Column `c` (length n): `y` for `c = 0`, else `y[perm[i]]` with
    /// `perm = permutation_from_seed(n, seeds[c - 1])`.
    pub(crate) fn column(&self, c: usize) -> Col<f64> {
        let y = self.y;
        if c == 0 {
            y.to_owned()
        } else {
            let perm = permutation_from_seed(y.nrows(), self.seeds[c - 1]);
            Col::<f64>::from_fn(y.nrows(), |i| y[perm[i]])
        }
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

/// `child_seeds(parent, n_rows)` first (the same parent consumption as `parallel_for_each_seeded`), then a
/// row-major `n_rows × row_len` buffer allocated once and filled in place,
/// row `i` by `f(i, &mut child_rng(seeds[i]), row)`, in parallel chunks.
/// Row `i` sees the same stream whichever worker runs it, so the buffer is
/// byte-identical across thread counts, and no per-row allocation outlives
/// its row. `f` must write every entry of its row.
/// With `row_len == 0` the buffer is empty and `f` is not called.
pub(crate) fn parallel_fill_rows_seeded(
    parent: &mut Rng,
    n_rows: usize,
    row_len: usize,
    f: impl Fn(usize, &mut Rng, &mut [f64]) + Sync,
) -> Vec<f64> {
    use rayon::prelude::*;
    let seeds = child_seeds(parent, n_rows);
    let mut buf = vec![0.0_f64; n_rows * row_len];
    if row_len == 0 {
        return buf;
    }
    buf.par_chunks_mut(row_len)
        .zip(seeds.par_iter())
        .enumerate()
        .for_each(|(i, (row, s))| {
            let mut crng = child_rng(*s);
            f(i, &mut crng, row);
        });
    buf
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
            y: y.as_ref(),
            seeds: &seeds,
        };
        assert_eq!(cols.len(), 4);
        let c0 = cols.column(0);
        assert!((0..13).all(|i| c0[i].to_bits() == y[i].to_bits()));
        for (c, s) in seeds.iter().enumerate() {
            let perm = permutation_from_seed(13, *s);
            let col = cols.column(c + 1);
            assert!(
                (0..13).all(|i| col[i].to_bits() == y[perm[i]].to_bits()),
                "column {}",
                c + 1
            );
        }
    }

    fn demo_row(i: usize, rng: &mut Rng) -> Vec<f64> {
        use rand::Rng as _;
        (0..5)
            .map(|j| (rng.next_u64() % 1000) as f64 + (i * 10 + j) as f64)
            .collect()
    }

    #[test]
    fn parallel_fill_rows_seeded_is_parallel_for_each_seeded_flattened() {
        use rand::Rng as _;
        let (_, mut a) = resolve_seed(Some(8)).unwrap();
        let (_, mut b) = resolve_seed(Some(8)).unwrap();
        let rows = parallel_for_each_seeded(&mut a, 37, demo_row);
        let flat = parallel_fill_rows_seeded(&mut b, 37, 5, |i, rng, out| {
            out.copy_from_slice(&demo_row(i, rng));
        });
        let expected: Vec<f64> = rows.concat();
        assert_eq!(
            flat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // Same parent consumption.
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn parallel_fill_rows_seeded_handles_empty_shapes() {
        use rand::Rng as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let (_, mut p) = resolve_seed(Some(5)).unwrap();
        let (_, mut q) = resolve_seed(Some(5)).unwrap();
        let zero_rows = parallel_fill_rows_seeded(&mut p, 0, 4, |_, _, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(zero_rows, Vec::<f64>::new());
        let zero_cols = parallel_fill_rows_seeded(&mut p, 3, 0, |_, _, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(zero_cols, Vec::<f64>::new());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let _ = child_seeds(&mut q, 3); // the seeds of the 3 × 0 call
        assert_eq!(p.next_u64(), q.next_u64());
    }
}
