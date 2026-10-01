//! Resampling engines used by signal_test, sequential, and find_k.
//! Crate-internal — public surface is the callers.

use faer::{Col, ColRef, Par};
use rand::seq::SliceRandom;

use crate::rng::{child_rng, child_seeds, Rng};

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

/// Map `0..n` in parallel (or sequentially), collecting in index order. No
/// randomness: element `i` is `f(i)` whichever worker computes it.
pub(crate) fn map_indexed<T: Send>(
    n: usize,
    disable_parallelism: bool,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    if disable_parallelism {
        (0..n).map(&f).collect()
    } else {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(&f).collect()
    }
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

/// Parallelism of a replicate loop's per-block precompute (a matrix built
/// once per fold, split or call and shared read-only by its columns):
/// `Par::Seq` under `disable_parallelism`, else the crate's fixed-degree
/// Rayon split (`fit::par_fixed`, never `Par::rayon(0)`, whose degree is
/// the pool size). Every driver hands it to its block builder, so all
/// routes share one policy. The primal blocks build nothing.
pub(crate) fn block_par(disable_parallelism: bool) -> Par {
    if disable_parallelism {
        Par::Seq
    } else {
        crate::fit::par_fixed()
    }
}

/// Sequentially compute J child seeds, then run `f(j, &mut child_rng)`
/// in parallel via Rayon (or serially when `disable_parallelism` is set).
/// The pre-computed seeds make both paths byte-identical.
pub(crate) fn parallel_for_each_seeded<T: Send>(
    parent: &mut Rng,
    n_iterations: usize,
    disable_parallelism: bool,
    f: impl Fn(usize, &mut Rng) -> T + Sync,
) -> Vec<T> {
    let seeds = child_seeds(parent, n_iterations);
    if disable_parallelism {
        seeds
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let mut crng = child_rng(s);
                f(i, &mut crng)
            })
            .collect()
    } else {
        use rayon::prelude::*;
        seeds
            .into_par_iter()
            .enumerate()
            .map(|(i, s)| {
                let mut crng = child_rng(s);
                f(i, &mut crng)
            })
            .collect()
    }
}

/// `child_seeds(parent, n_rows)` first (unchanged seed consumption), then a
/// row-major `n_rows × row_len` buffer allocated once and filled in place,
/// row `i` by `f(i, &mut child_rng(seeds[i]), row)`, in parallel chunks or
/// sequentially. Row `i` sees the same stream whichever worker runs it, so
/// the buffer is byte-identical across thread counts, and no per-row
/// allocation outlives its row. `f` must write every entry of its row.
/// With `row_len == 0` the buffer is empty and `f` is not called.
pub(crate) fn parallel_fill_rows_seeded(
    parent: &mut Rng,
    n_rows: usize,
    row_len: usize,
    disable_parallelism: bool,
    f: impl Fn(usize, &mut Rng, &mut [f64]) + Sync,
) -> Vec<f64> {
    let seeds = child_seeds(parent, n_rows);
    let mut buf = vec![0.0_f64; n_rows * row_len];
    if row_len == 0 {
        return buf;
    }
    if disable_parallelism {
        for (i, (row, s)) in buf.chunks_mut(row_len).zip(&seeds).enumerate() {
            let mut crng = child_rng(*s);
            f(i, &mut crng, row);
        }
    } else {
        use rayon::prelude::*;
        buf.par_chunks_mut(row_len)
            .zip(seeds.par_iter())
            .enumerate()
            .for_each(|(i, (row, s))| {
                let mut crng = child_rng(*s);
                f(i, &mut crng, row);
            });
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::resolve_seed;

    #[test]
    #[allow(clippy::similar_names)]
    fn split_sizes_halves_n() {
        // n=10, k=1: n/2 = 5 and k+2 = 3 does not bump
        let (n_tr, n_te) = split_sizes(10, 1);
        assert_eq!(n_tr, 5);
        assert_eq!(n_te, 5);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn split_sizes_bumps_for_small_train() {
        // n=10, k=5: want=5, max(5, 5+2=7)=7
        let (n_tr, n_te) = split_sizes(10, 5);
        assert_eq!(n_tr, 7);
        assert_eq!(n_te, 3);
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
    fn parallel_for_each_seeded_disable_parallelism_byte_exact() {
        let (_, mut a) = resolve_seed(Some(99)).unwrap();
        let (_, mut b) = resolve_seed(Some(99)).unwrap();
        let par = parallel_for_each_seeded(&mut a, 64, false, |i, rng| {
            use rand::Rng;
            (i, rng.next_u64())
        });
        let ser = parallel_for_each_seeded(&mut b, 64, true, |i, rng| {
            use rand::Rng;
            (i, rng.next_u64())
        });
        assert_eq!(par, ser);
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
        let drawn = parallel_for_each_seeded(&mut a, 9, true, |_, rng| permute_indices(31, rng));
        let seeds = child_seeds(&mut b, 9);
        let regenerated: Vec<Vec<usize>> = seeds
            .iter()
            .map(|&s| permutation_from_seed(31, s))
            .collect();
        assert_eq!(drawn, regenerated);
    }

    #[test]
    fn map_indexed_collects_in_index_order_serial_and_parallel() {
        let f = |i: usize| (i * 7919) % 101;
        let serial = map_indexed(500, true, f);
        assert_eq!(serial, (0..500).map(f).collect::<Vec<_>>());
        assert_eq!(serial, map_indexed(500, false, f));
        assert!(map_indexed(0, false, f).is_empty());
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

    #[test]
    fn block_par_is_sequential_exactly_when_parallelism_is_disabled() {
        assert!(matches!(block_par(true), faer::Par::Seq));
        assert!(
            matches!(block_par(false), faer::Par::Rayon(d) if d.get() == crate::fit::PAR_DEGREE),
            "{:?}",
            block_par(false)
        );
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
        for dp in [true, false] {
            let (_, mut a) = resolve_seed(Some(8)).unwrap();
            let (_, mut b) = resolve_seed(Some(8)).unwrap();
            let rows = parallel_for_each_seeded(&mut a, 37, true, demo_row);
            let flat = parallel_fill_rows_seeded(&mut b, 37, 5, dp, |i, rng, out| {
                out.copy_from_slice(&demo_row(i, rng));
            });
            let expected: Vec<f64> = rows.concat();
            assert_eq!(
                flat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "dp={dp}"
            );
            // Same parent consumption.
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn parallel_fill_rows_seeded_handles_empty_shapes() {
        use rand::Rng as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let (_, mut p) = resolve_seed(Some(5)).unwrap();
        let (_, mut q) = resolve_seed(Some(5)).unwrap();
        assert!(parallel_fill_rows_seeded(&mut p, 0, 4, false, |_, _, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .is_empty());
        assert!(parallel_fill_rows_seeded(&mut p, 3, 0, false, |_, _, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let _ = child_seeds(&mut q, 3); // the seeds of the 3 × 0 call
        assert_eq!(p.next_u64(), q.next_u64());
    }
}
