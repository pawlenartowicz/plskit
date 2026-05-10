//! Resampling engines used by signal_test, sequential, and find_k.
//! Crate-internal — public surface is the callers.

use rand::seq::SliceRandom;

use crate::rng::{child_rng, child_seeds, Rng};

/// Compute `(n_train, n_test)` for a 50/50 split-half given `(n, k)`.
/// Split fraction is hardcoded — NB calibration assumes balanced halves.
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn one_split(n: usize, n_train: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    let train = perm[..n_train].to_vec();
    let test = perm[n_train..].to_vec();
    (train, test)
}

/// Permute y in place via a child RNG.
#[allow(dead_code)]
pub(crate) fn permute_indices(n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    perm
}

/// Sequentially compute J child seeds, then run `f(j, &mut child_rng)`
/// in parallel via Rayon (or serially when `disable_parallelism` is set).
/// The pre-computed seeds make both paths byte-identical.
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::resolve_seed;

    #[test]
    #[allow(clippy::similar_names)]
    fn split_sizes_clamps_to_n_minus_three() {
        // n=10 → want=5, but k+2=2 doesn't bump and 5 ≤ n-3=7 so n_tr=5
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
        let (_, mut rng) = resolve_seed(Some(11));
        let (tr, te) = one_split(10, 7, &mut rng);
        assert_eq!(tr.len(), 7);
        assert_eq!(te.len(), 3);
        let mut all: Vec<usize> = tr.iter().chain(te.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_for_each_seeded_is_byte_exact_vs_serial() {
        // Same parent seed → same child seeds → same per-iter outputs in
        // the same order, regardless of Rayon scheduling.
        let (_, mut a) = resolve_seed(Some(99));
        let (_, mut b) = resolve_seed(Some(99));
        let par = parallel_for_each_seeded(&mut a, 64, false, |i, rng| {
            use rand::Rng;
            (i, rng.next_u64())
        });
        // Serial: pre-compute seeds the same way, run sequentially.
        let seeds = child_seeds(&mut b, 64);
        let ser: Vec<(usize, u64)> = seeds
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                use rand::Rng;
                let mut r = child_rng(s);
                (i, r.next_u64())
            })
            .collect();
        assert_eq!(par, ser);
    }

    #[test]
    fn parallel_for_each_seeded_disable_parallelism_byte_exact() {
        let (_, mut a) = resolve_seed(Some(99));
        let (_, mut b) = resolve_seed(Some(99));
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
        let (_, mut rng) = resolve_seed(Some(3));
        let p = permute_indices(20, &mut rng);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }
}
