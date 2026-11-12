//! Unit tests for the n-space PLS1 route: eligibility, the Gram builders
//! and their thread-count invariance, and the kernel.
#![allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::unreadable_literal
)]

use super::*;
use crate::fit::{ParChoice, PreparedFit, pls1_fit_prepared_fro};
use crate::linalg::{standardize, standardize1};
use crate::test_support::{assert_bits_eq, assert_mat_bits_eq, col_vals};
use faer::{Col, Mat, Par};
use rand::{RngExt, SeedableRng};

fn uniform(n: usize, p: usize, seed: u64) -> Mat<f64> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0))
}

#[test]
fn perm_null_is_eligible_from_k_one_to_k_dual_max() {
    assert!(nspace_eligible_perm_null(60, 3000, 200, 1, true));
    assert!(nspace_eligible_perm_null(60, 3000, 200, K_DUAL_MAX, true));
    assert!(!nspace_eligible_perm_null(60, 3000, 200, 0, true));
    assert!(!nspace_eligible_perm_null(
        60,
        3000,
        200,
        K_DUAL_MAX + 1,
        true
    ));
    // Weighted input stays primal.
    assert!(!nspace_eligible_perm_null(60, 3000, 200, 1, false));
    // k ≤ n − 1.
    assert!(nspace_eligible_perm_null(
        K_DUAL_MAX + 1,
        100_000,
        1000,
        K_DUAL_MAX,
        true
    ));
    assert!(!nspace_eligible_perm_null(
        K_DUAL_MAX, 100_000, 1000, K_DUAL_MAX, true
    ));
}

#[test]
fn raw_perm_starts_at_k_two_and_respects_the_smallest_training_fold() {
    // k = 1 belongs to the closed-form route.
    assert!(!nspace_eligible_raw_perm(60, 5, 3000, 200, 1, true));
    assert!(nspace_eligible_raw_perm(60, 5, 3000, 200, 2, true));
    assert!(!nspace_eligible_raw_perm(60, 5, 3000, 200, 2, false));
    assert!(!nspace_eligible_raw_perm(
        60,
        5,
        3000,
        200,
        K_DUAL_MAX + 1,
        true
    ));
    // Two folds of n = 2·r rows train on exactly r rows: eligible while
    // k ≤ r − 1.
    let r = K_DUAL_MAX + 1;
    assert!(nspace_eligible_raw_perm(
        2 * r,
        2,
        3000,
        200,
        K_DUAL_MAX,
        true
    ));
    assert!(!nspace_eligible_raw_perm(
        2 * (r - 1),
        2,
        3000,
        200,
        K_DUAL_MAX,
        true
    ));
    // Leave-one-out is eligible; more folds than rows, or one fold, is not.
    assert!(nspace_eligible_raw_perm(20, 20, 1500, 50, 2, true));
    assert!(!nspace_eligible_raw_perm(20, 21, 1500, 50, 2, true));
    assert!(!nspace_eligible_raw_perm(20, 1, 1500, 50, 2, true));
}

#[test]
fn raw_perm_nspace_rule_crosses_over_at_the_one_gram_cost() {
    // 60 x 100, 5 folds (n_tr 48, n_val 12), k = 2: 60²·100/2 +
    // 5·B'·(2·48² + 12·48) against 5·B'·(4·48 + 12)·100 pays from B' = 3.
    assert!(!nspace_eligible_raw_perm(60, 5, 100, 1, 2, true));
    assert!(nspace_eligible_raw_perm(60, 5, 100, 2, 2, true));
}

#[test]
fn split_exact_nspace_rule_crosses_over_at_the_one_gram_cost() {
    // 60 x 40, J = 10 (halves of 30), k = 2: without the gate's G the route
    // pays from B' = 3; holding it, from B' = 1.
    assert!(!nspace_eligible_split_exact(60, 40, 1, 10, 2, true, false));
    assert!(nspace_eligible_split_exact(60, 40, 2, 10, 2, true, false));
    assert!(nspace_eligible_split_exact(60, 40, 0, 10, 2, true, true));
}

#[test]
fn k_is_the_per_replicate_multiplicity_of_the_flop_rule() {
    // n = 60, p = 100, B = 100: at k = 1 the Gram costs more than it saves
    // (60·(100 + 100) = 12 000 ≥ 100·100), at k = 2 it pays
    // (60·(200 + 100) = 18 000 < 100·200).
    assert!(!nspace_eligible_perm_null(60, 100, 100, 1, true));
    assert!(nspace_eligible_perm_null(60, 100, 100, 2, true));
}

#[test]
fn gram_products_are_thread_count_invariant_at_the_route_shapes() {
    // (n_te, n_tr, p): the blocks this route builds for the corpus
    // fixtures (perm_null 40 × 2000 and 60 × 3000;
    // raw_perm folds 8 × 32 over 2000 and 12 × 48 over 3000; split_exact
    // halves 20 × 20 over 2000 and 30 × 30 over 3000), the leave-one-out M
    // (1 × 39 over 2000 and 1 × 19 over 4000: n_tr·p ≥ 65 536, where faer's
    // matrix-vector kernels turn parallel), and one product large enough
    // that faer surely splits it across threads.
    let shapes = [
        (1_usize, 40_usize, 2000_usize),
        (1, 60, 3000),
        (8, 32, 2000),
        (12, 48, 3000),
        (20, 20, 2000),
        (30, 30, 3000),
        (1, 39, 2000),
        (1, 19, 4000),
        (128, 256, 4000),
    ];
    for (si, &(n_te, n_tr, p)) in shapes.iter().enumerate() {
        let seed = 10 * si as u64;
        let xs = uniform(n_tr, p, seed + 1);
        let xt = uniform(n_te, p, seed + 2);
        let g_seq = gram_of(xs.as_ref(), Par::Seq);
        let m_seq = cross_of(xt.as_ref(), xs.as_ref(), Par::Seq);
        let what = format!("n_te={n_te} n_tr={n_tr} p={p}");
        assert_mat_bits_eq(
            gram_of(xs.as_ref(), Par::Seq).as_ref(),
            g_seq.as_ref(),
            &format!("G, {what}, serial"),
        );
        for threads in [2_usize, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let (g, m) = pool.install(|| {
                (
                    gram_of(xs.as_ref(), crate::fit::par_fixed()),
                    cross_of(xt.as_ref(), xs.as_ref(), crate::fit::par_fixed()),
                )
            });
            assert_mat_bits_eq(
                g.as_ref(),
                g_seq.as_ref(),
                &format!("G, {what}, {threads} threads"),
            );
            assert_mat_bits_eq(
                m.as_ref(),
                m_seq.as_ref(),
                &format!("M, {what}, {threads} threads"),
            );
        }
    }
}

fn standardized(n: usize, p: usize, seed: u64) -> Mat<f64> {
    standardize(uniform(n, p, seed).as_ref()).0
}

/// A standardized outcome: `snr` times the first column plus noise.
fn outcome(xs: &Mat<f64>, snr: f64, seed: u64) -> Col<f64> {
    let e = uniform(xs.nrows(), 1, seed);
    let y = Col::<f64>::from_fn(xs.nrows(), |i| snr * xs[(i, 0)] + e[(i, 0)]);
    standardize1(y.as_ref()).0
}

fn primal(xs: &Mat<f64>, z: &Col<f64>, k: usize) -> PreparedFit {
    pls1_fit_prepared_fro(
        xs.as_ref(),
        z.as_ref(),
        k,
        None,
        ParChoice::Seq,
        xs.norm_l2(),
    )
    .expect("primal fit")
}

#[test]
fn an_exactly_zero_outcome_is_the_zero_model() {
    let xs = standardized(20, 500, 4);
    let gram = NspaceGram::new(xs.as_ref(), Par::Seq);
    let z = Col::<f64>::from_fn(20, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert!(matches!(
        pls1_nspace_kernel(&gram.block(), z.as_ref(), 2),
        NspaceOutcome::ZeroModel
    ));
    assert_eq!(primal(&xs, &z, 2).k_used, 0);
}

/// A block built from `G` alone holds the same `G` bits and `p`, an
/// `x_fro` equal to `‖X̃‖_F` to rounding, and gives the same `alpha` bits:
/// `x_fro` reaches only the relative floor.
#[test]
fn a_block_from_the_gram_matrix_gives_the_alpha_bits_of_the_block_from_x() {
    let (n, p) = (40_usize, 2000_usize);
    let xs = standardized(n, p, 7);
    let from_x = NspaceGram::new(xs.as_ref(), Par::Seq);
    let g = gram_of(xs.as_ref(), Par::Seq);
    let from_g = NspaceGram::from_gram(g.as_ref(), p);
    assert_mat_bits_eq(from_g.g.as_ref(), from_x.g.as_ref(), "G");
    assert_eq!(from_g.p, p);
    let rel = (from_g.x_fro - from_x.x_fro).abs() / from_x.x_fro;
    assert!(rel <= 4.0 * p as f64 * f64::EPSILON, "x_fro: {rel:e}");
    for seed in 0..8_u64 {
        let z = outcome(&xs, 1.0, 300 + seed);
        match (
            pls1_nspace_kernel(&from_g.block(), z.as_ref(), 1),
            pls1_nspace_kernel(&from_x.block(), z.as_ref(), 1),
        ) {
            (
                NspaceOutcome::Resolved {
                    alpha: a,
                    k_used: 1,
                },
                NspaceOutcome::Resolved {
                    alpha: b,
                    k_used: 1,
                },
            ) => assert_bits_eq(
                &col_vals(a.as_ref()),
                &col_vals(b.as_ref()),
                &format!("seed {seed}"),
            ),
            _ => panic!("seed {seed}: both blocks must keep one component"),
        }
    }
}
