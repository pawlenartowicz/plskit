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
fn perm_null_is_allowed_from_k_one_to_k_dual_max() {
    assert!(nspace_allowed_perm_null(60, 1, true));
    assert!(nspace_allowed_perm_null(60, K_DUAL_MAX, true));
    assert!(!nspace_allowed_perm_null(60, 0, true));
    assert!(!nspace_allowed_perm_null(60, K_DUAL_MAX + 1, true));
    // Weighted input stays primal.
    assert!(!nspace_allowed_perm_null(60, 1, false));
    // k ≤ n − 1.
    assert!(nspace_allowed_perm_null(K_DUAL_MAX + 1, K_DUAL_MAX, true));
    assert!(!nspace_allowed_perm_null(K_DUAL_MAX, K_DUAL_MAX, true));
}

/// Whether `raw_perm`'s selector takes the n-space route on this shape,
/// `dense` false standing for weighted input.
fn raw_perm_nspace(
    n: usize,
    n_folds: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    dense: bool,
) -> bool {
    crate::signal_test::raw_perm_route(n, n_folds, p, n_perm, &[k], None, !dense)
        .unwrap()
        .route
        == crate::signal_test::ReplicateRoute::Nspace
}

/// Whether `split_exact`'s selector takes the n-space route on this shape;
/// with `held`, the same whether the call keeps the gate's matrix or frees
/// it off the routes that do not read it.
fn split_exact_nspace(n: usize, p: usize, n_perm: usize, j: usize, k: usize, held: bool) -> bool {
    let at = |kept| {
        crate::signal_test::split_exact_route_choice(n, p, n_perm, j, &[k], None, false, held, kept)
            .unwrap()
            .route
            .route
            == crate::signal_test::ReplicateRoute::Nspace
    };
    let nspace = at(held);
    assert!(!held || at(false) == nspace, "kept and freed gate matrix");
    nspace
}

#[test]
fn raw_perm_nspace_respects_the_smallest_training_fold() {
    assert!(raw_perm_nspace(60, 5, 3000, 200, 1, true));
    // Two folds of one row train on one row: k = 1 is not below it.
    assert!(!raw_perm_nspace(2, 2, 3000, 200, 1, true));
    assert!(raw_perm_nspace(60, 5, 3000, 200, 2, true));
    assert!(!raw_perm_nspace(60, 5, 3000, 200, 2, false));
    assert!(!raw_perm_nspace(60, 5, 3000, 200, K_DUAL_MAX + 1, true));
    // Two folds of n = 2·r rows train on exactly r rows: eligible while
    // k ≤ r − 1.
    let r = K_DUAL_MAX + 1;
    assert!(raw_perm_nspace(2 * r, 2, 3000, 200, K_DUAL_MAX, true));
    assert!(!raw_perm_nspace(
        2 * (r - 1),
        2,
        3000,
        200,
        K_DUAL_MAX,
        true
    ));
    // Leave-one-out is eligible; more folds than rows, or one fold, is not.
    assert!(raw_perm_nspace(20, 20, 1500, 50, 2, true));
    assert!(!nspace_allowed_raw_perm(20, 21, 1500, 2, true));
    assert!(!nspace_allowed_raw_perm(20, 1, 1500, 2, true));
}

#[test]
fn raw_perm_nspace_rule_crosses_over_at_the_one_gram_cost() {
    // 60 x 100, 5 folds (n_tr 48, n_val 12), k = 2: 60²·100/2 +
    // 5·B'·(2·48² + 12·48) against 5·B'·(4·48 + 12)·100 pays from B' = 3.
    assert!(!raw_perm_nspace(60, 5, 100, 1, 2, true));
    assert!(raw_perm_nspace(60, 5, 100, 2, 2, true));
}

#[test]
fn split_exact_nspace_rule_crosses_over_at_the_one_gram_cost() {
    // 60 x 40, J = 10 (halves of 30), k = 2: without the gate's G the route
    // pays from B' = 3; holding it, from B' = 1.
    assert!(!split_exact_nspace(60, 40, 1, 10, 2, false));
    assert!(split_exact_nspace(60, 40, 2, 10, 2, false));
    assert!(split_exact_nspace(60, 40, 0, 10, 2, true));
}

#[test]
fn perm_null_nspace_rule_crosses_over_at_its_work() {
    // n = 60, p = 100, k = 2: 60²·100 + B·2·60² multiply-adds against the
    // primal's 2·B·2·60·100 pays from B = 22 (no Gram-p route at this B).
    let route = |b| {
        crate::perm_null::perm_null_route(60, 100, b, 2, false, false)
            .unwrap()
            .route
    };
    assert_eq!(
        route(21),
        crate::signal_test::ReplicateRoute::Primal(crate::signal_test::PrimalSource::Copy)
    );
    assert_eq!(route(22), crate::signal_test::ReplicateRoute::Nspace);
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

/// Columns that stop at different components in one call keep the outcome
/// each has alone: a zero column, one orthogonal to `X̃` (a zero row of `X̃`
/// makes `G e_0` exactly zero, so it stops at the first component), one in
/// the span of two eigenvectors of `G` (the `wn` exit at the third
/// component), and three ordinary outcomes around them. The bound covers
/// the `G·R` product's last bits; a lane that leaks into another gives O(1)
/// errors.
#[test]
fn kernel_cols_mixed_stops_match_single_columns() {
    let (n, p, k) = (40_usize, 120_usize, 5_usize);
    let mut xs = standardized(n, p, 11);
    xs.row_mut(0).fill(0.0);
    let gram = NspaceGram::new(xs.as_ref(), Par::Seq);
    let (_, u) = crate::linalg::self_adjoint_eigen(gram.g.as_ref(), Par::Seq).expect("eigen");
    let ordinary = |seed| outcome(&xs, 1.0, seed);
    let zs = [
        ordinary(500),
        Col::<f64>::zeros(n),
        Col::<f64>::from_fn(n, |i| if i == 0 { 1.0 } else { 0.0 }),
        Col::<f64>::from_fn(n, |i| u[(i, n - 1)] + 0.5 * u[(i, n - 2)]),
        ordinary(501),
        ordinary(502),
    ];
    let zs = Mat::<f64>::from_fn(n, zs.len(), |i, j| zs[j][i]);
    let ks: Vec<usize> = (1..=k).collect();
    let all = pls1_nspace_kernel_cols(&gram.block(), zs.as_ref(), &ks);
    let k_full: Vec<usize> = all
        .iter()
        .map(|o| match &o[k - 1] {
            NspaceOutcome::ZeroModel => 0,
            NspaceOutcome::Resolved { k_used, .. } => *k_used,
        })
        .collect();
    assert_eq!(k_full, [k, 0, 0, 2, k, k], "components kept at k = {k}");
    for (j, all_j) in all.iter().enumerate() {
        let one = pls1_nspace_kernel_cols(&gram.block(), zs.subcols(j, 1), &ks);
        for (ki, (a, b)) in all_j.iter().zip(&one[0]).enumerate() {
            match (a, b) {
                (NspaceOutcome::ZeroModel, NspaceOutcome::ZeroModel) => {}
                (
                    NspaceOutcome::Resolved {
                        alpha: a,
                        k_used: ka,
                    },
                    NspaceOutcome::Resolved {
                        alpha: b,
                        k_used: kb,
                    },
                ) => {
                    assert_eq!(ka, kb, "column {j}, k = {}", ki + 1);
                    let (a, b) = (col_vals(a.as_ref()), col_vals(b.as_ref()));
                    let scale = b.iter().fold(1.0_f64, |s, v| s.max(v.abs()));
                    let err = a
                        .iter()
                        .zip(&b)
                        .fold(0.0_f64, |e, (x, y)| e.max((x - y).abs()));
                    assert!(
                        err <= 1e-12 * scale,
                        "column {j}, k = {}: alpha differs by {err:e}",
                        ki + 1
                    );
                }
                _ => panic!("column {j}, k = {}: one call is the zero model", ki + 1),
            }
        }
    }
}
