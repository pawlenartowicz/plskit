//! Unit tests for the n-space PLS1 route: eligibility, the Gram builders
//! and their thread-count invariance, the `‖G‖₂` bound, and the kernel.
#![allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::unreadable_literal
)]

use super::*;
use crate::fit::{pls1_fit_prepared_fro, ParChoice, PreparedFit};
use crate::linalg::{standardize, standardize1};
use crate::resample::block_par;
use faer::{Col, Mat, Par};
use rand::{RngExt, SeedableRng};

fn uniform(n: usize, p: usize, seed: u64) -> Mat<f64> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0))
}

fn assert_bits_eq(a: &Mat<f64>, b: &Mat<f64>, what: &str) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "{what}: shape"
    );
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_eq!(
                a[(i, j)].to_bits(),
                b[(i, j)].to_bits(),
                "{what}[({i}, {j})]"
            );
        }
    }
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
fn k_is_the_per_replicate_multiplicity_of_the_flop_rule() {
    // n = 60, p = 100, B = 100: at k = 1 the Gram costs more than it saves
    // (60·(100 + 100) = 12 000 ≥ 100·100), at k = 2 it pays
    // (60·(200 + 100) = 18 000 < 100·200).
    assert!(!nspace_eligible_perm_null(60, 100, 100, 1, true));
    assert!(nspace_eligible_perm_null(60, 100, 100, 2, true));
}

#[test]
fn gram_products_are_thread_count_invariant_at_the_route_shapes() {
    // (n_te, n_tr, p): the blocks this route builds in tests/byte_parity.rs
    // and for the corpus fixtures (perm_null 40 × 2000 and 60 × 3000;
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
        assert_bits_eq(
            &gram_of(xs.as_ref(), block_par(true)),
            &g_seq,
            &format!("G, {what}, serial"),
        );
        for threads in [2_usize, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let (g, m) = pool.install(|| {
                (
                    gram_of(xs.as_ref(), block_par(false)),
                    cross_of(xt.as_ref(), xs.as_ref(), block_par(false)),
                )
            });
            assert_bits_eq(&g, &g_seq, &format!("G, {what}, {threads} threads"));
            assert_bits_eq(&m, &m_seq, &format!("M, {what}, {threads} threads"));
        }
    }
}

/// `‖G‖₂` of `G = xs·xs'`: the largest squared singular value of `xs`.
fn spectral(xs: &Mat<f64>) -> f64 {
    let s = xs.thin_svd().expect("svd").S().column_vector().to_owned();
    let top = (0..s.nrows()).map(|i| s[i]).fold(0.0_f64, f64::max);
    top * top
}

#[test]
fn gram_norm_bound_brackets_the_spectral_norm() {
    let base = uniform(20, 1500, 3);
    let designs: Vec<(&str, Mat<f64>)> = vec![
        ("40 x 2000", standardize(uniform(40, 2000, 1).as_ref()).0),
        ("60 x 3000", standardize(uniform(60, 3000, 2).as_ref()).0),
        (
            "duplicated rows",
            standardize(Mat::<f64>::from_fn(40, 1500, |i, j| base[(i % 20, j)]).as_ref()).0,
        ),
        (
            "rank 3",
            standardize((uniform(30, 3, 4).as_ref() * uniform(3, 800, 5).as_ref()).as_ref()).0,
        ),
    ];
    for (what, xs) in &designs {
        let x_fro = xs.norm_l2();
        let g = gram_of(xs.as_ref(), Par::Seq);
        let bound = gram_norm_bound(g.as_ref(), x_fro);
        let exact = spectral(xs);
        assert!(bound >= exact, "{what}: {bound:e} < ‖G‖₂ = {exact:e}");
        assert!(
            bound <= 2.0 * exact * (1.0 + 1e-12),
            "{what}: {bound:e} > 2‖G‖₂"
        );
        assert!(bound <= x_fro * x_fro, "{what}: above ‖X̃‖_F²");
        // Sequential and deterministic.
        assert_eq!(
            bound.to_bits(),
            gram_norm_bound(g.as_ref(), x_fro).to_bits(),
            "{what}"
        );
    }
    // A zero Gram and a NaN Gram fall back to ‖X̃‖_F².
    assert_eq!(
        gram_norm_bound(Mat::<f64>::zeros(5, 5).as_ref(), 0.0).to_bits(),
        0.0_f64.to_bits()
    );
    let mut nan = Mat::<f64>::zeros(5, 5);
    nan[(1, 1)] = f64::NAN;
    assert_eq!(
        gram_norm_bound(nan.as_ref(), 3.0).to_bits(),
        9.0_f64.to_bits()
    );
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

/// Primal-side absolute error of `‖X_a'y_a‖` as a multiple of
/// `w_rel_floor`, adopted from the floor's measured margin: the larger of
/// the two margins measured in `fit::w_rel_floor`'s doc, `0.020×` for the
/// first noise component over all components and `0.33×` for an outcome
/// orthogonal to `X` at the first component.
const PRIMAL_W_MARGIN: f64 = 0.33;

#[test]
fn primal_margin_leaves_the_floor_gate_a_factor_two() {
    // Past gate 1 the exact wn is at least √(1 − 1/BOUND_BAND) of the
    // computed one; past gate 3 the computed one is RESOLVE_BAND × floor.
    let exact_min = (1.0 - 1.0 / BOUND_BAND).sqrt() * RESOLVE_BAND;
    assert!(exact_min - PRIMAL_W_MARGIN >= 2.0, "{exact_min}");
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

#[test]
fn a_non_finite_gram_is_unresolved() {
    let xs = standardized(20, 500, 5);
    let mut g = gram_of(xs.as_ref(), Par::Seq);
    g[(0, 0)] = f64::NAN;
    let x_fro = xs.norm_l2();
    let block = NspaceBlock {
        g: g.as_ref(),
        x_fro,
        g2: gram_norm_bound(g.as_ref(), x_fro),
        p: 500,
    };
    let z = outcome(&xs, 1.0, 105);
    assert!(matches!(
        pls1_nspace_kernel(&block, z.as_ref(), 2),
        NspaceOutcome::Unresolved
    ));
}

#[test]
fn out_of_contract_shapes_are_unresolved() {
    let xs = standardized(8, 200, 6);
    let gram = NspaceGram::new(xs.as_ref(), Par::Seq);
    let z = outcome(&xs, 1.0, 106);
    for k in [0_usize, 8, 9] {
        assert!(
            matches!(
                pls1_nspace_kernel(&gram.block(), z.as_ref(), k),
                NspaceOutcome::Unresolved
            ),
            "k={k}"
        );
    }
    let z_wrong_rows = outcome(&standardized(9, 200, 6), 1.0, 106);
    assert!(matches!(
        pls1_nspace_kernel(&gram.block(), z_wrong_rows.as_ref(), 2),
        NspaceOutcome::Unresolved
    ));
}

/// Printed by `python3 scripts/gate_feasibility.py nspace --reference`.
const SPIKE_G2: f64 = 2303.608226002213;
/// `(E_w(a), E_t(a), rho_a)` for `a = 1..=4`.
const SPIKE_REFERENCE: [(f64, f64, f64); 4] = [
    (
        5.563549621001585e-10,
        7.735252438578756e-06,
        1.8070190394459611e-12,
    ),
    (
        3.4107066852834623e-08,
        8.744123329737737e-05,
        2.3776744383566945e-10,
    ),
    (
        4.416140783316025e-06,
        0.010184348434120104,
        1.7510497446785014e-07,
    ),
    (
        0.0032314024669681933,
        7.443899012915223,
        8.529328862808795e-05,
    ),
];

#[test]
fn history_bounds_reproduce_the_feasibility_spike() {
    // `reference_design` of scripts/gate_feasibility.py, rebuilt here; the
    // kernel's per-component bounds must equal the script's (up to BLAS and
    // libm rounding), so the spike that set K_DUAL_MAX ran this recursion.
    let (n, p) = (24_usize, 300_usize);
    let x = Mat::<f64>::from_fn(n, p, |i, j| {
        ((i as f64 + 1.0) * (j as f64 + 1.0) * 0.37).sin()
            + 0.5 * (1.1 * i as f64 - 0.7 * j as f64).cos()
    });
    let xs = standardize(x.as_ref()).0;
    let y = Col::<f64>::from_fn(n, |i| (0.9 * i as f64 + 0.3).cos() + 0.5 * xs[(i, 0)]);
    let z = standardize1(y.as_ref()).0;
    let gram = NspaceGram::new(xs.as_ref(), Par::Seq);
    let block = gram.block();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs();
    assert!(
        close(block.g2, SPIKE_G2),
        "‖G‖₂ bound {:e} vs {SPIKE_G2:e}",
        block.g2
    );
    let (out, trace) = pls1_nspace_kernel_traced(&block, z.as_ref(), 4);
    assert!(matches!(out, NspaceOutcome::Resolved { k_used: 4, .. }));
    assert_eq!(trace.len(), 4);
    for (a, (t, &(e_w, e_t, rho))) in trace.iter().zip(SPIKE_REFERENCE.iter()).enumerate() {
        assert!(close(t.e_w, e_w), "a={}: E_w {:e} vs {e_w:e}", a + 1, t.e_w);
        assert!(close(t.e_t, e_t), "a={}: E_t {:e} vs {e_t:e}", a + 1, t.e_t);
        assert!(close(t.rho, rho), "a={}: rho {:e} vs {rho:e}", a + 1, t.rho);
    }
}
