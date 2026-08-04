//! Weighted-path coverage at the inference boundary (spec G1, bullets 1 & 3).
//!
//! Per-method parity-vs-calibration determination (see each test for the
//! justification grounded in the source):
//!
//! - `pls1_perm_null.beta_ref` → PARITY. `beta_ref` is the full-data reference
//!   fit (`fit_ref` in perm_null.rs), and it lives on the *standardized* scale —
//!   a regression coefficient, scale-free in the total weight. Mean-1 weights
//!   (Σw = n) and row duplication (Σ1 = total) give bit-equal weighted moments,
//!   so β matches. (The permutation SD/z are resampling-dependent and NOT
//!   asserted here.)
//! - `pls1_find_k_optimal` with `Selector::Bic` → PARITY on k_star only.
//!   `select_bic` draws no RNG, but its bic_scores use `n_eff` directly
//!   (`n_eff·log(SSR/n_eff) + k·log(n_eff)`): mean-1 weights carry n_eff = n,
//!   row duplication carries n_eff = total, so the *scores* legitimately differ
//!   by the sample-size inflation. The selected k_star (argmin) is what the
//!   entry point returns and is replication-stable — that is the parity claim.
//! - cross-entry-point consistency → weighted `pls1_perm_null(..).beta_ref`
//!   equals weighted `pls1_fit(..).coef` (standardized scale). Pins the
//!   convention unification (perm_null standardizes with weighted moments then
//!   fits pre_standardized; pls1_fit standardizes internally — same coef).
//!
//! Methods deliberately NOT given a parity test here (split_nb/split_exact/score
//! are FPR-calibrated in `calibration_mc.rs`; raw_perm, e, and the find_k
//! selectors get weighted end-to-end tests below):
//! - `score` → its statistic `T = ‖X̃'ỹ‖²` and the χ² p-value scale with the
//!   *total* weight: row duplication inflates n_eff from n to total, so weighted
//!   and duplicated p-values genuinely differ (more rows ⇒ more power). The test
//!   is sample-size-dependent by construction, not replication-equivalent.
//! - `raw_perm` / `split_nb` / `split_exact` and the CV/sequence find_k selectors
//!   draw folds/splits/permutations from the RNG, so the observed statistic is
//!   not replication-equivalent (different draws for n vs n_dup).

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
// Prose-heavy module/test docs: backticking every math token (n, β, n_dup, …)
// would bury the explanation in noise.
#![allow(clippy::doc_markdown)]

use faer::{Col, ColRef, Mat, MatRef};
use plskit::{
    pls1_confirmatory_test, pls1_find_k_optimal, pls1_find_k_sequence, pls1_fit, pls1_perm_null,
    ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts,
    FindKOptimalOpts, FindKSequenceOpts, FitOpts, KSpec, PermNullOpts, Selector,
};

/// Seeded synthetic data with planted signal in the first feature.
fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| x[(i, 0)] * snr + noise[i]);
    (x, y)
}

/// Expand `(x, y)` by integer weights into row-duplicated `(x_dup, y_dup)`.
fn duplicate_rows(x: MatRef<'_, f64>, y: ColRef<'_, f64>, w_int: &[u32]) -> (Mat<f64>, Col<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let total: usize = w_int.iter().map(|&w| w as usize).sum();
    let mut x_dup = Mat::<f64>::zeros(total, p);
    let mut y_dup = Col::<f64>::zeros(total);
    let mut row = 0;
    for i in 0..n {
        for _ in 0..w_int[i] {
            for j in 0..p {
                x_dup[(row, j)] = x[(i, j)];
            }
            y_dup[row] = y[i];
            row += 1;
        }
    }
    (x_dup, y_dup)
}

/// Small-integer weights covering all n rows (cycled).
fn integer_weights(n: usize) -> Vec<u32> {
    let pattern = [1u32, 2, 1, 3, 1, 1, 2, 1];
    (0..n).map(|i| pattern[i % pattern.len()]).collect()
}

#[test]
fn perm_null_beta_ref_integer_weights_match_row_duplication() {
    // PARITY: beta_ref is the full-data reference fit (deterministic). The
    // permutation SD/z are resampling-dependent and intentionally not asserted.
    let (x, y) = synth(40, 5, 3.0, 2);
    let w_int = integer_weights(40);
    let w = Col::<f64>::from_fn(40, |i| f64::from(w_int[i]));

    let opts = |n_perm| PermNullOpts {
        n_perm,
        return_perm_matrix: false,
        pre_standardized: false,
        disable_parallelism: true,
        verbose: false,
    };

    let weighted = pls1_perm_null(
        x.as_ref(),
        y.as_ref(),
        2,
        Some(w.as_ref()),
        opts(100),
        Some(7),
    )
    .unwrap();

    let (x_dup, y_dup) = duplicate_rows(x.as_ref(), y.as_ref(), &w_int);
    let dup = pls1_perm_null(x_dup.as_ref(), y_dup.as_ref(), 2, None, opts(100), Some(7)).unwrap();

    assert_eq!(weighted.beta_ref.len(), dup.beta_ref.len());
    for j in 0..weighted.beta_ref.len() {
        assert!(
            (weighted.beta_ref[j] - dup.beta_ref[j]).abs() < 1e-12,
            "beta_ref[{j}] differs: weighted={} dup={}",
            weighted.beta_ref[j],
            dup.beta_ref[j]
        );
    }
}

#[test]
fn find_k_optimal_bic_integer_weights_match_row_duplication() {
    // PARITY on k_star only. select_bic draws no RNG, but its bic_scores use
    // n_eff directly: weighted carries n_eff = n, row duplication carries
    // n_eff = total, so the scores differ by the sample-size inflation. The
    // argmin k_star is the entry point's output and is replication-stable; that
    // is the parity claim. The per-feature β the fit is selecting over is
    // scale-free (cf. perm_null beta_ref above), so the same k wins.
    let (x, y) = synth(40, 5, 3.0, 3);
    let w_int = integer_weights(40);
    let w = Col::<f64>::from_fn(40, |i| f64::from(w_int[i]));

    let opts = || FindKOptimalOpts {
        selector: Selector::Bic,
        seed: Some(13),
        ..Default::default()
    };

    let weighted =
        pls1_find_k_optimal(x.as_ref(), y.as_ref(), 4, Some(w.as_ref()), opts()).unwrap();

    let (x_dup, y_dup) = duplicate_rows(x.as_ref(), y.as_ref(), &w_int);
    let dup = pls1_find_k_optimal(x_dup.as_ref(), y_dup.as_ref(), 4, None, opts()).unwrap();

    assert_eq!(
        weighted.k_star, dup.k_star,
        "BIC k_star differs: weighted={} dup={}",
        weighted.k_star, dup.k_star
    );
}

#[test]
fn perm_null_beta_ref_matches_weighted_fit_coef() {
    // Cross-entry-point consistency (bullet 3): the standardized-scale reference
    // coefficient is the same object whether reached via pls1_perm_null
    // (standardize_weighted → fit pre_standardized) or via a direct weighted
    // pls1_fit (standardizes internally). Permanent pin of the convention
    // unification. NOT bit-exact (measured: ≤2 ulp): perm_null hands pls1_fit
    // already-normalized weights, and the fit normalizes again — mean-1
    // normalization is idempotent only to rounding, so the √w′ row-scaling
    // factors differ in the last bit. 1e-15 absorbs exactly that.
    let (x, y) = synth(50, 6, 2.5, 4);
    // Non-uniform, non-integer weights: stresses the weighted-moment path.
    let w = Col::<f64>::from_fn(50, |i| 0.5 + (i as f64).sin().abs());

    let pn = pls1_perm_null(
        x.as_ref(),
        y.as_ref(),
        2,
        Some(w.as_ref()),
        PermNullOpts {
            n_perm: 100,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        },
        Some(7),
    )
    .unwrap();

    let fit = pls1_fit(
        x.as_ref(),
        y.as_ref(),
        KSpec::Fixed(2),
        Some(w.as_ref()),
        FitOpts::default(),
    )
    .unwrap();

    assert_eq!(pn.beta_ref.len(), fit.coef.nrows());
    for j in 0..pn.beta_ref.len() {
        assert!(
            (pn.beta_ref[j] - fit.coef[j]).abs() < 1e-15,
            "beta_ref[{j}]={} != coef[{j}]={}",
            pn.beta_ref[j],
            fit.coef[j]
        );
    }
}

// ── Weighted end-to-end coverage for the resampling entry points ─────────────
//
// These complete the Accept-clause matrix for paths whose observed statistic is
// resampling-dependent (folds/splits/permutations drawn from the RNG) and so are
// not replication-equivalent: parity is meaningless, FPR calibration covers what
// is testable (calibration_mc.rs), and these pin that the weighted path runs
// end-to-end with non-uniform weights and returns a valid result. The `n_eff`
// field is the cheap observable invariant: Kish's effective sample size must sit
// strictly between 1 and n for genuinely non-uniform weights.

/// Strongly non-uniform weights, mean ≈ 1, every weight positive.
fn nonuniform_weights(n: usize) -> Col<f64> {
    Col::<f64>::from_fn(n, |i| 0.5 + (i as f64).cos().abs())
}

#[test]
fn raw_perm_weighted_runs_end_to_end() {
    // raw_perm: CV folds + permuted-y nulls are RNG-drawn ⇒ resampling, not
    // parity. Pin that the weighted path produces a bounded p and a reduced
    // n_eff. (Supersedes the deferred TODO(F11) internal parity attempt: the
    // weighted raw_perm path is now exercised through the public API.)
    let (x, y) = synth(40, 5, 3.0, 5);
    let w = nonuniform_weights(40);
    let r = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 2,
            weights: Some(w.as_ref()),
        },
        ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 100,
                n_folds: 5,
            },
            seed: Some(7),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.method, "raw_perm");
    assert!((0.0..=1.0).contains(&r.pvalue));
    assert!(r.n_eff > 1.0 && r.n_eff < 40.0, "n_eff={}", r.n_eff);
}

#[test]
fn e_weighted_runs_end_to_end() {
    // e: single random train/test split ⇒ resampling, not parity. e-method H0
    // calibration is deferred to spec G4 (calibration_mc.rs); here we pin the
    // weighted path returns a bounded p with reduced n_eff.
    let (x, y) = synth(60, 5, 3.0, 6);
    let w = nonuniform_weights(60);
    let r = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: 2,
            weights: Some(w.as_ref()),
        },
        ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::E,
            seed: Some(5),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.method, "e");
    assert!((0.0..=1.0).contains(&r.pvalue));
    assert!(r.n_eff > 1.0 && r.n_eff < 60.0, "n_eff={}", r.n_eff);
}

#[test]
fn find_k_sequence_weighted_runs_end_to_end() {
    // find_k_sequence drives the per-step split_nb test (RNG-drawn splits) ⇒
    // resampling. Pin that the weighted path returns a full pvalue vector and a
    // reduced n_eff.
    let (x, y) = synth(60, 5, 4.0, 7);
    let w = nonuniform_weights(60);
    let r = pls1_find_k_sequence(
        x.as_ref(),
        y.as_ref(),
        3,
        Some(w.as_ref()),
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitNb,
            n_splits: 30,
            alpha: 0.05,
            seed: Some(13),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.pvalues.nrows(), 3);
    assert_eq!(r.test_method, "split_nb");
    assert!(r.n_eff > 1.0 && r.n_eff < 60.0, "n_eff={}", r.n_eff);
}

#[test]
fn find_k_sequence_weighted_deflation_not_inflated() {
    // Regression pin for the weighted deflation in p_for_incremental: T·P′
    // lives on the √w′-row-scaled problem, so deflating unscaled Xs by T·P′
    // directly leaks (1−√w′ᵢ) of each removed component back into the step-h
    // test. With strongly non-uniform weights (alternating 2.0/0.25) that leak
    // made every deflated step "significant" and drove the weighted k_star to
    // k_max=4 while the unweighted run on the same data stopped at 2. The
    // corrected √W⁻¹-deflated residual must reach the same structural verdict
    // as the unweighted run. (A duplication-parity oracle is impossible here —
    // the per-step splits are RNG-drawn over different row counts; see module
    // doc — so equality of k_star on a fixed seed is the strongest pin.)
    let (x, y) = synth(60, 5, 6.0, 11);
    let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 2.0 } else { 0.25 });
    let opts = FindKSequenceOpts {
        test_method: ConfirmatoryMethod::SplitNb,
        n_splits: 30,
        alpha: 0.05,
        seed: Some(19),
        ..Default::default()
    };
    let weighted = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 4, Some(w.as_ref()), opts).unwrap();
    let unweighted = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 4, None, opts).unwrap();
    assert_eq!(
        weighted.k_star, unweighted.k_star,
        "weighted pvalues={:?}, unweighted pvalues={:?}",
        weighted.pvalues, unweighted.pvalues
    );
    assert_eq!(weighted.k_star, 2, "pvalues={:?}", weighted.pvalues);
}

#[test]
fn find_k_optimal_cv_weighted_runs_end_to_end() {
    // R2Se selector: CV folds are RNG-drawn ⇒ resampling (the BIC selector's
    // deterministic parity is covered above). Pin the weighted CV path returns
    // k_star, cv_scores, and a reduced n_eff.
    let (x, y) = synth(60, 5, 4.0, 8);
    let w = nonuniform_weights(60);
    let r = pls1_find_k_optimal(
        x.as_ref(),
        y.as_ref(),
        4,
        Some(w.as_ref()),
        FindKOptimalOpts {
            selector: Selector::R2Se,
            n_folds: 5,
            seed: Some(17),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(r.k_star >= 1);
    assert!(r.cv_scores.is_some());
    assert!(r.cv_scores_se.is_some());
    assert!(r.n_eff > 1.0 && r.n_eff < 60.0, "n_eff={}", r.n_eff);
}
