//! Weighted-path coverage at the inference boundary.
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
//! - `pls1_find_k_optimal` with `Selector::Bic` → PARITY on k_star and on the
//!   SSR behind every bic score. `select_bic` draws no RNG, but its bic_scores
//!   use `n_eff` directly (`n_eff·log(SSR/n_eff) + k·log(n_eff)`) and the SSR
//!   is formed with mean-1 weights, so the *scores* legitimately differ from
//!   row duplication's. Inverting the formula recovers each SSR, and
//!   SSR_weighted · (total / n) = SSR_dup is replication-invariant.
//! - cross-entry-point consistency → weighted `pls1_perm_null(..).beta_ref`
//!   equals weighted `pls1_fit(..).coef` (standardized scale). Pins that both
//!   paths give the same coef (perm_null standardizes with weighted moments
//!   then fits pre_standardized; pls1_fit standardizes internally).
//!
//! Methods deliberately NOT given a parity test here (split_nb/split_exact/score
//! are FPR-calibrated in `calibration_mc.rs`; weighted raw_perm / e numbers are
//! pinned by the corpus):
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

mod common;

use common::duplicate_rows;
use faer::{Col, Mat};
use plskit::{
    pls1_find_k_optimal, pls1_find_k_sequence, pls1_fit, pls1_perm_null, ConfirmatoryMethod,
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
    // PARITY on k_star and on the SSR behind each score. select_bic draws no
    // RNG, but its bic_scores use n_eff directly (Kish n_eff weighted, total
    // for row duplication), so the scores themselves differ by the sample-size
    // inflation. The per-feature β the fit is selecting over is scale-free
    // (cf. perm_null beta_ref above), so the same k wins.
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

    // SSR recovered from each score (inverse of bic = n_eff·ln(SSR/n_eff) +
    // k·ln(n_eff)). select_bic sums w·r² with mean-1 weights, i.e.
    // (n / total)·Σ w_int·r², which is (n / total)·SSR_dup when the weighted
    // and duplicated fits agree. Ignoring the weights breaks it.
    let ssr =
        |bic: f64, k: usize, n_eff: f64| n_eff * ((bic - k as f64 * n_eff.ln()) / n_eff).exp();
    let total = f64::from(w_int.iter().sum::<u32>());
    let (bw, bd) = (
        weighted.bic_scores.as_ref().expect("bic scores"),
        dup.bic_scores.as_ref().expect("bic scores"),
    );
    assert!(!bw.is_empty());
    assert_eq!(bw.keys().collect::<Vec<_>>(), bd.keys().collect::<Vec<_>>());
    for (&k, &b) in bw {
        let got = ssr(b, k, weighted.n_eff) * (total / 40.0);
        let want = ssr(bd[&k], k, dup.n_eff);
        assert!(
            (got - want).abs() <= 1e-9 * want.abs(),
            "k={k}: weighted SSR·total/n={got} dup SSR={want}"
        );
    }
}

#[test]
fn perm_null_beta_ref_matches_weighted_fit_coef() {
    // Cross-entry-point consistency (bullet 3): the standardized-scale reference
    // coefficient is the same object whether reached via pls1_perm_null
    // (standardize_weighted → fit pre_standardized) or via a direct weighted
    // pls1_fit (standardizes internally). NOT bit-exact (measured: ≤2 ulp): perm_null hands pls1_fit
    // already-normalized weights, and the fit normalizes again — mean-1
    // normalization is idempotent only to rounding, so the √w′ row-scaling
    // factors differ in the last bit. 1e-12 covers that on any platform.
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
            (pn.beta_ref[j] - fit.coef[j]).abs() < 1e-12,
            "beta_ref[{j}]={} != coef[{j}]={}",
            pn.beta_ref[j],
            fit.coef[j]
        );
    }
}

#[test]
fn find_k_sequence_weighted_deflation_not_inflated() {
    // Pins the weighted deflation in p_for_incremental: T·P′ lives on the
    // √w′-row-scaled problem, so deflating unscaled Xs by T·P′ directly would
    // leak (1−√w′ᵢ) of each removed component back into the step-h test. With
    // strongly non-uniform weights (alternating 2.0/0.25) that leak would make
    // every deflated step "significant" and drive the weighted k_star to
    // k_max=4 while the unweighted run on the same data stops at 2. The
    // √W⁻¹-deflated residual must reach the same structural verdict as the
    // unweighted run. (A duplication-parity oracle is impossible here —
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

/// A CV score map as `(k, bit pattern)` pairs, for exact comparison.
fn score_bits(scores: Option<&std::collections::BTreeMap<usize, f64>>) -> Vec<(usize, u64)> {
    scores
        .expect("CV selectors return the score map")
        .iter()
        .map(|(&k, v)| (k, v.to_bits()))
        .collect()
}

#[test]
fn find_k_optimal_cv_ignores_zero_weight_rows() {
    // PROPERTY: a row of weight exactly 0 cannot move the weighted CV result,
    // so overwriting the zero-weight rows of (X, y) with unrelated values
    // leaves k_star, cv_scores and cv_scores_se unchanged bit for bit. The
    // fold layout is a shuffle of 0..n from the seed alone, and the weighted
    // fold moments, the √w-scaled fold fit and the weighted validation sums
    // each take such a row as an exact `0.0 * finite` term. (The
    // standardizer's power-of-two prescale reads the overwritten entries, and
    // is exact.) A CV loop that dropped the weights at the fold
    // standardization, the fold fit or the validation R² lets the overwritten
    // rows through; the unweighted control at the end shows how far they move
    // the scores.
    let n = 60;
    let (x, y) = synth(n, 5, 3.0, 5);
    // Six exact zeros among non-uniform positive weights: fewer than the 12
    // rows of a fold, so no fold's weight slice sums to zero (such a slice
    // runs unweighted).
    let zeroed = |i: usize| i % 10 == 3;
    let w = Col::<f64>::from_fn(n, |i| {
        if zeroed(i) {
            0.0
        } else {
            0.5 + (i as f64).sin().abs()
        }
    });
    // Finite and of ordinary magnitude, so `0.0 * value` is exactly 0 and
    // nothing overflows.
    let (x_alt, y_alt) = {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
        let x_alt = Mat::<f64>::from_fn(n, 5, |i, j| {
            if zeroed(i) {
                rng.random_range(-50.0..50.0)
            } else {
                x[(i, j)]
            }
        });
        let y_alt = Col::<f64>::from_fn(n, |i| {
            if zeroed(i) {
                rng.random_range(-50.0..50.0)
            } else {
                y[i]
            }
        });
        (x_alt, y_alt)
    };

    let run = |x: &Mat<f64>, y: &Col<f64>, weights: Option<&Col<f64>>| {
        pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            4,
            weights.map(Col::as_ref),
            FindKOptimalOpts {
                selector: Selector::R2Se,
                seed: Some(21),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let base = run(&x, &y, Some(&w));
    let swapped = run(&x_alt, &y_alt, Some(&w));
    assert!(base.k_star >= 1, "no component selected: nothing compared");
    assert_eq!(base.k_star, swapped.k_star);
    assert_eq!(
        score_bits(base.cv_scores.as_ref()),
        score_bits(swapped.cv_scores.as_ref()),
        "cv_scores moved with the zero-weight rows"
    );
    assert_eq!(
        score_bits(base.cv_scores_se.as_ref()),
        score_bits(swapped.cv_scores_se.as_ref()),
        "cv_scores_se moved with the zero-weight rows"
    );

    // Control: without weights the same overwrite moves the scores.
    let plain = run(&x, &y, None);
    let plain_alt = run(&x_alt, &y_alt, None);
    let pa = plain.cv_scores.as_ref().expect("cv scores");
    let pb = plain_alt.cv_scores.as_ref().expect("cv scores");
    let moved = pa
        .iter()
        .map(|(k, v)| (v - pb[k]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        moved > 1e-3,
        "unweighted control: cv_scores moved by only {moved}"
    );
}
