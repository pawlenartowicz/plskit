//! Monte-Carlo H0 calibration of the weighted inference paths.
//!
//! Covers: weighted H0 FPR for every confirmatory method (split_nb, split_perm,
//! score, raw_perm, e) and the weighted sequential-deflation entry point
//! (`pls1_find_k_sequence`). The unweighted e-method test is a tripwire for
//! gross e-value inflation; the weighted tests exercise non-uniform observation
//! weights throughout.
//!
//! Under H0 (y ⟂ X) with NON-UNIFORM observation weights, a valid α-level test
//! rejects at rate ≤ α. We assert the empirical FPR over `N_REPS` seeded
//! replications stays within a one-sided binomial Monte-Carlo band of α.
//!
//! MC slack: the per-rep reject indicator is Bernoulli(p) with p ≤ α under H0,
//! so FPR_hat has SD ≤ √(α(1−α)/N_REPS). We allow a 3·SD upper band:
//!
//!     FPR_hat ≤ α + 3·√(α(1−α)/N_REPS).
//!
//! At α = 0.05, N_REPS = 200 this is 0.05 + 3·0.0154 ≈ 0.096. A correctly
//! calibrated test clears it with margin; a broken weighted path (e.g. a
//! convention mismatch inflating the statistic under permuted/split nulls)
//! blows past it. The band is intentionally one-sided — conservative tests
//! (FPR < α) are fine, only over-rejection is a failure.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
// Prose-heavy module docs: backticking every math token (α, n_reps, …) is noise.
#![allow(clippy::doc_markdown)]

use faer::{Col, Mat};
use plskit::{
    pls1_confirmatory_test, pls1_find_k_sequence, spls1_find_k_sequence, ConfirmatoryArgs,
    ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts, FindKSequenceOpts,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const ALPHA: f64 = 0.05;
const N_REPS: usize = 200;

/// 3·SD one-sided binomial Monte-Carlo upper band on the empirical FPR.
fn fpr_upper_band(alpha: f64, n_reps: usize) -> f64 {
    alpha + 3.0 * (alpha * (1.0 - alpha) / n_reps as f64).sqrt()
}

/// One H0 replication: y drawn independently of X (pure noise), plus fixed
/// non-uniform weights tied to row index. `rep` seeds the data so replications
/// are independent and reproducible.
fn null_data(n: usize, d: usize, rep: usize) -> (Mat<f64>, Col<f64>, Col<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(0x00C0_FFEE ^ rep as u64);
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    // y independent of X ⇒ H0 true.
    let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    // Deterministic, strongly non-uniform weights (range ~0.5..2.5). Tied to
    // row index, identical across replications — isolates the weighted path.
    let w = Col::<f64>::from_fn(n, |i| 0.5 + 2.0 * ((i as f64) / n as f64));
    (x, y, w)
}

/// Run the FPR loop for one method and assert the empirical rate clears the
/// MC band. `make_args` builds the per-method `ConfirmatoryArgs`; the test seed
/// is derived per-rep from `seed_base` so the resampling draws differ across
/// reps. Pass `weighted: true` for non-uniform observation weights, `false` for
/// `weights: None` (unweighted path). Pass `keep: Some(k)` to exercise the
/// sparse inner fitter; `None` for the dense path.
#[allow(clippy::too_many_arguments)]
fn assert_calibrated(
    n: usize,
    d: usize,
    k: usize,
    make_args: impl Fn() -> ConfirmatoryArgs,
    label: &str,
    weighted: bool,
    seed_base: u64,
    keep: Option<usize>,
) {
    let mut rejects = 0usize;
    for rep in 0..N_REPS {
        let (x, y, w) = null_data(n, d, rep);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k,
                weights: if weighted { Some(w.as_ref()) } else { None },
            },
            ConfirmatoryTestOpts {
                args: make_args(),
                seed: Some(seed_base + rep as u64),
                disable_parallelism: true,
                keep,
                ..Default::default()
            },
        )
        .unwrap();
        if r.pvalue <= ALPHA {
            rejects += 1;
        }
    }
    let fpr = rejects as f64 / N_REPS as f64;
    let band = fpr_upper_band(ALPHA, N_REPS);
    assert!(
        fpr <= band,
        "{label}: empirical FPR={fpr} (rejects={rejects}/{N_REPS}) exceeds MC band {band}"
    );
}

#[test]
fn split_nb_weighted_h0_fpr_within_band() {
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SplitNb { n_splits: 40 },
        "split_nb",
        true,
        1000,
        None,
    );
}

#[test]
fn split_perm_weighted_h0_fpr_within_band() {
    // split_perm is the costliest (n_perm × n_splits inner fits per rep); kept
    // small so the whole loop runs in a few seconds unignored.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SplitPerm {
            n_perm: 60,
            n_splits: 15,
        },
        "split_perm",
        true,
        1000,
        None,
    );
}

#[test]
fn score_weighted_h0_fpr_within_band() {
    // score is closed-form (no resampling): per-rep cost is a single Gram
    // eigendecomposition, so this loop is the fastest of the three.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::Score,
        "score",
        true,
        1000,
        None,
    );
}

#[test]
fn e_unweighted_h0_fpr_within_band() {
    // E-values are Markov-conservative: under H0 at n=40 the e-statistic sits
    // far below the rejection threshold (log_e ≈ −20..0 vs. ln(20) ≈ 3.0), so
    // expected FPR is ~0 and only the one-sided upper bound is meaningful.
    // This is a tripwire for GROSS e-value inflation (a regression inflating
    // log_e by ≳3 nats, e.g. a broken likelihood or wrong-order variance term),
    // not a sensitive detector of subtle σ²_alt-half regressions.
    assert_calibrated(40, 4, 1, || ConfirmatoryArgs::E, "e", false, 2000, None);
}

#[test]
fn raw_perm_weighted_h0_fpr_within_band() {
    // raw_perm: n_perm permutations of y over k-fold CV. Kept at n_perm=60,
    // n_folds=5 so the 200-rep loop stays under a few seconds.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::RawPerm {
            n_perm: 60,
            n_folds: 5,
        },
        "raw_perm",
        true,
        3000,
        None,
    );
}

#[test]
fn e_weighted_h0_fpr_within_band() {
    // E-values are Markov-conservative, so expected FPR ≈ 0 (see the unweighted
    // sibling above for the gross-inflation rationale). This weighted variant pins
    // that non-uniform observation weights do not break the e-value guarantee —
    // only the one-sided upper bound is meaningful here too.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::E,
        "e_weighted",
        true,
        4000,
        None,
    );
}

#[test]
fn find_k_sequence_weighted_h0_fpr_within_band() {
    // Pins the weighted sequential-deflation path: under H0, P(k_star ≥ 1) ≈ α.
    // Uses 100 reps (instead of 200) — the sequence test runs k_max=3 confirmatory
    // tests per rep, making each rep ~3× costlier than a single-component test.
    const REPS: usize = 100;
    let band = fpr_upper_band(ALPHA, REPS);
    let mut rejects = 0usize;
    for rep in 0..REPS {
        let (x, y, w) = null_data(40, 4, rep);
        let r = pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            Some(w.as_ref()),
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_splits: 15,
                // Rejection threshold inside the sequence test; must equal the
                // ALPHA the MC band is computed from.
                alpha: ALPHA,
                seed: Some(5000 + rep as u64),
                disable_parallelism: true,
                ..Default::default()
            },
        )
        .unwrap();
        if r.k_star >= 1 {
            rejects += 1;
        }
    }
    let fpr = rejects as f64 / REPS as f64;
    assert!(
        fpr <= band,
        "find_k_sequence: empirical FPR={fpr} (rejects={rejects}/{REPS}) exceeds MC band {band}"
    );
}

#[test]
fn split_nb_sparse_h0_fpr_within_band() {
    // Spec acceptance "inference reuse": split machinery is agnostic to the
    // fitter — sparse inner fits at keep=2 of d=4 stay calibrated under H0.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SplitNb { n_splits: 40 },
        "split_nb_sparse",
        false,
        6000,
        Some(2),
    );
}

#[test]
fn split_perm_sparse_h0_fpr_within_band() {
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SplitPerm {
            n_perm: 60,
            n_splits: 15,
        },
        "split_perm_sparse",
        false,
        7000,
        Some(2),
    );
}

#[test]
fn spls1_find_k_sequence_sparse_h0_fpr_within_band() {
    // Mirrors find_k_sequence_weighted_h0_fpr_within_band with the sparse
    // fitter: under H0, P(k_star ≥ 1) ≈ α through the sparse sequential path.
    const REPS: usize = 100;
    let band = fpr_upper_band(ALPHA, REPS);
    let mut rejects = 0usize;
    for rep in 0..REPS {
        let (x, y, _w) = null_data(40, 4, rep);
        let r = spls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            2,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_splits: 15,
                alpha: ALPHA,
                seed: Some(8000 + rep as u64),
                disable_parallelism: true,
                ..Default::default()
            },
        )
        .unwrap();
        if r.k_star >= 1 {
            rejects += 1;
        }
    }
    let fpr = rejects as f64 / REPS as f64;
    assert!(
        fpr <= band,
        "spls1_find_k_sequence: FPR={fpr} exceeds MC band {band}"
    );
}
