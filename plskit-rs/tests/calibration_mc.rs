//! Monte-Carlo H0 calibration of the weighted inference paths.
//!
//! Covers: weighted H0 FPR for every confirmatory method (split_nb, split_exact,
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
    linalg::{stable_rank, standardize_weighted},
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

/// One H0 replication with a concentrated spectrum: every column is the same
/// latent factor `f` plus a tiny jitter, so stable rank sits just above 1 —
/// well under `split_nb`'s gate floor of 3 (mirrors `synth_one_factor` in
/// `signal_test.rs`'s auto-gate tests). Weights follow `null_data`'s
/// non-uniform pattern.
#[allow(clippy::many_single_char_names)]
fn concentrated_null_data(n: usize, d: usize, rep: usize) -> (Mat<f64>, Col<f64>, Col<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0DE_00C0_FFEE ^ rep as u64);
    let f = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let x = Mat::<f64>::from_fn(n, d, |i, _| f[i] + 0.01 * rng.random_range(-1.0..1.0));
    // y independent of X ⇒ H0 true.
    let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
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
        // `force` — this suite measures NB's own false-positive rate, so NB has
        // to run on every replication. Without it most of this cell would not be
        // measuring NB at all: over 500 `null_data(40, 4, ·)` draws the stable
        // rank of the weighted-standardized X sits below the gate's floor of 3
        // in 59% of them (median 2.94, range 2.16–3.68), so the majority of reps
        // would silently swap to split_exact and the reported FPR would be a
        // blend of two methods. The design sits right on the floor because
        // d = 4 caps the stable rank at 4 and sampling noise in the top
        // eigenvalue costs about a further unit.
        || ConfirmatoryArgs::SplitNb {
            n_splits: 40,
            force: true,
        },
        "split_nb",
        true,
        1000,
        None,
    );
}

#[test]
fn split_exact_weighted_h0_fpr_within_band() {
    // split_exact is the costliest (n_perm × n_splits inner fits per rep); kept
    // small so the whole loop runs in a few seconds unignored.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SplitExact {
            n_perm: 60,
            n_splits: 15,
        },
        "split_exact",
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
    // Pins the weighted sequential-deflation path under split_nb: under H0,
    // P(k_star ≥ 1) ≈ α. Uses 100 reps (instead of 200) — the sequence test runs
    // k_max=3 confirmatory tests per rep, making each rep ~3× costlier than a
    // single-component test.
    //
    // d = 8, not the 4 the other weighted cells use, so that NB actually runs:
    // the auto-gate reroutes any X with ≤ 4 columns outright, and at d = 5 the
    // weighted stable rank falls under the floor of 3 in 7% of these draws
    // (min 2.77 over 100 reps) — a coin flip that would blend two methods into
    // one FPR. At d = 8 the weighted rank spans 3.79–5.38, and Kish n_eff = 34.7
    // clears the size floor, so every rep is an unforced NB run. Forcing instead
    // of widening would have hidden the gate rather than satisfied it.
    // The per-rep `test_method` assertion below is what keeps that true.
    const REPS: usize = 100;
    let band = fpr_upper_band(ALPHA, REPS);
    let mut rejects = 0usize;
    for rep in 0..REPS {
        let (x, y, w) = null_data(40, 8, rep);
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
        assert_eq!(
            r.test_method, "split_nb",
            "rep {rep} was rerouted — this cell must measure NB, not split_exact"
        );
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
        // `force` for the same reason as `split_nb_weighted_h0_fpr_within_band`
        // — unweighted here, and the gate floor catches 56% of the same draws.
        || ConfirmatoryArgs::SplitNb {
            n_splits: 40,
            force: true,
        },
        "split_nb_sparse",
        false,
        6000,
        Some(2),
    );
}

#[test]
fn split_exact_sparse_h0_fpr_within_band() {
    // k = 1 with `keep`: sparse `keep` breaks the no-refit shortcut even at
    // K = 1 (the selected column set moves with every permutation), so this
    // cell exercises split_exact's refit route — the dense k=1
    // split_exact_weighted_h0_fpr_within_band cell above takes the no-refit
    // route instead.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SplitExact {
            n_perm: 60,
            n_splits: 15,
        },
        "split_exact_sparse",
        false,
        7000,
        Some(2),
    );
}

#[test]
fn split_exact_concentrated_spectrum_h0_fpr_within_band() {
    // The gate's premise, under test directly: on a concentrated spectrum
    // (single dominant factor, stable rank < 3) split_nb's Fisher-z
    // correction drifts off level (that is exactly why the auto-gate reroutes
    // to split_exact here — see signal_test.rs's `gate_reroutes_when_stable_
    // rank_below_floor`), while split_exact calibrates by permutation and
    // must hold level regardless of the spectrum. n = 40 ≥ the gate's n_eff
    // floor of 25, so only the spectrum condition is in play.
    let (x0, _, w0) = concentrated_null_data(40, 5, 0);
    let (xs, ..) = standardize_weighted(x0.as_ref(), Some(w0.as_ref()));
    let sr = stable_rank(xs.as_ref());
    assert!(sr < 3.0, "design check: stable_rank={sr}, want < 3.0");

    let mut rejects = 0usize;
    for rep in 0..N_REPS {
        let (x, y, w) = concentrated_null_data(40, 5, rep);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: Some(w.as_ref()),
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 60,
                    n_splits: 15,
                },
                seed: Some(9000 + rep as u64),
                disable_parallelism: true,
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
        "split_exact_concentrated_spectrum: empirical FPR={fpr} (rejects={rejects}/{N_REPS}) exceeds MC band {band}"
    );
}

#[test]
fn spls1_find_k_sequence_sparse_h0_fpr_within_band() {
    // Mirrors find_k_sequence_weighted_h0_fpr_within_band with the sparse
    // fitter: under H0, P(k_star ≥ 1) ≈ α through the sparse sequential path.
    //
    // Left at d = 4 on purpose, so it reroutes to split_exact on every rep.
    // Widening it too would change `keep = 2` from "half the columns" to a much
    // sparser ratio — a second axis of change this cell was not written to
    // vary. The dense sibling above covers the sequence path under NB; what
    // this one pins is that the sparse deflation chain holds its level whatever
    // method the gate hands it, which is the method-agnostic half of the claim.
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
