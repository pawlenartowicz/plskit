//! Monte-Carlo H0 calibration of the inference paths.
//!
//! One cell per test; each draws data under H0 and checks the empirical
//! false-positive rate (FPR):
//!
//! - Weighted PLS1 cells (non-uniform observation weights): every
//!   confirmatory method (split_nb, split_exact, score, raw_perm, e),
//!   split_exact at k = 2 (its weighted refit route), split_exact on a
//!   concentrated spectrum, and the sequential-deflation entry point
//!   (`pls1_find_k_sequence`).
//! - Unweighted PLS1 cells: split_exact (its no-refit route) and e (a
//!   tripwire for gross e-value inflation).
//! - Sparse PLS1 cells (`keep = 2` of 4 columns, unweighted): split_nb,
//!   split_exact (its refit route) and `spls1_find_k_sequence`.
//! - PLS3 cells (two blocks; that family takes no weights): split_exact and
//!   split_nb of `pls3_confirmatory_test`.
//! - `sv_perm` cells: PLS3, PLS1 unweighted, and PLS1 with covariates that
//!   drive both X and y (so only the covariate-adjusted null holds).
//! - Ignored k-set cell (`k_set_level_under_null`): the min-p `split_exact`
//!   test over the default k set, upper edge of the 99% binomial band (see
//!   its doc).
//!
//! Under H0 (y ⟂ X; Y ⟂ X for PLS3) a valid α-level test rejects at rate
//! ≤ α. Every cell but the k-set cell runs seeded replications (`N_REPS`,
//! or `FIND_K_SEQUENCE_WEIGHTED_REPS` for the weighted sequential cell) and
//! asserts that the empirical FPR stays within a one-sided binomial
//! Monte-Carlo band of α. The allowed size is 0.065.
//!
//! MC slack: the per-rep reject indicator is Bernoulli(p) with p ≤ α under H0,
//! so FPR_hat has SD ≤ √(α(1−α)/N_REPS). We allow a 3·SD upper band:
//!
//!     FPR_hat ≤ α + 3·√(α(1−α)/N_REPS).
//!
//! At α = 0.05, N_REPS = 2000 this is 0.05 + 3·0.00487 ≈ 0.0646, so a cell
//! fails a true size of 0.08 or more with probability at least 0.995. A
//! correctly calibrated test clears it with margin; a broken path (e.g. a
//! convention mismatch inflating the statistic under permuted/split nulls)
//! blows past it. The band is intentionally one-sided: conservative tests
//! (FPR < α) are fine, only over-rejection is a failure.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
// Prose-heavy module docs: backticking every math token (α, n_reps, …) is noise.
#![allow(clippy::doc_markdown)]

use crate::{
    ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts,
    FindKSequenceOpts, Pls3ConfirmatoryTestOpts,
    linalg::{stable_rank, standardize_weighted},
    pls1_confirmatory_test, pls1_find_k_sequence, pls3_confirmatory_test, spls1_find_k_sequence,
};
use faer::{Col, Mat};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const ALPHA: f64 = 0.05;
const N_REPS: usize = 2000;

/// Assert the empirical FPR `rejects / n_reps` is at most the 3·SD one-sided
/// binomial Monte-Carlo upper band at `ALPHA`.
fn assert_fpr_within_band(label: &str, rejects: usize, n_reps: usize) {
    let fpr = rejects as f64 / n_reps as f64;
    let band = ALPHA + 3.0 * (ALPHA * (1.0 - ALPHA) / n_reps as f64).sqrt();
    assert!(
        fpr <= band,
        "{label}: empirical FPR={fpr} (rejects={rejects}/{n_reps}) exceeds MC band {band}"
    );
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
                k: Some(&[k]),
                weights: if weighted { Some(w.as_ref()) } else { None },
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: make_args(),
                seed: Some(seed_base + rep as u64),
                keep,
                ..Default::default()
            },
        )
        .unwrap();
        if r.pvalue <= ALPHA {
            rejects += 1;
        }
    }
    assert_fpr_within_band(label, rejects, N_REPS);
}

/// One H0 replication for the PLS3 cells: X and a multi-column Y drawn
/// independently of each other (pure noise on both sides). PLS3 takes no
/// observation weights, so none are returned.
fn pls3_null_data(n: usize, p: usize, q: usize, rep: usize) -> (Mat<f64>, Mat<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(0x3333_00C0_FFEE ^ rep as u64);
    let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
    // Y independent of X ⇒ H0 true.
    let y = Mat::<f64>::from_fn(n, q, |_, _| rng.random_range(-1.0..1.0));
    (x, y)
}

/// The FPR loop of `assert_calibrated` for `pls3_confirmatory_test`, which
/// has its own entry point and takes two blocks: n = 40, p = 4, q = 3,
/// `N_REPS` replications. `method` is the method the cell measures; every
/// replication must report it.
fn assert_pls3_calibrated(args: ConfirmatoryArgs, method: &str, seed_base: u64) {
    let mut rejects = 0usize;
    for rep in 0..N_REPS {
        let (x, y) = pls3_null_data(40, 4, 3, rep);
        let r = pls3_confirmatory_test(
            x.as_ref(),
            y.as_ref(),
            1,
            Pls3ConfirmatoryTestOpts {
                args,
                seed: Some(seed_base + rep as u64),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.test_method, method, "rep {rep} ran another method");
        if r.pvalue <= ALPHA {
            rejects += 1;
        }
    }
    assert_fpr_within_band(&format!("pls3 {method}"), rejects, N_REPS);
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
fn split_exact_unweighted_h0_fpr_within_band() {
    // k = 1, dense, unweighted: split_exact's no-refit route.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SplitExact {
            n_perm: 60,
            n_splits: 15,
        },
        "split_exact unweighted",
        false,
        3000,
        None,
    );
}

#[test]
fn split_exact_weighted_k2_h0_fpr_within_band() {
    // k = 2 with weights: split_exact's weighted refit route.
    assert_calibrated(
        40,
        4,
        2,
        || ConfirmatoryArgs::SplitExact {
            n_perm: 60,
            n_splits: 15,
        },
        "split_exact weighted k=2",
        true,
        6000,
        None,
    );
}

#[test]
fn score_weighted_h0_fpr_within_band() {
    // score is closed-form (no resampling): per-rep cost is one Gram matrix
    // and its trace and Frobenius norm, so this loop is the fastest of the
    // three.
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::Score { n_perm: None },
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
    // n_folds=5 so the loop stays under a few seconds.
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

/// Replications of [`find_k_sequence_weighted_h0_fpr_within_band`]: at
/// `N_REPS` its 2000 datasets lean high (131 rejections against an upper edge
/// of 129.2, while 8000 further datasets give 5.31%), so it runs five times
/// as many datasets under the same 3·SD band, upper edge 0.0565.
const FIND_K_SEQUENCE_WEIGHTED_REPS: usize = 10_000;

#[test]
fn find_k_sequence_weighted_h0_fpr_within_band() {
    // Pins the weighted sequential-deflation path under split_nb: under H0,
    // P(k_star ≥ 1) ≈ α.
    //
    // d = 8, not the 4 the other weighted cells use, so that NB actually runs:
    // the auto-gate reroutes any X with ≤ 4 columns outright, and at d = 5 the
    // weighted stable rank falls under the floor of 3 in 7% of the first 100
    // draws (min 2.77), a coin flip that would blend two methods into one FPR.
    // At d = 8 the weighted rank over the same 100 draws spans 3.79 to 5.38,
    // and Kish n_eff = 34.7 clears the size floor, so every rep is an unforced
    // NB run. Forcing instead of widening would have hidden the gate rather
    // than satisfied it.
    // The per-rep `test_method` assertion below is what keeps that true.
    let mut rejects = 0usize;
    for rep in 0..FIND_K_SEQUENCE_WEIGHTED_REPS {
        let (x, y, w) = null_data(40, 8, rep);
        let r = pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            Some(w.as_ref()),
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_splits: 15,
                // Rejection threshold inside the sequence test; must equal the
                // ALPHA the MC band is computed from.
                alpha: ALPHA,
                seed: Some(5000 + rep as u64),
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
    assert_fpr_within_band("find_k_sequence", rejects, FIND_K_SEQUENCE_WEIGHTED_REPS);
}

#[test]
fn split_nb_sparse_h0_fpr_within_band() {
    // Inference reuse: split machinery is agnostic to the
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
                k: Some(&[1]),
                weights: Some(w.as_ref()),
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 60,
                    n_splits: 15,
                },
                seed: Some(9000 + rep as u64),
                ..Default::default()
            },
        )
        .unwrap();
        if r.pvalue <= ALPHA {
            rejects += 1;
        }
    }
    assert_fpr_within_band("split_exact_concentrated_spectrum", rejects, N_REPS);
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
    let mut rejects = 0usize;
    for rep in 0..N_REPS {
        let (x, y, _w) = null_data(40, 4, rep);
        let r = spls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            2,
            None,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_splits: 15,
                alpha: ALPHA,
                seed: Some(8000 + rep as u64),
                ..Default::default()
            },
        )
        .unwrap();
        if r.k_star >= 1 {
            rejects += 1;
        }
    }
    assert_fpr_within_band("spls1_find_k_sequence", rejects, N_REPS);
}

#[test]
fn pls3_split_exact_h0_fpr_within_band() {
    // The PLS1 split_exact cells' sizes on the two-block null. At p = 4 and
    // q = 3 every replication takes the per-column refit route, so a rep
    // costs (n_perm + 1) × n_splits = 915 training-half fits, as in
    // split_exact_sparse_h0_fpr_within_band.
    assert_pls3_calibrated(
        ConfirmatoryArgs::SplitExact {
            n_perm: 60,
            n_splits: 15,
        },
        "split_exact",
        10_000,
    );
}

#[test]
fn pls3_split_nb_h0_fpr_within_band() {
    assert_pls3_calibrated(
        // `force`: the gate reroutes any X with 4 or fewer columns to
        // split_exact whatever its stable rank, so at p = 4 no replication
        // would run NB without it. The PLS3 gate reads X only.
        ConfirmatoryArgs::SplitNb {
            n_splits: 40,
            force: true,
        },
        "split_nb",
        11_000,
    );
}

#[test]
fn pls3_sv_perm_h0_fpr_within_band() {
    assert_pls3_calibrated(
        ConfirmatoryArgs::SvPerm {
            n_perm: 199,
            rotate: false,
        },
        "sv_perm",
        100_000,
    );
}

#[test]
fn sv_perm_unweighted_h0_fpr_within_band() {
    assert_calibrated(
        40,
        4,
        1,
        || ConfirmatoryArgs::SvPerm {
            n_perm: 199,
            rotate: false,
        },
        "sv_perm unweighted",
        false,
        110_000,
        None,
    );
}

/// One H0 replication with two covariates Z that drive both X (a shared
/// loading on every column) and y (`y = Z·γ + ε`); X's own noise is
/// independent of ε, so y ⟂ X given Z.
fn covariate_null_data(n: usize, d: usize, rep: usize) -> (Mat<f64>, Col<f64>, Mat<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(0x2222_00C0_FFEE ^ rep as u64);
    let z = Mat::<f64>::from_fn(n, 2, |_, _| rng.random_range(-1.0..1.0));
    let x = Mat::<f64>::from_fn(n, d, |i, _| {
        z[(i, 0)] + 0.5 * z[(i, 1)] + rng.random_range(-1.0..1.0)
    });
    let y = Col::<f64>::from_fn(n, |i| {
        2.0 * z[(i, 0)] - z[(i, 1)] + rng.random_range(-1.0..1.0)
    });
    (x, y, z)
}

#[test]
fn sv_perm_covariates_h0_fpr_within_band() {
    let mut rejects = 0usize;
    for rep in 0..N_REPS {
        let (x, y, z) = covariate_null_data(40, 4, rep);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[1]),
                weights: None,
                covariates: Some(z.as_ref()),
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SvPerm {
                    n_perm: 199,
                    rotate: false,
                },
                seed: Some(120_000 + rep as u64),
                ..Default::default()
            },
        )
        .unwrap();
        if r.pvalue <= ALPHA {
            rejects += 1;
        }
    }
    assert_fpr_within_band("sv_perm covariates", rejects, N_REPS);
}

/// The upper edge of the 99% binomial band of rejection counts at `ALPHA`
/// over `r` draws: the 0.995 quantile of Binomial(r, ALPHA).
fn binomial_upper_99(r: usize) -> usize {
    let odds = ALPHA / (1.0 - ALPHA);
    let mut pmf = (1.0 - ALPHA).powi(i32::try_from(r).unwrap());
    let mut cdf = pmf;
    for c in 0..r {
        if cdf >= 0.995 {
            return c;
        }
        pmf *= (r - c) as f64 / (c + 1) as f64 * odds;
        cdf += pmf;
    }
    r
}

/// One H0 draw for `k_set_level_under_null`: X uniform on (-1, 1), or with
/// `rank = Some(m)` an n x m by m x p product of uniforms plus 0.01 uniform
/// noise; y independent of X. `stream` seeds the draw.
fn k_set_null_data(n: usize, p: usize, rank: Option<usize>, stream: u64) -> (Mat<f64>, Col<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0C5E_7000_C0FF_EE00 ^ stream);
    let mut unif = |r: usize, c: usize, s: f64| {
        Mat::<f64>::from_fn(r, c, |_, _| s * rng.random_range(-1.0..1.0))
    };
    let x = match rank {
        None => unif(n, p, 1.0),
        Some(m) => {
            let f = unif(n, m, 1.0);
            let l = unif(m, p, 1.0);
            &f * &l + unif(n, p, 0.01)
        }
    };
    // y independent of X => H0 true.
    let y = unif(n, 1, 1.0).col(0).to_owned();
    (x, y)
}

/// Level of the min-p `split_exact` test over the default k set (`k: None`,
/// 1..8 on all three designs) under H0: tall (n = 60, p = 20), wide
/// (n = 40, p = 200) and low stable rank (n = 60, p = 100, rank-3 X plus 0.01
/// noise), unweighted, `n_perm = 99`, default `n_splits = 32`. Each design's
/// rejection count at `ALPHA` over `R` draws must be at most the upper edge of
/// the 99% binomial band: min-p over p-values on a 1/(n_perm + 1) grid has
/// ties, so its exact property is P(p <= alpha) <= alpha. Prints one line
/// per design before asserting.
#[test]
#[ignore = "Monte Carlo, about 10 minutes on 4 cores"]
fn k_set_level_under_null() {
    const R: usize = 10_000;
    const N_PERM: usize = 99;
    let designs: [(&str, usize, usize, Option<usize>); 3] = [
        ("tall", 60, 20, None),
        ("wide", 40, 200, None),
        ("low stable rank", 60, 100, Some(3)),
    ];
    let hi = binomial_upper_99(R);
    let mut outside = Vec::new();
    for (d, &(label, n, p, rank)) in designs.iter().enumerate() {
        let start = Instant::now();
        let mut rejects = 0usize;
        for rep in 0..R {
            let stream = ((d as u64) << 32) | rep as u64;
            let (x, y) = k_set_null_data(n, p, rank, stream);
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: None,
                    weights: None,
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: N_PERM,
                        n_splits: 32,
                    },
                    seed: Some(20_000 + stream),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(r.k, (1..=8).collect::<Vec<_>>(), "{label}: default set");
            if r.pvalue <= ALPHA {
                rejects += 1;
            }
        }
        println!(
            "{label}: n={n} p={p} R={R} n_perm={N_PERM} rejects={rejects} rate={:.4} upper={:.4} wall={:.1}s",
            rejects as f64 / R as f64,
            hi as f64 / R as f64,
            start.elapsed().as_secs_f64()
        );
        if rejects > hi {
            outside.push(label);
        }
    }
    assert!(outside.is_empty(), "above the 99% band: {outside:?}");
}
