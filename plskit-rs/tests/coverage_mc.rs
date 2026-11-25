//! Monte Carlo coverage / calibration test for the confirmatory CI engine.
//!
//! Slow (36 cells × 200 datasets × 300 resampling replicates each, plus
//! 200 oracle datasets per cell). Gated with `#[ignore]`. Run before tagging
//! a release:
//!
//!     cargo test -p plskit --release --test coverage_mc -- --ignored --nocapture
//!
//! The release-gate check covers two CI metrics on the
//! `pls1_confirmatory_test` engine across the cell grid
//! `n ∈ {100, 200, 500} × d ∈ {6, 20} × K ∈ {1, 2, 3} × SNR ∈ {1, 4}`.
//! The band at level=0.95 has lower edge `level − 0.05 = 0.90` and upper
//! edge `level + 3·√(level·(1 − level)/N_DATASETS)`, which is 0.9962 at 200
//! datasets. The upper edge sits three binomial standard errors above the
//! level, so a calibrated interval passes and a cell fails high only when
//! all 200 datasets cover. Leverage must lie inside both edges;
//! `holdout_corr` must reach the lower edge.
//!
//!  * **`holdout_corr`**, every cell, lower edge only. Its NB-Wald interval
//!    is built on half splits, where the Nadeau–Bengio 2003 factor
//!    `(n − m)/m` is about 1. The factor assumes two splits' correlations
//!    correlate at about 0.5. That holds when the noise of the held-out rows
//!    dominates, and the interval is then a little wide (1.1 to 1.2 times
//!    the needed width at snr = 1, coverage about 0.97). When the noise
//!    of the training fit dominates (snr = 4 here), two splits correlate
//!    nearer 0.25, and the interval is conservative: up to about twice as
//!    wide as needed, with coverage near 1.
//!  * **`leverage_ci_*` on each signal coordinate (`j < 2`) at
//!    `k ≤ SIGNAL_RANK`.** The interval is a normal-theory bootstrap
//!    interval centered on the full-data leverage (n-out-of-n refits; see
//!    `docs/python/results.md`, "Per-variable readouts"). Noise coordinates
//!    and `k > SIGNAL_RANK` are printed but not asserted: a `k = 1` noise
//!    coordinate's leverage is of order `1/n` and sits at the boundary (its
//!    interval over-covers by construction), and at `k > SIGNAL_RANK` the
//!    extra components fit noise, so the target is not identified across
//!    datasets.
//!
//! ## Coverage target (the oracle)
//!
//! Both metrics are *biased at finite n* — `holdout_corr` by its train size
//! `⌊n/2⌋` (a model trained on half the rows generalizes worse than one
//! trained on all of them), and `leverage` by
//! noise-dimension contamination of the weight vector at small `n/d`. So the
//! coverage target is NOT the asymptotic (large-n) value: that quantity
//! differs from the estimand each finite-n CI is centered on, and scoring
//! against it makes coverage collapse wherever the finite-n bias is
//! non-negligible (e.g. `holdout_corr` → 0.000 at `d=20, snr=4`). Instead the
//! target is the *population value of the same finite-n estimand*, estimated
//! by Monte Carlo over `N_ORACLE` fresh size-`n` datasets run through the
//! identical engine path. Coverage then tests purely whether the CI WIDTH is
//! calibrated, which is the question a coverage gate should ask.
//!
//! All seeds are deterministic per (cell, dataset) so a re-run reproduces
//! identical numbers. Failures panic with the cell `(n, d, k, snr)`, the
//! offending metric, and the empirical coverage vs. the band — surfacing
//! release-gate findings without blocking the default test run.
#![allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs

use faer::{Col, Mat};
use plskit::{
    CIOpts, ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, FitOpts,
    pls1_confirmatory_test, pls1_fit,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const N_DATASETS: usize = 200;
/// Monte-Carlo replicate count for the coverage target (see module doc). The
/// target SE is `sd(estimand) / sqrt(N_ORACLE)`; with per-dataset SD ~0.05 this
/// gives an oracle precise to ~0.004, well inside the band.
const N_ORACLE: usize = 200;
const N_BOOT: usize = 300;
const LEVEL: f64 = 0.95;
/// Distance of the band's lower edge below `LEVEL`.
const LOWER_MARGIN: f64 = 0.05;
/// True signal rank of the synthetic DGP: `y` depends on `X` only through the
/// single linear combination `x0 + x1` (see `synth`), so exactly one PLS
/// component carries signal. Leverage coverage is asserted only for
/// `k ≤ SIGNAL_RANK`; higher `k` extracts noise components whose leverage is
/// not identified across resamples.
const SIGNAL_RANK: usize = 1;

const CELL_NS: &[usize] = &[100, 200, 500];
const CELL_DS: &[usize] = &[6, 20];
const CELL_KS: &[usize] = &[1, 2, 3];
const CELL_SNRS: &[f64] = &[1.0, 4.0];

/// Synthetic DGP. Signal coordinates are `j ∈ {0, 1}` regardless of `d`;
/// remaining `d − 2` coordinates are pure noise predictors. `y` is a linear
/// function of the two signal coords plus i.i.d. uniform noise, scaled so
/// that `snr` is the per-coord signal multiplier (NOT the variance ratio).
///
/// Because `y = snr·(x0 + x1) + noise`, `y` depends on `X` through the single
/// direction `x0 + x1`; the X-columns are independent, so `Cov(X, y) ∝
/// [1,1,0,…,0]` and the residual after one PLS component carries no signal.
/// The true signal rank is therefore **1** (= `SIGNAL_RANK`), not 2 — coords 0
/// and 1 are both signal-bearing, but only one component is needed to capture
/// them.
fn synth(rng: &mut ChaCha8Rng, n: usize, d: usize, snr: f64) -> (Mat<f64>, Col<f64>) {
    let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
    let beta_signal: Vec<f64> = (0..d).map(|j| if j < 2 { 1.0 } else { 0.0 }).collect();
    let signal: Col<f64> =
        Col::<f64>::from_fn(n, |i| (0..d).map(|j| x[(i, j)] * beta_signal[j]).sum());
    let noise: Col<f64> = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
    (x, y)
}

/// Compute oracle per-coordinate leverage by fitting `pls1_fit` directly on
/// a large dataset and replicating the leverage formula from
/// `linalg::leverage_diag`:
///   `leverage[j] = W_star[j,:] · (W_starᵀ W_star)⁻¹ · W_star[j,:]ᵀ`
#[allow(clippy::similar_names, clippy::many_single_char_names)]
fn oracle_leverage(x: faer::MatRef<f64>, y: faer::ColRef<f64>, k: usize) -> Vec<f64> {
    let fit =
        pls1_fit(x, y, k, None, None, FitOpts::default()).expect("oracle pls1_fit must succeed");

    let mut wtw = faer::Mat::<f64>::zeros(k, k);
    faer::linalg::matmul::matmul(
        wtw.as_mut(),
        faer::Accum::Replace,
        fit.w_star.transpose(),
        fit.w_star.as_ref(),
        1.0,
        faer::Par::Seq,
    );
    let lu = faer::linalg::solvers::PartialPivLu::new(wtw.as_ref());
    let mut m_inv = faer::Mat::<f64>::zeros(k, k);
    for i in 0..k {
        m_inv[(i, i)] = 1.0;
    }
    {
        use faer::prelude::Solve;
        lu.solve_in_place(m_inv.as_mut());
    }

    let d = x.ncols();
    let mut leverage = vec![0.0_f64; d];
    let mut tmp = vec![0.0_f64; k];
    #[allow(clippy::needless_range_loop)]
    for j in 0..d {
        for kk in 0..k {
            let mut s = 0.0;
            for ll in 0..k {
                s += fit.w_star[(j, ll)] * m_inv[(ll, kk)];
            }
            tmp[kk] = s;
        }
        let mut q = 0.0;
        for kk in 0..k {
            q += tmp[kk] * fit.w_star[(j, kk)];
        }
        leverage[j] = q;
    }
    leverage
}

/// Stable per-cell seed mixer. Each `(n, d, k, snr_idx)` cell gets a unique
/// 64-bit base; per-dataset seeds are derived as `base + dataset_idx`.
#[allow(clippy::cast_possible_truncation)]
fn cell_base_seed(n: usize, d: usize, k: usize, snr_idx: usize) -> u64 {
    // Mix into a single u64 with a fixed salt so cells stay disjoint.
    let salt: u64 = 0x5EED_C0FF_BAAD_F00D;
    let mut s = salt;
    s ^= (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= (d as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    s ^= (k as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    s ^= (snr_idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    s
}

#[test]
#[ignore = "slow MC coverage test; run before release with --ignored"]
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn coverage_mc_two_sided_grid() {
    let band_lo = LEVEL - LOWER_MARGIN;
    // Three binomial standard errors of coverage over N_DATASETS above LEVEL.
    #[allow(clippy::cast_precision_loss)]
    let band_hi = LEVEL + 3.0 * (LEVEL * (1.0 - LEVEL) / N_DATASETS as f64).sqrt();

    println!(
        "coverage_mc grid: cells = {} × {} × {} × {} = {}, datasets/cell = {}",
        CELL_NS.len(),
        CELL_DS.len(),
        CELL_KS.len(),
        CELL_SNRS.len(),
        CELL_NS.len() * CELL_DS.len() * CELL_KS.len() * CELL_SNRS.len(),
        N_DATASETS,
    );
    println!(
        "level = {LEVEL:.2}, two-sided band = [{band_lo:.4}, {band_hi:.4}], n_boot = {N_BOOT}",
    );

    let mut failures: Vec<String> = Vec::new();
    let (mut n_holdout_checks, mut n_leverage_checks) = (0_usize, 0_usize);

    for &n in CELL_NS {
        for &d in CELL_DS {
            for &k in CELL_KS {
                for (snr_idx, &snr) in CELL_SNRS.iter().enumerate() {
                    let base = cell_base_seed(n, d, k, snr_idx);

                    // Coverage target: the population value of the SAME finite-n
                    // estimand each per-dataset CI is built for (NOT an
                    // asymptotic value — see module doc). Estimated by Monte
                    // Carlo over N_ORACLE fresh size-n datasets drawn from a seed
                    // stream disjoint from the coverage datasets below, run
                    // through the identical engine path (n_splits, n_boot, m_rate
                    // all match the per-dataset opts) so the estimand is the same.
                    let mut sum_holdout = 0.0_f64;
                    let mut sum_lev = vec![0.0_f64; d];
                    for o_idx in 0..N_ORACLE {
                        // High-bit salt puts oracle seeds in a region disjoint
                        // from the per-dataset seeds (`base + d_idx`, d_idx < 200).
                        let oracle_seed = base ^ 0xA5A5_A5A5_0000_0000 ^ (o_idx as u64);
                        let mut orng = ChaCha8Rng::seed_from_u64(oracle_seed);
                        let (xo, yo) = synth(&mut orng, n, d, snr);
                        let oracle_opts = ConfirmatoryTestOpts {
                            args: ConfirmatoryArgs::SplitNb {
                                n_splits: 30,
                                force: false,
                            },
                            ci: Some(CIOpts {
                                n_boot: N_BOOT,
                                m_rate: 0.7,
                                level: LEVEL,
                                max_failure_rate: 0.0,
                            }),
                            seed: Some(oracle_seed ^ 0xC0FF_EE00_C0FF_EE00),
                            ..Default::default()
                        };
                        let oracle_r = pls1_confirmatory_test(
                            ConfirmatoryTestInput::Raw {
                                x: xo.as_ref(),
                                y: yo.as_ref(),
                                k: Some(&[k]),
                                weights: None,
                                covariates: None,
                                null_y: None,
                            },
                            oracle_opts,
                        )
                        .expect("oracle MC dataset must succeed");
                        sum_holdout += oracle_r
                            .ci
                            .expect("oracle CI must be Some")
                            .holdout_corr
                            .point;
                        let lev = oracle_leverage(xo.as_ref(), yo.as_ref(), k);
                        for j in 0..d {
                            sum_lev[j] += lev[j];
                        }
                    }
                    #[allow(clippy::cast_precision_loss)]
                    let oracle_holdout_corr = sum_holdout / N_ORACLE as f64;
                    let oracle_lev: Vec<f64> = sum_lev
                        .iter()
                        .map(|&s| {
                            #[allow(clippy::cast_precision_loss)]
                            let v = s / N_ORACLE as f64;
                            v
                        })
                        .collect();

                    let mut covered_holdout = 0_usize;
                    let mut covered_lev = vec![0_usize; d];

                    for d_idx in 0..N_DATASETS {
                        let dataset_seed = base.wrapping_add(d_idx as u64);
                        let mut rng = ChaCha8Rng::seed_from_u64(dataset_seed);
                        let (x, y) = synth(&mut rng, n, d, snr);
                        let opts = ConfirmatoryTestOpts {
                            args: ConfirmatoryArgs::SplitNb {
                                n_splits: 30,
                                force: false,
                            },
                            ci: Some(CIOpts {
                                n_boot: N_BOOT,
                                m_rate: 0.7,
                                level: LEVEL,
                                max_failure_rate: 0.0,
                            }),
                            seed: Some(dataset_seed ^ 0xBEEF_BEEF_BEEF_BEEF),
                            ..Default::default()
                        };
                        let r = pls1_confirmatory_test(
                            ConfirmatoryTestInput::Raw {
                                x: x.as_ref(),
                                y: y.as_ref(),
                                k: Some(&[k]),
                                weights: None,
                                covariates: None,
                                null_y: None,
                            },
                            opts,
                        )
                        .expect("strict-mode MC dataset must not fail");
                        let ci = r.ci.expect("ci=Some must produce ci.is_some");

                        if ci.holdout_corr.lower <= oracle_holdout_corr
                            && oracle_holdout_corr <= ci.holdout_corr.upper
                        {
                            covered_holdout += 1;
                        }
                        for j in 0..d {
                            if ci.leverage_ci_lower[j] <= oracle_lev[j]
                                && oracle_lev[j] <= ci.leverage_ci_upper[j]
                            {
                                covered_lev[j] += 1;
                            }
                        }
                    }

                    #[allow(clippy::cast_precision_loss)]
                    let denom = N_DATASETS as f64;
                    #[allow(clippy::cast_precision_loss)]
                    let cov_holdout = covered_holdout as f64 / denom;
                    let cov_lev: Vec<f64> = covered_lev
                        .iter()
                        .map(|&c| {
                            #[allow(clippy::cast_precision_loss)]
                            let v = c as f64 / denom;
                            v
                        })
                        .collect();

                    // Per-cell summary line: cell params, holdout_corr coverage,
                    // signal-coord leverage coverage (asserted), noise-coord
                    // leverage coverage (diagnostic only).
                    let signal_str: Vec<String> =
                        (0..2).map(|j| format!("{:.3}", cov_lev[j])).collect();
                    let noise_str: Vec<String> =
                        (2..d).map(|j| format!("{:.3}", cov_lev[j])).collect();
                    // `*` flags signal coords whose leverage is diagnostic-only
                    // (k > SIGNAL_RANK, not asserted).
                    let lev_flag = if k <= SIGNAL_RANK { "" } else { "*" };
                    println!(
                        "[cell n={:>3} d={:>2} k={} snr={:.0}] holdout_corr={:.3} \
                         leverage_signal{}=[{}] leverage_noise=[{}]",
                        n,
                        d,
                        k,
                        snr,
                        cov_holdout,
                        lev_flag,
                        signal_str.join(", "),
                        noise_str.join(", "),
                    );

                    // holdout_corr: lower edge only, the interval is
                    // conservative at high signal (see module doc).
                    n_holdout_checks += 1;
                    if cov_holdout < band_lo {
                        failures.push(format!(
                            "cell (n={n}, d={d}, k={k}, snr={snr}): holdout_corr \
                             coverage {cov_holdout:.3} below {band_lo:.4}",
                        ));
                    }
                    // Leverage: asserted on the signal coordinates at
                    // k ≤ SIGNAL_RANK. The target is E[ĥ_n], the expected
                    // leverage of a size-n fit (oracle above), which at small
                    // n/d sits well below the large-sample leverage of the
                    // signal coordinates (≈0.39 vs 0.5 at d=20, n=100,
                    // snr=1). The CI is centered on ĥ with the SD of
                    // n-out-of-n bootstrap refits and no bias correction, so
                    // it targets that estimand. A subsampling interval
                    // reflected around ĥ at the rate √(m/(n − m)) would
                    // cover it only 0.40 to 0.65 at d=20: a fit on m rows
                    // puts less leverage on the signal coordinates than the
                    // full fit, and the reflection turned that gap into an
                    // offset. Noise coordinates and k > SIGNAL_RANK are
                    // printed only (see module doc).
                    if k <= SIGNAL_RANK {
                        for (j, &cov) in cov_lev.iter().enumerate().take(2) {
                            n_leverage_checks += 1;
                            if !(band_lo..=band_hi).contains(&cov) {
                                failures.push(format!(
                                    "cell (n={n}, d={d}, k={k}, snr={snr}): leverage \
                                     signal coord {j} coverage {cov:.3} outside band \
                                     [{band_lo:.4}, {band_hi:.4}]",
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    let n_cells = CELL_NS.len() * CELL_DS.len() * CELL_KS.len() * CELL_SNRS.len();
    let n_lev_cells =
        n_cells / CELL_KS.len() * CELL_KS.iter().filter(|&&k| k <= SIGNAL_RANK).count();
    assert_eq!(n_holdout_checks, n_cells, "holdout_corr checks run");
    assert_eq!(n_leverage_checks, 2 * n_lev_cells, "leverage checks run");
    if !failures.is_empty() {
        let n_fail = failures.len();
        let joined = failures.join("\n  ");
        panic!(
            "coverage_mc grid: {n_fail} cell-metric failures (band \
             [{band_lo:.4}, {band_hi:.4}], holdout_corr lower edge only):\n  {joined}",
        );
    }
    println!(
        "coverage_mc grid: {n_holdout_checks} holdout_corr checks at or above {band_lo:.4}, \
         {n_leverage_checks} leverage checks within [{band_lo:.4}, {band_hi:.4}]"
    );
}
