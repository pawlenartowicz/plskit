//! Monte Carlo coverage / calibration test for the confirmatory CI engine.
//!
//! Slow (36 cells × 200 datasets × 300 resamples each, plus a 50k-sample
//! oracle fit per cell). Gated with `#[ignore]`. Run before tagging a release:
//!
//!     cargo test -p plskit --release --test coverage_mc -- --ignored --nocapture
//!
//! The release-gate check covers two surviving CI metrics on the
//! `pls1_confirmatory_test` engine across the cell grid
//! `n ∈ {100, 200, 500} × d ∈ {6, 20} × K ∈ {1, 2, 3} × SNR ∈ {1, 4}`:
//!
//!  * **`holdout_corr` two-sided coverage at level=0.95.** Empirical coverage
//!    must lie in `[0.90, 1.00]` (`level ± 0.05`). NB-Wald is conservative
//!    by construction (Nadeau–Bengio 2003), so empirical coverage tends to
//!    sit at or above 0.95 — over-coverage near 1.00 is expected and
//!    accepted by the upper edge of the band.
//!  * **Per-coordinate `leverage_ci_*` coverage on signal coordinates only
//!    (`j < 2`) at level=0.95.** Each signal coordinate must have empirical
//!    coverage in `[0.90, 1.00]`. Noise-coordinate (`j ≥ 2`) numbers are
//!    printed for diagnostic context but do NOT participate in the band
//!    assertion — under any well-specified DGP the leverage population
//!    value at noise coords pins to a boundary, so the centered-scaled CI
//!    inherits a one-sided geometry that makes coverage MC the wrong
//!    calibration check there.
//!
//! All seeds are deterministic per (cell, dataset) so a re-run reproduces
//! identical numbers. Failures panic with the cell `(n, d, k, snr)`, the
//! offending metric, and the empirical coverage vs. the band — surfacing
//! release-gate findings without blocking Phase 1.

use faer::{Col, Mat};
use plskit::{
    pls1_confirmatory_test, pls1_fit, CIOpts, ConfirmatoryArgs, ConfirmatoryTestInput,
    ConfirmatoryTestOpts, FitOpts, KSpec,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const N_DATASETS: usize = 200;
const N_BOOT: usize = 300;
const LEVEL: f64 = 0.95;
const BAND_HALF_WIDTH: f64 = 0.05;
const ORACLE_N: usize = 50_000;

const CELL_NS: &[usize] = &[100, 200, 500];
const CELL_DS: &[usize] = &[6, 20];
const CELL_KS: &[usize] = &[1, 2, 3];
const CELL_SNRS: &[f64] = &[1.0, 4.0];

/// Synthetic DGP. Signal coordinates are `j ∈ {0, 1}` regardless of `d`;
/// remaining `d − 2` coordinates are pure noise predictors. `y` is a linear
/// function of the two signal coords plus i.i.d. uniform noise, scaled so
/// that `snr` is the per-coord signal multiplier (NOT the variance ratio).
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
/// `signal_test.rs::compute_leverage_ref`:
///   `leverage[j] = W_star[j,:] · (W_starᵀ W_star)⁻¹ · W_star[j,:]ᵀ`
#[allow(clippy::similar_names, clippy::many_single_char_names)]
fn oracle_leverage(x: faer::MatRef<f64>, y: faer::ColRef<f64>, k: usize) -> Vec<f64> {
    let fit = pls1_fit(x, y, KSpec::Fixed(k), None, FitOpts::default())
        .expect("oracle pls1_fit must succeed");

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
    let band_lo = LEVEL - BAND_HALF_WIDTH;
    let band_hi = LEVEL + BAND_HALF_WIDTH;

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
        "level = {LEVEL:.2}, two-sided band = [{band_lo:.2}, {band_hi:.2}], n_boot = {N_BOOT}",
    );

    let mut failures: Vec<String> = Vec::new();

    for &n in CELL_NS {
        for &d in CELL_DS {
            for &k in CELL_KS {
                for (snr_idx, &snr) in CELL_SNRS.iter().enumerate() {
                    let base = cell_base_seed(n, d, k, snr_idx);

                    // Oracle: 50k-sample fit, deterministic per cell.
                    let mut oracle_rng = ChaCha8Rng::seed_from_u64(base ^ 0xDEAD_BEEF_DEAD_BEEF);
                    let (x_oracle, y_oracle) = synth(&mut oracle_rng, ORACLE_N, d, snr);

                    // holdout_corr oracle uses a confirmatory CI fit (the same path the
                    // per-dataset test uses), so the two are comparable.
                    let oracle_opts = ConfirmatoryTestOpts {
                        args: ConfirmatoryArgs::SplitNb { n_splits: 50 },
                        ci: Some(CIOpts {
                            n_boot: N_BOOT,
                            m_rate: 0.7,
                            level: LEVEL,
                            max_failure_rate: 0.0,
                        }),
                        seed: Some(base ^ 0xC0FF_EE00_C0FF_EE00),
                        disable_parallelism: false,
                        ..Default::default()
                    };
                    let oracle_r = pls1_confirmatory_test(
                        ConfirmatoryTestInput::Raw {
                            x: x_oracle.as_ref(),
                            y: y_oracle.as_ref(),
                            k,
                            weights: None,
                        },
                        oracle_opts,
                    )
                    .expect("oracle confirmatory_test must succeed");
                    let oracle_holdout_corr = oracle_r
                        .ci
                        .expect("oracle CI must be Some")
                        .holdout_corr
                        .point;

                    // Per-coordinate leverage oracle uses a direct pls1_fit (no
                    // resampling) on the same 50k dataset.
                    let oracle_lev = oracle_leverage(x_oracle.as_ref(), y_oracle.as_ref(), k);

                    let mut covered_holdout = 0_usize;
                    let mut covered_lev = vec![0_usize; d];

                    for d_idx in 0..N_DATASETS {
                        let dataset_seed = base.wrapping_add(d_idx as u64);
                        let mut rng = ChaCha8Rng::seed_from_u64(dataset_seed);
                        let (x, y) = synth(&mut rng, n, d, snr);
                        let opts = ConfirmatoryTestOpts {
                            args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                            ci: Some(CIOpts {
                                n_boot: N_BOOT,
                                m_rate: 0.7,
                                level: LEVEL,
                                max_failure_rate: 0.0,
                            }),
                            seed: Some(dataset_seed ^ 0xBEEF_BEEF_BEEF_BEEF),
                            disable_parallelism: false,
                            ..Default::default()
                        };
                        let r = pls1_confirmatory_test(
                            ConfirmatoryTestInput::Raw {
                                x: x.as_ref(),
                                y: y.as_ref(),
                                k,
                                weights: None,
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
                    println!(
                        "[cell n={:>3} d={:>2} k={} snr={:.0}] holdout_corr={:.3} \
                         leverage_signal=[{}] leverage_noise=[{}]",
                        n,
                        d,
                        k,
                        snr,
                        cov_holdout,
                        signal_str.join(", "),
                        noise_str.join(", "),
                    );

                    // Assertions: holdout_corr two-sided band.
                    if !(cov_holdout >= band_lo && cov_holdout <= band_hi) {
                        failures.push(format!(
                            "cell (n={n}, d={d}, k={k}, snr={snr}): holdout_corr \
                             coverage {cov_holdout:.3} outside band \
                             [{band_lo:.2}, {band_hi:.2}]",
                        ));
                    }
                    // Per-coordinate leverage band on signal coords only (j < 2).
                    for (j, &c) in cov_lev.iter().take(2).enumerate() {
                        if !(c >= band_lo && c <= band_hi) {
                            failures.push(format!(
                                "cell (n={n}, d={d}, k={k}, snr={snr}): leverage[{j}] \
                                 (signal) coverage {c:.3} outside band [{band_lo:.2}, {band_hi:.2}]",
                            ));
                        }
                    }
                }
            }
        }
    }

    if !failures.is_empty() {
        let n_fail = failures.len();
        let joined = failures.join("\n  ");
        panic!(
            "coverage_mc grid: {n_fail} cell-metric failures outside two-sided \
             band [{band_lo:.2}, {band_hi:.2}]:\n  {joined}",
        );
    }
    println!("coverage_mc grid: all cells within band [{band_lo:.2}, {band_hi:.2}]");
}
