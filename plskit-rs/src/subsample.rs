//! Resampling engine for PLS1 inference: Politis–Romano subsampling for β
//! and `holdout_corr`, and an n-out-of-n bootstrap for leverage. The engine
//! entry points are internal (`pub(crate)`): callers are
//! `signal_test::pls1_confirmatory_test(CI=Some)` and
//! `rotation_stability::pls1_rotation_stability`. Only the result types
//! `CIScalar` and `ConfirmatoryCI` are public, re-exported from `lib.rs`.
//!
//! Reduction formulas are documented inline on `reduce_centered_scaled`
//! (centered-scaled CI; Politis–Romano 1999 Ch. 3), `reduce_leverage`
//! (normal-theory bootstrap CI centered on the full-data leverage) and
//! `reduce_holdout_corr` (Fisher z-transformed NB-Wald CI; variance via
//! Nadeau–Bengio 2003 inflation applied on the variance-stabilized atanh
//! scale).

/// Resampling CI for a scalar functional, plus its SD.
/// Marshalled to a frozen dataclass on the Python side; fields must not be reordered.
///
/// Scale convention: `point`, `lower`, `upper` are always reported on the natural
/// (user-facing) scale of the statistic. `sd` is on the inference scale used to
/// build the CI. For the centered-scaled reductions the inference scale is the
/// natural scale. For `reduce_holdout_corr` the inference scale is Fisher
/// z = atanh(r), so `sd` is on the z-scale and the CI is asymmetric on the
/// r-scale; do not reconstruct it from `point ± sd`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CIScalar {
    /// Full-data point estimate `θ̂_n`, on the natural scale of the statistic.
    pub point: f64,
    /// Lower endpoint of the CI at `level`, on the natural scale.
    pub lower: f64,
    /// Upper endpoint of the CI at `level`, on the natural scale.
    pub upper: f64,
    /// Resampling SE on the inference scale (natural scale for centered-scaled
    /// reductions; Fisher z-scale for `reduce_holdout_corr`). See struct doc.
    pub sd: f64,
}

/// Subsampling rate factor `√(m/(n − m))` that maps the spread of size-`m`
/// replicates to the SE of the size-`n` statistic. Subsamples are drawn
/// without replacement, so for an asymptotically linear statistic
/// `Var(θ̂_m | data) ≈ σ²·(1/m − 1/n)` and
/// `SE(θ̂_n) = σ/√n = √(m/(n − m)) · sd(θ̂_m)`. The textbook `√(m/n)` drops
/// the finite-population factor `1 − m/n`, which does not vanish at the
/// default `m_rate` (`m/n = 0.26` at `n = 100`, `0.10` at `n = 2000`). The
/// argument uses only the linearization, not the form of the functional.
/// Used by the β reduction; leverage is bootstrapped at size `n` and needs
/// no rate (`reduce_leverage`). Caller guarantees `m < n`.
#[allow(clippy::cast_precision_loss)]
fn fpc_rate(n: usize, m: usize) -> f64 {
    (m as f64 / (n - m) as f64).sqrt()
}

/// Reduce a per-resample B-vector of `θ̂_m,b` values to a `CIScalar` using
/// centered-scaled subsampling (Politis–Romano 1999 Ch. 3) at the rate
/// factor `scale`:
///   Δ_b = θ̂_m,b − θ̂_n
///   lower = θ̂_n − scale · quantile(Δ, 1 − α/2)
///   upper = θ̂_n − scale · quantile(Δ, α/2)
///   sd    = scale · stddev(Δ)
/// `level` is the CI level (e.g. 0.95 → α = 0.05). `reduce_beta` passes the
/// finite-population rate `fpc_rate(n, m)` for the κ̂-rescaled replicates,
/// and 1 to read off the raw replicate spread for `κ̂`.
///
/// NaN samples (produced by failed worker resamples) are filtered before
/// building the deltas vector; all denominators use the finite count
/// `b_finite`. If all samples are NaN, returns a degenerate CI at `point`.
#[allow(clippy::cast_precision_loss, clippy::doc_markdown)]
fn reduce_centered_scaled(samples_b: &[f64], point: f64, scale: f64, level: f64) -> CIScalar {
    let mut deltas: Vec<f64> = samples_b
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&v| v - point)
        .collect();
    let b_finite = deltas.len();
    if b_finite == 0 {
        return CIScalar {
            point,
            lower: point,
            upper: point,
            sd: 0.0,
        };
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));

    let alpha = 1.0 - level;

    let q_lo = crate::linalg::empirical_quantile(&deltas, alpha / 2.0);
    let q_hi = crate::linalg::empirical_quantile(&deltas, 1.0 - alpha / 2.0);

    // lower = θ̂_n − scale · q_{1−α/2};  upper = θ̂_n − scale · q_{α/2}.
    let lower = point - scale * q_hi;
    let upper = point - scale * q_lo;

    // Two-pass stable variance: compute mean first, then sum squared deviations.
    let mean: f64 = deltas.iter().sum::<f64>() / b_finite as f64;
    let var: f64 = if b_finite > 1 {
        deltas.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (b_finite - 1) as f64
    } else {
        0.0
    };
    let sd = scale * var.sqrt();

    CIScalar {
        point,
        lower,
        upper,
        sd,
    }
}

/// Normal-theory bootstrap CI for one variable's leverage, centered on the
/// full-data value `h` (`point`), from the n-out-of-n bootstrap replicates
/// `h_b` (`samples_b`):
///
/// ```text
/// sd    = stddev(h_b)
/// lower = h − Φ⁻¹(1 − α/2) · sd
/// upper = h + Φ⁻¹(1 − α/2) · sd
/// ```
///
/// Targets `E[ĥ_n]`, the expected leverage of a size-`n` fit, under the
/// assumption that `ĥ_n` is roughly normal around it with an SD the
/// bootstrap spread estimates. There is deliberately no bias correction:
/// the replicate mean sits below `h` for signal variables and above it for
/// noise variables (a resample duplicates rows and spreads leverage
/// further), and neither that offset nor its reflection moves the interval.
/// Unlike the subsampling SE, `sd(h_b)` needs no rate factor, because the
/// replicates are fits at the full size `n`.
///
/// NaN samples (failed worker resamples) are filtered. With fewer than two
/// finite samples the CI is degenerate at `point` with `sd = 0`. The caller
/// clamps the bounds to `[0, 1]`.
#[allow(clippy::cast_precision_loss)]
fn reduce_leverage(samples_b: &[f64], point: f64, level: f64) -> CIScalar {
    let finite: Vec<f64> = samples_b.iter().copied().filter(|v| !v.is_nan()).collect();
    let b_finite = finite.len();
    if b_finite < 2 {
        return CIScalar {
            point,
            lower: point,
            upper: point,
            sd: 0.0,
        };
    }
    let mean = finite.iter().sum::<f64>() / b_finite as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (b_finite - 1) as f64;
    let sd = var.sqrt();
    let half = standard_normal_inv(1.0 - (1.0 - level) / 2.0) * sd;
    CIScalar {
        point,
        lower: point - half,
        upper: point + half,
        sd,
    }
}

/// Resolve `m` from `(n, m_rate)`: `m = ceil(n^m_rate)`.
/// Caller has already validated `0.5 < m_rate < 0.95`.
pub(crate) fn resolve_m(n: usize, m_rate: f64) -> usize {
    #[allow(clippy::cast_precision_loss)]
    let m_real = (n as f64).powf(m_rate);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let m = m_real.ceil() as usize;
    m
}

/// Draw a subsample of size `m` without replacement from `0..n`. Returns
/// `(sample_idx, holdout_idx)` partitioning `0..n`. Uses `rng` (a child RNG).
pub(crate) fn subsample_indices(
    n: usize,
    m: usize,
    rng: &mut crate::rng::Rng,
) -> (Vec<usize>, Vec<usize>) {
    use rand::seq::SliceRandom;
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    let sample = perm[..m].to_vec();
    let holdout = perm[m..].to_vec();
    (sample, holdout)
}

#[cfg(test)]
mod tests_indices {
    use super::*;
    use crate::rng::resolve_seed;

    #[test]
    fn resolve_m_known_values() {
        // Known values (m_rate=0.7): n=100 → m=26; n=1000 → m=126; n=10_000 → m=631.
        assert_eq!(resolve_m(100, 0.7), 26);
        assert_eq!(resolve_m(1000, 0.7), 126);
        assert_eq!(resolve_m(10_000, 0.7), 631);
    }

    #[test]
    fn subsample_indices_partitions_range() {
        let (_, mut rng) = resolve_seed(Some(7)).unwrap();
        let (s, h) = subsample_indices(100, 26, &mut rng);
        assert_eq!(s.len(), 26);
        assert_eq!(h.len(), 74);
        let mut all: Vec<usize> = s.iter().chain(h.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..100).collect::<Vec<_>>());
    }
}

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::{pls1_fit, FitOpts, KSpec};
use crate::linalg::{col_row_subset, row_subset};

/// Per-resample outputs for the confirmatory CI branch. Per-variable arrays
/// only — composite scalars have been removed.
///
/// Each row carries its own `beta` vector; the reducer turns these into the
/// per-coordinate β CIs and the per-coordinate subsampling z (`beta_sign_z`).
/// Storing per-row keeps the row-disjoint invariant (per-resample writes are
/// independent — no shared counters across threads).
#[derive(Debug)]
pub(crate) struct ConfirmatoryWorkerRow {
    /// Per-variable subspace leverage on the bootstrap fit's `W_b` (n-out-of-n,
    /// with replacement; see `run_one_confirmatory`). Invariant under
    /// right-orthogonal rotation of W (leverage = diag(W(WᵀW)⁻¹Wᵀ); W → W·R
    /// leaves the hat matrix fixed), so computed on the unaligned `W_b`, with
    /// no procrustes alignment. Length D.
    pub leverage: Vec<f64>,
    /// `corr(X[holdout] · β_b, y[holdout])`. NaN if undefined (e.g. constant holdout y).
    pub holdout_corr: f64,
    /// Per-variable subsample regression coefficient `β_b`. Length D. PLS1-only:
    /// β is invariant under component sign flips and within-subspace rotations
    /// (the (Pᵀ W)⁻¹·Q product cancels both), so per-coordinate centered-scaled
    /// CIs are well-defined without procrustes alignment.
    pub beta: Vec<f64>,
}

/// Draw an n-out-of-n bootstrap resample: `n` indices from `0..n` with
/// replacement. Uses `rng` (a child RNG).
pub(crate) fn bootstrap_indices(n: usize, rng: &mut crate::rng::Rng) -> Vec<usize> {
    use rand::RngExt;
    (0..n).map(|_| rng.random_range(0..n)).collect()
}

/// A fit on a row selection of `(x, y)`, with the column and target scales
/// it was standardized with (unit / zero under `pre_standardized`).
struct RowFit {
    w: Mat<f64>,
    beta: Col<f64>,
    x_mean: Col<f64>,
    x_scale: Col<f64>,
    y_scale: f64,
}

/// Fit `k` components on the rows `idx` of `(x, y)` (repeats allowed) with
/// the same standardization regime as the full-data fit: the rows are
/// re-standardized with their own (weighted) moments unless
/// `pre_standardized_x`. Weights are sliced to `idx` and re-validated, so an
/// insufficient effective sample size surfaces as `InvalidWeights`.
///
/// A NIPALS short-circuit (`k_used < k`) is an error, so the caller's
/// driver counts the resample as failed instead of reducing a truncated
/// weight matrix against length-`k` references.
fn fit_rows(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    idx: &[usize],
    k: usize,
    pre_standardized_x: bool,
    weights: Option<ColRef<'_, f64>>,
) -> PlsKitResult<RowFit> {
    let d = x.ncols();
    let y_rows = col_row_subset(y, idx);

    let w_sub: Option<faer::Col<f64>> = weights.map(|w| crate::linalg::col_row_subset(w, idx));
    let (w_sub_norm, _, _) = crate::fit::validate_and_normalize_weights(
        w_sub.as_ref().map(faer::Col::as_ref),
        idx.len(),
        k,
    )?;

    let (xs_sub, x_mean, x_scale, ys_sub, y_scale) = if pre_standardized_x {
        // Caller asserts already standardized: the row subsets are the
        // blocks. Mean/scale are no-ops (zeros / ones) for the holdout
        // standardization and for the β back-projection (β_b stays on the
        // standardized scale, matching β_ref which the caller's full-data fit
        // also leaves on that scale).
        (
            row_subset(x, idx),
            Col::<f64>::zeros(d),
            Col::<f64>::from_fn(d, |_| 1.0),
            y_rows,
            1.0_f64,
        )
    } else {
        let (xs, mu, sigma) = crate::linalg::standardize_rows(
            x,
            idx,
            w_sub_norm.as_ref().map(faer::Col::as_ref),
            None,
        );
        let (ys, _, ys_sigma) = crate::linalg::standardize1_weighted(
            y_rows.as_ref(),
            w_sub_norm.as_ref().map(faer::Col::as_ref),
        );
        (xs, mu, sigma, ys, ys_sigma)
    };

    let fit = pls1_fit(
        xs_sub.as_ref(),
        ys_sub.as_ref(),
        KSpec::Fixed(k),
        w_sub_norm.as_ref().map(faer::Col::as_ref),
        FitOpts {
            pre_standardized: true,
            // Seq inside the per-resample worker: outer Rayon owns the threadpool.
            par: crate::fit::ParChoice::Seq,
            ..FitOpts::default()
        },
    )?;
    if fit.w_star.ncols() != k {
        return Err(PlsKitError::Internal(format!(
            "resample fit truncated to {} of {k} components",
            fit.w_star.ncols()
        )));
    }
    Ok(RowFit {
        w: fit.w_star,
        beta: fit.beta,
        x_mean,
        x_scale,
        y_scale,
    })
}

/// Run one confirmatory replicate: a size-`m` subsample (without
/// replacement) for β and `holdout_corr`, and an n-out-of-n bootstrap
/// resample (with replacement) for leverage. Both index sets come from the
/// same child `rng`, subsample first, so the subsample draw does not depend
/// on the bootstrap one. `pre_standardized_x` matches the user's flag.
///
/// Leverage uses the bootstrap because its bias depends on the sample size
/// (at small `n/D` a fit spreads leverage from signal onto noise
/// variables, more so on fewer rows): a size-`m` fit measures the sampling
/// distribution at the wrong size, while a size-`n` resample reproduces the
/// spread of the full-data leverage. See `reduce_leverage`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn run_one_confirmatory(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    m: usize,
    _w_ref: MatRef<'_, f64>, // retained for call-site shape parity; leverage no longer uses it
    pre_standardized_x: bool,
    weights: Option<ColRef<'_, f64>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<ConfirmatoryWorkerRow> {
    let n = x.nrows();
    let d = x.ncols();

    // 1. Index draws: subsample, then bootstrap resample.
    let (sample_idx, holdout_idx) = subsample_indices(n, m, rng);
    let boot_idx = bootstrap_indices(n, rng);

    // 2. Subsample fit (β, holdout_corr) and bootstrap fit (leverage).
    let RowFit {
        beta: beta_b,
        x_mean,
        x_scale,
        y_scale,
        ..
    } = fit_rows(x, y, &sample_idx, k, pre_standardized_x, weights)?;
    let boot = fit_rows(x, y, &boot_idx, k, pre_standardized_x, weights)?;

    // 3. Per-variable leverage on the bootstrap fit's unaligned W. Leverage
    // is invariant under right-orthogonal rotation (leverage =
    // diag(W(WᵀW)⁻¹Wᵀ); W → W·R leaves the hat matrix fixed), so no
    // procrustes alignment is needed.
    let leverage = crate::linalg::leverage_diag(boot.w.as_ref());

    // 4. Holdout predictive correlation of the subsample fit.
    // Pre-standardized: the holdout rows are the block. Otherwise they are
    // gathered and standardized with the subsample's moments in one pass.
    let ys_h_owned = col_row_subset(y, &holdout_idx);
    let xs_h = if pre_standardized_x {
        row_subset(x, &holdout_idx)
    } else {
        crate::linalg::standardize_apply_rows(
            x,
            &holdout_idx,
            x_mean.as_ref(),
            x_scale.as_ref(),
            None,
        )
    };

    // Seq inside the replicate worker: outer Rayon owns the threadpool.
    let y_pred = crate::linalg::mat_vec(xs_h.as_ref(), beta_b.as_ref(), faer::Par::Seq);
    let n_h = ys_h_owned.nrows();
    #[allow(clippy::cast_precision_loss)]
    let holdout_corr = if n_h >= 2 {
        let yp_mean: f64 = (0..n_h).map(|i| y_pred[i]).sum::<f64>() / n_h as f64;
        let yh_mean: f64 = (0..n_h).map(|i| ys_h_owned[i]).sum::<f64>() / n_h as f64;
        let mut s_pp = 0.0_f64;
        let mut s_yy = 0.0_f64;
        let mut s_py = 0.0_f64;
        for i in 0..n_h {
            #[allow(clippy::many_single_char_names)]
            let dp = y_pred[i] - yp_mean;
            #[allow(clippy::many_single_char_names)]
            let dy = ys_h_owned[i] - yh_mean;
            s_pp += dp * dp;
            s_yy += dy * dy;
            s_py += dp * dy;
        }
        if s_pp > 1e-30 && s_yy > 1e-30 {
            (s_py / (s_pp * s_yy).sqrt()).clamp(-1.0, 1.0)
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    // 5. Back-project β_b to the same scale as β_ref so deltas are meaningful.
    // Mirrors `pls1_fit`'s full-data back-projection (`fit.rs:143-147`):
    //   β_raw[j] = β_std[j] * y_scale / x_scale[j]
    // For pre_standardized_x, both scales are 1.0 → no-op (β_b stays standardized,
    // matching β_ref which the caller's full-data fit also leaves on that scale).
    let beta_vec: Vec<f64> = (0..d).map(|j| beta_b[j] * y_scale / x_scale[j]).collect();

    Ok(ConfirmatoryWorkerRow {
        leverage,
        holdout_corr,
        beta: beta_vec,
    })
}

#[cfg(test)]
mod tests_worker {
    use super::*;
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::rng::resolve_seed;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < 2 { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
        (x, y)
    }

    #[test]
    fn worker_returns_finite_outputs_with_signal() {
        let (x, y) = synth(100, 6, 4.0, 1);
        let m_ref_fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let w_ref = m_ref_fit.w_star.clone();

        let (_, mut rng) = resolve_seed(Some(2)).unwrap();
        let row = run_one_confirmatory(
            x.as_ref(),
            y.as_ref(),
            2,
            resolve_m(100, 0.7),
            w_ref.as_ref(),
            false,
            None,
            &mut rng,
        )
        .unwrap();

        assert_eq!(row.leverage.len(), 6);
        for v in &row.leverage {
            assert!(
                (0.0..=1.5).contains(v),
                "leverage out of expected range: {v}"
            );
        }
        assert!(row.holdout_corr.is_finite() || row.holdout_corr.is_nan());
        assert_eq!(row.beta.len(), 6);
    }

    /// Regression for review-finding R4 (ticket #1, 2026-05-10): a
    /// subsample whose NIPALS fit short-circuits with `k_used < k`
    /// produces a truncated `w_b`; procrustes returns `DimensionMismatch`.
    /// The pre-fix `.expect(...)` in `run_one_confirmatory` panicked the
    /// Rayon worker; the post-fix `?` propagates as `Err`, which the
    /// outer driver maps to `WorkerOutcome::Failed`.
    ///
    /// Trigger: tiny y values + `pre_standardized=true` so plskit skips
    /// re-standardization. NIPALS sees `‖X'y‖ ≪ 1e-14` at component 0
    /// and short-circuits, yielding `w_b` with zero columns even though
    /// `w_ref` has `k=2`. Caller must not panic.
    #[test]
    fn worker_propagates_err_on_truncated_w_b() {
        let n = 40;
        let d = 4;
        let k = 2;
        let mut rng_data = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng_data.random_range(-1.0..1.0));
        let y_tiny = Col::<f64>::from_fn(n, |_| rng_data.random_range(-1e-20..1e-20));
        let w_ref = Mat::<f64>::from_fn(d, k, |i, j| if i == j { 1.0 } else { 0.0 });

        let (_, mut rng) = resolve_seed(Some(11)).unwrap();
        let res = run_one_confirmatory(
            x.as_ref(),
            y_tiny.as_ref(),
            k,
            resolve_m(n, 0.7),
            w_ref.as_ref(),
            true,
            None,
            &mut rng,
        );
        assert!(res.is_err(), "expected Err from truncated-w_b path, got Ok");
    }
}

/// Output of the confirmatory subsampling engine. Marshalled into
/// `ConfirmatoryCI` at the wrapper layer.
#[derive(Debug, Clone)]
pub struct ConfirmatoryCI {
    /// Number of resampling replicates. Each replicate draws one size-`m`
    /// subsample (β, `holdout_corr`) and one n-out-of-n bootstrap resample
    /// (leverage).
    pub n_boot: usize,
    /// Subsample size used for each subsample (β, `holdout_corr`).
    pub m: usize,
    /// Subsample rate `m_rate` such that `m = ceil(n^m_rate)`.
    pub m_rate: f64,
    /// Nominal CI level (e.g. 0.95).
    pub level: f64,
    /// Per-variable folded subsampling z for `β[j]`: `|z_j|` with
    /// `z_j = β_ref[j] / beta_se[j]`, i.e. `β_ref[j]` over a subsampling SE
    /// that carries the without-replacement finite-population correction
    /// and the shrinkage factor `κ̂` of the subsample fits (see
    /// `reduce_beta`). Length D. Approximately half-normal when the
    /// population `β[j]` is 0 (K = 1; see `_docs/python/results.md`,
    /// "Per-variable readouts", for the calibration range); does not grow
    /// with `n_boot`. NaN when the subsample spread of `β_b[j]` is zero or
    /// undefined and `β_ref[j] ≠ 0`.
    pub beta_sign_z: Vec<f64>,
    /// Per-variable signed subsampling z = `z_j` above, so
    /// `beta_sign_z_signed[j] = sign(β_ref[j]) · beta_sign_z[j]`. Length D.
    /// Approximately `N(0, 1)` when the population `β[j]` is 0 (K = 1).
    pub beta_sign_z_signed: Vec<f64>,
    /// Per-variable leverage CI lower bound, `h[j] − Φ⁻¹(1 − α/2) ·
    /// leverage_se[j]` (normal-theory bootstrap interval centered on the
    /// full-data leverage; see `reduce_leverage`), clamped to [0, 1]
    /// (leverage is a diagonal entry of a projection matrix). Length D.
    /// Covers the expected leverage of a size-`n` fit, not its large-`n`
    /// limit. Reads 0 for weakly loaded variables.
    pub leverage_ci_lower: Vec<f64>,
    /// Per-variable leverage CI upper bound, `h[j] + Φ⁻¹(1 − α/2) ·
    /// leverage_se[j]`, clamped to [0, 1]. Length D. See `leverage_ci_lower`.
    pub leverage_ci_upper: Vec<f64>,
    /// Per-variable leverage bootstrap SE = `sd(h_b[j])` over the leverages
    /// `h_b[j]` of n-out-of-n bootstrap refits (rows drawn with
    /// replacement). Length D.
    pub leverage_se: Vec<f64>,
    /// Per-coordinate β CI lower bound. Length D. PLS1-only. Centered-scaled
    /// subsampling on the replicates rescaled by `1/κ̂`, at the
    /// finite-population rate `√(m/(n − m))` (see `reduce_beta`), so the
    /// shrinkage of the subsample fits neither moves the midpoint nor
    /// narrows the interval. Calibrated at `K = 1`; at `K ≥ 2` a directional
    /// sanity check. `(−∞, +∞)` when `κ̂ = 0`. See `_docs/python/results.md`
    /// ("Per-coordinate β CIs — diagnostic, with caveats") for the full caveats.
    pub beta_ci_lower: Vec<f64>,
    /// Per-coordinate β CI upper bound. Length D. See `beta_ci_lower`.
    pub beta_ci_upper: Vec<f64>,
    /// Per-coordinate β subsampling SE = `√(m/(n − m)) · sd(β_b[j]) / κ̂`,
    /// the SE `beta_sign_z` divides by. Length D. `+∞` when `κ̂ = 0`.
    pub beta_se: Vec<f64>,
    /// Holdout predictive correlation. CI is built via Fisher z-transform
    /// (atanh) with NB inflation applied on the z-scale, then back-transformed
    /// via tanh; CI bounds are guaranteed in (−1, 1) and are asymmetric on the
    /// r-scale. `point` is the subsample mean on the r-scale; `sd` is on the
    /// z-scale (see `CIScalar` doc).
    pub holdout_corr: CIScalar,
    /// Number of resamples whose worker fit succeeded (contributors to
    /// leverage, `beta_sign_z`). Equals `n_boot − n_worker_failed`.
    pub n_boot_finite: usize,
    /// Number of resamples whose `holdout_corr` is finite (contributors to
    /// the `holdout_corr` CI). Strictly ≤ `n_boot_finite`.
    pub n_boot_finite_holdout_corr: usize,
}

/// Fisher-transformed NB-Wald CI for the holdout predictive correlation.
///
/// Inference is performed on the variance-stabilized scale ζ = atanh(r):
///   ζ_b   = atanh(r_b)
///   Var_z = (1/B + (n − m)/m) · stddev²(ζ_b)
///   ci_z  = atanh(mean(r_b)) ± Φ⁻¹(1 − α/2) · sqrt(Var_z)
///   ci    = tanh(ci_z)        // bounds always in (−1, 1)
/// `point` stays on the r-scale (subsample mean) — the unbiased plug-in estimate
/// of ρ — while the CI center on the z-scale is `atanh(point)`. `sd` is the
/// z-scale SE; the r-scale CI is asymmetric and cannot be reconstructed from
/// `point ± sd`.
///
/// Filtering: NaN samples (failed workers) are dropped, and `|r_b| ≥ 1`
/// (degenerate subsamples — `atanh(±1) = ±∞`) are dropped from the z pool.
/// If the surviving pool is empty, returns a degenerate CI at `mean(r_b)` over
/// the non-NaN inputs (or 0.0 if all inputs are NaN).
#[allow(clippy::doc_markdown)]
fn reduce_holdout_corr(r_b: &[f64], n: usize, m: usize, level: f64) -> CIScalar {
    // Pool for the Fisher z-transform: finite and strictly inside (−1, 1).
    let z_b: Vec<f64> = r_b
        .iter()
        .filter(|v| v.is_finite() && v.abs() < 1.0)
        .map(|v| v.atanh())
        .collect();
    let b_count = z_b.len();

    // Point on r-scale uses the same filtered pool so it lies inside (−1, 1)
    // and `atanh(point)` is finite. Caller-side diagnostics already report the
    // pre-filter finite count via `n_boot_finite_holdout_corr`.
    #[allow(clippy::cast_precision_loss)]
    let point: f64 = if b_count == 0 {
        // Fall back to the mean of all non-NaN r_b (may include ±1). If the
        // entire set is NaN, default to 0.0.
        let nan_filtered: Vec<f64> = r_b.iter().copied().filter(|v| !v.is_nan()).collect();
        if nan_filtered.is_empty() {
            0.0
        } else {
            nan_filtered.iter().sum::<f64>() / nan_filtered.len() as f64
        }
    } else {
        // `mean of values strictly inside (−1, 1)` is itself strictly inside
        // (−1, 1), so `atanh(point)` below is finite.
        r_b.iter()
            .filter(|v| v.is_finite() && v.abs() < 1.0)
            .sum::<f64>()
            / b_count as f64
    };

    if b_count == 0 {
        return CIScalar {
            point,
            lower: point,
            upper: point,
            sd: 0.0,
        };
    }

    // Variance on the z-scale, centered on `mean(z_b)` (standard unbiased
    // estimator). The NB inflation factor is dimensionless and applies on
    // either scale; we apply it here to z-scale variance.
    #[allow(clippy::cast_precision_loss)]
    let mean_z: f64 = z_b.iter().sum::<f64>() / b_count as f64;
    let var_z: f64 = if b_count > 1 {
        z_b.iter().map(|z| (z - mean_z).powi(2)).sum::<f64>() / (b_count - 1) as f64
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)]
    let nb_factor = 1.0 / (b_count as f64) + (n as f64 - m as f64) / (m as f64);
    let se_z_nb = (nb_factor * var_z).sqrt();

    let alpha = 1.0 - level;
    let z_crit = standard_normal_inv(1.0 - alpha / 2.0);

    // Build the CI on the z-scale centered at atanh(point) (textbook Fisher
    // recipe), then back-transform endpoints via tanh.
    let center_z = point.atanh();
    let lower = (center_z - z_crit * se_z_nb).tanh();
    let upper = (center_z + z_crit * se_z_nb).tanh();

    CIScalar {
        point,
        lower,
        upper,
        sd: se_z_nb,
    }
}

/// Inverse standard-normal CDF (Acklam / Beasley-Springer / Wichura). Used
/// for the NB-Wald CI on `holdout_corr` and the normal-theory leverage CI
/// (level ∈ [0.5, 0.99] → no extreme tails).
#[allow(unused_parens, clippy::unreadable_literal, clippy::excessive_precision)]
fn standard_normal_inv(p: f64) -> f64 {
    // Wichura AS241. Reproduced from numerical-recipes idiom; tolerance better
    // than 1e-9 over [1e-300, 1 − 1e-9].
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = q * q;
        let num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r
            + 67265.770927008700853)
            * r
            + 45921.953931549871457)
            * r
            + 13731.693765509461125)
            * r
            + 1971.5909503065514427)
            * r
            + 133.14166789178437745)
            * r
            + 3.387132872796366608);
        let den = (((((((5226.495278852854561 * r + 28729.085735721942674) * r
            + 39307.89580009271061)
            * r
            + 21213.794301586595867)
            * r
            + 5394.1960214247511077)
            * r
            + 687.1870074920579083)
            * r
            + 42.313330701600911252)
            * r
            + 1.0);
        return q * num / den;
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-(r.ln())).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        let num = (((((((0.00077454501427834140764 * r + 0.0227238449892691845833) * r
            + 0.24178072517745061177)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734);
        let den = (((((((0.00000105075007164441684324 * r + 0.0005475938084995344946) * r
            + 0.0151986665636164571966)
            * r
            + 0.14810397642748007459)
            * r
            + 0.68976733498510000455)
            * r
            + 1.6763848301838038494)
            * r
            + 2.05319162663775882187)
            * r
            + 1.0);
        num / den
    } else {
        let r = r - 5.0;
        let num = (((((((0.000000201033439929228813265 * r + 0.0000271155556874348757815) * r
            + 0.0012426609473880784386)
            * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772);
        let den =
            (((((((0.00000000000204426310338993978564 * r + 0.00000014215117583164458887) * r
                + 0.000018463183175100546818)
                * r
                + 0.0007868691311456132591)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0);
        num / den
    };
    if q < 0.0 {
        -val
    } else {
        val
    }
}

/// Reduce `Option<ConfirmatoryWorkerRow>` rows under the `max_failure_rate`
/// contract. Errors with `ResampleFailureRateExceeded` if the combined
/// failure rate (`n_worker_failed + n_holdout_only_nan`) exceeds the
/// threshold; otherwise filters `None` rows and patches `n_boot`,
/// `n_boot_finite`, `n_boot_finite_holdout_corr` onto the result.
pub(crate) fn reduce_with_failure_check(
    opt_rows: Vec<Option<ConfirmatoryWorkerRow>>,
    opts: SubsampleOpts,
    n: usize,
    m: usize,
    leverage_ref: &[f64],
    beta_ref: ColRef<'_, f64>,
) -> PlsKitResult<ConfirmatoryCI> {
    let n_boot = opts.n_boot;
    let n_worker_failed = opt_rows.iter().filter(|r| r.is_none()).count();
    let rows: Vec<ConfirmatoryWorkerRow> = opt_rows.into_iter().flatten().collect();
    let n_holdout_only_nan = rows.iter().filter(|r| r.holdout_corr.is_nan()).count();
    let n_holdout_corr_failed = n_worker_failed + n_holdout_only_nan;

    #[allow(clippy::cast_precision_loss)]
    let observed_worker = n_worker_failed as f64 / n_boot as f64;
    #[allow(clippy::cast_precision_loss)]
    let observed_holdout_corr = n_holdout_corr_failed as f64 / n_boot as f64;

    if observed_holdout_corr > opts.max_failure_rate {
        return Err(PlsKitError::ResampleFailureRateExceeded {
            max_failure_rate: opts.max_failure_rate,
            observed_worker,
            observed_holdout_corr,
            n_worker_failed,
            n_holdout_corr_failed,
            n_boot,
        });
    }

    let mut ci = reduce_confirmatory(&rows, n, m, opts.m_rate, opts.level, leverage_ref, beta_ref);
    ci.n_boot = n_boot;
    ci.n_boot_finite = n_boot - n_worker_failed;
    ci.n_boot_finite_holdout_corr = n_boot - n_holdout_corr_failed;
    Ok(ci)
}

/// Per-coordinate β readouts from the subsample replicates `β_b[j]`: the CI
/// bounds, the SE, and the signed subsampling z. Built by `reduce_beta`.
struct BetaReduction {
    ci_lower: Vec<f64>,
    ci_upper: Vec<f64>,
    se: Vec<f64>,
    z_signed: Vec<f64>,
}

/// Per-coordinate β CI, SE and signed z (`beta_ci_*`, `beta_se`,
/// `beta_sign_z_signed`), all resting on one corrected subsampling SE:
///
/// ```text
/// se_j = √(m/(n − m)) · sd(β_b[j]) / κ̂
/// z_j  = β_ref[j] / se_j
/// CI_j = β_ref[j] − √(m/(n − m)) · [q_{1−α/2}, q_{α/2}] of (β_b[j]/κ̂ − β_ref[j])
/// ```
///
/// Two corrections to textbook centered-scaled subsampling at `√(m/n)`:
///
/// 1. Finite-population correction: the rate `√(m/(n − m))` (`fpc_rate`).
/// 2. Shrinkage factor `κ̂` (`shrinkage_kappa`). A PLS fit on `m < n` rows
///    shrinks the whole coefficient vector toward 0 more than the full fit
///    does (at `K = 1`, `β = w·(wᵀXᵀy)/(wᵀXᵀXw)` with `w ∝ Xᵀy`, and the
///    scalar factor falls as `m` shrinks), so `β_b ≈ κ·β_ref + noise` with
///    the noise shrunk by the same `κ`. Dividing the replicates by `κ̂`
///    removes both effects: the `(κ − 1)·β_ref` offset that would move the
///    CI midpoint away from 0, and the understated spread.
///
/// Calibrated at `K = 1` only: at `K ≥ 2` the shrinkage is per component,
/// not one scalar, and neither the z nor the CI is calibrated.
///
/// If `κ̂ = 0` (the subsample fits do not reproduce the full fit) the SE is
/// `+∞`, every CI is `(−∞, +∞)` and every `z_j` is 0. `z_j` is NaN when
/// `β_ref[j]` is not finite, or when `se_j` is 0 or undefined while
/// `β_ref[j] ≠ 0` (no subsample spread to scale by); it is 0 when
/// `β_ref[j] = 0`. A coordinate with no finite replicate gets the degenerate
/// CI at `β_ref[j]` and `se_j = 0`, as in `reduce_centered_scaled`.
fn reduce_beta(
    rows: &[ConfirmatoryWorkerRow],
    beta_ref: &[f64],
    n: usize,
    m: usize,
    level: f64,
) -> BetaReduction {
    let d = beta_ref.len();
    // One replicate column at a time through a reused buffer, as the
    // leverage reduction does: a `d × B` copy of the replicates would add
    // `d·B` floats on top of the rows at voxel-scale `d`. κ̂ is a slope over
    // all `d` coordinates, so pass 2 cannot start before pass 1 has seen
    // every column; what it needs from pass 1 is only `beta_mean` and
    // `beta_sd`, so each column is gathered twice rather than kept.
    let mut col = vec![0.0_f64; rows.len()];

    // Pass 1: raw replicate mean and sd (unit rate factor) for κ̂.
    let mut beta_mean = vec![f64::NAN; d];
    let mut beta_sd = vec![0.0_f64; d];
    for j in 0..d {
        for (c, r) in col.iter_mut().zip(rows) {
            *c = r.beta[j];
        }
        beta_sd[j] = reduce_centered_scaled(&col, beta_ref[j], 1.0, level).sd;
        let (sum, count) = col
            .iter()
            .filter(|v| !v.is_nan())
            .fold((0.0_f64, 0_usize), |(s, c), &v| (s + v, c + 1));
        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let mean = sum / count as f64;
            beta_mean[j] = mean;
        }
    }
    let kappa = shrinkage_kappa(beta_ref, &beta_mean, &beta_sd);

    // Pass 2: CI and SE on the κ̂-rescaled replicates at the FPC rate.
    let scale = fpc_rate(n, m);
    let mut ci_lower = vec![f64::NEG_INFINITY; d];
    let mut ci_upper = vec![f64::INFINITY; d];
    let mut se = vec![f64::INFINITY; d];
    if kappa > 0.0 {
        for j in 0..d {
            for (c, r) in col.iter_mut().zip(rows) {
                *c = r.beta[j] / kappa;
            }
            let r = reduce_centered_scaled(&col, beta_ref[j], scale, level);
            ci_lower[j] = r.lower;
            ci_upper[j] = r.upper;
            se[j] = r.sd;
        }
    }
    let z_signed = (0..d)
        .map(|j| {
            let b = beta_ref[j];
            if !b.is_finite() {
                f64::NAN
            } else if b == 0.0 {
                0.0
            } else if se[j] > 0.0 {
                // `+∞` (κ̂ = 0) gives 0; NaN fails the test above.
                b / se[j]
            } else {
                f64::NAN
            }
        })
        .collect();
    BetaReduction {
        ci_lower,
        ci_upper,
        se,
        z_signed,
    }
}

/// Shrinkage factor `κ̂` of the subsample fits relative to the full fit:
/// the precision-weighted least-squares slope of the subsample means on
/// `β_ref`, `κ̂ = Σ_j t̄_j·t_j / Σ_j t_j²` with `t_j = β_ref[j]/sd_j` and
/// `t̄_j = mean(β_b[j])/sd_j`. Invariant to rescaling any column of X, and
/// to a common factor on every `sd_j`. Clamped to `[0, 1]`: values above 1
/// are estimation noise, and a non-positive slope means the subsample fits
/// do not reproduce the full fit. If no coordinate has a positive finite
/// `sd_j`, `κ̂ = 1`.
fn shrinkage_kappa(beta_ref: &[f64], beta_mean: &[f64], beta_sd: &[f64]) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for j in 0..beta_ref.len() {
        let sd = beta_sd[j];
        if beta_ref[j].is_finite() && beta_mean[j].is_finite() && sd.is_finite() && sd > 0.0 {
            let t = beta_ref[j] / sd;
            let t_bar = beta_mean[j] / sd;
            num += t_bar * t;
            den += t * t;
        }
    }
    if den > 0.0 && (num / den).is_finite() {
        (num / den).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Reduce per-resample worker rows into the final `ConfirmatoryCI` payload.
/// Computes per-variable leverage CIs and SEs from the bootstrap replicates
/// (`reduce_leverage`, bounds clamped to [0, 1]), per-coordinate β CIs, SEs
/// and subsampling z (`reduce_beta`), and the `holdout_corr` CI
/// (`reduce_holdout_corr`).
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::similar_names
)]
pub(crate) fn reduce_confirmatory(
    rows: &[ConfirmatoryWorkerRow],
    n: usize,
    m: usize,
    m_rate: f64,
    level: f64,
    leverage_ref: &[f64],
    beta_ref: ColRef<'_, f64>,
) -> ConfirmatoryCI {
    let b = rows.len();
    let d = leverage_ref.len();

    // ── per-variable leverage CI ──
    let mut leverage_ci_lower = vec![0.0_f64; d];
    let mut leverage_ci_upper = vec![0.0_f64; d];
    let mut leverage_se = vec![0.0_f64; d];
    let mut col_buf = vec![0.0_f64; b];
    for j in 0..d {
        for i in 0..b {
            col_buf[i] = rows[i].leverage[j];
        }
        let r = reduce_leverage(&col_buf, leverage_ref[j], level);
        // Leverage is a diagonal entry of a projection matrix, so it lies in
        // [0, 1]. The normal interval does not know that: for a weakly
        // loaded variable `h − 1.96·se` is below 0. Intersect with the
        // parameter space; `leverage_se` is unchanged.
        leverage_ci_lower[j] = r.lower.clamp(0.0, 1.0);
        leverage_ci_upper[j] = r.upper.clamp(0.0, 1.0);
        leverage_se[j] = r.sd;
    }

    // ── per-coordinate β CI, SE and z (PLS1 only: β is rotation/sign invariant) ──
    let beta_ref_vec: Vec<f64> = (0..d).map(|j| beta_ref[j]).collect();
    let beta = reduce_beta(rows, &beta_ref_vec, n, m, level);
    let beta_sign_z: Vec<f64> = beta.z_signed.iter().map(|z| z.abs()).collect();

    // ── holdout_corr (NB-Wald CI) ──
    let mut hc_buf = vec![0.0_f64; b];
    for i in 0..b {
        hc_buf[i] = rows[i].holdout_corr;
    }
    let holdout_corr = reduce_holdout_corr(&hc_buf, n, m, level);

    ConfirmatoryCI {
        n_boot: b,
        m,
        m_rate,
        level,
        beta_sign_z,
        beta_sign_z_signed: beta.z_signed,
        leverage_ci_lower,
        leverage_ci_upper,
        leverage_se,
        beta_ci_lower: beta.ci_lower,
        beta_ci_upper: beta.ci_upper,
        beta_se: beta.se,
        holdout_corr,
        n_boot_finite: b,
        n_boot_finite_holdout_corr: b,
    }
}

#[cfg(test)]
mod tests_reduce {
    use super::*;

    #[test]
    fn standard_normal_inv_known_values() {
        // Φ⁻¹(0.975) ≈ 1.959964
        assert!((standard_normal_inv(0.975) - 1.959_964).abs() < 1e-4);
        // Φ⁻¹(0.5) = 0
        assert!(standard_normal_inv(0.5).abs() < 1e-6);
    }

    #[test]
    fn reduce_holdout_corr_widens_with_overlap_factor() {
        // Generate B deterministic samples around 0.4 with mild dispersion. The
        // NB inflation (1/B + (n−m)/m) must widen the SE relative to a vanilla
        // 1/B-scaled estimator, and the back-transformed CI bounds must lie
        // strictly inside (−1, 1).
        let b_count = 1000;
        let mut samples = vec![0.0_f64; b_count];
        for (i, s) in samples.iter_mut().enumerate() {
            let phi = (i as f64).sin();
            *s = 0.4 + 0.05 * phi;
        }
        let n = 1000_usize;
        let m = 126_usize;
        let nb = reduce_holdout_corr(&samples, n, m, 0.95);

        // Self-comparison on the z-scale: the NB factor inflates Var by the
        // ratio (1/B + (n−m)/m) / (1/B), which exceeds 1 whenever (n−m)/m > 0.
        #[allow(clippy::cast_precision_loss)]
        let mean_z: f64 = samples.iter().map(|r| r.atanh()).sum::<f64>() / b_count as f64;
        #[allow(clippy::cast_precision_loss)]
        let var_z: f64 = samples
            .iter()
            .map(|r| (r.atanh() - mean_z).powi(2))
            .sum::<f64>()
            / (b_count - 1) as f64;
        #[allow(clippy::cast_precision_loss)]
        let se_no_nb = (var_z / b_count as f64).sqrt();
        assert!(
            nb.sd > se_no_nb,
            "NB-inflated z-scale SE must exceed the un-inflated SE: nb.sd={} se_no_nb={}",
            nb.sd,
            se_no_nb
        );

        // Bounds always within the correlation domain after tanh().
        assert!(nb.lower > -1.0 && nb.lower < 1.0, "lower={}", nb.lower);
        assert!(nb.upper > -1.0 && nb.upper < 1.0, "upper={}", nb.upper);
        // Point sits inside the CI (Fisher CI is monotone in the z-scale center).
        assert!(nb.lower < nb.point && nb.point < nb.upper);
    }

    #[test]
    fn reduce_holdout_corr_drops_degenerate_pm_one() {
        // Mix in a single r_b = 1.0 (degenerate subsample); it must be dropped
        // from the Fisher pool without producing an infinite SE.
        let mut samples: Vec<f64> = (0..50).map(|i| 0.3 + 0.02 * f64::from(i).sin()).collect();
        samples.push(1.0);
        samples.push(-1.0);
        let ci = reduce_holdout_corr(&samples, 200, 60, 0.95);
        assert!(ci.sd.is_finite());
        assert!(ci.lower > -1.0 && ci.upper < 1.0);
    }

    #[test]
    fn reduce_holdout_corr_bounds_clip_for_strong_signal() {
        // Samples concentrated near 1.0; the un-transformed Wald CI would
        // overflow ±1, but the Fisher CI must stay strictly inside.
        let samples: Vec<f64> = (0..200)
            .map(|i| 0.97 + 0.005 * f64::from(i).cos())
            .collect();
        let ci = reduce_holdout_corr(&samples, 200, 60, 0.95);
        assert!(ci.upper < 1.0, "upper={} must be < 1", ci.upper);
        assert!(ci.lower > -1.0);
        // Asymmetric on the r-scale: the upper arm is shorter than the lower
        // (atanh stretches near 1), so the CI is no longer symmetric.
        let upper_arm = ci.upper - ci.point;
        let lower_arm = ci.point - ci.lower;
        assert!(
            upper_arm < lower_arm,
            "expected upper arm < lower arm near r=0.97: upper_arm={upper_arm} lower_arm={lower_arm}"
        );
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::needless_range_loop)]
mod tests_beta_z_and_leverage_clamp {
    use super::*;
    use faer::Col;

    fn row(leverage: Vec<f64>, beta: Vec<f64>) -> ConfirmatoryWorkerRow {
        ConfirmatoryWorkerRow {
            leverage,
            holdout_corr: 0.5,
            beta,
        }
    }

    /// Evenly spaced deviations on `[-a, a]`: exact mean 0.
    #[allow(clippy::cast_precision_loss)]
    fn grid(b: usize, a: f64) -> Vec<f64> {
        (0..b)
            .map(|i| a * (2.0 * (i as f64 + 0.5) / b as f64 - 1.0))
            .collect()
    }

    fn sd(v: &[f64]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let nf = v.len() as f64;
        let mean = v.iter().sum::<f64>() / nf;
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt()
    }

    /// `beta_sign_z` is `β_ref / se` with the finite-population-corrected
    /// subsampling SE, and it neither grows with the number of replicates nor
    /// depends on a common shrinkage factor of the subsample fits. The former
    /// `(2p̂ − 1)·√B` formula fails every assertion here: all replicates share
    /// the sign of `β_ref`, so it returns `√B` (14.1 and 44.7).
    #[test]
    fn beta_sign_z_is_shrinkage_and_n_boot_invariant() {
        let (n, m) = (100_usize, 26_usize);
        let beta_ref = [0.3_f64, -0.3];
        let a = 0.25; // |δ| ≤ a < |β_ref|: every replicate keeps β_ref's sign
        let beta_ref_col = Col::<f64>::from_fn(2, |j| beta_ref[j]);
        #[allow(clippy::cast_precision_loss)]
        let c = ((m as f64) / ((n - m) as f64)).sqrt();
        for &b in &[200_usize, 2000] {
            let delta = grid(b, a);
            let expected = [
                beta_ref[0] / (c * sd(&delta)),
                beta_ref[1] / (c * sd(&delta)),
            ];
            for &kappa in &[1.0_f64, 0.6] {
                let rows: Vec<ConfirmatoryWorkerRow> = delta
                    .iter()
                    .map(|&dl| {
                        row(
                            vec![0.5, 0.5],
                            vec![kappa * (beta_ref[0] + dl), kappa * (beta_ref[1] - dl)],
                        )
                    })
                    .collect();
                let ci =
                    reduce_confirmatory(&rows, n, m, 0.7, 0.95, &[0.5, 0.5], beta_ref_col.as_ref());
                for j in 0..2 {
                    assert!(
                        (ci.beta_sign_z_signed[j] - expected[j]).abs() < 1e-9,
                        "b={b} kappa={kappa} j={j}: z={} expected={}",
                        ci.beta_sign_z_signed[j],
                        expected[j]
                    );
                    assert!((ci.beta_sign_z[j] - expected[j].abs()).abs() < 1e-9);
                }
            }
        }
    }

    /// The β CI is built from the same corrected replicates as
    /// `beta_sign_z`: replicates rescaled by `1/κ̂` (so the shrinkage offset
    /// does not move the midpoint) and spread scaled by `√(m/(n − m))` (the
    /// finite-population correction). With symmetric `δ` it is therefore
    /// centered on `β_ref` for any common shrinkage `κ`, its half-width is
    /// `√(m/(n − m))` times the `δ` quantile, `beta_se` is the SE the z
    /// divides by, and `beta_sign_z_signed = β_ref / beta_se` exactly.
    #[test]
    fn beta_ci_is_shrinkage_corrected_and_fpc_scaled() {
        let (n, m, b) = (100_usize, 26_usize, 2000_usize);
        let beta_ref = [0.3_f64, -0.3];
        let beta_ref_col = Col::<f64>::from_fn(2, |j| beta_ref[j]);
        #[allow(clippy::cast_precision_loss)]
        let c = ((m as f64) / ((n - m) as f64)).sqrt();
        let delta = grid(b, 0.25);
        let mut sorted = delta.clone();
        sorted.sort_by(f64::total_cmp);
        let half = c * crate::linalg::empirical_quantile(&sorted, 0.975);
        for &kappa in &[1.0_f64, 0.6] {
            let rows: Vec<ConfirmatoryWorkerRow> = delta
                .iter()
                .map(|&dl| {
                    row(
                        vec![0.5, 0.5],
                        vec![kappa * (beta_ref[0] + dl), kappa * (beta_ref[1] - dl)],
                    )
                })
                .collect();
            let ci =
                reduce_confirmatory(&rows, n, m, 0.7, 0.95, &[0.5, 0.5], beta_ref_col.as_ref());
            for j in 0..2 {
                let what = format!("kappa={kappa} j={j}");
                let mid = f64::midpoint(ci.beta_ci_lower[j], ci.beta_ci_upper[j]);
                let hw = 0.5 * (ci.beta_ci_upper[j] - ci.beta_ci_lower[j]);
                assert!((mid - beta_ref[j]).abs() < 1e-9, "{what}: midpoint {mid}");
                assert!(
                    (hw - half).abs() < 1e-9,
                    "{what}: half-width {hw} vs {half}"
                );
                assert!(
                    (ci.beta_se[j] - c * sd(&delta)).abs() < 1e-9,
                    "{what}: beta_se {} vs {}",
                    ci.beta_se[j],
                    c * sd(&delta)
                );
                assert!(
                    (ci.beta_sign_z_signed[j] - beta_ref[j] / ci.beta_se[j]).abs() < 1e-12,
                    "{what}: z {} vs β/se {}",
                    ci.beta_sign_z_signed[j],
                    beta_ref[j] / ci.beta_se[j]
                );
            }
        }
    }

    #[test]
    fn beta_reduction_edge_cases() {
        // κ̂ skips coordinates without a positive finite sd, clamps to
        // [0, 1], and is 1 when no coordinate is usable.
        assert_eq!(shrinkage_kappa(&[1.0, 1.0], &[1.0, 0.5], &[0.0, 0.5]), 0.5);
        assert_eq!(
            shrinkage_kappa(&[1.0, 2.0], &[-1.0, -2.0], &[0.5, 0.5]),
            0.0
        );
        assert_eq!(shrinkage_kappa(&[1.0], &[3.0], &[0.5]), 1.0);
        assert_eq!(shrinkage_kappa(&[1.0], &[1.0], &[0.0]), 1.0);

        // β_ref = 0 → z = 0; zero spread with β_ref ≠ 0 → NaN (degenerate
        // CI at β_ref, se = 0); an ordinary coordinate → z = β_ref / se.
        let (n, m) = (100_usize, 26_usize);
        let delta = grid(200, 0.25);
        let beta_ref = [0.0_f64, 1.0, 1.0];
        let rows: Vec<ConfirmatoryWorkerRow> = delta
            .iter()
            .map(|&dl| row(vec![0.5; 3], vec![dl, 1.0, 1.0 + dl]))
            .collect();
        let r = reduce_beta(&rows, &beta_ref, n, m, 0.95);
        assert_eq!(r.z_signed[0], 0.0);
        assert!(r.z_signed[1].is_nan());
        assert_eq!((r.ci_lower[1], r.ci_upper[1], r.se[1]), (1.0, 1.0, 0.0));
        assert!(r.se[2] > 0.0 && (r.z_signed[2] - 1.0 / r.se[2]).abs() < 1e-12);

        // Subsample means anti-aligned with β_ref: κ̂ = 0, so the SE is
        // infinite, every CI is the whole line and every z is 0.
        let beta_ref = [1.0_f64, 2.0];
        let rows: Vec<ConfirmatoryWorkerRow> = delta
            .iter()
            .map(|&dl| row(vec![0.5; 2], vec![-1.0 - dl, -2.0 + dl]))
            .collect();
        let r = reduce_beta(&rows, &beta_ref, n, m, 0.95);
        for j in 0..2 {
            assert_eq!(r.z_signed[j], 0.0);
            assert_eq!(r.se[j], f64::INFINITY);
            assert_eq!(
                (r.ci_lower[j], r.ci_upper[j]),
                (f64::NEG_INFINITY, f64::INFINITY)
            );
        }
    }

    /// Leverage lies in [0, 1]; the normal interval can leave it on either
    /// side, and the reducer clamps it back.
    #[test]
    fn leverage_ci_is_clamped_to_unit_interval() {
        let (n, m, b) = (100_usize, 26_usize, 200_usize);
        let delta = grid(b, 1.0);
        // Coordinate 0: h = 0.05 with a wide replicate spread (noise-like).
        // Coordinate 1: h = 0.95 with the same spread.
        let lev0: Vec<f64> = delta.iter().map(|dl| 0.05 + 0.3 * dl).collect();
        let lev1: Vec<f64> = delta.iter().map(|dl| 0.95 + 0.3 * dl).collect();
        let raw0 = reduce_leverage(&lev0, 0.05, 0.95);
        let raw1 = reduce_leverage(&lev1, 0.95, 0.95);
        assert!(
            raw0.lower < 0.0 && raw1.upper > 1.0,
            "fixture must leave [0, 1]"
        );
        let rows: Vec<ConfirmatoryWorkerRow> = (0..b)
            .map(|i| row(vec![lev0[i], lev1[i]], vec![1.0, 1.0]))
            .collect();
        let beta_ref_col = Col::<f64>::from_fn(2, |_| 1.0);
        let ci = reduce_confirmatory(&rows, n, m, 0.7, 0.95, &[0.05, 0.95], beta_ref_col.as_ref());
        assert_eq!(ci.leverage_ci_lower[0], 0.0);
        assert_eq!(ci.leverage_ci_upper[1], 1.0);
        assert_eq!(ci.leverage_ci_upper[0], raw0.upper.clamp(0.0, 1.0));
        assert_eq!(ci.leverage_ci_lower[1], raw1.lower.clamp(0.0, 1.0));
        assert_eq!(ci.leverage_se[0], raw0.sd);
        assert_eq!(ci.leverage_se[1], raw1.sd);
        for j in 0..2 {
            assert!((0.0..=1.0).contains(&ci.leverage_ci_lower[j]));
            assert!((0.0..=1.0).contains(&ci.leverage_ci_upper[j]));
            assert!(ci.leverage_ci_lower[j] <= ci.leverage_ci_upper[j]);
        }
    }

    /// The leverage CI is the normal interval `h ± Φ⁻¹(1 − α/2)·sd(h_b)`
    /// over the bootstrap replicates, with `leverage_se = sd(h_b)`: no
    /// subsampling rate (the replicates are size-`n` fits, so `n` and `m`
    /// do not enter), and centered on `h` whatever the replicate mean. A
    /// common offset of the replicates (the resampling bias of leverage)
    /// and the skew of their distribution leave the interval unchanged; the
    /// former reflected subsampling interval moved by both.
    #[test]
    fn leverage_ci_is_normal_bootstrap_interval_centered_on_h() {
        let b = 2000_usize;
        let lev_ref = [0.4_f64, 0.3];
        // Skewed replicate deviations (squared grid) so a quantile-based
        // interval would be asymmetric; nothing touches the [0, 1] clamp.
        let delta: Vec<f64> = grid(b, 1.0).iter().map(|v| 0.1 * v * v.abs()).collect();
        let z = standard_normal_inv(0.975);
        let beta_ref_col = Col::<f64>::from_fn(2, |_| 1.0);
        let mut reference: Option<ConfirmatoryCI> = None;
        for &(n, m) in &[(100_usize, 26_usize), (2000, 206)] {
            for &offset in &[0.0_f64, -0.08] {
                let rows: Vec<ConfirmatoryWorkerRow> = delta
                    .iter()
                    .map(|&dl| {
                        row(
                            vec![lev_ref[0] + offset + dl, lev_ref[1] + offset + dl],
                            vec![1.0, 1.0],
                        )
                    })
                    .collect();
                let ci =
                    reduce_confirmatory(&rows, n, m, 0.7, 0.95, &lev_ref, beta_ref_col.as_ref());
                for j in 0..2 {
                    let what = format!("n={n} offset={offset} j={j}");
                    assert!((ci.leverage_se[j] - sd(&delta)).abs() < 1e-12, "{what}");
                    let lo = lev_ref[j] - z * ci.leverage_se[j];
                    let hi = lev_ref[j] + z * ci.leverage_se[j];
                    assert!((ci.leverage_ci_lower[j] - lo).abs() < 1e-12, "{what}");
                    assert!((ci.leverage_ci_upper[j] - hi).abs() < 1e-12, "{what}");
                }
                if let Some(r) = &reference {
                    for j in 0..2 {
                        assert!((ci.leverage_ci_lower[j] - r.leverage_ci_lower[j]).abs() < 1e-12);
                        assert!((ci.leverage_ci_upper[j] - r.leverage_ci_upper[j]).abs() < 1e-12);
                    }
                } else {
                    reference = Some(ci);
                }
            }
        }
    }

    /// Fewer than two finite replicates give the degenerate CI at `h`.
    #[test]
    fn leverage_ci_degenerate_without_spread() {
        let r = reduce_leverage(&[f64::NAN, 0.3, f64::NAN], 0.25, 0.95);
        assert_eq!((r.lower, r.upper, r.sd), (0.25, 0.25, 0.0));
        let r = reduce_leverage(&[0.375; 64], 0.25, 0.95);
        assert_eq!((r.lower, r.upper, r.sd), (0.25, 0.25, 0.0));
    }
}

#[cfg(test)]
mod tests_failure_check {
    use super::*;
    use faer::Col;

    fn dummy_row(holdout_corr: f64) -> ConfirmatoryWorkerRow {
        ConfirmatoryWorkerRow {
            leverage: vec![0.5_f64; 2],
            holdout_corr,
            beta: vec![1.0_f64, 1.0_f64],
        }
    }

    fn opts_with(n_boot: usize, max_failure_rate: f64) -> SubsampleOpts {
        SubsampleOpts {
            n_boot,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            disable_parallelism: true,
            max_failure_rate,
            max_skip_rate: 1.0,
        }
    }

    fn run(
        opt_rows: Vec<Option<ConfirmatoryWorkerRow>>,
        opts: SubsampleOpts,
    ) -> PlsKitResult<ConfirmatoryCI> {
        let leverage_ref = vec![0.5_f64; 2];
        let beta_ref_col = Col::<f64>::from_fn(2, |_| 1.0);
        reduce_with_failure_check(
            opt_rows,
            opts,
            100,
            26,
            &leverage_ref,
            beta_ref_col.as_ref(),
        )
    }

    #[test]
    fn boundary_pass_99_some_1_none_at_threshold_001() {
        let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..99).map(|_| Some(dummy_row(0.5))).collect();
        rows.push(None);
        let ci = run(rows, opts_with(100, 0.01)).expect("at-threshold should pass");
        assert_eq!(ci.n_boot, 100);
        assert_eq!(ci.n_boot_finite, 99);
        assert_eq!(ci.n_boot_finite_holdout_corr, 99);
    }

    #[test]
    fn boundary_fail_98_some_2_none_at_threshold_001() {
        let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..98).map(|_| Some(dummy_row(0.5))).collect();
        rows.push(None);
        rows.push(None);
        let err = run(rows, opts_with(100, 0.01)).unwrap_err();
        match err {
            PlsKitError::ResampleFailureRateExceeded {
                observed_holdout_corr,
                observed_worker,
                n_worker_failed,
                n_holdout_corr_failed,
                n_boot,
                ..
            } => {
                assert!((observed_holdout_corr - 0.02).abs() < 1e-12);
                assert!((observed_worker - 0.02).abs() < 1e-12);
                assert_eq!(n_worker_failed, 2);
                assert_eq!(n_holdout_corr_failed, 2);
                assert_eq!(n_boot, 100);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn strict_clean_passes() {
        let rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..100).map(|_| Some(dummy_row(0.5))).collect();
        let ci = run(rows, opts_with(100, 0.0)).expect("clean strict should pass");
        assert_eq!(ci.n_boot, 100);
        assert_eq!(ci.n_boot_finite, 100);
        assert_eq!(ci.n_boot_finite_holdout_corr, 100);
    }

    #[test]
    fn strict_one_failure_errors() {
        let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..99).map(|_| Some(dummy_row(0.5))).collect();
        rows.push(None);
        let err = run(rows, opts_with(100, 0.0)).unwrap_err();
        match err {
            PlsKitError::ResampleFailureRateExceeded {
                observed_holdout_corr,
                n_worker_failed,
                ..
            } => {
                assert!((observed_holdout_corr - 0.01).abs() < 1e-12);
                assert_eq!(n_worker_failed, 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn holdout_pathology_only_strict_errors() {
        let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..50).map(|_| Some(dummy_row(0.5))).collect();
        for _ in 0..50 {
            rows.push(Some(dummy_row(f64::NAN)));
        }
        let err = run(rows, opts_with(100, 0.0)).unwrap_err();
        match err {
            PlsKitError::ResampleFailureRateExceeded {
                observed_worker,
                observed_holdout_corr,
                n_worker_failed,
                n_holdout_corr_failed,
                ..
            } => {
                assert!((observed_worker - 0.0).abs() < 1e-12);
                assert!((observed_holdout_corr - 0.5).abs() < 1e-12);
                assert_eq!(n_worker_failed, 0);
                assert_eq!(n_holdout_corr_failed, 50);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn holdout_pathology_under_threshold_passes() {
        let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..99).map(|_| Some(dummy_row(0.5))).collect();
        rows.push(Some(dummy_row(f64::NAN)));
        let ci = run(rows, opts_with(100, 0.01)).expect("1 holdout NaN under threshold");
        assert_eq!(ci.n_boot, 100);
        assert_eq!(ci.n_boot_finite, 100);
        assert_eq!(ci.n_boot_finite_holdout_corr, 99);
    }

    #[test]
    fn permissive_all_none_returns_degenerate_ci() {
        let rows: Vec<Option<ConfirmatoryWorkerRow>> = (0..100).map(|_| None).collect();
        let ci = run(rows, opts_with(100, 1.0)).expect("permissive must not error");
        assert_eq!(ci.n_boot, 100);
        assert_eq!(ci.n_boot_finite, 0);
        assert_eq!(ci.n_boot_finite_holdout_corr, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_centered_scaled_recovers_point_when_no_variation() {
        // All samples equal to point → Δ ≡ 0 → lower = upper = point, sd = 0.
        let samples = vec![3.0_f64; 100];
        let r = reduce_centered_scaled(&samples, 3.0, fpc_rate(1000, 126), 0.95);
        assert!((r.point - 3.0).abs() < 1e-12);
        assert!((r.lower - 3.0).abs() < 1e-12);
        assert!((r.upper - 3.0).abs() < 1e-12);
        assert!(r.sd.abs() < 1e-12);
    }

    #[test]
    fn reduce_centered_scaled_lower_le_upper_for_dispersed_samples() {
        // Synthetic dispersed Δ around point=0; symmetric → ci is symmetric-ish.
        #[allow(clippy::cast_lossless)]
        let samples: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) * 0.001).collect();
        let r = reduce_centered_scaled(&samples, 0.0, fpc_rate(1000, 100), 0.95);
        assert!(r.lower < r.upper, "lower={} upper={}", r.lower, r.upper);
        assert!(r.sd > 0.0);
    }
}

/// Tuning knobs for the subsampling engine driving `pls1_confirmatory_test(CI=Some)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SubsampleOpts {
    /// Number of bootstrap resamples. Must be ≥ 100.
    pub n_boot: usize,
    /// Subsample rate: `m = ceil(n^m_rate)`. Must satisfy `0.5 < m_rate < 0.95`.
    pub m_rate: f64,
    /// Nominal CI level (e.g. 0.95). Must satisfy `0.5 ≤ level ≤ 0.99`.
    pub level: f64,
    /// Whether the input `X` has already been column-standardized by the caller.
    pub pre_standardized: bool,
    /// If `true`, run resamples sequentially (disables Rayon parallelism). Useful for tests.
    pub disable_parallelism: bool,
    /// Maximum tolerable combined per-resample failure rate
    /// (`n_holdout_corr_failed / n_boot`). Default `0.01`. Range `[0.0, 1.0]`.
    /// `0.0` is strict; `1.0` is the legacy permissive behaviour.
    /// Distinct from `max_skip_rate`: covers numerical/fit failures, not weight-validation skips.
    pub max_failure_rate: f64,
    /// Threshold on the fraction of resamples skipped by weight validation.
    /// Default `0.01`. Range `[0.0, 1.0]`.
    /// Distinct from `max_failure_rate` (numerical failures): when a subsample's `pls1_fit`
    /// returns `InvalidWeights { reason: "insufficient_effective_n" }` due to insufficient
    /// effective sample size after weight-based row selection, that's a *skip* rather than
    /// a numerical *failure*. Fires `PlsKitError::ResamplingDegenerate` when exceeded.
    pub max_skip_rate: f64,
}

impl SubsampleOpts {
    /// Validate args (`m_rate` ∈ (0.5, 0.95), level ∈ [0.5, 0.99], `n_boot` ≥ 100).
    ///
    /// # Errors
    ///
    /// Returns `PlsKitError::InvalidArgument` if `m_rate`, `level`, `n_boot`,
    /// `max_failure_rate`, or `max_skip_rate` are out of the allowed ranges.
    #[allow(clippy::manual_range_contains, clippy::nonminimal_bool)]
    pub(crate) fn validate(&self) -> PlsKitResult<()> {
        if !(self.m_rate > 0.5 && self.m_rate < 0.95) {
            return Err(PlsKitError::InvalidArgument(format!(
                "m_rate must satisfy 0.5 < m_rate < 0.95, got {}",
                self.m_rate
            )));
        }
        if !(self.level >= 0.5 && self.level <= 0.99) {
            return Err(PlsKitError::InvalidArgument(format!(
                "level must satisfy 0.5 ≤ level ≤ 0.99, got {}",
                self.level
            )));
        }
        if self.n_boot < 100 {
            return Err(PlsKitError::InvalidArgument(format!(
                "n_boot must be ≥ 100, got {}",
                self.n_boot
            )));
        }
        if !(0.0..=1.0).contains(&self.max_failure_rate) {
            return Err(PlsKitError::InvalidArgument(format!(
                "max_failure_rate must be in [0.0, 1.0], got {}",
                self.max_failure_rate
            )));
        }
        if !(0.0..=1.0).contains(&self.max_skip_rate) {
            return Err(PlsKitError::InvalidArgument(format!(
                "max_skip_rate must be in [0.0, 1.0], got {}",
                self.max_skip_rate
            )));
        }
        Ok(())
    }
}

/// Three-way outcome of a single confirmatory worker call.
/// Distinguishes weight-validation skips from numerical failures so the
/// driver can apply two independent thresholds (`max_skip_rate` vs
/// `max_failure_rate`).
enum WorkerOutcome {
    /// Worker succeeded; carries the row data.
    Ok(ConfirmatoryWorkerRow),
    /// `pls1_fit` returned `InvalidWeights { reason: "insufficient_effective_n" }`.
    /// Counts toward `max_skip_rate`, not `max_failure_rate`.
    Skipped,
    /// Any other error (numerical failure, NaN, etc.).
    /// Counts toward `max_failure_rate` via the `None` path in `reduce_with_failure_check`.
    Failed,
}

/// Engine entry for the confirmatory CI branch. Caller has already done the
/// full-data reference fit and validated arguments.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
pub(crate) fn pls1_subsample_inference_confirmatory(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    w_ref: MatRef<'_, f64>,
    beta_ref: ColRef<'_, f64>,
    leverage_ref: &[f64],
    opts: SubsampleOpts,
    weights: Option<ColRef<'_, f64>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<ConfirmatoryCI> {
    opts.validate()?;
    let n = x.nrows();
    let m = resolve_m(n, opts.m_rate);
    // Empty holdout (m == n) → NaN holdout_corr on every resample →
    // ResampleFailureRateExceeded before any useful output. Reject early.
    if m >= n {
        return Err(PlsKitError::InvalidArgument(format!(
            "resolved m = {m} (from n={n}, m_rate={}) leaves no holdout; need m < n",
            opts.m_rate
        )));
    }
    if m < k + 2 {
        return Err(PlsKitError::InvalidArgument(format!(
            "resolved m = {m} (from n={n}, m_rate={}) is too small for k={k}; need m ≥ k+2",
            opts.m_rate
        )));
    }

    let pre_std = opts.pre_standardized;
    let outcomes: Vec<WorkerOutcome> = crate::resample::parallel_for_each_seeded(
        rng,
        opts.n_boot,
        opts.disable_parallelism,
        |_, child| match run_one_confirmatory(x, y, k, m, w_ref, pre_std, weights, child) {
            std::result::Result::Ok(row) => WorkerOutcome::Ok(row),
            Err(PlsKitError::InvalidWeights {
                reason: "insufficient_effective_n",
            }) => WorkerOutcome::Skipped,
            Err(_) => WorkerOutcome::Failed,
        },
    );

    // Check max_skip_rate before passing to reduce_with_failure_check.
    let n_skipped = outcomes
        .iter()
        .filter(|o| matches!(o, WorkerOutcome::Skipped))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let skip_rate = n_skipped as f64 / opts.n_boot as f64;
    if skip_rate > opts.max_skip_rate {
        return Err(PlsKitError::ResamplingDegenerate {
            skipped: n_skipped,
            total: opts.n_boot,
            skip_rate,
            threshold: opts.max_skip_rate,
        });
    }

    // Map outcomes to Option<ConfirmatoryWorkerRow>: Ok→Some, Skipped→None, Failed→None.
    // Both Skipped and Failed become None so reduce_with_failure_check counts them as
    // worker failures for the max_failure_rate check. This is intentionally conservative:
    // a skipped subsample still cannot contribute data, so it should be treated as failed
    // for the purpose of that secondary check.
    let opt_rows: Vec<Option<ConfirmatoryWorkerRow>> = outcomes
        .into_iter()
        .map(|o| match o {
            WorkerOutcome::Ok(row) => Some(row),
            WorkerOutcome::Skipped | WorkerOutcome::Failed => None,
        })
        .collect();

    reduce_with_failure_check(opt_rows, opts, n, m, leverage_ref, beta_ref)
}

#[cfg(test)]
mod tests_engine {
    use super::*;
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::rng::resolve_seed;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    fn synth(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < 2 { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
        (x, y)
    }

    #[test]
    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
    fn engine_runs_end_to_end_with_signal() {
        let (x, y) = synth(100, 6, 4.0, 42);
        let fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();

        let d = x.ncols();
        let leverage_ref = crate::linalg::leverage_diag(fit.w_star.as_ref());

        let opts = SubsampleOpts {
            n_boot: 200,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            disable_parallelism: true,
            max_failure_rate: 0.0,
            max_skip_rate: 1.0,
        };
        let (_, mut rng) = resolve_seed(Some(2026)).unwrap();
        let ci = pls1_subsample_inference_confirmatory(
            x.as_ref(),
            y.as_ref(),
            2,
            fit.w_star.as_ref(),
            fit.beta.as_ref(),
            &leverage_ref,
            opts,
            None,
            &mut rng,
        )
        .unwrap();

        assert_eq!(ci.n_boot, 200);
        assert_eq!(ci.m, 26); // ceil(100^0.7)
        assert_eq!(ci.beta_sign_z.len(), d);
        assert_eq!(ci.leverage_ci_lower.len(), d);
        assert!(ci.holdout_corr.lower.is_finite());
    }

    #[test]
    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
    fn engine_emits_signed_beta_sign_z() {
        let (x, y) = synth(120, 6, 5.0, 7);
        let fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();

        let d = x.ncols();
        let leverage_ref = crate::linalg::leverage_diag(fit.w_star.as_ref());

        let opts = SubsampleOpts {
            n_boot: 200,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            disable_parallelism: true,
            max_failure_rate: 0.0,
            max_skip_rate: 1.0,
        };
        let (_, mut rng) = resolve_seed(Some(2026)).unwrap();
        let ci = pls1_subsample_inference_confirmatory(
            x.as_ref(),
            y.as_ref(),
            2,
            fit.w_star.as_ref(),
            fit.beta.as_ref(),
            &leverage_ref,
            opts,
            None,
            &mut rng,
        )
        .unwrap();

        // Must have the new field, same length as folded form.
        assert_eq!(ci.beta_sign_z_signed.len(), d);
        // Magnitudes equal the folded form's magnitudes.
        for j in 0..d {
            assert!(
                (ci.beta_sign_z_signed[j].abs() - ci.beta_sign_z[j].abs()).abs() < 1e-12,
                "magnitude mismatch at j={}: signed={}, folded={}",
                j,
                ci.beta_sign_z_signed[j],
                ci.beta_sign_z[j],
            );
            // Sign matches sign of beta_ref[j] (when β_ref[j] ≠ 0).
            if fit.beta[j].abs() > 1e-12 {
                assert!(
                    ci.beta_sign_z_signed[j].signum() == fit.beta[j].signum()
                        || ci.beta_sign_z_signed[j].abs() < 1e-12,
                    "sign mismatch at j={}: signed={}, β_ref={}",
                    j,
                    ci.beta_sign_z_signed[j],
                    fit.beta[j],
                );
            }
        }
    }

    #[test]
    fn validate_rejects_bad_m_rate() {
        let opts = SubsampleOpts {
            n_boot: 1000,
            m_rate: 0.4,
            level: 0.95,
            pre_standardized: false,
            disable_parallelism: false,
            max_failure_rate: 1.0,
            max_skip_rate: 1.0,
        };
        let err = opts.validate().unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
    }

    #[test]
    fn validate_rejects_bad_level() {
        let opts = SubsampleOpts {
            n_boot: 1000,
            m_rate: 0.7,
            level: 0.999,
            pre_standardized: false,
            disable_parallelism: false,
            max_failure_rate: 1.0,
            max_skip_rate: 1.0,
        };
        assert_eq!(opts.validate().unwrap_err().code(), "invalid_argument");
    }

    #[test]
    fn validate_rejects_low_n_boot() {
        let opts = SubsampleOpts {
            n_boot: 50,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            disable_parallelism: false,
            max_failure_rate: 1.0,
            max_skip_rate: 1.0,
        };
        assert_eq!(opts.validate().unwrap_err().code(), "invalid_argument");
    }

    #[test]
    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
    fn engine_emits_n_boot_finite_diagnostics() {
        let (x, y) = synth(100, 6, 4.0, 42);
        let fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();

        let _d = x.ncols();
        let leverage_ref = crate::linalg::leverage_diag(fit.w_star.as_ref());

        let opts = SubsampleOpts {
            n_boot: 200,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            disable_parallelism: true,
            max_failure_rate: 0.0,
            max_skip_rate: 1.0,
        };
        let (_, mut rng) = resolve_seed(Some(2026)).unwrap();
        let ci = pls1_subsample_inference_confirmatory(
            x.as_ref(),
            y.as_ref(),
            2,
            fit.w_star.as_ref(),
            fit.beta.as_ref(),
            &leverage_ref,
            opts,
            None,
            &mut rng,
        )
        .unwrap();

        assert_eq!(ci.n_boot, 200);
        assert_eq!(ci.n_boot_finite, 200);
        assert_eq!(ci.n_boot_finite_holdout_corr, 200);
        assert!(ci.n_boot_finite_holdout_corr <= ci.n_boot_finite);
        assert!(ci.n_boot_finite <= ci.n_boot);
    }

    fn run_engine(
        x: &Mat<f64>,
        y: &Col<f64>,
        k: usize,
        n_boot: usize,
        seed: u64,
    ) -> ConfirmatoryCI {
        let fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(k),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let leverage_ref = crate::linalg::leverage_diag(fit.w_star.as_ref());
        let opts = SubsampleOpts {
            n_boot,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            disable_parallelism: false,
            max_failure_rate: 0.0,
            max_skip_rate: 1.0,
        };
        let (_, mut rng) = resolve_seed(Some(seed)).unwrap();
        pls1_subsample_inference_confirmatory(
            x.as_ref(),
            y.as_ref(),
            k,
            fit.w_star.as_ref(),
            fit.beta.as_ref(),
            &leverage_ref,
            opts,
            None,
            &mut rng,
        )
        .unwrap()
    }

    /// Under a pure null (y independent of X, K = 1) every population β[j]
    /// is 0, so `beta_sign_z` should be roughly half-normal: mean
    /// √(2/π) ≈ 0.80 and about 5% above 1.96. The former `(2p̂ − 1)·√B`
    /// formula averaged about 4.6 here with most values above 1.96.
    #[test]
    fn beta_sign_z_is_calibrated_under_pure_null() {
        let (n, d, reps) = (200_usize, 8_usize, 60_u64);
        let mut pooled = Vec::new();
        let mut excluded = 0_usize;
        for rep in 0..reps {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5000 + rep);
            let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
            let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
            let ci = run_engine(&x, &y, 1, 200, 7000 + rep);
            pooled.extend(ci.beta_sign_z.iter().copied());
            excluded += (0..d)
                .filter(|&j| ci.beta_ci_lower[j] > 0.0 || ci.beta_ci_upper[j] < 0.0)
                .count();
        }
        #[allow(clippy::cast_precision_loss)]
        let total = pooled.len() as f64;
        let mean = pooled.iter().sum::<f64>() / total;
        #[allow(clippy::cast_precision_loss)]
        let share = pooled.iter().filter(|z| **z > 1.96).count() as f64 / total;
        assert!(pooled.iter().all(|z| z.is_finite() && *z >= 0.0));
        assert!((0.6..=1.0).contains(&mean), "mean |z| = {mean}");
        assert!(share <= 0.10, "share |z| > 1.96 = {share}");
        // The β CI rests on the same corrected SE, so it covers the null
        // β[j] = 0 at roughly the nominal rate. Uncorrected (no FPC, no
        // 1/κ̂) it was κ̂·√(1 − m/n) too narrow and missed 0 far more often.
        #[allow(clippy::cast_precision_loss)]
        let miss = excluded as f64 / total;
        assert!(miss <= 0.10, "share of β CIs excluding 0 = {miss}");
    }

    /// The leverage CI covers the expected leverage of a size-`n` fit in
    /// the regime that exposed the reflected subsampling interval: `D = 20`,
    /// `n = 100`, `K = 1`, where a fit on `m = 26` rows puts much less
    /// leverage on the signal variables than the full fit. The target
    /// `E[ĥ_n]` is the mean full-data leverage over fresh datasets. The
    /// former interval covered it about half the time here (0.51 over 1000
    /// datasets); the bootstrap interval about 0.93.
    #[test]
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn leverage_ci_covers_expected_leverage_at_small_n_over_d() {
        let (n, d, snr) = (100_usize, 20_usize, 1.0_f64);
        let (n_oracle, reps) = (400_u64, 100_u64);
        let mut target = [0.0_f64; 2];
        for o in 0..n_oracle {
            let (x, y) = synth(n, d, snr, 90_000 + o);
            let fit = pls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(1),
                None,
                FitOpts::default(),
            )
            .unwrap();
            let h = crate::linalg::leverage_diag(fit.w_star.as_ref());
            target[0] += h[0] / n_oracle as f64;
            target[1] += h[1] / n_oracle as f64;
        }
        let mut covered = 0_usize;
        for rep in 0..reps {
            let (x, y) = synth(n, d, snr, 10_000 + rep);
            let ci = run_engine(&x, &y, 1, 200, 20_000 + rep);
            covered += (0..2)
                .filter(|&j| {
                    ci.leverage_ci_lower[j] <= target[j] && target[j] <= ci.leverage_ci_upper[j]
                })
                .count();
        }
        let coverage = covered as f64 / (2 * reps) as f64;
        assert!(coverage >= 0.85, "signal leverage coverage {coverage}");
    }

    /// Signal coordinates keep large z, noise coordinates stay in the
    /// half-normal range, and the leverage CI stays inside [0, 1].
    #[test]
    fn beta_sign_z_separates_signal_and_leverage_ci_in_unit_interval() {
        let (x, y) = synth(200, 8, 4.0, 42);
        let ci = run_engine(&x, &y, 1, 300, 2026);
        for j in 0..2 {
            assert!(
                ci.beta_sign_z[j] > 5.0,
                "signal j={j}: {}",
                ci.beta_sign_z[j]
            );
        }
        for j in 2..8 {
            assert!(
                ci.beta_sign_z[j] < 4.0,
                "noise j={j}: {}",
                ci.beta_sign_z[j]
            );
        }
        for j in 0..8 {
            assert!((0.0..=1.0).contains(&ci.leverage_ci_lower[j]));
            assert!((0.0..=1.0).contains(&ci.leverage_ci_upper[j]));
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]
mod copy_free_reference {
    use super::*;
    use crate::linalg::{col_row_subset, row_subset, standardize_apply};
    use crate::signal_test::with_new_routes_disabled;
    use crate::test_support::{assert_bits_eq, col_vals, copy_free_families, mat_vals, Layouts};

    /// Pre-change body, verbatim.
    fn fit_rows_reference(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        idx: &[usize],
        k: usize,
        pre_standardized_x: bool,
        weights: Option<ColRef<'_, f64>>,
    ) -> PlsKitResult<RowFit> {
        let d = x.ncols();
        let x_rows = row_subset(x, idx);
        let y_rows = col_row_subset(y, idx);

        let w_sub: Option<faer::Col<f64>> = weights.map(|w| crate::linalg::col_row_subset(w, idx));
        let (w_sub_norm, _, _) = crate::fit::validate_and_normalize_weights(
            w_sub.as_ref().map(faer::Col::as_ref),
            idx.len(),
            k,
        )?;

        let (xs_sub, x_mean, x_scale, ys_sub, y_scale) = if pre_standardized_x {
            // Caller asserts already standardized: the row subsets are the
            // blocks, moved rather than copied. Mean/scale are no-ops (zeros /
            // ones) for the holdout standardization and for the β
            // back-projection (β_b stays on the standardized scale, matching
            // β_ref which the caller's full-data fit also leaves on that scale).
            (
                x_rows,
                Col::<f64>::zeros(d),
                Col::<f64>::from_fn(d, |_| 1.0),
                y_rows,
                1.0_f64,
            )
        } else {
            let (xs, mu, sigma) = crate::linalg::standardize_weighted(
                x_rows.as_ref(),
                w_sub_norm.as_ref().map(faer::Col::as_ref),
            );
            let (ys, _, ys_sigma) = crate::linalg::standardize1_weighted(
                y_rows.as_ref(),
                w_sub_norm.as_ref().map(faer::Col::as_ref),
            );
            (xs, mu, sigma, ys, ys_sigma)
        };

        let fit = pls1_fit(
            xs_sub.as_ref(),
            ys_sub.as_ref(),
            KSpec::Fixed(k),
            w_sub_norm.as_ref().map(faer::Col::as_ref),
            FitOpts {
                pre_standardized: true,
                // Seq inside the per-resample worker: outer Rayon owns the threadpool.
                par: crate::fit::ParChoice::Seq,
                ..FitOpts::default()
            },
        )?;
        if fit.w_star.ncols() != k {
            return Err(PlsKitError::Internal(format!(
                "resample fit truncated to {} of {k} components",
                fit.w_star.ncols()
            )));
        }
        Ok(RowFit {
            w: fit.w_star,
            beta: fit.beta,
            x_mean,
            x_scale,
            y_scale,
        })
    }

    /// Pre-change body, verbatim.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    fn run_one_confirmatory_reference(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k: usize,
        m: usize,
        _w_ref: MatRef<'_, f64>, // retained for call-site shape parity; leverage no longer uses it
        pre_standardized_x: bool,
        weights: Option<ColRef<'_, f64>>,
        rng: &mut crate::rng::Rng,
    ) -> PlsKitResult<ConfirmatoryWorkerRow> {
        let n = x.nrows();
        let d = x.ncols();

        // 1. Index draws: subsample, then bootstrap resample.
        let (sample_idx, holdout_idx) = subsample_indices(n, m, rng);
        let boot_idx = bootstrap_indices(n, rng);

        // 2. Subsample fit (β, holdout_corr) and bootstrap fit (leverage).
        let RowFit {
            beta: beta_b,
            x_mean,
            x_scale,
            y_scale,
            ..
        } = fit_rows_reference(x, y, &sample_idx, k, pre_standardized_x, weights)?;
        let boot = fit_rows_reference(x, y, &boot_idx, k, pre_standardized_x, weights)?;

        // 3. Per-variable leverage on the bootstrap fit's unaligned W. Leverage
        // is invariant under right-orthogonal rotation (leverage =
        // diag(W(WᵀW)⁻¹Wᵀ); W → W·R leaves the hat matrix fixed), so no
        // procrustes alignment is needed.
        let leverage = crate::linalg::leverage_diag(boot.w.as_ref());

        // 4. Holdout predictive correlation of the subsample fit.
        let x_h_view = row_subset(x, &holdout_idx);
        let y_h_view = col_row_subset(y, &holdout_idx);
        let (xs_h, ys_h_owned) = if pre_standardized_x {
            (
                Mat::<f64>::from_fn(x_h_view.nrows(), d, |i, j| x_h_view[(i, j)]),
                Col::<f64>::from_fn(y_h_view.nrows(), |i| y_h_view[i]),
            )
        } else {
            let xs_h = standardize_apply(x_h_view.as_ref(), x_mean.as_ref(), x_scale.as_ref());
            (xs_h, Col::<f64>::from_fn(y_h_view.nrows(), |i| y_h_view[i]))
        };

        let y_pred: Col<f64> = &xs_h * &beta_b;
        let n_h = ys_h_owned.nrows();
        #[allow(clippy::cast_precision_loss)]
        let holdout_corr = if n_h >= 2 {
            let yp_mean: f64 = (0..n_h).map(|i| y_pred[i]).sum::<f64>() / n_h as f64;
            let yh_mean: f64 = (0..n_h).map(|i| ys_h_owned[i]).sum::<f64>() / n_h as f64;
            let mut s_pp = 0.0_f64;
            let mut s_yy = 0.0_f64;
            let mut s_py = 0.0_f64;
            for i in 0..n_h {
                #[allow(clippy::many_single_char_names)]
                let dp = y_pred[i] - yp_mean;
                #[allow(clippy::many_single_char_names)]
                let dy = ys_h_owned[i] - yh_mean;
                s_pp += dp * dp;
                s_yy += dy * dy;
                s_py += dp * dy;
            }
            if s_pp > 1e-30 && s_yy > 1e-30 {
                (s_py / (s_pp * s_yy).sqrt()).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        // 5. Back-project β_b to the same scale as β_ref so deltas are meaningful.
        // Mirrors `pls1_fit`'s full-data back-projection (`fit.rs:143-147`):
        //   β_raw[j] = β_std[j] * y_scale / x_scale[j]
        // For pre_standardized_x, both scales are 1.0 → no-op (β_b stays standardized,
        // matching β_ref which the caller's full-data fit also leaves on that scale).
        let beta_vec: Vec<f64> = (0..d).map(|j| beta_b[j] * y_scale / x_scale[j]).collect();

        Ok(ConfirmatoryWorkerRow {
            leverage,
            holdout_corr,
            beta: beta_vec,
        })
    }

    fn row_fit_bits(a: &RowFit, b: &RowFit, what: &str) {
        assert_bits_eq(
            &mat_vals(a.w.as_ref()),
            &mat_vals(b.w.as_ref()),
            &format!("{what}.w"),
        );
        assert_bits_eq(
            &col_vals(a.beta.as_ref()),
            &col_vals(b.beta.as_ref()),
            &format!("{what}.beta"),
        );
        assert_bits_eq(
            &col_vals(a.x_mean.as_ref()),
            &col_vals(b.x_mean.as_ref()),
            &format!("{what}.x_mean"),
        );
        assert_bits_eq(
            &col_vals(a.x_scale.as_ref()),
            &col_vals(b.x_scale.as_ref()),
            &format!("{what}.x_scale"),
        );
        assert_eq!(a.y_scale.to_bits(), b.y_scale.to_bits(), "{what}.y_scale");
    }

    // Per-unit body: no parallel axis; serial vs parallel is covered by
    // byte_parity (confirmatory_ci_bundle_byte_parity).
    #[test]
    fn fit_rows_and_the_confirmatory_worker_match_reference() {
        with_new_routes_disabled(|| {
            for f in copy_free_families() {
                let n = f.x.nrows();
                let w =
                    f.w.as_ref()
                        .map(|w| crate::linalg::normalize_weights(w.as_ref()).unwrap());
                let wr = w.as_ref().map(Col::as_ref);
                let (xs, _, _) = crate::linalg::standardize(f.x.as_ref());
                let (ys, _, _) = crate::linalg::standardize1(f.y.as_ref());
                let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
                // Bootstrap indices repeat rows.
                let boot = bootstrap_indices(n, &mut rng);
                let (sub, _) = subsample_indices(n, n * 2 / 3, &mut rng);
                for pre in [false, true] {
                    let (x0, y0) = if pre { (&xs, &ys) } else { (&f.x, &f.y) };
                    let lay = Layouts::new(x0.as_ref());
                    for (view, xv) in lay.all(x0) {
                        for (label, idx) in [("boot", &boot), ("subsample", &sub)] {
                            for k in [1_usize, 2] {
                                let what = format!("{} {view} {label} pre={pre} k={k}", f.name);
                                match (
                                    fit_rows(xv, y0.as_ref(), idx, k, pre, wr),
                                    fit_rows_reference(xv, y0.as_ref(), idx, k, pre, wr),
                                ) {
                                    (Ok(a), Ok(b)) => row_fit_bits(&a, &b, &what),
                                    (Err(a), Err(b)) => {
                                        assert_eq!(a.to_string(), b.to_string(), "{what}");
                                    }
                                    (a, b) => panic!("{what}: ok {} vs {}", a.is_ok(), b.is_ok()),
                                }
                            }
                        }
                        for seed in [1_u64, 2, 3] {
                            let (_, mut r1) = crate::rng::resolve_seed(Some(seed)).unwrap();
                            let (_, mut r2) = crate::rng::resolve_seed(Some(seed)).unwrap();
                            let w_ref = Mat::<f64>::zeros(f.x.ncols(), 2);
                            let what = format!("{} {view} pre={pre} worker seed {seed}", f.name);
                            match (
                                run_one_confirmatory(
                                    xv,
                                    y0.as_ref(),
                                    2,
                                    n * 2 / 3,
                                    w_ref.as_ref(),
                                    pre,
                                    wr,
                                    &mut r1,
                                ),
                                run_one_confirmatory_reference(
                                    xv,
                                    y0.as_ref(),
                                    2,
                                    n * 2 / 3,
                                    w_ref.as_ref(),
                                    pre,
                                    wr,
                                    &mut r2,
                                ),
                            ) {
                                (Ok(a), Ok(b)) => {
                                    assert_bits_eq(
                                        &a.leverage,
                                        &b.leverage,
                                        &format!("{what}.leverage"),
                                    );
                                    assert_bits_eq(&a.beta, &b.beta, &format!("{what}.beta"));
                                    assert_eq!(
                                        a.holdout_corr.to_bits(),
                                        b.holdout_corr.to_bits(),
                                        "{what}.holdout_corr"
                                    );
                                }
                                (Err(a), Err(b)) => {
                                    assert_eq!(a.to_string(), b.to_string(), "{what}");
                                }
                                (a, b) => panic!("{what}: ok {} vs {}", a.is_ok(), b.is_ok()),
                            }
                        }
                    }
                }
            }
        });
    }
}
