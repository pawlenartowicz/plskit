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
use crate::fit::{pls1_fit_impl, FitOpts, KSpec};
use crate::linalg::{col_row_subset, row_subset};

/// Per-resample outputs for the confirmatory CI branch. Per-variable arrays
/// only.
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
    let (w_sub_norm, _) = crate::fit::validate_and_normalize_weights(
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

    let fit = pls1_fit_impl(
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
    _w_ref: MatRef<'_, f64>, // unused: kept so the call site matches the other workers' signature
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
    // Mirrors `pls1_fit`'s full-data back-projection in `pls1_fit`:
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
    use crate::rng::resolve_seed;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    /// A subsample whose NIPALS fit short-circuits with `k_used < k` must
    /// not panic the Rayon worker. The replicate fit runs `pls1_fit` with `pre_standardized = true`, whose
    /// `k_used < k` guard returns `InvalidInput` ("truncated"); the worker
    /// propagates it with `?` and the outer driver maps it to
    /// `WorkerOutcome::Failed`.
    ///
    /// Trigger: tiny y values + `pre_standardized=true` so plskit skips
    /// re-standardization. NIPALS sees `‖X'y‖ ≪ 1e-14` at component 0
    /// and short-circuits with no component although `k = 2`.
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
        assert!(
            matches!(&res, Err(PlsKitError::InvalidInput(msg)) if msg.contains("truncated")),
            "{res:?}"
        );
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

/// Inverse standard-normal CDF (Wichura 1988, AS241 `PPND16`). Used for the
/// NB-Wald CI on `holdout_corr` and the normal-theory leverage CI (level ∈
/// [0.5, 0.99] → p ∈ [0.75, 0.995]: the central and intermediate branches).
#[allow(unused_parens, clippy::unreadable_literal, clippy::excessive_precision)]
fn standard_normal_inv(p: f64) -> f64 {
    // Wichura AS241 (Applied Statistics 37:477-484), accurate to about 1e-16.
    // Coefficients as published (also Python's `statistics.NormalDist.inv_cdf`).
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
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
        let den = (((((((0.00000000105075007164441684324 * r + 0.0005475938084995344946) * r
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
        let den = (((((((0.00000000000000204426310338993978564 * r
            + 0.00000014215117583164458887)
            * r
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

    /// AS241 against `statistics.NormalDist().inv_cdf`, to 1e-12, in every
    /// branch: central (|p − 0.5| ≤ 0.425, reached by `level < 0.85`),
    /// intermediate (the default level's 0.975, and 0.995 for `level = 0.99`)
    /// and far tail (p < e⁻²⁵, one-sided since `1 − p` rounds to 1).
    #[test]
    fn standard_normal_inv_matches_as241() {
        check_standard_normal_inv(
            &[
                (0.5, 0.0),
                (0.75, 0.674_489_750_196_081_7),
                (0.9, 1.281_551_565_544_600_6),
                (0.975, 1.959_963_984_540_054),
                (0.995, 2.575_829_303_548_900_4),
            ],
            1e-12,
        );
        let far = standard_normal_inv(1e-50);
        assert!(
            (far + 14.933_337_534_788_489).abs() < 1e-12,
            "Φ⁻¹(1e-50) = {far}"
        );
    }

    /// `|Φ⁻¹(p) − q| < tol` and `Φ⁻¹(1 − p) = −Φ⁻¹(p)` to `tol` on every row.
    fn check_standard_normal_inv(rows: &[(f64, f64)], tol: f64) {
        for &(p, q) in rows {
            let got = standard_normal_inv(p);
            assert!((got - q).abs() < tol, "Φ⁻¹({p}) = {got}, want {q}");
            assert!(
                (standard_normal_inv(1.0 - p) + got).abs() < tol,
                "Φ⁻¹(1 − {p}) = {} is not −Φ⁻¹({p})",
                standard_normal_inv(1.0 - p)
            );
        }
    }

    #[test]
    fn reduce_holdout_corr_widens_with_overlap_factor() {
        // Generate B deterministic samples around 0.4 with mild dispersion.
        // The point is the plain mean of r_b; the z-scale SE carries the NB
        // inflation (1/B + (n−m)/m); the bounds are the Fisher Wald interval
        // back-transformed with tanh, strictly inside (−1, 1).
        let b_count = 1000;
        let mut samples = vec![0.0_f64; b_count];
        for (i, s) in samples.iter_mut().enumerate() {
            let phi = (i as f64).sin();
            *s = 0.4 + 0.05 * phi;
        }
        let n = 1000_usize;
        let m = 126_usize;
        let nb = reduce_holdout_corr(&samples, n, m, 0.95);

        #[allow(clippy::cast_precision_loss)]
        let mean_z: f64 = samples.iter().map(|r| r.atanh()).sum::<f64>() / b_count as f64;
        #[allow(clippy::cast_precision_loss)]
        let var_z: f64 = samples
            .iter()
            .map(|r| (r.atanh() - mean_z).powi(2))
            .sum::<f64>()
            / (b_count - 1) as f64;
        #[allow(clippy::cast_precision_loss)]
        let point = samples.iter().sum::<f64>() / b_count as f64;
        #[allow(clippy::cast_precision_loss)]
        let sd = ((1.0 / b_count as f64 + (n - m) as f64 / m as f64) * var_z).sqrt();
        let z = standard_normal_inv(0.975);
        assert!((nb.sd - sd).abs() < 1e-12, "sd {} vs {sd}", nb.sd);
        assert!(
            (nb.point - point).abs() < 1e-12,
            "point {} vs {point}",
            nb.point
        );
        let lower = (point.atanh() - z * sd).tanh();
        let upper = (point.atanh() + z * sd).tanh();
        assert!(
            (nb.lower - lower).abs() < 1e-12,
            "lower {} vs {lower}",
            nb.lower
        );
        assert!(
            (nb.upper - upper).abs() < 1e-12,
            "upper {} vs {upper}",
            nb.upper
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

    /// The β CI is built from the same corrected replicates as
    /// `beta_sign_z`: replicates rescaled by `1/κ̂` (so the shrinkage offset
    /// does not move the midpoint) and spread scaled by `√(m/(n − m))` (the
    /// finite-population correction). With symmetric `δ` it is therefore
    /// centered on `β_ref` for any common shrinkage `κ`, its half-width is
    /// `√(m/(n − m))` times the `δ` quantile, `beta_se` is the SE the z
    /// divides by, `beta_sign_z_signed = β_ref / beta_se` exactly, and the
    /// folded `beta_sign_z` is its absolute value. None of it grows with
    /// the number of replicates B: a `(2p̂ − 1)·√B` statistic
    /// would return √B here (every replicate shares the sign of `β_ref`), 14.1
    /// and 44.7.
    #[test]
    fn beta_ci_is_shrinkage_corrected_and_fpc_scaled() {
        let (n, m) = (100_usize, 26_usize);
        let beta_ref = [0.3_f64, -0.3];
        let beta_ref_col = Col::<f64>::from_fn(2, |j| beta_ref[j]);
        #[allow(clippy::cast_precision_loss)]
        let c = ((m as f64) / ((n - m) as f64)).sqrt();
        for &b in &[200_usize, 2000] {
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
                    let what = format!("b={b} kappa={kappa} j={j}");
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
                    assert!(
                        (ci.beta_sign_z[j] - ci.beta_sign_z_signed[j].abs()).abs() < 1e-12,
                        "{what}: folded"
                    );
                }
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
    /// and the skew of their distribution leave the interval unchanged; a reflected subsampling interval would
    /// move with both.
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

        // Strict threshold: a single failure already exceeds 0.0.
        let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..99).map(|_| Some(dummy_row(0.5))).collect();
        rows.push(None);
        match run(rows, opts_with(100, 0.0)).unwrap_err() {
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
        // No finite replicate: every CI collapses onto its reference
        // (leverage_ref 0.5, β_ref 1.0) with zero SE; the β z is NaN
        // (se = 0 with β_ref ≠ 0) and holdout_corr is all zeros.
        assert_eq!(ci.leverage_ci_lower, [0.5, 0.5]);
        assert_eq!(ci.leverage_ci_upper, [0.5, 0.5]);
        assert_eq!(ci.leverage_se, [0.0, 0.0]);
        assert_eq!(ci.beta_ci_lower, [1.0, 1.0]);
        assert_eq!(ci.beta_ci_upper, [1.0, 1.0]);
        assert_eq!(ci.beta_se, [0.0, 0.0]);
        assert!(ci.beta_sign_z_signed.iter().all(|z| z.is_nan()));
        let h = &ci.holdout_corr;
        assert_eq!((h.point, h.lower, h.upper, h.sd), (0.0, 0.0, 0.0, 0.0));
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
    /// Maximum tolerable combined per-resample failure rate
    /// (`n_holdout_corr_failed / n_boot`). Default `0.01`. Range `[0.0, 1.0]`.
    /// `0.0` is strict; `1.0` never fails on resample failures.
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
    let outcomes: Vec<WorkerOutcome> =
        crate::resample::parallel_for_each_seeded(rng, opts.n_boot, |_, child| {
            match run_one_confirmatory(x, y, k, m, w_ref, pre_std, weights, child) {
                std::result::Result::Ok(row) => WorkerOutcome::Ok(row),
                Err(PlsKitError::InvalidWeights {
                    reason: "insufficient_effective_n",
                }) => WorkerOutcome::Skipped,
                Err(_) => WorkerOutcome::Failed,
            }
        });

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
    use crate::test_support::synth;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    /// Every knob range `validate` enforces, at both edges: each rejected
    /// row names the knob in its message, each accepted edge passes.
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn validate_rejects_out_of_range_knobs() {
        let base = SubsampleOpts {
            n_boot: 100,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            max_failure_rate: 0.01,
            max_skip_rate: 0.01,
        };
        let with = |knobs: &[(&str, f64)]| {
            let mut o = base;
            for &(knob, v) in knobs {
                match knob {
                    "n_boot" => o.n_boot = v as usize,
                    "m_rate" => o.m_rate = v,
                    "level" => o.level = v,
                    "max_failure_rate" => o.max_failure_rate = v,
                    "max_skip_rate" => o.max_skip_rate = v,
                    _ => unreachable!("{knob}"),
                }
            }
            o
        };
        for (knob, v) in [
            ("m_rate", 0.4),
            ("m_rate", 0.5),
            ("m_rate", 0.95),
            ("level", 0.49),
            ("level", 0.991),
            ("n_boot", 99.0),
            ("max_failure_rate", -0.01),
            ("max_failure_rate", 1.01),
            ("max_skip_rate", -0.01),
            ("max_skip_rate", 1.01),
        ] {
            let err = with(&[(knob, v)]).validate().unwrap_err();
            assert_eq!(err.code(), "invalid_argument", "{knob}={v}");
            assert!(
                err.to_string().contains(&format!("{knob} must")),
                "{knob}={v}: {err}"
            );
        }
        for knobs in [
            &[][..],
            &[("level", 0.5)],
            &[("level", 0.99)],
            &[("max_failure_rate", 0.0), ("max_skip_rate", 0.0)],
            &[("max_failure_rate", 1.0), ("max_skip_rate", 1.0)],
        ] {
            assert!(with(knobs).validate().is_ok(), "{knobs:?}");
        }
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
    /// √(2/π) ≈ 0.80, sd √(1 − 2/π) ≈ 0.60 and about 5% above 1.96. A `(2p̂ − 1)·√B`
    /// statistic would average about 4.6 here, with most values above 1.96.
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
        // Half-normal spread too: sd √(1 − 2/π) ≈ 0.6028.
        let sd = (pooled.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (total - 1.0)).sqrt();
        assert!(
            (sd - (1.0 - 2.0 / std::f64::consts::PI).sqrt()).abs() <= 0.15,
            "sd |z| = {sd}"
        );
        // The β CI rests on the same corrected SE, so it covers the null
        // β[j] = 0 at roughly the nominal rate. Uncorrected (no FPC, no
        // 1/κ̂) it would be κ̂·√(1 − m/n) too narrow and
        // miss 0 far more often.
        #[allow(clippy::cast_precision_loss)]
        let miss = excluded as f64 / total;
        assert!(miss <= 0.10, "share of β CIs excluding 0 = {miss}");
    }

    /// The leverage CI covers the expected leverage of a size-`n` fit in
    /// the regime where a reflected subsampling interval fails: `D = 20`,
    /// `n = 100`, `K = 1`, where a fit on `m = 26` rows puts much less
    /// leverage on the signal variables than the full fit. The target
    /// `E[ĥ_n]` is the mean full-data leverage over fresh datasets. A
    /// reflected subsampling interval covers it about half the time here
    /// (0.51 over 1000 datasets); the bootstrap interval about 0.93.
    #[test]
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn leverage_ci_covers_expected_leverage_at_small_n_over_d() {
        let (n, d, snr) = (100_usize, 20_usize, 1.0_f64);
        let (n_oracle, reps) = (400_u64, 100_u64);
        let mut target = [0.0_f64; 2];
        for o in 0..n_oracle {
            let (x, y) = synth(n, d, 2, snr, 90_000 + o);
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
            let (x, y) = synth(n, d, 2, snr, 10_000 + rep);
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

    /// End-to-end β CI at K = 2 on the 2-signal / 6-noise design with
    /// unequal column scales (raw-scale true β = 8 and 4 on the signal
    /// coordinates, 0 elsewhere). Signal CIs exclude 0 on the positive
    /// side, noise CIs stay inside (−1, 1), signal midpoints sit ≥ 10×
    /// further from 0 than any noise midpoint, and the full-data β lies
    /// inside its CI on both signal coordinates and on at least d − 1
    /// coordinates overall. The bracketing catches
    /// replicates left on the standardized scale (no `y_scale / x_scale`
    /// back-projection): the two signal coordinates have equal standardized
    /// β but raw β 8 and 4, a gap the common factor κ̂ cannot absorb (with
    /// equal column scales it would).
    #[test]
    fn beta_ci_separates_signal_and_brackets_full_data_beta() {
        let (mut x, y) = synth(200, 8, 2, 4.0, 42);
        let scale = [0.5, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        for (j, s) in scale.iter().enumerate() {
            for i in 0..x.nrows() {
                x[(i, j)] *= s;
            }
        }
        let ci = run_engine(&x, &y, 2, 300, 2026);
        let beta_ref = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap()
        .beta;
        let (lo, hi) = (&ci.beta_ci_lower, &ci.beta_ci_upper);
        let mid = |j: usize| f64::midpoint(lo[j], hi[j]).abs();
        let noise_mid = (2..8).map(mid).fold(0.0_f64, f64::max);
        for j in 0..2 {
            assert!(lo[j] > 0.0, "signal j={j}: [{}, {}]", lo[j], hi[j]);
            assert!(
                mid(j) >= 10.0 * noise_mid,
                "signal j={j}: |mid| {} vs noise max {noise_mid}",
                mid(j)
            );
        }
        for j in 2..8 {
            assert!(
                lo[j] > -1.0 && hi[j] < 1.0,
                "noise j={j}: [{}, {}]",
                lo[j],
                hi[j]
            );
        }
        let inside = |j: usize| lo[j] <= beta_ref[j] && beta_ref[j] <= hi[j];
        for j in 0..2 {
            assert!(
                inside(j),
                "signal j={j}: β_ref {} outside [{}, {}]",
                beta_ref[j],
                lo[j],
                hi[j]
            );
        }
        let n_inside = (0..8).filter(|&j| inside(j)).count();
        assert!(
            n_inside >= 7,
            "β_ref inside its CI on {n_inside} of 8: {lo:?} {beta_ref:?} {hi:?}"
        );
    }

    /// The replicates read X only through row gathers into owned
    /// column-major copies, so every layout of X gives the owned matrix's
    /// CI bundle to the bit once the references are fixed. The references
    /// are fitted on the owned matrix: `pls1_fit` itself is only
    /// rounding-equal across layouts (CHANGELOG 0.6.1; owned by fit.rs).
    /// Strict rates make any replicate failure an `Err`, which fails the
    /// table, so it cannot pass on an empty bundle.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn confirmatory_ci_engine_is_layout_invariant() {
        crate::test_support::assert_families_layout_invariant(
            "confirmatory_ci",
            |c| {
                let k = 2;
                let (w_norm, _) = crate::fit::validate_and_normalize_weights(c.w, c.x.nrows(), k)?;
                let wr = w_norm.as_ref().map(Col::as_ref);
                let fit = pls1_fit(
                    c.x_owned.as_ref(),
                    c.y,
                    KSpec::Fixed(k),
                    wr,
                    FitOpts {
                        pre_standardized: c.pre,
                        ..FitOpts::default()
                    },
                )?;
                let lev = crate::linalg::leverage_diag(fit.w_star.as_ref());
                let (_, mut rng) = resolve_seed(Some(3))?;
                pls1_subsample_inference_confirmatory(
                    c.x,
                    c.y,
                    k,
                    fit.w_star.as_ref(),
                    fit.beta.as_ref(),
                    &lev,
                    SubsampleOpts {
                        n_boot: 100,
                        m_rate: 0.7,
                        level: 0.95,
                        pre_standardized: c.pre,
                        max_failure_rate: 0.0,
                        max_skip_rate: 0.0,
                    },
                    wr,
                    &mut rng,
                )
            },
            |ci: &ConfirmatoryCI| {
                let mut v = vec![
                    ci.n_boot as f64,
                    ci.m as f64,
                    ci.n_boot_finite as f64,
                    ci.n_boot_finite_holdout_corr as f64,
                ];
                for a in [
                    &ci.beta_sign_z,
                    &ci.beta_sign_z_signed,
                    &ci.leverage_ci_lower,
                    &ci.leverage_ci_upper,
                    &ci.leverage_se,
                    &ci.beta_ci_lower,
                    &ci.beta_ci_upper,
                    &ci.beta_se,
                ] {
                    v.extend_from_slice(a);
                }
                let h = &ci.holdout_corr;
                v.extend([h.point, h.lower, h.upper, h.sd]);
                v
            },
        );
    }
}
