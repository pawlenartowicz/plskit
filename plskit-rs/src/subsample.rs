//! Politis–Romano subsampling engine for PLS1 inference. Internal: callers
//! are `signal_test::pls1_confirmatory_test(CI=Some)` and
//! `rotation_stability::pls1_rotation_stability`.
//!
//! Reduction formulas are documented inline on `reduce_centered_scaled`
//! (centered-scaled CI; Politis–Romano 1999 Ch. 3) and `reduce_holdout_corr`
//! (Fisher z-transformed NB-Wald CI; variance via Nadeau–Bengio 2003 inflation
//! applied on the variance-stabilized atanh scale).

/// Centered-scaled subsampling CI for a scalar functional, plus its SD.
/// Marshalled to a frozen dataclass on the Python side; fields must not be reordered.
///
/// Scale convention: `point`, `lower`, `upper` are always reported on the natural
/// (user-facing) scale of the statistic. `sd` is on the inference scale used to
/// build the CI. For `reduce_centered_scaled` (used for leverage) the inference
/// scale is the natural scale, so `point ± Φ⁻¹(1−α/2) · sd` reconstructs the CI.
/// For `reduce_holdout_corr` the inference scale is Fisher z = atanh(r), so `sd`
/// is on the z-scale and the CI is asymmetric on the r-scale; do not reconstruct
/// it from `point ± sd`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CIScalar {
    /// Full-data point estimate `θ̂_n`, on the natural scale of the statistic.
    pub point: f64,
    /// Lower endpoint of the CI at `level`, on the natural scale.
    pub lower: f64,
    /// Upper endpoint of the CI at `level`, on the natural scale.
    pub upper: f64,
    /// Subsampling SE on the inference scale (natural scale for centered-scaled
    /// reductions; Fisher z-scale for `reduce_holdout_corr`). See struct doc.
    pub sd: f64,
}

/// Reduce a per-resample B-vector of `θ̂_m,b` values to a `CIScalar` using
/// centered-scaled subsampling (Politis–Romano 1999 Ch. 3):
///   Δ_b = θ̂_m,b − θ̂_n
///   lower = θ̂_n − sqrt(m/n) · quantile(Δ, 1 − α/2)
///   upper = θ̂_n − sqrt(m/n) · quantile(Δ, α/2)
///   sd    = sqrt(m/n) · stddev(Δ)
/// `level` is the CI level (e.g. 0.95 → α = 0.05).
///
/// NaN samples (produced by failed worker resamples) are filtered before
/// building the deltas vector; all denominators use the finite count
/// `b_finite`. If all samples are NaN, returns a degenerate CI at `point`.
#[allow(clippy::doc_markdown)]
pub(crate) fn reduce_centered_scaled(
    samples_b: &[f64],
    point: f64,
    n: usize,
    m: usize,
    level: f64,
) -> CIScalar {
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
    let scale = ((m as f64) / (n as f64)).sqrt();

    let q_lo = crate::linalg::empirical_quantile(&deltas, alpha / 2.0);
    let q_hi = crate::linalg::empirical_quantile(&deltas, 1.0 - alpha / 2.0);

    // lower = θ̂_n − √(m/n) · q_{1−α/2};  upper = θ̂_n − √(m/n) · q_{α/2}.
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
        let (_, mut rng) = resolve_seed(Some(7));
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
use crate::linalg::{col_row_subset, row_subset, standardize_apply};

/// Per-resample outputs for the confirmatory CI branch. Per-variable arrays
/// only — composite scalars have been removed.
///
/// Each row carries its own `beta_sign` vector (per-coordinate `+/−/0` tally
/// for the resample); the reducer aggregates these into per-coordinate counts.
/// Storing per-row keeps the row-disjoint invariant (per-resample writes are
/// independent — no shared counters across threads).
#[derive(Debug)]
pub(crate) struct ConfirmatoryWorkerRow {
    /// Per-variable subspace leverage on aligned `W_b`. Length D.
    pub leverage: Vec<f64>,
    /// `corr(X[holdout] · β_b, y[holdout])`. NaN if undefined (e.g. constant holdout y).
    pub holdout_corr: f64,
    /// Per-variable `+/−/0` tally for the resample, used by the sign-z reduction.
    /// Length D; entries are `1` for positive `β_b`[j], `-1` for negative, `0` for exact zero.
    pub beta_sign: Vec<i8>,
    /// Per-variable subsample regression coefficient `β_b`. Length D. PLS1-only:
    /// β is invariant under component sign flips and within-subspace rotations
    /// (the (Pᵀ W)⁻¹·Q product cancels both), so per-coordinate centered-scaled
    /// CIs are well-defined without procrustes alignment.
    pub beta: Vec<f64>,
}

/// Run one confirmatory subsample: draw indices, fit, align, compute readouts.
/// `w_ref` is the `(D, K)` reference weight matrix; leverage scores are
/// computed internally via `crate::linalg::leverage_diag`.
/// `pre_standardized` matches the user's flag.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::similar_names)]
fn run_one_confirmatory(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    m: usize,
    w_ref: MatRef<'_, f64>,
    pre_standardized_x: bool,
    weights: Option<ColRef<'_, f64>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<ConfirmatoryWorkerRow> {
    let n = x.nrows();
    let d = x.ncols();

    // 1. Index draw
    let (sample_idx, holdout_idx) = subsample_indices(n, m, rng);

    // 2. Fit on the subsample with the same standardization regime as the full-data fit.
    let x_sub_view = row_subset(x, &sample_idx);
    let y_sub_view = col_row_subset(y, &sample_idx);

    // Slice weights to the subsample and validate per-subsample effective n.
    let w_sub: Option<faer::Col<f64>> =
        weights.map(|w| crate::linalg::col_row_subset(w, &sample_idx));
    let (w_sub_norm, _, _) =
        crate::fit::validate_and_normalize_weights(w_sub.as_ref().map(faer::Col::as_ref), m, k)?;

    #[allow(clippy::cast_precision_loss)]
    let (xs_sub, x_mean, x_scale, ys_sub, y_scale) = if pre_standardized_x {
        // Caller asserts already standardized — pass through, but row_subset still
        // copies. Mean/scale are no-ops (zeros / ones) for the holdout standardization
        // and for the β back-projection below (β_b stays on the standardized scale,
        // matching β_ref which the caller's full-data fit also leaves on that scale).
        let xs = Mat::<f64>::from_fn(x_sub_view.nrows(), d, |i, j| x_sub_view[(i, j)]);
        let ys = Col::<f64>::from_fn(y_sub_view.nrows(), |i| y_sub_view[i]);
        (
            xs,
            Col::<f64>::zeros(d),
            Col::<f64>::from_fn(d, |_| 1.0),
            ys,
            1.0_f64,
        )
    } else {
        let (xs, mu, sigma) = crate::linalg::standardize_weighted(
            x_sub_view.as_ref(),
            w_sub_norm.as_ref().map(faer::Col::as_ref),
        );
        let (ys, _, ys_sigma) = crate::linalg::standardize1_weighted(
            y_sub_view.as_ref(),
            w_sub_norm.as_ref().map(faer::Col::as_ref),
        );
        (xs, mu, sigma, ys, ys_sigma)
    };

    let fit_b = pls1_fit(
        xs_sub.as_ref(),
        ys_sub.as_ref(),
        KSpec::Fixed(k),
        w_sub_norm.as_ref().map(faer::Col::as_ref),
        FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        },
    )?;

    let w_b = fit_b.w_star;
    let beta_b = fit_b.beta;

    // 3. Procrustes alignment
    let r_b = procrustes::orthogonal(w_b.as_ref(), w_ref, false)
        .expect("procrustes invariants pre-validated by plskit")
        .rotation;
    let mut w_b_aligned = Mat::<f64>::zeros(d, k);
    faer::linalg::matmul::matmul(
        w_b_aligned.as_mut(),
        faer::Accum::Replace,
        w_b.as_ref(),
        r_b.as_ref(),
        1.0,
        faer::Par::Seq,
    );

    // 5. Per-variable leverage
    let leverage = crate::linalg::leverage_diag(w_b_aligned.as_ref());

    // 6. beta-sign tally
    let mut beta_sign = vec![0_i8; d];
    for i in 0..d {
        beta_sign[i] = if beta_b[i] > 0.0 {
            1
        } else if beta_b[i] < 0.0 {
            -1
        } else {
            0
        };
    }

    // 7. Holdout predictive correlation
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

    // Back-project β_b to the same scale as β_ref so deltas are meaningful.
    // Mirrors `pls1_fit`'s full-data back-projection (`fit.rs:143-147`):
    //   β_raw[j] = β_std[j] * y_scale / x_scale[j]
    // For pre_standardized_x, both scales are 1.0 → no-op (β_b stays standardized,
    // matching β_ref which the caller's full-data fit also leaves on that scale).
    let beta_vec: Vec<f64> = (0..d).map(|j| beta_b[j] * y_scale / x_scale[j]).collect();

    Ok(ConfirmatoryWorkerRow {
        leverage,
        holdout_corr,
        beta_sign,
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

        let (_, mut rng) = resolve_seed(Some(2));
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
        assert_eq!(row.beta_sign.len(), 6);
    }
}

/// Output of the confirmatory subsampling engine. Marshalled into
/// `ConfirmatoryCI` at the wrapper layer.
#[derive(Debug, Clone)]
pub struct ConfirmatoryCI {
    /// Number of bootstrap resamples used.
    pub n_boot: usize,
    /// Subsample size used for each resample.
    pub m: usize,
    /// Subsample rate `m_rate` such that `m = ceil(n^m_rate)`.
    pub m_rate: f64,
    /// Nominal CI level (e.g. 0.95).
    pub level: f64,
    /// Per-variable sign-stability z (folded). Length D. Under H0: P(sign-match)=0.5.
    pub beta_sign_z: Vec<f64>,
    /// Per-variable signed sign-stability z = `sign(β_ref[j]) · |beta_sign_z[j]|`. Length D.
    /// Use for descriptive directional-stability maps (TFCE-style displays). Folded form is canonical for hypothesis testing.
    pub beta_sign_z_signed: Vec<f64>,
    /// Per-variable leverage centered-scaled CI lower bound. Length D.
    pub leverage_ci_lower: Vec<f64>,
    /// Per-variable leverage centered-scaled CI upper bound. Length D.
    pub leverage_ci_upper: Vec<f64>,
    /// Per-variable leverage subsampling SE. Length D.
    pub leverage_se: Vec<f64>,
    /// Per-coordinate β centered-scaled CI lower bound. Length D. PLS1-only;
    /// the centered-scaled formula treats shrinkage bias and sampling variance
    /// as sharing a √(n/m) rate — well-calibrated in the easy regime, a
    /// directional sanity check elsewhere. See `RESULTS_FORMAT.md` for caveats.
    pub beta_ci_lower: Vec<f64>,
    /// Per-coordinate β centered-scaled CI upper bound. Length D. See `beta_ci_lower`.
    pub beta_ci_upper: Vec<f64>,
    /// Per-coordinate β subsampling SE = √(m/n) · `sd(β_b[j])`. Length D.
    pub beta_se: Vec<f64>,
    /// Holdout predictive correlation. CI is built via Fisher z-transform
    /// (atanh) with NB inflation applied on the z-scale, then back-transformed
    /// via tanh; CI bounds are guaranteed in (−1, 1) and are asymmetric on the
    /// r-scale. `point` is the subsample mean on the r-scale; `sd` is on the
    /// z-scale (see `CIScalar` doc).
    pub holdout_corr: CIScalar,
    /// Number of resamples whose worker fit succeeded (contributors to
    /// leverage, `beta_sign`). Equals `n_boot − n_worker_failed`.
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
/// only for NB-Wald CI on `holdout_corr` (level ∈ [0.5, 0.99] → no extreme tails).
/// The NB-adjusted Wald form requires Φ⁻¹ once.
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

/// Reduce per-resample worker rows into the final `ConfirmatoryCI` payload.
/// Computes per-variable leverage CIs and per-variable sign-z statistics by
/// tallying `+/−/0` signs from each row's `beta_sign` against `beta_ref`.
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
        let r = reduce_centered_scaled(&col_buf, leverage_ref[j], n, m, level);
        leverage_ci_lower[j] = r.lower;
        leverage_ci_upper[j] = r.upper;
        leverage_se[j] = r.sd;
    }

    // ── per-coordinate β CI (PLS1 only — β is rotation/sign invariant) ──
    let mut beta_ci_lower = vec![0.0_f64; d];
    let mut beta_ci_upper = vec![0.0_f64; d];
    let mut beta_se = vec![0.0_f64; d];
    for j in 0..d {
        for i in 0..b {
            col_buf[i] = rows[i].beta[j];
        }
        let r = reduce_centered_scaled(&col_buf, beta_ref[j], n, m, level);
        beta_ci_lower[j] = r.lower;
        beta_ci_upper[j] = r.upper;
        beta_se[j] = r.sd;
    }

    // ── per-variable sign-z ──
    let mut pos = vec![0_usize; d];
    let mut zero = vec![0_usize; d];
    for row in rows {
        for j in 0..d {
            match row.beta_sign[j] {
                1 => pos[j] += 1,
                0 => zero[j] += 1,
                _ => {}
            }
        }
    }
    let mut beta_sign_z = vec![0.0_f64; d];
    let mut beta_sign_z_signed = vec![0.0_f64; d];
    for j in 0..d {
        let neg = b.saturating_sub(pos[j] + zero[j]);
        let s_ref = beta_ref[j];
        #[allow(clippy::cast_precision_loss)]
        let m_count: f64 = if s_ref > 0.0 {
            pos[j] as f64
        } else if s_ref < 0.0 {
            neg as f64
        } else {
            pos[j].max(neg) as f64
        };
        #[allow(clippy::cast_precision_loss)]
        let p_hat = m_count / b as f64;
        #[allow(clippy::cast_precision_loss)]
        let folded = (2.0 * p_hat - 1.0) * (b as f64).sqrt();
        beta_sign_z[j] = folded;
        beta_sign_z_signed[j] = if s_ref > 0.0 {
            folded.abs()
        } else if s_ref < 0.0 {
            -folded.abs()
        } else {
            folded
        };
    }

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
        beta_sign_z_signed,
        leverage_ci_lower,
        leverage_ci_upper,
        leverage_se,
        beta_ci_lower,
        beta_ci_upper,
        beta_se,
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
mod tests_failure_check {
    use super::*;
    use faer::Col;

    fn dummy_row(holdout_corr: f64) -> ConfirmatoryWorkerRow {
        ConfirmatoryWorkerRow {
            leverage: vec![0.5_f64; 2],
            holdout_corr,
            beta_sign: vec![1_i8, 1_i8],
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
        let r = reduce_centered_scaled(&samples, 3.0, 1000, 126, 0.95);
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
        let r = reduce_centered_scaled(&samples, 0.0, 1000, 100, 0.95);
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
    /// Spec §6.3 threshold for weight-validation skips. Default `0.01`. Range `[0.0, 1.0]`.
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
        let (_, mut rng) = resolve_seed(Some(2026));
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
        let (_, mut rng) = resolve_seed(Some(2026));
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
        let (_, mut rng) = resolve_seed(Some(2026));
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
}
