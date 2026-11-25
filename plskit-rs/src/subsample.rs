//! Resampling engine for PLS1 inference: Politis–Romano subsampling for β,
//! half splits for `holdout_corr`, and an n-out-of-n bootstrap for leverage. The engine
//! entry points are internal (`pub(crate)`): the caller is
//! `signal_test::pls1_confirmatory_test(CI=Some)`. Only the result types
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
    /// Point estimate on the natural scale of the statistic: the full-data
    /// `θ̂_n`, or for `holdout_corr` the mean over the half splits.
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
    // Formed on the deltas divided by a power of two near the largest
    // (`linalg::pow2_scale`), so the squares stay finite and normal for a β
    // at any magnitude (β of X·2^±600 under `pre_standardized` is about
    // 2^∓600); the division is exact, so no bit changes elsewhere.
    let (s, inv) = crate::linalg::pow2_scale(deltas.iter().fold(0.0_f64, |m, d| m.max(d.abs())));
    let mean: f64 = deltas.iter().map(|d| d * inv).sum::<f64>() / b_finite as f64;
    let var: f64 = if b_finite > 1 {
        deltas.iter().map(|d| (d * inv - mean).powi(2)).sum::<f64>() / (b_finite - 1) as f64
    } else {
        0.0
    };
    let sd = scale * (s * var.sqrt());

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

/// The confirmatory CI's subsample size on `n` rows, or `InvalidArgument`
/// when it cannot serve a `k`-component CI: `m ≥ n` leaves no rows outside
/// the subsample, so the β rate `√(m/(n − m))` is undefined, and
/// `m < k + 1 + rank` (or a `holdout_train_size(n) < k + 1 + rank`) is too
/// small to fit `k` components and leave residual degrees of freedom.
/// `rank` is that of `[1, Z]` with covariates, 1 without.
pub(crate) fn resolve_ci_m(n: usize, k: usize, rank: usize, m_rate: f64) -> PlsKitResult<usize> {
    let m = resolve_m(n, m_rate);
    if m >= n {
        return Err(PlsKitError::InvalidArgument(format!(
            "resolved m = {m} (from n={n}, m_rate={m_rate}) leaves no rows outside the subsample; need m < n"
        )));
    }
    let (d, cov) = (
        rank + 1,
        if rank > 1 {
            format!(" and {} covariates", rank - 1)
        } else {
            String::new()
        },
    );
    if m < k + d {
        return Err(PlsKitError::InvalidArgument(format!(
            "resolved m = {m} (from n={n}, m_rate={m_rate}) is too small for k={k}{cov}; need m ≥ k+{d}"
        )));
    }
    let m_h = holdout_train_size(n);
    if m_h < k + d {
        return Err(PlsKitError::InvalidArgument(format!(
            "the holdout_corr split trains on n/2 = {m_h} rows (n={n}), too few for k={k}{cov}; need n/2 ≥ k+{d}"
        )));
    }
    Ok(m)
}

/// Training rows of each replicate's `holdout_corr` split: `⌊n/2⌋`, the
/// training size of the `split_exact` / `split_nb` statistic, so the
/// interval and the test target the same held-out correlation.
pub(crate) fn holdout_train_size(n: usize) -> usize {
    n / 2
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

/// Draw a subsample of size `m` without replacement from `0..n`: one
/// shuffle of `0..n` on `rng`, the first `m` entries the sample. Returns
/// `(sample_idx, holdout_idx)` partitioning `0..n`.
pub(crate) fn subsample_indices(
    n: usize,
    m: usize,
    rng: &mut CommonStatsRng,
) -> (Vec<usize>, Vec<usize>) {
    let mut perm: Vec<usize> = (0..n).collect();
    crate::rng::shuffle(rng, &mut perm);
    let holdout = perm.split_off(m);
    (perm, holdout)
}

/// The index draws of CI replicate `r` at run seed `seed`:
/// `(sample, boot, (half, holdout))`. On stream `(seed, r)` of `TAG_SUBSM`
/// the size-`m` subsample, then the half split (`holdout_train_size(n)`
/// training rows); on stream `(seed, r)` of `TAG_BOOTS` the `n` bootstrap
/// draws.
#[allow(clippy::type_complexity)]
pub(crate) fn replicate_indices(
    n: usize,
    m: usize,
    seed: u64,
    r: usize,
) -> (Vec<usize>, Vec<usize>, (Vec<usize>, Vec<usize>)) {
    let mut subsm = CommonStatsRng::new_tagged(seed, r as u64, crate::rng::TAG_SUBSM);
    let (sample, _) = subsample_indices(n, m, &mut subsm);
    let mut boots = CommonStatsRng::new_tagged(seed, r as u64, crate::rng::TAG_BOOTS);
    let boot = bootstrap_indices(n, &mut boots);
    let half = subsample_indices(n, holdout_train_size(n), &mut subsm);
    (sample, boot, half)
}

#[cfg(test)]
mod tests_indices {
    use super::*;

    #[test]
    fn resolve_m_known_values() {
        // Known values (m_rate=0.7): n=100 → m=26; n=1000 → m=126; n=10_000 → m=631.
        assert_eq!(resolve_m(100, 0.7), 26);
        assert_eq!(resolve_m(1000, 0.7), 126);
        assert_eq!(resolve_m(10_000, 0.7), 631);
    }

    #[test]
    fn subsample_indices_partitions_range() {
        let mut rng = CommonStatsRng::new_tagged(7, 0, crate::rng::TAG_SUBSM);
        let (s, h) = subsample_indices(100, 26, &mut rng);
        assert_eq!(s.len(), 26);
        assert_eq!(h.len(), 74);
        let mut all: Vec<usize> = s.iter().chain(h.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..100).collect::<Vec<_>>());
    }

    /// Replicate `r`'s draws, rebuilt by hand from their streams: the
    /// subsample shuffle, the half-split shuffle continuing the same
    /// `TAG_SUBSM` stream, and `n` bounded draws on `TAG_BOOTS`.
    #[test]
    fn replicate_draws_follow_their_streams() {
        use crate::rng::{TAG_BOOTS, TAG_SUBSM, shuffle};
        let mut subsm = CommonStatsRng::new_tagged(7, 3, TAG_SUBSM);
        let mut first: Vec<usize> = (0..40).collect();
        shuffle(&mut subsm, &mut first);
        let mut boots = CommonStatsRng::new_tagged(7, 3, TAG_BOOTS);
        let boot: Vec<usize> = (0..40).map(|_| boots.bounded(40) as usize).collect();
        let mut second: Vec<usize> = (0..40).collect();
        shuffle(&mut subsm, &mut second);
        let h = holdout_train_size(40);
        let want = (
            first[..13].to_vec(),
            boot,
            (second[..h].to_vec(), second[h..].to_vec()),
        );
        assert_eq!(replicate_indices(40, 13, 7, 3), want);
    }
}

use commonstats::CommonStatsRng;
use faer::{Col, ColRef, Mat, MatRef};

use crate::covariates::CovariateBasis;
use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::{FitOpts, InternalFitOpts, pls1_fit_impl};
use crate::linalg::{center_rows_fro, col_row_subset, row_subset, standardize_apply_rows};

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
    /// `corr(X[holdout] · β_h, y[holdout])`, `β_h` fitted on the half split's
    /// training rows. NaN if undefined (e.g. constant holdout y).
    pub holdout_corr: f64,
    /// The half-split fit failed on the weights (`InvalidWeights` with
    /// reason `insufficient_effective_n` or `all_zero`), so `holdout_corr`
    /// is NaN and counts as a skip under `max_skip_rate`, not as a failure.
    pub holdout_skipped: bool,
    /// Per-variable subsample regression coefficient `β_b`. Length D. PLS1-only:
    /// β is invariant under component sign flips and within-subspace rotations
    /// (the (Pᵀ W)⁻¹·Q product cancels both), so per-coordinate centered-scaled
    /// CIs are well-defined without procrustes alignment.
    pub beta: Vec<f64>,
}

/// Draw an n-out-of-n bootstrap resample: `n` indices from `0..n` with
/// replacement, `n` draws of `rng.bounded(n)`.
pub(crate) fn bootstrap_indices(n: usize, rng: &mut CommonStatsRng) -> Vec<usize> {
    let bound = u32::try_from(n).expect("resolve_seed's row check keeps n <= u32::MAX");
    (0..n).map(|_| rng.bounded(bound) as usize).collect()
}

/// A fit on a row selection of `(x, y)`, with the rows' (weighted) means
/// of X and y it was centred with.
struct RowFit {
    w: Mat<f64>,
    beta: Col<f64>,
    x_mean: Col<f64>,
    y_mean: f64,
    /// With covariates: the basis of the rows' `[1, Z]` and the WLS
    /// coefficients of the scaled X `(rank, d)` and y `(rank, 1)` on it.
    cov: Option<(CovariateBasis, Mat<f64>, Mat<f64>)>,
}

/// Fit `k` components on the rows `idx` of `(x, y)` (repeats allowed).
/// With `scales = Some((x_scale, y_scale))` the rows are read as the
/// held-out tests read X: centred with the rows' own (weighted) means (when
/// `centre`; else as given, with zero means) and divided by the all-row
/// scales (`y_scale` one entry), the caller's `linalg::held_out_scale` of X
/// and of y. A fit whose coefficients score other rows needs these: a
/// row-subset scale would blow up the other rows wherever a column barely
/// varies on these. With `scales = None` the rows are standardized on their
/// own (weighted) moments, as the full-data fit standardizes all rows
/// (`linalg::standardize_rows`, which gives a column with no spread on
/// these rows the standardizer's fallback scale), and `centre` is not read.
/// Weights are sliced to `idx` and re-validated, so an insufficient
/// effective sample size surfaces as `InvalidWeights`.
///
/// With covariates `z`, the scaled rows of X and y are replaced by
/// their WLS residuals on the rows' own `[1, Z]` (weighted with the rows'
/// normalized weights) before the fit, as the full-data fit does on all
/// rows, a column `[1, Z]` explains set to 0; too few residual degrees of
/// freedom is `InsufficientResidualDf`.
///
/// A NIPALS short-circuit (`k_used < k`) is an error, so the caller's
/// driver counts the resample as failed instead of reducing a truncated
/// weight matrix against length-`k` references.
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn fit_rows(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    idx: &[usize],
    k: usize,
    scales: Option<(ColRef<'_, f64>, ColRef<'_, f64>)>,
    weights: Option<ColRef<'_, f64>>,
    z: Option<MatRef<'_, f64>>,
    centre: bool,
) -> PlsKitResult<RowFit> {
    let w_sub: Option<faer::Col<f64>> = weights.map(|w| crate::linalg::col_row_subset(w, idx));
    let (w_sub_norm, _) = crate::fit::validate_and_normalize_weights(
        w_sub.as_ref().map(faer::Col::as_ref),
        idx.len(),
    )?;
    let wsr = w_sub_norm.as_ref().map(faer::Col::as_ref);

    let rows = |a: MatRef<'_, f64>, scale: Option<ColRef<'_, f64>>| match scale {
        None => crate::linalg::standardize_rows(a, idx, wsr),
        Some(scale) if centre => {
            let (s, mean, _) = center_rows_fro(a, idx, wsr, None, scale, faer::Par::Seq);
            (s, mean)
        }
        Some(scale) => {
            let mean = Col::<f64>::zeros(a.ncols());
            (
                standardize_apply_rows(a, idx, mean.as_ref(), scale, None),
                mean,
            )
        }
    };
    let (mut xs_sub, x_mean) = rows(x, scales.map(|s| s.0));
    let (ys_sub, y_mean) = rows(y.as_mat(), scales.map(|s| s.1));
    let (mut ys_sub, y_mean) = (ys_sub.col(0).to_owned(), y_mean[0]);
    let cov = match z {
        None => None,
        Some(z) => {
            let z_rows = row_subset(z, idx);
            let basis =
                CovariateBasis::new(z_rows.as_ref(), w_sub_norm.as_ref().map(faer::Col::as_ref))?;
            basis.check_df(k)?;
            // The coefficients serve the held-out rows; the rows they came
            // from take `residualize`, whose aliasing rule zeroes a column
            // `[1, Z]` explains, as the full-data fit does.
            let bx = basis.coef(xs_sub.as_ref());
            let by = basis.coef(ys_sub.as_ref().as_mat());
            basis.residualize(xs_sub.as_mut());
            basis.residualize(ys_sub.as_mut().as_mat_mut());
            Some((basis, bx, by))
        }
    };

    let fit = pls1_fit_impl(
        xs_sub.as_ref(),
        ys_sub.as_ref(),
        k,
        w_sub_norm.as_ref().map(faer::Col::as_ref),
        None,
        FitOpts {
            pre_standardized: true,
            // Auto: the fit picks its arm from its shape (`fit::resolve_par`),
            // so the bits do not depend on the batch or the pool.
            par: crate::fit::ParChoice::Auto,
        },
        InternalFitOpts::default(),
    )?;
    Ok(RowFit {
        w: fit.w_star,
        beta: fit.beta,
        x_mean,
        y_mean,
        cov,
    })
}

/// Run one confirmatory replicate: a size-`m` subsample (without
/// replacement) for β, an n-out-of-n bootstrap resample (with replacement)
/// for leverage, and a half split (`holdout_train_size(n)` training rows,
/// the rest held out) for `holdout_corr`. The index sets are
/// [`replicate_indices`]`(n, m, seed, r)`, replicate `r`'s own streams, so
/// they do not depend on which replicates ran before. The β fit and the
/// half split read X and y on the all-row scales `x_scale` and `y_scale`
/// (`fit_rows`); the leverage fit standardizes its resample on the
/// resample's own (weighted) moments, as the full-data reference fit
/// standardizes all rows. Under `pre_standardized` the β and leverage fits
/// take the rows as given, as the reference fit does, and the half split
/// still centres, as the split tests do.
///
/// Leverage uses the bootstrap because its bias depends on the sample size
/// (at small `n/D` a fit spreads leverage from signal onto noise
/// variables, more so on fewer rows): a size-`m` fit measures the sampling
/// distribution at the wrong size, while a size-`n` resample reproduces the
/// spread of the full-data leverage. See `reduce_leverage`.
///
/// `holdout_corr` uses a half split, not the size-`m` subsample, because
/// its target is the held-out correlation of a fit trained on that many
/// rows: at `m = ⌊n/2⌋` it is close to what the split tests estimate (X
/// scaled once over all rows, each half centred with the training means),
/// and the NB factor `(n − m)/m` is about 1 (at `m = n^0.7` it is 2.8 at
/// `n = 100`, and the interval nearly always covers). The factor assumes two
/// splits' correlations correlate at `(n − m)/n`, about 0.5. That holds when
/// the noise of the held-out rows dominates the split-to-split spread, and
/// the interval then covers a little above its level (about 1.1 to 1.2
/// times the needed width in the grid). When the noise of the training
/// fit dominates (typically at high signal), two splits correlate nearer
/// 0.25, and the interval is conservative: in the Monte Carlo coverage grid
/// it is up to about twice as wide as needed.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
fn run_one_confirmatory(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    m: usize,
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    z: Option<MatRef<'_, f64>>,
    pre_standardized: bool,
    seed: u64,
    r: usize,
) -> PlsKitResult<ConfirmatoryWorkerRow> {
    let n = x.nrows();
    let d = x.ncols();

    // 1. Index draws: subsample, bootstrap resample, then the half split.
    let (sample_idx, boot_idx, (half_idx, holdout_idx)) = replicate_indices(n, m, seed, r);

    // 2. Subsample fit (β) and bootstrap fit (leverage).
    let centre = !pre_standardized;
    let all_row = Some((x_scale, y_scale));
    let beta_b = fit_rows(x, y, &sample_idx, k, all_row, weights, z, centre)?.beta;
    // The leverage fit scores no other rows, so it standardizes the
    // resample on its own moments, as the reference fit standardizes all
    // rows. On the all-row scale a column's weight moves with its resample
    // variance over the fixed SD, not with its resample SD, which at high
    // signal roughly doubles the spread of a signal column's leverage.
    let boot_scales = pre_standardized.then_some((x_scale, y_scale));
    let boot = fit_rows(x, y, &boot_idx, k, boot_scales, weights, z, centre)?;

    // 3. Per-variable leverage on the bootstrap fit's unaligned W. Leverage
    // is invariant under right-orthogonal rotation (leverage =
    // diag(W(WᵀW)⁻¹Wᵀ); W → W·R leaves the hat matrix fixed), so no
    // procrustes alignment is needed.
    let leverage = crate::linalg::leverage_diag(boot.w.as_ref());
    drop(boot);

    // 4. Holdout predictive correlation of the half-split fit. A failed
    // half-split fit costs the replicate its holdout_corr only (NaN), not its
    // β and leverage rows; one that fails on the weights is a skip.
    // The holdout rows are gathered, centred with the training half's means
    // and divided by the all-row scale in one pass, y too (the r floors
    // below are absolute). With covariates, both
    // are then the residuals on the holdout rows' `[1, Z]` from the training
    // half's coefficients.
    #[allow(clippy::cast_precision_loss)]
    let half_fit = fit_rows(x, y, &half_idx, k, all_row, weights, z, true);
    let holdout_skipped = matches!(
        half_fit,
        Err(PlsKitError::InvalidWeights {
            reason: "insufficient_effective_n" | "all_zero",
        })
    );
    let holdout_corr = match half_fit {
        Err(_) => f64::NAN,
        Ok(RowFit {
            beta: beta_h,
            x_mean,
            y_mean: y_mean_h,
            cov,
            ..
        }) => {
            let mut ys_h_owned = col_row_subset(y, &holdout_idx);
            let mut xs_h = standardize_apply_rows(x, &holdout_idx, x_mean.as_ref(), x_scale, None);
            for v in ys_h_owned.iter_mut() {
                *v = (*v - y_mean_h) / y_scale[0];
            }
            if let (Some(z), Some((basis, bx, by))) = (z, cov) {
                let z_h = row_subset(z, &holdout_idx);
                basis.residualize_new(z_h.as_ref(), xs_h.as_mut(), bx.as_ref());
                basis.residualize_new(z_h.as_ref(), ys_h_owned.as_mut().as_mat_mut(), by.as_ref());
            }

            // Seq inside the replicate worker: outer Rayon owns the threadpool.
            let y_pred = crate::linalg::mat_vec(xs_h.as_ref(), beta_h.as_ref(), faer::Par::Seq);
            let n_h = ys_h_owned.nrows();
            if n_h >= 2 {
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
            }
        }
    };

    // 5. Back-project β_b to the same scale as β_ref so deltas are meaningful.
    // Mirrors `pls1_fit`'s full-data back-projection, with the same all-row
    // scales:
    //   β_raw[j] = β_std[j] * y_scale / x_scale[j]
    // Under pre_standardized the scales are 1 (a power of two for a block
    // outside [2^-20, 2^20]), so β_b is on X as given, as β_ref is.
    let beta_vec: Vec<f64> = (0..d)
        .map(|j| beta_b[j] * y_scale[0] / x_scale[j])
        .collect();

    Ok(ConfirmatoryWorkerRow {
        leverage,
        holdout_corr,
        holdout_skipped,
        beta: beta_vec,
    })
}

#[cfg(test)]
mod tests_worker {
    use super::*;
    use crate::linalg::held_out_scale;
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
    /// re-standardization. y's scale is passed as ones, so y stays at
    /// `1e-20` (`held_out_scale` would divide it by a power of two): NIPALS
    /// sees `‖X'y‖ ≪ 1e-14` at component 0 and short-circuits with no
    /// component although `k = 2`. The test is about the error reaching the
    /// caller, not about the scale.
    #[test]
    fn worker_propagates_err_on_truncated_w_b() {
        let n = 40;
        let d = 4;
        let k = 2;
        let mut rng_data = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng_data.random_range(-1.0..1.0));
        let y_tiny = Col::<f64>::from_fn(n, |_| rng_data.random_range(-1e-20..1e-20));
        let res = run_one_confirmatory(
            x.as_ref(),
            y_tiny.as_ref(),
            k,
            resolve_m(n, 0.7),
            held_out_scale(x.as_ref(), None, true).as_ref(),
            Col::<f64>::from_fn(1, |_| 1.0).as_ref(),
            None,
            None,
            true,
            11,
            0,
        );
        assert!(
            matches!(&res, Err(PlsKitError::InvalidInput(msg)) if msg.contains("truncated")),
            "{res:?}"
        );
    }

    /// A failed half-split fit costs the replicate its `holdout_corr` only.
    /// `k = 55` is above the rank of a 50-row half split but not of the
    /// 99-row subsample or of the bootstrap resample (about 63 distinct rows).
    #[test]
    #[allow(clippy::many_single_char_names)] // n, d, k, x, y are the math names
    fn failed_half_split_keeps_beta_and_leverage() {
        let (n, d, k) = (100, 80, 55);
        let mut rng_data = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng_data.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |_| rng_data.random_range(-1.0..1.0));
        let (xs, ys) = (
            held_out_scale(x.as_ref(), None, false),
            held_out_scale(y.as_mat(), None, false),
        );
        let row = run_one_confirmatory(
            x.as_ref(),
            y.as_ref(),
            k,
            99,
            xs.as_ref(),
            ys.as_ref(),
            None,
            None,
            false,
            11,
            0,
        )
        .unwrap();
        assert!(row.holdout_corr.is_nan());
        assert!(row.beta.iter().chain(&row.leverage).all(|v| v.is_finite()));
    }

    /// With covariates, a replicate fit is the plain fit on its rows
    /// centred with their own (weighted) means, divided by the all-row
    /// scales, and then residualized on the rows' own `[1, Z]` with their
    /// normalized weights.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn ci_with_covariates_adjusts_each_subsample() {
        use crate::test_support::{assert_close_rel, synth, test_weights, unif};
        let (n, d, k) = (60, 8, 2);
        let (x, y) = synth(n, d, 3, 1.0, 71);
        let x = Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)] + 3.0);
        let z = unif(n, 2, 72);
        let idx: Vec<usize> = (0..n).filter(|i| i % 3 != 1).collect();
        for w in [None, Some(test_weights(n))] {
            let wr = w.as_ref().map(Col::as_ref);
            let (xsc, ysc) = (
                held_out_scale(x.as_ref(), wr, false),
                held_out_scale(y.as_mat(), wr, false),
            );
            let (xsc, ysc) = (xsc.as_ref(), ysc.as_ref());
            let got = fit_rows(
                x.as_ref(),
                y.as_ref(),
                &idx,
                k,
                Some((xsc, ysc)),
                wr,
                Some(z.as_ref()),
                true,
            )
            .unwrap();
            let w_sub = wr.map(|w| col_row_subset(w, &idx));
            let (w_sub, _) = crate::fit::validate_and_normalize_weights(
                w_sub.as_ref().map(Col::as_ref),
                idx.len(),
            )
            .unwrap();
            let wsr = w_sub.as_ref().map(Col::as_ref);
            let basis = CovariateBasis::new(row_subset(z.as_ref(), &idx).as_ref(), wsr).unwrap();
            let (mut xs, _, _) = center_rows_fro(x.as_ref(), &idx, wsr, None, xsc, faer::Par::Seq);
            let (mut ys, _, _) = center_rows_fro(y.as_mat(), &idx, wsr, None, ysc, faer::Par::Seq);
            basis.residualize(xs.as_mut());
            basis.residualize(ys.as_mut());
            let want = pls1_fit_impl(
                xs.as_ref(),
                ys.col(0),
                k,
                wsr,
                None,
                FitOpts {
                    pre_standardized: true,
                    ..FitOpts::default()
                },
                InternalFitOpts::default(),
            )
            .unwrap();
            assert_close_rel(got.beta.as_mat(), want.beta.as_mat(), 1e-12, "beta");
        }
    }

    /// A y that `[1, Z]` explains gives no component on a subsample, as on
    /// all rows: its residual is set to 0, not fitted as rounding noise, and
    /// the replicate fails (`truncated`).
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn a_y_the_covariates_explain_fits_no_component_on_a_subsample() {
        use crate::test_support::{synth, unif};
        let n = 60;
        let (x, _) = synth(n, 8, 3, 1.0, 71);
        let z = unif(n, 2, 72);
        let y = Col::<f64>::from_fn(n, |i| 1.0 + 2.0 * z[(i, 0)] - z[(i, 1)]);
        let idx: Vec<usize> = (0..n).filter(|i| i % 3 != 1).collect();
        let r = fit_rows(
            x.as_ref(),
            y.as_ref(),
            &idx,
            1,
            Some((
                held_out_scale(x.as_ref(), None, false).as_ref(),
                held_out_scale(y.as_mat(), None, false).as_ref(),
            )),
            None,
            Some(z.as_ref()),
            true,
        )
        .map(|f| f.beta);
        assert!(
            matches!(&r, Err(PlsKitError::InvalidInput(m)) if m.contains("truncated")),
            "{r:?}"
        );
    }

    /// Under `pre_standardized` the β and leverage fits of a replicate take
    /// X and y as given, as the full-data reference fit does: on X + 7 they
    /// are the as-given fits of the drawn rows. The half split centres, so
    /// `holdout_corr` is the one on X.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pre_standardized_replicate_fits_beta_and_leverage_as_given() {
        use crate::test_support::{assert_close_rel, synth, test_weights};
        let (n, d, k, m) = (60, 8, 2, 40);
        let (x, y) = synth(n, d, 3, 1.0, 71);
        let x7 = Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)] + 7.0);
        let w = test_weights(n);
        let run = |x: &Mat<f64>| {
            run_one_confirmatory(
                x.as_ref(),
                y.as_ref(),
                k,
                m,
                held_out_scale(x.as_ref(), None, true).as_ref(),
                held_out_scale(y.as_mat(), None, true).as_ref(),
                Some(w.as_ref()),
                None,
                true,
                11,
                0,
            )
            .unwrap()
        };
        let (plain, shifted) = (run(&x), run(&x7));
        let (hp, hs) = (plain.holdout_corr, shifted.holdout_corr);
        assert!(
            (hp - hs).abs() <= 1e-12,
            "holdout_corr {hp} on X, {hs} on X + 7"
        );

        let (sample_idx, boot_idx, _) = replicate_indices(n, m, 11, 0);
        let fit = |idx: &[usize]| {
            pls1_fit_impl(
                row_subset(x7.as_ref(), idx).as_ref(),
                col_row_subset(y.as_ref(), idx).as_ref(),
                k,
                Some(col_row_subset(w.as_ref(), idx).as_ref()),
                None,
                FitOpts {
                    pre_standardized: true,
                    ..FitOpts::default()
                },
                InternalFitOpts::default(),
            )
            .unwrap()
        };
        let beta = fit(&sample_idx).beta;
        let lev = crate::linalg::leverage_diag(fit(&boot_idx).w_star.as_ref());
        let col = |v: &[f64]| Col::<f64>::from_fn(v.len(), |i| v[i]);
        let got_beta = col(&shifted.beta);
        assert_close_rel(got_beta.as_mat(), beta.as_mat(), 1e-12, "beta");
        let (got_lev, want_lev) = (col(&shifted.leverage), col(&lev));
        assert_close_rel(got_lev.as_mat(), want_lev.as_mat(), 1e-12, "leverage");
    }

    /// The leverage fit of a replicate standardizes the bootstrap resample
    /// on its own (weighted) moments, as the full-data reference fit
    /// standardizes X: its leverage is that of the plain fit on the drawn
    /// rows, with and without weights and covariates.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn replicate_leverage_is_the_plain_fit_on_the_bootstrap_rows() {
        use crate::test_support::{assert_close_rel, synth, test_weights, unif};
        let (n, d, k, m) = (60, 8, 2, 40);
        let (x, y) = synth(n, d, 3, 1.0, 71);
        let x = Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)] * (j + 1) as f64 + 3.0);
        let z = unif(n, 2, 72);
        let (_, boot_idx, _) = replicate_indices(n, m, 11, 0);
        for w in [None, Some(test_weights(n))] {
            for zz in [None, Some(z.as_ref())] {
                let wr = w.as_ref().map(Col::as_ref);
                let row = run_one_confirmatory(
                    x.as_ref(),
                    y.as_ref(),
                    k,
                    m,
                    held_out_scale(x.as_ref(), wr, false).as_ref(),
                    held_out_scale(y.as_mat(), wr, false).as_ref(),
                    wr,
                    zz,
                    false,
                    11,
                    0,
                )
                .unwrap();
                let w_b = wr.map(|w| col_row_subset(w, &boot_idx));
                let z_b = zz.map(|z| row_subset(z, &boot_idx));
                let fit = pls1_fit_impl(
                    row_subset(x.as_ref(), &boot_idx).as_ref(),
                    col_row_subset(y.as_ref(), &boot_idx).as_ref(),
                    k,
                    w_b.as_ref().map(Col::as_ref),
                    z_b.as_ref().map(Mat::as_ref),
                    FitOpts::default(),
                    InternalFitOpts::default(),
                )
                .unwrap();
                let want = crate::linalg::leverage_diag(fit.w_star.as_ref());
                let col = |v: &[f64]| Col::<f64>::from_fn(v.len(), |i| v[i]);
                let (got, want) = (col(&row.leverage), col(&want));
                assert_close_rel(got.as_mat(), want.as_mat(), 1e-12, "leverage");
            }
        }
    }

    #[test]
    fn a_half_split_too_small_for_k_is_invalid_argument() {
        // m = ⌈10^0.9⌉ = 8 passes m ≥ k + 2 = 6; the half split trains on 5 rows.
        assert_eq!(resolve_m(10, 0.9), 8);
        let r = resolve_ci_m(10, 4, 1, 0.9);
        assert!(
            matches!(&r, Err(PlsKitError::InvalidArgument(m)) if m.contains("n/2 = 5")),
            "{r:?}"
        );
    }

    #[test]
    fn a_subsample_too_small_for_the_covariates_is_invalid_argument() {
        // m = ⌈100^0.7⌉ = 26 serves k = 1 alone, not with 24 covariates.
        assert_eq!(resolve_m(100, 0.7), 26);
        assert_eq!(resolve_ci_m(100, 1, 1, 0.7).unwrap(), 26);
        let r = resolve_ci_m(100, 1, 25, 0.7);
        assert!(
            matches!(&r, Err(PlsKitError::InvalidArgument(m)) if m.contains("24 covariates")),
            "{r:?}"
        );
    }
}

/// Output of the confirmatory subsampling engine. Marshalled into
/// `ConfirmatoryCI` at the wrapper layer.
#[derive(Debug, Clone)]
pub struct ConfirmatoryCI {
    /// Number of resampling replicates. Each replicate draws one size-`m`
    /// subsample (β), one n-out-of-n bootstrap resample (leverage) and one
    /// half split of `⌊n/2⌋` training rows (`holdout_corr`).
    pub n_boot: usize,
    /// Subsample size used for each β subsample.
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
    /// population `β[j]` is 0 (K = 1; see `docs/python/results.md`,
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
    /// sanity check. `(−∞, +∞)` when `κ̂ = 0`. See `docs/python/results.md`
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
    /// r-scale. `point` is the mean over the half splits on the r-scale; `sd`
    /// is on the z-scale (see `CIScalar` doc).
    pub holdout_corr: CIScalar,
    /// Number of resamples whose worker fit succeeded (contributors to
    /// leverage, `beta_sign_z`). Equals `n_boot − skipped − n_worker_failed`,
    /// where `skipped` counts the resamples skipped by weight validation.
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
/// with `m` the training size of each split (`holdout_train_size(n)` from
/// `reduce_confirmatory`). The target is ρ(m), the expected held-out
/// correlation of a fit trained on `m` rows.
/// `point` stays on the r-scale (mean over splits) — the plug-in estimate
/// of ρ(m) — while the CI center on the z-scale is `atanh(point)`. `sd` is the
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
    if q < 0.0 { -val } else { val }
}

/// Reduce `Option<ConfirmatoryWorkerRow>` rows under the `max_skip_rate`
/// and `max_failure_rate` contracts. `opt_rows` holds the draws that were
/// not skipped by weight validation (`n_skipped` of them were); a `None` is
/// a failed draw. A row whose half split failed on the weights
/// (`holdout_skipped`) is a skip too, of its `holdout_corr` only.
/// Errors with `ResamplingDegenerate` when the skips (`n_skipped` plus the
/// `holdout_skipped` rows, over `n_boot`) exceed `max_skip_rate`, and when
/// no row is `Some`, or no `Some` row has a non-NaN `holdout_corr`,
/// whatever the rates, since there is nothing to reduce; with
/// `ResampleFailureRateExceeded` if the combined failure rate
/// (`n_worker_failed` plus the rows with a NaN `holdout_corr` that are not
/// `holdout_skipped`, over `n_boot`) exceeds the threshold; otherwise
/// filters `None` rows and patches `n_boot`, `n_boot_finite`,
/// `n_boot_finite_holdout_corr` onto the result.
pub(crate) fn reduce_with_failure_check(
    opt_rows: Vec<Option<ConfirmatoryWorkerRow>>,
    n_skipped: usize,
    opts: SubsampleOpts,
    n: usize,
    m: usize,
    leverage_ref: &[f64],
    beta_ref: ColRef<'_, f64>,
) -> PlsKitResult<ConfirmatoryCI> {
    let n_boot = opts.n_boot;
    let n_holdout_skipped = opt_rows
        .iter()
        .flatten()
        .filter(|r| r.holdout_skipped)
        .count();
    let skipped = n_skipped + n_holdout_skipped;
    #[allow(clippy::cast_precision_loss)]
    let skip_rate = skipped as f64 / n_boot as f64;
    let degenerate = PlsKitError::ResamplingDegenerate {
        skipped,
        total: n_boot,
        skip_rate,
        threshold: opts.max_skip_rate,
    };
    if skip_rate > opts.max_skip_rate
        || opt_rows
            .iter()
            .all(|r| r.as_ref().is_none_or(|r| r.holdout_corr.is_nan()))
    {
        return Err(degenerate);
    }
    let n_worker_failed = opt_rows.iter().filter(|r| r.is_none()).count();
    let rows: Vec<ConfirmatoryWorkerRow> = opt_rows.into_iter().flatten().collect();
    let n_holdout_only_nan = rows
        .iter()
        .filter(|r| r.holdout_corr.is_nan() && !r.holdout_skipped)
        .count();
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
    ci.n_boot_finite = n_boot - n_skipped - n_worker_failed;
    ci.n_boot_finite_holdout_corr = n_boot - skipped - n_holdout_corr_failed;
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

    // ── holdout_corr (NB-Wald CI, on the half splits) ──
    let mut hc_buf = vec![0.0_f64; b];
    for i in 0..b {
        hc_buf[i] = rows[i].holdout_corr;
    }
    let holdout_corr = reduce_holdout_corr(&hc_buf, n, holdout_train_size(n), level);

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
            holdout_skipped: false,
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
            holdout_skipped: false,
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
        n_skipped: usize,
        opts: SubsampleOpts,
    ) -> PlsKitResult<ConfirmatoryCI> {
        let leverage_ref = vec![0.5_f64; 2];
        let beta_ref_col = Col::<f64>::from_fn(2, |_| 1.0);
        reduce_with_failure_check(
            opt_rows,
            n_skipped,
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
        let ci = run(rows, 0, opts_with(100, 0.01)).expect("at-threshold should pass");
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
        let err = run(rows, 0, opts_with(100, 0.01)).unwrap_err();
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
        match run(rows, 0, opts_with(100, 0.0)).unwrap_err() {
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
        let err = run(rows, 0, opts_with(100, 0.0)).unwrap_err();
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
        let ci = run(rows, 0, opts_with(100, 0.01)).expect("1 holdout NaN under threshold");
        assert_eq!(ci.n_boot, 100);
        assert_eq!(ci.n_boot_finite, 100);
        assert_eq!(ci.n_boot_finite_holdout_corr, 99);
    }

    #[test]
    fn no_row_left_is_resampling_degenerate() {
        let all_failed: Vec<Option<ConfirmatoryWorkerRow>> = (0..100).map(|_| None).collect();
        for (rows, n_skipped) in [(all_failed, 0), (Vec::new(), 100)] {
            match run(rows, n_skipped, opts_with(100, 1.0)) {
                Err(PlsKitError::ResamplingDegenerate { total: 100, .. }) => {}
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn no_finite_holdout_corr_is_resampling_degenerate() {
        // Every draw fits with a NaN holdout correlation (one skipped), or
        // one fails and the rest have NaN.
        for n_skipped in [1, 0] {
            let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
                (0..99).map(|_| Some(dummy_row(f64::NAN))).collect();
            if n_skipped == 0 {
                rows.push(None);
            }
            match run(rows, n_skipped, opts_with(100, 1.0)) {
                Err(PlsKitError::ResamplingDegenerate { total: 100, .. }) => {}
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn skips_reduce_the_finite_counts_not_the_failure_rate() {
        let mut rows: Vec<Option<ConfirmatoryWorkerRow>> =
            (0..94).map(|_| Some(dummy_row(0.5))).collect();
        rows.push(Some(dummy_row(f64::NAN)));
        let ci = run(rows, 5, opts_with(100, 0.01)).expect("skips are not failures");
        assert_eq!(ci.n_boot, 100);
        assert_eq!(ci.n_boot_finite, 95);
        assert_eq!(ci.n_boot_finite_holdout_corr, 94);
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
    /// returns `InvalidWeights { reason: "insufficient_effective_n" }` (too small an
    /// effective sample size) or `InvalidWeights { reason: "all_zero" }` (a subsample whose
    /// weights are all zero), that's a *skip* rather than a numerical *failure*. A half-split
    /// fit that fails this way is a skip of the replicate's holdout correlation only. Fires
    /// `PlsKitError::ResamplingDegenerate` when exceeded, and also when no draw is left to
    /// reduce after skips and failures, or every draw left has a NaN holdout correlation,
    /// whatever the two rates.
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
    /// `pls1_fit` returned `InvalidWeights` with reason `"insufficient_effective_n"` or
    /// `"all_zero"` (a subsample whose weights are all zero).
    /// Counts toward `max_skip_rate`, not `max_failure_rate`.
    Skipped,
    /// Any other error (numerical failure, NaN, etc.).
    /// Counts toward `max_failure_rate`, as a `None` row in `reduce_with_failure_check`.
    Failed,
}

/// Engine entry for the confirmatory CI branch. Caller has already done the
/// full-data reference fit and validated arguments. X and y are scaled once
/// over all rows, as the held-out tests scale X (`linalg::held_out_scale`:
/// the full-data fit's weighted moments, or as given under
/// `pre_standardized`), for the β fits and the half splits, which centre
/// their own rows (see `fit_rows`). The leverage fits standardize their
/// resample on its own moments, as the reference fit standardizes all rows.
/// Under `pre_standardized` the β and leverage fits take X and y as given,
/// as the reference fit does. With covariates
/// `z`, `(n, p_z)`, every replicate fit redoes the adjustment on its own
/// rows.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
pub(crate) fn pls1_subsample_inference_confirmatory(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    beta_ref: ColRef<'_, f64>,
    leverage_ref: &[f64],
    opts: SubsampleOpts,
    weights: Option<ColRef<'_, f64>>,
    z: Option<MatRef<'_, f64>>,
    seed: u64,
) -> PlsKitResult<ConfirmatoryCI> {
    opts.validate()?;
    let n = x.nrows();
    // The caller checked m against the covariates' rank.
    let m = resolve_ci_m(n, k, 1, opts.m_rate)?;

    let pre_std = opts.pre_standardized;
    let x_scale = crate::linalg::held_out_scale(x, weights, pre_std);
    let y_scale = crate::linalg::held_out_scale(y.as_mat(), weights, pre_std);
    // The replicates run in batches by the memory budget. One replicate
    // holds at most two n-row copies of X at once: the bootstrap fit's
    // standardized rows and, when weighted, their √w-scaled copy (the
    // subsample fit's copies, of m rows, the half split's, of n/2 rows, and
    // its n - n/2 hold-out rows are smaller and not alive at the same
    // time), and one fit's buffers. The
    // unit counts the two copies whether weighted or not. The call holds
    // every replicate's `beta` and `leverage` rows (`d` entries each) until
    // the reduction.
    let unit = crate::resample::copy_bytes(n, x.ncols())
        .saturating_mul(2)
        .saturating_add(crate::fit::pls1_fit_bytes(x.ncols(), k));
    let rows = opts.n_boot.saturating_mul(x.ncols().saturating_mul(16));
    let outcomes: Vec<WorkerOutcome> = crate::resample::map_units(opts.n_boot, rows, unit, |r| {
        let (xs, ys) = (x_scale.as_ref(), y_scale.as_ref());
        match run_one_confirmatory(x, y, k, m, xs, ys, weights, z, pre_std, seed, r) {
            std::result::Result::Ok(row) => WorkerOutcome::Ok(row),
            Err(PlsKitError::InvalidWeights {
                reason: "insufficient_effective_n" | "all_zero",
            }) => WorkerOutcome::Skipped,
            Err(_) => WorkerOutcome::Failed,
        }
    })?;

    let n_skipped = outcomes
        .iter()
        .filter(|o| matches!(o, WorkerOutcome::Skipped))
        .count();

    // Skipped draws are dropped; a failed one becomes `None`, which
    // `reduce_with_failure_check` counts against `max_failure_rate`, and
    // it checks `max_skip_rate`.
    let opt_rows: Vec<Option<ConfirmatoryWorkerRow>> = outcomes
        .into_iter()
        .filter_map(|o| match o {
            WorkerOutcome::Ok(row) => Some(Some(row)),
            WorkerOutcome::Skipped => None,
            WorkerOutcome::Failed => Some(None),
        })
        .collect();

    reduce_with_failure_check(opt_rows, n_skipped, opts, n, m, leverage_ref, beta_ref)
}

#[cfg(test)]
mod tests_engine {
    use super::*;
    use crate::fit::{FitOpts, pls1_fit};
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
        run_engine_at(x, y, k, n_boot, seed, 0.7)
    }

    fn run_engine_at(
        x: &Mat<f64>,
        y: &Col<f64>,
        k: usize,
        n_boot: usize,
        seed: u64,
        m_rate: f64,
    ) -> ConfirmatoryCI {
        let fit = pls1_fit(x.as_ref(), y.as_ref(), k, None, None, FitOpts::default()).unwrap();
        let leverage_ref = crate::linalg::leverage_diag(fit.w_star.as_ref());
        let opts = SubsampleOpts {
            n_boot,
            m_rate,
            level: 0.95,
            pre_standardized: false,
            max_failure_rate: 0.0,
            max_skip_rate: 1.0,
        };
        pls1_subsample_inference_confirmatory(
            x.as_ref(),
            y.as_ref(),
            k,
            fit.beta.as_ref(),
            &leverage_ref,
            opts,
            None,
            None,
            seed,
        )
        .unwrap()
    }

    /// Replicate 0's half split trains on rows that all have weight 0, so
    /// its fit fails on the weights. That costs the replicate its
    /// `holdout_corr` only, and counts as a skip under `max_skip_rate`, not
    /// as a failure under `max_failure_rate`.
    #[test]
    #[allow(clippy::many_single_char_names)] // n, k, x, y, w are the math names
    fn a_half_split_without_weight_is_a_skip() {
        let (n, k, seed) = (60, 1, 5);
        let (x, y) = synth(n, 4, 1, 1.0, 81);
        let (_, _, (half, _)) = replicate_indices(n, resolve_m(n, 0.7), seed, 0);
        let w = Col::<f64>::from_fn(n, |i| if half.contains(&i) { 0.0 } else { 1.0 });
        let fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            k,
            Some(w.as_ref()),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let leverage_ref = crate::linalg::leverage_diag(fit.w_star.as_ref());
        let run = |max_skip_rate| {
            let opts = SubsampleOpts {
                n_boot: 100,
                m_rate: 0.7,
                level: 0.95,
                pre_standardized: false,
                max_failure_rate: 0.0,
                max_skip_rate,
            };
            pls1_subsample_inference_confirmatory(
                x.as_ref(),
                y.as_ref(),
                k,
                fit.beta.as_ref(),
                &leverage_ref,
                opts,
                Some(w.as_ref()),
                None,
                seed,
            )
        };
        let ci = run(0.01).unwrap();
        assert_eq!(ci.n_boot_finite, 100);
        assert_eq!(ci.n_boot_finite_holdout_corr, 99);
        match run(0.0) {
            Err(PlsKitError::ResamplingDegenerate {
                skipped: 1,
                total: 100,
                ..
            }) => {}
            other => panic!("{other:?}"),
        }
    }

    /// A column that barely varies (1e-44 noise, one row at 0.5) is read on
    /// the all-row scale, as the held-out tests read it, so a half split
    /// that holds out the 0.5 row does not blow that row up by about 1e43:
    /// `holdout_corr` stays where it is without the column.
    #[test]
    fn a_near_flat_column_does_not_move_holdout_corr() {
        let (x, y) = synth(100, 20, 2, 1.0, 61);
        let u = crate::test_support::unif_col(100, 62);
        let x_flat = Mat::<f64>::from_fn(100, 21, |i, j| match (i, j) {
            (_, 0..20) => x[(i, j)],
            (0, _) => 0.5,
            _ => 1e-44 * u[i],
        });
        let plain = run_engine(&x, &y, 1, 100, 63).holdout_corr;
        let flat = run_engine(&x_flat, &y, 1, 100, 63).holdout_corr;
        assert!(
            (plain.point - flat.point).abs() < 0.01,
            "holdout_corr {} without the column, {} with it",
            plain.point,
            flat.point
        );
    }

    /// `holdout_corr` is built on half splits, so its target, the held-out
    /// correlation of a fit on `n/2` rows, does not move with `m_rate`: the
    /// same seed gives the same interval at any `m_rate`, while β follows
    /// its own size-`m` subsample.
    #[test]
    fn holdout_corr_does_not_depend_on_m_rate() {
        let (x, y) = synth(120, 10, 2, 1.0, 31);
        let lo = run_engine_at(&x, &y, 1, 100, 32, 0.6);
        let hi = run_engine_at(&x, &y, 1, 100, 32, 0.9);
        assert!((lo.m, hi.m) == (18, 75), "m = {}, {}", lo.m, hi.m);
        assert_eq!(lo.holdout_corr, hi.holdout_corr);
        assert!(lo.n_boot_finite_holdout_corr == 100 && lo.holdout_corr.sd > 0.0);
        assert_ne!(lo.beta_se, hi.beta_se);
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
            let fit = pls1_fit(x.as_ref(), y.as_ref(), 1, None, None, FitOpts::default()).unwrap();
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
        let beta_ref = pls1_fit(x.as_ref(), y.as_ref(), 2, None, None, FitOpts::default())
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

    /// Under `pre_standardized`, X·2^±600 gives every CI readout of X:
    /// β's bounds and SE times 2^±600, the rest unchanged.
    #[test]
    #[allow(clippy::many_single_char_names)] // n, d, k, x, y, s are the math names
    fn ci_under_pre_standardized_reads_x_at_any_magnitude() {
        let (n, d, k) = (60, 8, 1);
        let (x, y) = synth(n, d, 2, 2.0, 17);
        let run = |s: f64| {
            let xs = Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)] * s);
            let opts = FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            };
            let fit = pls1_fit(xs.as_ref(), y.as_ref(), k, None, None, opts).unwrap();
            let lev = crate::linalg::leverage_diag(fit.w_star.as_ref());
            let ci = pls1_subsample_inference_confirmatory(
                xs.as_ref(),
                y.as_ref(),
                k,
                fit.beta.as_ref(),
                &lev,
                SubsampleOpts {
                    n_boot: 100,
                    m_rate: 0.7,
                    level: 0.95,
                    pre_standardized: true,
                    max_failure_rate: 0.0,
                    max_skip_rate: 0.0,
                },
                None,
                None,
                19,
            )
            .unwrap();
            let mut v: Vec<f64> = [&ci.beta_ci_lower, &ci.beta_ci_upper, &ci.beta_se]
                .iter()
                .flat_map(|a| a.iter().map(|b| b * s))
                .collect();
            for a in [
                &ci.beta_sign_z_signed,
                &ci.leverage_ci_lower,
                &ci.leverage_ci_upper,
                &ci.leverage_se,
            ] {
                v.extend_from_slice(a);
            }
            let h = &ci.holdout_corr;
            v.extend([h.point, h.lower, h.upper, h.sd]);
            v
        };
        let want = run(1.0);
        for s in [2f64.powi(600), 2f64.powi(-600)] {
            for (i, (g, w)) in run(s).iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                    "s = {s:e} [{i}]: {g} vs {w}"
                );
            }
        }
    }

    /// The held-out y is read on the all-row scale, so a y in small units
    /// (y·1e-17) gives the `holdout_corr` of y; raw, its Pearson sums fall
    /// under the 1e-30 floors and every replicate fails.
    #[test]
    fn holdout_corr_reads_y_at_any_unit() {
        let (x, y) = synth(200, 20, 2, 1.0, 61);
        let run = |s: f64| {
            let ys = Col::<f64>::from_fn(200, |i| y[i] * s);
            let h = run_engine(&x, &ys, 1, 100, 5).holdout_corr;
            [h.point, h.lower, h.upper, h.sd]
        };
        let want = run(1.0);
        for (g, w) in run(1e-17).iter().zip(&want) {
            assert!((g - w).abs() <= 1e-12 * w.abs(), "{g} vs {w}");
        }
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
                let (w_norm, _) = crate::fit::validate_and_normalize_weights(c.w, c.x.nrows())?;
                let wr = w_norm.as_ref().map(Col::as_ref);
                let fit = pls1_fit(
                    c.x_owned.as_ref(),
                    c.y,
                    k,
                    wr,
                    None,
                    FitOpts {
                        pre_standardized: c.pre,
                        ..FitOpts::default()
                    },
                )?;
                let lev = crate::linalg::leverage_diag(fit.w_star.as_ref());
                pls1_subsample_inference_confirmatory(
                    c.x,
                    c.y,
                    k,
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
                    None,
                    3,
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

    /// The confirmatory CI gives the same bits with its replicates one at a
    /// time (budget 0) and all in one batch (no limit), unweighted and
    /// weighted.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn confirmatory_ci_does_not_depend_on_the_budget() {
        let (n, k) = (60_usize, 2_usize);
        let (x, y) = synth(n, 6, 2, 2.0, 11);
        let w = crate::test_support::test_weights(n);
        for weights in [None, Some(w.as_ref())] {
            let (w_norm, _) = crate::fit::validate_and_normalize_weights(weights, n).unwrap();
            let wr = w_norm.as_ref().map(Col::as_ref);
            let fit = pls1_fit(x.as_ref(), y.as_ref(), k, wr, None, FitOpts::default()).unwrap();
            let lev = crate::linalg::leverage_diag(fit.w_star.as_ref());
            let run = |b| {
                let (ci, log) = crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(b, || {
                        pls1_subsample_inference_confirmatory(
                            x.as_ref(),
                            y.as_ref(),
                            k,
                            fit.beta.as_ref(),
                            &lev,
                            SubsampleOpts {
                                n_boot: 100,
                                m_rate: 0.7,
                                level: 0.95,
                                pre_standardized: false,
                                max_failure_rate: 0.0,
                                max_skip_rate: 1.0,
                            },
                            wr,
                            None,
                            5,
                        )
                        .unwrap()
                    })
                });
                (format!("{ci:?}"), log)
            };
            let ((one, log_one), (all, log_all)) = (run(0), run(u64::MAX));
            assert_eq!(one, all, "weighted={}", weights.is_some());
            crate::resample::assert_batches(&log_one, 100, 1, 1);
            crate::resample::assert_batches(&log_all, 100, 100, 1);
            // A replicate counts two n-row copies of X and one fit's
            // buffers; the call holds every replicate's β and leverage rows.
            let unit = 2 * crate::resample::copy_bytes(n, 6) + crate::fit::pls1_fit_bytes(6, k);
            let rows = 100 * 2 * 6 * 8;
            for (budget, batch) in [(rows + 2 * unit - 1, 1), (rows + 2 * unit, 2)] {
                let (_, log) = run(u64::try_from(budget).unwrap());
                crate::resample::assert_batches(&log, 100, batch, 1);
            }
        }
    }
}
