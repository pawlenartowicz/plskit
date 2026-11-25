//! Permutation-null engine for signed per-voxel z statistics on PLS1 β.
//!
//! Produces a signed per-feature test statistic suitable for downstream
//! TFCE / cluster-mass / max-stat FWER pipelines (PALM, FSL randomise,
//! nltools). Independent from the subsampling engine in `subsample.rs` —
//! different question (null distribution under permuted y, not sampling
//! distribution under the true DGP) and different per-resample workload
//! (full-size fit on permuted y, no procrustes alignment).
//!
//! At k = 1 a row is not a refit: β = c·X̃'ỹ, and the row holds `c` at its
//! observed value (`c_ref·X̃'ỹ_p`), because a refit's `c` depends on how the
//! permuted y lines up with X's spectrum and makes β non-pivotal (see
//! `run_engine`).

use faer::{ColRef, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::resample::copy_bytes;
use crate::route_select::{Candidate, Choice, select};
use crate::signal_test::{PrimalSource, ReplicateRoute};

/// Tuning knobs for `pls1_perm_null`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct PermNullOpts {
    /// Number of permutations. Must be ≥ 100 (per-voxel-z noise floor).
    pub n_perm: usize,
    /// Whether to return the full `(n_perm, D)` β-matrix on the output.
    /// Without it the rows run in batches by the memory budget and only one
    /// batch is held; mean, SD and z are the same bits either way (pinned
    /// by `streaming_matches_retained_byte_exact`).
    pub return_perm_matrix: bool,
    /// Caller asserts X is already column-standardized; skips centering/scaling.
    pub pre_standardized: bool,
}

impl Default for PermNullOpts {
    fn default() -> Self {
        Self {
            n_perm: 1000,
            return_perm_matrix: false,
            pre_standardized: false,
        }
    }
}

impl PermNullOpts {
    /// Validate args (`n_perm` ≥ 100, k ≥ 1).
    ///
    /// # Errors
    ///
    /// Returns `PlsKitError::InvalidArgument` if `n_perm < 100` or `k < 1`.
    pub fn validate(&self, k: usize) -> PlsKitResult<()> {
        if self.n_perm < 100 {
            return Err(PlsKitError::InvalidArgument(format!(
                "n_perm must be ≥ 100, got {}",
                self.n_perm
            )));
        }
        if k < 1 {
            return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
        }
        Ok(())
    }
}

/// Output of `pls1_perm_null`. All per-voxel arrays length `D`.
#[derive(Debug, Clone)]
#[allow(clippy::doc_markdown)]
pub struct PermNullOutput {
    /// Number of permutations actually run.
    pub n_perm: usize,
    /// K used for fitting.
    pub k: usize,
    /// RNG seed actually used.
    pub seed: u64,
    /// Effective sample size (sum(w)² / sum(w²)); equals n when weights are uniform.
    pub n_eff: f64,
    /// Full-data β reference. Length D.
    pub beta_ref: Vec<f64>,
    /// Mean of β under permuted y. Length D. ≈ 0 under H0 (calibration diagnostic).
    /// At k = 1 "β under permuted y" is `c_ref·X̃'ỹ_p`, the permuted fit with
    /// its scalar `c` held at the observed fit's (module docs); here and in
    /// `beta_perm_sd`, `beta_perm_z` and `beta_perm_matrix`. A non-finite
    /// entry is left out of its coordinate's mean; `n_failed_perms` counts
    /// the rows that hold one.
    pub beta_perm_mean: Vec<f64>,
    /// SD of β under permuted y, `count − 1` in the denominator. Length D.
    /// A non-finite entry is left out of its coordinate's SD; `n_failed_perms`
    /// counts the rows that hold one.
    pub beta_perm_sd: Vec<f64>,
    /// Signed z = β_ref / β_perm_sd per coordinate. NaN where the SD is not
    /// above `√ε · max(|β_ref|, |β_perm_mean|)` (no spread, or spread at the
    /// rounding level of β), and for a column of X whose entries are all
    /// equal. Length D.
    pub beta_perm_z: Vec<f64>,
    /// Optional `(n_perm, D)` β matrix in row-major layout (length n_perm·D).
    /// Some when `opts.return_perm_matrix == true`.
    pub beta_perm_matrix: Option<Vec<f64>>,
    /// `rank([1, Z]) − 1`, the covariates kept after aliased columns are
    /// dropped. `None` without covariates.
    pub n_covariates: Option<usize>,
    /// Of the `n_perm` permutation rows, the number with at least one entry
    /// that is not finite. Each such entry is left out of its coordinate's
    /// `beta_perm_mean` and `beta_perm_sd`; the rest of the row is used.
    pub n_failed_perms: usize,
}

/// Permutation-null engine for PLS1 β. See module docs.
///
/// # Shapes
/// - `x`: `(n, d)`
/// - `y`: `(n,)`
/// - `k`: components retained per fit; `1 ≤ k ≤ d`
/// - `covariates`: optional `Z`, `(n, p_z)`. X is standardized, then X and
///   y are replaced by their (weighted) residuals on `[1, Z]`, as `pls1_fit`
///   does with covariates; β maps are computed on those residuals, and null
///   row c fits the Freedman–Lane column `M_Z·π_c(ỹ)`, the residual on
///   `[1, Z]` of the permuted residual.
///
/// # Errors
/// - `PlsKitError::InvalidArgument` when `n_perm < 100` or `k < 1`
/// - `PlsKitError::DimensionMismatch` when `y.len() != x.nrows()`
/// - `PlsKitError::KExceedsMax` when `k > d`
/// - `PlsKitError::NonFiniteInput` when X or y contains NaN/inf
/// - `PlsKitError::InvalidWeights` when weights are invalid
/// - `PlsKitError::ShapeMismatch` when `covariates` does not have `n` rows
/// - `PlsKitError::NonFiniteInput` when `covariates` contains NaN/inf
/// - `PlsKitError::InsufficientResidualDf` when `n − rank([1, Z]) < k + 1`
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::needless_pass_by_value, clippy::many_single_char_names)]
pub fn pls1_perm_null(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    covariates: Option<MatRef<'_, f64>>,
    opts: PermNullOpts,
    seed: Option<u64>,
) -> PlsKitResult<PermNullOutput> {
    crate::fit::with_thread_limit(|| perm_null_impl(x, y, k, weights, covariates, opts, seed))
}

/// Body of [`pls1_perm_null`], on the caller's pool.
#[allow(
    clippy::needless_pass_by_value,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
pub(crate) fn perm_null_impl(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    covariates: Option<MatRef<'_, f64>>,
    opts: PermNullOpts,
    seed: Option<u64>,
) -> PlsKitResult<PermNullOutput> {
    use crate::fit::{
        FitOpts, InternalFitOpts, fit_row_scale, pls1_fit_impl, rows_scale_in_band, scale_col,
        scale_rows, scale_rows_in_place, validate_weights_for_k,
    };
    use crate::linalg::{standardize, standardize_weighted, standardize1, standardize1_weighted};
    use faer::{Col, Mat};

    opts.validate(k)?;

    let n = x.nrows();
    let d = x.ncols();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, d),
            y: y.nrows(),
        });
    }
    if k > d {
        return Err(PlsKitError::KExceedsMax { k, k_max: d });
    }
    crate::fit::check_finite_mat(x)?;
    crate::fit::check_finite_col(y)?;
    let seed_used = crate::rng::resolve_seed(seed, n)?;

    let (w_norm, n_eff_val) = validate_weights_for_k(weights, n, k)?;
    let wref = w_norm.as_ref().map(Col::as_ref);
    // The covariate basis and its residual df are arguments: checked before
    // the budget, so a bad `covariates` gives its own error at any budget.
    let pending_basis = covariates
        .map(|z| {
            let b = crate::covariates::basis_for(x, z, wref)?;
            b.check_df(k)?;
            Ok::<_, PlsKitError>(b)
        })
        .transpose()?;
    // The route and the rows at once, before the standardized copy of X or
    // any buffer of the loop exists.
    let plan = perm_null_plan(
        n,
        d,
        opts.n_perm,
        k,
        wref.is_some(),
        opts.return_perm_matrix,
    )?;

    // Standardize once. Subsequent permutations operate on standardized arrays
    // — permuting y after standardization is equivalent to permuting raw y and
    // re-standardizing because mean/scale are permutation-invariant.
    //
    // Convention: weighted moments when weights are present, matching pls1_fit's
    // own path (standardize_weighted, fit.rs). beta_ref is taken from a reference
    // fit run on these pre-standardized arrays, so it equals the standardized-scale
    // coefficient (Pls1Model.coef) of a direct weighted pls1_fit on the same input
    // — verified to f64 epsilon. (Not the raw-scale Pls1Model.beta, which fit
    // back-projects by x_scale/y_scale; beta_ref stays on the standardized scale,
    // which is the scale the per-voxel z statistic is defined on.) The
    // y-permutation argument still holds: the weighted moments depend on w, and w
    // is NOT permuted (it stays tied to row i), so the transform applied to
    // observed and permuted y is identical and the null is internally consistent.
    //
    // Under `pre_standardized` the caller's X is used as given: a
    // column-major view is borrowed and any other layout is copied into
    // column-major storage (`linalg::col_major_or_copy` says why the layout
    // matters).
    //
    // With covariates, X and y are standardized the same way and then
    // replaced by their residuals on `[1, Z]` (`covariates::adjust_on`, as
    // `pls1_fit` does), so `beta_ref` is the covariate fit's `coef`.
    let mut basis = None;
    let (xs_owned, ys_owned): (Option<Mat<f64>>, Col<f64>) = if let Some(b) = pending_basis {
        let adj = crate::covariates::adjust_on(x, y, b, wref, opts.pre_standardized);
        basis = Some(adj.basis);
        (Some(adj.x), adj.y)
    } else if opts.pre_standardized {
        (
            crate::linalg::col_major_or_copy(x),
            Col::<f64>::from_fn(n, |i| y[i]),
        )
    } else if wref.is_some() {
        // wref are the same mean-1 normalized weights pls1_fit standardizes with.
        let (xs, _, _) = standardize_weighted(x, wref);
        let (ys, _, _) = standardize1_weighted(y, wref);
        (Some(xs), ys)
    } else {
        let (xs, _, _) = standardize(x);
        let (ys, _, _) = standardize1(y);
        (Some(xs), ys)
    };
    // Under `pre_standardized`, an X far from unit scale is divided by a
    // power of two `c` (`linalg::pre_standardized_div`): every β below is
    // then `c` times its value on X, and is divided back at the end.
    let div = opts
        .pre_standardized
        .then(|| {
            crate::linalg::pre_standardized_div(xs_owned.as_ref().map_or(x, Mat::as_ref), wref)
        })
        .flatten();
    let c = div.as_ref().map(|d| d.1);
    let xs_owned = div.map_or(xs_owned, |d| Some(d.0));
    let xs: MatRef<'_, f64> = xs_owned.as_ref().map_or(x, Mat::as_ref);
    let ys = ys_owned.as_ref();

    // Reference fit on full standardized data. Weighted, it reads the
    // scaled copy built below for the loop, unweighted, with y scaled
    // alike: the weighted fit's kernel input without a second copy.
    // Outside the band `fit::rows_scale_in_band` checks, it is the weighted
    // fit on `xs`, before the copy.
    let fit_on = |xf: MatRef<'_, f64>, yf: ColRef<'_, f64>, w: Option<ColRef<'_, f64>>| {
        pls1_fit_impl(
            xf,
            yf,
            k,
            w,
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
            InternalFitOpts {
                // check_n_eff: false — n_eff was already validated at the top-level
                // entry; truncation here just yields β at k_used (fit at what exists).
                check_n_eff: false,
                ..InternalFitOpts::default()
            },
        )
    };

    // Weights stay tied to rows under every permutation, so the √w row
    // scaling `pls1_fit` applies to X̃ is the same matrix in every replicate:
    // build it once. The factor is `pls1_fit`'s own (it renormalizes the
    // weights it is handed), so it is √(normalize_weights(w')), not √w'.
    // A standardized copy that is ours is scaled in place, so a single n×d
    // buffer lives through the permutation loop.
    let sqw: Option<Col<f64>> = wref.map(fit_row_scale);
    let ys_scaled: Option<Col<f64>> = sqw
        .as_ref()
        .filter(|s| rows_scale_in_band(xs, s.as_ref()))
        .map(|s| scale_col(ys, s.as_ref()));
    let fit_ref = match ys_scaled {
        None => Some(fit_on(xs, ys, wref)?),
        Some(_) => None,
    };
    let xs_fit_owned: Option<Mat<f64>> = match (sqw.as_ref(), xs_owned) {
        (Some(s), Some(mut m)) => {
            scale_rows_in_place(&mut m, s.as_ref());
            Some(m)
        }
        // No copy of ours: `xs` is `x` itself.
        (Some(s), None) => Some(scale_rows(x, s.as_ref())),
        (None, owned) => owned,
    };
    let xs_fit: MatRef<'_, f64> = xs_fit_owned.as_ref().map_or(x, Mat::as_ref);
    let fit_ref = match fit_ref {
        Some(f) => f,
        None => fit_on(xs_fit, ys_scaled.as_ref().map_or(ys, Col::as_ref), None)?,
    };
    let beta_ref: Vec<f64> = (0..d).map(|j| fit_ref.beta[j]).collect();

    // Freedman–Lane: null column c is the residual on `[1, Z]` of the
    // permuted residual, `M_Z·π_c(ỹ)`, on the raw (not √w-scaled) scale.
    let fl = basis
        .as_ref()
        .map(|b| move |_: usize, m: faer::MatMut<'_, f64>| b.residualize(m));
    let mut out = run_engine(
        xs_fit,
        ys,
        k,
        sqw.as_ref().map(Col::as_ref),
        beta_ref,
        n_eff_val,
        opts,
        seed_used,
        fl.as_ref().map(|f| f as crate::resample::NullMap<'_>),
        plan,
    );
    out.n_covariates = basis
        .as_ref()
        .map(crate::covariates::CovariateBasis::n_covariates);
    if let Some(c) = c {
        let o = &mut out;
        let betas = o.beta_ref.iter_mut().chain(&mut o.beta_perm_mean);
        let betas = betas.chain(&mut o.beta_perm_sd);
        for v in betas.chain(o.beta_perm_matrix.iter_mut().flatten()) {
            *v /= c;
        }
    }
    // A column whose entries are all equal standardizes to zero on the
    // default path, where z is NaN. Under `pre_standardized` it stays as
    // given and its β, null mean and null SD are rounding noise, so z is
    // set NaN on every path from X itself.
    #[allow(clippy::float_cmp)]
    for (j, z) in out.beta_perm_z.iter_mut().enumerate() {
        let col = x.col(j);
        if (1..n).all(|i| col[i] == col[0]) {
            *z = f64::NAN;
        }
    }
    Ok(out)
}

// The row driver of the permutation null: on the route and lanes of
// `perm_null_plan`, the block is built once, then the rows run in batches and
// fold into per-coordinate moments in row order. With
// `opts.return_perm_matrix` every row is kept in one row-major (B, D)
// buffer, which is returned; otherwise only one batch's rows are held.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn run_engine(
    xs_fit: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    sqw: Option<faer::ColRef<'_, f64>>,
    beta_ref: Vec<f64>,
    n_eff_val: f64,
    opts: PermNullOpts,
    seed_used: u64,
    null_map: Option<crate::resample::NullMap<'_>>,
    (route, lanes): (Option<ReplicateRoute>, usize),
) -> PermNullOutput {
    debug_assert!(k <= xs_fit.ncols(), "k <= p is the caller's precondition");
    let d = xs_fit.ncols();
    let b = opts.n_perm;
    // The null targets: column c ≥ 1 of `cols` is `ys` with its rows of
    // positive weight permuted (`resample::Columns::permutation`), then
    // `null_map`, the columns every route takes.
    let cols = crate::resample::Columns {
        y: ys.as_mat(),
        seed: seed_used,
        n_perm: b,
        null_map,
        w: sqw,
    };
    // The fit target X̃'s rows see: column `c` of `cols`, `√w`-scaled when weighted.
    let target = |c: usize| {
        let y = cols.column(c);
        match sqw {
            Some(s) => crate::fit::scale_col(y.as_ref(), s),
            None => y,
        }
    };

    // One row per permutation, in batches of `lanes` rows. The kept (B, D)
    // matrix is the dominant allocation (~0.8–8 GB at D = 1e5–1e6, B ≥ 1e3,
    // the fMRI target scale), counted in the call's bytes when it is returned.
    let retain = opts.return_perm_matrix;
    let mut moments = RowMoments::new(d, ys.norm_max());
    let fold = |rows: &[f64]| moments.fold(rows);
    let flat = match route {
        None => {
            // At k = 1, β = c·X̃'ỹ with c = ỹ'Kỹ / ỹ'K²ỹ (K = X̃X̃'). A
            // permuted y does not line up with the top of K's spectrum the
            // way a real one can, so its c is larger and the null spread of
            // β is too wide for the observed scale: β is not pivotal. Each
            // row holds c at its observed value, `c_ref·X̃'ỹ_p`, so the null
            // is that of X_j'y, the same null as β_j = 0 at k = 1.
            let s_ref =
                crate::linalg::mat_vec(xs_fit.transpose(), target(0).as_ref(), faer::Par::Seq);
            let ss: f64 = (0..d).map(|j| s_ref[j] * s_ref[j]).sum();
            let c_ref = if ss > 0.0 {
                (0..d).map(|j| beta_ref[j] * s_ref[j]).sum::<f64>() / ss
            } else {
                0.0
            };
            let row = |i: usize, row: &mut [f64]| {
                let s = crate::linalg::mat_vec(
                    xs_fit.transpose(),
                    target(i + 1).as_ref(),
                    faer::Par::Seq,
                );
                for (j, v) in row.iter_mut().enumerate() {
                    *v = c_ref * s[j];
                }
            };
            crate::resample::fill_rows_in_batches(b, d, lanes, retain, row, fold)
        }
        Some(route) => {
            let block = perm_block(route, xs_fit, crate::fit::par_fixed());
            let row = |i: usize, row: &mut [f64]| {
                perm_row(&block, xs_fit, sqw, cols.column(i + 1).as_ref(), k, row);
            };
            crate::resample::fill_rows_in_batches(b, d, lanes, retain, row, fold)
        }
    };
    let (beta_perm_mean, beta_perm_sd, n_failed_perms) = moments.finish();
    let beta_perm_z = signed_z(&beta_ref, &beta_perm_mean, &beta_perm_sd);

    PermNullOutput {
        n_perm: b,
        k,
        seed: seed_used,
        n_eff: n_eff_val,
        beta_ref,
        beta_perm_mean,
        beta_perm_sd,
        beta_perm_z,
        beta_perm_matrix: retain.then_some(flat),
        n_covariates: None,
        n_failed_perms,
    }
}

/// Per-coordinate mean and SD of the rows folded into it, Welford's update
/// in row order, so the result depends only on the rows and their order,
/// not on how they are batched. Non-finite entries are skipped per
/// coordinate, in both numerator and denominator; `failed` counts the rows
/// with at least one, the `n_failed_perms` of [`PermNullOutput`].
///
/// The moments are formed on the rows divided by `s`, the power of two
/// [`crate::linalg::pow2_scale`] picks for a magnitude the rows scale with,
/// and multiplied back by `s` at the end. β is linear in y, so `s` taken
/// from y keeps the squared deviations in range for β near `1e±158`, where
/// they overflow to inf or underflow to 0 on the raw rows. A power of two
/// moves no bit, so in range the result is the one on the raw rows.
struct RowMoments {
    count: Vec<usize>,
    mean: Vec<f64>,
    m2: Vec<f64>,
    failed: usize,
    scale: (f64, f64),
}

impl RowMoments {
    /// `d` coordinates; `max_abs` is the magnitude `s` is taken from.
    fn new(d: usize, max_abs: f64) -> Self {
        Self {
            count: vec![0; d],
            mean: vec![0.0; d],
            m2: vec![0.0; d],
            failed: 0,
            scale: crate::linalg::pow2_scale(max_abs),
        }
    }

    /// Fold row-major rows of `d` entries (`d ≥ 1`), in order.
    fn fold(&mut self, rows: &[f64]) {
        let inv = self.scale.1;
        for row in rows.chunks_exact(self.mean.len()) {
            let mut row_failed = false;
            for (j, &v) in row.iter().enumerate() {
                if v.is_finite() {
                    let v = v * inv;
                    self.count[j] += 1;
                    let delta = v - self.mean[j];
                    self.mean[j] += delta / self.count[j] as f64;
                    self.m2[j] += delta * (v - self.mean[j]);
                } else {
                    row_failed = true;
                }
            }
            self.failed += usize::from(row_failed);
        }
    }

    /// `(mean, sd, failed)`, the SD with `count − 1` in the denominator and 0
    /// below two finite values.
    fn finish(self) -> (Vec<f64>, Vec<f64>, usize) {
        let s = self.scale.0;
        let sd = self
            .m2
            .iter()
            .zip(&self.count)
            .map(|(m2, &c)| {
                if c > 1 {
                    (m2 / (c - 1) as f64).sqrt() * s
                } else {
                    0.0
                }
            })
            .collect();
        let mean = self.mean.iter().map(|m| m * s).collect();
        (mean, sd, self.failed)
    }
}

/// β_ref / β_perm_sd per coordinate, NaN where the SD is not above
/// `√ε · max(|β_ref|, |β_perm_mean|)`, so the rule does not depend on units.
#[allow(clippy::doc_markdown)]
fn signed_z(beta_ref: &[f64], mean: &[f64], sd: &[f64]) -> Vec<f64> {
    let eps = f64::EPSILON.sqrt();
    beta_ref
        .iter()
        .zip(mean)
        .zip(sd)
        .map(|((b, m), s)| {
            if *s > eps * b.abs().max(m.abs()) {
                b / s
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// `(call_bytes, row_bytes)` of `B` rows of `d` entries on `n × d` X: the
/// call holds the standardized copy of X the rows read (`n × d`), the three
/// per-coordinate vectors of [`RowMoments`] and, with `retain`, the kept
/// `B × D` matrix; without it a row holds its entry of the batch buffer.
fn rows_bytes(n: usize, b: usize, d: usize, retain: bool) -> (usize, usize) {
    // shortcut: one copy of X counted. A pre-standardized column-major X without weights is borrowed, not copied; two rare paths hold a second n × d copy for a while (the weighted reference fit outside `fit::rows_scale_in_band`, whose `scale_rows` copy sits beside `xs`, and a pre-standardized non-column-major X whose `col_major_or_copy` copy overlaps `pre_standardized_div`'s); count them if a capped run is killed there.
    let x_copy = copy_bytes(n, d);
    let row = d.saturating_mul(8);
    let call = x_copy.saturating_add(row.saturating_mul(3));
    if retain {
        (call.saturating_add(b.saturating_mul(row)), 0)
    } else {
        (call, row)
    }
}

/// The route and the rows at once of `pls1_perm_null` on `n × d` X with
/// `B` permutations: at k = 1 no route, the rows' buffers ([`rows_bytes`])
/// and one row's `X̃'ỹ_p` checked alone; else [`perm_null_route`], which
/// owns the rule. Called once at the call's entry, on the calling thread.
///
/// # Errors
/// `MemoryBudget` when one row does not fit.
fn perm_null_plan(
    n: usize,
    d: usize,
    b: usize,
    k: usize,
    weighted: bool,
    retain: bool,
) -> PlsKitResult<(Option<ReplicateRoute>, usize)> {
    if k == 1 {
        let (call, row_bytes) = rows_bytes(n, b, d, retain);
        let unit = row_bytes.saturating_add(copy_bytes(d, 1));
        return Ok((None, crate::resample::units_at_once(b, call, unit)?));
    }
    let choice = perm_null_route(n, d, b, k, weighted, retain)?;
    Ok((Some(choice.route), choice.lanes))
}

/// `pls1_perm_null`'s route at `k ≥ 2` (`k = 1` has a closed form and no
/// route), called once per call on the calling thread before any parallel
/// work: the least work per `√lanes` among the routes its rules allow
/// (`route_select::select`), in this order, one unit per permutation row,
/// with `B = n_perm` (the observed fit stays on the Primal route), in
/// primal multiply-adds:
///
/// - `Nspace` (`dual_route::nspace_allowed_perm_null`: dense, unweighted):
///   `dual_route::nspace_work` with `n²·p` for `G = X̃X̃'`, a full product;
/// - `GramP` (`gram_p::gram_p_allowed`): `gram_p::gram_p_work`;
/// - `Primal(Copy)`: `gram_p::gram_p_primal_work`, `2·B·k·n·p`.
///
/// `Nspace` and `GramP` only while `gram_routes_disabled()` is false. The
/// call holds the rows' buffers ([`rows_bytes`]) and the
/// route's block: `G` (`n × n`) or `C` (`p × p`). A row holds its entry of
/// the batch buffer (without `retain`) and one PLS1 fit's buffers: the
/// n-space kernel's (`dual_route::nspace_run_bytes`) on that route,
/// `fit::pls1_fit_bytes` off it.
///
/// # Errors
/// `MemoryBudget` when no route fits one row.
#[allow(clippy::many_single_char_names)]
pub(crate) fn perm_null_route(
    n: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    weighted: bool,
    retain: bool,
) -> PlsKitResult<Choice<ReplicateRoute>> {
    use ReplicateRoute::{GramP, Nspace, Primal};
    let gram_on = !crate::signal_test::gram_routes_disabled();
    let (held, row) = rows_bytes(n, n_perm, p, retain);
    let fit_row = row.saturating_add(crate::fit::pls1_fit_bytes(p, k));
    let nspace = |_: usize| {
        (
            held.saturating_add(crate::signal_test::block_gram_bytes(n)),
            row.saturating_add(crate::dual_route::nspace_run_bytes(n, n, k, 1)),
        )
    };
    let gram_p = |_: usize| (held.saturating_add(copy_bytes(p, p)), fit_row);
    let primal = |_: usize| (held, fit_row);
    let (nf, pf, bf, kf) = (n as f64, p as f64, n_perm as f64, k as f64);
    let mut candidates = Vec::with_capacity(3);
    if gram_on && crate::dual_route::nspace_allowed_perm_null(n, k, !weighted) {
        candidates.push(Candidate {
            route: Nspace,
            work: crate::dual_route::nspace_work(nf * nf * pf, 1, n_perm, k, n, 0),
            bytes: &nspace,
            group: None,
        });
    }
    if gram_on && crate::gram_p::gram_p_allowed(n, p, n_perm, k) {
        candidates.push(Candidate {
            route: GramP,
            work: crate::gram_p::gram_p_work(nf, pf, bf, kf),
            bytes: &gram_p,
            group: None,
        });
    }
    candidates.push(Candidate {
        route: Primal(PrimalSource::Copy),
        work: crate::gram_p::gram_p_primal_work(nf, pf, bf, kf),
        bytes: &primal,
        group: None,
    });
    select(n_perm, &candidates)
}

/// Per-call state of a route, built once by `run_engine` and shared
/// read-only by every permutation row. The primal route needs only
/// `‖xs_fit‖_F`.
enum PermBlock {
    /// `x_fro = ‖xs_fit‖_F` (`norm_l2` of the view), the norm the kernel's
    /// truncation floor reads. Taken once per call instead of once per row:
    /// the same call on the same view, so the same bits.
    Primal {
        /// `xs_fit.norm_l2()`.
        x_fro: f64,
    },
    /// n-space Gram route: `G = X̃X̃'` with its norms, built once per call on
    /// the calling thread.
    Nspace(crate::dual_route::NspaceGram),
    /// The p-space Gram block of `xs_fit`, built once per call.
    GramP(crate::gram_p::GramPBlock),
}

/// The block of `route` for the prepared `xs_fit`, its precompute run under
/// `par` (`fit::par_fixed()`). `Special` is not a `perm_null` route;
/// handed it, the builder returns the primal block. `Nspace` builds `G` and
/// its norms once; `GramP` builds `C = X̃'X̃` once; `Primal` takes
/// `‖xs_fit‖_F` once.
fn perm_block(route: ReplicateRoute, xs_fit: MatRef<'_, f64>, par: faer::Par) -> PermBlock {
    match route {
        ReplicateRoute::Nspace => {
            PermBlock::Nspace(crate::dual_route::NspaceGram::new(xs_fit, par))
        }
        ReplicateRoute::GramP => PermBlock::GramP(crate::gram_p::GramPBlock::new(xs_fit, par)),
        ReplicateRoute::Special | ReplicateRoute::Primal(_) => PermBlock::Primal {
            x_fro: xs_fit.norm_l2(),
        },
    }
}

/// One permutation row: β of the refit on `y_null`, a null column of the
/// standardized outcome (scaled by `sqw` when weighted), written into `out`
/// (length d). `xs_fit` is standardized and, when weighted, √w-scaled.
/// Weights are NOT permuted: `w[i]` stays tied to row `i` whichever `y`
/// value lands there (key invariant), which is why `xs_fit` can be shared.
/// Every input check `pls1_fit` would make already passed on the reference
/// fit (same X, a permutation of the same y, same `k` and weights), so the
/// primal arm runs the kernel tail directly, on the block's `‖xs_fit‖_F`;
/// β is `coef` because the arrays are pre-standardized.
fn perm_row(
    block: &PermBlock,
    xs_fit: MatRef<'_, f64>,
    sqw: Option<ColRef<'_, f64>>,
    y_null: ColRef<'_, f64>,
    k: usize,
    out: &mut [f64],
) {
    use crate::fit::{ParChoice, pls1_fit_prepared_fro, scale_col};
    match block {
        &PermBlock::Primal { x_fro } => {
            // Seq inside the per-permutation worker: outer Rayon owns the threadpool.
            let fit = match sqw {
                Some(s) => {
                    let ys_fit = scale_col(y_null, s);
                    pls1_fit_prepared_fro(xs_fit, ys_fit.as_ref(), k, None, ParChoice::Seq, x_fro)
                }
                None => pls1_fit_prepared_fro(xs_fit, y_null, k, None, ParChoice::Seq, x_fro),
            };
            for (j, v) in out.iter_mut().enumerate() {
                *v = fit.coef[j];
            }
        }
        PermBlock::Nspace(gram) => {
            debug_assert!(k >= 1 && sqw.is_none());
            crate::dual_route::nspace_perm_row(gram, xs_fit, y_null, k, out);
        }
        PermBlock::GramP(gram) => {
            debug_assert!(k >= 1);
            let coef = perm_row_coef(gram, xs_fit, sqw, y_null, k);
            for (j, v) in out.iter_mut().enumerate() {
                *v = coef[j];
            }
        }
    }
}

/// The Gram-p arm's fit of one permutation row, from the block's `C` alone
/// after the GEMV `s₁ = xs_fit'ys`: the fit target the Primal arm builds
/// (the null column `y_null`, `scale_col`-ed by `sqw` when weighted,
/// weights tied to rows), `s₁` under the Primal arm's `ParChoice::Seq`.
/// `pre_standardized = true` inside the loop, so the row is `coef`.
fn perm_row_coef(
    gram: &crate::gram_p::GramPBlock,
    xs_fit: MatRef<'_, f64>,
    sqw: Option<ColRef<'_, f64>>,
    y_null: ColRef<'_, f64>,
    k: usize,
) -> faer::Col<f64> {
    let scaled = sqw.map(|s| crate::fit::scale_col(y_null, s));
    let ys = scaled.as_ref().map_or(y_null, faer::Col::as_ref);
    let s1 = crate::gram_p::first_product(xs_fit, ys, crate::fit::ParChoice::Seq, k);
    gram.fit(s1.as_ref(), ys.norm_l2(), k, None).coef
}

#[cfg(test)]
mod layout_invariance {
    use super::*;
    use crate::test_support::{assert_bits_eq, copy_free_families, for_each_layout};
    use faer::Col;

    fn assert_out_bits(a: &PermNullOutput, b: &PermNullOutput, what: &str) {
        assert_eq!(
            (a.n_perm, a.k, a.seed),
            (b.n_perm, b.k, b.seed),
            "{what}: counts"
        );
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "{what}.n_eff");
        assert_bits_eq(&a.beta_ref, &b.beta_ref, &format!("{what}.beta_ref"));
        assert_bits_eq(
            &a.beta_perm_mean,
            &b.beta_perm_mean,
            &format!("{what}.beta_perm_mean"),
        );
        assert_bits_eq(
            &a.beta_perm_sd,
            &b.beta_perm_sd,
            &format!("{what}.beta_perm_sd"),
        );
        assert_bits_eq(
            &a.beta_perm_z,
            &b.beta_perm_z,
            &format!("{what}.beta_perm_z"),
        );
        match (&a.beta_perm_matrix, &b.beta_perm_matrix) {
            (Some(x), Some(y)) => assert_bits_eq(x, y, &format!("{what}.beta_perm_matrix")),
            (None, None) => {}
            _ => panic!("{what}: beta_perm_matrix presence differs"),
        }
    }

    /// `pls1_perm_null` copies X column-major before the loop (standardizing,
    /// √w-scaling, or `col_major_or_copy` under `pre_standardized`), so every
    /// memory layout of X gives the owned column-major matrix's bits, on the
    /// Primal route and (the unweighted `wide` family) the n-space route.
    #[test]
    fn pls1_perm_null_is_bit_identical_across_layouts() {
        for f in copy_free_families() {
            let wr = f.w.as_ref().map(Col::as_ref);
            for pre in [false, true] {
                let (x0, y0) = f.inputs(pre);
                let opts = PermNullOpts {
                    n_perm: 100,
                    return_perm_matrix: true,
                    pre_standardized: pre,
                };
                for_each_layout(
                    &x0,
                    |_, xv| pls1_perm_null(xv, y0.as_ref(), 2, wr, None, opts, Some(31)).unwrap(),
                    |view, got, want| {
                        assert_out_bits(got, want, &format!("{} {view} pre={pre}", f.name));
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests_contract {
    use super::*;
    use faer::{Col, Mat};

    // Recorded from commonstats 0.3.0: gen_resample_indices(Scheme::Permutation, 10, c, 2809).
    const PERMS_N10_S2809: [[u32; 10]; 3] = [
        [4, 1, 0, 6, 5, 7, 2, 3, 9, 8],
        [7, 9, 4, 5, 1, 2, 6, 3, 8, 0],
        [1, 2, 4, 8, 5, 7, 9, 0, 3, 6],
    ];

    /// The rank of each entry, 0 for the smallest (entries distinct).
    fn ranks(v: &[f64]) -> Vec<u32> {
        v.iter()
            .map(|a| u32::try_from(v.iter().filter(|b| *b < a).count()).unwrap())
            .collect()
    }

    /// With X the identity and `k = 1`, row `c − 1` of the β matrix is an
    /// increasing affine map of `π_c(y)`, so on `y = 0..10` its ranks are
    /// null column `c`'s permutation itself.
    #[test]
    fn public_null_columns_are_the_neurostats_draws() {
        let x = Mat::<f64>::identity(10, 10);
        let y = Col::<f64>::from_fn(10, |i| i as f64);
        let opts = PermNullOpts {
            n_perm: 100,
            return_perm_matrix: true,
            ..Default::default()
        };
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, None, opts, Some(2809)).unwrap();
        assert_eq!(out.seed, 2809);
        assert_eq!(ranks(&out.beta_ref), (0..10).collect::<Vec<u32>>());
        let rows = out.beta_perm_matrix.unwrap();
        for (i, want) in PERMS_N10_S2809.iter().enumerate() {
            assert_eq!(
                ranks(&rows[i * 10..(i + 1) * 10]),
                want.to_vec(),
                "null column {}",
                i + 1
            );
        }
    }
}

#[cfg(test)]
mod tests_engine_streaming {
    use super::*;
    use crate::test_support::{bits, synth};

    #[test]
    fn streaming_matches_retained_byte_exact() {
        // Both paths fold the same rows in row order; only retained keeps
        // and returns the matrix.
        let (x, y) = synth(100, 5, 2, 4.0, 42);
        let opts_retained = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: true,
            pre_standardized: false,
        };
        let opts_streaming = PermNullOpts {
            return_perm_matrix: false,
            ..opts_retained
        };
        let r1 = pls1_perm_null(
            x.as_ref(),
            y.as_ref(),
            2,
            None,
            None,
            opts_retained,
            Some(99),
        )
        .unwrap();
        let r2 = pls1_perm_null(
            x.as_ref(),
            y.as_ref(),
            2,
            None,
            None,
            opts_streaming,
            Some(99),
        )
        .unwrap();
        assert_eq!((r1.n_perm, r1.k), (200, 2));
        assert_eq!(r1.beta_perm_matrix.as_ref().map(Vec::len), Some(200 * 5));
        assert!(r2.beta_perm_matrix.is_none());
        assert_eq!(
            bits(&r1.beta_perm_mean),
            bits(&r2.beta_perm_mean),
            "beta_perm_mean must be byte-exact between retained and streaming"
        );
        assert_eq!(
            bits(&r1.beta_perm_sd),
            bits(&r2.beta_perm_sd),
            "beta_perm_sd must be byte-exact between retained and streaming"
        );
        assert_eq!(
            bits(&r1.beta_perm_z),
            bits(&r2.beta_perm_z),
            "beta_perm_z must be byte-exact between retained and streaming"
        );
    }

    /// At a budget below the `B × D` matrix's bytes but above one batch of
    /// rows, the call that returns the matrix does not fit and the call
    /// without it runs, its rows in batches, and gives the retained run's
    /// mean, SD and z bit for bit; at budget 1 neither fits.
    #[test]
    fn perm_null_streams_without_matrix() {
        let (x, y) = synth(60, 50, 2, 4.0, 42);
        let retained = PermNullOpts {
            n_perm: 1000,
            return_perm_matrix: true,
            pre_standardized: false,
        };
        let streaming = PermNullOpts {
            return_perm_matrix: false,
            ..retained
        };
        let run = |o, budget| {
            crate::fit::with_budget(budget, || {
                pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, None, o, Some(99))
            })
        };
        let full = run(retained, crate::fit::DEFAULT_MEMORY_BUDGET).unwrap();
        let low = 1000 * 50 * 8 - 1;
        for (o, budget) in [(retained, low), (streaming, 1)] {
            let r = run(o, budget);
            assert!(
                matches!(r, Err(PlsKitError::MemoryBudget { .. })),
                "budget {budget}: {r:?}"
            );
        }
        let s = run(streaming, low).unwrap();
        assert!(s.beta_perm_matrix.is_none());
        // Two batches run here (856 rows fit beside the call's bytes), and
        // the rows fold in the same order, so the summaries are the same bits.
        for (name, a, b) in [
            ("mean", &s.beta_perm_mean, &full.beta_perm_mean),
            ("sd", &s.beta_perm_sd, &full.beta_perm_sd),
            ("z", &s.beta_perm_z, &full.beta_perm_z),
        ] {
            assert_eq!(bits(a), bits(b), "beta_perm_{name}");
        }
    }

    /// The standardized copy of X (`n × d`) is counted before it is
    /// written: a budget one byte below its bytes is `MemoryBudget` naming
    /// a need above them, on the routed and the closed-form path.
    #[test]
    fn perm_null_counts_the_x_copy() {
        let (x, y) = synth(60, 50, 2, 4.0, 42);
        let copy = crate::resample::copy_bytes(60, 50) as u64;
        let opts = PermNullOpts {
            n_perm: 100,
            ..PermNullOpts::default()
        };
        for k in [1, 2] {
            let r = crate::fit::with_budget(copy - 1, || {
                pls1_perm_null(x.as_ref(), y.as_ref(), k, None, None, opts, Some(5))
            });
            assert!(
                matches!(r, Err(PlsKitError::MemoryBudget { needed, .. }) if needed > copy),
                "k = {k}: {r:?}"
            );
        }
    }

    /// A bad `covariates` gives its own error at budget 1, before the
    /// budget is checked: covariates of the wrong row count, and covariates
    /// that leave fewer residual degrees of freedom than k needs.
    #[test]
    fn covariate_errors_win_over_the_budget() {
        let (x, y) = synth(60, 50, 2, 4.0, 42);
        let opts = PermNullOpts {
            n_perm: 100,
            ..PermNullOpts::default()
        };
        let short = faer::Mat::<f64>::from_fn(59, 1, |i, _| f64::from(u32::try_from(i).unwrap()));
        // 58 random columns and the intercept leave 1 residual df, below k = 2.
        let (wide, _) = synth(60, 58, 2, 1.0, 7);
        let run = |z: &faer::Mat<f64>| {
            crate::fit::with_budget(1, || {
                pls1_perm_null(
                    x.as_ref(),
                    y.as_ref(),
                    2,
                    None,
                    Some(z.as_ref()),
                    opts,
                    Some(5),
                )
            })
        };
        let r = run(&short);
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))), "{r:?}");
        let r = run(&wide);
        assert!(
            matches!(r, Err(PlsKitError::InsufficientResidualDf { .. })),
            "{r:?}"
        );
    }
}

#[cfg(test)]
mod tests_row_moments {
    use super::RowMoments;

    #[test]
    fn row_moments_counts_rows_with_a_non_finite_entry() {
        let rows = [
            [1.0, 2.0, 3.0],
            [f64::NAN, 5.0, 7.0],
            [4.0, -1.0, 0.5],
            [2.5, 3.0, f64::INFINITY],
        ];
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let mut m = RowMoments::new(3, 7.0);
        m.fold(&flat[..3]);
        m.fold(&flat[3..]);
        let (mean, sd, failed) = m.finish();
        assert_eq!(failed, 2);
        for j in 0..3 {
            let v: Vec<f64> = rows
                .iter()
                .map(|r| r[j])
                .filter(|v| v.is_finite())
                .collect();
            let c = v.len() as f64;
            let mu = v.iter().sum::<f64>() / c;
            let s = (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (c - 1.0)).sqrt();
            assert!(
                (mean[j] - mu).abs() <= 1e-15 * mu.abs(),
                "mean[{j}]: {} vs {mu}",
                mean[j]
            );
            assert!(
                (sd[j] - s).abs() <= 1e-15 * s.abs(),
                "sd[{j}]: {} vs {s}",
                sd[j]
            );
        }
    }
}

#[cfg(test)]
mod tests_calibration {
    use super::*;
    use crate::test_support::synth;
    use faer::{Col, Mat};
    use rand::RngExt;
    use rand::SeedableRng;

    /// Planted sparse signal in feature 0 with positive sign.
    fn synth_h1_signed(n: usize, d: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| 2.0 * x[(i, 0)] + 0.3 * rng.random_range(-1.0..1.0));
        (x, y)
    }

    #[test]
    fn h0_mean_perm_close_to_zero() {
        // Under H0, β under permuted y has population mean 0; sampling SD scales like 1/√B.
        let (x, y) = synth(80, 5, 0, 0.0, 1);
        let opts = PermNullOpts {
            n_perm: 1000,
            return_perm_matrix: false,
            pre_standardized: false,
        };
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, None, opts, Some(7)).unwrap();
        // Tolerance: 5σ band around 0 with σ ≈ sd/√B. `sd[j]/√1000 * 5` ~ generous.
        for j in 0..5 {
            let band = 5.0 * out.beta_perm_sd[j] / (1000.0_f64).sqrt();
            assert!(
                out.beta_perm_mean[j].abs() < band,
                "mean_perm[{j}] = {} exceeds band {}",
                out.beta_perm_mean[j],
                band,
            );
        }
    }

    #[test]
    fn h0_uncorrected_fpr_close_to_alpha() {
        // Average across many features and a few seeds: fraction of |z| > 1.96
        // should land near 0.05. Use a loose Monte-Carlo band (D × n_seeds is small,
        // so the test is loose but should catch order-of-magnitude regressions).
        let mut total = 0_usize;
        let mut rejects = 0_usize;
        for seed in 0..3_u64 {
            let (x, y) = synth(80, 30, 0, 0.0, seed * 17 + 3);
            let opts = PermNullOpts {
                n_perm: 500,
                return_perm_matrix: false,
                pre_standardized: false,
            };
            let out = pls1_perm_null(
                x.as_ref(),
                y.as_ref(),
                1,
                None,
                None,
                opts,
                Some(seed * 11 + 7),
            )
            .unwrap();
            for &z in &out.beta_perm_z {
                if z.is_finite() {
                    total += 1;
                    if z.abs() > 1.96 {
                        rejects += 1;
                    }
                }
            }
        }
        let fpr = rejects as f64 / total as f64;
        // σ ≈ √(0.05·0.95/total) ≈ 0.023 at total = 90, so [0.01, 0.15] spans
        // about −1.7σ to +4.4σ around 0.05. Loose but informative.
        assert!(
            (0.01..=0.15).contains(&fpr),
            "FPR={fpr} (rejects={rejects} / total={total}) outside [0.01, 0.15]",
        );
    }

    #[test]
    fn h1_signed_signal_recovered() {
        let (x, y) = synth_h1_signed(150, 8, 23);
        let opts = PermNullOpts {
            n_perm: 500,
            return_perm_matrix: false,
            pre_standardized: false,
        };
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, None, opts, Some(31)).unwrap();
        // Feature 0 carries the planted signal with positive sign.
        assert!(
            out.beta_perm_z[0] > 0.0,
            "z[0] = {} not positive",
            out.beta_perm_z[0]
        );
        let abs_z: Vec<f64> = out.beta_perm_z.iter().map(|z| z.abs()).collect();
        let max_other = abs_z[1..].iter().copied().fold(0.0_f64, f64::max);
        assert!(
            abs_z[0] > max_other,
            "|z[0]|={} not larger than max |z[1..]|={}",
            abs_z[0],
            max_other,
        );
    }

    /// y on the dominant factor of a collinear block (columns 0..10) and 10
    /// independent columns. A β pivot puts the permuted fits' weaker
    /// shrinkage into the null spread and loses the block (`docs/python/
    /// results.md`, `PermNullResult`); the k = 1 null of `X_j'y` gives large
    /// block z, and N(0, 1)-sized z on the independent columns.
    #[test]
    #[allow(clippy::many_single_char_names)] // n, d, f, x, y, z are the math names
    fn k1_z_keeps_power_on_a_dominant_factor() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(41);
        let (n, d) = (200_usize, 20_usize);
        let f = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let x = Mat::<f64>::from_fn(n, d, |i, j| {
            let e = rng.random_range(-1.0..1.0);
            if j < 10 { 2.0 * f[i] + e } else { e }
        });
        let y = Col::<f64>::from_fn(n, |i| f[i] + 2.0 * rng.random_range(-1.0..1.0));
        let opts = PermNullOpts {
            n_perm: 500,
            return_perm_matrix: false,
            pre_standardized: false,
        };
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 1, None, None, opts, Some(43)).unwrap();
        let z = &out.beta_perm_z;
        assert!(z[..10].iter().all(|v| *v > 3.0), "block z {:?}", &z[..10]);
        assert!(
            z[10..].iter().all(|v| v.abs() < 3.5),
            "independent z {:?}",
            &z[10..]
        );
        // Permuted rows center on 0: about a 5σ band, σ = sd/√B.
        assert!(
            out.beta_perm_mean
                .iter()
                .zip(&out.beta_perm_sd)
                .all(|(m, s)| m.abs() < 0.25 * s)
        );
    }
}

#[cfg(test)]
mod tests_validation {
    use super::*;
    use faer::{Col, Mat};

    #[test]
    fn rejects_dim_mismatch() {
        let x = Mat::<f64>::zeros(10, 5);
        let y = Col::<f64>::zeros(9);
        let opts = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
        };
        let err = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, None, opts, Some(1)).unwrap_err();
        assert_eq!(err.code(), "dimension_mismatch");
    }

    #[test]
    fn rejects_k_exceeds_max() {
        let x = Mat::<f64>::zeros(20, 4);
        let y = Col::<f64>::zeros(20);
        let opts = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
        };
        let err = pls1_perm_null(x.as_ref(), y.as_ref(), 5, None, None, opts, Some(1)).unwrap_err();
        assert_eq!(err.code(), "k_exceeds_max");
    }

    #[test]
    fn rejects_low_n_perm() {
        // `opts.validate(k)` runs first, so the all-zero X is never reached.
        let x = Mat::<f64>::zeros(20, 4);
        let y = Col::<f64>::zeros(20);
        let opts = PermNullOpts {
            n_perm: 50,
            return_perm_matrix: false,
            pre_standardized: false,
        };
        let err = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, None, opts, Some(1)).unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(format!("{err}").contains("n_perm must be"), "{err}");
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests_nspace {
    use super::*;
    use crate::dual_route::{K_DUAL_MAX, nspace_blocks_built};
    use crate::linalg::{standardize, standardize1};
    use crate::signal_test::with_gram_routes_disabled;
    use crate::test_support::{assert_rows_close, assert_summaries_close, bits, synth};
    use faer::{Col, Mat};

    fn opts() -> PermNullOpts {
        PermNullOpts {
            n_perm: 100,
            return_perm_matrix: true,
            pre_standardized: false,
        }
    }

    fn run(x: &Mat<f64>, y: &Col<f64>, k: usize, o: PermNullOpts) -> PermNullOutput {
        pls1_perm_null(x.as_ref(), y.as_ref(), k, None, None, o, Some(17)).expect("perm_null")
    }

    #[test]
    fn the_public_call_builds_one_nspace_block() {
        let (x, y) = synth(40, 2000, 1, 1.0, 1);
        let before = nspace_blocks_built();
        let _ = run(&x, &y, 2, opts());
        assert_eq!(
            nspace_blocks_built(),
            before + 1,
            "the n-space route must run"
        );
        let before = nspace_blocks_built();
        let _ = with_gram_routes_disabled(|| run(&x, &y, 2, opts()));
        assert_eq!(nspace_blocks_built(), before, "override: primal only");
        let w = Col::<f64>::from_fn(40, |i| 0.5 + (i % 3) as f64 * 0.5);
        let before = nspace_blocks_built();
        let _ = pls1_perm_null(
            x.as_ref(),
            y.as_ref(),
            2,
            Some(w.as_ref()),
            None,
            opts(),
            Some(17),
        )
        .expect("weighted perm_null");
        assert_eq!(nspace_blocks_built(), before, "weighted input stays primal");
    }

    #[test]
    fn every_replicate_matches_the_primal_route() {
        // Closeness alone would pass if the n-space route fell back on every
        // row and returned the primal answer, so the bits must also differ
        // from the primal route's somewhere: the n-space kernel must run.
        let mut differing = 0usize;
        for k in 1..=K_DUAL_MAX {
            let (x, y) = synth(40, 2000, 1, 0.5, 20 + k as u64);
            let gram = run(&x, &y, k, opts());
            let primal = with_gram_routes_disabled(|| run(&x, &y, k, opts()));
            assert_rows_close(
                gram.beta_perm_matrix.as_deref().expect("matrix"),
                primal.beta_perm_matrix.as_deref().expect("matrix"),
                gram.beta_ref.len(),
                &format!("k={k}"),
            );
            assert_summaries_close(&gram, &primal, &format!("k={k}"));
            assert_eq!(
                bits(&gram.beta_ref),
                bits(&primal.beta_ref),
                "the observed fit stays primal"
            );
            if bits(gram.beta_perm_matrix.as_deref().expect("matrix"))
                != bits(primal.beta_perm_matrix.as_deref().expect("matrix"))
            {
                differing += 1;
            }
        }
        assert!(
            differing > 0,
            "no replicate's bits differed from the primal route: the n-space \
             route may be falling back on every row"
        );
    }

    #[test]
    fn tiny_scale_pre_standardized_x_matches_the_primal_route() {
        // X at 1e-8 and below is divided by a power of two first, so the
        // kernel's absolute floors are not reached there; 1e-4 and 1e-6 stay
        // inside the band (the standardized X has entries of a few units at most).
        let (x, y) = synth(40, 2000, 1, 1.0, 3);
        let (xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let o = PermNullOpts {
            pre_standardized: true,
            ..opts()
        };
        for s in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
            let xt = Mat::<f64>::from_fn(40, 2000, |i, j| xs[(i, j)] * s);
            let gram = run(&xt, &ys, 2, o);
            let primal = with_gram_routes_disabled(|| run(&xt, &ys, 2, o));
            assert_rows_close(
                gram.beta_perm_matrix.as_deref().expect("matrix"),
                primal.beta_perm_matrix.as_deref().expect("matrix"),
                gram.beta_ref.len(),
                &format!("scale {s:e}"),
            );
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pre_standardized_outcomes_of_extreme_magnitude_match_the_primal_route() {
        // Squares of z underflow (1e-160) or overflow (1e160), and below
        // about 1e-16 the absolute exit `‖X_a'y_a‖ < 1e-14` decides the
        // components; the kernel divides z by a power of two first and
        // applies that exit on the scale of z, so every row is the primal's
        // to `1e-10` of its own largest entry, and a zero row stays zero.
        let (x, y) = synth(40, 2000, 1, 1.0, 4);
        let (xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let o = PermNullOpts {
            pre_standardized: true,
            ..opts()
        };
        // z does not depend on the scale of y wherever the fit itself is
        // scale-free. Below about 1e-16 the fit's absolute exit stops at zero
        // components and every β is 0.
        let z1 = run(&xs, &ys, 2, o).beta_perm_z;
        let z_scale = z1.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        for m in [1e-15, 1e-12, 1e-8, 1e-4, 1e120] {
            let yt = Col::<f64>::from_fn(40, |i| ys[i] * m);
            let zm = run(&xs, &yt, 2, o).beta_perm_z;
            for (j, (a, b)) in zm.iter().zip(&z1).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() <= 1e-10 * z_scale,
                    "|y| ~ {m:e}, z[{j}] = {a:e} against {b:e} at magnitude 1"
                );
            }
        }
        for m in [1e-160, 1e-120, 1e-20, 1e-16, 1e120, 1e160] {
            let yt = Col::<f64>::from_fn(40, |i| ys[i] * m);
            let gram = run(&xs, &yt, 2, o);
            let primal = with_gram_routes_disabled(|| run(&xs, &yt, 2, o));
            let d = gram.beta_ref.len();
            let (a, b) = (
                gram.beta_perm_matrix.as_deref().expect("matrix"),
                primal.beta_perm_matrix.as_deref().expect("matrix"),
            );
            for (r, (ra, rb)) in a.chunks(d).zip(b.chunks(d)).enumerate() {
                let scale = rb.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
                let diff = ra
                    .iter()
                    .zip(rb)
                    .fold(0.0_f64, |s, (u, v)| s.max((u - v).abs()));
                assert!(
                    diff <= 1e-10 * scale,
                    "|y| ~ {m:e}, row {r}: max |Δ| {diff:e} against row scale {scale:e}"
                );
            }
        }
    }

    #[test]
    fn null_moments_hold_at_extreme_outcome_magnitudes() {
        // β rows near 1e158 square past the f64 range; the moments are
        // formed on rows divided by a power of two, so the SD stays finite
        // and z is the one at magnitude 1. (Near 1e-160 the fit's absolute
        // exit stops at zero components before any row is formed.)
        let (x, y) = synth(40, 2000, 1, 1.0, 4);
        let (xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let o = PermNullOpts {
            pre_standardized: true,
            ..opts()
        };
        let z1 = run(&xs, &ys, 2, o).beta_perm_z;
        let z_scale = z1.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        let yt = Col::<f64>::from_fn(40, |i| ys[i] * 1e160);
        let out = run(&xs, &yt, 2, o);
        for (j, (a, b)) in out.beta_perm_z.iter().zip(&z1).enumerate() {
            let sd = out.beta_perm_sd[j];
            assert!(sd.is_finite() && sd > 0.0, "sd[{j}] = {sd:e}");
            assert!(
                (a - b).abs() <= 1e-10 * z_scale,
                "z[{j}] = {a:e} against {b:e} at magnitude 1"
            );
        }
    }

    #[test]
    fn a_zero_column_has_a_nan_z() {
        let (mut x, y) = synth(40, 6, 1, 1.0, 4);
        for i in 0..40 {
            x[(i, 2)] = 0.0;
        }
        let z = run(&x, &y, 2, opts()).beta_perm_z;
        for (j, v) in z.iter().enumerate() {
            let ok = if j == 2 { v.is_nan() } else { v.is_finite() };
            assert!(ok, "z[{j}] = {v:e}");
        }
    }

    #[test]
    fn a_constant_column_has_a_nan_z_under_pre_standardized() {
        // The pre-standardized column stays constant and nonzero, so its β,
        // null mean and null SD are rounding noise; z is NaN as on the
        // default path, and every other column keeps the `signed_z` value.
        let (x, y) = synth(40, 6, 1, 1.0, 4);
        let (mut xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        for i in 0..40 {
            xs[(i, 2)] = 1.0;
        }
        let o = PermNullOpts {
            pre_standardized: true,
            ..opts()
        };
        let out = run(&xs, &ys, 2, o);
        let want = signed_z(&out.beta_ref, &out.beta_perm_mean, &out.beta_perm_sd);
        for (j, (v, w)) in out.beta_perm_z.iter().zip(&want).enumerate() {
            if j == 2 {
                assert!(v.is_nan(), "z[{j}] = {v:e}");
            } else {
                assert!(
                    v.is_finite() && v.to_bits() == w.to_bits(),
                    "z[{j}] = {v:e}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests_gram_p {
    use super::*;
    use crate::signal_test::{ReplicateRoute, with_gram_routes_disabled};
    use crate::test_support::{
        assert_rows_close, assert_summaries_close, bits, gram_p_weights, linear_y, unif,
    };
    use faer::{Col, Mat};

    #[test]
    fn perm_null_gram_route_is_invisible() {
        let x = unif(2000, 50, 75);
        let y = linear_y(&x, 2.0, 76);
        let w = gram_p_weights(2000);
        for (label, wopt, k) in [("dense", None, 3usize), ("weighted", Some(&w), 2usize)] {
            assert_eq!(
                perm_null_route(2000, 50, 300, k, wopt.is_some(), true)
                    .unwrap()
                    .route,
                ReplicateRoute::GramP,
                "{label}: premise"
            );
            let opts = PermNullOpts {
                n_perm: 300,
                return_perm_matrix: true,
                pre_standardized: false,
            };
            let call = || {
                pls1_perm_null(
                    x.as_ref(),
                    y.as_ref(),
                    k,
                    wopt.map(Col::as_ref),
                    None,
                    opts,
                    Some(77),
                )
                .expect("perm_null")
            };
            let g = call();
            let xr = with_gram_routes_disabled(call);
            // The observed fit stays on the Primal route on both runs.
            assert!(
                g.beta_ref
                    .iter()
                    .zip(&xr.beta_ref)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_summaries_close(&g, &xr, label);
            let (gm, xm) = (
                g.beta_perm_matrix.as_deref().expect("matrix"),
                xr.beta_perm_matrix.as_deref().expect("matrix"),
            );
            assert_rows_close(gm, xm, 50, "beta_perm_matrix");
            // The public call must actually run the Gram route: some row's
            // bits differ from the Primal route's.
            assert_ne!(
                bits(gm),
                bits(xm),
                "{label}: the public call ran no Gram row"
            );
        }
    }

    #[test]
    fn tiny_scale_pre_standardized_x_matches_the_primal_route() {
        // At 1e-8, X is divided by a power of two first, so `t't` is not
        // near the kernel's absolute exit of 1e-14; at 1e-6 X stays inside
        // the band (the standardized uniform X has entries up to about 1.7). The rows
        // are the Primal route's to the row rule (beta scales as 1/scale).
        let x = unif(2000, 50, 78);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize1(linear_y(&x, 1.0, 79).as_ref());
        for scale in [1e-6, 1e-8] {
            let xt = Mat::<f64>::from_fn(2000, 50, |i, j| xs[(i, j)] * scale);
            assert_eq!(
                perm_null_route(2000, 50, 300, 2, false, true)
                    .unwrap()
                    .route,
                ReplicateRoute::GramP,
                "premise"
            );
            let opts = PermNullOpts {
                n_perm: 300,
                return_perm_matrix: true,
                pre_standardized: true,
            };
            let call = || {
                pls1_perm_null(xt.as_ref(), ys.as_ref(), 2, None, None, opts, Some(80))
                    .expect("perm_null")
            };
            let g = call();
            let xr = with_gram_routes_disabled(call);
            assert_rows_close(
                &g.beta_perm_matrix.expect("matrix"),
                &xr.beta_perm_matrix.expect("matrix"),
                50,
                &format!("scale {scale:e}"),
            );
        }
    }

    /// With covariates, `beta_ref` is the covariate fit's `coef`, the
    /// result records the kept covariates, and the null rows are the
    /// engine's on the adjusted data with the Freedman–Lane map, bit for
    /// bit. The engine reads its null columns through the map: one
    /// that writes `ỹ` into every null column makes every null row the
    /// observed fit, so `beta_perm_sd` is 0 to rounding.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn perm_null_with_covariates() {
        use crate::fit::{FitOpts, fit_row_scale, pls1_fit, scale_rows};
        use crate::test_support::{assert_close_rel, synth, test_weights, unif};
        use faer::{Col, Mat};
        let n = 60;
        let (x, y) = synth(n, 8, 2, 1.0, 81);
        let z = unif(n, 2, 82);
        let opts = PermNullOpts {
            n_perm: 100,
            ..PermNullOpts::default()
        };
        for (k, w) in [(1, None), (2, None), (2, Some(test_weights(n)))] {
            let wr = w.as_ref().map(Col::as_ref);
            let out = pls1_perm_null(
                x.as_ref(),
                y.as_ref(),
                k,
                wr,
                Some(z.as_ref()),
                opts,
                Some(5),
            )
            .unwrap();
            let fit = pls1_fit(
                x.as_ref(),
                y.as_ref(),
                k,
                wr,
                Some(z.as_ref()),
                FitOpts::default(),
            )
            .unwrap();
            let beta_ref = Col::<f64>::from_fn(8, |j| out.beta_ref[j]);
            assert_close_rel(beta_ref.as_mat(), fit.coef.as_mat(), 1e-12, "beta_ref");
            assert_eq!(out.n_covariates, Some(2));

            let (wn, _) = crate::fit::validate_and_normalize_weights(wr, n).unwrap();
            let adj = crate::covariates::adjust(
                x.as_ref(),
                y.as_ref(),
                z.as_ref(),
                wn.as_ref().map(Col::as_ref),
                false,
            )
            .unwrap();
            let sqw = wn.as_ref().map(|w| fit_row_scale(w.as_ref()));
            let xs_fit: Mat<f64> = match &sqw {
                Some(s) => scale_rows(adj.x.as_ref(), s.as_ref()),
                None => adj.x.clone(),
            };
            let engine = |map: Option<crate::resample::NullMap<'_>>| {
                run_engine(
                    xs_fit.as_ref(),
                    adj.y.as_ref(),
                    k,
                    sqw.as_ref().map(Col::as_ref),
                    out.beta_ref.clone(),
                    out.n_eff,
                    opts,
                    5,
                    map,
                    perm_null_plan(n, 8, opts.n_perm, k, sqw.is_some(), false).unwrap(),
                )
            };
            let fl = |_: usize, m: faer::MatMut<'_, f64>| adj.basis.residualize(m);
            assert_eq!(engine(Some(&fl)).beta_perm_sd, out.beta_perm_sd, "k = {k}");
            let fill =
                |_: usize, mut m: faer::MatMut<'_, f64>| m.copy_from(adj.y.as_ref().as_mat());
            let scale = out.beta_ref.iter().fold(0.0_f64, |m, b| m.max(b.abs()));
            let sd = engine(Some(&fill)).beta_perm_sd;
            assert!(sd.iter().all(|s| *s <= 1e-12 * scale), "k = {k}: {sd:?}");
        }
    }
}
