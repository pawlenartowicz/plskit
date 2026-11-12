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
use crate::signal_test::ReplicateRoute;

/// Tuning knobs for `pls1_perm_null`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct PermNullOpts {
    /// Number of permutations. Must be ≥ 100 (per-voxel-z noise floor).
    pub n_perm: usize,
    /// Whether to return the full `(n_perm, D)` β-matrix on the output.
    /// The b×d buffer is always materialized — the two-pass reduction over it
    /// is the byte-exact reduction path (pinned by `streaming_matches_retained_byte_exact`),
    /// so no true single-pass Welford accumulator exists; this flag only
    /// controls whether that buffer is handed back to the caller.
    pub return_perm_matrix: bool,
    /// Caller asserts X is already column-standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
}

impl Default for PermNullOpts {
    fn default() -> Self {
        Self {
            n_perm: 1000,
            return_perm_matrix: false,
            pre_standardized: false,
            verbose: false,
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
    /// `beta_perm_sd`, `beta_perm_z` and `beta_perm_matrix`.
    pub beta_perm_mean: Vec<f64>,
    /// SD of β under permuted y. Length D.
    pub beta_perm_sd: Vec<f64>,
    /// Signed per-voxel z = β_ref / β_perm_sd. NaN where SD ≈ 0. Length D.
    pub beta_perm_z: Vec<f64>,
    /// Optional `(n_perm, D)` β matrix in row-major layout (length n_perm·D).
    /// Some when `opts.return_perm_matrix == true`.
    pub beta_perm_matrix: Option<Vec<f64>>,
    /// `rank([1, Z]) − 1`, the covariates kept after aliased columns are
    /// dropped. `None` without covariates.
    pub n_covariates: Option<usize>,
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
#[allow(clippy::needless_pass_by_value, clippy::many_single_char_names)]
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
        FitOpts, KSpec, fit_row_scale, pls1_fit_impl, scale_rows, validate_weights_for_k,
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

    let (w_norm, n_eff_val) = validate_weights_for_k(weights, n, k)?;
    let wref = w_norm.as_ref().map(Col::as_ref);

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
    // replaced by their residuals on `[1, Z]` (`covariates::adjust`, as
    // `pls1_fit` does), so `beta_ref` is the covariate fit's `coef`.
    let mut basis = None;
    let (xs_owned, ys_owned): (Option<Mat<f64>>, Col<f64>) = if let Some(z) = covariates {
        let adj = crate::covariates::adjust(x, y, z, wref, opts.pre_standardized)?;
        adj.basis.check_df(k)?;
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
    let xs: MatRef<'_, f64> = xs_owned.as_ref().map_or(x, Mat::as_ref);
    let ys = ys_owned.as_ref();

    // Reference fit on full standardized data.
    let fit_ref = pls1_fit_impl(
        xs,
        ys,
        KSpec::Fixed(k),
        wref,
        None,
        FitOpts {
            pre_standardized: true,
            // check_n_eff: false — n_eff was already validated at the top-level
            // entry; truncation here just yields β at k_used (fit at what exists).
            check_n_eff: false,
            ..FitOpts::default()
        },
    )?;
    let beta_ref: Vec<f64> = (0..d).map(|j| fit_ref.beta[j]).collect();

    // Weights stay tied to rows under every permutation, so the √w row
    // scaling `pls1_fit` applies to X̃ is the same matrix in every replicate:
    // build it once. The factor is `pls1_fit`'s own (it renormalizes the
    // weights it is handed), so it is √(normalize_weights(w')), not √w'.
    // Once the scaled copy exists nothing reads the standardized one, so it
    // is freed here: a single n×d buffer lives through the permutation loop.
    let sqw: Option<Col<f64>> = wref.map(fit_row_scale);
    let xs_fit_owned: Option<Mat<f64>> = match sqw.as_ref() {
        Some(s) => {
            let scaled = scale_rows(xs, s.as_ref());
            drop(xs_owned);
            Some(scaled)
        }
        None => xs_owned,
    };
    let xs_fit: MatRef<'_, f64> = xs_fit_owned.as_ref().map_or(x, Mat::as_ref);

    let (seed_used, mut rng) = crate::rng::resolve_seed(seed)?;

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
        &mut rng,
    )?;
    out.n_covariates = basis
        .as_ref()
        .map(crate::covariates::CovariateBasis::n_covariates);
    Ok(out)
}

// The row driver of the permutation null: the route's block is built once,
// then one row per permutation is written in place into the single
// row-major (B, D) buffer the two-pass reduction reads (no single-pass
// Welford path exists; `streaming_matches_retained_byte_exact` pins the two
// outputs equal). `opts.return_perm_matrix` only decides whether that buffer
// is returned. Returns Result to leave room for per-permutation hard
// failures in future revisions.
#[allow(
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    clippy::many_single_char_names
)]
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
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<PermNullOutput> {
    debug_assert!(k <= xs_fit.ncols(), "k <= p is the caller's precondition");
    let n = xs_fit.nrows();
    let d = xs_fit.ncols();
    let b = opts.n_perm;
    // The null targets: column c ≥ 1 of `cols` is `ys` permuted by
    // `permutation_from_seed(n, seeds[c - 1])`, then `null_map`, the columns
    // every route takes, from one child seed per permutation.
    let seeds = crate::rng::child_seeds(rng, b);
    let cols = crate::resample::Columns {
        y: ys.as_mat(),
        seeds: &seeds,
        null_map,
    };
    // The fit target X̃'s rows see: column `c` of `cols`, `√w`-scaled when weighted.
    let target = |c: usize| {
        let y = cols.column(c);
        match sqw {
            Some(s) => crate::fit::scale_col(y.as_ref(), s),
            None => y,
        }
    };

    // One row per permutation, written in place into the single row-major
    // (B, D) buffer; a failed permutation leaves a NaN row (fail-soft), the
    // whole row overwritten whatever the unit wrote before failing. This
    // buffer is the dominant allocation (~0.8–8 GB at D = 1e5–1e6, B ≥ 1e3,
    // the fMRI target scale), held once: a deliberate
    // determinism-over-memory trade, since no streaming Welford accumulator
    // exists.
    let flat = if k == 1 {
        // At k = 1, β = c·X̃'ỹ with c = ỹ'Kỹ / ỹ'K²ỹ (K = X̃X̃'). A permuted y
        // does not line up with the top of K's spectrum the way a real one
        // can, so its c is larger and the null spread of β is too wide for
        // the observed scale: β is not pivotal. Each row holds c at its
        // observed value, `c_ref·X̃'ỹ_p`, so the null is that of X_j'y, the
        // same null as β_j = 0 at k = 1.
        let s_ref = crate::linalg::mat_vec(xs_fit.transpose(), target(0).as_ref(), faer::Par::Seq);
        let ss: f64 = (0..d).map(|j| s_ref[j] * s_ref[j]).sum();
        let c_ref = if ss > 0.0 {
            (0..d).map(|j| beta_ref[j] * s_ref[j]).sum::<f64>() / ss
        } else {
            0.0
        };
        crate::resample::parallel_fill_rows(b, d, |i, row| {
            let s =
                crate::linalg::mat_vec(xs_fit.transpose(), target(i + 1).as_ref(), faer::Par::Seq);
            for (j, v) in row.iter_mut().enumerate() {
                *v = c_ref * s[j];
            }
        })
    } else {
        // Route choice: `perm_null_route` owns the rule. Decided here, on the
        // calling thread, before any parallel work.
        let route = perm_null_route(n, d, b, k, sqw.is_some());
        let block = perm_block(route, xs_fit, crate::fit::par_fixed());
        crate::resample::parallel_fill_rows(b, d, |i, row| {
            if perm_row(&block, xs_fit, sqw, cols.column(i + 1).as_ref(), k, row).is_err() {
                row.fill(f64::NAN);
            }
        })
    };

    // Two-pass per-column reduction.
    let (beta_perm_mean, beta_perm_sd) = reduce_two_pass(&flat, b, d);
    let beta_perm_z = signed_z(&beta_ref, &beta_perm_sd);

    Ok(PermNullOutput {
        n_perm: b,
        k,
        seed: seed_used,
        n_eff: n_eff_val,
        beta_ref,
        beta_perm_mean,
        beta_perm_sd,
        beta_perm_z,
        beta_perm_matrix: if opts.return_perm_matrix {
            Some(flat)
        } else {
            None
        },
        n_covariates: None,
    })
}

/// Two-pass mean / SD over a flat row-major (B, D) buffer.
/// Skips NaN rows in both numerator and denominator (failed permutations).
fn reduce_two_pass(flat: &[f64], b: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0_f64; d];
    let mut count = vec![0_usize; d];
    for bi in 0..b {
        let off = bi * d;
        for j in 0..d {
            let v = flat[off + j];
            if v.is_finite() {
                mean[j] += v;
                count[j] += 1;
            }
        }
    }
    for j in 0..d {
        if count[j] > 0 {
            mean[j] /= count[j] as f64;
        }
    }
    let mut m2 = vec![0.0_f64; d];
    for bi in 0..b {
        let off = bi * d;
        for j in 0..d {
            let v = flat[off + j];
            if v.is_finite() {
                let dv = v - mean[j];
                m2[j] += dv * dv;
            }
        }
    }
    let sd: Vec<f64> = (0..d)
        .map(|j| {
            if count[j] > 1 {
                (m2[j] / (count[j] - 1) as f64).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    (mean, sd)
}

/// β_ref / β_perm_sd, NaN-guarded with ε = √f64::EPSILON.
#[allow(clippy::doc_markdown)]
fn signed_z(beta_ref: &[f64], sd: &[f64]) -> Vec<f64> {
    let eps = f64::EPSILON.sqrt();
    beta_ref
        .iter()
        .zip(sd.iter())
        .map(|(b, s)| if *s > eps { b / s } else { f64::NAN })
        .collect()
}

/// `pls1_perm_null`'s selector at `k ≥ 2` (`k = 1` has a closed form and no
/// route), called once per call on the calling thread before any parallel
/// work. `Nspace` for dense, unweighted input the n-space rule admits
/// (`dual_route::nspace_eligible_perm_null`), unless
/// `gram_routes_disabled()`. `GramP` when `gram_p::gram_p_eligible` admits
/// the block (tried after `Nspace`: in the overlap band
/// `n_tr < p < 1.54·n_tr` the n-space route is cheaper), unless
/// `gram_routes_disabled()`; its replicate count is `n_perm`, since the
/// observed fit stays on the Primal route. `Primal` otherwise.
pub(crate) fn perm_null_route(
    n: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    weighted: bool,
) -> ReplicateRoute {
    if crate::signal_test::gram_routes_disabled() {
        return ReplicateRoute::Primal;
    }
    if crate::dual_route::nspace_eligible_perm_null(n, p, n_perm, k, !weighted) {
        ReplicateRoute::Nspace
    } else if crate::gram_p::gram_p_eligible(n, p, n_perm, k, None) {
        ReplicateRoute::GramP
    } else {
        ReplicateRoute::Primal
    }
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
        ReplicateRoute::Special | ReplicateRoute::Primal => PermBlock::Primal {
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
///
/// # Errors
/// What the kernel returns; `run_engine` then writes a NaN row.
fn perm_row(
    block: &PermBlock,
    xs_fit: MatRef<'_, f64>,
    sqw: Option<ColRef<'_, f64>>,
    y_null: ColRef<'_, f64>,
    k: usize,
    out: &mut [f64],
) -> PlsKitResult<()> {
    use crate::fit::{ParChoice, pls1_fit_prepared_fro, scale_col};
    match block {
        &PermBlock::Primal { x_fro } => {
            // Seq inside the per-permutation worker: outer Rayon owns the threadpool.
            let fit = match sqw {
                Some(s) => {
                    let ys_fit = scale_col(y_null, s);
                    pls1_fit_prepared_fro(xs_fit, ys_fit.as_ref(), k, None, ParChoice::Seq, x_fro)?
                }
                None => pls1_fit_prepared_fro(xs_fit, y_null, k, None, ParChoice::Seq, x_fro)?,
            };
            for (j, v) in out.iter_mut().enumerate() {
                *v = fit.coef[j];
            }
            Ok(())
        }
        PermBlock::Nspace(gram) => {
            debug_assert!(k >= 1 && sqw.is_none());
            crate::dual_route::nspace_perm_row(gram, xs_fit, y_null, k, out);
            Ok(())
        }
        PermBlock::GramP(gram) => {
            debug_assert!(k >= 1);
            let coef = perm_row_coef(gram, xs_fit, sqw, y_null, k);
            for (j, v) in out.iter_mut().enumerate() {
                *v = coef[j];
            }
            Ok(())
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
                    verbose: false,
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
mod tests_engine_streaming {
    use super::*;
    use crate::test_support::{bits, synth};

    #[test]
    fn streaming_matches_retained_byte_exact() {
        // Both paths fill the same flat buffer (parallel_fill_rows) and
        // run the two-pass reduce over it; only retained returns the matrix.
        let (x, y) = synth(100, 5, 2, 4.0, 42);
        let opts_retained = PermNullOpts {
            n_perm: 200,
            return_perm_matrix: true,
            pre_standardized: false,
            verbose: false,
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
            verbose: false,
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
                verbose: false,
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
            verbose: false,
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
            verbose: false,
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
            verbose: false,
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
            verbose: false,
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
            verbose: false,
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
            verbose: false,
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
        // The one input that reaches the kernel's absolute floors.
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
                perm_null_route(2000, 50, 300, k, wopt.is_some()),
                ReplicateRoute::GramP,
                "{label}: premise"
            );
            let opts = PermNullOpts {
                n_perm: 300,
                return_perm_matrix: true,
                pre_standardized: false,
                verbose: false,
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
        // At 1e-8, t't = ‖Xs·w‖² is about 2e-13 (2000 rows of unit-SD
        // columns times 1e-16), near the kernel's absolute exit of 1e-14.
        // The rows are the Primal route's to the row rule (beta scales as
        // 1/scale).
        let x = unif(2000, 50, 78);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize1(linear_y(&x, 1.0, 79).as_ref());
        for scale in [1e-6, 1e-8] {
            let xt = Mat::<f64>::from_fn(2000, 50, |i, j| xs[(i, j)] * scale);
            assert_eq!(
                perm_null_route(2000, 50, 300, 2, false),
                ReplicateRoute::GramP,
                "premise"
            );
            let opts = PermNullOpts {
                n_perm: 300,
                return_perm_matrix: true,
                pre_standardized: true,
                verbose: false,
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
        use crate::fit::{FitOpts, KSpec, fit_row_scale, pls1_fit, scale_rows};
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
                KSpec::Fixed(k),
                wr,
                Some(z.as_ref()),
                FitOpts::default(),
            )
            .unwrap();
            let beta_ref = Col::<f64>::from_fn(8, |j| out.beta_ref[j]);
            assert_close_rel(beta_ref.as_mat(), fit.coef.as_mat(), 1e-12, "beta_ref");
            assert_eq!(out.n_covariates, Some(2));

            let (wn, _) = crate::fit::validate_and_normalize_weights(wr, n, k).unwrap();
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
                let (_, mut rng) = crate::rng::resolve_seed(Some(5)).unwrap();
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
                    &mut rng,
                )
                .unwrap()
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
