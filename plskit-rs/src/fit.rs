//! PLS1 fit: the NIPALS PLS1 model, computed by Improved Kernel PLS (one
//! shared component loop, `pls1_component_loop`, with the X backend; see
//! `pls1_kernel`). Public entry points: `pls1_fit`, `spls1_fit`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};

/// How `pls1_fit` decides how many components to extract.
#[derive(Debug, Clone, Copy)]
pub enum KSpec {
    /// Fixed component count requested by the caller.
    Fixed(usize),
}

/// Parallelism strategy for the PLS1 kernel (`pls1_kernel`) called by `pls1_fit`.
///
/// `Auto` (the default) selects per-fit based on problem size:
/// runs sequentially when `n * d * k < 1_000_000` and otherwise on the
/// current Rayon pool (an installed pool, if the call runs inside one),
/// split into a fixed number of pieces (see below). The `1_000_000` threshold reflects the
/// crossover measured on Arrow Lake-H: at smaller sizes faer's matmul
/// dispatch over-eagerly parallelizes and the thread-overhead dominates
/// (~1.8× slowdown observed at `(200, 800, 5)`).
///
/// The parallel arm splits each product into a fixed number of pieces
/// (8), however many threads the pool has, so a fit's bits
/// depend on its inputs and on whether `Auto` chose the parallel arm (a
/// function of the shape), never on the pool size: `RAYON_NUM_THREADS=1`
/// and `RAYON_NUM_THREADS=16` give the same result. `Seq` and the parallel
/// arm of `Auto` can differ from each other in the last bits.
///
/// Resamplers (`pls1_perm_null`, `pls1_rotation_stability`,
/// `pls1_confirmatory_test`, `pls1_find_k_*`) force `Seq` for the inner
/// fits: outer Rayon is already saturating the cores, and nested
/// parallelism would oversubscribe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParChoice {
    /// Decide per fit based on `n * d * k`.
    Auto,
    /// Force sequential execution.
    Seq,
}

/// Knobs for `pls1_fit`.
#[derive(Debug, Clone, Copy)]
pub struct FitOpts {
    /// Skip the centering/scaling step; caller asserts X and y are already standardized.
    ///
    /// **Scale contract.** When `pre_standardized=true`, the PLS1 kernel
    /// uses a fixed `1e-14` absolute threshold on the per-component norms
    /// of `X_a'y_a` and `t = X_a w` (every component must also clear a floor relative
    /// to `‖X‖_F·‖y‖`, which is scale-free). If raw data is scaled below
    /// ~`1e-7` (frobenius norm of `X` < `1e-6`), the loop short-circuits at
    /// the first component and the fit silently returns a zero-beta model. Callers
    /// passing `pre_standardized=true` must ensure the inputs are
    /// genuinely zero-mean / unit-variance (or at least scale-comparable
    /// to that). The default `pre_standardized=false` path absorbs raw
    /// scale automatically and is unaffected by this contract.
    ///
    /// As a guard, when `pre_standardized=true` and `check_n_eff=true`
    /// (the default for top-level public entry points), `pls1_fit`
    /// returns `InvalidInput` if the PLS1 kernel truncates below the requested `k`.
    /// Per-iteration internal callers (CV folds, per-half split fits,
    /// permutation refits, the BIC full-k fit, the sequential deflation
    /// fit) set `check_n_eff=false` and tolerate truncation by design.
    pub pre_standardized: bool,
    /// When true (default), `pls1_fit` errors with `InvalidWeights{reason:"insufficient_effective_n"}`
    /// (non-uniform weights) or `InvalidArgument` (uniform or absent weights)
    /// if `n_eff < k + 1`. Set to false for per-iteration internal calls (CV folds,
    /// bootstrap subsamples) where the upstream accumulator handles degeneracy.
    /// See `_docs/internals/n-eff-check.md`.
    pub check_n_eff: bool,
    /// Parallelism strategy for the PLS1 kernel. See `ParChoice`.
    pub par: ParChoice,
    /// Sparse keep-count (spls1 family plumbing): retain the `keep`
    /// largest-|w| coordinates per component, zero the rest — hard selection
    /// at the keep-th order statistic of |w|; exact ties break by lowest
    /// column index (reproducibility contract). `None` (default) = dense
    /// PLS1. Wrapper surfaces never expose this on dense functions;
    /// call `spls1_fit` instead of setting it directly.
    pub keep: Option<usize>,
}

impl Default for FitOpts {
    fn default() -> Self {
        Self {
            pre_standardized: false,
            check_n_eff: true,
            par: ParChoice::Auto,
            keep: None,
        }
    }
}

/// Number of pieces every parallel faer call in this crate is split into.
///
/// faer's `Par::rayon(0)` (and its global default, which operator `*` and
/// the high-level decompositions read) takes the degree from
/// `rayon::current_num_threads()`, and the degree is part of the
/// arithmetic, not only of the scheduling: a column-major GEMV `X v` sums
/// `degree` partial products over column blocks, in order, so the rounding
/// of `t = X r` changes with the pool size. A fixed degree fixes the split;
/// Rayon then only schedules the pieces, on however many threads it has.
///
/// 8 spreads the kernel GEMVs over the cores of a typical workstation. The
/// column-major GEMV adds its partial vectors in one serial pass (`n` adds
/// per piece), so a much larger degree costs more than it gains. A pool
/// with fewer threads runs the 8 pieces on the threads it has; a larger
/// pool leaves the rest free for that call.
pub(crate) const PAR_DEGREE: usize = 8;

/// The one parallel `faer::Par` of this crate: `Par::Rayon(PAR_DEGREE)`.
/// Never `Par::rayon(0)`, whose degree is the pool size (see
/// [`PAR_DEGREE`]).
pub(crate) fn par_fixed() -> Par {
    Par::rayon(PAR_DEGREE)
}

/// Translate a `ParChoice` into a concrete `faer::Par` for the given problem size.
///
/// `n` is the number of samples and `d` the number of features. `n_sweeps`
/// is the number of `O(n * d)` passes over `X` the caller performs, so that
/// `n * d * n_sweeps` estimates the flop count of the dominant kernel: the
/// requested component count `k` for PLS1 (the PLS1 kernel reads `X`
/// `2k + 1` times; `pls1_fit` passes `k`, the count the `1_000_000`
/// threshold was measured with), and the target count `q` for PLS3 (the `X'Y` and score
/// matmuls are `n * p * q`).
///
/// `ParChoice::Seq` always maps to `Par::Seq`; `ParChoice::Auto` uses
/// [`par_fixed`] when `n * d * n_sweeps >= 1_000_000`, else `Par::Seq`.
/// Saturating arithmetic guards against `usize` overflow on absurd inputs.
pub(crate) fn resolve_par(choice: ParChoice, n: usize, d: usize, n_sweeps: usize) -> Par {
    match choice {
        ParChoice::Seq => Par::Seq,
        ParChoice::Auto => {
            let work = n.saturating_mul(d).saturating_mul(n_sweeps);
            if work >= 1_000_000 {
                par_fixed()
            } else {
                Par::Seq
            }
        }
    }
}

/// Owned PLS1 fit. Fields use long `snake_case` names;
/// the wrapper translates to short Python-facing names at the FFI seam.
#[derive(Debug, Clone)]
pub struct Pls1Model {
    /// X-scores `T`; shape `(n_samples, k_used)`.
    pub t_scores: Mat<f64>,
    /// X-loadings `P`; shape `(n_features, k_used)`.
    pub p_loadings: Mat<f64>,
    /// X-weights `W` (raw NIPALS weights, unit-normed per component); shape `(n_features, k_used)`.
    /// Note: this is raw W, not the modified W* = W·(P'W)^{-1} used to back-solve coefficients — see `pls1_coef_at_k`.
    pub w_star: Mat<f64>,
    /// y-loadings `Q`; shape `(k_used,)`.
    pub q_loadings: Col<f64>,
    /// Regression coefficients in standardized space; shape `(n_features,)`.
    pub coef: Col<f64>,
    /// Regression coefficients back-projected to raw X scale; shape `(n_features,)`.
    pub beta: Col<f64>,
    /// y intercept in raw scale (0 when `pre_standardized=true`).
    pub intercept: f64,
    /// Number of components actually retained (≤ requested `k`).
    pub k_used: usize,
    /// Echoes the caller's `pre_standardized` flag.
    pub pre_standardized: bool,
    /// Resolved (post-normalization) weight vector. `None` when input was uniform
    /// or absent (all-equal weights fit identically to none). Length = `n_samples` when present.
    pub weights: Option<Col<f64>>,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
    /// Resolved sparse keep-count. `None` for dense fits — mirrors the
    /// `weights: Option<Col<f64>>` precedent.
    pub keep: Option<usize>,
}

/// Validate weights and produce `(normalized vector, n_eff, all_uniform_flag)`.
/// Returns `Ok((None, n as f64, true))` when `weights` is `None` **and** when
/// every weight is equal: all-equal weights are no weights (uniform-weight
/// invariance), so every downstream branch on `w_norm.is_some()` takes the
/// unweighted path and the call is bit-identical to one without weights.
/// That includes `n_eff`, which is exactly `n` (Kish's ratio of equal
/// non-unit weights can round an ulp below it).
///
/// # Errors
/// - `InvalidWeights { reason: "length_mismatch" }` if `weights.len() != n`
/// - `NonFiniteInput` for any NaN / infinity
/// - `InvalidWeights { reason: "negative" }` for any `w < 0`
/// - `InvalidWeights { reason: "all_zero" }` if `Σw == 0`
///
/// All-equal means every normalized entry is 1.0 within 1e-12; such weights
/// come back as `None` with `n_eff = n`, exactly like absent weights.
/// Top-level entries that take a component count call
/// `validate_weights_for_k` instead.
pub(crate) fn validate_and_normalize_weights(
    weights: Option<ColRef<'_, f64>>,
    n: usize,
    k_requested: usize,
) -> PlsKitResult<(Option<Col<f64>>, f64)> {
    let Some(w) = weights else {
        #[allow(clippy::cast_precision_loss)]
        return Ok((None, n as f64));
    };

    if w.nrows() != n {
        // Weights length is a weights problem, not a dimension mismatch between X and y.
        // Mirrors the same convention in preprocess.rs — change together.
        return Err(PlsKitError::InvalidWeights {
            reason: "length_mismatch",
        });
    }
    // mirrors preprocess.rs weight-validation loop — change together.
    for i in 0..n {
        if !w[i].is_finite() {
            return Err(PlsKitError::NonFiniteInput);
        }
        if w[i] < 0.0 {
            return Err(PlsKitError::InvalidWeights { reason: "negative" });
        }
    }
    let wn = crate::linalg::normalize_weights(w)
        .ok_or(PlsKitError::InvalidWeights { reason: "all_zero" })?;
    let _ = k_requested; // the n_eff check lives in validate_weights_for_k; see _docs/internals/n-eff-check.md
    if weights_all_equal(wn.as_ref()) {
        #[allow(clippy::cast_precision_loss)]
        return Ok((None, n as f64));
    }
    Ok((Some(wn), crate::linalg::compute_n_eff(w)))
}

/// The all-equal predicate on normalized weights `w'` (mean 1): every
/// entry is 1.0 within 1e-12. Such weights are no weights, and their
/// `n_eff` is exactly `n`; `preprocess` reports the same `n_eff` for them.
pub(crate) fn weights_all_equal(wn: ColRef<'_, f64>) -> bool {
    let max_dev = (0..wn.nrows())
        .map(|i| (wn[i] - 1.0).abs())
        .fold(0.0_f64, f64::max);
    max_dev < 1e-12
}

/// `validate_and_normalize_weights` followed by the `n_eff ≥ k + 1` check:
/// the one call every top-level entry with weights and a component count
/// makes. Returns `(normalized weights, n_eff)`. Whether the weights are "in
/// play" for the error kind is read off the normalized vector, never off
/// the caller's `Option`, so all-equal weights fail exactly as absent ones.
///
/// # Errors
/// Those of `validate_and_normalize_weights`, then
/// `InvalidWeights { reason: "insufficient_effective_n" }` (non-uniform
/// weights) or `InvalidArgument` (uniform or absent weights) when
/// `n_eff < k + 1`.
pub(crate) fn validate_weights_for_k(
    weights: Option<ColRef<'_, f64>>,
    n: usize,
    k: usize,
) -> PlsKitResult<(Option<Col<f64>>, f64)> {
    let (w_norm, n_eff) = validate_and_normalize_weights(weights, n, k)?;
    check_n_eff_for_k(n_eff, k, w_norm.is_some())?;
    Ok((w_norm, n_eff))
}

/// Check that every entry of `x` is finite. Used at top-level public
/// entry points to guarantee the boundary contract; per-iteration inner
/// callers can rely on this having run upstream.
///
/// # Errors
/// `NonFiniteInput` on any NaN or infinity.
pub(crate) fn check_finite_mat(x: MatRef<'_, f64>) -> PlsKitResult<()> {
    let n = x.nrows();
    let d = x.ncols();
    // A row-major view (a C-ordered host array read in place) is swept row
    // by row over contiguous slices; the answer does not depend on the order.
    if x.try_as_col_major().is_none() {
        if let Some(xr) = x.try_as_row_major() {
            for i in 0..n {
                // A non-short-circuiting fold, so the row sweep vectorizes.
                let row_ok = xr
                    .row(i)
                    .as_slice()
                    .iter()
                    .fold(true, |ok, v| ok & v.is_finite());
                if !row_ok {
                    return Err(PlsKitError::NonFiniteInput);
                }
            }
            return Ok(());
        }
    }
    // j-outer/i-inner matches faer's column-major storage (cache-friendly sweep).
    // Mirrors `rotate::mat_is_finite`, which has no row-major branch.
    for j in 0..d {
        for i in 0..n {
            if !x[(i, j)].is_finite() {
                return Err(PlsKitError::NonFiniteInput);
            }
        }
    }
    Ok(())
}

/// Check that every entry of `y` is finite.
///
/// # Errors
/// `NonFiniteInput` on any NaN or infinity.
pub(crate) fn check_finite_col(y: ColRef<'_, f64>) -> PlsKitResult<()> {
    let n = y.nrows();
    for i in 0..n {
        if !y[i].is_finite() {
            return Err(PlsKitError::NonFiniteInput);
        }
    }
    Ok(())
}

/// Check that effective sample size supports the requested number of components.
///
/// When `weighted=true` (non-uniform observation weights), returns
/// `Err(InvalidWeights { reason: "insufficient_effective_n" })` when `n_eff < k + 1`.
/// When `weighted=false` (uniform or absent weights), returns
/// `Err(InvalidArgument)` — no weights are in play, so the failure is a plain
/// data-size problem, not a weights problem (callers branch on `code()`).
/// Reached only through `validate_weights_for_k`, which derives `weighted`
/// from the normalized vector; see `_docs/internals/n-eff-check.md`.
///
/// # Errors
/// `InvalidWeights { reason: "insufficient_effective_n" }` (non-uniform
/// weights) or `InvalidArgument` (uniform or absent weights) when
/// `n_eff < k + 1`.
fn check_n_eff_for_k(n_eff: f64, k: usize, weighted: bool) -> PlsKitResult<()> {
    #[allow(clippy::cast_precision_loss)]
    if n_eff < (k as f64) + 1.0 {
        return Err(if weighted {
            PlsKitError::InvalidWeights {
                reason: "insufficient_effective_n",
            }
        } else {
            PlsKitError::InvalidArgument(format!(
                "insufficient n for k={k}: need n >= k + 1 (got n={n_eff})"
            ))
        });
    }
    Ok(())
}

/// Range-check a keep-count against the dimension it selects within,
/// naming both in the message. `validate_keep` is the PLS1 spelling
/// (`keep` within `n_features`); the sPLS3 sides pass their own names.
///
/// # Errors
/// `InvalidArgument` when `keep == 0` or `keep > dim`.
pub(crate) fn validate_keep_arg(
    keep: usize,
    dim: usize,
    arg: &str,
    dim_name: &str,
) -> PlsKitResult<()> {
    if keep == 0 {
        return Err(PlsKitError::InvalidArgument(format!(
            "{arg} must be >= 1 ({arg}=0 would produce an empty component)"
        )));
    }
    if keep > dim {
        return Err(PlsKitError::InvalidArgument(format!(
            "{arg}={keep} exceeds {dim_name}={dim}"
        )));
    }
    Ok(())
}

/// PLS1 spelling of [`validate_keep_arg`].
///
/// # Errors
/// `InvalidArgument` when `keep == 0` or `keep > n_features`.
pub(crate) fn validate_keep(keep: usize, n_features: usize) -> PlsKitResult<()> {
    validate_keep_arg(keep, n_features, "keep", "n_features")
}

/// The argument checks `pls1_fit` makes after the shapes and before the
/// weights, in its order: finite y, `k ≥ 1`, `k ≤ n_features`, `keep`. A replicate loop that
/// checked its prepared X once (per fold or per split) runs only these per
/// outcome column, and fails with the error `pls1_fit` would return.
///
/// # Errors
/// `NonFiniteInput`, `InvalidArgument` (for `k = 0` or a bad `keep`) or
/// `KExceedsMax`.
pub(crate) fn check_fit_y_and_k(
    n_features: usize,
    y: ColRef<'_, f64>,
    k: usize,
    keep: Option<usize>,
) -> PlsKitResult<()> {
    check_finite_col(y)?;
    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    if k > n_features {
        return Err(PlsKitError::KExceedsMax {
            k,
            k_max: n_features,
        });
    }
    if let Some(kp) = keep {
        validate_keep(kp, n_features)?;
    }
    Ok(())
}

/// Fit a PLS1 regression: the NIPALS PLS1 model, computed by Improved
/// Kernel PLS (`pls1_kernel`).
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - `y`: `(n_samples,)`
/// - `weights`: optional per-observation weights; `None` is equivalent to all-ones.
/// - returns `Pls1Model { t_scores: (n_samples, k_used), p_loadings: (n_features, k_used),
///   w_star: (n_features, k_used), q_loadings: (k_used,), coef: (n_features,),
///   beta: (n_features,), ... }`
///
/// # Errors
/// - `PlsKitError::DimensionMismatch` when `y.nrows() != x.nrows()`
/// - `PlsKitError::InvalidWeights { reason: "length_mismatch" }` when `weights.len() != n`
/// - `PlsKitError::InvalidArgument` when `k == 0`
/// - `PlsKitError::KExceedsMax` when `k > n_features`
/// - `PlsKitError::NonFiniteInput` when X, y, or weights contains NaN/inf
/// - `PlsKitError::InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub fn pls1_fit(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: KSpec,
    weights: Option<ColRef<'_, f64>>,
    opts: FitOpts,
) -> PlsKitResult<Pls1Model> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let KSpec::Fixed(k_requested) = k;
    if y.nrows() != n_samples {
        return Err(PlsKitError::DimensionMismatch {
            x: (n_samples, n_features),
            y: y.nrows(),
        });
    }
    // X's finiteness is read from the standardization moments or the
    // sum-of-squares screen below. When a later check fails first, the exact
    // scan runs before its error is returned, so a non-finite X is still
    // reported ahead of y, k, keep and the weights, as the replicate loops
    // (which check X first) report it.
    // Weights: finite, non-negative, Σw > 0, normalized to mean 1.
    let checked = check_fit_y_and_k(n_features, y, k_requested, opts.keep).and_then(|()| {
        if opts.check_n_eff {
            validate_weights_for_k(weights, n_samples, k_requested)
        } else {
            validate_and_normalize_weights(weights, n_samples, k_requested)
        }
    });
    let (w_norm, n_eff_val) = match checked {
        Ok(v) => v,
        Err(e) => {
            check_finite_mat(x)?;
            return Err(e);
        }
    };
    let wref: Option<ColRef<'_, f64>> = w_norm.as_ref().map(Col::as_ref);

    // √w' row factor. Row-scaling is the Cholesky factor of diag(w'),
    // *not* preprocessing, so it applies even when pre_standardized=true.
    let sqw: Option<Col<f64>> = wref.map(crate::linalg::sqrt_col);

    // Standardization moments, or none when pre_standardized, weighted when
    // weights is Some. The standardized X itself is written only when its
    // columns' means are too large for the implicit products (below).
    let mut fx: Option<crate::linalg::FitX> =
        (!opts.pre_standardized).then(|| crate::linalg::fit_x_moments(x, wref));
    let (ys_owned, y_mean, y_scale) = if opts.pre_standardized {
        (None, 0.0, 1.0)
    } else {
        let (zs, ym, ysc) = crate::linalg::standardize1_weighted(y, wref);
        (Some(zs), ym, ysc)
    };
    let ys_view: ColRef<'_, f64> = match &ys_owned {
        Some(a) => a.as_ref(),
        None => y,
    };

    // A pre-standardized X is still to be row-scaled; the implicit kernel
    // applies √w' itself.
    let x_scaled_owned: Option<Mat<f64>> = match (&fx, &sqw) {
        (None, Some(s)) => Some(scale_rows(x, s.as_ref())),
        _ => None,
    };
    let y_scaled_owned: Option<Col<f64>> = sqw.as_ref().map(|s| scale_col(ys_view, s.as_ref()));

    // The pre-standardized path's kernel input.
    let x_for_nipals: MatRef<'_, f64> = match &x_scaled_owned {
        Some(a) => a.as_ref(),
        None => x,
    };
    let y_for_nipals: ColRef<'_, f64> = match &y_scaled_owned {
        Some(a) => a.as_ref(),
        None => ys_view,
    };

    // X's finiteness and ‖X‖_F (for the kernel's truncation floor) come from
    // the pass already made: a non-finite entry makes its column's mean
    // non-finite, and the pre-standardized sum of squares is non-finite for a
    // non-finite entry or an overflowing square. A non-finite verdict is
    // confirmed by the exact scan, so a finite X is never rejected.
    let x_fro = if let Some(f) = &fx {
        if !f.mean.iter().chain(f.scale.iter()).all(|v| v.is_finite()) {
            check_finite_mat(x)?;
        }
        f.fro
    } else {
        let ss = crate::linalg::sum_of_squares(x_for_nipals);
        if ss.is_finite() {
            ss.sqrt()
        } else {
            check_finite_mat(x)?;
            x_for_nipals.norm_l2()
        }
    };
    let (x_mean, x_scale) = match &fx {
        Some(f) => (f.mean.clone(), f.scale.clone()),
        None => (
            Col::<f64>::zeros(n_features),
            Col::<f64>::from_fn(n_features, |_| 1.0),
        ),
    };

    // Within `IMPLICIT_MAX_MEAN_RATIO` the kernel forms the standardized
    // products from the raw X. Past it, or when the implicit fit cannot decide
    // where to stop, X is standardized into a copy (its layout, rows scaled
    // by √w'), and the fit is the copy's to the bit.
    let implicit_fit = match &fx {
        Some(f) => {
            let ratio = f.max_mean_ratio();
            if ratio > IMPLICIT_MAX_MEAN_RATIO {
                None
            } else {
                pls1_fit_implicit(
                    x,
                    y_for_nipals,
                    crate::linalg::owned_col_slice(&f.mean),
                    crate::linalg::owned_col_slice(&f.scale),
                    sqw.as_ref().map(crate::linalg::owned_col_slice),
                    k_requested,
                    opts.keep,
                    opts.par,
                    x_fro,
                    ratio,
                )
            }
        }
        None => None,
    };
    let fit = match (implicit_fit, fx.as_mut()) {
        (Some(fit), _) => fit,
        (None, Some(f)) => {
            f.materialize(x, sqw.as_ref().map(Col::as_ref));
            pls1_fit_prepared_fro(
                f.xs(),
                y_for_nipals,
                k_requested,
                opts.keep,
                opts.par,
                x_fro,
            )?
        }
        (None, None) => pls1_fit_prepared_fro(
            x_for_nipals,
            y_for_nipals,
            k_requested,
            opts.keep,
            opts.par,
            x_fro,
        )?,
    };
    let k_used = fit.k_used;
    if opts.pre_standardized && opts.check_n_eff && k_used < k_requested {
        return Err(PlsKitError::InvalidInput(format!(
            "pls1_fit(pre_standardized=true) truncated to k_used={k_used} < requested k={k_requested}: \
             NIPALS short-circuited on the {kth} component (norm < 1e-14, or at the \
             rounding floor relative to ‖X‖_F·‖y‖). Either X or y is exhausted (fewer \
             than k informative directions: lower k; at k_used=0, y is orthogonal to X \
             up to rounding), or the inputs \
             violate the pre_standardized scale contract (see `FitOpts::pre_standardized`): \
             re-fit with `pre_standardized=false` to let plskit standardize, or rescale your \
             inputs so that ‖X‖_F ≥ 1e-6.",
            kth = k_used + 1
        )));
    }
    let coef = fit.coef;

    // Back-project to raw scale: beta[j] = coef[j] * y_scale / x_scale[j]
    let beta = if opts.pre_standardized {
        coef.clone()
    } else {
        Col::<f64>::from_fn(n_features, |j| coef[j] * y_scale / x_scale[j])
    };
    let intercept = if opts.pre_standardized {
        0.0
    } else {
        // y_hat_raw = mean_y + sum_j beta_j (x_j - mean_x_j)
        let dot: f64 = (0..n_features).map(|j| beta[j] * x_mean[j]).sum();
        y_mean - dot
    };

    Ok(Pls1Model {
        t_scores: fit.t_scores,
        p_loadings: fit.p_loadings,
        w_star: fit.w_star,
        q_loadings: fit.q_loadings,
        coef,
        beta,
        intercept,
        k_used,
        pre_standardized: opts.pre_standardized,
        weights: w_norm,
        n_eff: n_eff_val,
        keep: opts.keep,
    })
}

/// Sparse PLS1 fit, head of the `spls1_*` family: the PLS1 (NIPALS) model
/// with hard keep-count selection on the weight vector per component (Chun &
/// Keleş 2010 lineage, keep-count formulation), so each latent direction
/// loads on exactly `keep` X variables. Everything downstream of the
/// selection step (rotation, scores, loadings, the update of `X'y`,
/// `coef = W(P'W)⁻¹Q`, raw-scale β, intercept) is the same code as
/// `pls1_fit`; `keep = n_features` reduces bit-exactly to the dense fit.
///
/// `keep` is a scalar broadcast to all `k` components (per-component budget
/// deferred — rule of three). Selection on `w` does not guarantee a nested
/// selection path across components; acceptable for v1 (tune, don't
/// interpret the path).
///
/// # Errors
/// Everything `pls1_fit` returns, plus `InvalidArgument` for `keep == 0`
/// or `keep > n_features`.
pub fn spls1_fit(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: KSpec,
    keep: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: FitOpts,
) -> PlsKitResult<Pls1Model> {
    pls1_fit(
        x,
        y,
        k,
        weights,
        FitOpts {
            keep: Some(keep),
            ..opts
        },
    )
}

/// The kernel-and-coefficient tail of `pls1_fit`, on arrays that are
/// already standardized and already √w-scaled. `pls1_fit` runs it as
/// [`pls1_fit_prepared_fro`] when it does not standardize, or when it
/// standardizes X into a copy (past [`IMPLICIT_MAX_MEAN_RATIO`]); otherwise
/// it runs [`pls1_fit_implicit`] on the raw X, which returns this struct
/// too. Callers that prepare the arrays once for many replicates run
/// [`check_finite_mat`] on X and [`check_fit_y_and_k`] first when their
/// inputs can fail them.
pub(crate) struct PreparedFit {
    /// X-scores `T` of the matrix the kernel ran on; `(n, k_used)`.
    pub(crate) t_scores: Mat<f64>,
    /// X-loadings `P`; `(d, k_used)`.
    pub(crate) p_loadings: Mat<f64>,
    /// Raw weights `W`; `(d, k_used)`.
    pub(crate) w_star: Mat<f64>,
    /// y-loadings `Q`; `(k_used,)`.
    pub(crate) q_loadings: Col<f64>,
    /// `W (P'W)⁻¹ Q` in the prepared (standardized) space; `(d,)`.
    pub(crate) coef: Col<f64>,
    /// Components kept (≤ `k`).
    pub(crate) k_used: usize,
}

/// [`PreparedFit`] of `xs`, `ys`, with `‖xs‖_F` supplied as `x_fro`
/// (see [`pls1_kernel`]): a loop over one fixed block takes `xs.norm_l2()`
/// once and passes it to every call. `par` is resolved from `xs`'s shape and
/// `k` exactly as `pls1_fit` resolves it.
///
/// # Errors
/// None today (the kernel has no error path); the `Result` is kept so a
/// kernel error can surface without a signature change.
pub(crate) fn pls1_fit_prepared_fro(
    xs: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    keep: Option<usize>,
    par: ParChoice,
    x_fro: f64,
) -> PlsKitResult<PreparedFit> {
    let par = resolve_par(par, xs.nrows(), xs.ncols(), k);
    let (t_mat, p_mat, w_mat, q_vec) = pls1_kernel(xs, ys, k, keep, par, x_fro)?;
    let k_used = w_mat.ncols();
    let coef = pls1_coef_at_k(&w_mat, &p_mat, &q_vec, k_used, par);
    Ok(PreparedFit {
        t_scores: t_mat,
        p_loadings: p_mat,
        w_star: w_mat,
        q_loadings: q_vec,
        coef,
        k_used,
    })
}

/// [`pls1_fit_prepared_fro`] on `Xs = diag(sqw)·(x − 1·mean')·diag(1/scale)`
/// through [`ImplicitXBackend`], without writing `Xs`. `x_fro` is `‖Xs‖_F`;
/// `mean_ratio` is `max_j |mean_j| / scale_j`. `None` when a component's
/// stop decision falls in the backend's rounding band; the caller then fits
/// the standardized copy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pls1_fit_implicit(
    x: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    mean: &[f64],
    scale: &[f64],
    sqw: Option<&[f64]>,
    k: usize,
    keep: Option<usize>,
    par: ParChoice,
    x_fro: f64,
    mean_ratio: f64,
) -> Option<PreparedFit> {
    let par = resolve_par(par, x.nrows(), x.ncols(), k);
    let w_floor = w_rel_floor(x.nrows(), x.ncols(), x_fro, ys.norm_l2());
    let mut backend = ImplicitXBackend {
        x,
        ys,
        mean,
        scale,
        sqw,
        par,
        band: 2.0 * (1.0 + mean_ratio) * NIPALS_ABS_FLOOR.max(w_floor),
    };
    let s0 = backend.xs_t_mul(ys, 1.0);
    let LoopOutcome::Done { t, p, w, q } = pls1_component_loop(&mut backend, s0, w_floor, k, keep)
    else {
        return None;
    };
    let t = t.unwrap_or_else(|| Mat::<f64>::zeros(x.nrows(), 0));
    let k_used = w.ncols();
    let coef = pls1_coef_at_k(&w, &p, &q, k_used, par);
    Some(PreparedFit {
        t_scores: t,
        p_loadings: p,
        w_star: w,
        q_loadings: q,
        coef,
        k_used,
    })
}

/// The exact √w vector `pls1_fit` scales rows by when handed weights `w`:
/// `sqrt(normalize_weights(w)[i])`. `pls1_fit` renormalizes the weights it
/// is given, so a caller that hoists the scaling out of a replicate loop
/// uses this, not `√w`: the two can differ in the last bit.
///
/// # Panics
/// When `Σw == 0`, where `pls1_fit` returns `InvalidWeights`. Callers pass
/// weights that went through `validate_and_normalize_weights` or a per-fold
/// renormalization (`None` for a zero-sum slice), whose sum is positive.
pub(crate) fn fit_row_scale(w: ColRef<'_, f64>) -> Col<f64> {
    let wn = crate::linalg::normalize_weights(w).expect("fit_row_scale: positive weight sum");
    crate::linalg::sqrt_col(wn.as_ref())
}

/// `sqw[i] * x[(i, j)]`, the operand order of `pls1_fit`'s row scaling.
pub(crate) fn scale_rows(x: MatRef<'_, f64>, sqw: ColRef<'_, f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| sqw[i] * x[(i, j)])
}

/// `sqw[i] * y[i]`.
pub(crate) fn scale_col(y: ColRef<'_, f64>, sqw: ColRef<'_, f64>) -> Col<f64> {
    Col::<f64>::from_fn(y.nrows(), |i| sqw[i] * y[i])
}

/// Zero all but the `keep` largest-|w| coordinates (hard thresholding at
/// the keep-th order statistic of |w| — NOT soft thresholding; survivors
/// keep their magnitudes). Exact ties break deterministically: order by
/// (|w| desc, index asc), so the lowest column index wins.
///
/// The index tiebreak is what makes an *unstable* partial select safe
/// here, and is why this is `select_nth_unstable_by` rather than a full
/// sort. `total_cmp` is a total order on bit patterns (and `.abs()`
/// collapses `+0.0` / `-0.0` onto one key), while `a.cmp(&b)` returns
/// `Equal` only for `a == b`, which never happens between two distinct
/// elements of a permutation of `0..d`. So no two elements ever compare
/// `Equal`, the sorted order is unique, and the selected *set* is fixed by
/// the comparator alone, independent of which algorithm computes it.
/// Drop the tiebreak and that stops being true; the comparator and the
/// unstable select stand or fall together.
///
/// Shared with `pls3::spls3_component`, which applies it to `u` and to
/// `v` in turn. Do not copy it: the tie rule is the reproducibility
/// contract and must have exactly one implementation.
pub(crate) fn hard_select_keep(w: &mut Col<f64>, keep: usize) {
    let d = w.nrows();
    // `keep >= d` selects everything, so there is no complement to zero.
    // Guarded rather than left to `select_nth_unstable_by`, which panics
    // on `keep == d` where the old `&idx[keep..]` yielded an empty slice:
    // this function is total and stays total.
    if keep >= d {
        return;
    }
    let mut idx: Vec<usize> = (0..d).collect();
    idx.select_nth_unstable_by(keep, |&a, &b| {
        w[b].abs().total_cmp(&w[a].abs()).then_with(|| a.cmp(&b))
    });
    for &j in &idx[keep..] {
        w[j] = 0.0;
    }
}

/// Keep the `keep` largest-magnitude entries of `v` ([`hard_select_keep`]),
/// then scale `v` to unit norm. Returns the norm it was divided by, or
/// `None`, leaving `v` selected but unscaled, when that norm is below
/// `floor`.
///
/// The one select-and-normalize step of every hard-threshold iteration:
/// the `w` step of `pls1_component_loop`, the `u` and `v` steps of
/// `pls3::spls3_component`, and the Gram mirror of the `v` step in
/// `dual_route::pls3_split_zbars_columns`. Sharing it is what keeps those
/// in the same float sequence: select, then norm, then one reciprocal, then
/// multiply. `v *= 1/nv` and `v /= nv` round differently, and on the Gram
/// route a one-ulp split can move the stopping sweep.
///
/// `keep >= v.nrows()` is a literal skip of the selection, not a no-op
/// call, so a dense endpoint's float sequence is provably untouched.
pub(crate) fn select_and_normalize(v: &mut Col<f64>, keep: usize, floor: f64) -> Option<f64> {
    let d = v.nrows();
    if keep < d {
        hard_select_keep(v, keep);
    }
    let nv = v.norm_l2();
    if nv < floor {
        return None;
    }
    let inv_nv = 1.0 / nv;
    for j in 0..d {
        v[j] *= inv_nv;
    }
    Some(nv)
}

/// Absolute floor of the PLS1 kernel (`pls1_component_loop`): a component
/// stops when `w_norm` or `t't` falls below it. The Gram-route gates that mirror those exits
/// (`dual_route::pls1_cv_r2_columns`, `signal_test::split_perm_nr_zbars`)
/// name this constant rather than restate the literal.
pub(crate) const NIPALS_ABS_FLOOR: f64 = 1e-14;

/// Relative floor on `w_norm = ‖X_a'y_a‖` for every PLS1 component:
/// `max(n, d)·ε·‖X‖_F·‖y‖`, where `X` (`n × d`) and `y` are the kernel's
/// inputs (`Xs`, `ys` in `pls1_kernel`), undeflated.
///
/// The kernel never forms `X_a'y_a` from a deflated matrix: it carries
/// `s_a = X_a'y_a` as a d-vector, `s_{a+1} = s_a − (q_a·t_a't_a)·p_a` (see
/// `pls1_kernel`). The `1e-14` absolute floor does not catch `y` running
/// out. Once the part of `y` that `X` can explain has been fitted, the exact
/// `s_a` is zero and the computed one is rounding noise, which sits at a
/// small multiple of `ε·‖X‖_F·‖y‖`: every step that feeds `s_a` (the initial
/// `X'y` sum, the scores and loadings read from the undeflated `X`, the
/// updates of `s`) carries absolute errors proportional to the entries of
/// the undeflated inputs, not the deflated ones. Standardizing with ddof 0
/// makes `‖X‖_F·‖y‖ = n·√d`, so that noise clears `1e-14` as soon as `n` is
/// in the hundreds. Normalizing that noise and continuing on it loses
/// orthogonality, and `w_norm` can climb back up over the following
/// components, so the floor must fire at the first noise component rather
/// than wait for a monotone decline. Measured at `n = 2000`, `d = 40` (rank
/// 39) with the explicit-deflation kernel this crate used up to 0.5.0:
/// `w_norm` fell from 339 to about `1.4e-13` by the 18th component, regrew
/// to 40, and all 40 components were kept. With the current kernel and the
/// floor disabled, `w_norm` falls to about `4.6e-13` by component 19,
/// reaches at most `7.207e2` after that, and 40 of the 40 components are
/// kept.
///
/// The first component is no exception: when `y` is orthogonal to the
/// columns of `X` (for instance an outcome residualized on a set of
/// covariates that spans `X`), `X'y` is itself rounding noise, and without
/// the floor the fit keeps one component whose weight vector `w` is a
/// normalized noise vector pointing nowhere in particular. With it such a fit
/// returns `k_used = 0`, the same zero model a constant `y` gives.
///
/// The reference scale is `‖X‖_F·‖y‖` rather than the first `w_norm`
/// (`sigma_rel_floor` in `pls3.rs`, the PLS3 counterpart, likewise uses the
/// block norms `‖X̃‖_F·‖Ỹ‖_F` rather than `σ₁`). The
/// first `w_norm` is only a lower bound on the rounding scale and can be
/// arbitrarily far below it when `y` is nearly orthogonal to `X`; a floor
/// on `max(n, d)·ε·‖X'y‖` kept noise components on such designs. The
/// current residual `‖y_a‖` is no reference either: when `y` lies in the
/// span of `X`, `y_a` is itself rounding noise, and `‖X_a'y_a‖` relative to
/// `‖X_a‖·‖y_a‖` is then of order one.
///
/// `max(n, d)` is the factor `sigma_rel_floor` uses: `n·ε` bounds the
/// rounding of a length-`n` dot product relative to the product of the
/// norms, and at most `d` updates of `s` feed into any component. On a
/// sweep of rank-deficient, `y`-exhausted, `y`-orthogonal, weak-signal,
/// `p ≫ n`, sparse, weighted and tiny designs (`n` from 5 to 2e5;
/// `fit::kernel_tests::floor_calibration_sweep`, run on the current
/// kernel), the first noise component, located from each design's
/// construction (`y` in the span of `m` singular directions, `y`
/// orthogonal to `X`, or `X` deflated to zero at its rank), sat at most at
/// `0.020×` this floor (`first_noise_max`), while every
/// component contributing more than `1e-9` of `‖y‖` to the fit sat at least
/// `38.807×` above it. The real components it drops, which have
/// decayed into rounding, contributed at most about `4.4e-12` of `‖y‖`
/// to the fitted values of the unfloored fit. For the first component in
/// particular (880 designs over the same families, `n` from 5 to 1e5,
/// weighted and not; measured on the
/// explicit-deflation kernel, whose first component is the same float
/// sequence as the current kernel's, so the numbers carry over), a `y`
/// orthogonal to the standardized `X` up to rounding sat at most at `0.33×`
/// the floor, a true effect of `1e-9` of `‖y‖` added to it at least `14×`
/// above it (the margin shrinks as `1/n`), and an unstructured random `y`
/// at least `2.7e7×` above it.
///
/// The K = 1 Gram mirror `dual_route::pls1_cv_r2_columns` evaluates this
/// same function on bit-identical inputs, but it cannot form `‖X'y‖` to the
/// accuracy the floor needs, so a column it cannot resolve is recomputed
/// by this kernel (see the doc comment there).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn w_rel_floor(n: usize, d: usize, x_fro: f64, y_norm: f64) -> f64 {
    (n.max(d) as f64) * f64::EPSILON * x_fro * y_norm
}

/// What a [`ComponentBackend`] returns for one rotation vector `r`.
pub(crate) struct Scored {
    /// `t't` on the X backend, `r'Cr` on the Gram backend.
    pub(crate) tt: f64,
    /// The scores `t = Xs r`; `None` on a backend that never forms them.
    pub(crate) t: Option<Col<f64>>,
}

/// Backend control after a gate hook of [`ComponentBackend`].
pub(crate) enum Gate {
    /// The backend decides this component the way the X backend would.
    Continue,
    /// It cannot; the caller recomputes the whole fit on the X backend.
    // Constructed only by a Gram backend; the X backend never returns it.
    Unresolved,
}

/// One way to form scores, loadings and `q` from a rotation vector `r`.
///
/// Every rule that decides numbers (the weight step, `keep` selection, the
/// floors, truncation, the rotation, the update of `s`) lives in
/// [`pls1_component_loop`], once, and not in a backend.
pub(crate) trait ComponentBackend {
    /// Called at every `a` from 1 up to `min(k, k_used + 1)` (1-based),
    /// before selection, on the current `s`, including the component whose
    /// selection stops the loop. The X backend always continues; the Gram
    /// backend checks `‖s‖` bands and the keep-boundary gap, both of which
    /// it only evaluates starting at `a = 2`.
    fn gate_s(&mut self, a: usize, s: &Col<f64>, keep: Option<usize>) -> Gate;
    /// `t` (X backend) and `tt = ‖t‖²` or `r'Cr`.
    fn score(&mut self, r: &Col<f64>) -> Scored;
    /// After `tt` is known, before the `tt` floor test.
    fn gate_tt(&mut self, a: usize, r: &Col<f64>, tt: f64) -> Gate;
    /// `p = Xs't·inv_tt` (X) or `C r·inv_tt` (Gram).
    fn loading(&mut self, r: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64>;
    /// `q`: X backend `(Σ ys[i]·t[i])·inv_tt`; Gram `(s'w)·inv_tt`.
    fn q(&mut self, s: &Col<f64>, w: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> f64;
}

/// Result of [`pls1_component_loop`].
#[allow(clippy::large_enum_variant)] // one per fit, moved, never copied
pub(crate) enum LoopOutcome {
    /// `k_used = w.ncols()` components. `t` is `None` when the backend forms
    /// no scores, or when no component was kept.
    Done {
        t: Option<Mat<f64>>,
        p: Mat<f64>,
        w: Mat<f64>,
        q: Col<f64>,
    },
    /// A gate returned [`Gate::Unresolved`].
    Unresolved,
}

/// `r_a = w_a − R_{<a}(P_{<a}'w_a)`, the rotation that makes `Xs r_a` the
/// deflated score `X_a w_a` (see [`pls1_kernel`]). `a` is the 0-based index
/// of the component being built, so `R_{<a}`, `P_{<a}` are the first `a`
/// columns. Two small GEMVs, sequential, in ascending order: first
/// `c_j = Σ_i p_{ij}·w_i`, then `r_i = w_i − Σ_j r_{ij}·c_j`. At `a = 0` it
/// is a literal copy of `w`, which keeps every K = 1 fit the float sequence
/// of the explicit-deflation kernel.
#[allow(clippy::many_single_char_names)]
fn rotation(w: &Col<f64>, r_mat: &Mat<f64>, p_mat: &Mat<f64>, a: usize) -> Col<f64> {
    if a == 0 {
        return w.clone();
    }
    let d = w.nrows();
    let c: Vec<f64> = (0..a)
        .map(|j| (0..d).map(|i| p_mat[(i, j)] * w[i]).sum::<f64>())
        .collect();
    Col::<f64>::from_fn(d, |i| {
        w[i] - (0..a).map(|j| r_mat[(i, j)] * c[j]).sum::<f64>()
    })
}

/// The shared PLS1 component loop.
///
/// Starts from `s0 = Xs'ys` and carries `s = X_a'y_a` as a d-vector. Per
/// component `a`: the backend's `gate_s`; `w = s`, selected and normalized
/// by [`select_and_normalize`] against `max(NIPALS_ABS_FLOOR, w_floor)` (a
/// `None` stops the loop); the rotation `r = w − R(P'w)`; the backend's
/// `score` and `gate_tt`; the stop on `tt < NIPALS_ABS_FLOOR`; the backend's
/// `loading` and `q` with the same `inv_tt = 1/tt`; then
/// `s ← s − (q·tt)·p` in ascending `j`. Outputs are truncated to the
/// components kept, as the explicit-deflation kernel truncated them.
///
/// Call order is a guarantee, not an implementation detail: `gate_s` runs
/// at every `a` up to `min(k, k_used + 1)`, `a = 1` included, including the
/// component whose selection stops the loop; a `tt` stop is preceded by
/// `gate_tt` at that same `a`; after either gate returns `Unresolved`, no
/// backend method is called for the rest of the fit.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub(crate) fn pls1_component_loop<B: ComponentBackend>(
    backend: &mut B,
    s0: Col<f64>,
    w_floor: f64,
    k: usize,
    keep: Option<usize>,
) -> LoopOutcome {
    let d = s0.nrows();
    // `w_norm < max(a, b)` is `w_norm < a || w_norm < b`, NaN included
    // (`f64::max` ignores a NaN operand, and a NaN `w_norm` fails either test).
    let floor = NIPALS_ABS_FLOOR.max(w_floor);
    let mut s = s0;
    let mut t_cols: Vec<Col<f64>> = Vec::with_capacity(k);
    let mut r_mat = Mat::<f64>::zeros(d, k);
    let mut p_mat = Mat::<f64>::zeros(d, k);
    let mut w_mat = Mat::<f64>::zeros(d, k);
    let mut q_vec = Col::<f64>::zeros(k);
    let mut k_actual = 0usize;

    for a in 0..k {
        if matches!(backend.gate_s(a + 1, &s, keep), Gate::Unresolved) {
            return LoopOutcome::Unresolved;
        }
        // Sparse keep-count selection sits between `s` and the norm guard, so
        // an all-zero surviving set truncates via the absolute floor. `keep =
        // None` and `keep = d` are a literal skip of the selection (dense
        // bit-parity at `keep = n_features`).
        let mut w = s.clone();
        if select_and_normalize(&mut w, keep.unwrap_or(d), floor).is_none() {
            break;
        }
        let r = rotation(&w, &r_mat, &p_mat, a);
        let scored = backend.score(&r);
        let tt = scored.tt;
        if matches!(backend.gate_tt(a + 1, &r, tt), Gate::Unresolved) {
            return LoopOutcome::Unresolved;
        }
        if tt < NIPALS_ABS_FLOOR {
            break;
        }
        let inv_tt = 1.0 / tt;
        let p = backend.loading(&r, scored.t.as_ref(), inv_tt);
        let q = backend.q(&s, &w, scored.t.as_ref(), inv_tt);
        let qtt = q * tt;
        for j in 0..d {
            s[j] -= qtt * p[j];
        }
        r_mat.col_mut(a).copy_from(&r);
        p_mat.col_mut(a).copy_from(&p);
        w_mat.col_mut(a).copy_from(&w);
        q_vec[a] = q;
        if let Some(t) = scored.t {
            t_cols.push(t);
        }
        k_actual = a + 1;
    }

    let t = if t_cols.is_empty() {
        None
    } else {
        let n = t_cols[0].nrows();
        Some(Mat::<f64>::from_fn(n, k_actual, |i, j| t_cols[j][i]))
    };
    if k_actual == k {
        LoopOutcome::Done {
            t,
            p: p_mat,
            w: w_mat,
            q: q_vec,
        }
    } else {
        LoopOutcome::Done {
            t,
            p: p_mat.subcols(0, k_actual).to_owned(),
            w: w_mat.subcols(0, k_actual).to_owned(),
            q: Col::<f64>::from_fn(k_actual, |i| q_vec[i]),
        }
    }
}

/// Algorithm 1 of Dayal and MacGregor (1997): scores and loadings from the
/// undeflated `Xs`, two reads per component, no writes, no copy.
#[allow(clippy::doc_markdown)] // "MacGregor"
struct XBackend<'a> {
    xs: MatRef<'a, f64>,
    ys: ColRef<'a, f64>,
    par: Par,
}

impl ComponentBackend for XBackend<'_> {
    fn gate_s(&mut self, _a: usize, _s: &Col<f64>, _keep: Option<usize>) -> Gate {
        Gate::Continue
    }

    fn score(&mut self, r: &Col<f64>) -> Scored {
        // t = Xs r  (GEMV)
        let mut t = Col::<f64>::zeros(self.xs.nrows());
        matmul(
            t.as_mut().as_mat_mut(),
            Accum::Replace,
            self.xs,
            r.as_ref().as_mat(),
            1.0,
            self.par,
        );
        let tt = t.squared_norm_l2();
        Scored { tt, t: Some(t) }
    }

    fn gate_tt(&mut self, _a: usize, _r: &Col<f64>, _tt: f64) -> Gate {
        Gate::Continue
    }

    fn loading(&mut self, _r: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64> {
        // p = Xs' t / (t't)  (GEMV, alpha = 1/tt as in the explicit-deflation kernel)
        let t = t.expect("the X backend always forms t");
        let mut p = Col::<f64>::zeros(self.xs.ncols());
        matmul(
            p.as_mut().as_mat_mut(),
            Accum::Replace,
            self.xs.transpose(),
            t.as_ref().as_mat(),
            inv_tt,
            self.par,
        );
        p
    }

    fn q(&mut self, _s: &Col<f64>, _w: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> f64 {
        // The explicit-deflation kernel's expression, on the undeflated ys:
        // a scalar sum in ascending i, times the same 1/tt.
        // `ImplicitXBackend::q` repeats it; change the two together.
        let t = t.expect("the X backend always forms t");
        let n = self.ys.nrows();
        (0..n).map(|i| self.ys[i] * t[i]).sum::<f64>() * inv_tt
    }
}

/// Largest `max_j |mean_j| / scale_j` for which `pls1_fit` forms the
/// standardized products from the raw X ([`ImplicitXBackend`]); above it the
/// standardized X is written into a copy. The implicit products lose about
/// `log10(1 + |mean_j| / scale_j)` digits to cancellation: on Gaussian designs
/// with a common offset the coefficients moved by about `ratio·1e-16`, at most
/// `6·ratio·1e-16` at `k = 5`, so this bound keeps them within about `1e-12` of
/// the copy's.
pub(crate) const IMPLICIT_MAX_MEAN_RATIO: f64 = 1e3;

/// The X backend on `Xs = diag(sqw)·(X − 1·mean')·diag(1/scale)` formed from
/// the raw `X` on the fly, so the standardized matrix is never written: the
/// kernel only multiplies by `Xs` and `Xs'`, and each product is the raw one
/// plus a rank-one correction,
///
/// - `Xs·v  = sqw ⊙ (X·u − 1·(mean'u))`, with `u = v / scale`;
/// - `Xs'·t = (X'·t̃ − mean·Σt̃) / scale`, with `t̃ = sqw ⊙ t`.
///
/// Same float model as [`XBackend`] on a materialized `Xs` up to rounding, but
/// the correction subtracts two terms of size `|mean_j|·|u_j|`, so column `j`
/// loses about `log10(1 + |mean_j| / scale_j)` digits to cancellation that
/// the materialized `x − mean` does not.
///
/// That cancellation also lifts the rounding noise in `s`, which the
/// truncation floor ([`w_rel_floor`]) is sized for on the materialized `Xs`:
/// with `y` orthogonal to `X`, the implicit first `‖s‖` reached `27×` the floor
/// (`n = 8`, ratio 900), where the materialized one stays below it; without a
/// gate the fit keeps a noise component. So `gate_s` hands every stop decision
/// on the `w` floor to the caller: a selected `‖s‖` below `band =
/// 2·(1 + max_j |mean_j| / scale_j)·floor` is `Unresolved`. On orthogonal-`y`
/// sweeps (`n` from 8 to 1000, ratio up to 900) the implicit noise sat at most
/// at `0.03·(1 + ratio)·floor` and the materialized at `0.58·floor`, so the
/// factor 2 also covers the copy's own rounding at a ratio near 0.
struct ImplicitXBackend<'a> {
    x: MatRef<'a, f64>,
    ys: ColRef<'a, f64>,
    mean: &'a [f64],
    scale: &'a [f64],
    sqw: Option<&'a [f64]>,
    par: Par,
    band: f64,
}

impl ImplicitXBackend<'_> {
    /// `Xs·v`, `(n,)`.
    fn xs_mul(&self, v: &Col<f64>) -> Col<f64> {
        let d = self.x.ncols();
        let u = Col::<f64>::from_fn(d, |j| v[j] / self.scale[j]);
        let c: f64 = (0..d).map(|j| self.mean[j] * u[j]).sum();
        let mut xu = Col::<f64>::zeros(self.x.nrows());
        matmul(
            xu.as_mut().as_mat_mut(),
            Accum::Replace,
            self.x,
            u.as_ref().as_mat(),
            1.0,
            self.par,
        );
        match self.sqw {
            None => {
                for i in 0..xu.nrows() {
                    xu[i] -= c;
                }
            }
            Some(sqw) => {
                for i in 0..xu.nrows() {
                    xu[i] = (xu[i] - c) * sqw[i];
                }
            }
        }
        xu
    }

    /// `alpha·Xs'·t`, `(d,)`.
    fn xs_t_mul(&self, t: ColRef<'_, f64>, alpha: f64) -> Col<f64> {
        let n = self.x.nrows();
        let tw_owned: Option<Col<f64>> = self
            .sqw
            .map(|sqw| Col::<f64>::from_fn(n, |i| t[i] * sqw[i]));
        let tw = tw_owned.as_ref().map_or(t, Col::as_ref);
        let sum_tw: f64 = (0..n).map(|i| tw[i]).sum();
        // A wide row-major X (a C-ordered host array) reads its rows in
        // blocks; any other X goes to faer.
        let mut g = crate::linalg::row_major_t_mul(self.x, tw, self.par).unwrap_or_else(|| {
            let mut g = Col::<f64>::zeros(self.x.ncols());
            matmul(
                g.as_mut().as_mat_mut(),
                Accum::Replace,
                self.x.transpose(),
                tw.as_mat(),
                1.0,
                self.par,
            );
            g
        });
        for j in 0..g.nrows() {
            g[j] = (g[j] - self.mean[j] * sum_tw) / self.scale[j] * alpha;
        }
        g
    }
}

impl ComponentBackend for ImplicitXBackend<'_> {
    fn gate_s(&mut self, _a: usize, s: &Col<f64>, keep: Option<usize>) -> Gate {
        if crate::gram_p::selected_norm_and_gap(s, keep).0 < self.band {
            Gate::Unresolved
        } else {
            Gate::Continue
        }
    }

    fn score(&mut self, r: &Col<f64>) -> Scored {
        let t = self.xs_mul(r);
        let tt = t.squared_norm_l2();
        Scored { tt, t: Some(t) }
    }

    fn gate_tt(&mut self, _a: usize, _r: &Col<f64>, _tt: f64) -> Gate {
        Gate::Continue
    }

    fn loading(&mut self, _r: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64> {
        let t = t.expect("the implicit X backend always forms t");
        self.xs_t_mul(t.as_ref(), inv_tt)
    }

    fn q(&mut self, _s: &Col<f64>, _w: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> f64 {
        // As in `XBackend::q`; change the two together.
        let t = t.expect("the implicit X backend always forms t");
        let n = self.ys.nrows();
        (0..n).map(|i| self.ys[i] * t[i]).sum::<f64>() * inv_tt
    }
}

/// The PLS1 kernel: the NIPALS PLS1 model, computed by Improved Kernel PLS
/// (Dayal and MacGregor 1997, Algorithm 1) through [`pls1_component_loop`]
/// with the X backend. Same inputs and outputs (`T`, `P`, `W`, `Q`) as the
/// explicit-deflation kernel it replaced.
///
/// # Derivation
/// NIPALS deflates `X_{a+1} = X_a − t_a p_a'` and `y_{a+1} = y_a − q_a t_a`
/// and takes `w_a ∝ X_a'y_a`, `t_a = X_a w_a`, `p_a = X_a't_a / t_a't_a`,
/// `q_a = y_a't_a / t_a't_a`. This kernel forms none of `X_a`, `y_a` and
/// keeps `s_a = X_a'y_a`. In exact arithmetic, by induction on `a`:
///
/// 1. `s_{a+1} = s_a − p_a·(q_a·t_a't_a)`. `X_{a+1}'t_a = 0` gives
///    `X_{a+1}'y_{a+1} = X_{a+1}'y_a = X_a'y_a − p_a·(t_a'y_a)`, and
///    `t_a'y_a = q_a·t_a't_a`.
/// 2. `t_a = Xs r_a` with `r_a = w_a − R_{<a}(P_{<a}'w_a)`. Writing
///    `X_a = Xs·M_a`, the deflation gives `M_{a+1} = M_a − r_a p_a'`, so
///    `M_a = I − Σ_{j<a} r_j p_j'` and `r_a = M_a w_a`.
/// 3. `p_a = Xs't_a / t_a't_a`. `X_a't_a = Xs't_a − Σ_{j<a} p_j (t_j't_a)`,
///    and the scores are mutually orthogonal (`t_a = X_a w_a` lies in the
///    complement of `t_1 … t_{a−1}`), so the sum vanishes.
/// 4. `q_a = ys't_a / t_a't_a = s_a'w_a / t_a't_a`. `ys − y_a` is a
///    combination of `t_1 … t_{a−1}`, orthogonal to `t_a`, and
///    `y_a't_a = y_a'X_a w_a = s_a'w_a`.
///
/// So the floors test `‖s_a‖`, the undeflated `‖Xs‖_F` (passed in as
/// `x_fro`), `‖ys‖`, and `t't`, and at `a = 1` every step is the
/// same operation on the same inputs as the explicit-deflation kernel: a
/// K = 1 fit on a column-major `Xs` is bit-identical to it. At `a ≥ 2` the
/// two differ in rounding only; `fit::kernel_tests` bounds the difference.
///
/// Each component reads `Xs` twice (`t = Xs r`, `p = Xs't/tt`) and never
/// writes it; `s_1 = Xs'ys` is one more read per fit. `par` threads the
/// three GEMVs, resolved by the caller as before (`resolve_par`).
///
/// # Errors
/// None today; the `Result` matches the call site.
#[allow(clippy::doc_markdown)] // "MacGregor"
#[allow(clippy::many_single_char_names)]
#[allow(clippy::type_complexity)]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn pls1_kernel(
    xs: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    keep: Option<usize>,
    par: Par,
    x_fro: f64,
) -> PlsKitResult<(Mat<f64>, Mat<f64>, Mat<f64>, Col<f64>)> {
    let n = xs.nrows();
    let d = xs.ncols();
    // s_1 = Xs' ys  (GEMV)
    let mut s0 = Col::<f64>::zeros(d);
    matmul(
        s0.as_mut().as_mat_mut(),
        Accum::Replace,
        xs.transpose(),
        ys.as_mat(),
        1.0,
        par,
    );
    // `x_fro` is `‖xs‖_F`, supplied by the caller: `pls1_fit` forms it from
    // its standardization moments or its sum-of-squares screen, the other
    // callers take `xs.norm_l2()`. It only ever gates a `break`, so no output
    // bit depends on it unless it truncates.
    let w_floor = w_rel_floor(n, d, x_fro, ys.norm_l2());
    let mut backend = XBackend { xs, ys, par };
    match pls1_component_loop(&mut backend, s0, w_floor, k, keep) {
        LoopOutcome::Done { t, p, w, q } => {
            Ok((t.unwrap_or_else(|| Mat::<f64>::zeros(n, 0)), p, w, q))
        }
        LoopOutcome::Unresolved => unreachable!("the X backend never returns Gate::Unresolved"),
    }
}

/// Regression coefficient using first `k` PLS components.
/// Formula: coef = W (P'W)^{-1} Q.
///
/// `par` threads the two GEMMs (`P'W` and `W·z`) explicitly: operator-`*`
/// would dispatch on faer's global parallelism (default `Rayon` at the
/// pool's size), which leaks into the global pool when this runs inside a
/// resampler's Rayon worker and rounds differently per pool size. The K×K
/// solve is `linalg::lu_solve_in_place`, sequential for the same reason.
#[allow(clippy::many_single_char_names)]
pub(crate) fn pls1_coef_at_k(
    w: &Mat<f64>,
    p: &Mat<f64>,
    q: &Col<f64>,
    k: usize,
    par: Par,
) -> Col<f64> {
    let d = w.nrows();
    let wk = w.subcols(0, k);
    let pk = p.subcols(0, k);
    let qk = q.subrows(0, k);
    // P' W is (k, k); solve (P'W) z = Q via faer's LU, then coef = W z.
    let mut pwk = Mat::<f64>::zeros(k, k);
    matmul(pwk.as_mut(), Accum::Replace, pk.transpose(), wk, 1.0, par);
    let mut z: Col<f64> = qk.to_owned();
    crate::linalg::lu_solve_in_place(pwk.as_ref(), z.as_mut().as_mat_mut());
    let mut coef = Col::<f64>::zeros(d);
    matmul(
        coef.as_mut().as_mat_mut(),
        Accum::Replace,
        wk,
        z.as_ref().as_mat(),
        1.0,
        par,
    );
    coef
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests {
    use super::*;
    use crate::test_support::{orthonormal_basis, project_off};
    use approx::assert_relative_eq;

    /// The explicit-deflation PLS1 kernel that `pls1_kernel` replaced, kept
    /// verbatim as the oracle for the IKPLS kernel (`pls1_kernel`). Do not
    /// edit: every equivalence test in `fit::kernel_tests` measures against
    /// this exact float sequence.
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    #[allow(clippy::type_complexity)]
    #[allow(clippy::unnecessary_wraps)] // reserved for future variants that may return Err
    pub(super) fn nipals_pls1_reference(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k: usize,
        keep: Option<usize>,
        par: Par,
    ) -> PlsKitResult<(Mat<f64>, Mat<f64>, Mat<f64>, Col<f64>)> {
        let n = x.nrows();
        let d = x.ncols();
        // Owned working copies — deflated in place across components.
        // The caller resolves `par` from `FitOpts::par` (see `resolve_par`).
        // Resamplers explicitly force `ParChoice::Seq`; oversubscribing the
        // outer Rayon pool with nested parallelism here would tank throughput.
        let mut xk: Mat<f64> = x.to_owned();
        let mut yk: Col<f64> = y.to_owned();
        // From the undeflated inputs, once. It only ever gates a `break`, so no
        // output bit depends on it unless it truncates.
        let w_floor = w_rel_floor(n, d, x.norm_l2(), y.norm_l2());

        // Pre-allocate output matrices (truncated at end if convergence stops short).
        let mut t_mat = Mat::<f64>::zeros(n, k);
        let mut p_mat = Mat::<f64>::zeros(d, k);
        let mut w_mat = Mat::<f64>::zeros(d, k);
        let mut q_vec = Col::<f64>::zeros(k);
        let mut k_actual = 0usize;

        for a in 0..k {
            // w = X' y  (GEMV)
            let mut w: Col<f64> = Col::<f64>::zeros(d);
            matmul(
                w.as_mut().as_mat_mut(),
                Accum::Replace,
                xk.as_ref().transpose(),
                yk.as_ref().as_mat(),
                1.0,
                par,
            );
            // Sparse keep-count selection (spls1 family) sits between the GEMV
            // and the norm guard, so an all-zero surviving set truncates via the
            // absolute floor. `keep = None` and the dense endpoint (keep ==
            // n_features) are a LITERAL skip of the selection, so the dense
            // float sequence is provably untouched (bit-parity tripwire), not
            // merely value-preserving. `w_norm < max(a, b)` is `w_norm < a ||
            // w_norm < b`, NaN included (`f64::max` ignores a NaN operand, and
            // a NaN `w_norm` fails either test).
            if select_and_normalize(&mut w, keep.unwrap_or(d), NIPALS_ABS_FLOOR.max(w_floor))
                .is_none()
            {
                break;
            }
            // t = X w  (GEMV)
            let mut t: Col<f64> = Col::<f64>::zeros(n);
            matmul(
                t.as_mut().as_mat_mut(),
                Accum::Replace,
                xk.as_ref(),
                w.as_ref().as_mat(),
                1.0,
                par,
            );
            let tt = t.squared_norm_l2();
            if tt < NIPALS_ABS_FLOOR {
                break;
            }
            let inv_tt = 1.0 / tt;
            // p = X' t / (t't)  (GEMV)
            let mut p: Col<f64> = Col::<f64>::zeros(d);
            matmul(
                p.as_mut().as_mat_mut(),
                Accum::Replace,
                xk.as_ref().transpose(),
                t.as_ref().as_mat(),
                inv_tt,
                par,
            );
            // q = y' t / (t't) — small dot, scalar is fine
            let q: f64 = (0..n).map(|i| yk[i] * t[i]).sum::<f64>() * inv_tt;

            // Rank-1 deflation: Xk -= t · p'  (GER via matmul with alpha=-1)
            matmul(
                xk.as_mut(),
                Accum::Add,
                t.as_ref().as_mat(),
                p.as_ref().as_mat().transpose(),
                -1.0,
                par,
            );
            // y -= q · t  (AXPY; scalar n-pass is fine)
            for i in 0..n {
                yk[i] -= q * t[i];
            }

            t_mat.col_mut(a).copy_from(&t);
            p_mat.col_mut(a).copy_from(&p);
            w_mat.col_mut(a).copy_from(&w);
            q_vec[a] = q;
            k_actual = a + 1;
        }

        if k_actual == k {
            Ok((t_mat, p_mat, w_mat, q_vec))
        } else {
            // Truncate to actually-fitted columns.
            let t_out = t_mat.subcols(0, k_actual).to_owned();
            let p_out = p_mat.subcols(0, k_actual).to_owned();
            let w_out = w_mat.subcols(0, k_actual).to_owned();
            let q_out = Col::<f64>::from_fn(k_actual, |i| q_vec[i]);
            Ok((t_out, p_out, w_out, q_out))
        }
    }

    fn linear_data(n: usize, d: usize, k_true: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta_true = Col::<f64>::from_fn(d, |j| if j < k_true { 1.0 } else { 0.0 });
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-0.1..0.1));
        let y_signal: Col<f64> = &x * &beta_true;
        let y = Col::<f64>::from_fn(n, |i| y_signal[i] + noise[i]);
        (x, y)
    }

    #[test]
    fn fit_returns_correct_shapes() {
        let (x, y) = linear_data(50, 8, 3, 1);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!((m.t_scores.nrows(), m.t_scores.ncols()), (50, 3));
        assert_eq!((m.p_loadings.nrows(), m.p_loadings.ncols()), (8, 3));
        assert_eq!((m.w_star.nrows(), m.w_star.ncols()), (8, 3));
        assert_eq!(m.q_loadings.nrows(), 3);
        assert_eq!(m.coef.nrows(), 8);
        assert_eq!(m.beta.nrows(), 8);
        assert_eq!(m.k_used, 3);
    }

    #[test]
    fn fit_pre_standardized_skips_centering() {
        let (x, y) = linear_data(50, 8, 3, 1);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize1(y.as_ref());
        let m = pls1_fit(
            xs.as_ref(),
            ys.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
        )
        .unwrap();
        assert!(m.pre_standardized);
        for j in 0..m.coef.nrows() {
            assert_relative_eq!(m.beta[j], m.coef[j], epsilon = 1e-15);
        }
        assert_relative_eq!(m.intercept, 0.0, epsilon = 1e-15);
    }

    // ── spls1 sparse kernel ──────────────────────────────────────────

    #[test]
    fn spls1_keep_eq_n_features_is_bit_identical_to_dense() {
        // THE bit-parity tripwire (`_docs/rust/api.md`): keep = n_features must be a literal
        // skip — exact equality, not approx.
        let (x, y) = linear_data(50, 8, 3, 1);
        let dense = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let sparse = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            8,
            None,
            FitOpts::default(),
        )
        .unwrap();
        for j in 0..8 {
            assert_eq!(
                dense.coef[j].to_bits(),
                sparse.coef[j].to_bits(),
                "coef[{j}]"
            );
            assert_eq!(
                dense.beta[j].to_bits(),
                sparse.beta[j].to_bits(),
                "beta[{j}]"
            );
        }
        assert_eq!(dense.intercept.to_bits(), sparse.intercept.to_bits());
        assert_eq!(dense.k_used, sparse.k_used);
        assert_eq!(sparse.keep, Some(8));
        assert_eq!(dense.keep, None);
    }

    #[test]
    fn spls1_exact_nonzero_count_per_component() {
        // At keep = m, exactly m nonzeros per w column — always exact
        // (ties break by lowest index, so never m±1).
        let (x, y) = linear_data(50, 10, 3, 7);
        for keep in [1usize, 3, 7] {
            let m = spls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(3),
                keep,
                None,
                FitOpts::default(),
            )
            .unwrap();
            for a in 0..m.k_used {
                let nnz = (0..10).filter(|&j| m.w_star[(j, a)] != 0.0).count();
                assert_eq!(nnz, keep, "component {a} at keep={keep}");
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // asserting exact-zero from hard_select_keep — intentional bit-equality
    fn spls1_tie_break_lowest_index_wins() {
        // X built so that |X'y| has exact ties: column 1 duplicates column 0
        // and column 3 duplicates column 2. keep=1 must select column 0;
        // keep=3 must select {0, 1, 2} (not {0, 1, 3}).
        use rand::RngExt;
        use rand::SeedableRng;
        let n = 16;
        let mut x = Mat::<f64>::zeros(n, 4);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        for i in 0..n {
            let a = rng.random_range(-1.0..1.0);
            let b = rng.random_range(-1.0..1.0);
            x[(i, 0)] = a;
            x[(i, 1)] = a; // exact duplicate → exact |w| tie with col 0
            x[(i, 2)] = b;
            x[(i, 3)] = b; // exact duplicate → exact |w| tie with col 2
        }
        let y = Col::<f64>::from_fn(n, |i| x[(i, 0)] + 0.5 * x[(i, 2)]);
        let m1 = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(1),
            1,
            None,
            FitOpts {
                pre_standardized: true,
                check_n_eff: false,
                ..FitOpts::default()
            },
        )
        .unwrap();
        assert!(
            m1.w_star[(0, 0)] != 0.0,
            "keep=1 must keep column 0 (lowest index in tie)"
        );
        for j in 1..4 {
            assert_eq!(m1.w_star[(j, 0)], 0.0, "col {j} must be zeroed");
        }
        let m3 = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(1),
            3,
            None,
            FitOpts {
                pre_standardized: true,
                check_n_eff: false,
                ..FitOpts::default()
            },
        )
        .unwrap();
        assert!(m3.w_star[(2, 0)] != 0.0, "col 2 (higher |w|-rank than its duplicate at idx 3 via index tie-break path) must survive");
        assert_eq!(m3.w_star[(3, 0)], 0.0, "col 3 loses the tie against col 2");
    }

    // ── relative floor on w_norm (y exhausted) ──────────────────────

    pub(super) fn uniform_mat(n: usize, d: usize, seed: u64) -> Mat<f64> {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0))
    }

    /// `n = 2000`, `d = 40`, last column the sum of the first two (rank
    /// 39), `y` independent of `X`: `y`'s projection onto the span of `X`
    /// is fitted within about 17 components, after which `‖X_a'y_a‖` is
    /// rounding noise near `1e-13` that the absolute floor let through.
    fn exhausted_y_design() -> (Mat<f64>, Col<f64>) {
        let (n, d) = (2000, 40);
        let mut x = uniform_mat(n, d, 11);
        for i in 0..n {
            x[(i, d - 1)] = x[(i, 0)] + x[(i, 1)];
        }
        let yc = uniform_mat(n, 1, 12);
        let y = Col::<f64>::from_fn(n, |i| yc[(i, 0)]);
        (x, y)
    }

    #[test]
    fn nipals_drops_noise_components_once_y_is_exhausted() {
        let (x, y) = exhausted_y_design();
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(40),
            None,
            FitOpts::default(),
        )
        .unwrap();
        // Without the relative floor all 40 components were kept, the
        // last ~20 of them fitted to rounding noise.
        assert!(
            (5..=20).contains(&m.k_used),
            "expected truncation near the numerical rank, got k_used={}",
            m.k_used
        );
        assert!(m.coef.norm_l2().is_finite());
    }

    #[test]
    fn spls1_drops_noise_components_once_y_is_exhausted() {
        // Same design through the sparse path: selection runs before the
        // floor, so the floor sees the selected norm.
        let (x, y) = exhausted_y_design();
        let m = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(40),
            10,
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert!(
            m.k_used < 36,
            "expected truncation before the noise tail, got k_used={}",
            m.k_used
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nipals_stops_at_m_when_y_is_in_span_of_m_directions() {
        // y is an exact combination of two left singular vectors of the
        // standardized X, so exactly two components are real and the
        // third `‖X_a'y_a‖` is rounding noise (about 1e-13 here, above the
        // absolute floor).
        let (n, d) = (500, 30);
        let x = uniform_mat(n, d, 21);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let svd = xs.thin_svd().unwrap();
        let u = svd.U();
        let y = Col::<f64>::from_fn(n, |i| 1.3 * u[(i, 0)] - 0.7 * u[(i, 5)]);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(10),
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!(m.k_used, 2);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nipals_floor_is_relative_to_x_and_y_not_to_the_first_component() {
        // y = (component orthogonal to X) + 1e-6 · (signal in X): the first
        // `‖X'y‖` is about 1e-6 of `‖X‖_F·‖y‖`, so a floor relative to it
        // would sit below the rounding noise and keep all 20 components.
        let (n, d) = (1000, 20);
        let x = uniform_mat(n, d, 31);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let svd = xs.thin_svd().unwrap();
        let u = svd.U();
        let e = uniform_mat(n, 1, 32);
        let mut e_perp = Col::<f64>::from_fn(n, |i| e[(i, 0)]);
        for j in 0..d {
            let c: f64 = (0..n).map(|i| u[(i, j)] * e_perp[i]).sum();
            for i in 0..n {
                e_perp[i] -= c * u[(i, j)];
            }
        }
        let b = uniform_mat(d, 1, 33);
        let y = Col::<f64>::from_fn(n, |i| {
            e_perp[i] + 1e-6 * (0..d).map(|j| xs[(i, j)] * b[(j, 0)]).sum::<f64>()
        });
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(20),
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert!(m.k_used < 20, "got k_used={}", m.k_used);
    }

    /// `n = 200`, `d = 10`, and a `y` orthogonal to the span of `1` and the
    /// standardized columns of `X` (Gram-Schmidt, each projection applied
    /// twice), plus the standardized `X`. `X'y` is then rounding noise near
    /// `1e-14`: under the relative floor but, for this `y`, not always under
    /// the absolute one.
    #[allow(clippy::many_single_char_names)]
    fn orthogonal_y_design(seed: u64) -> (Mat<f64>, Mat<f64>, Col<f64>) {
        let (n, d) = (200, 10);
        let x = uniform_mat(n, d, seed);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let basis = orthonormal_basis(Col::<f64>::from_fn(n, |_| 1.0).as_ref(), xs.as_ref(), 0.0);
        let e = uniform_mat(n, 1, seed + 1);
        let mut y = Col::<f64>::from_fn(n, |i| 5.0 + e[(i, 0)]);
        project_off(&basis, &mut y);
        (x, xs, y)
    }

    #[test]
    fn nipals_drops_the_first_component_when_y_is_orthogonal_to_x() {
        // Without the floor on the first component the fit kept one
        // component whose `w` is normalized rounding noise.
        for seed in [41, 43, 45] {
            let (x, _, y) = orthogonal_y_design(seed);
            let m = pls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(3),
                None,
                FitOpts::default(),
            )
            .unwrap();
            assert_eq!(m.k_used, 0, "seed {seed}");
            assert_eq!(m.w_star.ncols(), 0);
            assert_eq!(m.q_loadings.nrows(), 0);
            assert!((0..10).all(|j| m.coef[j] == 0.0 && m.beta[j] == 0.0));
            // The zero model predicts the mean of y.
            let y_mean = (0..y.nrows()).map(|i| y[i]).sum::<f64>() / y.nrows() as f64;
            assert_relative_eq!(m.intercept, y_mean, epsilon = 1e-12);
            // Same decision on the sparse path.
            let s = spls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(1),
                4,
                None,
                FitOpts::default(),
            )
            .unwrap();
            assert_eq!(s.k_used, 0, "seed {seed} (spls1)");
        }
    }

    #[test]
    fn orthogonal_y_errors_when_strict_and_is_a_zero_model_when_internal() {
        // pre_standardized=true: a top-level call reports the truncation,
        // a per-iteration internal call gets the zero model.
        let (_, xs, y) = orthogonal_y_design(41);
        let (ys, _, _) = crate::linalg::standardize1(y.as_ref());
        let strict = pls1_fit(
            xs.as_ref(),
            ys.as_ref(),
            KSpec::Fixed(1),
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
        );
        assert!(
            matches!(strict, Err(PlsKitError::InvalidInput(_))),
            "expected InvalidInput, got {strict:?}"
        );
        let internal = pls1_fit(
            xs.as_ref(),
            ys.as_ref(),
            KSpec::Fixed(1),
            None,
            FitOpts {
                pre_standardized: true,
                check_n_eff: false,
                ..FitOpts::default()
            },
        )
        .unwrap();
        assert_eq!(internal.k_used, 0);
        assert!((0..10).all(|j| internal.coef[j] == 0.0));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nipals_keeps_a_tiny_real_first_component() {
        // Guard against an over-eager first-component floor: the orthogonal
        // y plus a true effect of 1e-9 of ‖y‖ along X keeps its component.
        let (x, xs, y_perp) = orthogonal_y_design(41);
        let (n, d) = (xs.nrows(), xs.ncols());
        let b = uniform_mat(d, 1, 47);
        let s = Col::<f64>::from_fn(n, |i| (0..d).map(|j| xs[(i, j)] * b[(j, 0)]).sum::<f64>());
        let scale = 1e-9 * y_perp.norm_l2() / s.norm_l2();
        let y = Col::<f64>::from_fn(n, |i| y_perp[i] + scale * s[i]);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(1),
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!(m.k_used, 1);
    }

    #[test]
    fn nipals_floor_keeps_every_component_of_full_signal_data() {
        // Guard against an over-eager floor: ordinary data with signal in
        // every direction keeps all requested components, weighted or not.
        let (x, y) = linear_data(100, 20, 20, 3);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(20),
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!(m.k_used, 20);
        let w = Col::<f64>::from_fn(100, |i| 0.5 + (i % 7) as f64 * 0.25);
        let mw = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(20),
            Some(w.as_ref()),
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!(mw.k_used, 20);
    }
}

#[cfg(test)]
mod kernel_tests;

/// `pls1_fit` at its public boundary: layout invariance (to rounding), input
/// screening, error codes and precedence, and the prepared tail the
/// resamplers use.
#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]
mod boundary_tests {
    use super::*;
    use crate::linalg::{
        normalize_weights, standardize1, standardize1_weighted, standardize_weighted,
    };
    use crate::test_support::{
        assert_agree, assert_bits_eq, assert_layout_agree, col_vals, copy_free_families, mat_vals,
        signal_data, Agree, Family,
    };

    /// Layouts of X agree to rounding, not bit for bit: `pls1_fit` forms its
    /// products from X in X's own layout (CHANGELOG 0.6.1), so each layout
    /// sums in its own order. `Agree::Corpus(1e-10)` is the corpus array
    /// tolerance, `1e-10 + 1e-14 · |value|`, which the Python docs promise
    /// across layouts.
    const FIT_LAYOUT_TOL: f64 = 1e-10;

    /// Every numeric output of a fit, flattened; `k_used` as an `f64` fails
    /// `Agree::Corpus(FIT_LAYOUT_TOL)` on any difference.
    fn model_vals(m: &Pls1Model) -> Vec<f64> {
        let mut v = mat_vals(m.t_scores.as_ref());
        v.extend(mat_vals(m.p_loadings.as_ref()));
        v.extend(mat_vals(m.w_star.as_ref()));
        v.extend(col_vals(m.q_loadings.as_ref()));
        v.extend(col_vals(m.coef.as_ref()));
        v.extend(col_vals(m.beta.as_ref()));
        #[allow(clippy::cast_precision_loss)]
        v.extend([m.intercept, m.n_eff, m.k_used as f64]);
        v
    }

    /// `pls1_fit` on every layout of `x` agrees with the owned column-major
    /// run to `FIT_LAYOUT_TOL`; returns the owned run's values. `keep` routes
    /// through `spls1_fit`'s path (`spls1_fit` only sets `FitOpts::keep`).
    fn fit_layouts(
        what: &str,
        x: &Mat<f64>,
        y: &Col<f64>,
        w: Option<ColRef<'_, f64>>,
        k: usize,
        opts: FitOpts,
    ) -> Vec<f64> {
        assert_layout_agree(x, what, Agree::Corpus(FIT_LAYOUT_TOL), |xv| {
            pls1_fit(xv, y.as_ref(), KSpec::Fixed(k), w, opts).map(|m| model_vals(&m))
        })
        .unwrap_or_else(|e| panic!("{what}: {e}"))
    }

    fn offset(x: &Mat<f64>, c: f64) -> Mat<f64> {
        Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] + c)
    }

    #[test]
    fn pls1_fit_is_layout_invariant_to_rounding() {
        let opts = |pre, keep, par| FitOpts {
            pre_standardized: pre,
            par,
            keep,
            ..FitOpts::default()
        };
        for f in copy_free_families() {
            let w = f.w.as_ref().map(Col::as_ref);
            for pre in [false, true] {
                let (x, y) = f.inputs(pre);
                for k in [1_usize, 3] {
                    for keep in [None, Some(3)] {
                        for par in [ParChoice::Seq, ParChoice::Auto] {
                            let what = format!("{} pre={pre} k={k} keep={keep:?} {par:?}", f.name);
                            fit_layouts(&what, &x, &y, w, k, opts(pre, keep, par));
                        }
                    }
                }
            }
        }
        // Sizes where `ParChoice::Auto` resolves to the Rayon path (at k = 3):
        // the owned Seq and Auto runs agree too (nothing else runs the Rayon
        // implicit products against Seq at this size). At 2100 columns
        // (`linalg::ROW_BLOCK_MIN_COLS` and up) the row-major view reads its
        // rows in blocks for the moments and `Xs'·t`
        // (`linalg::row_major_t_mul`); 203 rows leave a partial block, also in
        // each Rayon piece. At 2000 it keeps faer's product.
        for (n, d, seed) in [(200, 2000, 9), (203, 2100, 10)] {
            let (x, y) = signal_data(n, d, seed);
            // The row-major view of `Layouts` (the transpose of a stored
            // transpose) takes the blocked product exactly at 2100 columns,
            // under both arms.
            let x_t = x.transpose().to_owned();
            for par in [Par::Seq, crate::fit::par_fixed()] {
                assert_eq!(
                    crate::linalg::row_major_t_mul(x_t.transpose(), y.as_ref(), par).is_some(),
                    d == 2100,
                    "{n}x{d} {par:?}: left its row-major route"
                );
            }
            for k in [1_usize, 3] {
                for keep in [None, Some(3)] {
                    let what = format!("large {n}x{d} k={k} keep={keep:?}");
                    let seq = fit_layouts(
                        &format!("{what} Seq"),
                        &x,
                        &y,
                        None,
                        k,
                        opts(false, keep, ParChoice::Seq),
                    );
                    let auto = fit_layouts(
                        &format!("{what} Auto"),
                        &x,
                        &y,
                        None,
                        k,
                        opts(false, keep, ParChoice::Auto),
                    );
                    assert_agree(
                        &auto,
                        &seq,
                        Agree::Corpus(FIT_LAYOUT_TOL),
                        &format!("{what} Auto vs Seq"),
                    );
                }
            }
        }
        // Offset X on both sides of `IMPLICIT_MAX_MEAN_RATIO`: +300 on the
        // dense family stays on the implicit products, +1e6 on the weighted
        // family takes the standardized copy.
        let fams = copy_free_families();
        for (f, c, implicit) in [(&fams[0], 300.0, true), (&fams[1], 1e6, false)] {
            let x = offset(&f.x, c);
            let w = f.w.as_ref().map(Col::as_ref);
            let ratio = crate::linalg::fit_x_moments(x.as_ref(), w).max_mean_ratio();
            let what = format!("{} + {c:e} (ratio {ratio:.0})", f.name);
            assert_eq!(
                ratio > 100.0 && ratio < IMPLICIT_MAX_MEAN_RATIO,
                implicit,
                "{what}: left its route"
            );
            assert!(ratio > 100.0, "{what}: not mean-heavy");
            for k in [1_usize, 3] {
                fit_layouts(
                    &format!("{what} k={k}"),
                    &x,
                    &f.y,
                    w,
                    k,
                    opts(false, None, ParChoice::Seq),
                );
            }
        }
    }

    /// A non-finite entry is found on every layout, on both paths, weighted
    /// or not, also when its row has weight zero.
    #[test]
    fn non_finite_x_is_found_on_every_layout() {
        let (xl, yl) = signal_data(200, 2000, 9);
        let mut families = copy_free_families();
        families.push(Family {
            name: "large",
            x: xl,
            y: yl,
            w: None,
        });
        for f in &families {
            let n = f.x.nrows();
            let mut bad = f.x.clone();
            bad[(n - 1, f.x.ncols() - 1)] = f64::INFINITY;
            let zero_last = Col::<f64>::from_fn(n, |i| if i + 1 == n { 0.0 } else { 1.0 });
            for pre in [false, true] {
                for w in [None, Some(zero_last.as_ref())] {
                    let what = format!("{} pre={pre} weighted={}", f.name, w.is_some());
                    let opts = FitOpts {
                        pre_standardized: pre,
                        ..FitOpts::default()
                    };
                    let r = assert_layout_agree(&bad, &what, Agree::Bits, |xv| {
                        pls1_fit(xv, f.y.as_ref(), KSpec::Fixed(1), w, opts).map(|_| vec![])
                    });
                    assert!(
                        matches!(r, Err(PlsKitError::NonFiniteInput)),
                        "{what}: {r:?}"
                    );
                }
            }
        }
    }

    /// A `y` orthogonal to an offset X, below `IMPLICIT_MAX_MEAN_RATIO`: two
    /// offset columns (largest `|mean_j| / scale_j` about 270), and a centered
    /// column beside the offset column `a + 999.3` (ratio 999.3; a constant
    /// column no longer counts, its ratio is at most 1). The implicit products
    /// would put the first `‖X'y‖` above the floor and keep one component;
    /// the copy puts it below, and the fit decides on the copy: `k_used = 0`.
    #[test]
    #[allow(clippy::float_cmp)] // the zero model's intercept is mean(y) exactly
    fn orthogonal_y_on_offset_x_keeps_no_component() {
        use rand::{RngExt, SeedableRng};
        let designs = [[0.37, 99.7, -1.3, 89.73], [0.37, 0.0, 1.0, 999.3]];
        for (c, n) in designs
            .iter()
            .flat_map(|c| [8_usize, 12, 16].map(|n| (c, n)))
        {
            // Column j is `c[2j]·a + c[2j + 1]`, `a = ±1` alternating.
            let x = Mat::<f64>::from_fn(n, 2, |i, j| {
                let a = if i % 2 == 0 { 1.0 } else { -1.0 };
                c[2 * j] * a + c[2 * j + 1]
            });
            let ones = Col::<f64>::from_fn(n, |_| 1.0);
            let basis = crate::test_support::orthonormal_basis(ones.as_ref(), x.as_ref(), 1e-12);
            for seed in 0..20_u64 {
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                let mut y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
                crate::test_support::project_off(&basis, &mut y);
                let what = format!("{c:?} n={n} seed={seed}");
                let m = pls1_fit(
                    x.as_ref(),
                    y.as_ref(),
                    KSpec::Fixed(1),
                    None,
                    FitOpts::default(),
                )
                .unwrap();
                assert_eq!(m.k_used, 0, "{what}");
                assert_eq!(
                    (m.t_scores.ncols(), m.w_star.ncols(), m.q_loadings.nrows()),
                    (0, 0, 0),
                    "{what}"
                );
                assert!(
                    (0..2).all(|j| m.coef[j] == 0.0 && m.beta[j] == 0.0),
                    "{what}"
                );
                let (_, y_mean, _) = standardize1(y.as_ref());
                assert_eq!(
                    m.intercept, y_mean,
                    "{what}: the zero model predicts mean(y)"
                );
            }
        }
    }

    /// A pre-standardized X of finite entries whose squares overflow is not
    /// reported as non-finite: the sum-of-squares screen falls back to the
    /// exact scan and `norm_l2`.
    #[test]
    fn pre_standardized_huge_finite_x_is_not_non_finite() {
        let (x, y) = signal_data(40, 6, 3);
        let big = Mat::<f64>::from_fn(40, 6, |i, j| x[(i, j)] * 1e200);
        let (ys, _, _) = standardize1(y.as_ref());
        for check_n_eff in [true, false] {
            let opts = FitOpts {
                pre_standardized: true,
                check_n_eff,
                ..FitOpts::default()
            };
            let r = pls1_fit(big.as_ref(), ys.as_ref(), KSpec::Fixed(1), None, opts);
            assert!(!matches!(r, Err(PlsKitError::NonFiniteInput)), "{r:?}");
        }
    }

    /// Each fault gets its documented code (and, for `InvalidWeights`, its
    /// reason); with several faults at once the first check in `pls1_fit`'s
    /// order wins.
    #[test]
    fn pls1_fit_errors_have_documented_codes_in_precedence_order() {
        let (x, y) = signal_data(20, 5, 7);
        let mut bad_x = x.clone();
        bad_x[(3, 2)] = f64::NAN;
        let short_y = Col::<f64>::zeros(19);
        let zero_w = Col::<f64>::zeros(20);
        let neg_w = Col::<f64>::from_fn(20, |i| if i == 4 { -1.0 } else { 1.0 });
        let mut nan_y = y.clone();
        nan_y[6] = f64::NAN;
        let nan_w = Col::<f64>::from_fn(20, |i| if i == 8 { f64::NAN } else { 1.0 });
        let long_w = Col::<f64>::from_fn(21, |_| 1.0);
        // n_eff about 1: k = 2 needs n_eff >= 3.
        let one_row_w = Col::<f64>::from_fn(20, |i| if i == 0 { 1.0 } else { 1e-6 });
        let (x3, y3) = signal_data(3, 5, 7);
        let tiny_x = Mat::<f64>::from_fn(30, 4, |i, j| {
            1e-9 * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
        });
        let tiny_y = Col::<f64>::from_fn(30, |i| 1e-9 * (i as f64 - 15.0));
        let strict_pre = FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        };
        let d = FitOpts::default();
        let (ia, nf, iw) = ("invalid_argument", "non_finite_input", "invalid_weights");
        let cases: Vec<(
            &str,
            MatRef<'_, f64>,
            ColRef<'_, f64>,
            usize,
            Option<ColRef<'_, f64>>,
            FitOpts,
            &str,
            Option<&str>,
        )> = vec![
            ("k = 0", x.as_ref(), y.as_ref(), 0, None, d, ia, None),
            (
                "k > d",
                x.as_ref(),
                y.as_ref(),
                6,
                None,
                d,
                "k_exceeds_max",
                None,
            ),
            (
                "keep = 0",
                x.as_ref(),
                y.as_ref(),
                2,
                None,
                FitOpts { keep: Some(0), ..d },
                ia,
                None,
            ),
            (
                "keep > d",
                x.as_ref(),
                y.as_ref(),
                2,
                None,
                FitOpts { keep: Some(6), ..d },
                ia,
                None,
            ),
            ("NaN in X", bad_x.as_ref(), y.as_ref(), 2, None, d, nf, None),
            (
                "short y",
                x.as_ref(),
                short_y.as_ref(),
                2,
                None,
                d,
                "dimension_mismatch",
                None,
            ),
            (
                "all-zero weights",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(zero_w.as_ref()),
                d,
                iw,
                Some("all_zero"),
            ),
            (
                "negative weight",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(neg_w.as_ref()),
                d,
                iw,
                Some("negative"),
            ),
            ("NaN in y", x.as_ref(), nan_y.as_ref(), 2, None, d, nf, None),
            (
                "NaN weight",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(nan_w.as_ref()),
                d,
                nf,
                None,
            ),
            (
                "wrong-length weights",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(long_w.as_ref()),
                d,
                iw,
                Some("length_mismatch"),
            ),
            (
                "weight on one row",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(one_row_w.as_ref()),
                d,
                iw,
                Some("insufficient_effective_n"),
            ),
            // Unweighted, n < k + 1. (All-equal weights raise the same:
            // `equal_weights_are_absent_weights_at_every_entry`.)
            ("n < k + 1", x3.as_ref(), y3.as_ref(), 3, None, d, ia, None),
            // Several faults at once: the first check in `pls1_fit`'s order wins.
            (
                "NaN X + k = 0",
                bad_x.as_ref(),
                y.as_ref(),
                0,
                None,
                d,
                nf,
                None,
            ),
            (
                "short y + NaN X",
                bad_x.as_ref(),
                short_y.as_ref(),
                2,
                None,
                d,
                "dimension_mismatch",
                None,
            ),
            (
                "NaN y + all-zero weights",
                x.as_ref(),
                nan_y.as_ref(),
                2,
                Some(zero_w.as_ref()),
                d,
                nf,
                None,
            ),
            (
                "NaN y + keep = 0",
                x.as_ref(),
                nan_y.as_ref(),
                2,
                None,
                FitOpts { keep: Some(0), ..d },
                nf,
                None,
            ),
            // Review finding H1/N5 (ticket #3): with `pre_standardized = true`
            // and inputs so far below unit scale that ‖X'y‖ < 1e-14, the
            // kernel stops at the first component. A top-level call
            // (`check_n_eff = true`) reports the truncation instead of
            // returning a k_used = 0 model.
            (
                "strict truncation",
                tiny_x.as_ref(),
                tiny_y.as_ref(),
                3,
                None,
                strict_pre,
                "invalid_input",
                None,
            ),
        ];
        for (what, xv, yv, k, w, opts, code, reason) in cases {
            let e = pls1_fit(xv, yv, KSpec::Fixed(k), w, opts).expect_err(what);
            assert_eq!(e.code(), code, "{what}: {e}");
            if let Some(want) = reason {
                assert!(
                    matches!(&e, PlsKitError::InvalidWeights { reason } if *reason == want),
                    "{what}: {e:?}"
                );
            }
        }
        // The public sparse entry range-checks its keep-count the same way.
        for keep in [0, 6] {
            let e = spls1_fit(x.as_ref(), y.as_ref(), KSpec::Fixed(2), keep, None, d)
                .expect_err("spls1_fit keep");
            assert_eq!(e.code(), ia, "spls1_fit keep={keep}: {e}");
        }
    }

    /// Every public entry that takes weights and a component count, as its
    /// whole output's `Debug` (an `f64` prints its shortest round-trip form,
    /// so equal strings are equal bits) or its error code and message.
    /// `confirmatory` runs one call per listed method.
    fn entry_outcomes(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k: usize,
        w: Option<ColRef<'_, f64>>,
        confirmatory: &[crate::signal_test::ConfirmatoryArgs],
    ) -> Vec<(String, String)> {
        use crate::find_k::{
            pls1_find_k_optimal, pls1_find_k_sequence, spls1_find_keep_optimal, FindKOptimalOpts,
            FindKSequenceOpts, FindKeepOptimalOpts, Selector,
        };
        use crate::perm_null::{pls1_perm_null, PermNullOpts};
        use crate::rotation_stability::{
            pls1_rotation_stability, RotationStabilityMethod, RotationStabilityOpts,
        };
        use crate::signal_test::{
            pls1_confirmatory_test, ConfirmatoryTestInput, ConfirmatoryTestOpts,
        };
        fn show<T: std::fmt::Debug>(r: PlsKitResult<T>) -> String {
            match r {
                Ok(v) => format!("ok: {v:?}"),
                Err(e) => format!("{}: {e}", e.code()),
            }
        }
        let seed = Some(5);
        let mut out = vec![
            (
                "pls1_fit".to_owned(),
                show(pls1_fit(x, y, KSpec::Fixed(k), w, FitOpts::default())),
            ),
            (
                "perm_null".to_owned(),
                show(pls1_perm_null(
                    x,
                    y,
                    k,
                    w,
                    PermNullOpts {
                        n_perm: 100,
                        ..Default::default()
                    },
                    seed,
                )),
            ),
            (
                "rotation_stability".to_owned(),
                show(pls1_rotation_stability(
                    x,
                    y,
                    k,
                    RotationStabilityMethod::Varimax(crate::rotate::VarimaxArgs::default()),
                    None,
                    w,
                    RotationStabilityOpts {
                        n_boot: 100,
                        seed,
                        ..Default::default()
                    },
                )),
            ),
            (
                "find_k_sequence".to_owned(),
                show(pls1_find_k_sequence(
                    x,
                    y,
                    k,
                    w,
                    FindKSequenceOpts {
                        n_splits: 10,
                        seed,
                        ..Default::default()
                    },
                )),
            ),
            (
                "find_keep_optimal".to_owned(),
                show(spls1_find_keep_optimal(
                    x,
                    y,
                    k,
                    w,
                    FindKeepOptimalOpts {
                        seed,
                        ..Default::default()
                    },
                )),
            ),
        ];
        for selector in [Selector::R2Se, Selector::Bic] {
            let r = pls1_find_k_optimal(
                x,
                y,
                k,
                w,
                FindKOptimalOpts {
                    selector,
                    seed,
                    ..Default::default()
                },
            );
            out.push((format!("find_k_optimal {selector:?}"), show(r)));
        }
        for &args in confirmatory {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x,
                    y,
                    k,
                    weights: w,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed,
                    ..Default::default()
                },
            );
            out.push((format!("confirmatory {:?}", args.method()), show(r)));
        }
        out
    }

    /// All-equal weights are no weights, at every top-level entry that
    /// takes a component count: at n < k + 1 they raise the unweighted
    /// `invalid_argument` (not `insufficient_effective_n`), and at a
    /// feasible n every output is bit-identical to the call without
    /// weights: `rho_hat`, `n_eff`, the replicate route (n = 40, p = 60
    /// puts `perm_null` on the n-space route) and the BIC penalty included.
    /// 0.3 is there because Kish's ratio of equal non-unit weights rounds
    /// an ulp below n (2.999999999999999 at n = 3).
    #[test]
    fn equal_weights_are_absent_weights_at_every_entry() {
        use crate::signal_test::ConfirmatoryArgs;
        let every_method = [
            ConfirmatoryArgs::RawPerm {
                n_perm: 19,
                n_folds: 5,
            },
            ConfirmatoryArgs::SplitNb {
                n_splits: 10,
                force: false,
            },
            ConfirmatoryArgs::SplitExact {
                n_perm: 19,
                n_splits: 10,
            },
            ConfirmatoryArgs::Score,
            ConfirmatoryArgs::E,
        ];
        let tiny = signal_data(3, 5, 7);
        let feasible = signal_data(40, 60, 8);
        // The n < k + 1 case runs one method: `raw_perm`'s own n_folds < n
        // floor would fire first and pass for the wrong reason.
        for ((x, y), k, methods, fails) in [
            (&tiny, 3, &every_method[2..3], true),
            (&feasible, 2, &every_method[..], false),
        ] {
            let n = x.nrows();
            let absent = entry_outcomes(x.as_ref(), y.as_ref(), k, None, methods);
            for (entry, got) in &absent {
                let ok = if fails {
                    got.starts_with("invalid_argument: ") && got.contains("insufficient n for k=")
                } else {
                    got.starts_with("ok: ")
                };
                assert!(ok, "{entry}: {got}");
            }
            for c in [1.0, 0.3, 1e6] {
                let w = Col::<f64>::from_fn(n, |_| c);
                let equal = entry_outcomes(x.as_ref(), y.as_ref(), k, Some(w.as_ref()), methods);
                for ((entry, a), (_, e)) in absent.iter().zip(&equal) {
                    assert!(
                        a == e,
                        "n = {n}, w = {c}: {entry}\nabsent: {a}\nequal:  {e}"
                    );
                }
            }
        }
    }

    /// The public entries that take X, besides the PLS1 fits and the
    /// engines whose own tests pin layout bit-identity (the PLS3 family's
    /// is `pls3::tests::pls3_family_is_bit_identical_across_layouts`). The
    /// Python wrapper hands each its X in place (C or F order).
    /// `preprocess` / `preprocess_block` and the `split_nb` gate give the
    /// same bits on every layout. `pls1_predict` and
    /// `pls1_rotation_stability` (whose reference fits read X in place)
    /// agree to `FIT_LAYOUT_TOL`.
    #[test]
    fn x_taking_entries_agree_across_layouts() {
        use crate::preprocess::{
            preprocess, preprocess_block, PreprocessBlockInput, PreprocessInput,
        };
        use crate::rotation_stability::{
            pls1_rotation_stability, RotationStabilityMethod, RotationStabilityOpts,
        };
        use crate::signal_test::split_nb_gate;
        let agree = Agree::Corpus(FIT_LAYOUT_TOL);
        let bits = Agree::Bits;
        for f in copy_free_families() {
            let w = f.w.as_ref().map(Col::as_ref);
            let n = f.x.nrows();
            let yy = Mat::<f64>::from_fn(n, 3, |i, j| f.y[i] * (j + 1) as f64 + f.x[(i, j)]);
            let what = |e: &str| format!("{e} {}", f.name);
            assert_layout_agree(&f.x, &what("preprocess"), bits, |xv| {
                let r = preprocess(PreprocessInput {
                    x: Some(xv),
                    y: Some(f.y.as_ref()),
                    weights: w,
                })?;
                let (xs, m, sd) = r.x_std.unwrap();
                let mut v = mat_vals(xs.as_ref());
                v.extend(col_vals(m.as_ref()));
                v.extend(col_vals(sd.as_ref()));
                Ok(v)
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("preprocess_block"), bits, |xv| {
                let r = preprocess_block(PreprocessBlockInput {
                    x: Some(xv),
                    y: Some(yy.as_ref()),
                    weights: w,
                })?;
                Ok(mat_vals(r.x_std.unwrap().0.as_ref()))
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("split_nb_gate"), bits, |xv| {
                let g = split_nb_gate(xv, w)?;
                Ok(vec![f64::from(u8::from(g.fires)), g.stable_rank, g.n_eff])
            })
            .unwrap();
            let model = pls1_fit(
                f.x.as_ref(),
                f.y.as_ref(),
                KSpec::Fixed(2),
                w,
                FitOpts::default(),
            )
            .unwrap();
            assert_layout_agree(&f.x, &what("pls1_predict"), agree, |xv| {
                Ok(col_vals(crate::predict::pls1_predict(&model, xv)?.as_ref()))
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("pls1_rotation_stability"), agree, |xv| {
                let r = pls1_rotation_stability(
                    xv,
                    f.y.as_ref(),
                    2,
                    RotationStabilityMethod::Varimax(crate::rotate::VarimaxArgs::default()),
                    None,
                    w,
                    RotationStabilityOpts {
                        n_boot: 100,
                        seed: Some(5),
                        ..Default::default()
                    },
                )?;
                let ci = |c: &crate::subsample::CIScalar| [c.point, c.lower, c.upper, c.sd];
                let mut v = ci(&r.variance_ratio).to_vec();
                v.extend(r.variance_ratio_per_axis.iter().flat_map(ci));
                v.extend([r.variance_unrot, r.variance_rot, r.n_eff]);
                v.extend(&r.variance_unrot_per_axis);
                v.extend(&r.variance_rot_per_axis);
                #[allow(clippy::cast_precision_loss)]
                v.push(r.n_boot_finite as f64);
                Ok(v)
            })
            .unwrap();
        }
    }

    /// A zero count is `invalid_argument` at every public entry, with one
    /// message per argument: `"{arg} must be >= 1"` (`k`, `k_max`, and the
    /// keep-counts, which add why). `k_exceeds_max` is for a count above its
    /// maximum only. The `find_k` rows of `entry_outcomes` pass their `k` as
    /// `k_max`; `pls3_confirmatory_test` says this rather than its own
    /// `k = 1` restriction. `score` is rejected too although its statistic
    /// never reads k; `split_nb` is forced so the gate cannot reroute it.
    #[test]
    fn zero_count_is_invalid_argument_at_every_entry() {
        use crate::find_k::{
            spls1_find_k_optimal, spls1_find_k_sequence, FindKOptimalOpts, FindKSequenceOpts,
        };
        use crate::pls3::{pls3_fit, plssvd_fit, spls3_fit, Pls3FitOpts};
        use crate::pls3_signal_test::{pls3_confirmatory_test, Pls3ConfirmatoryTestOpts};
        use crate::signal_test::ConfirmatoryArgs;
        fn show<T>(r: PlsKitResult<T>) -> String {
            match r {
                Ok(_) => "ok".to_owned(),
                Err(e) => format!("{}: {e}", e.code()),
            }
        }
        let (x, y) = signal_data(40, 6, 3);
        let (x, y) = (x.as_ref(), y.as_ref());
        let yy = Mat::<f64>::from_fn(40, 3, |i, j| x[(i, j)] + x[(i, 5 - j)]);
        let yy = yy.as_ref();
        let every_method = [
            ConfirmatoryArgs::RawPerm {
                n_perm: 19,
                n_folds: 5,
            },
            ConfirmatoryArgs::SplitNb {
                n_splits: 10,
                force: true,
            },
            ConfirmatoryArgs::SplitExact {
                n_perm: 19,
                n_splits: 10,
            },
            ConfirmatoryArgs::Score,
            ConfirmatoryArgs::E,
        ];
        let seed = Some(5);
        let fk = || FindKOptimalOpts {
            seed,
            ..Default::default()
        };
        let fs = || FindKSequenceOpts {
            n_splits: 10,
            seed,
            ..Default::default()
        };
        let p3 = Pls3FitOpts::default;
        let p3c = |keep_x, keep_y| Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 19,
                n_splits: 6,
            },
            seed,
            keep_x,
            keep_y,
            ..Default::default()
        };
        let mut rows: Vec<(String, &str, String)> = entry_outcomes(x, y, 0, None, &every_method)
            .into_iter()
            .map(|(entry, got)| {
                let arg = if entry.starts_with("find_k_") {
                    "k_max"
                } else {
                    "k"
                };
                (entry, arg, got)
            })
            .collect();
        let d = FitOpts::default();
        let mut row = |entry: &str, arg: &'static str, got: String| {
            rows.push((entry.to_owned(), arg, got));
        };
        row(
            "spls1_fit",
            "k",
            show(spls1_fit(x, y, KSpec::Fixed(0), 3, None, d)),
        );
        row(
            "spls1_fit keep",
            "keep",
            show(spls1_fit(x, y, KSpec::Fixed(2), 0, None, d)),
        );
        row(
            "spls1_find_k_optimal",
            "k_max",
            show(spls1_find_k_optimal(x, y, 0, 3, None, fk())),
        );
        row(
            "spls1_find_k_optimal keep",
            "keep",
            show(spls1_find_k_optimal(x, y, 2, 0, None, fk())),
        );
        row(
            "spls1_find_k_sequence",
            "k_max",
            show(spls1_find_k_sequence(x, y, 0, 3, None, fs())),
        );
        row(
            "spls1_find_k_sequence keep",
            "keep",
            show(spls1_find_k_sequence(x, y, 2, 0, None, fs())),
        );
        row("pls3_fit", "k", show(pls3_fit(x, yy, 0, None, p3())));
        row("plssvd_fit", "k", show(plssvd_fit(x, yy, 0, None, p3())));
        row(
            "spls3_fit",
            "k",
            show(spls3_fit(x, yy, 0, 2, 2, None, p3())),
        );
        row(
            "spls3_fit keep_x",
            "keep_x",
            show(spls3_fit(x, yy, 1, 0, 2, None, p3())),
        );
        row(
            "spls3_fit keep_y",
            "keep_y",
            show(spls3_fit(x, yy, 1, 2, 0, None, p3())),
        );
        row(
            "pls3_confirmatory_test",
            "k",
            show(pls3_confirmatory_test(x, yy, 0, p3c(None, None))),
        );
        row(
            "pls3_confirmatory_test keep_x",
            "keep_x",
            show(pls3_confirmatory_test(x, yy, 1, p3c(Some(0), None))),
        );
        row(
            "pls3_confirmatory_test keep_y",
            "keep_y",
            show(pls3_confirmatory_test(x, yy, 1, p3c(None, Some(0)))),
        );
        for (entry, arg, got) in rows {
            let want = format!("invalid_argument: invalid argument: {arg} must be >= 1");
            assert!(got.starts_with(&want), "{entry}: {got}");
        }
    }

    #[test]
    fn prepared_tail_on_hoisted_scaling_is_pls1_fit() {
        for f in copy_free_families() {
            let Some(w_raw) = f.w.as_ref() else {
                continue;
            };
            // The caller's weights as `perm_null` and the CV folds hold them.
            let wn = normalize_weights(w_raw.as_ref()).unwrap();
            let (xs, _, _) = standardize_weighted(f.x.as_ref(), Some(wn.as_ref()));
            let (ys, _, _) = standardize1_weighted(f.y.as_ref(), Some(wn.as_ref()));
            let sqw = fit_row_scale(wn.as_ref());
            let xs_fit = scale_rows(xs.as_ref(), sqw.as_ref());
            let ys_fit = scale_col(ys.as_ref(), sqw.as_ref());
            for i in 0..xs.nrows() {
                assert_eq!(xs_fit[(i, 1)].to_bits(), (sqw[i] * xs[(i, 1)]).to_bits());
                assert_eq!(ys_fit[i].to_bits(), (sqw[i] * ys[i]).to_bits());
            }
            // Taken once for the block, as `perm_null` and the CV folds do.
            let x_fro = xs_fit.norm_l2();
            for k in [1_usize, 3] {
                for keep in [None, Some(3)] {
                    let opts = FitOpts {
                        pre_standardized: true,
                        check_n_eff: false,
                        par: ParChoice::Seq,
                        keep,
                    };
                    let direct = pls1_fit(
                        xs.as_ref(),
                        ys.as_ref(),
                        KSpec::Fixed(k),
                        Some(wn.as_ref()),
                        opts,
                    )
                    .unwrap();
                    let hoisted = pls1_fit_prepared_fro(
                        xs_fit.as_ref(),
                        ys_fit.as_ref(),
                        k,
                        keep,
                        ParChoice::Seq,
                        x_fro,
                    )
                    .unwrap();
                    let what = format!("{} k={k} keep={keep:?}", f.name);
                    assert_eq!(direct.k_used, hoisted.k_used, "{what}");
                    assert_bits_eq(
                        &mat_vals(direct.t_scores.as_ref()),
                        &mat_vals(hoisted.t_scores.as_ref()),
                        &what,
                    );
                    assert_bits_eq(
                        &mat_vals(direct.w_star.as_ref()),
                        &mat_vals(hoisted.w_star.as_ref()),
                        &what,
                    );
                    assert_bits_eq(
                        &col_vals(direct.coef.as_ref()),
                        &col_vals(hoisted.coef.as_ref()),
                        &what,
                    );
                    // pre_standardized: beta is coef.
                    assert_bits_eq(
                        &col_vals(direct.beta.as_ref()),
                        &col_vals(hoisted.coef.as_ref()),
                        &what,
                    );
                }
            }
        }
    }
}
