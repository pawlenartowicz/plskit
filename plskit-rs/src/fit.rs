//! NIPALS PLS1 fit. Public entry point: `pls1_fit`.

use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::{Accum, Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};

/// How `pls1_fit` decides how many components to extract.
#[derive(Debug, Clone, Copy)]
pub enum KSpec {
    /// Fixed component count requested by the caller.
    Fixed(usize),
}

/// Parallelism strategy for the NIPALS kernel called by `pls1_fit`.
///
/// `Auto` (the default) selects per-fit based on problem size:
/// runs sequentially when `n * d * k < 1_000_000` and on the global
/// rayon threadpool otherwise. The `1_000_000` threshold reflects the
/// crossover measured on Arrow Lake-H — at smaller sizes faer's matmul
/// dispatch over-eagerly parallelizes and the thread-overhead dominates
/// (~1.8× slowdown observed at `(200, 800, 5)`).
///
/// Resamplers (`pls1_perm_null`, `pls1_rotation_stability`,
/// `pls1_confirmatory_test`, `pls1_find_k_*`) force `Seq` for the inner
/// fits — outer Rayon is already saturating the cores, and nested
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
    /// **Scale contract.** When `pre_standardized=true`, the NIPALS kernel
    /// uses a fixed `1e-14` absolute threshold on the per-component norms
    /// of `X'y` and `Xw`. If raw data is scaled below ~`1e-7` (frobenius
    /// norm of `X` < `1e-6`), the loop short-circuits at the first
    /// component and the fit silently returns a zero-beta model. Callers
    /// passing `pre_standardized=true` must ensure the inputs are
    /// genuinely zero-mean / unit-variance (or at least scale-comparable
    /// to that). The default `pre_standardized=false` path absorbs raw
    /// scale automatically and is unaffected by this contract.
    ///
    /// As a guard, when `pre_standardized=true` and `check_n_eff=true`
    /// (the default for top-level public entry points), `pls1_fit`
    /// returns `InvalidInput` if NIPALS truncates below the requested `k`.
    /// Per-iteration internal callers (CV folds, per-half split fits,
    /// permutation refits, the BIC full-k fit, the sequential deflation
    /// fit) set `check_n_eff=false` and tolerate truncation by design.
    pub pre_standardized: bool,
    /// When true (default), `pls1_fit` errors with `InvalidWeights{reason:"insufficient_effective_n"}`
    /// (weighted inputs) or `InvalidArgument` (uniform/absent weights)
    /// if `n_eff < k + 1`. Set to false for per-iteration internal calls (CV folds,
    /// bootstrap subsamples) where the upstream accumulator handles degeneracy.
    /// See `_docs/concepts/effective-sample-size.md`.
    pub check_n_eff: bool,
    /// Parallelism strategy for the NIPALS kernel. See `ParChoice`.
    pub par: ParChoice,
    /// Sparse keep-count (spls1 family plumbing): retain the `keep`
    /// largest-|w| coordinates per component, zero the rest — hard selection
    /// at the keep-th order statistic of |w|; exact ties break by lowest
    /// column index (reproducibility contract). `None` (default) = dense
    /// NIPALS. Wrapper surfaces never expose this on dense functions;
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

/// Translate a `ParChoice` into a concrete `faer::Par` for the given problem size.
///
/// `ParChoice::Seq` always maps to `Par::Seq`; `ParChoice::Auto` uses
/// `Par::rayon(0)` (default thread pool) when `n * d * k ≥ 1_000_000`,
/// else `Par::Seq`. Saturating arithmetic guards against `usize` overflow
/// on absurd inputs.
fn resolve_par(choice: ParChoice, n: usize, d: usize, k: usize) -> Par {
    match choice {
        ParChoice::Seq => Par::Seq,
        ParChoice::Auto => {
            let work = n.saturating_mul(d).saturating_mul(k);
            if work >= 1_000_000 {
                Par::rayon(0)
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
    /// or absent — see spec §3.6. Length = `n_samples` when present.
    pub weights: Option<Col<f64>>,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
    /// Resolved sparse keep-count. `None` for dense fits — mirrors the
    /// `weights: Option<Col<f64>>` precedent.
    pub keep: Option<usize>,
}

/// Validate weights and produce `(normalized vector, n_eff, all_uniform_flag)`.
/// Returns `Ok((None, n as f64, true))` when `weights` is `None`.
///
/// # Errors
/// - `InvalidWeights { reason: "length_mismatch" }` if `weights.len() != n`
/// - `NonFiniteInput` for any NaN / infinity
/// - `InvalidWeights { reason: "negative" }` for any `w < 0`
/// - `InvalidWeights { reason: "all_zero" }` if `Σw == 0`
///
/// The `all_uniform` flag is `true` when post-normalization every entry equals 1.0 (within 1e-12).
/// Callers should echo `None` for `weights` on the result struct when this flag is set
/// (uniform-weight invariance, spec §3.6).
pub(crate) fn validate_and_normalize_weights(
    weights: Option<ColRef<'_, f64>>,
    n: usize,
    k_requested: usize,
) -> PlsKitResult<(Option<Col<f64>>, f64, bool)> {
    let Some(w) = weights else {
        #[allow(clippy::cast_precision_loss)]
        return Ok((None, n as f64, true));
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
    let n_eff = crate::linalg::compute_n_eff(w);
    let _ = k_requested; // n_eff check moved to check_n_eff_for_k; see _docs/concepts/effective-sample-size.md
    let max_dev = (0..n).map(|i| (wn[i] - 1.0).abs()).fold(0.0_f64, f64::max);
    let all_uniform = max_dev < 1e-12;
    Ok((Some(wn), n_eff, all_uniform))
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
    // j-outer/i-inner matches faer's column-major storage (cache-friendly sweep).
    // Mirrors `rotate::mat_is_finite` — same traversal, bool vs Result signature is
    // the only difference.
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
/// When `weighted=true` (observation weights were supplied), returns
/// `Err(InvalidWeights { reason: "insufficient_effective_n" })` when `n_eff < k + 1`.
/// When `weighted=false` (uniform / absent weights), returns
/// `Err(InvalidArgument)` — no weights are in play, so the failure is a plain
/// data-size problem, not a weights problem (callers branch on `code()`).
/// Called at every TOP-LEVEL public entry that takes weights;
/// NOT called by per-iteration internals (CV folds, bootstrap subsamples,
/// permutation refits) — see `_docs/concepts/effective-sample-size.md`.
///
/// # Errors
/// `InvalidWeights { reason: "insufficient_effective_n" }` (weighted) or
/// `InvalidArgument` (unweighted) when `n_eff < k + 1`.
pub(crate) fn check_n_eff_for_k(n_eff: f64, k: usize, weighted: bool) -> PlsKitResult<()> {
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

/// Validate a sparse keep-count against the feature dimension (spec:
/// keep ∈ \[1, `n_features`]; keep = `n_features` is the dense special case).
///
/// # Errors
/// `InvalidArgument` for `keep == 0` (empty component) or `keep > n_features`.
pub(crate) fn validate_keep(keep: usize, n_features: usize) -> PlsKitResult<()> {
    if keep == 0 {
        return Err(PlsKitError::InvalidArgument(
            "keep must be >= 1 (keep=0 would produce an empty component)".into(),
        ));
    }
    if keep > n_features {
        return Err(PlsKitError::InvalidArgument(format!(
            "keep={keep} exceeds n_features={n_features}"
        )));
    }
    Ok(())
}

/// Fit a PLS1 regression by NIPALS.
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
    if y.nrows() != n_samples {
        return Err(PlsKitError::DimensionMismatch {
            x: (n_samples, n_features),
            y: y.nrows(),
        });
    }
    check_finite_mat(x)?;
    check_finite_col(y)?;

    let KSpec::Fixed(k_requested) = k;

    if k_requested == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }

    if k_requested > n_features {
        return Err(PlsKitError::KExceedsMax {
            k: k_requested,
            k_max: n_features,
        });
    }

    if let Some(kp) = opts.keep {
        validate_keep(kp, n_features)?;
    }

    // Validate + normalize weights (spec §3.3, §3.4).
    let (w_norm, n_eff_val, all_uniform) =
        validate_and_normalize_weights(weights, n_samples, k_requested)?;
    if opts.check_n_eff {
        check_n_eff_for_k(n_eff_val, k_requested, weights.is_some())?;
    }
    let wref: Option<ColRef<'_, f64>> = w_norm.as_ref().map(Col::as_ref);

    // Standardize OR skip (spec §4.2). Use weighted versions when weights is Some.
    let (xs_owned, x_mean, x_scale, ys_owned, y_mean, y_scale) = if opts.pre_standardized {
        (
            None,
            Col::<f64>::zeros(n_features),
            Col::<f64>::from_fn(n_features, |_| 1.0),
            None,
            0.0,
            1.0,
        )
    } else {
        let (xs, m, s) = crate::linalg::standardize_weighted(x, wref);
        let (zs, ym, ysc) = crate::linalg::standardize1_weighted(y, wref);
        (Some(xs), m, s, Some(zs), ym, ysc)
    };

    let xs_view: MatRef<'_, f64> = match &xs_owned {
        Some(a) => a.as_ref(),
        None => x,
    };
    let ys_view: ColRef<'_, f64> = match &ys_owned {
        Some(a) => a.as_ref(),
        None => y,
    };

    // Apply √w' row-scaling — spec §4.2: row-scaling is the Cholesky factor,
    // *not* preprocessing, so it runs even when pre_standardized=true.
    let (x_scaled_owned, y_scaled_owned): (Option<Mat<f64>>, Option<Col<f64>>) = match wref {
        None => (None, None),
        Some(w) => {
            let sqw: Vec<f64> = (0..n_samples).map(|i| w[i].sqrt()).collect();
            let xt = Mat::<f64>::from_fn(n_samples, n_features, |i, j| sqw[i] * xs_view[(i, j)]);
            let yt = Col::<f64>::from_fn(n_samples, |i| sqw[i] * ys_view[i]);
            (Some(xt), Some(yt))
        }
    };

    let x_for_nipals: MatRef<'_, f64> = match &x_scaled_owned {
        Some(a) => a.as_ref(),
        None => xs_view,
    };
    let y_for_nipals: ColRef<'_, f64> = match &y_scaled_owned {
        Some(a) => a.as_ref(),
        None => ys_view,
    };

    let par = resolve_par(opts.par, n_samples, n_features, k_requested);
    let (t_mat, p_mat, w_mat, q_vec) =
        nipals_pls1(x_for_nipals, y_for_nipals, k_requested, opts.keep, par)?;

    let k_used = w_mat.ncols();
    if opts.pre_standardized && opts.check_n_eff && k_used < k_requested {
        return Err(PlsKitError::InvalidInput(format!(
            "pls1_fit(pre_standardized=true) truncated to k_used={k_used} < requested k={k_requested}: \
             NIPALS short-circuited on the {kth} component (norm < 1e-14). Either X is \
             rank-deficient (fewer than k informative directions — lower k), or the inputs \
             violate the pre_standardized scale contract (see `FitOpts::pre_standardized`): \
             re-fit with `pre_standardized=false` to let plskit standardize, or rescale your \
             inputs so that ‖X‖_F ≥ 1e-6.",
            kth = k_used + 1
        )));
    }
    let coef = pls1_coef_at_k(&w_mat, &p_mat, &q_vec, k_used, par);

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
        t_scores: t_mat,
        p_loadings: p_mat,
        w_star: w_mat,
        q_loadings: q_vec,
        coef,
        beta,
        intercept,
        k_used,
        pre_standardized: opts.pre_standardized,
        weights: if all_uniform { None } else { w_norm },
        n_eff: n_eff_val,
        keep: opts.keep,
    })
}

/// Sparse PLS1 fit — head of the `spls1_*` family. NIPALS with hard
/// keep-count selection on the weight vector per component (Chun & Keleş
/// 2010 lineage, keep-count formulation): each latent direction loads on
/// exactly `keep` X variables. Everything downstream of the selection step —
/// scores, loadings, deflation, `coef = W(P'W)⁻¹Q`, raw-scale β, intercept —
/// is byte-identical to `pls1_fit`; `keep = n_features` reduces bit-exactly
/// to the dense fit.
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

/// Zero all but the `keep` largest-|w| coordinates (hard thresholding at
/// the keep-th order statistic of |w| — NOT soft thresholding; survivors
/// keep their magnitudes). Exact ties break deterministically: order by
/// (|w| desc, index asc), so the lowest column index wins. An unstable
/// partial select (`select_nth_unstable_by`) would NOT honor this —
/// selection must not depend on sort order (reproducibility contract).
fn hard_select_keep(w: &mut Col<f64>, keep: usize) {
    let d = w.nrows();
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_unstable_by(|&a, &b| w[b].abs().total_cmp(&w[a].abs()).then_with(|| a.cmp(&b)));
    for &j in &idx[keep..] {
        w[j] = 0.0;
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::type_complexity)]
#[allow(clippy::unnecessary_wraps)] // reserved for future variants that may return Err
fn nipals_pls1(
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
        // existing < 1e-14 break. The `kp < d` guard makes the dense endpoint
        // (keep == n_features) a LITERAL skip — the dense float sequence is
        // provably untouched (bit-parity tripwire), not merely value-preserving.
        if let Some(kp) = keep {
            if kp < d {
                hard_select_keep(&mut w, kp);
            }
        }
        let w_norm = w.norm_l2();
        if w_norm < 1e-14 {
            break;
        }
        let inv_w_norm = 1.0 / w_norm;
        for j in 0..d {
            w[j] *= inv_w_norm;
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
        if tt < 1e-14 {
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

/// Regression coefficient using first `k` PLS components.
/// Formula: coef = W (P'W)^{-1} Q.
///
/// `par` threads the two GEMMs (`P'W` and `W·z`) explicitly: operator-`*`
/// would dispatch on faer's global parallelism (default `Rayon`), which
/// leaks into the global pool when this runs inside a resampler's Rayon
/// worker. The K×K `PartialPivLu` stays on the high-level API — faer's LU
/// `par_threshold` (128²) keeps a K≤~20 factor on `Par::Seq` regardless.
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
    let lu = PartialPivLu::new(pwk.as_ref());
    let z: Col<f64> = lu.solve(&qk);
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
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
    fn fit_recovers_signal_directionally() {
        let (x, y) = linear_data(200, 8, 3, 1);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let y_hat: Col<f64> = &x * &m.beta;
        let y_mean: f64 = (0..y.nrows()).map(|i| y[i]).sum::<f64>() / y.nrows() as f64;
        let ss_tot: f64 = (0..y.nrows()).map(|i| (y[i] - y_mean).powi(2)).sum();
        let ss_res: f64 = (0..y.nrows())
            .map(|i| (y[i] - (y_hat[i] + m.intercept)).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.9, "R² too low: {r2}");
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

    #[test]
    fn fit_dimension_mismatch_errors() {
        let x = Mat::<f64>::zeros(10, 5);
        let y = Col::<f64>::zeros(9);
        let err = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        );
        assert!(matches!(err, Err(PlsKitError::DimensionMismatch { .. })));
    }

    #[test]
    fn fit_k_exceeds_max_errors() {
        let (x, y) = linear_data(20, 5, 2, 1);
        let err = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(20),
            None,
            FitOpts::default(),
        );
        assert!(matches!(err, Err(PlsKitError::KExceedsMax { .. })));
    }

    #[test]
    fn pls1_fit_rejects_k_zero() {
        let (x, y) = linear_data(20, 5, 2, 1);
        let err = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(0),
            None,
            FitOpts::default(),
        );
        assert!(
            matches!(err, Err(PlsKitError::InvalidArgument(_))),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pre_standardized_below_scale_contract_errors_when_strict() {
        // Regression for review-finding H1/N5 (ticket #3): with
        // `pre_standardized=true` and inputs scaled so far below 1.0 that
        // ‖X'y‖ < 1e-14, NIPALS short-circuits at the first component.
        // Before the guard, this returned a finite k_used=0 model;
        // afterwards it returns InvalidInput because `check_n_eff=true`.
        let n = 30;
        let d = 4;
        let x = Mat::<f64>::from_fn(n, d, |i, j| {
            // Roughly orthogonal columns at amplitude ~1e-9.
            let s = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            1e-9 * s
        });
        let y = Col::<f64>::from_fn(n, |i| 1e-9 * (i as f64 - 15.0));
        let err = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
        );
        assert!(
            matches!(err, Err(PlsKitError::InvalidInput(_))),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pre_standardized_below_scale_contract_silent_when_internal() {
        // Mirror of the above with `check_n_eff=false`: per-fold internal
        // callers must keep their fail-soft behavior (truncated k_used,
        // upstream aggregator handles it). The guard MUST stay opt-in.
        let n = 30;
        let d = 4;
        let x = Mat::<f64>::from_fn(n, d, |i, j| {
            let s = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            1e-9 * s
        });
        let y = Col::<f64>::from_fn(n, |i| 1e-9 * (i as f64 - 15.0));
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts {
                pre_standardized: true,
                check_n_eff: false,
                ..FitOpts::default()
            },
        )
        .expect("internal-style call must not error on truncation");
        assert!(m.k_used < 3, "expected truncation; got k_used={}", m.k_used);
    }

    // ── spls1 sparse kernel ──────────────────────────────────────────

    #[test]
    fn spls1_keep_eq_n_features_is_bit_identical_to_dense() {
        // THE bit-parity tripwire (spec): keep = n_features must be a literal
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

    #[test]
    fn spls1_rejects_keep_zero_and_keep_gt_d() {
        let (x, y) = linear_data(20, 5, 2, 1);
        let e0 = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            0,
            None,
            FitOpts::default(),
        );
        assert!(
            matches!(e0, Err(PlsKitError::InvalidArgument(_))),
            "keep=0: {e0:?}"
        );
        let e6 = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            6,
            None,
            FitOpts::default(),
        );
        assert!(
            matches!(e6, Err(PlsKitError::InvalidArgument(_))),
            "keep>d: {e6:?}"
        );
    }

    #[test]
    fn dense_pls1_fit_has_keep_none() {
        let (x, y) = linear_data(30, 5, 2, 1);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!(m.keep, None);
    }
}
