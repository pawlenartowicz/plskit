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

/// Knobs for `pls1_fit`.
#[derive(Debug, Clone, Copy)]
pub struct FitOpts {
    /// Skip the centering/scaling step; caller asserts X and y are already standardized.
    pub pre_standardized: bool,
    /// Convergence tolerance. Unused for PLS1 (single-pass), reserved for symmetric variants.
    pub tol: f64,
    /// Iteration cap. Unused for PLS1, reserved for future variants.
    pub max_iter: usize,
    /// When true (default), `pls1_fit` errors with `InvalidWeights{reason:"insufficient_effective_n"}`
    /// if `n_eff < k + 1`. Set to false for per-iteration internal calls (CV folds,
    /// bootstrap subsamples) where the upstream accumulator handles degeneracy.
    /// See `_docs/concepts/effective-sample-size.md`.
    pub check_n_eff: bool,
}

impl Default for FitOpts {
    fn default() -> Self {
        Self {
            pre_standardized: false,
            tol: 1e-9,
            max_iter: 500,
            check_n_eff: true,
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
}

/// Validate weights and produce `(normalized vector, n_eff, all_uniform_flag)`.
/// Returns `Ok((None, n as f64, true))` when `weights` is `None`.
///
/// # Errors
/// - `DimensionMismatch` if `weights.len() != n`
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
        return Err(PlsKitError::DimensionMismatch {
            x: (n, 0),
            y: w.nrows(),
        });
    }
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

/// Check that effective sample size supports the requested number of components.
///
/// Returns `Err(InvalidWeights { reason: "insufficient_effective_n" })` when
/// `n_eff < k + 1`. Called at every TOP-LEVEL public entry that takes weights;
/// NOT called by per-iteration internals (CV folds, bootstrap subsamples,
/// permutation refits) — see `_docs/concepts/effective-sample-size.md`.
///
/// # Errors
/// `InvalidWeights { reason: "insufficient_effective_n" }` when `n_eff < k + 1`.
pub(crate) fn check_n_eff_for_k(n_eff: f64, k: usize) -> PlsKitResult<()> {
    #[allow(clippy::cast_precision_loss)]
    if n_eff < (k as f64) + 1.0 {
        return Err(PlsKitError::InvalidWeights {
            reason: "insufficient_effective_n",
        });
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
/// - `PlsKitError::DimensionMismatch` when `y.nrows() != x.nrows()` or `weights.len() != n`
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
    let x_finite = (0..n_samples).all(|i| (0..n_features).all(|j| x[(i, j)].is_finite()));
    let y_finite = (0..n_samples).all(|i| y[i].is_finite());
    if !x_finite || !y_finite {
        return Err(PlsKitError::NonFiniteInput);
    }

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

    // Validate + normalize weights (spec §3.3, §3.4).
    let (w_norm, n_eff_val, all_uniform) =
        validate_and_normalize_weights(weights, n_samples, k_requested)?;
    if opts.check_n_eff {
        check_n_eff_for_k(n_eff_val, k_requested)?;
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

    let (t_mat, p_mat, w_mat, q_vec) = nipals_pls1(
        x_for_nipals,
        y_for_nipals,
        k_requested,
        opts.tol,
        opts.max_iter,
    )?;

    let k_used = w_mat.ncols();
    let coef = pls1_coef_at_k(&w_mat, &p_mat, &q_vec, k_used);

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
    })
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::type_complexity)]
#[allow(clippy::unnecessary_wraps)] // reserved for future variants that may return Err
fn nipals_pls1(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    tol: f64,
    max_iter: usize,
) -> PlsKitResult<(Mat<f64>, Mat<f64>, Mat<f64>, Col<f64>)> {
    let n = x.nrows();
    let d = x.ncols();
    // Owned working copies — deflated in place across components.
    // Par::Seq inside this kernel: outer Rayon (split-half, CV folds,
    // permutations) already saturates cores, and nested parallelism
    // would oversubscribe them.
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
            Par::Seq,
        );
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
            Par::Seq,
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
            Par::Seq,
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
            Par::Seq,
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
    // tol/max_iter currently unused — NIPALS PLS1 converges in one
    // power-iteration pass per component. Reserved for symmetric variants.
    let _ = (tol, max_iter);

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
#[allow(clippy::many_single_char_names)]
pub(crate) fn pls1_coef_at_k(w: &Mat<f64>, p: &Mat<f64>, q: &Col<f64>, k: usize) -> Col<f64> {
    let d = w.nrows();
    let wk = Mat::<f64>::from_fn(d, k, |i, a| w[(i, a)]);
    let pk = Mat::<f64>::from_fn(d, k, |i, a| p[(i, a)]);
    let qk = Col::<f64>::from_fn(k, |a| q[a]);
    // P' W is (k, k); solve (P'W) z = Q via faer's LU, then coef = W z.
    let pwk: Mat<f64> = pk.transpose() * &wk;
    let lu: PartialPivLu<f64> = pwk.partial_piv_lu();
    let z: Col<f64> = lu.solve(&qk);
    &wk * &z
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
}
