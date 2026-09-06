//! PLS3 / PLSSVD — symmetric X↔Y covariance analysis. Public entry points:
//! `pls3_fit` (alias `plssvd_fit`) and `pls3_transform` (alias
//! `plssvd_transform`).
//!
//! The fit is one thin SVD of the standardized cross-covariance `X̃'Ỹ`
//! (`p × q`), not `k` sequential passes: there is no deflation, so all
//! components come out orthogonal by construction. The SVD never touches a
//! `p × p` matrix, which is why `p ≫ n` is the ordinary case here rather
//! than a problem.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::{check_finite_mat, resolve_par, ParChoice};

/// Knobs for [`pls3_fit`].
#[derive(Debug, Clone, Copy)]
pub struct Pls3FitOpts {
    /// Skip X centering/scaling; caller asserts X is already standardized.
    /// The returned `x_mean` / `x_scale` are then the identity moments
    /// (0 / 1), matching what `pls1_fit` records on the same branch.
    ///
    /// **Scale contract.** Same failure mode as `FitOpts::pre_standardized`
    /// in `fit.rs` (see its doc comment, `fit.rs:40-57`) — deliberately
    /// reused here. When `pre_standardized_x=true`, `pls3_fit` compares the
    /// singular values of `X̃'Ỹ` against the fixed `SIGMA_FLOOR = 1e-14`
    /// absolute threshold. If the "pre-standardized" X passed in is not
    /// genuinely close to zero-mean / unit-variance, that threshold can
    /// silently truncate `k_used` below the requested `k` instead of
    /// erroring. Unlike `pls1_fit`, PLS3 has no `check_n_eff`-equivalent
    /// opt-in to turn truncation into a hard error (out of scope for this
    /// family) — callers passing `pre_standardized_x=true` must ensure the
    /// inputs are genuinely standardized.
    pub pre_standardized_x: bool,
    /// Skip Y centering/scaling; same contract as `pre_standardized_x`.
    pub pre_standardized_y: bool,
    /// Parallelism strategy for the score matmuls. See [`ParChoice`].
    /// Resamplers force `Seq` — outer Rayon already owns the cores.
    pub par: ParChoice,
}

impl Default for Pls3FitOpts {
    fn default() -> Self {
        Self {
            pre_standardized_x: false,
            pre_standardized_y: false,
            par: ParChoice::Auto,
        }
    }
}

/// Owned PLS3 fit. Fields use long `snake_case` names; the wrapper
/// translates to short Python-facing names at the FFI seam.
#[derive(Debug, Clone)]
pub struct Pls3Model {
    /// X-side saliences `U`; shape `(n_features, k_used)`, orthonormal columns.
    pub u_saliences: Mat<f64>,
    /// Y-side saliences `V`; shape `(n_targets, k_used)`, orthonormal columns.
    pub v_saliences: Mat<f64>,
    /// Singular values of `X̃'Ỹ`, descending; shape `(k_used,)`.
    pub singular_values: Col<f64>,
    /// In-sample X-side LV scores `X̃·U`; shape `(n_samples, k_used)`.
    pub x_scores: Mat<f64>,
    /// In-sample Y-side LV scores `Ỹ·V`; shape `(n_samples, k_used)`.
    pub y_scores: Mat<f64>,
    /// Column means of X used by the fit; zeros when `pre_standardized_x`.
    pub x_mean: Col<f64>,
    /// Column scales of X used by the fit; ones when `pre_standardized_x`.
    pub x_scale: Col<f64>,
    /// Column means of Y used by the fit; zeros when `pre_standardized_y`.
    pub y_mean: Col<f64>,
    /// Column scales of Y used by the fit; ones when `pre_standardized_y`.
    pub y_scale: Col<f64>,
    /// Number of components actually retained (≤ requested `k`).
    pub k_used: usize,
    /// Echoes the caller's `pre_standardized_x` flag.
    pub pre_standardized_x: bool,
    /// Echoes the caller's `pre_standardized_y` flag.
    pub pre_standardized_y: bool,
}

/// Absolute floor on a retained singular value. Mirrors the `1e-14` norm
/// guard in `fit::nipals_pls1` (change together): below it the component
/// is numerical dust and is dropped rather than reported, so a
/// zero-variance block truncates instead of emitting NaN saliences.
/// Mirrored as `LAMBDA_FLOOR` = `SIGMA_FLOOR²` in
/// `dual_route::pls3_split_zbars_columns` (change together), where the
/// eigenvalues of Ỹ'GỸ are the σ².
const SIGMA_FLOOR: f64 = 1e-14;

/// Fit PLS3 / PLSSVD: SVD of the standardized cross-covariance `X̃'Ỹ`.
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - `y`: `(n_samples, n_targets)`
/// - `k`: `1..=min(n_features, n_targets)`
/// - returns `Pls3Model` with `u_saliences: (n_features, k_used)`,
///   `v_saliences: (n_targets, k_used)`, `singular_values: (k_used,)`,
///   `x_scores` / `y_scores`: `(n_samples, k_used)`.
///
/// # Errors
/// - `PlsKitError::DimensionMismatch` when `y.nrows() != x.nrows()`
/// - `PlsKitError::InvalidArgument` when `k == 0`, or when `weights` is `Some`
///   (observation weights are not implemented for this family; see
///   api-surface §1.1)
/// - `PlsKitError::KExceedsMax` when `k > min(n_features, n_targets)`
/// - `PlsKitError::NonFiniteInput` when X or Y contains NaN/inf
/// - `PlsKitError::Internal` when faer's SVD fails to converge
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub fn pls3_fit(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<Pls3Model> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let n_targets = y.ncols();

    if y.nrows() != n_samples {
        return Err(PlsKitError::DimensionMismatch {
            x: (n_samples, n_features),
            y: y.nrows(),
        });
    }
    if weights.is_some() {
        return Err(PlsKitError::InvalidArgument(
            "observation weights are not implemented for pls3_fit; pass weights=None. \
             The weighted PLS3 path row-scales (X, Y) by √w before forming X'Y and is \
             not implemented yet across the PLS2/PLS3 family."
                .into(),
        ));
    }
    check_finite_mat(x)?;
    check_finite_mat(y)?;

    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    let k_max = n_features.min(n_targets);
    if k > k_max {
        return Err(PlsKitError::KExceedsMax { k, k_max });
    }

    // Standardize OR skip, recording the moments either way so
    // `pls3_transform` can apply the same transform to new data.
    let (xs_owned, x_mean, x_scale) = if opts.pre_standardized_x {
        (
            None,
            Col::<f64>::zeros(n_features),
            Col::<f64>::from_fn(n_features, |_| 1.0),
        )
    } else {
        let (xs, m, s) = crate::linalg::standardize(x);
        (Some(xs), m, s)
    };
    let (ys_owned, y_mean, y_scale) = if opts.pre_standardized_y {
        (
            None,
            Col::<f64>::zeros(n_targets),
            Col::<f64>::from_fn(n_targets, |_| 1.0),
        )
    } else {
        let (ys, m, s) = crate::linalg::standardize(y);
        (Some(ys), m, s)
    };
    let xs: MatRef<'_, f64> = xs_owned.as_ref().map_or(x, faer::Mat::as_ref);
    let ys: MatRef<'_, f64> = ys_owned.as_ref().map_or(y, faer::Mat::as_ref);

    // The SVD cost is min(p,q)·p·q; the matmuls below are n·p·q. Size the
    // par decision on the matmul term, which dominates whenever n ≥ min(p,q).
    let par = resolve_par(opts.par, n_samples, n_features, n_targets);

    // A = X̃'Ỹ, (p × q). Never a p × p matrix — that is the whole point.
    let mut a = Mat::<f64>::zeros(n_features, n_targets);
    matmul(a.as_mut(), Accum::Replace, xs.transpose(), ys, 1.0, par);

    let svd = a
        .as_ref()
        .thin_svd()
        .map_err(|_| PlsKitError::Internal("SVD of X'Y failed to converge in pls3_fit".into()))?;
    let sigma = svd.S().column_vector();

    // Truncate at the first below-floor singular value, mirroring NIPALS'
    // early break rather than emitting components with unidentified directions.
    let mut k_used = 0usize;
    while k_used < k && sigma[k_used] >= SIGMA_FLOOR {
        k_used += 1;
    }

    let mut u = svd.U().subcols(0, k_used).to_owned();
    let mut v = svd.V().subcols(0, k_used).to_owned();
    pin_component_signs(&mut u, &mut v);
    let singular_values = Col::<f64>::from_fn(k_used, |i| sigma[i]);

    let mut x_scores = Mat::<f64>::zeros(n_samples, k_used);
    matmul(x_scores.as_mut(), Accum::Replace, xs, u.as_ref(), 1.0, par);
    let mut y_scores = Mat::<f64>::zeros(n_samples, k_used);
    matmul(y_scores.as_mut(), Accum::Replace, ys, v.as_ref(), 1.0, par);

    Ok(Pls3Model {
        u_saliences: u,
        v_saliences: v,
        singular_values,
        x_scores,
        y_scores,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        k_used,
        pre_standardized_x: opts.pre_standardized_x,
        pre_standardized_y: opts.pre_standardized_y,
    })
}

/// Pin the SVD sign convention: flip `(u_i, v_i)` together until the
/// largest-magnitude entry of `u_i` is positive, ties broken by lowest row
/// index. faer promises no sign, and the reference corpus stores the
/// saliences verbatim, so without this the fixtures would be
/// platform-dependent. A *joint* flip is what keeps it free: it leaves
/// `u_i σ_i v_i'`, and therefore `X'Y` and every held-out LV correlation,
/// exactly where it was (spec, "Sign indeterminacy is not a problem").
fn pin_component_signs(u: &mut Mat<f64>, v: &mut Mat<f64>) {
    for a in 0..u.ncols() {
        let mut best = 0usize;
        for i in 1..u.nrows() {
            if u[(i, a)].abs() > u[(best, a)].abs() {
                best = i;
            }
        }
        if u.nrows() > 0 && u[(best, a)] < 0.0 {
            for i in 0..u.nrows() {
                u[(i, a)] = -u[(i, a)];
            }
            for i in 0..v.nrows() {
                v[(i, a)] = -v[(i, a)];
            }
        }
    }
}

/// Alias for [`pls3_fit`] under the SVD-PLS name the chemometrics
/// literature uses. Same function, same arguments, same result — registered
/// as an alias in api-surface §1.1, mirroring how `spls1_fit` delegates to
/// `pls1_fit` in `fit.rs`.
///
/// # Errors
/// Everything [`pls3_fit`] returns.
pub fn plssvd_fit(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: Pls3FitOpts,
) -> PlsKitResult<Pls3Model> {
    pls3_fit(x, y, k, weights, opts)
}

/// Which score block [`pls3_transform`] should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformWhich {
    /// Project new X only.
    XScores,
    /// Project new Y only.
    YScores,
    /// Project both blocks.
    Both,
}

impl TransformWhich {
    /// Public string identifier (`snake_case`) used in wrapper APIs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TransformWhich::XScores => "x_scores",
            TransformWhich::YScores => "y_scores",
            TransformWhich::Both => "both",
        }
    }

    fn needs_x(self) -> bool {
        matches!(self, TransformWhich::XScores | TransformWhich::Both)
    }

    fn needs_y(self) -> bool {
        matches!(self, TransformWhich::YScores | TransformWhich::Both)
    }
}

/// LV scores for new data. A block is `None` exactly when `which` did not
/// ask for it.
#[derive(Debug, Clone)]
pub struct Pls3Scores {
    /// `X_new` projected on the X-side saliences; shape `(n_new, k_used)`.
    pub x_scores: Option<Mat<f64>>,
    /// `Y_new` projected on the Y-side saliences; shape `(n_new, k_used)`.
    pub y_scores: Option<Mat<f64>>,
}

/// Project new data onto a fitted PLS3's LV scores.
///
/// New data is standardized with the *fit's* moments — the same
/// train-moments-on-test-data rule `split_half_correlations` uses in
/// `signal_test.rs`. On a `pre_standardized_*` fit those moments are the
/// identity, so the caller must apply its own transform first, exactly as
/// `pls1_predict` requires.
///
/// PLS3 is symmetric, so there is no regression `predict` here: neither
/// block is the outcome (api-surface §1.2).
///
/// # Shapes
/// - `x_new`: `(n_new, n_features)` — `n_features` must equal `model.u_saliences.nrows()`
/// - `y_new`: `(n_new, n_targets)` — `n_targets` must equal `model.v_saliences.nrows()`
/// - returns scores of shape `(n_new, model.k_used)` per requested block
///
/// # Errors
/// - `PlsKitError::InvalidArgument` when `which` asks for a block that was not supplied
/// - `PlsKitError::ShapeMismatch` on a column-count mismatch against the model, or
///   on a model whose saliences, `k_used` and moments disagree with each other
/// - `PlsKitError::NonFiniteInput` when the supplied data contains NaN/inf
///
/// # Panics
/// Never (shapes validated at entry).
pub fn pls3_transform(
    model: &Pls3Model,
    x_new: Option<MatRef<'_, f64>>,
    y_new: Option<MatRef<'_, f64>>,
    which: TransformWhich,
) -> PlsKitResult<Pls3Scores> {
    // Par::Seq: transform is reachable from inside resampler Rayon workers,
    // so it must not dispatch to the global pool (mirrors pls1_predict).
    let par = Par::Seq;

    let x_scores = if which.needs_x() {
        let xn = x_new.ok_or_else(|| {
            PlsKitError::InvalidArgument(format!(
                "which='{}' requires x_new, which was not supplied",
                which.as_str()
            ))
        })?;
        let p = model.u_saliences.nrows();
        if xn.ncols() != p {
            // ShapeMismatch, not DimensionMismatch: the latter is documented
            // and formatted as "X has shape {x}, y has length {y}" for a
            // *row*-count disagreement between X and y, which would print
            // nonsense here ("y has length 7" for a 9-column X).
            return Err(PlsKitError::ShapeMismatch(format!(
                "x_new has {} columns, model was fit on {p}",
                xn.ncols()
            )));
        }
        // A hand-built model (the Python surface exposes every field
        // separately) can carry a k_used, a salience width and moments that
        // disagree. matmul and standardize_apply would panic on the
        // mismatch instead of reporting it.
        if model.u_saliences.ncols() != model.k_used {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model u_saliences has {} columns, k_used is {}",
                model.u_saliences.ncols(),
                model.k_used
            )));
        }
        if model.x_mean.nrows() != p || model.x_scale.nrows() != p {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model x_mean/x_scale have lengths {}/{}, expected {p}",
                model.x_mean.nrows(),
                model.x_scale.nrows()
            )));
        }
        check_finite_mat(xn)?;
        let xs =
            crate::linalg::standardize_apply(xn, model.x_mean.as_ref(), model.x_scale.as_ref());
        let mut t = Mat::<f64>::zeros(xn.nrows(), model.k_used);
        matmul(
            t.as_mut(),
            Accum::Replace,
            xs.as_ref(),
            model.u_saliences.as_ref(),
            1.0,
            par,
        );
        Some(t)
    } else {
        None
    };

    let y_scores = if which.needs_y() {
        let yn = y_new.ok_or_else(|| {
            PlsKitError::InvalidArgument(format!(
                "which='{}' requires y_new, which was not supplied",
                which.as_str()
            ))
        })?;
        let q = model.v_saliences.nrows();
        if yn.ncols() != q {
            // ShapeMismatch — see the x_new branch.
            return Err(PlsKitError::ShapeMismatch(format!(
                "y_new has {} columns, model was fit on {q}",
                yn.ncols()
            )));
        }
        // Model self-consistency — see the x_new branch.
        if model.v_saliences.ncols() != model.k_used {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model v_saliences has {} columns, k_used is {}",
                model.v_saliences.ncols(),
                model.k_used
            )));
        }
        if model.y_mean.nrows() != q || model.y_scale.nrows() != q {
            return Err(PlsKitError::ShapeMismatch(format!(
                "model y_mean/y_scale have lengths {}/{}, expected {q}",
                model.y_mean.nrows(),
                model.y_scale.nrows()
            )));
        }
        check_finite_mat(yn)?;
        let ys =
            crate::linalg::standardize_apply(yn, model.y_mean.as_ref(), model.y_scale.as_ref());
        let mut t = Mat::<f64>::zeros(yn.nrows(), model.k_used);
        matmul(
            t.as_mut(),
            Accum::Replace,
            ys.as_ref(),
            model.v_saliences.as_ref(),
            1.0,
            par,
        );
        Some(t)
    } else {
        None
    };

    Ok(Pls3Scores { x_scores, y_scores })
}

/// Alias for [`pls3_transform`] under the SVD-PLS name (api-surface §1.2).
///
/// # Errors
/// Everything [`pls3_transform`] returns.
pub fn plssvd_transform(
    model: &Pls3Model,
    x_new: Option<MatRef<'_, f64>>,
    y_new: Option<MatRef<'_, f64>>,
    which: TransformWhich,
) -> PlsKitResult<Pls3Scores> {
    pls3_transform(model, x_new, y_new, which)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// (X, Y) sharing one latent factor: X's first two columns and Y's
    /// first column all load on `f`, so LV1 is a strong, well-separated
    /// singular triplet and the tests below are not measuring noise.
    #[allow(clippy::many_single_char_names)]
    fn shared_factor_data(n: usize, p: usize, q: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let f: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x = Mat::<f64>::from_fn(n, p, |i, j| {
            let noise = rng.random_range(-0.3..0.3);
            if j < 2 {
                f[i] + noise
            } else {
                noise
            }
        });
        let y = Mat::<f64>::from_fn(n, q, |i, j| {
            let noise = rng.random_range(-0.3..0.3);
            if j == 0 {
                f[i] + noise
            } else {
                noise
            }
        });
        (x, y)
    }

    #[test]
    fn fit_returns_requested_shapes() {
        let (x, y) = shared_factor_data(60, 8, 4, 11);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        assert_eq!(m.k_used, 2);
        assert_eq!((m.u_saliences.nrows(), m.u_saliences.ncols()), (8, 2));
        assert_eq!((m.v_saliences.nrows(), m.v_saliences.ncols()), (4, 2));
        assert_eq!(m.singular_values.nrows(), 2);
        assert_eq!((m.x_scores.nrows(), m.x_scores.ncols()), (60, 2));
        assert_eq!((m.y_scores.nrows(), m.y_scores.ncols()), (60, 2));
    }

    #[test]
    fn saliences_are_orthonormal_and_sigmas_descending() {
        let (x, y) = shared_factor_data(60, 8, 4, 11);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 3, None, Pls3FitOpts::default()).unwrap();
        for a in 0..m.k_used {
            let norm: f64 = (0..8).map(|j| m.u_saliences[(j, a)].powi(2)).sum::<f64>();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
            let norm_v: f64 = (0..4).map(|j| m.v_saliences[(j, a)].powi(2)).sum::<f64>();
            assert_relative_eq!(norm_v, 1.0, epsilon = 1e-12);
        }
        let cross: f64 = (0..8)
            .map(|j| m.u_saliences[(j, 0)] * m.u_saliences[(j, 1)])
            .sum();
        assert!(cross.abs() < 1e-10, "u1 ⟂ u2 violated: {cross}");
        for a in 1..m.k_used {
            assert!(m.singular_values[a - 1] >= m.singular_values[a]);
        }
    }

    #[test]
    fn singular_triplet_reproduces_cross_covariance_action() {
        // A v₁ = σ₁ u₁ is the identity the dual (Gram) route rests on;
        // assert it here so the fit owns the invariant.
        let (x, y) = shared_factor_data(60, 8, 4, 11);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 1, None, Pls3FitOpts::default()).unwrap();
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize(y.as_ref());
        let a: Mat<f64> = xs.transpose() * ys.as_ref();
        let av: Col<f64> = a.as_ref() * m.v_saliences.col(0);
        for j in 0..8 {
            assert_relative_eq!(
                av[j],
                m.singular_values[0] * m.u_saliences[(j, 0)],
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn signs_are_pinned_to_largest_u_entry_positive() {
        let (x, y) = shared_factor_data(60, 8, 4, 11);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 3, None, Pls3FitOpts::default()).unwrap();
        for a in 0..m.k_used {
            let mut best = 0usize;
            for j in 1..8 {
                if m.u_saliences[(j, a)].abs() > m.u_saliences[(best, a)].abs() {
                    best = j;
                }
            }
            assert!(
                m.u_saliences[(best, a)] > 0.0,
                "component {a}: largest-|.| u entry is negative"
            );
        }
    }

    #[test]
    fn pre_standardized_path_matches_manual_standardization() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize(y.as_ref());
        let a = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let b = pls3_fit(
            xs.as_ref(),
            ys.as_ref(),
            2,
            None,
            Pls3FitOpts {
                pre_standardized_x: true,
                pre_standardized_y: true,
                ..Pls3FitOpts::default()
            },
        )
        .unwrap();
        for j in 0..6 {
            assert_relative_eq!(
                a.u_saliences[(j, 0)],
                b.u_saliences[(j, 0)],
                epsilon = 1e-12
            );
        }
        assert_relative_eq!(a.singular_values[0], b.singular_values[0], epsilon = 1e-10);
        // Pre-standardized fits report identity moments, matching pls1_fit.
        for j in 0..6 {
            assert_relative_eq!(b.x_mean[j], 0.0, epsilon = 1e-15);
            assert_relative_eq!(b.x_scale[j], 1.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn weights_are_rejected() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let w = Col::<f64>::from_fn(40, |_| 1.0);
        let r = pls3_fit(
            x.as_ref(),
            y.as_ref(),
            1,
            Some(w.as_ref()),
            Pls3FitOpts::default(),
        );
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn k_above_min_p_q_errors() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let r = pls3_fit(x.as_ref(), y.as_ref(), 4, None, Pls3FitOpts::default());
        assert!(matches!(
            r,
            Err(PlsKitError::KExceedsMax { k: 4, k_max: 3 })
        ));
    }

    #[test]
    fn k_zero_errors() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let r = pls3_fit(x.as_ref(), y.as_ref(), 0, None, Pls3FitOpts::default());
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn row_count_mismatch_errors() {
        let x = Mat::<f64>::zeros(10, 4);
        let y = Mat::<f64>::zeros(9, 2);
        let r = pls3_fit(x.as_ref(), y.as_ref(), 1, None, Pls3FitOpts::default());
        assert!(matches!(r, Err(PlsKitError::DimensionMismatch { .. })));
    }

    #[test]
    fn non_finite_input_errors() {
        let (mut x, y) = shared_factor_data(20, 4, 2, 3);
        x[(0, 0)] = f64::NAN;
        let r = pls3_fit(x.as_ref(), y.as_ref(), 1, None, Pls3FitOpts::default());
        assert!(matches!(r, Err(PlsKitError::NonFiniteInput)));
    }

    #[test]
    fn constant_columns_truncate_to_zero_components() {
        // Every X column constant ⇒ standardize's zero-variance branch zeroes
        // X̃ ⇒ X̃'Ỹ is the zero matrix ⇒ σ₁ = 0 ⇒ nothing is retained.
        let x = Mat::<f64>::from_fn(30, 4, |_, _| 2.5);
        let (_, y) = shared_factor_data(30, 4, 3, 9);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        assert_eq!(m.k_used, 0);
        assert_eq!(m.u_saliences.ncols(), 0);
    }

    #[test]
    fn plssvd_fit_is_the_same_function() {
        let (x, y) = shared_factor_data(40, 6, 3, 5);
        let a = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let b = plssvd_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        for j in 0..6 {
            assert_eq!(
                a.u_saliences[(j, 0)].to_bits(),
                b.u_saliences[(j, 0)].to_bits()
            );
        }
    }

    #[test]
    fn transform_on_training_data_reproduces_in_sample_scores() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let s =
            pls3_transform(&m, Some(x.as_ref()), Some(y.as_ref()), TransformWhich::Both).unwrap();
        let xs = s.x_scores.unwrap();
        let ys = s.y_scores.unwrap();
        for a in 0..2 {
            for i in 0..50 {
                assert_relative_eq!(xs[(i, a)], m.x_scores[(i, a)], epsilon = 1e-10);
                assert_relative_eq!(ys[(i, a)], m.y_scores[(i, a)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn transform_x_only_leaves_y_scores_none() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let s = pls3_transform(&m, Some(x.as_ref()), None, TransformWhich::XScores).unwrap();
        assert!(s.x_scores.is_some());
        assert!(s.y_scores.is_none());
    }

    #[test]
    fn transform_missing_required_block_errors() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let r = pls3_transform(&m, None, Some(y.as_ref()), TransformWhich::Both);
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
        let r = pls3_transform(&m, None, None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn transform_column_count_mismatch_errors() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let bad = Mat::<f64>::zeros(5, 9);
        let r = pls3_transform(&m, Some(bad.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))));
    }

    #[test]
    fn transform_inconsistent_model_errors() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();

        let mut bad = m.clone();
        bad.x_mean = Col::<f64>::zeros(3);
        let r = pls3_transform(&bad, Some(x.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))));

        let mut bad = m;
        bad.k_used = 5;
        let r = pls3_transform(&bad, Some(x.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))));
    }

    #[test]
    fn transform_rejects_non_finite_new_data() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let mut bad = Mat::<f64>::zeros(5, 7);
        bad[(0, 0)] = f64::INFINITY;
        let r = pls3_transform(&m, Some(bad.as_ref()), None, TransformWhich::XScores);
        assert!(matches!(r, Err(PlsKitError::NonFiniteInput)));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn plssvd_transform_is_the_same_function() {
        let (x, y) = shared_factor_data(50, 7, 3, 21);
        let m = pls3_fit(x.as_ref(), y.as_ref(), 2, None, Pls3FitOpts::default()).unwrap();
        let a = pls3_transform(&m, Some(x.as_ref()), None, TransformWhich::XScores).unwrap();
        let b = plssvd_transform(&m, Some(x.as_ref()), None, TransformWhich::XScores).unwrap();
        let (a, b) = (a.x_scores.unwrap(), b.x_scores.unwrap());
        assert_eq!(a[(0, 0)].to_bits(), b[(0, 0)].to_bits());
    }
}
