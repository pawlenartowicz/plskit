//! Public preprocess helper. Normalizes weights and standardizes (X, y).
//!
//! [`preprocess()`](crate::preprocess::preprocess) takes a 1-D `y`;
//! [`preprocess_block`](crate::preprocess::preprocess_block) is the same
//! helper for a multi-column response block `Y`. The wrappers expose both as one
//! `preprocess` whose `Y` may be 1-D or 2-D, dispatching on its shape, so
//! the validation and the moments of a 2-D `Y` live here, not in a wrapper.

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::linalg::{
    compute_n_eff, normalize_weights, standardize1_weighted, standardize_weighted,
};

/// Input for `preprocess`. All three fields are independently optional.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessInput<'a> {
    /// Predictor matrix `(n_samples, n_features)` if provided.
    pub x: Option<MatRef<'a, f64>>,
    /// Response vector `(n_samples,)` if provided.
    pub y: Option<ColRef<'a, f64>>,
    /// Length-n observation weights if provided.
    pub weights: Option<ColRef<'a, f64>>,
}

/// Result of `preprocess`. Each field is populated only if the matching input was provided.
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// `Some((X_std, X_mean, X_scale))` when X was passed.
    pub x_std: Option<(Mat<f64>, Col<f64>, Col<f64>)>,
    /// `Some((y_std, y_mean, y_scale))` when y was passed.
    pub y_std: Option<(Col<f64>, f64, f64)>,
    /// `Some(w')` (normalized to mean 1, Σ = n) when weights were passed.
    pub weights_normalized: Option<Col<f64>>,
    /// Always populated when weights were passed; `None` otherwise. Kish's
    /// `(Σw)² / Σw²`, and exactly `n` when the weights are all equal (the
    /// value every fit reports for them).
    pub n_eff: Option<f64>,
}

/// Public preprocess helper. Validates weights (length, finiteness, non-negativity, Σ > 0)
/// and standardizes (X, y) with weighted moments: weights are renormalized to mean 1
/// (`w'`), then `mean = Σ w'x / n` and `var = Σ w'(x − mean)² / n` (population, ddof=0).
///
/// Note: `n_eff ≥ k+1` is **not** validated here (k is unknown to `preprocess`); fit-side
/// entry points perform that check via `validate_and_normalize_weights`.
///
/// # Errors
///
/// - [`PlsKitError::DimensionMismatch`] if X and y have inconsistent row counts.
/// - [`PlsKitError::InvalidWeights`]`{ reason: "length_mismatch" }` if weights length disagrees with X or y.
/// - [`PlsKitError::NonFiniteInput`] if X, y, or any weight contains NaN or infinity.
/// - [`PlsKitError::InvalidWeights`] if any weight is negative or all weights are zero.
/// - [`PlsKitError::InvalidArgument`] if X or y is given with zero rows (the
///   moments of an empty column are undefined).
pub fn preprocess(input: PreprocessInput<'_>) -> PlsKitResult<PreprocessResult> {
    check_rows(input.x, input.y.map(|y| y.nrows()), input.weights)?;
    // Top-level boundary contract: X / y finiteness. Mirrors pls1_fit's
    // check_finite_mat / check_finite_col entry sequence (fit.rs).
    if let Some(x) = input.x {
        crate::fit::check_finite_mat(x)?;
    }
    if let Some(y) = input.y {
        crate::fit::check_finite_col(y)?;
    }
    if let Some(n) = input.x.map(|x| x.nrows()).or(input.y.map(|y| y.nrows())) {
        check_has_rows(n)?;
    }
    let (w_norm, n_eff_val) = normalized_weights(input.weights)?;
    let wref = w_norm.as_ref().map(Col::as_ref);

    let x_std = input.x.map(|x| standardize_weighted(x, wref));
    let y_std = input.y.map(|y| standardize1_weighted(y, wref));

    Ok(PreprocessResult {
        x_std,
        y_std,
        weights_normalized: w_norm,
        n_eff: n_eff_val,
    })
}

/// Input for [`preprocess_block`]: [`PreprocessInput`] with a multi-column
/// response block `Y` in place of the 1-D `y`. All three fields are
/// independently optional.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessBlockInput<'a> {
    /// Predictor matrix `(n_samples, n_features)` if provided.
    pub x: Option<MatRef<'a, f64>>,
    /// Response block `(n_samples, n_targets)` if provided.
    pub y: Option<MatRef<'a, f64>>,
    /// Length-n observation weights if provided.
    pub weights: Option<ColRef<'a, f64>>,
}

/// Result of [`preprocess_block`]. Each field is populated only if the
/// matching input was provided.
#[derive(Debug, Clone)]
pub struct PreprocessBlockResult {
    /// `Some((X_std, X_mean, X_scale))` when X was passed.
    pub x_std: Option<(Mat<f64>, Col<f64>, Col<f64>)>,
    /// `Some((Y_std, Y_mean, Y_scale))` when Y was passed; one mean and one
    /// scale per column.
    pub y_std: Option<(Mat<f64>, Col<f64>, Col<f64>)>,
    /// `Some(w')` (normalized to mean 1, Σ = n) when weights were passed.
    pub weights_normalized: Option<Col<f64>>,
    /// Always populated when weights were passed; `None` otherwise. Kish's
    /// `(Σw)² / Σw²`, and exactly `n` when the weights are all equal (the
    /// value every fit reports for them).
    pub n_eff: Option<f64>,
}

/// [`preprocess`] for a multi-column response block `Y`: the same
/// validation (row counts, finiteness, weights) and the same weighted
/// moments, applied to every column of `Y` as [`preprocess`] applies them
/// to `y`. Column `j` of
/// `Y_std` is bit for bit what [`preprocess`] returns for `y = Y[:, j]`.
///
/// # Errors
///
/// As [`preprocess`], with `Y` in place of `y`:
/// [`PlsKitError::DimensionMismatch`] if X and Y have different row counts,
/// [`PlsKitError::InvalidWeights`]`{ reason: "length_mismatch" }` if the
/// weights length disagrees with X or Y, [`PlsKitError::NonFiniteInput`] if
/// X, Y, or any weight contains NaN or infinity,
/// [`PlsKitError::InvalidWeights`] if any weight is negative or all are zero,
/// and [`PlsKitError::InvalidArgument`] if X or Y has zero rows.
pub fn preprocess_block(input: PreprocessBlockInput<'_>) -> PlsKitResult<PreprocessBlockResult> {
    check_rows(input.x, input.y.map(|y| y.nrows()), input.weights)?;
    if let Some(x) = input.x {
        crate::fit::check_finite_mat(x)?;
    }
    if let Some(y) = input.y {
        crate::fit::check_finite_mat(y)?;
    }
    if let Some(n) = input.x.or(input.y).map(|m| m.nrows()) {
        check_has_rows(n)?;
    }
    let (w_norm, n_eff_val) = normalized_weights(input.weights)?;
    let wref = w_norm.as_ref().map(Col::as_ref);

    Ok(PreprocessBlockResult {
        x_std: input.x.map(|x| standardize_weighted(x, wref)),
        y_std: input.y.map(|y| standardize_weighted(y, wref)),
        weights_normalized: w_norm,
        n_eff: n_eff_val,
    })
}

/// The zero-row guard of every entry that standardizes a block without
/// running `pls1_fit`'s `n >= k + 1` check: [`preprocess`],
/// [`preprocess_block`] and the PLS3 fits. The moments of an empty column
/// are NaN (`0 / 0`), so no row is an error, worded like `pls1_fit`'s
/// "insufficient n for k=..." and with its code (`invalid_argument`). One
/// row passes: its columns are constant, which standardization handles.
///
/// # Errors
/// `InvalidArgument` when `n == 0`.
pub(crate) fn check_has_rows(n: usize) -> PlsKitResult<()> {
    if n == 0 {
        return Err(PlsKitError::InvalidArgument(
            "insufficient n: need n >= 1 (got n=0)".into(),
        ));
    }
    Ok(())
}

/// Row-count agreement of X, the response (`n_y` rows) and the weights.
fn check_rows(
    x: Option<MatRef<'_, f64>>,
    n_y: Option<usize>,
    weights: Option<ColRef<'_, f64>>,
) -> PlsKitResult<()> {
    if let (Some(x), Some(ny)) = (x, n_y) {
        if x.nrows() != ny {
            return Err(PlsKitError::DimensionMismatch {
                x: (x.nrows(), x.ncols()),
                y: ny,
            });
        }
    }
    if let Some(nw) = weights.map(|w| w.nrows()) {
        if x.map(|x| x.nrows()).is_some_and(|nx| nx != nw) || n_y.is_some_and(|ny| ny != nw) {
            return Err(PlsKitError::InvalidWeights {
                reason: "length_mismatch",
            });
        }
    }
    Ok(())
}

/// Validate and normalize weights: `(w', n_eff)`, both `None` without weights.
/// Mirrors fit.rs `validate_and_normalize_weights` (finite / non-negative /
/// Σ>0 loop): change together. preprocess keeps its own copy because it has
/// no `n` or `k` to feed that helper, and it echoes `w'` even when the
/// weights are all equal (where the fits drop them). Its `n_eff` still
/// follows the fits: exactly `n` for all-equal weights
/// (`fit::weights_all_equal`), where Kish's ratio can round an ulp below.
fn normalized_weights(
    weights: Option<ColRef<'_, f64>>,
) -> PlsKitResult<(Option<Col<f64>>, Option<f64>)> {
    let Some(w) = weights else {
        return Ok((None, None));
    };
    for i in 0..w.nrows() {
        if !w[i].is_finite() {
            return Err(PlsKitError::NonFiniteInput);
        }
        if w[i] < 0.0 {
            return Err(PlsKitError::InvalidWeights { reason: "negative" });
        }
    }
    let wn = normalize_weights(w).ok_or(PlsKitError::InvalidWeights { reason: "all_zero" })?;
    #[allow(clippy::cast_precision_loss)]
    let n_eff = if crate::fit::weights_all_equal(wn.as_ref()) {
        w.nrows() as f64
    } else {
        compute_n_eff(w)
    };
    Ok((Some(wn), Some(n_eff)))
}
