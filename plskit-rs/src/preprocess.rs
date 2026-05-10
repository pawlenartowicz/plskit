//! Public preprocess helper. Normalizes weights and standardizes (X, y) using §3.2 math.
//!
//! Spec: docs/specs/2026-05-01-pls1-observation-weights.md §5.

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
    /// Always populated when weights were passed; `None` otherwise.
    pub n_eff: Option<f64>,
}

/// Public preprocess helper. Validates weights (length, finiteness, non-negativity, Σ > 0)
/// and standardizes (X, y) using the weighted formulas in spec §3.2.
///
/// Note: `n_eff ≥ k+1` is **not** validated here (k is unknown to `preprocess`); fit-side
/// entry points perform that check via `validate_and_normalize_weights`.
///
/// # Errors
///
/// Returns [`PlsKitError::DimensionMismatch`] if X, y, and weights have inconsistent lengths.
/// Returns [`PlsKitError::NonFiniteInput`] if any weight is NaN or infinite.
/// Returns [`PlsKitError::InvalidWeights`] if any weight is negative or all weights are zero.
pub fn preprocess(input: PreprocessInput<'_>) -> PlsKitResult<PreprocessResult> {
    // Determine n from whichever of X / y is provided; weights consistency-check.
    let n_from_x = input.x.map(|x| x.nrows());
    let n_from_y = input.y.map(|y| y.nrows());
    let n_from_w = input.weights.map(|w| w.nrows());

    if let (Some(nx), Some(ny)) = (n_from_x, n_from_y) {
        if nx != ny {
            return Err(PlsKitError::DimensionMismatch { x: (nx, 0), y: ny });
        }
    }
    if let Some(nw) = n_from_w {
        if let Some(nx) = n_from_x {
            if nx != nw {
                return Err(PlsKitError::DimensionMismatch { x: (nx, 0), y: nw });
            }
        }
        if let Some(ny) = n_from_y {
            if ny != nw {
                return Err(PlsKitError::DimensionMismatch { x: (nw, 0), y: ny });
            }
        }
    }

    // Validate and normalize weights.
    let (w_norm, n_eff_val) = match input.weights {
        None => (None, None),
        Some(w) => {
            for i in 0..w.nrows() {
                if !w[i].is_finite() {
                    return Err(PlsKitError::NonFiniteInput);
                }
                if w[i] < 0.0 {
                    return Err(PlsKitError::InvalidWeights { reason: "negative" });
                }
            }
            let wn =
                normalize_weights(w).ok_or(PlsKitError::InvalidWeights { reason: "all_zero" })?;
            let neff = compute_n_eff(w);
            (Some(wn), Some(neff))
        }
    };
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
