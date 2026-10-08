//! Apply a fitted PLS1 model to new X.
//!
//! Default path (`pre_standardized=false`): returns ŷ in raw scale with the
//! intercept restored (`ŷ = X_new · β + intercept`).
//!
//! Pre-standardized path (`pre_standardized=true`): returns ŷ in standardized
//! scale; `model.intercept` is 0.0 (set by `pls1_fit`), so the formula reduces to
//! `ŷ = X_new · β`. The caller must pre-standardize `X_new` with the same mean
//! and scale used during fit — this module does not re-apply those transforms.

use faer::{Col, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::Pls1Model;

/// Score new observations under a fitted PLS1 model.
///
/// # Shapes
/// - `x_new`: `(n_new, n_features)`
/// - returns: `(n_new,)`
///
/// # Errors
/// - `PlsKitError::DimensionMismatch` when `x_new.ncols() != model.beta.nrows()`
/// - `PlsKitError::NonFiniteInput` when `x_new` contains NaN/inf
///
/// # Panics
/// Never (shape validated at entry).
pub fn pls1_predict(model: &Pls1Model, x_new: MatRef<'_, f64>) -> PlsKitResult<Col<f64>> {
    let d = model.beta.nrows();
    if x_new.ncols() != d {
        return Err(PlsKitError::DimensionMismatch {
            x: (x_new.nrows(), x_new.ncols()),
            y: d,
        });
    }
    crate::fit::check_finite_mat(x_new)?;
    // model.intercept is already 0.0 for pre_standardized models (set in
    // pls1_fit), so no re-check needed here.
    // Par::Seq: predict is reachable from inside resampler Rayon workers
    // (holdout scoring), so it must not dispatch to the global pool.
    let mut raw = Col::<f64>::zeros(x_new.nrows());
    faer::linalg::matmul::matmul(
        raw.as_mut().as_mat_mut(),
        faer::Accum::Replace,
        x_new,
        model.beta.as_ref().as_mat(),
        1.0,
        faer::Par::Seq,
    );
    Ok(Col::<f64>::from_fn(raw.nrows(), |i| {
        raw[i] + model.intercept
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use approx::assert_relative_eq;
    use faer::Mat;

    fn linear_data(n: usize, d: usize, k_true: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta_true = Col::<f64>::from_fn(d, |j| if j < k_true { 1.0 } else { 0.0 });
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-0.05..0.05));
        let y_signal: Col<f64> = &x * &beta_true;
        let y = Col::<f64>::from_fn(n, |i| y_signal[i] + noise[i]);
        (x, y)
    }

    #[test]
    fn predict_recovers_in_sample_y() {
        // Offset X and an offset, scaled y, so a dropped intercept, `coef`
        // read for `beta`, or a re-standardized X each move the prediction.
        // The expected value is built from the scores, `mean(y) + sd(y)·(T q)`
        // (population sd), which reads neither `beta` nor `intercept`.
        let (x, y) = linear_data(80, 6, 3, 7);
        let n = y.nrows();
        let x = Mat::<f64>::from_fn(n, 6, |i, j| x[(i, j)] + 3.0);
        let y = Col::<f64>::from_fn(n, |i| 2.0 * y[i] + 10.0);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let y_hat = pls1_predict(&m, x.as_ref()).unwrap();
        let mean_y: f64 = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;
        let sd_y = ((0..n).map(|i| (y[i] - mean_y).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert_eq!(m.k_used, 3);
        for i in 0..n {
            let tq: f64 = (0..m.k_used)
                .map(|a| m.t_scores[(i, a)] * m.q_loadings[a])
                .sum();
            assert_relative_eq!(y_hat[i], mean_y + sd_y * tq, max_relative = 1e-10);
        }
    }

    #[test]
    fn predict_dimension_mismatch_errors() {
        let (x, y) = linear_data(30, 5, 2, 1);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let bad = Mat::<f64>::zeros(4, 6);
        let r = pls1_predict(&m, bad.as_ref());
        assert!(matches!(r, Err(PlsKitError::DimensionMismatch { .. })));
    }

    #[test]
    fn predict_rejects_non_finite_x_new() {
        let (x, y) = linear_data(30, 5, 2, 1);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();
        for bad_value in [f64::NAN, f64::INFINITY] {
            let mut bad = x.clone();
            bad[(3, 2)] = bad_value;
            let r = pls1_predict(&m, bad.as_ref());
            assert!(
                matches!(r, Err(PlsKitError::NonFiniteInput)),
                "{bad_value}: {r:?}"
            );
        }
    }
}
