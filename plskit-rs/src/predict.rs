//! Apply a fitted PLS1 model to new X → ŷ in raw scale.

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
    let raw: Col<f64> = x_new * &model.beta;
    let intercept = if model.pre_standardized {
        0.0
    } else {
        model.intercept
    };
    Ok(Col::<f64>::from_fn(raw.nrows(), |i| raw[i] + intercept))
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
        let (x, y) = linear_data(80, 6, 3, 7);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(3),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let y_hat = pls1_predict(&m, x.as_ref()).unwrap();
        let mean_y: f64 = (0..y.nrows()).map(|i| y[i]).sum::<f64>() / y.nrows() as f64;
        let ss_tot: f64 = (0..y.nrows()).map(|i| (y[i] - mean_y).powi(2)).sum();
        let ss_res: f64 = (0..y.nrows()).map(|i| (y[i] - y_hat[i]).powi(2)).sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.9, "R² = {r2}");
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
    fn predict_with_pre_standardized_skips_intercept() {
        let (x, y) = linear_data(30, 5, 2, 1);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = crate::linalg::standardize1(y.as_ref());
        let m = pls1_fit(
            xs.as_ref(),
            ys.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
        )
        .unwrap();
        let y_hat = pls1_predict(&m, xs.as_ref()).unwrap();
        assert_relative_eq!(m.intercept, 0.0, epsilon = 1e-15);
        let direct: Col<f64> = &xs * &m.coef;
        for i in 0..y_hat.nrows() {
            assert_relative_eq!(y_hat[i], direct[i], epsilon = 1e-12);
        }
    }
}
