//! Apply a fitted PLS1 model to new X.
//!
//! Default path (`pre_standardized=false`): returns ŷ in raw scale with the
//! intercept restored (`ŷ = X_new · β + intercept`).
//!
//! Pre-standardized path (`pre_standardized=true`): returns ŷ in standardized
//! scale; `model.intercept` is 0.0 (set by `pls1_fit`), so the formula reduces to
//! `ŷ = X_new · β`. The caller must pre-standardize `X_new` with the same mean
//! and scale used during fit — this module does not re-apply those transforms.
//!
//! A model fitted with covariates adds them: `ŷ = X_new · β + intercept +
//! Z_new · covariate_coef`, on either path (its intercept is the fit's own,
//! also under `pre_standardized`).

use faer::{Col, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::Pls1Model;

/// Score new observations under a fitted PLS1 model.
///
/// # Shapes
/// - `x_new`: `(n_new, n_features)`
/// - `covariates`: `(n_new, p_z)`; required when the model was fitted with
///   covariates, refused otherwise
/// - returns: `(n_new,)`
///
/// # Errors
/// - `PlsKitError::DimensionMismatch` when `x_new.ncols() != model.beta.nrows()`
/// - `PlsKitError::NonFiniteInput` when `x_new` or `covariates` contains NaN/inf
/// - `PlsKitError::InvalidArgument` when `covariates` is missing for a model
///   fitted with covariates, or given for a model fitted without them
/// - `PlsKitError::ShapeMismatch` when `covariates` does not have `n_new`
///   rows and `p_z` columns
///
/// # Panics
/// Never (shape validated at entry).
pub fn pls1_predict(
    model: &Pls1Model,
    x_new: MatRef<'_, f64>,
    covariates: Option<MatRef<'_, f64>>,
) -> PlsKitResult<Col<f64>> {
    let d = model.beta.nrows();
    if x_new.ncols() != d {
        return Err(PlsKitError::DimensionMismatch {
            x: (x_new.nrows(), x_new.ncols()),
            y: d,
        });
    }
    crate::fit::check_finite_mat(x_new)?;
    let cov = match (&model.covariate_coef, covariates) {
        (None, None) => None,
        (Some(c), Some(z)) => {
            if z.nrows() != x_new.nrows() || z.ncols() != c.nrows() {
                return Err(PlsKitError::ShapeMismatch(format!(
                    "covariates have shape ({}, {}); expected ({}, {})",
                    z.nrows(),
                    z.ncols(),
                    x_new.nrows(),
                    c.nrows()
                )));
            }
            crate::fit::check_finite_mat(z)?;
            Some((z, c))
        }
        (Some(_), None) => {
            return Err(PlsKitError::InvalidArgument(
                "the model was fitted with covariates: pass covariates for the new rows".into(),
            ));
        }
        (None, Some(_)) => {
            return Err(PlsKitError::InvalidArgument(
                "the model was fitted without covariates: covariates are not accepted".into(),
            ));
        }
    };
    // model.intercept is what pls1_fit set for either path (0.0 for a
    // pre_standardized model without covariates), so it is always added.
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
    if let Some((z, c)) = cov {
        faer::linalg::matmul::matmul(
            raw.as_mut().as_mat_mut(),
            faer::Accum::Add,
            z,
            c.as_ref().as_mat(),
            1.0,
            faer::Par::Seq,
        );
    }
    Ok(Col::<f64>::from_fn(raw.nrows(), |i| {
        raw[i] + model.intercept
    }))
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::fit::{FitOpts, KSpec, pls1_fit};
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
            None,
            FitOpts::default(),
        )
        .unwrap();
        let y_hat = pls1_predict(&m, x.as_ref(), None).unwrap();
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
            None,
            FitOpts::default(),
        )
        .unwrap();
        let bad = Mat::<f64>::zeros(4, 6);
        let r = pls1_predict(&m, bad.as_ref(), None);
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
            None,
            FitOpts::default(),
        )
        .unwrap();
        for bad_value in [f64::NAN, f64::INFINITY] {
            let mut bad = x.clone();
            bad[(3, 2)] = bad_value;
            let r = pls1_predict(&m, bad.as_ref(), None);
            assert!(
                matches!(r, Err(PlsKitError::NonFiniteInput)),
                "{bad_value}: {r:?}"
            );
        }
    }

    /// `(x, y, z)` for training (`n` rows) and `(x_new, z_new)` for 15 new
    /// rows: X offset from zero, y driven by X and by the first covariate.
    #[allow(clippy::type_complexity)]
    fn covariate_data(n: usize) -> (Mat<f64>, Col<f64>, Mat<f64>, Mat<f64>, Mat<f64>) {
        use crate::test_support::unif;
        let (x, y) = linear_data(n, 6, 3, 21);
        let x = Mat::<f64>::from_fn(n, 6, |i, j| x[(i, j)] + 3.0);
        let z = unif(n, 2, 22);
        let y = Col::<f64>::from_fn(n, |i| 2.0 * y[i] + 1.5 * z[(i, 0)] + 10.0);
        let u = unif(15, 6, 23);
        let x_new = Mat::<f64>::from_fn(15, 6, |i, j| u[(i, j)] + 3.0);
        (x, y, z, x_new, unif(15, 2, 24))
    }

    #[test]
    fn predict_with_covariates_equals_long_form() {
        use crate::linalg::{
            mat_mul, normalize_weights, standardize_apply, standardize_weighted,
            standardize1_weighted,
        };
        use crate::test_support::{
            assert_close_rel, test_weights, with_intercept, wls_normal_equations,
        };
        let n = 60;
        let (x, y, z, x_new, z_new) = covariate_data(n);
        let (xs_pre, _, _) = crate::linalg::standardize(x.as_ref());
        for (w, pre) in [(None, false), (Some(test_weights(n)), false), (None, true)] {
            let x_in = if pre { xs_pre.clone() } else { x.clone() };
            let m = pls1_fit(
                x_in.as_ref(),
                y.as_ref(),
                KSpec::Fixed(2),
                w.as_ref().map(Col::as_ref),
                Some(z.as_ref()),
                FitOpts {
                    pre_standardized: pre,
                    ..FitOpts::default()
                },
            )
            .unwrap();
            if pre {
                assert!(m.intercept.abs() > 1.0, "intercept {}", m.intercept);
            }
            let x_new_in = if pre {
                let (_, mu, sc) = crate::linalg::standardize(x.as_ref());
                standardize_apply(x_new.as_ref(), mu.as_ref(), sc.as_ref())
            } else {
                x_new.clone()
            };
            let got = pls1_predict(&m, x_new_in.as_ref(), Some(z_new.as_ref())).unwrap();

            // Long form: ŷ = [1, Z_new]·γ_y + (X_new,std − [1, Z_new]·Γ_X)·β_std.
            let wn = w.as_ref().and_then(|w| normalize_weights(w.as_ref()));
            let wr = wn.as_ref().map(Col::as_ref);
            let (xs, mu, sc, y_scale) = if pre {
                let d = x_in.ncols();
                (
                    x_in.clone(),
                    Col::<f64>::zeros(d),
                    Col::<f64>::from_fn(d, |_| 1.0),
                    1.0,
                )
            } else {
                let (xs, mu, sc) = standardize_weighted(x_in.as_ref(), wr);
                (xs, mu, sc, standardize1_weighted(y.as_ref(), wr).2)
            };
            let a = with_intercept(z.as_ref());
            let gamma_y = wls_normal_equations(a.as_ref(), wr, y.as_ref().as_mat());
            let gamma_x = wls_normal_equations(a.as_ref(), wr, xs.as_ref());
            let beta_std = Col::<f64>::from_fn(m.coef.nrows(), |j| m.coef[j] * y_scale);
            let a_new = with_intercept(z_new.as_ref());
            let xn_std = standardize_apply(x_new_in.as_ref(), mu.as_ref(), sc.as_ref());
            let resid_x = &xn_std - mat_mul(a_new.as_ref(), gamma_x.as_ref(), faer::Par::Seq);
            let want = mat_mul(a_new.as_ref(), gamma_y.as_ref(), faer::Par::Seq)
                + mat_mul(resid_x.as_ref(), beta_std.as_mat(), faer::Par::Seq);
            assert_close_rel(got.as_mat(), want.as_ref(), 1e-10, "prediction");
        }
    }

    #[test]
    fn predict_covariate_errors() {
        let n = 60;
        let (x, y, z, x_new, z_new) = covariate_data(n);
        let fit = |z: Option<MatRef<'_, f64>>| {
            pls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(2),
                None,
                z,
                FitOpts::default(),
            )
            .unwrap()
        };
        let with = fit(Some(z.as_ref()));
        let without = fit(None);
        assert!(matches!(
            pls1_predict(&with, x_new.as_ref(), None),
            Err(PlsKitError::InvalidArgument(_))
        ));
        assert!(matches!(
            pls1_predict(&without, x_new.as_ref(), Some(z_new.as_ref())),
            Err(PlsKitError::InvalidArgument(_))
        ));
        assert!(matches!(
            pls1_predict(&with, x_new.as_ref(), Some(z_new.as_ref().subcols(0, 1))),
            Err(PlsKitError::ShapeMismatch(_))
        ));
        assert!(matches!(
            pls1_predict(&with, x_new.as_ref(), Some(z_new.as_ref().subrows(0, 14))),
            Err(PlsKitError::ShapeMismatch(_))
        ));
    }
}
