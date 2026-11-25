//! `pls1_predict` fixture cases.

use crate::cases::{
    CasePaths, ensure_moved, faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat,
    scalar_f64, scalar_i64, synth_covariates, synth_data,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{NpzWriter, sha256_of_file};
use anyhow::Result;
use plskit::{FitOpts, pls1_fit, pls1_predict};
use std::path::Path;

/// Case: basic `pls1_predict` — fit on `(60, 6)` train data, predict on `(20, 6)` holdout.
///
/// Uses `seed=42` for training data and `seed=43` for holdout X.
///
/// Inputs: `X_train`, `y_train`, `X_new`.
/// Outputs: `y_pred`, `coef`, `beta`, `intercept`, `k_used` (from the fitted model).
///
/// # Errors
/// Returns an error if fixture files cannot be written or either plskit call fails.
pub fn basic_n80_d6_k2(root: &Path) -> Result<Case> {
    let name = "pls1_predict_basic_n80_d6_k2";
    let function = "pls1_predict";
    let paths = CasePaths::build(root, function, name)?;

    let (x_train, y_train) = synth_data(60, 6, 2, 4.0, 42);
    let (x_new, _) = synth_data(20, 6, 2, 4.0, 43);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X_train", &x_train.clone().into_dyn())?;
        w.add_f64("y_train", &y_train.clone().into_dyn())?;
        w.add_f64("X_new", &x_new.clone().into_dyn())?;
        w.finish()?;
    }

    let x_train_faer = ndarray_to_faer_mat(&x_train);
    let y_train_faer = ndarray_to_faer_col(&y_train);
    let model = pls1_fit(
        x_train_faer.as_ref(),
        y_train_faer.as_ref(),
        2,
        None,
        None,
        FitOpts::default(),
    )?;

    let x_new_faer = ndarray_to_faer_mat(&x_new);
    let y_pred = pls1_predict(&model, x_new_faer.as_ref(), None)?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("y_pred", &faer_col_to_array(&y_pred))?;
        w.add_f64("coef", &faer_col_to_array(&model.coef))?;
        w.add_f64("beta", &faer_col_to_array(&model.beta))?;
        w.add_f64("intercept", &scalar_f64(model.intercept))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(model.k_used)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"k": 2, "n_train": 60, "n_new": 20, "d": 6, "seed_train": 42, "seed_new": 43}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
    })
}

/// Case: [`basic_n80_d6_k2`]'s data with three covariates (the third an alias
/// of the first) on the training rows and on the new rows.
///
/// Inputs: `X_train`, `y_train`, `X_new`, `covariates` (training rows),
/// `covariates_new`. Outputs: those of [`basic_n80_d6_k2`] plus
/// `covariate_coef` and `n_covariates`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or either plskit call fails.
pub fn covariates_n80_d6_k2(root: &Path) -> Result<Case> {
    let name = "pls1_predict_covariates";
    let function = "pls1_predict";
    let paths = CasePaths::build(root, function, name)?;

    let (x_train, y_train) = synth_data(60, 6, 2, 4.0, 42);
    let (x_new, _) = synth_data(20, 6, 2, 4.0, 43);
    let z_train = synth_covariates(&x_train, 7);
    let z_new = synth_covariates(&x_new, 8);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X_train", &x_train.clone().into_dyn())?;
        w.add_f64("y_train", &y_train.clone().into_dyn())?;
        w.add_f64("X_new", &x_new.clone().into_dyn())?;
        w.add_f64("covariates", &z_train.clone().into_dyn())?;
        w.add_f64("covariates_new", &z_new.clone().into_dyn())?;
        w.finish()?;
    }

    let x_train_faer = ndarray_to_faer_mat(&x_train);
    let y_train_faer = ndarray_to_faer_col(&y_train);
    let z_train_faer = ndarray_to_faer_mat(&z_train);
    let model = pls1_fit(
        x_train_faer.as_ref(),
        y_train_faer.as_ref(),
        2,
        None,
        Some(z_train_faer.as_ref()),
        FitOpts::default(),
    )?;

    let x_new_faer = ndarray_to_faer_mat(&x_new);
    let z_new_faer = ndarray_to_faer_mat(&z_new);
    let y_pred = pls1_predict(&model, x_new_faer.as_ref(), Some(z_new_faer.as_ref()))?;
    let without = pls1_predict(
        &pls1_fit(
            x_train_faer.as_ref(),
            y_train_faer.as_ref(),
            2,
            None,
            None,
            FitOpts::default(),
        )?,
        x_new_faer.as_ref(),
        None,
    )?;
    let col = |c: &faer::Col<f64>| (0..c.nrows()).map(|i| c[i]).collect::<Vec<_>>();
    ensure_moved(name, "y_pred", &col(&y_pred), &col(&without))?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("y_pred", &faer_col_to_array(&y_pred))?;
        w.add_f64("coef", &faer_col_to_array(&model.coef))?;
        w.add_f64("beta", &faer_col_to_array(&model.beta))?;
        w.add_f64("intercept", &scalar_f64(model.intercept))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(model.k_used)?))?;
        if let Some(cc) = &model.covariate_coef {
            w.add_f64("covariate_coef", &faer_col_to_array(cc))?;
        }
        if let Some(nc) = model.n_covariates {
            w.add_i64("n_covariates", &scalar_i64(i64::try_from(nc)?))?;
        }
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "k": 2, "n_train": 60, "n_new": 20, "d": 6,
            "seed_train": 42, "seed_new": 43, "covariates": "Z"
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
    })
}
