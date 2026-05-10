//! `preprocess` fixture cases.

use crate::cases::{ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, synth_data, CasePaths};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::Result;
use faer::Col;
use plskit::{preprocess, PreprocessInput};
use std::path::Path;

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

/// Convert a `faer::Col<f64>` to a 1-D `ndarray::ArrayD<f64>`.
fn faer_col_to_array(col: &Col<f64>) -> ndarray::ArrayD<f64> {
    let n = col.nrows();
    let v: Vec<f64> = (0..n).map(|i| col[i]).collect();
    ndarray::Array1::from_vec(v).into_dyn()
}

/// Case: `preprocess` on (n=50, d=10) data with non-uniform weights.
///
/// Weights are `2.0` for observations 0..25 and `1.0` for 25..50.
///
/// Inputs: `X`, `y`, `weights`.
/// Outputs: `X_std`, `X_mean`, `X_scale`, `y_std`, `y_mean`, `y_scale`,
///          `weights_normalized`, `n_eff`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `preprocess` fails.
pub fn n50_d10_with_weights(root: &Path) -> Result<Case> {
    let name = "preprocess_n50_d10_with_weights";
    let function = "preprocess";
    let paths = CasePaths::build(root, function, name)?;

    let (x, y) = synth_data(50, 10, 2, 4.0, 42);
    // Non-uniform weights: first 25 observations get weight 2.0, rest get 1.0.
    let weights_nd = ndarray::Array1::from_shape_fn(50, |i| if i < 25 { 2.0_f64 } else { 1.0_f64 });
    let weights_faer = ndarray_to_faer_col(&weights_nd);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.add_f64("weights", &weights_nd.clone().into_dyn())?;
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let result = preprocess(PreprocessInput {
        x: Some(x_faer.as_ref()),
        y: Some(y_faer.as_ref()),
        weights: Some(weights_faer.as_ref()),
    })?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        if let Some((x_std, x_mean, x_scale)) = &result.x_std {
            let x_std_arr =
                ndarray::Array2::from_shape_fn((x_std.nrows(), x_std.ncols()), |(i, j)| {
                    x_std[(i, j)]
                })
                .into_dyn();
            w.add_f64("X_std", &x_std_arr)?;
            w.add_f64("X_mean", &faer_col_to_array(x_mean))?;
            w.add_f64("X_scale", &faer_col_to_array(x_scale))?;
        }
        if let Some((y_std, y_mean, y_scale)) = &result.y_std {
            w.add_f64("y_std", &faer_col_to_array(y_std))?;
            w.add_f64("y_mean", &scalar_f64(*y_mean))?;
            w.add_f64("y_scale", &scalar_f64(*y_scale))?;
        }
        if let Some(wn) = &result.weights_normalized {
            w.add_f64("weights_normalized", &faer_col_to_array(wn))?;
        }
        if let Some(n_eff) = result.n_eff {
            w.add_f64("n_eff", &scalar_f64(n_eff))?;
        }
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"n": 50, "d": 10, "seed": 42, "weights": "nonuniform"}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
