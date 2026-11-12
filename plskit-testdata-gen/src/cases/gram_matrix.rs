//! `gram_matrix` fixture cases.

use crate::cases::{
    case_files, ensure_moved, faer_mat_to_array, manifest_case, ndarray_to_faer_mat, scalar_i64,
    synth_data,
};
use crate::manifest::Case;
use crate::npz::NpzWriter;
use anyhow::Result;
use plskit::gram_matrix;
use std::path::Path;

/// Inputs of both cases: `synth_data(40, 120, 2, 4.0, 42)` with column `j`
/// scaled by `1 + j/10` and shifted by `1 + j/20`, so X is uncentred and
/// its column scales differ, and the flag moves `gram`.
const INPUTS: &str = "inputs/gram_matrix_n40_d120.npz";

/// Case: `gram_matrix` on [`INPUTS`] with `pre_standardized` as given.
///
/// Inputs: `X` `(40, 120)`.
/// Outputs: `gram` `(40, 40)`, `n_features`, `pre_standardized` (0/1).
///
/// # Errors
/// Returns an error if the fixture files cannot be written, if
/// `gram_matrix` fails, or if the flag does not move `gram`.
#[allow(clippy::cast_precision_loss)]
pub fn n40_d120(root: &Path, pre_standardized: bool) -> Result<Case> {
    let name = if pre_standardized {
        "gram_matrix_n40_d120_pre_standardized"
    } else {
        "gram_matrix_n40_d120"
    };
    let function = "gram_matrix";
    let rel_outputs = format!("outputs/{function}/{name}.npz");
    let (abs_inputs, abs_outputs) = case_files(root, INPUTS, &rel_outputs)?;
    let (raw, _) = synth_data(40, 120, 2, 4.0, 42);
    let x = ndarray::Array2::from_shape_fn(raw.dim(), |(i, j)| {
        raw[(i, j)] * (1.0 + j as f64 / 10.0) + 1.0 + j as f64 / 20.0
    });
    {
        let mut w = NpzWriter::create(&abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.finish()?;
    }

    let xf = ndarray_to_faer_mat(&x);
    let r = gram_matrix(xf.as_ref(), pre_standardized)?;
    let other = gram_matrix(xf.as_ref(), !pre_standardized)?;
    let flat = |m: &faer::Mat<f64>| -> Vec<f64> {
        (0..m.ncols())
            .flat_map(|j| (0..m.nrows()).map(move |i| m[(i, j)]))
            .collect()
    };
    ensure_moved(name, "gram", &flat(&r.gram), &flat(&other.gram))?;
    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_f64("gram", &faer_mat_to_array(&r.gram))?;
        w.add_i64("n_features", &scalar_i64(i64::try_from(r.n_features)?))?;
        w.add_i64(
            "pre_standardized",
            &scalar_i64(i64::from(r.pre_standardized)),
        )?;
        w.finish()?;
    }

    manifest_case(
        root,
        name,
        function,
        INPUTS.to_string(),
        rel_outputs,
        serde_json::json!({ "pre_standardized": pre_standardized }),
    )
}
