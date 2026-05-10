//! `rotate` fixture cases.

use crate::cases::{
    ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_data, CasePaths,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::Result;
use faer::Mat;
use plskit::{pls1_fit, rotate, FitOpts, KSpec, RotationMethod, VarimaxArgs};
use std::path::Path;

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

/// Convert a `faer::Mat<f64>` (d × k) into a 2-D `ndarray::ArrayD<f64>`.
fn faer_mat_to_array(m: &Mat<f64>) -> ndarray::ArrayD<f64> {
    ndarray::Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_dyn()
}

/// Case: varimax rotation of a `W*` matrix derived from a synthetic PLS1 fit.
///
/// Fits PLS1 at `k=2` on `(n=20, d=6)` data to obtain a `W*` matrix (shape `6×2`),
/// then rotates it with varimax using default args.
///
/// `VarimaxArgs::default()` is deterministic (no external RNG — varimax uses purely
/// deterministic closed-form Kaiser angle sweeps starting from the identity).
///
/// Inputs: `X`, `y` (the source data; consumer reproduces W* by fitting PLS1 at k=2).
/// Outputs: `w_rot`, `r`, `sweeps`, `v_converged`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or either plskit call fails.
pub fn varimax_d6_k2(root: &Path) -> Result<Case> {
    let name = "rotate_varimax_d6_k2";
    let function = "rotate";
    let paths = CasePaths::build(root, function, name)?;

    let (x, y) = synth_data(20, 6, 2, 4.0, 42);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let model = pls1_fit(
        x_faer.as_ref(),
        y_faer.as_ref(),
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    )?;

    let rot_out = rotate(
        model.w_star.as_ref(),
        RotationMethod::Varimax(VarimaxArgs::default()),
        None,
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("w_rot", &faer_mat_to_array(&rot_out.w_rot))?;
        w.add_f64("r", &faer_mat_to_array(&rot_out.r))?;
        w.add_i64("sweeps", &scalar_i64(i64::try_from(rot_out.sweeps)?))?;
        w.add_f64("v_converged", &scalar_f64(rot_out.v_converged))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"method": "varimax", "n": 20, "d": 6, "k": 2, "seed": 42}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
