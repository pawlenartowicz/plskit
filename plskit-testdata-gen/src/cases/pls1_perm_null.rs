//! `pls1_perm_null` fixture cases.

use crate::cases::{
    ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_data, CasePaths,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::Result;
use plskit::{pls1_perm_null, PermNullOpts};
use std::path::Path;

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

/// Convert a `Vec<f64>` to a 1-D `ndarray::ArrayD<f64>`.
fn vec_to_array(v: &[f64]) -> ndarray::ArrayD<f64> {
    ndarray::Array1::from_vec(v.to_vec()).into_dyn()
}

/// Case: `pls1_perm_null` on `(n=80, d=6)` data at `k=2`, 200 permutations.
///
/// Uses `disable_parallelism: true` for byte-exact determinism and `seed=42`.
///
/// Inputs: `X`, `y`.
/// Outputs: `beta_ref`, `beta_perm_mean`, `beta_perm_sd`, `beta_perm_z`,
///          `n_perm`, `k`, `seed`, `n_eff`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn basic_n80_d6_k2(root: &Path) -> Result<Case> {
    let name = "pls1_perm_null_basic_n80_d6_k2";
    let function = "pls1_perm_null";
    let paths = CasePaths::build(root, function, name)?;

    let (x, y) = synth_data(80, 6, 2, 4.0, 42);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let result = pls1_perm_null(
        x_faer.as_ref(),
        y_faer.as_ref(),
        2,
        None,
        PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        },
        Some(42),
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("beta_ref", &vec_to_array(&result.beta_ref))?;
        w.add_f64("beta_perm_mean", &vec_to_array(&result.beta_perm_mean))?;
        w.add_f64("beta_perm_sd", &vec_to_array(&result.beta_perm_sd))?;
        w.add_f64("beta_perm_z", &vec_to_array(&result.beta_perm_z))?;
        w.add_i64("n_perm", &scalar_i64(i64::try_from(result.n_perm)?))?;
        w.add_i64("k", &scalar_i64(i64::try_from(result.k)?))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(result.seed)?))?;
        w.add_f64("n_eff", &scalar_f64(result.n_eff))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "n": 80, "d": 6, "k": 2, "n_perm": 200,
            "seed": 42, "disable_parallelism": true
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
