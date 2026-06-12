//! `spls1_find_keep_optimal` fixture case (mode 2 — keep sweep at fixed k).

use std::path::Path;

use anyhow::Result;

use crate::cases::pls1_find_k_optimal::write_btreemap;
use crate::cases::{ndarray_to_faer_col, ndarray_to_faer_mat, scalar_i64, synth_data, CasePaths};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use plskit::{spls1_find_keep_optimal, FindKeepOptimalOpts};

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

/// Case: keep sweep at fixed k=1 on (n=80, d=6), seed=42.
///
/// # Errors
/// Returns an error if fixture files cannot be written or the sweep fails.
pub fn k1(root: &Path) -> Result<Case> {
    let function = "spls1_find_keep_optimal";
    let name = "spls1_find_keep_optimal_k1";
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
    let r = spls1_find_keep_optimal(
        x_faer.as_ref(),
        y_faer.as_ref(),
        1,
        None,
        FindKeepOptimalOpts {
            seed: Some(42),
            ..Default::default()
        },
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_i64("keep_star", &scalar_i64(i64::try_from(r.keep_star)?))?;
        w.add_i64("k", &scalar_i64(i64::try_from(r.k)?))?;
        write_btreemap(&mut w, "cv_scores__keys", "cv_scores__values", &r.cv_scores)?;
        write_btreemap(
            &mut w,
            "cv_scores_se__keys",
            "cv_scores_se__values",
            &r.cv_scores_se,
        )?;
        let grid: Vec<i64> = r
            .keep_grid
            .iter()
            .map(|&v| i64::try_from(v))
            .collect::<std::result::Result<_, _>>()?;
        w.add_i64("keep_grid", &ndarray::Array1::from_vec(grid).into_dyn())?;
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"k": 1, "seed": 42}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
