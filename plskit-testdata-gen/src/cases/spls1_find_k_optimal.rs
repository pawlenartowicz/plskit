//! `spls1_find_k_optimal` fixture case (sparse PLS1, mode 3 — k sweep at fixed keep).

use std::path::Path;

use anyhow::Result;

use crate::cases::pls1_find_k_optimal::write_btreemap;
use crate::cases::{
    faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_i64, synth_data,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use plskit::{spls1_find_k_optimal, FindKOptimalOpts, Selector};

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

/// Shared synth parameters (mirrors dense `pls1_find_k_optimal` cases).
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const SYNTH_SEED: u64 = 42;
const K_MAX: usize = 4;
const KEEP: usize = 3;
/// Shared inputs filename stem.
const INPUTS_NAME: &str = "spls1_find_k_optimal_inputs";
const FUNCTION: &str = "spls1_find_k_optimal";

/// Case: `spls1_find_k_optimal` with `selector=r2_se`, `keep=3`, `n_folds=5`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_find_k_optimal` fails.
pub fn r2_se_keep3(root: &Path) -> Result<Case> {
    let name = "spls1_find_k_optimal_r2_se_keep3";
    let rel_inputs = format!("inputs/{INPUTS_NAME}.npz");
    let rel_outputs = format!("outputs/{FUNCTION}/{name}.npz");
    let abs_inputs = root.join(&rel_inputs);
    let abs_outputs = root.join(&rel_outputs);

    if let Some(p) = abs_inputs.parent() {
        std::fs::create_dir_all(p)?;
    }
    if let Some(p) = abs_outputs.parent() {
        std::fs::create_dir_all(p)?;
    }

    let (x, y) = synth_data(SYNTH_N, SYNTH_D, SYNTH_K_SIGNAL, SYNTH_SNR, SYNTH_SEED);

    // Write shared inputs (idempotent — same bytes every call).
    {
        let mut w = NpzWriter::create(&abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let r = spls1_find_k_optimal(
        x_faer.as_ref(),
        y_faer.as_ref(),
        K_MAX,
        KEEP,
        None,
        FindKOptimalOpts {
            selector: Selector::R2Se,
            n_folds: 5,
            seed: Some(42),
            ..FindKOptimalOpts::default()
        },
    )?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_i64("k_star", &scalar_i64(i64::try_from(r.k_star)?))?;
        w.add_string("selector", &r.selector)?;
        if let Some(m) = &r.cv_scores {
            write_btreemap(&mut w, "cv_scores__keys", "cv_scores__values", m)?;
        }
        if let Some(m) = &r.cv_scores_se {
            write_btreemap(&mut w, "cv_scores_se__keys", "cv_scores_se__values", m)?;
        }
        if let Some(m) = &r.bic_scores {
            write_btreemap(&mut w, "bic_scores__keys", "bic_scores__values", m)?;
        }
        if let Some(ref col) = r.pvalues {
            w.add_f64("pvalues", &faer_col_to_array(col))?;
        }
        if let Some(ref s) = r.diagnostic {
            w.add_string("diagnostic", s)?;
        }
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: FUNCTION.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs: serde_json::json!({
            "k_max": K_MAX,
            "keep": KEEP,
            "selector": "r2_se",
            "args": {"n_folds": 5},
            "seed": 42
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
