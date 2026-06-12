//! `spls1_find_k_sequence` fixture case (sparse PLS1, mode 4 — sequential test at fixed keep).

use std::path::Path;

use anyhow::Result;

use crate::cases::{
    faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_data,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use plskit::{spls1_find_k_sequence, ConfirmatoryMethod, FindKSequenceOpts};

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

/// Shared synth parameters (mirrors dense `pls1_find_k_sequence` cases).
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const SYNTH_SEED: u64 = 42;
const K_MAX: usize = 4;
const KEEP: usize = 3;
/// Shared inputs filename stem.
const INPUTS_NAME: &str = "spls1_find_k_sequence_inputs";
const FUNCTION: &str = "spls1_find_k_sequence";

/// Case: `spls1_find_k_sequence` with `test_method=split_nb`, `keep=3`, `n_splits=20`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_find_k_sequence` fails.
pub fn split_nb_keep3(root: &Path) -> Result<Case> {
    let name = "spls1_find_k_sequence_split_nb_keep3";
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
    let r = spls1_find_k_sequence(
        x_faer.as_ref(),
        y_faer.as_ref(),
        K_MAX,
        KEEP,
        None,
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitNb,
            n_splits: 20,
            seed: Some(42),
            ..FindKSequenceOpts::default()
        },
    )?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_i64("k_star", &scalar_i64(i64::try_from(r.k_star)?))?;
        w.add_f64("pvalues", &faer_col_to_array(&r.pvalues))?;
        w.add_string("test_method", &r.test_method)?;
        w.add_f64("alpha", &scalar_f64(r.alpha))?;
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
            "test_method": "split_nb",
            "args": {"n_splits": 20},
            "seed": 42
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
