//! `pls1_find_k_sequence` fixture cases (Family C of Task 5).

use std::path::Path;

use anyhow::Result;

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}
use plskit::{pls1_find_k_sequence, ConfirmatoryMethod, FindKSequenceOpts};

use crate::cases::{
    faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_data,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};

/// Shared synth parameters for all `pls1_find_k_sequence` cases.
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const SYNTH_SEED: u64 = 42;
/// Shared `k_max` for all `pls1_find_k_sequence` cases.
const K_MAX: usize = 4;
/// Shared inputs filename stem (not case-name-derived).
const INPUTS_NAME: &str = "pls1_find_k_sequence_inputs";
/// Function name for the manifest.
const FUNCTION: &str = "pls1_find_k_sequence";

/// Descriptor for one `pls1_find_k_sequence` case.
struct SequenceCase {
    name: &'static str,
    opts: FindKSequenceOpts,
    kwargs: serde_json::Value,
}

/// Generic runner shared by all four `pls1_find_k_sequence` cases.
///
/// Writes the shared inputs file (idempotent — same bytes every call) and the
/// case-specific outputs file, then returns the manifest `Case`.
#[allow(clippy::many_single_char_names)]
fn run_sequence_case(root: &Path, c: &SequenceCase) -> Result<Case> {
    let rel_inputs = format!("inputs/{INPUTS_NAME}.npz");
    let rel_outputs = format!("outputs/{FUNCTION}/{}.npz", c.name);
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
    let r = pls1_find_k_sequence(x_faer.as_ref(), y_faer.as_ref(), K_MAX, None, c.opts)?;

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
        name: c.name.to_string(),
        function: FUNCTION.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs: c.kwargs.clone(),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Case: `pls1_find_k_sequence` with `test_method=raw_perm`, `n_perm=100`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_sequence` fails.
pub fn raw_perm(root: &Path) -> Result<Case> {
    run_sequence_case(
        root,
        &SequenceCase {
            name: "pls1_find_k_sequence_raw_perm",
            opts: FindKSequenceOpts {
                test_method: ConfirmatoryMethod::RawPerm,
                n_perm: 100,
                seed: Some(42),
                ..FindKSequenceOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "test_method": "raw_perm",
                "args": {"n_perm": 100},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_find_k_sequence` with `test_method=split_nb`, `n_splits=20`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_sequence` fails.
pub fn split_nb(root: &Path) -> Result<Case> {
    run_sequence_case(
        root,
        &SequenceCase {
            name: "pls1_find_k_sequence_split_nb",
            opts: FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_splits: 20,
                seed: Some(42),
                ..FindKSequenceOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "test_method": "split_nb",
                "args": {"n_splits": 20},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_find_k_sequence` with `test_method=split_perm`, `n_perm=100`, `n_splits=20`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_sequence` fails.
pub fn split_perm(root: &Path) -> Result<Case> {
    run_sequence_case(
        root,
        &SequenceCase {
            name: "pls1_find_k_sequence_split_perm",
            opts: FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitPerm,
                n_perm: 100,
                n_splits: 20,
                seed: Some(42),
                ..FindKSequenceOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "test_method": "split_perm",
                "args": {"n_perm": 100, "n_splits": 20},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_find_k_sequence` with `test_method=e`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_sequence` fails.
pub fn e(root: &Path) -> Result<Case> {
    run_sequence_case(
        root,
        &SequenceCase {
            name: "pls1_find_k_sequence_e",
            opts: FindKSequenceOpts {
                test_method: ConfirmatoryMethod::E,
                seed: Some(42),
                ..FindKSequenceOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "test_method": "e",
                "args": {},
                "seed": 42
            }),
        },
    )
}
