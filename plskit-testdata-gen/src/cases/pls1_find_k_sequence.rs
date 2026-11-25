//! `pls1_find_k_sequence` fixture cases.

use std::path::Path;

use anyhow::Result;

use plskit::{ConfirmatoryMethod, FindKSequenceOpts, pls1_find_k_sequence};

use crate::cases::{
    Xyw, add_counts, array_descriptors, faer_col_to_array, scalar_f64, scalar_i64,
    synth_covariates, synth_data,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{NpzWriter, sha256_of_file};

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
    let (x, y) = synth_data(SYNTH_N, SYNTH_D, SYNTH_K_SIGNAL, SYNTH_SNR, SYNTH_SEED);
    run_sequence_on(
        root,
        c,
        INPUTS_NAME,
        &Xyw {
            x,
            y,
            ..Default::default()
        },
    )
}

/// [`run_sequence_case`] on `data`, written to `inputs/{inputs_name}.npz`
/// (with `covariates` when `data` has them, which the call takes).
fn run_sequence_on(root: &Path, c: &SequenceCase, inputs_name: &str, data: &Xyw) -> Result<Case> {
    let rel_inputs = format!("inputs/{inputs_name}.npz");
    let rel_outputs = format!("outputs/{FUNCTION}/{}.npz", c.name);
    let abs_inputs = root.join(&rel_inputs);
    let abs_outputs = root.join(&rel_outputs);

    if let Some(p) = abs_inputs.parent() {
        std::fs::create_dir_all(p)?;
    }
    if let Some(p) = abs_outputs.parent() {
        std::fs::create_dir_all(p)?;
    }

    // Write the inputs (idempotent — same bytes every call).
    data.write(&abs_inputs)?;

    let z_faer = data.z_faer();
    let r = pls1_find_k_sequence(
        data.x_faer().as_ref(),
        data.y_faer().as_ref(),
        K_MAX,
        None,
        z_faer.as_ref().map(faer::Mat::as_ref),
        c.opts,
    )?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_i64("k_star", &scalar_i64(i64::try_from(r.k_star)?))?;
        w.add_f64("pvalues", &faer_col_to_array(&r.pvalues))?;
        w.add_string("test_method", &r.test_method)?;
        w.add_f64("alpha", &scalar_f64(r.alpha))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        add_counts(&mut w, "n_failed_splits", r.n_failed_splits.as_deref())?;
        add_counts(&mut w, "n_failed_perms", r.n_failed_perms.as_deref())?;
        if let Some(nc) = r.n_covariates {
            w.add_i64("n_covariates", &scalar_i64(i64::try_from(nc)?))?;
        }
        w.finish()?;
    }

    let mut kwargs = c.kwargs.clone();
    array_descriptors(&mut kwargs, data);
    Ok(Case {
        name: c.name.to_string(),
        function: FUNCTION.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs,
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
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

/// Case: `pls1_find_k_sequence` with `test_method=split_exact`, `n_perm=100`, `n_splits=20`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_sequence` fails.
pub fn split_exact(root: &Path) -> Result<Case> {
    run_sequence_case(
        root,
        &SequenceCase {
            name: "pls1_find_k_sequence_split_exact",
            opts: FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitExact,
                n_perm: 100,
                n_splits: 20,
                seed: Some(42),
                ..FindKSequenceOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "test_method": "split_exact",
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

/// Case: `pls1_find_k_sequence` with `test_method=auto`, `n_perm=100`, `n_splits=20`, `seed=42`.
/// At n = 80 the rule resolves every step to `split_exact`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_sequence` fails.
pub fn auto(root: &Path) -> Result<Case> {
    run_sequence_case(
        root,
        &SequenceCase {
            name: "pls1_find_k_sequence_auto",
            opts: FindKSequenceOpts {
                test_method: ConfirmatoryMethod::Auto,
                n_perm: 100,
                n_splits: 20,
                seed: Some(42),
                ..FindKSequenceOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "test_method": "auto",
                "args": {"n_perm": 100, "n_splits": 20},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_find_k_sequence` with `test_method=split_exact`, `n_perm=100`,
/// `n_splits=20`, `seed=42`, on the shared design with three covariates, the
/// third an alias of the first.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_sequence` fails.
pub fn covariates(root: &Path) -> Result<Case> {
    let (x, y) = synth_data(SYNTH_N, SYNTH_D, SYNTH_K_SIGNAL, SYNTH_SNR, SYNTH_SEED);
    let z = Some(synth_covariates(&x, 7));
    run_sequence_on(
        root,
        &SequenceCase {
            name: "pls1_find_k_sequence_covariates",
            opts: FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitExact,
                n_perm: 100,
                n_splits: 20,
                seed: Some(42),
                ..FindKSequenceOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "test_method": "split_exact",
                "args": {"n_perm": 100, "n_splits": 20},
                "seed": 42
            }),
        },
        "pls1_find_k_sequence_covariates_inputs",
        &Xyw {
            x,
            y,
            z,
            ..Default::default()
        },
    )
}
