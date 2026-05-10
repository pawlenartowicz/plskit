//! `pls1_find_k_optimal` fixture cases (Family B of Task 5).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}
use plskit::{pls1_find_k_optimal, ConfirmatoryMethod, FindKOptimalOpts, Selector};

use crate::cases::{
    faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_i64, synth_data,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};

/// Shared synth parameters for all `pls1_find_k_optimal` cases.
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const SYNTH_SEED: u64 = 42;
/// Shared `k_max` for all `pls1_find_k_optimal` cases.
const K_MAX: usize = 4;
/// Shared inputs filename stem (not case-name-derived).
const INPUTS_NAME: &str = "pls1_find_k_optimal_inputs";
/// Function name for the manifest.
const FUNCTION: &str = "pls1_find_k_optimal";

/// Descriptor for one `pls1_find_k_optimal` case.
struct OptimalCase {
    name: &'static str,
    opts: FindKOptimalOpts,
    kwargs: serde_json::Value,
}

/// Write a `BTreeMap<usize, f64>` as two parallel 1-D arrays into `w`.
///
/// Keys are written to `<key_field>` (1-D i64) and values to `<value_field>` (1-D f64).
/// `BTreeMap` iteration is sorted by key, which is exactly the required order.
fn write_btreemap(
    w: &mut NpzWriter,
    key_field: &str,
    value_field: &str,
    map: &BTreeMap<usize, f64>,
) -> Result<()> {
    let keys: Vec<i64> = map
        .keys()
        .map(|k| i64::try_from(*k))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let values: Vec<f64> = map.values().copied().collect();
    let keys_arr = ndarray::Array1::from_vec(keys).into_dyn();
    let values_arr = ndarray::Array1::from_vec(values).into_dyn();
    w.add_i64(key_field, &keys_arr)?;
    w.add_f64(value_field, &values_arr)?;
    Ok(())
}

/// Generic runner shared by all four `pls1_find_k_optimal` cases.
///
/// Writes the shared inputs file (idempotent — same bytes every call) and the
/// case-specific outputs file, then returns the manifest `Case`.
#[allow(clippy::many_single_char_names)]
fn run_optimal_case(root: &Path, c: &OptimalCase) -> Result<Case> {
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
    let r = pls1_find_k_optimal(x_faer.as_ref(), y_faer.as_ref(), K_MAX, None, c.opts)?;

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

/// Case: `pls1_find_k_optimal` with `selector=r2_se`, `n_folds=5`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_optimal` fails.
pub fn r2_se(root: &Path) -> Result<Case> {
    run_optimal_case(
        root,
        &OptimalCase {
            name: "pls1_find_k_optimal_r2_se",
            opts: FindKOptimalOpts {
                selector: Selector::R2Se,
                n_folds: 5,
                seed: Some(42),
                ..FindKOptimalOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "selector": "r2_se",
                "args": {"n_folds": 5},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_find_k_optimal` with `selector=r2_max`, `n_folds=5`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_optimal` fails.
pub fn r2_max(root: &Path) -> Result<Case> {
    run_optimal_case(
        root,
        &OptimalCase {
            name: "pls1_find_k_optimal_r2_max",
            opts: FindKOptimalOpts {
                selector: Selector::R2Max,
                n_folds: 5,
                seed: Some(42),
                ..FindKOptimalOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "selector": "r2_max",
                "args": {"n_folds": 5},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_find_k_optimal` with `selector=bic`, `seed=42`.
///
/// BIC does not use `n_folds`, so it is absent from `kwargs`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_optimal` fails.
pub fn bic(root: &Path) -> Result<Case> {
    run_optimal_case(
        root,
        &OptimalCase {
            name: "pls1_find_k_optimal_bic",
            opts: FindKOptimalOpts {
                selector: Selector::Bic,
                seed: Some(42),
                ..FindKOptimalOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "selector": "bic",
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_find_k_optimal` with `selector=r2_se`, diagnostic enabled via `split_nb`, `seed=42`.
///
/// Exercises the diagnostic path and verifies that `pvalues` and `diagnostic`
/// are present in the output.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_find_k_optimal` fails.
pub fn r2_se_diagnostic(root: &Path) -> Result<Case> {
    run_optimal_case(
        root,
        &OptimalCase {
            name: "pls1_find_k_optimal_r2_se_diagnostic",
            opts: FindKOptimalOpts {
                selector: Selector::R2Se,
                n_folds: 5,
                diagnostic: Some(ConfirmatoryMethod::SplitNb),
                n_splits: 30,
                seed: Some(42),
                ..FindKOptimalOpts::default()
            },
            kwargs: serde_json::json!({
                "k_max": 4,
                "selector": "r2_se",
                "diagnostic": "split_nb",
                "args": {"n_folds": 5, "n_splits": 30},
                "seed": 42
            }),
        },
    )
}
