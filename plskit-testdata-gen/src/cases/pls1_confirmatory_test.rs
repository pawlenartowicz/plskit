//! `pls1_confirmatory_test` fixture cases (Family D of Task 5).
//!
//! Six cases share one inputs file (`inputs/pls1_confirmatory_inputs.npz`):
//! five base-method cases (no CI) and one CI-bundle variant.

use std::path::Path;

use anyhow::Result;

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}
use plskit::{
    pls1_confirmatory_test, CIOpts, ConfirmatoryArgs, ConfirmatoryCI, ConfirmatoryTestInput,
    ConfirmatoryTestOpts,
};

use crate::cases::{ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_data};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};

/// Shared synth parameters for all `pls1_confirmatory_test` cases.
const SYNTH_N: usize = 80;
const SYNTH_D: usize = 6;
const SYNTH_K_SIGNAL: usize = 2;
const SYNTH_SNR: f64 = 4.0;
const SYNTH_SEED: u64 = 42;
/// Shared `k` for all `pls1_confirmatory_test` cases.
const K: usize = 2;
/// Shared RNG seed for all cases.
const CASE_SEED: u64 = 42;
/// Shared inputs filename stem (legacy name — no `_test_` in middle).
const INPUTS_NAME: &str = "pls1_confirmatory_inputs";
/// Function name for the manifest.
const FUNCTION: &str = "pls1_confirmatory_test";

/// Descriptor for one `pls1_confirmatory_test` case.
struct ConfirmatoryCase {
    name: &'static str,
    args: ConfirmatoryArgs,
    ci: Option<CIOpts>,
    disable_parallelism: bool,
    kwargs: serde_json::Value,
}

/// Write the `ConfirmatoryCI` bundle fields into `w`.
///
/// Each `Vec<f64>` field is encoded as a 1-D `ArrayD<f64>`.
/// Each `CIScalar` is split into four separate 0-D float fields.
/// Integral counts use `i64::try_from` to avoid silent truncation.
fn write_ci_fields(w: &mut NpzWriter, ci: &ConfirmatoryCI) -> Result<()> {
    let to_arr = |v: &Vec<f64>| ndarray::Array1::from_vec(v.clone()).into_dyn();

    w.add_i64("n_boot", &scalar_i64(i64::try_from(ci.n_boot)?))?;
    w.add_i64("m", &scalar_i64(i64::try_from(ci.m)?))?;
    w.add_f64("m_rate", &scalar_f64(ci.m_rate))?;
    w.add_f64("level", &scalar_f64(ci.level))?;
    w.add_f64("beta_sign_z", &to_arr(&ci.beta_sign_z))?;
    w.add_f64("beta_sign_z_signed", &to_arr(&ci.beta_sign_z_signed))?;
    w.add_f64("leverage_ci_lower", &to_arr(&ci.leverage_ci_lower))?;
    w.add_f64("leverage_ci_upper", &to_arr(&ci.leverage_ci_upper))?;
    w.add_f64("leverage_se", &to_arr(&ci.leverage_se))?;
    w.add_f64("beta_ci_lower", &to_arr(&ci.beta_ci_lower))?;
    w.add_f64("beta_ci_upper", &to_arr(&ci.beta_ci_upper))?;
    w.add_f64("beta_se", &to_arr(&ci.beta_se))?;
    w.add_f64("holdout_corr_point", &scalar_f64(ci.holdout_corr.point))?;
    w.add_f64("holdout_corr_lower", &scalar_f64(ci.holdout_corr.lower))?;
    w.add_f64("holdout_corr_upper", &scalar_f64(ci.holdout_corr.upper))?;
    w.add_f64("holdout_corr_sd", &scalar_f64(ci.holdout_corr.sd))?;
    w.add_i64(
        "n_boot_finite",
        &scalar_i64(i64::try_from(ci.n_boot_finite)?),
    )?;
    w.add_i64(
        "n_boot_finite_holdout_corr",
        &scalar_i64(i64::try_from(ci.n_boot_finite_holdout_corr)?),
    )?;
    Ok(())
}

/// Generic runner shared by all six `pls1_confirmatory_test` cases.
///
/// Writes the shared inputs file (idempotent — same bytes every call) and the
/// case-specific outputs file, then returns the manifest `Case`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
#[allow(clippy::many_single_char_names)]
fn run_confirmatory_case(root: &Path, c: &ConfirmatoryCase) -> Result<Case> {
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
    let r = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x_faer.as_ref(),
            y: y_faer.as_ref(),
            k: K,
            weights: None,
        },
        ConfirmatoryTestOpts {
            args: c.args,
            pre_standardized: false,
            seed: Some(CASE_SEED),
            disable_parallelism: c.disable_parallelism,
            verbose: false,
            ci: c.ci,
            max_skip_rate: 0.01,
        },
    )?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_f64("pvalue", &scalar_f64(r.pvalue))?;
        w.add_f64("statistic", &scalar_f64(r.statistic))?;
        w.add_string("method", &r.method)?;
        w.add_i64("k", &scalar_i64(i64::try_from(r.k)?))?;
        if let Some(np) = r.n_perm {
            w.add_i64("n_perm", &scalar_i64(i64::try_from(np)?))?;
        }
        if let Some(ns) = r.n_splits {
            w.add_i64("n_splits", &scalar_i64(i64::try_from(ns)?))?;
        }
        w.add_i64("seed", &scalar_i64(i64::try_from(r.seed)?))?;
        if let Some(ci) = &r.ci {
            write_ci_fields(&mut w, ci)?;
        }
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

/// Case: `pls1_confirmatory_test` with `method=raw_perm`, `n_perm=200`, `n_folds=5`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn raw_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_raw_perm",
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 200,
                n_folds: 5,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "raw_perm",
                "args": {"n_perm": 200, "n_folds": 5},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_confirmatory_test` with `method=split_nb`, `n_splits=30`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_nb(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_nb",
            args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "split_nb",
                "args": {"n_splits": 30},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_confirmatory_test` with `method=split_perm`, `n_perm=200`, `n_splits=30`, `seed=42`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_perm",
            args: ConfirmatoryArgs::SplitPerm {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "split_perm",
                "args": {"n_perm": 200, "n_splits": 30},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_confirmatory_test` with `method=score`, `seed=42`.
///
/// Closed-form score test — no permutation or split count.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn score(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_score",
            args: ConfirmatoryArgs::Score,
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "score",
                "args": {},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_confirmatory_test` with `method=e`, `seed=42`.
///
/// Universal-inference split-LR e-value — no permutation or split count.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn e(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_e",
            args: ConfirmatoryArgs::E,
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "e",
                "args": {},
                "seed": 42
            }),
        },
    )
}

/// Case: `pls1_confirmatory_test` with `method=split_nb` + CI bundle (`n_boot=300`), `seed=42`.
///
/// Exercises the `ci = Some(CIOpts { ... })` path. Parallelism is disabled
/// (`disable_parallelism: true`) for fully deterministic output across runs.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_nb_ci(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_nb_ci",
            args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
            ci: Some(CIOpts {
                n_boot: 300,
                m_rate: 0.7,
                level: 0.95,
                max_failure_rate: 0.0,
            }),
            disable_parallelism: true,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "split_nb",
                "args": {"n_splits": 30},
                "ci": true,
                "n_boot": 300,
                "m_rate": 0.7,
                "level": 0.95,
                "seed": 42,
                "disable_parallelism": true,
                "max_failure_rate": 0.0
            }),
        },
    )
}
