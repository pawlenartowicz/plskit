//! `pls1_confirmatory_test` fixture cases (Family D of Task 5).
//!
//! Seven unweighted cases share one inputs file (`inputs/pls1_confirmatory_inputs.npz`):
//! six base-method cases (no CI; two of them, `split_exact` and `split_exact_k1`, cover
//! `split_exact`'s refit and no-refit routes respectively) and one CI-bundle variant.
//! Five weighted cases share a separate inputs file
//! (`inputs/pls1_confirmatory_weighted_inputs.npz`) that also carries `weights`.

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
/// Shared RNG seed for all cases.
const CASE_SEED: u64 = 42;
/// Function name for the manifest.
const FUNCTION: &str = "pls1_confirmatory_test";

/// Descriptor for one `pls1_confirmatory_test` case.
struct ConfirmatoryCase {
    name: &'static str,
    args: ConfirmatoryArgs,
    ci: Option<CIOpts>,
    disable_parallelism: bool,
    kwargs: serde_json::Value,
    /// Inputs file stem and synth seed; set to the weighted variants for weighted cases.
    inputs_name: &'static str,
    synth_seed: u64,
    k: usize,
    /// Non-uniform weights array (first 40 obs get 2.0, rest 1.0); `None` for unweighted cases.
    weights: Option<ndarray::Array1<f64>>,
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

/// Generic runner shared by all eleven `pls1_confirmatory_test` cases (unweighted + weighted).
///
/// Writes the shared inputs file (idempotent — same bytes every call) and the
/// case-specific outputs file, then returns the manifest `Case`.
/// Weighted cases set `c.weights = Some(...)` and use distinct `c.inputs_name`/`c.synth_seed`/`c.k`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
#[allow(clippy::many_single_char_names)]
fn run_confirmatory_case(root: &Path, c: &ConfirmatoryCase) -> Result<Case> {
    let rel_inputs = format!("inputs/{}.npz", c.inputs_name);
    let rel_outputs = format!("outputs/{FUNCTION}/{}.npz", c.name);
    let abs_inputs = root.join(&rel_inputs);
    let abs_outputs = root.join(&rel_outputs);

    if let Some(p) = abs_inputs.parent() {
        std::fs::create_dir_all(p)?;
    }
    if let Some(p) = abs_outputs.parent() {
        std::fs::create_dir_all(p)?;
    }

    let (x, y) = synth_data(SYNTH_N, SYNTH_D, SYNTH_K_SIGNAL, SYNTH_SNR, c.synth_seed);

    // Write shared inputs (idempotent — same bytes every call).
    {
        let mut w = NpzWriter::create(&abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        if let Some(ref wt) = c.weights {
            w.add_f64("weights", &wt.clone().into_dyn())?;
        }
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let weights_faer = c.weights.as_ref().map(ndarray_to_faer_col);
    let r = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x_faer.as_ref(),
            y: y_faer.as_ref(),
            k: c.k,
            weights: weights_faer.as_ref().map(faer::Col::as_ref),
        },
        ConfirmatoryTestOpts {
            args: c.args,
            pre_standardized: false,
            seed: Some(CASE_SEED),
            disable_parallelism: c.disable_parallelism,
            verbose: false,
            ci: c.ci,
            max_skip_rate: 0.01,
            keep: None,
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
        if let Some(sr) = r.stable_rank {
            w.add_f64("stable_rank", &scalar_f64(sr))?;
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
            inputs_name: "pls1_confirmatory_inputs",
            synth_seed: 42,
            k: 2,
            weights: None,
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
            args: ConfirmatoryArgs::SplitNb {
                n_splits: 30,
                force: false,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "split_nb",
                "args": {"n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_inputs",
            synth_seed: 42,
            k: 2,
            weights: None,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `method=split_exact`, `n_perm=200`, `n_splits=30`,
/// `seed=42`. `k=2` sends this through `split_exact`'s honest-refit route.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_exact",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 2,
                "method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_inputs",
            synth_seed: 42,
            k: 2,
            weights: None,
        },
    )
}

/// Case: `pls1_confirmatory_test` with `method=split_exact`, `n_perm=200`, `n_splits=30`,
/// `seed=42`, `k=1`. Unweighted dense K = 1: exercises `split_exact`'s no-refit route
/// (the [`split_exact`](self::split_exact) case above covers the refit route via K = 2;
/// [`weighted_split_exact`] covers the no-refit route under weights).
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn split_exact_k1(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_split_exact_k1",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 30,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 1,
                "method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 30},
                "seed": 42
            }),
            inputs_name: "pls1_confirmatory_inputs",
            synth_seed: 42,
            k: 1,
            weights: None,
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
            inputs_name: "pls1_confirmatory_inputs",
            synth_seed: 42,
            k: 2,
            weights: None,
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
            inputs_name: "pls1_confirmatory_inputs",
            synth_seed: 42,
            k: 2,
            weights: None,
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
            args: ConfirmatoryArgs::SplitNb {
                n_splits: 30,
                force: false,
            },
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
            inputs_name: "pls1_confirmatory_inputs",
            synth_seed: 42,
            k: 2,
            weights: None,
        },
    )
}

/// Non-uniform weights for weighted confirmatory cases: first 40 of 80 obs get 2.0, rest 1.0.
fn weighted_confirmatory_weights() -> ndarray::Array1<f64> {
    ndarray::Array1::from_shape_fn(SYNTH_N, |i| if i < 40 { 2.0_f64 } else { 1.0_f64 })
}

/// Case: weighted `pls1_confirmatory_test` with `method=raw_perm`, `n_perm=200`, `n_folds=5`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_raw_perm(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_raw_perm",
            args: ConfirmatoryArgs::RawPerm {
                n_perm: 200,
                n_folds: 5,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 1,
                "method": "raw_perm",
                "args": {"n_perm": 200, "n_folds": 5},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            synth_seed: 77,
            k: 1,
            weights: Some(weighted_confirmatory_weights()),
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `method=split_nb`, `n_splits=50`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_split_nb(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_split_nb",
            args: ConfirmatoryArgs::SplitNb {
                n_splits: 50,
                force: false,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 1,
                "method": "split_nb",
                "args": {"n_splits": 50},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            synth_seed: 77,
            k: 1,
            weights: Some(weighted_confirmatory_weights()),
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `method=split_exact`, `n_perm=200`,
/// `n_splits=50`. `k=1` dense under weights: exercises `split_exact`'s weighted
/// no-refit route.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_split_exact(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_split_exact",
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 200,
                n_splits: 50,
            },
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 1,
                "method": "split_exact",
                "args": {"n_perm": 200, "n_splits": 50},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            synth_seed: 77,
            k: 1,
            weights: Some(weighted_confirmatory_weights()),
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `method=score`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_score(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_score",
            args: ConfirmatoryArgs::Score,
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 1,
                "method": "score",
                "args": {},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            synth_seed: 77,
            k: 1,
            weights: Some(weighted_confirmatory_weights()),
        },
    )
}

/// Case: weighted `pls1_confirmatory_test` with `method=e`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_confirmatory_test` fails.
pub fn weighted_e(root: &Path) -> Result<Case> {
    run_confirmatory_case(
        root,
        &ConfirmatoryCase {
            name: "pls1_confirmatory_weighted_e",
            args: ConfirmatoryArgs::E,
            ci: None,
            disable_parallelism: false,
            kwargs: serde_json::json!({
                "k": 1,
                "method": "e",
                "args": {},
                "seed": 42,
                "weights": "nonuniform"
            }),
            inputs_name: "pls1_confirmatory_weighted_inputs",
            synth_seed: 77,
            k: 1,
            weights: Some(weighted_confirmatory_weights()),
        },
    )
}
