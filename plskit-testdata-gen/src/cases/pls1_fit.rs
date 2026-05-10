//! `pls1_fit` fixture cases.

use crate::cases::{
    faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64,
    synth_data, CasePaths,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::{bail, Result};

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}
use plskit::{
    pls1_find_k_sequence, pls1_fit, ConfirmatoryMethod, FindKSequenceOpts, FitOpts, KSpec,
};
use std::path::Path;

/// Synth parameters for a fixed-k `pls1_fit` case.
struct FitFixedKCase<'a> {
    name: &'a str,
    n: usize,
    d: usize,
    k_signal: usize,
    snr: f64,
    seed: u64,
    kspec: KSpec,
    kwargs: serde_json::Value,
}

/// Generic fixed-k `pls1_fit` case writer. Used by all 4 fixed-k cases below.
fn fit_fixed_k(root: &Path, c: &FitFixedKCase<'_>) -> Result<Case> {
    let function = "pls1_fit";
    let paths = CasePaths::build(root, function, c.name)?;
    let (x, y) = synth_data(c.n, c.d, c.k_signal, c.snr, c.seed);

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
        c.kspec,
        None,
        FitOpts::default(),
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("coef", &faer_col_to_array(&model.coef))?;
        w.add_f64("beta", &faer_col_to_array(&model.beta))?;
        w.add_f64("intercept", &scalar_f64(model.intercept))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(model.k_used)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: c.name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: c.kwargs.clone(),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Case: small (n=50, d=10), `k_signal=2`, fixed k=1, seed=42.
///
/// Legacy field set: `coef`, `beta`, `intercept`, `k_used`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_fit` fails.
pub fn small_n50_d10_k1(root: &Path) -> Result<Case> {
    fit_fixed_k(
        root,
        &FitFixedKCase {
            name: "pls1_fit_small_n50_d10_k1",
            n: 50,
            d: 10,
            k_signal: 2,
            snr: 4.0,
            seed: 42,
            kspec: KSpec::Fixed(1),
            kwargs: serde_json::json!({"k": 1, "seed": 42}),
        },
    )
}

/// Case: small (n=50, d=10), `k_signal=2`, fixed k=3, seed=42.
///
/// Legacy field set: `coef`, `beta`, `intercept`, `k_used`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_fit` fails.
pub fn small_n50_d10_k3(root: &Path) -> Result<Case> {
    fit_fixed_k(
        root,
        &FitFixedKCase {
            name: "pls1_fit_small_n50_d10_k3",
            n: 50,
            d: 10,
            k_signal: 2,
            snr: 4.0,
            seed: 42,
            kspec: KSpec::Fixed(3),
            kwargs: serde_json::json!({"k": 3, "seed": 42}),
        },
    )
}

/// Case: wide (n=30, d=100), `k_signal=2`, fixed k=1, seed=42.
///
/// Legacy field set: `coef`, `beta`, `intercept`, `k_used`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_fit` fails.
pub fn wide_n30_d100_k1(root: &Path) -> Result<Case> {
    fit_fixed_k(
        root,
        &FitFixedKCase {
            name: "pls1_fit_wide_n30_d100_k1",
            n: 30,
            d: 100,
            k_signal: 2,
            snr: 4.0,
            seed: 42,
            kspec: KSpec::Fixed(1),
            kwargs: serde_json::json!({"k": 1, "seed": 42}),
        },
    )
}

/// Case: wide (n=30, d=100), `k_signal=2`, fixed k=3, seed=42.
///
/// Legacy field set: `coef`, `beta`, `intercept`, `k_used`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_fit` fails.
pub fn wide_n30_d100_k3(root: &Path) -> Result<Case> {
    fit_fixed_k(
        root,
        &FitFixedKCase {
            name: "pls1_fit_wide_n30_d100_k3",
            n: 30,
            d: 100,
            k_signal: 2,
            snr: 4.0,
            seed: 42,
            kspec: KSpec::Fixed(3),
            kwargs: serde_json::json!({"k": 3, "seed": 42}),
        },
    )
}

/// Case: small (n=50, d=10), `k_signal=2`, k selected via sequence test, seed=42.
///
/// Calls `pls1_find_k_sequence` with `k_max=4` first; errors hard if `k_star == 0`
/// (no rejection — unexpected given the current seed+parameters).
/// Otherwise fits PLS1 at `k_star` and writes both input and output `.npz` fixtures.
///
/// Legacy field set: `coef`, `beta`, `intercept`, `k_used`.
///
/// # Errors
/// Returns an error if fixture files cannot be written, either plskit call fails,
/// or `pls1_find_k_sequence` rejects no components (`k_star == 0`).
pub fn small_n50_d10_sequence(root: &Path) -> Result<Case> {
    let (x, y) = synth_data(50, 10, 2, 4.0, 42);
    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);

    let seq = pls1_find_k_sequence(
        x_faer.as_ref(),
        y_faer.as_ref(),
        4_usize,
        None,
        FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitNb,
            alpha: 0.05,
            n_perm: 1000,
            n_splits: 50,
            pre_standardized: false,
            seed: Some(42),
            disable_parallelism: false,
            verbose: false,
        },
    )?;

    if seq.k_star == 0 {
        bail!("pls1_find_k_sequence rejected no components (k_star == 0) — unexpected for this seed/parameters");
    }

    let name = "pls1_fit_small_n50_d10_sequence".to_string();
    let function = "pls1_fit";
    let paths = CasePaths::build(root, function, &name)?;

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let model = pls1_fit(
        x_faer.as_ref(),
        y_faer.as_ref(),
        KSpec::Fixed(seq.k_star),
        None,
        FitOpts::default(),
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("coef", &faer_col_to_array(&model.coef))?;
        w.add_f64("beta", &faer_col_to_array(&model.beta))?;
        w.add_f64("intercept", &scalar_f64(model.intercept))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(model.k_used)?))?;
        w.finish()?;
    }

    Ok(Case {
        name,
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"k": "sequence", "seed": 42, "k_max": 4}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Case: skinny (n=200, d=5), `k_signal=2`, fixed k=1, seed=42.
///
/// Tests behavior when n >> d (skinny regime).
/// Legacy field set: `coef`, `beta`, `intercept`, `k_used`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_fit` fails.
pub fn skinny_n200_d5_k1(root: &Path) -> Result<Case> {
    fit_fixed_k(
        root,
        &FitFixedKCase {
            name: "pls1_fit_skinny_n200_d5_k1",
            n: 200,
            d: 5,
            k_signal: 2,
            snr: 4.0,
            seed: 42,
            kspec: KSpec::Fixed(1),
            kwargs: serde_json::json!({"k": 1, "seed": 42}),
        },
    )
}
