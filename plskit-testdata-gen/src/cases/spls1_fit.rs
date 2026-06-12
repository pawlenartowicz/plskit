//! `spls1_fit` fixture cases (sparse PLS1).

use std::path::Path;

use anyhow::Result;

use crate::cases::{
    faer_col_to_array, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64,
    synth_data, CasePaths,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use plskit::{spls1_fit, FitOpts, KSpec};

/// Default numerical tolerances: atol_scalar=1e-12, atol_array=1e-10.
fn default_tolerance() -> serde_json::Value {
    serde_json::json!({"atol_scalar": 1e-12, "atol_array": 1e-10})
}

struct Spls1FitCase<'a> {
    name: &'a str,
    n: usize,
    d: usize,
    k_signal: usize,
    snr: f64,
    seed: u64,
    k: usize,
    keep: usize,
    kwargs: serde_json::Value,
}

fn spls1_fit_case(root: &Path, c: &Spls1FitCase<'_>) -> Result<Case> {
    let function = "spls1_fit";
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
    let model = spls1_fit(
        x_faer.as_ref(),
        y_faer.as_ref(),
        KSpec::Fixed(c.k),
        c.keep,
        None,
        FitOpts::default(),
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("coef", &faer_col_to_array(&model.coef))?;
        w.add_f64("beta", &faer_col_to_array(&model.beta))?;
        w.add_f64("intercept", &scalar_f64(model.intercept))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(model.k_used)?))?;
        w.add_i64(
            "keep",
            &scalar_i64(i64::try_from(
                model.keep.expect("spls1_fit always sets keep"),
            )?),
        )?;
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

/// Case: wide (n=30, d=100), k=2, keep=8 — the p ≫ n motivation regime.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_fit` fails.
pub fn wide_n30_d100_k2_keep8(root: &Path) -> Result<Case> {
    spls1_fit_case(
        root,
        &Spls1FitCase {
            name: "spls1_fit_wide_n30_d100_k2_keep8",
            n: 30,
            d: 100,
            k_signal: 2,
            snr: 4.0,
            seed: 42,
            k: 2,
            keep: 8,
            kwargs: serde_json::json!({"k": 2, "keep": 8, "seed": 42}),
        },
    )
}

/// Case: small (n=50, d=10), k=2, keep=3.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `spls1_fit` fails.
pub fn small_n50_d10_k2_keep3(root: &Path) -> Result<Case> {
    spls1_fit_case(
        root,
        &Spls1FitCase {
            name: "spls1_fit_small_n50_d10_k2_keep3",
            n: 50,
            d: 10,
            k_signal: 2,
            snr: 4.0,
            seed: 42,
            k: 2,
            keep: 3,
            kwargs: serde_json::json!({"k": 2, "keep": 3, "seed": 42}),
        },
    )
}
