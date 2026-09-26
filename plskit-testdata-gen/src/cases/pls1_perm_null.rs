//! `pls1_perm_null` fixture cases.

use crate::cases::{
    default_tolerance, ndarray_to_faer_col, ndarray_to_faer_mat, scalar_f64, scalar_i64,
    synth_data, CasePaths,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::Result;
use plskit::{pls1_perm_null, PermNullOpts};
use std::path::Path;

/// Convert a `Vec<f64>` to a 1-D `ndarray::ArrayD<f64>`.
fn vec_to_array(v: &[f64]) -> ndarray::ArrayD<f64> {
    ndarray::Array1::from_vec(v.to_vec()).into_dyn()
}

/// Case: `pls1_perm_null` on `(n=80, d=6)` data at `k=2`, 200 permutations.
///
/// Uses `disable_parallelism: true` for byte-exact determinism and `seed=42`.
///
/// Inputs: `X`, `y`.
/// Outputs: `beta_ref`, `beta_perm_mean`, `beta_perm_sd`, `beta_perm_z`,
///          `n_perm`, `k`, `seed`, `n_eff`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn basic_n80_d6_k2(root: &Path) -> Result<Case> {
    let name = "pls1_perm_null_basic_n80_d6_k2";
    let function = "pls1_perm_null";
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
    let result = pls1_perm_null(
        x_faer.as_ref(),
        y_faer.as_ref(),
        2,
        None,
        PermNullOpts {
            n_perm: 200,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        },
        Some(42),
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("beta_ref", &vec_to_array(&result.beta_ref))?;
        w.add_f64("beta_perm_mean", &vec_to_array(&result.beta_perm_mean))?;
        w.add_f64("beta_perm_sd", &vec_to_array(&result.beta_perm_sd))?;
        w.add_f64("beta_perm_z", &vec_to_array(&result.beta_perm_z))?;
        w.add_i64("n_perm", &scalar_i64(i64::try_from(result.n_perm)?))?;
        w.add_i64("k", &scalar_i64(i64::try_from(result.k)?))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(result.seed)?))?;
        w.add_f64("n_eff", &scalar_f64(result.n_eff))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "n": 80, "d": 6, "k": 2, "n_perm": 200,
            "seed": 42, "disable_parallelism": true
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Weights of [`weighted_n80_d6_k2`]: uneven, mean not one, and zero on
/// every tenth row. `pls1_perm_null` renormalizes them on entry and
/// `pls1_fit` renormalizes them again, so the fixture pins the √w row
/// scaling of the weighted permutation loop, zero-weight rows included.
#[allow(clippy::cast_precision_loss)]
fn perm_null_weights(n: usize) -> ndarray::Array1<f64> {
    ndarray::Array1::from_shape_fn(n, |i| {
        if i % 10 == 9 {
            0.0
        } else {
            1.0 + (i % 4) as f64 * 0.5
        }
    })
}

/// One `pls1_perm_null` fixture: `synth_data(n, d, 2, 4.0, 42)` written to
/// `inputs/{inputs_name}.npz` (plus `weights` when given), then
/// `pls1_perm_null` at `seed = 42`, serially, with the output fields of
/// [`basic_n80_d6_k2`].
#[allow(clippy::many_single_char_names)]
fn perm_null_case(
    root: &Path,
    name: &str,
    inputs_name: &str,
    (n, d): (usize, usize),
    k: usize,
    n_perm: usize,
    weights: Option<&ndarray::Array1<f64>>,
) -> Result<Case> {
    let function = "pls1_perm_null";
    let rel_inputs = format!("inputs/{inputs_name}.npz");
    let rel_outputs = format!("outputs/{function}/{name}.npz");
    let abs_inputs = root.join(&rel_inputs);
    let abs_outputs = root.join(&rel_outputs);
    for dir in [abs_inputs.parent(), abs_outputs.parent()]
        .into_iter()
        .flatten()
    {
        std::fs::create_dir_all(dir)?;
    }

    let (x, y) = synth_data(n, d, 2, 4.0, 42);
    {
        let mut w = NpzWriter::create(&abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        if let Some(wt) = weights {
            w.add_f64("weights", &wt.clone().into_dyn())?;
        }
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let w_faer = weights.map(ndarray_to_faer_col);
    let result = pls1_perm_null(
        x_faer.as_ref(),
        y_faer.as_ref(),
        k,
        w_faer.as_ref().map(faer::Col::as_ref),
        PermNullOpts {
            n_perm,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: true,
            verbose: false,
        },
        Some(42),
    )?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        w.add_f64("beta_ref", &vec_to_array(&result.beta_ref))?;
        w.add_f64("beta_perm_mean", &vec_to_array(&result.beta_perm_mean))?;
        w.add_f64("beta_perm_sd", &vec_to_array(&result.beta_perm_sd))?;
        w.add_f64("beta_perm_z", &vec_to_array(&result.beta_perm_z))?;
        w.add_i64("n_perm", &scalar_i64(i64::try_from(result.n_perm)?))?;
        w.add_i64("k", &scalar_i64(i64::try_from(result.k)?))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(result.seed)?))?;
        w.add_f64("n_eff", &scalar_f64(result.n_eff))?;
        w.finish()?;
    }

    let mut kwargs = serde_json::json!({
        "n": n, "d": d, "k": k, "n_perm": n_perm,
        "seed": 42, "disable_parallelism": true
    });
    if weights.is_some() {
        kwargs["weights"] = serde_json::json!("nonuniform");
    }
    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: rel_inputs,
        outputs: rel_outputs,
        kwargs,
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&abs_inputs)?,
            outputs_sha256: sha256_of_file(&abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Case: weighted `pls1_perm_null` on `(n=80, d=6)` at `k=2`, 200
/// permutations, weights from [`perm_null_weights`]. The primal reference
/// for the weighted permutation loop's √w hoist.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn weighted_n80_d6_k2(root: &Path) -> Result<Case> {
    let weights = perm_null_weights(80);
    perm_null_case(
        root,
        "pls1_perm_null_weighted_n80_d6_k2",
        "pls1_perm_null_weighted_n80_d6_k2",
        (80, 6),
        2,
        200,
        Some(&weights),
    )
}

/// Case: `pls1_perm_null` on a wide design `(n=60, d=3000)` at `k=1`, 200
/// permutations. Dense and unweighted, so eligible for an n-space Gram
/// route (pinned in `plskit-rs/src/fixture_route_pins.rs`); generated by
/// the primal engine, it is the reference such a route must reproduce.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn wide_n60_d3000_k1(root: &Path) -> Result<Case> {
    perm_null_case(
        root,
        "pls1_perm_null_wide_n60_d3000_k1",
        "pls1_perm_null_wide_n60_d3000_inputs",
        (60, 3000),
        1,
        200,
        None,
    )
}

/// Case: as [`wide_n60_d3000_k1`] at `k=2`, on the same inputs file.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn wide_n60_d3000_k2(root: &Path) -> Result<Case> {
    perm_null_case(
        root,
        "pls1_perm_null_wide_n60_d3000_k2",
        "pls1_perm_null_wide_n60_d3000_inputs",
        (60, 3000),
        2,
        200,
        None,
    )
}

/// Synth design shared by the two tall cases: three active columns at a
/// moderate signal-to-noise ratio, so `beta_perm_z` carries both signal and
/// null columns.
const TALL_K_SIGNAL: usize = 3;
const TALL_SNR: f64 = 0.5;
const TALL_SYNTH_SEED: u64 = 2000;
/// Tall shape of both cases.
const TALL_N: usize = 2000;
const TALL_D: usize = 50;
/// `B = 1000` puts the X backend's work per block, `B·(2k + 1)·n·d`, at
/// `7e8` (k = 3) and `5e8` (k = 2): at least 4× the Gram backend's starting
/// work floor of `1e8`, pinned in `fixture_route_pins`.
const TALL_N_PERM: usize = 1000;

/// Case: `pls1_perm_null` on a tall `(n=2000, d=50)` design at `k=3`, 1000
/// permutations, `seed=42`.
///
/// Generated by the explicit-deflation engine before any p-space Gram route
/// exists, so it is an independent reference for that route. Its route and
/// Gram-backend shape numbers are pinned in
/// `plskit-rs/src/fixture_route_pins.rs`: shrinking `n`, `d` or `n_perm`
/// silently turns it into a test of the X backend only.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn tall_n2000_d50_k3(root: &Path) -> Result<Case> {
    tall_case(root, "pls1_perm_null_tall_n2000_d50_k3", 3, None)
}

/// Case: weighted `pls1_perm_null` on the same tall design at `k=2`, 1000
/// permutations, `seed=42`. Weights cycle through 0.5, 1.0, 1.5, 2.0 by row.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn tall_weighted_n2000_d50_k2(root: &Path) -> Result<Case> {
    let weights = ndarray::Array1::from_shape_fn(TALL_N, |i| match i % 4 {
        0 => 0.5,
        1 => 1.0,
        2 => 1.5,
        _ => 2.0,
    });
    tall_case(
        root,
        "pls1_perm_null_tall_weighted_n2000_d50_k2",
        2,
        Some(&weights),
    )
}

/// Shared body of the two tall cases. Writes `X`, `y` (and `weights` when
/// given) as the case's own inputs file.
fn tall_case(
    root: &Path,
    name: &str,
    k: usize,
    weights: Option<&ndarray::Array1<f64>>,
) -> Result<Case> {
    let function = "pls1_perm_null";
    let paths = CasePaths::build(root, function, name)?;

    let (x, y) = synth_data(TALL_N, TALL_D, TALL_K_SIGNAL, TALL_SNR, TALL_SYNTH_SEED);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        if let Some(wt) = weights {
            w.add_f64("weights", &wt.clone().into_dyn())?;
        }
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let w_faer = weights.map(ndarray_to_faer_col);
    let result = pls1_perm_null(
        x_faer.as_ref(),
        y_faer.as_ref(),
        k,
        w_faer.as_ref().map(faer::Col::as_ref),
        PermNullOpts {
            n_perm: TALL_N_PERM,
            return_perm_matrix: false,
            pre_standardized: false,
            disable_parallelism: false,
            verbose: false,
        },
        Some(42),
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("beta_ref", &vec_to_array(&result.beta_ref))?;
        w.add_f64("beta_perm_mean", &vec_to_array(&result.beta_perm_mean))?;
        w.add_f64("beta_perm_sd", &vec_to_array(&result.beta_perm_sd))?;
        w.add_f64("beta_perm_z", &vec_to_array(&result.beta_perm_z))?;
        w.add_i64("n_perm", &scalar_i64(i64::try_from(result.n_perm)?))?;
        w.add_i64("k", &scalar_i64(i64::try_from(result.k)?))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(result.seed)?))?;
        w.add_f64("n_eff", &scalar_f64(result.n_eff))?;
        w.finish()?;
    }

    let mut kwargs = serde_json::json!({
        "n": TALL_N, "d": TALL_D, "k": k, "n_perm": TALL_N_PERM,
        "seed": 42, "disable_parallelism": false
    });
    if weights.is_some() {
        kwargs["weights"] = serde_json::json!("nonuniform");
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs,
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}
