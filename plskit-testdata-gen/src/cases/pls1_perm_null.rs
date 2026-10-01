//! `pls1_perm_null` fixture cases.

use crate::cases::{
    case_files, default_tolerance, ensure_moved, manifest_case, ndarray_to_faer_col,
    ndarray_to_faer_mat, perm_null_weights, scalar_f64, scalar_i64, synth_data,
    weighted_prestd_n80_d6, CasePaths, Xyw, WEIGHTED_PRESTD_N80_D6_INPUTS,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::Result;
use plskit::{pls1_perm_null, PermNullOpts, PermNullOutput};
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
/// Generated by the explicit-deflation (primal) kernel, so it is an
/// independent reference for the p-space Gram route. Its route and
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

/// One `pls1_perm_null` call on `data` (written to `inputs/{inputs_name}.npz`)
/// at `seed = 42`, serially.
struct PermNullCall<'a> {
    name: &'a str,
    inputs_name: &'a str,
    data: &'a Xyw,
    k: usize,
    n_perm: usize,
    pre_standardized: bool,
    return_perm_matrix: bool,
}

/// The call `c` describes, on `c.data`.
fn perm_null_run(c: &PermNullCall<'_>) -> Result<PermNullOutput> {
    let weights = c.data.w_faer();
    Ok(pls1_perm_null(
        c.data.x_faer().as_ref(),
        c.data.y_faer().as_ref(),
        c.k,
        weights.as_ref().map(faer::Col::as_ref),
        PermNullOpts {
            n_perm: c.n_perm,
            return_perm_matrix: c.return_perm_matrix,
            pre_standardized: c.pre_standardized,
            disable_parallelism: true,
            verbose: false,
        },
        Some(42),
    )?)
}

/// Write `c.data`, run the call, and write the output fields of
/// [`basic_n80_d6_k2`] plus, when `c.return_perm_matrix`, `beta_perm_matrix`
/// as a 2-D `(n_perm, d)` array (the core's flat buffer is row-major, which is
/// also the Python wrapper's `(n_perm, D)` layout). The manifest `kwargs`
/// record the two flags only when they are set.
fn perm_null_call(root: &Path, c: &PermNullCall<'_>) -> Result<Case> {
    let function = "pls1_perm_null";
    let rel_inputs = format!("inputs/{}.npz", c.inputs_name);
    let rel_outputs = format!("outputs/{function}/{}.npz", c.name);
    let (abs_inputs, abs_outputs) = case_files(root, &rel_inputs, &rel_outputs)?;
    c.data.write(&abs_inputs)?;

    let (n, d) = c.data.x.dim();
    let result = perm_null_run(c)?;

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
        if let Some(flat) = result.beta_perm_matrix {
            let matrix = ndarray::Array2::from_shape_vec((result.n_perm, d), flat)?;
            w.add_f64("beta_perm_matrix", &matrix.into_dyn())?;
        }
        w.finish()?;
    }

    let mut kwargs = serde_json::json!({
        "n": n, "d": d, "k": c.k, "n_perm": c.n_perm,
        "seed": 42, "disable_parallelism": true
    });
    if c.data.w.is_some() {
        kwargs["weights"] = serde_json::json!("nonuniform");
    }
    if c.pre_standardized {
        kwargs["pre_standardized"] = serde_json::json!(true);
    }
    if c.return_perm_matrix {
        kwargs["return_perm_matrix"] = serde_json::json!(true);
    }
    manifest_case(root, c.name, function, rel_inputs, rel_outputs, kwargs)
}

/// Case: [`basic_n80_d6_k2`]'s call with `pre_standardized = true` and
/// `return_perm_matrix = true`, on its inputs file (same bytes). The inputs
/// are standard-normal columns, inside the scale contract of the
/// pre-standardized path. The only fixture of that path and of the returned
/// `(200, 6)` permutation matrix's rows.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn pre_standardized_n80_d6_k2(root: &Path) -> Result<Case> {
    let (x, y) = synth_data(80, 6, 2, 4.0, 42);
    perm_null_call(
        root,
        &PermNullCall {
            name: "pls1_perm_null_pre_standardized_n80_d6_k2",
            inputs_name: "pls1_perm_null_basic_n80_d6_k2",
            data: &Xyw { x, y, w: None },
            k: 2,
            n_perm: 200,
            pre_standardized: true,
            return_perm_matrix: true,
        },
    )
}

/// Case: weighted `pls1_perm_null` on a single predictor (`n=40, d=1`) at
/// `k=1`, 200 permutations, weights [`perm_null_weights`]`(40)`. The corpus's
/// only `d = 1` perm-null fixture.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_perm_null` fails.
pub fn weighted_n40_d1_k1(root: &Path) -> Result<Case> {
    // `synth_data` sums the first `k_signal` columns into y: at d = 1 that is 1.
    let (x, y) = synth_data(40, 1, 1, 4.0, 42);
    perm_null_call(
        root,
        &PermNullCall {
            name: "pls1_perm_null_weighted_n40_d1_k1",
            inputs_name: "pls1_perm_null_weighted_n40_d1_k1",
            data: &Xyw {
                x,
                y,
                w: Some(perm_null_weights(40)),
            },
            k: 1,
            n_perm: 200,
            pre_standardized: false,
            return_perm_matrix: false,
        },
    )
}

/// Case: [`weighted_n80_d6_k2`]'s call with `pre_standardized = true` on X
/// and y standardized by their unweighted moments
/// (`cases::weighted_prestd_n80_d6`). Under the flag the weights still drive
/// the reference fit and the √w row scaling of the permutation loop; the
/// only fixture of that combination. The generator refuses it unless
/// `beta_perm_z` moves when the flag is dropped.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_perm_null`
/// fails, or the output does not depend on `pre_standardized`.
pub fn weighted_prestd_n80_d6_k2(root: &Path) -> Result<Case> {
    let data = weighted_prestd_n80_d6();
    let call = PermNullCall {
        name: "pls1_perm_null_weighted_prestd_n80_d6_k2",
        inputs_name: WEIGHTED_PRESTD_N80_D6_INPUTS,
        data: &data,
        k: 2,
        n_perm: 200,
        pre_standardized: true,
        return_perm_matrix: false,
    };
    let with = perm_null_run(&call)?;
    let without = perm_null_run(&PermNullCall {
        pre_standardized: false,
        ..call
    })?;
    ensure_moved(
        call.name,
        "beta_perm_z",
        &with.beta_perm_z,
        &without.beta_perm_z,
    )?;
    perm_null_call(root, &call)
}
