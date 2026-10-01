//! `pls1_rotation_stability` fixture cases.

use crate::cases::{
    case_files, default_tolerance, ensure_moved, manifest_case, ndarray_to_faer_col,
    ndarray_to_faer_mat, scalar_f64, scalar_i64, synth_data, weighted_n80_d6,
    weighted_prestd_n80_d6, CasePaths, Xyw, WEIGHTED_N80_D6_INPUTS, WEIGHTED_PRESTD_N80_D6_INPUTS,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::Result;
use plskit::{
    pls1_rotation_stability, CIScalar, RotationStabilityMethod, RotationStabilityOpts,
    RotationStabilityOutput, VarimaxArgs,
};
use std::path::Path;

/// Write the four scalar fields of a `CIScalar` under a given prefix.
fn write_ci_scalar(w: &mut crate::npz::NpzWriter, prefix: &str, ci: &CIScalar) -> Result<()> {
    w.add_f64(&format!("{prefix}_point"), &scalar_f64(ci.point))?;
    w.add_f64(&format!("{prefix}_lower"), &scalar_f64(ci.lower))?;
    w.add_f64(&format!("{prefix}_upper"), &scalar_f64(ci.upper))?;
    w.add_f64(&format!("{prefix}_sd"), &scalar_f64(ci.sd))?;
    Ok(())
}

/// Convert a `Vec<f64>` to a 1-D `ndarray::ArrayD<f64>`.
fn vec_to_array(v: &[f64]) -> ndarray::ArrayD<f64> {
    ndarray::Array1::from_vec(v.to_vec()).into_dyn()
}

/// Case: `pls1_rotation_stability` on `(n=80, d=6)` data at `k=2`.
///
/// Uses `n_boot=100`, `m_rate=0.7`, `level=0.95`, `seed=42`, and
/// `disable_parallelism=true` for byte-exact determinism.
///
/// Inputs: `X`, `y`.
/// Outputs: scalar fields for `variance_ratio` (point/lower/upper/sd),
///          `variance_unrot`, `variance_rot`, `variance_unrot_per_axis`,
///          `variance_rot_per_axis`, scalars `n_boot`, `m`, `seed`, `m_rate`,
///          `level`, `degenerate_baseline` (0/1 i64), `n_boot_finite`.
///
/// Per-axis CI scalars are flattened into `variance_ratio_per_axis_k{kk}_{field}`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_rotation_stability` fails.
pub fn n80_d6_k2(root: &Path) -> Result<Case> {
    let name = "pls1_rotation_stability_n80_d6_k2";
    let function = "pls1_rotation_stability";
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
    let result = pls1_rotation_stability(
        x_faer.as_ref(),
        y_faer.as_ref(),
        2,
        RotationStabilityMethod::Varimax(VarimaxArgs::default()),
        None,
        None,
        RotationStabilityOpts {
            n_boot: 100,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized: false,
            seed: Some(42),
            disable_parallelism: true,
            verbose: false,
            max_skip_rate: 0.01,
        },
    )?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;

        // Headline variance_ratio CIScalar (flattened).
        write_ci_scalar(&mut w, "variance_ratio", &result.variance_ratio)?;

        // Per-axis CI scalars (flattened with axis index suffix).
        for (kk, ci) in result.variance_ratio_per_axis.iter().enumerate() {
            write_ci_scalar(&mut w, &format!("variance_ratio_per_axis_k{kk}"), ci)?;
        }

        w.add_f64("variance_unrot", &scalar_f64(result.variance_unrot))?;
        w.add_f64("variance_rot", &scalar_f64(result.variance_rot))?;
        w.add_f64(
            "variance_unrot_per_axis",
            &vec_to_array(&result.variance_unrot_per_axis),
        )?;
        w.add_f64(
            "variance_rot_per_axis",
            &vec_to_array(&result.variance_rot_per_axis),
        )?;

        w.add_i64("n_boot", &scalar_i64(i64::try_from(result.n_boot)?))?;
        w.add_i64("m", &scalar_i64(i64::try_from(result.m)?))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(result.seed)?))?;
        w.add_f64("m_rate", &scalar_f64(result.m_rate))?;
        w.add_f64("level", &scalar_f64(result.level))?;
        // Encode bool as 0/1 i64 (NPZ has no bool dtype).
        w.add_i64(
            "degenerate_baseline",
            &scalar_i64(i64::from(result.degenerate_baseline)),
        )?;
        w.add_i64(
            "n_boot_finite",
            &scalar_i64(i64::try_from(result.n_boot_finite)?),
        )?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({
            "n": 80, "d": 6, "k": 2, "n_boot": 100, "m_rate": 0.7,
            "level": 0.95, "seed": 42, "disable_parallelism": true
        }),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// [`n80_d6_k2`]'s call (`k=2`, `n_boot=100`, `m_rate=0.7`, `level=0.95`,
/// `seed=42`, serial) on `data`, with its weights and `pre_standardized`.
fn rotation_stability_call(data: &Xyw, pre_standardized: bool) -> Result<RotationStabilityOutput> {
    let w = data.w_faer();
    Ok(pls1_rotation_stability(
        data.x_faer().as_ref(),
        data.y_faer().as_ref(),
        2,
        RotationStabilityMethod::Varimax(VarimaxArgs::default()),
        None,
        w.as_ref().map(faer::Col::as_ref),
        RotationStabilityOpts {
            n_boot: 100,
            m_rate: 0.7,
            level: 0.95,
            pre_standardized,
            seed: Some(42),
            disable_parallelism: true,
            verbose: false,
            max_skip_rate: 0.01,
        },
    )?)
}

/// Write `data` to `inputs/{inputs_name}.npz`, run
/// [`rotation_stability_call`], apply the generator guard `check`, and write
/// the fields of [`n80_d6_k2`].
fn rotation_stability_on(
    root: &Path,
    name: &str,
    inputs_name: &str,
    data: &Xyw,
    pre_standardized: bool,
    check: impl FnOnce(&RotationStabilityOutput) -> Result<()>,
) -> Result<Case> {
    let function = "pls1_rotation_stability";
    let rel_inputs = format!("inputs/{inputs_name}.npz");
    let rel_outputs = format!("outputs/{function}/{name}.npz");
    let (abs_inputs, abs_outputs) = case_files(root, &rel_inputs, &rel_outputs)?;
    data.write(&abs_inputs)?;

    let result = rotation_stability_call(data, pre_standardized)?;
    check(&result)?;

    {
        let mut w = NpzWriter::create(&abs_outputs)?;
        write_ci_scalar(&mut w, "variance_ratio", &result.variance_ratio)?;
        for (kk, ci) in result.variance_ratio_per_axis.iter().enumerate() {
            write_ci_scalar(&mut w, &format!("variance_ratio_per_axis_k{kk}"), ci)?;
        }
        w.add_f64("variance_unrot", &scalar_f64(result.variance_unrot))?;
        w.add_f64("variance_rot", &scalar_f64(result.variance_rot))?;
        w.add_f64(
            "variance_unrot_per_axis",
            &vec_to_array(&result.variance_unrot_per_axis),
        )?;
        w.add_f64(
            "variance_rot_per_axis",
            &vec_to_array(&result.variance_rot_per_axis),
        )?;
        w.add_i64("n_boot", &scalar_i64(i64::try_from(result.n_boot)?))?;
        w.add_i64("m", &scalar_i64(i64::try_from(result.m)?))?;
        w.add_i64("seed", &scalar_i64(i64::try_from(result.seed)?))?;
        w.add_f64("m_rate", &scalar_f64(result.m_rate))?;
        w.add_f64("level", &scalar_f64(result.level))?;
        w.add_i64(
            "degenerate_baseline",
            &scalar_i64(i64::from(result.degenerate_baseline)),
        )?;
        w.add_i64(
            "n_boot_finite",
            &scalar_i64(i64::try_from(result.n_boot_finite)?),
        )?;
        w.finish()?;
    }

    let (n, d) = data.x.dim();
    let mut kwargs = serde_json::json!({
        "n": n, "d": d, "k": 2, "n_boot": 100, "m_rate": 0.7,
        "level": 0.95, "seed": 42, "disable_parallelism": true,
        "weights": "nonuniform"
    });
    if pre_standardized {
        kwargs["pre_standardized"] = serde_json::json!(true);
    }
    manifest_case(root, name, function, rel_inputs, rel_outputs, kwargs)
}

/// Case: [`n80_d6_k2`]'s call under weights (`cases::weighted_n80_d6`, the
/// inputs file of `pls1_perm_null_weighted_n80_d6_k2`). The only weighted
/// rotation-stability fixture. The generator refuses it unless all 100
/// replicates are finite and the baseline is not degenerate.
///
/// # Errors
/// Returns an error if fixture files cannot be written,
/// `pls1_rotation_stability` fails, or the guard fails.
pub fn weighted_n80_d6_k2(root: &Path) -> Result<Case> {
    let name = "pls1_rotation_stability_weighted_n80_d6_k2";
    rotation_stability_on(
        root,
        name,
        WEIGHTED_N80_D6_INPUTS,
        &weighted_n80_d6(),
        false,
        |r| {
            anyhow::ensure!(
                r.n_boot_finite == 100 && !r.degenerate_baseline,
                "{name}: n_boot_finite = {}, degenerate_baseline = {}",
                r.n_boot_finite,
                r.degenerate_baseline
            );
            Ok(())
        },
    )
}

/// Case: [`weighted_n80_d6_k2`]'s call with `pre_standardized = true` on X
/// and y standardized by their unweighted moments
/// (`cases::weighted_prestd_n80_d6`). The generator refuses it unless the
/// unrotated per-axis variances move when the flag is dropped.
///
/// # Errors
/// Returns an error if fixture files cannot be written,
/// `pls1_rotation_stability` fails, or the output does not depend on
/// `pre_standardized`.
pub fn weighted_prestd_n80_d6_k2(root: &Path) -> Result<Case> {
    let name = "pls1_rotation_stability_weighted_prestd_n80_d6_k2";
    let data = weighted_prestd_n80_d6();
    let without = rotation_stability_call(&data, false)?;
    rotation_stability_on(
        root,
        name,
        WEIGHTED_PRESTD_N80_D6_INPUTS,
        &data,
        true,
        |r| {
            ensure_moved(
                name,
                "variance_unrot_per_axis",
                &r.variance_unrot_per_axis,
                &without.variance_unrot_per_axis,
            )
        },
    )
}
