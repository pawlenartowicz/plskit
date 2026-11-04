//! `factors` fixture cases.

use crate::cases::{
    CasePaths, faer_col_to_array, faer_mat_to_array, ndarray_to_faer_col, ndarray_to_faer_mat,
    scalar_i64, synth_data,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{NpzWriter, sha256_of_file};
use anyhow::Result;
use faer::Mat;
use ndarray::{Array1, Array2};
use plskit::linalg::standardize;
use plskit::{
    FitOpts, KSpec, ProfilesFindNOpts, RotationMethod, VarimaxArgs, factors, pls1_fit,
    profiles_find_n, rotate,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use std::path::Path;

/// Case: a fixed-`k` PLS1 coefficient cut into 3 factors on the PCA basis of
/// standardized `X` (`factors` standardizes it itself), then varimax-rotated on `L = Xs W_f`.
///
/// Inputs: `X`, `y` (consumer repeats the fit, the split and the rotation).
/// Outputs: `F`, `W`, `size` (rotated) and `R` (after sorting and sign pinning).
///
/// # Errors
/// Returns an error if fixture files cannot be written or a plskit call fails.
#[allow(clippy::many_single_char_names)] // x, y, f, l, r are the math names
pub fn pls1_n50_d10_k2_n3(root: &Path) -> Result<Case> {
    let name = "factors_pls1_n50_d10_k2_n3";
    let function = "factors";
    let paths = CasePaths::build(root, function, name)?;

    let (x, y) = synth_data(50, 10, 2, 4.0, 42);
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
        KSpec::Fixed(2),
        None,
        FitOpts::default(),
    )?;
    let (xs, _, _) = standardize(x_faer.as_ref());
    let f = factors(model.coef.as_ref(), x_faer.as_ref(), 3, false)?;
    let l = Mat::<f64>::from_fn(xs.nrows(), f.w.ncols(), |i, j| {
        (0..xs.ncols()).map(|t| xs[(i, t)] * f.w[(t, j)]).sum()
    });
    let varimax = rotate(
        f.w.as_ref(),
        RotationMethod::Varimax(VarimaxArgs::default()),
        Some(l.as_ref()),
    )?;
    let (fr, r) = f.rotated(varimax.r.as_ref())?;

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("F", &faer_mat_to_array(&fr.f))?;
        w.add_f64("W", &faer_mat_to_array(&fr.w))?;
        w.add_f64("size", &faer_col_to_array(&fr.size))?;
        w.add_f64("R", &faer_mat_to_array(&r))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"k": 2, "n_factors": 3}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
    })
}

/// Case: `profiles_find_n` at `seed = 7` with default options, which pins
/// the seeded fold split and column shuffles across the wrappers.
///
/// Inputs: `X`, `y`. Outputs: every field of the result.
///
/// # Errors
/// Returns an error if fixture files cannot be written or a plskit call fails.
pub fn find_n_n100_d12_k2(root: &Path) -> Result<Case> {
    let name = "profiles_find_n_n100_d12_k2";
    let function = "profiles_find_n";
    let paths = CasePaths::build(root, function, name)?;

    // Three factors, four columns each; y loads on the first two. synth_data
    // has no factor structure, so the parallel analysis would count one.
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let t = Array2::<f64>::from_shape_fn((100, 3), |_| rng.sample(StandardNormal));
    let x = Array2::<f64>::from_shape_fn((100, 12), |(i, j)| {
        t[(i, j / 4)] + 0.5 * rng.sample::<f64, _>(StandardNormal)
    });
    let y = Array1::<f64>::from_shape_fn(100, |i| {
        t[(i, 0)] + t[(i, 1)] + rng.sample::<f64, _>(StandardNormal)
    });
    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.finish()?;
    }

    let out = profiles_find_n(
        ndarray_to_faer_mat(&x).as_ref(),
        ndarray_to_faer_col(&y).as_ref(),
        2,
        4,
        ProfilesFindNOpts {
            seed: Some(7),
            ..ProfilesFindNOpts::default()
        },
    )?;
    let ints = |v: &[usize]| -> Result<ndarray::ArrayD<i64>> {
        let v: Vec<i64> = v
            .iter()
            .map(|&i| i64::try_from(i))
            .collect::<Result<_, _>>()?;
        Ok(ndarray::Array1::from(v).into_dyn())
    };

    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_i64("n_star", &scalar_i64(i64::try_from(out.n_star)?))?;
        w.add_i64("n_grid", &ints(&out.n_grid)?)?;
        w.add_f64(
            "r2_mean",
            &ndarray::Array1::from(out.r2_mean.clone()).into_dyn(),
        )?;
        w.add_f64(
            "r2_se",
            &ndarray::Array1::from(out.r2_se.clone()).into_dyn(),
        )?;
        w.add_i64("k_star", &scalar_i64(i64::try_from(out.k_star)?))?;
        w.add_i64(
            "n_failed_folds",
            &scalar_i64(i64::try_from(out.n_failed_folds)?),
        )?;
        w.add_i64("seed", &scalar_i64(i64::try_from(out.seed)?))?;
        w.finish()?;
    }

    Ok(Case {
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"k": 2, "n_max": 4, "seed": 7}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
    })
}
