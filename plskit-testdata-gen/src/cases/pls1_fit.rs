//! `pls1_fit` fixture cases.

use crate::cases::{
    default_tolerance, faer_col_to_array, manifest_case, ndarray_to_faer_col, ndarray_to_faer_mat,
    scalar_f64, scalar_i64, synth_data, CasePaths, Xyw,
};
use crate::manifest::{Case, Hashes};
use crate::npz::{sha256_of_file, NpzWriter};
use anyhow::{bail, Result};

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
            force: false,
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

/// Case: small (n=50, d=10), `k_signal=2`, fixed k=2, seed=99, non-uniform weights.
///
/// Weights: `2.0` for observations 0..25, `1.0` for 25..50.
/// Exercises the weighted preprocessing + NIPALS path.
/// Legacy field set: `coef`, `beta`, `intercept`, `k_used`.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_fit` fails.
pub fn weighted_n50_d10_k2(root: &Path) -> Result<Case> {
    let name = "pls1_fit_weighted_n50_d10_k2";
    let function = "pls1_fit";
    let paths = CasePaths::build(root, function, name)?;
    let (x, y) = synth_data(50, 10, 2, 4.0, 99);
    let weights_nd = ndarray::Array1::from_shape_fn(50, |i| if i < 25 { 2.0_f64 } else { 1.0_f64 });
    let weights_faer = ndarray_to_faer_col(&weights_nd);

    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("X", &x.clone().into_dyn())?;
        w.add_f64("y", &y.clone().into_dyn())?;
        w.add_f64("weights", &weights_nd.clone().into_dyn())?;
        w.finish()?;
    }

    let x_faer = ndarray_to_faer_mat(&x);
    let y_faer = ndarray_to_faer_col(&y);
    let model = pls1_fit(
        x_faer.as_ref(),
        y_faer.as_ref(),
        KSpec::Fixed(2),
        Some(weights_faer.as_ref()),
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
        name: name.to_string(),
        function: function.into(),
        inputs: paths.rel_inputs,
        outputs: paths.rel_outputs,
        kwargs: serde_json::json!({"k": 2, "seed": 99, "weights": "nonuniform"}),
        hashes: Hashes {
            inputs_sha256: sha256_of_file(&paths.abs_inputs)?,
            outputs_sha256: sha256_of_file(&paths.abs_outputs)?,
        },
        tolerance: Some(default_tolerance()),
    })
}

/// Weights of [`weighted_n50_d10_k2`] and the cases below: `2.0` for
/// observations 0..25, `1.0` for 25..50.
fn half_double_weights() -> ndarray::Array1<f64> {
    ndarray::Array1::from_shape_fn(50, |i| if i < 25 { 2.0_f64 } else { 1.0_f64 })
}

/// `max_j |mean_j| / sd_j` over the columns of `x`, with the population
/// (ddof 0) sd: the mean-to-scale ratio `pls1_fit` compares with its
/// implicit-route bound (`IMPLICIT_MAX_MEAN_RATIO = 1e3`, `fit.rs`).
#[allow(clippy::cast_precision_loss)]
fn max_mean_ratio(x: &ndarray::Array2<f64>) -> f64 {
    let n = x.nrows() as f64;
    x.columns()
        .into_iter()
        .map(|c| {
            let mean = c.sum() / n;
            let sd = (c.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
            mean.abs() / sd
        })
        .fold(0.0, f64::max)
}

/// Write `data` as the case's own inputs file, fit `pls1_fit` at fixed `k`
/// with its weights and `pre_standardized`, and write the fields of
/// [`fit_fixed_k`].
fn fit_on(
    root: &Path,
    name: &str,
    data: &Xyw,
    k: usize,
    pre_standardized: bool,
    kwargs: serde_json::Value,
) -> Result<Case> {
    let function = "pls1_fit";
    let paths = CasePaths::build(root, function, name)?;
    data.write(&paths.abs_inputs)?;
    let w = data.w_faer();
    let model = pls1_fit(
        data.x_faer().as_ref(),
        data.y_faer().as_ref(),
        KSpec::Fixed(k),
        w.as_ref().map(faer::Col::as_ref),
        FitOpts {
            pre_standardized,
            ..FitOpts::default()
        },
    )?;
    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("coef", &faer_col_to_array(&model.coef))?;
        w.add_f64("beta", &faer_col_to_array(&model.beta))?;
        w.add_f64("intercept", &scalar_f64(model.intercept))?;
        w.add_i64("k_used", &scalar_i64(i64::try_from(model.k_used)?))?;
        w.finish()?;
    }
    manifest_case(
        root,
        name,
        function,
        paths.rel_inputs,
        paths.rel_outputs,
        kwargs,
    )
}

/// Case: pre-standardized, weighted (n=50, d=10), fixed k=3.
///
/// X and y are `synth_data(50, 10, 2, 4.0, 42)` standardized by their
/// unweighted moments; the weights are [`weighted_n50_d10_k2`]'s. With
/// `pre_standardized = true` the fit scales the caller's rows by √w and never
/// recenters: the only fixture of that path.
///
/// # Errors
/// Returns an error if fixture files cannot be written or `pls1_fit` fails.
pub fn pre_standardized_weighted_n50_d10_k3(root: &Path) -> Result<Case> {
    let (x, y) = synth_data(50, 10, 2, 4.0, 42);
    let (xs, _, _) = plskit::linalg::standardize(ndarray_to_faer_mat(&x).as_ref());
    let (ys, _, _) = plskit::linalg::standardize1(ndarray_to_faer_col(&y).as_ref());
    let data = Xyw {
        x: ndarray::Array2::from_shape_fn((xs.nrows(), xs.ncols()), |(i, j)| xs[(i, j)]),
        y: ndarray::Array1::from_shape_fn(ys.nrows(), |i| ys[i]),
        w: Some(half_double_weights()),
    };
    fit_on(
        root,
        "pls1_fit_pre_standardized_weighted_n50_d10_k3",
        &data,
        3,
        true,
        serde_json::json!({"k": 3, "seed": 42, "weights": "nonuniform", "pre_standardized": true}),
    )
}

/// Case: `synth_data(50, 10, 2, 4.0, 42)` with every X entry offset by
/// `+300` and y scaled by `1e-3`, unweighted, fixed k=3.
///
/// The column means sit a few hundred scales from zero but under the
/// implicit-route bound, so the fit forms its standardized products from the
/// raw offset X. The generator refuses the case unless the mean-to-scale
/// ratio lands in (100, 1000), so it cannot silently leave that regime.
///
/// The intercept is `ȳ − Σ x̄_j β_j`: at y's synth scale its terms are
/// about `1e3` and their last-bit differences across hosts would exceed the
/// corpus's absolute scalar tolerance (`1e-12`). Scaling y keeps the terms
/// near one; `coef` is scale-free and `β` scales with y.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_fit` fails, or
/// the ratio leaves (100, 1000).
pub fn offset_n50_d10_k3(root: &Path) -> Result<Case> {
    let (x, y) = synth_data(50, 10, 2, 4.0, 42);
    let (x, y) = (x + 300.0, y * 1e-3);
    let r = max_mean_ratio(&x);
    if !(100.0 < r && r < 1000.0) {
        bail!("pls1_fit_offset_n50_d10_k3: mean/scale ratio {r} is outside (100, 1000)");
    }
    fit_on(
        root,
        "pls1_fit_offset_n50_d10_k3",
        &Xyw { x, y, w: None },
        3,
        false,
        serde_json::json!({"k": 3, "seed": 42}),
    )
}

/// Case: `synth_data(50, 10, 2, 4.0, 99)` with every X entry offset by
/// `+1e6` and y scaled by `1e-6`, [`weighted_n50_d10_k2`]'s weights, fixed k=3.
///
/// Past the implicit-route bound the fit standardizes X into a copy: the only
/// numeric pin of that public route at k > 1. The generator refuses the case
/// unless the mean-to-scale ratio exceeds 1000. y is scaled for the reason
/// given on [`offset_n50_d10_k3`]: unscaled, the intercept is about `-9e6`,
/// whose last bit alone is above the `1e-12` scalar tolerance.
///
/// # Errors
/// Returns an error if fixture files cannot be written, `pls1_fit` fails, or
/// the ratio is at most 1000.
pub fn mean_heavy_weighted_n50_d10_k3(root: &Path) -> Result<Case> {
    let (x, y) = synth_data(50, 10, 2, 4.0, 99);
    let (x, y) = (x + 1e6, y * 1e-6);
    let r = max_mean_ratio(&x);
    if r <= 1000.0 {
        bail!("pls1_fit_mean_heavy_weighted_n50_d10_k3: mean/scale ratio {r} is not above 1000");
    }
    fit_on(
        root,
        "pls1_fit_mean_heavy_weighted_n50_d10_k3",
        &Xyw {
            x,
            y,
            w: Some(half_double_weights()),
        },
        3,
        false,
        serde_json::json!({"k": 3, "seed": 99, "weights": "nonuniform"}),
    )
}
