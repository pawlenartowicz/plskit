//! plskit-py — `PyO3` wrapper for plskit.

// The pyo3::create_exception! macro generates a struct without doc comments;
// the workspace `missing_docs = "warn"` lint can't be applied to macro output.
#![allow(missing_docs)]

use faer::{Col, Mat};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use plskit::error::{PlsKitError, PlsKitResult};
use plskit::preprocess::{preprocess as core_preprocess, PreprocessInput};
use plskit::{
    pls1_confirmatory_test as core_pls1_confirmatory_test,
    pls1_find_k_optimal as core_pls1_find_k_optimal,
    pls1_find_k_sequence as core_pls1_find_k_sequence, pls1_fit as core_pls1_fit,
    pls1_predict as core_pls1_predict, pls1_rotation_stability as core_pls1_rotation_stability,
    rotate as core_rotate, ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestInput,
    ConfirmatoryTestOpts, FindKOptimalOpts, FindKOptimalOutput, FindKSequenceOpts,
    FindKSequenceOutput, FitOpts, KSpec, Pls1Model, RotateOutput, RotationMethod, Selector,
    VarimaxArgs,
};
use plskit::{pls1_perm_null as core_pls1_perm_null, PermNullOpts, PermNullOutput};

// ── numpy ↔ faer bridge ─────────────────────────────────────────────────
// The only place ndarray and faer touch in the repo. Inputs copy +
// transpose row-major numpy → column-major faer in one pass; outputs
// copy back the same way.

#[allow(clippy::needless_pass_by_value)]
fn np_mat_to_faer(arr: PyReadonlyArray2<'_, f64>) -> Mat<f64> {
    let v = arr.as_array();
    let (n, d) = v.dim();
    Mat::<f64>::from_fn(n, d, |i, j| v[(i, j)])
}

#[allow(clippy::needless_pass_by_value)]
fn np_col_to_faer(arr: PyReadonlyArray1<'_, f64>) -> Col<f64> {
    let v = arr.as_array();
    Col::<f64>::from_fn(v.len(), |i| v[i])
}

#[allow(clippy::needless_pass_by_value)]
fn faer_mat_to_np(py: Python<'_>, m: Mat<f64>) -> Bound<'_, PyArray2<f64>> {
    let arr = Array2::<f64>::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]);
    arr.into_pyarray(py)
}

#[allow(clippy::needless_pass_by_value)]
fn faer_col_to_np(py: Python<'_>, v: Col<f64>) -> Bound<'_, PyArray1<f64>> {
    let arr = Array1::<f64>::from_shape_fn(v.nrows(), |i| v[i]);
    arr.into_pyarray(py)
}

pyo3::create_exception!(_plskit, PlsKitException, PyException);

/// Variant-specific fields extracted from a `PlsKitError` before it is moved
/// into the `PyErr` message string, so they can be attached as Python attributes.
enum ErrorExtra {
    None,
    InvalidWeights {
        reason: &'static str,
    },
    ResamplingDegenerate {
        skipped: usize,
        total: usize,
        skip_rate: f64,
        threshold: f64,
    },
}

fn map_res<T>(r: PlsKitResult<T>) -> PyResult<T> {
    r.map_err(|e| {
        let code = e.code();
        let msg = format!("{e}");
        // Capture variant-specific fields before moving `e`.
        let extra = match e {
            PlsKitError::InvalidWeights { reason } => ErrorExtra::InvalidWeights { reason },
            PlsKitError::ResamplingDegenerate {
                skipped,
                total,
                skip_rate,
                threshold,
            } => ErrorExtra::ResamplingDegenerate {
                skipped,
                total,
                skip_rate,
                threshold,
            },
            _ => ErrorExtra::None,
        };
        let py_err = PlsKitException::new_err(msg);
        Python::attach(|py| {
            let _ = py_err.value(py).setattr("code", code);
            match extra {
                ErrorExtra::InvalidWeights { reason } => {
                    let _ = py_err.value(py).setattr("reason", reason);
                }
                ErrorExtra::ResamplingDegenerate {
                    skipped,
                    total,
                    skip_rate,
                    threshold,
                } => {
                    let _ = py_err.value(py).setattr("skipped", skipped);
                    let _ = py_err.value(py).setattr("total", total);
                    let _ = py_err.value(py).setattr("skip_rate", skip_rate);
                    let _ = py_err.value(py).setattr("threshold", threshold);
                }
                ErrorExtra::None => {}
            }
        });
        py_err
    })
}

fn pls1_model_to_dict(py: Python<'_>, m: Pls1Model) -> Bound<'_, PyDict> {
    // Dict keys are the short Python-facing names;
    // Rust struct fields use long snake_case.
    let d = PyDict::new(py);
    d.set_item("T", faer_mat_to_np(py, m.t_scores)).unwrap();
    d.set_item("P", faer_mat_to_np(py, m.p_loadings)).unwrap();
    d.set_item("W", faer_mat_to_np(py, m.w_star)).unwrap();
    d.set_item("Q", faer_col_to_np(py, m.q_loadings)).unwrap();
    d.set_item("coef", faer_col_to_np(py, m.coef)).unwrap();
    d.set_item("beta", faer_col_to_np(py, m.beta)).unwrap();
    d.set_item("intercept", m.intercept).unwrap();
    d.set_item("k_used", m.k_used).unwrap();
    d.set_item("pre_standardized", m.pre_standardized).unwrap();
    d.set_item("n_eff", m.n_eff).unwrap();
    d.set_item("weights", m.weights.map(|c| faer_col_to_np(py, c)))
        .unwrap();
    d
}

fn ciscalar_to_dict(py: Python<'_>, ci: plskit::CIScalar) -> Bound<'_, PyDict> {
    let d = PyDict::new(py);
    d.set_item("point", ci.point).unwrap();
    d.set_item("lower", ci.lower).unwrap();
    d.set_item("upper", ci.upper).unwrap();
    d.set_item("sd", ci.sd).unwrap();
    d
}

fn confirmatory_ci_to_dict(py: Python<'_>, ci: plskit::ConfirmatoryCI) -> Bound<'_, PyDict> {
    let d = PyDict::new(py);
    d.set_item("n_boot", ci.n_boot).unwrap();
    d.set_item("m", ci.m).unwrap();
    d.set_item("m_rate", ci.m_rate).unwrap();
    d.set_item("level", ci.level).unwrap();

    // Per-variable arrays → np.ndarray
    let bsz = Array1::<f64>::from_vec(ci.beta_sign_z);
    d.set_item("beta_sign_z", bsz.into_pyarray(py)).unwrap();
    let bszs = Array1::<f64>::from_vec(ci.beta_sign_z_signed);
    d.set_item("beta_sign_z_signed", bszs.into_pyarray(py))
        .unwrap();
    let llo = Array1::<f64>::from_vec(ci.leverage_ci_lower);
    d.set_item("leverage_ci_lower", llo.into_pyarray(py))
        .unwrap();
    let lhi = Array1::<f64>::from_vec(ci.leverage_ci_upper);
    d.set_item("leverage_ci_upper", lhi.into_pyarray(py))
        .unwrap();
    let lse = Array1::<f64>::from_vec(ci.leverage_se);
    d.set_item("leverage_se", lse.into_pyarray(py)).unwrap();
    let blo = Array1::<f64>::from_vec(ci.beta_ci_lower);
    d.set_item("beta_ci_lower", blo.into_pyarray(py)).unwrap();
    let bhi = Array1::<f64>::from_vec(ci.beta_ci_upper);
    d.set_item("beta_ci_upper", bhi.into_pyarray(py)).unwrap();
    let bse = Array1::<f64>::from_vec(ci.beta_se);
    d.set_item("beta_se", bse.into_pyarray(py)).unwrap();

    d.set_item("holdout_corr", ciscalar_to_dict(py, ci.holdout_corr))
        .unwrap();
    d.set_item("n_boot_finite", ci.n_boot_finite).unwrap();
    d.set_item("n_boot_finite_holdout_corr", ci.n_boot_finite_holdout_corr)
        .unwrap();
    d
}

fn rotation_stability_to_dict(
    py: Python<'_>,
    out: plskit::RotationStabilityOutput,
) -> Bound<'_, PyDict> {
    let d = PyDict::new(py);
    d.set_item("method", out.method).unwrap();
    d.set_item("n_boot", out.n_boot).unwrap();
    d.set_item("m", out.m).unwrap();
    d.set_item("m_rate", out.m_rate).unwrap();
    d.set_item("level", out.level).unwrap();
    d.set_item("seed", out.seed).unwrap();

    d.set_item("variance_ratio", ciscalar_to_dict(py, out.variance_ratio))
        .unwrap();
    let per_axis: Vec<Bound<'_, PyDict>> = out
        .variance_ratio_per_axis
        .into_iter()
        .map(|ci| ciscalar_to_dict(py, ci))
        .collect();
    d.set_item("variance_ratio_per_axis", per_axis).unwrap();

    d.set_item("variance_unrot", out.variance_unrot).unwrap();
    d.set_item("variance_rot", out.variance_rot).unwrap();
    let v_unrot_per_axis = Array1::<f64>::from_vec(out.variance_unrot_per_axis);
    d.set_item("variance_unrot_per_axis", v_unrot_per_axis.into_pyarray(py))
        .unwrap();
    let v_rot_per_axis = Array1::<f64>::from_vec(out.variance_rot_per_axis);
    d.set_item("variance_rot_per_axis", v_rot_per_axis.into_pyarray(py))
        .unwrap();

    d.set_item("degenerate_baseline", out.degenerate_baseline)
        .unwrap();
    d.set_item("n_boot_finite", out.n_boot_finite).unwrap();
    d.set_item("n_eff", out.n_eff).unwrap();
    d
}

// ── args-dict parsers ──────────────────────────────────────────────────
//
// `method` selects the variant; `args` (optional dict) carries
// method-specific kwargs. Unknown keys raise.

/// Build a `PlsKitException` with `.code = "invalid_args"` already
/// set. Used wherever args-dict validation fails (unknown key,
/// wrong-typed value, etc.).
fn invalid_args_err(msg: &str) -> PyErr {
    let py_err = PlsKitException::new_err(msg.to_owned());
    Python::attach(|py| {
        let _ = py_err.value(py).setattr("code", "invalid_args");
    });
    py_err
}

fn validate_keys(method: &str, args: &Bound<'_, PyDict>, allowed: &[&str]) -> PyResult<()> {
    for (k, _) in args.iter() {
        let ks: String = k.extract()?;
        if !allowed.iter().any(|a| *a == ks) {
            return Err(invalid_args_err(&format!(
                "method='{method}' does not accept arg '{ks}'; allowed: {allowed:?}"
            )));
        }
    }
    Ok(())
}

fn btreemap_to_dict<'py>(
    py: Python<'py>,
    m: &std::collections::BTreeMap<usize, f64>,
) -> Bound<'py, PyDict> {
    let d = PyDict::new(py);
    for (k, v) in m {
        d.set_item(*k, *v).expect("dict insert");
    }
    d
}

fn parse_confirmatory_method(s: &str) -> PyResult<ConfirmatoryMethod> {
    match s {
        "raw_perm" => Ok(ConfirmatoryMethod::RawPerm),
        "split_nb" => Ok(ConfirmatoryMethod::SplitNb),
        "split_perm" => Ok(ConfirmatoryMethod::SplitPerm),
        "score" => Ok(ConfirmatoryMethod::Score),
        "e" => Ok(ConfirmatoryMethod::E),
        _ => Err(PlsKitException::new_err(format!("unknown method: {s}"))),
    }
}

fn parse_confirmatory_args(
    method: &str,
    args: Option<&Bound<'_, PyDict>>,
) -> PyResult<ConfirmatoryArgs> {
    match method {
        "raw_perm" => {
            let allowed: &[&str] = &["n_perm", "n_folds"];
            if let Some(a) = args {
                validate_keys("raw_perm", a, allowed)?;
            }
            let n_perm = match args.and_then(|a| a.get_item("n_perm").ok().flatten()) {
                Some(v) => v.extract::<usize>()?,
                None => 1000,
            };
            let n_folds = match args.and_then(|a| a.get_item("n_folds").ok().flatten()) {
                Some(v) => v.extract::<usize>()?,
                None => 5,
            };
            Ok(ConfirmatoryArgs::RawPerm { n_perm, n_folds })
        }
        "split_nb" => {
            let allowed: &[&str] = &["n_splits"];
            if let Some(a) = args {
                validate_keys("split_nb", a, allowed)?;
            }
            let n_splits = match args.and_then(|a| a.get_item("n_splits").ok().flatten()) {
                Some(v) => v.extract::<usize>()?,
                None => 50,
            };
            Ok(ConfirmatoryArgs::SplitNb { n_splits })
        }
        "split_perm" => {
            let allowed: &[&str] = &["n_perm", "n_splits"];
            if let Some(a) = args {
                validate_keys("split_perm", a, allowed)?;
            }
            let n_perm = match args.and_then(|a| a.get_item("n_perm").ok().flatten()) {
                Some(v) => v.extract::<usize>()?,
                None => 1000,
            };
            let n_splits = match args.and_then(|a| a.get_item("n_splits").ok().flatten()) {
                Some(v) => v.extract::<usize>()?,
                None => 50,
            };
            Ok(ConfirmatoryArgs::SplitPerm { n_perm, n_splits })
        }
        "score" => {
            let allowed: &[&str] = &[];
            if let Some(a) = args {
                validate_keys("score", a, allowed)?;
            }
            Ok(ConfirmatoryArgs::Score)
        }
        "e" => {
            let allowed: &[&str] = &[];
            if let Some(a) = args {
                validate_keys("e", a, allowed)?;
            }
            Ok(ConfirmatoryArgs::E)
        }
        _ => Err(PlsKitException::new_err(format!(
            "unknown method: {method}"
        ))),
    }
}

fn parse_optimal_selector(s: &str) -> PyResult<Selector> {
    match s {
        "r2_se" => Ok(Selector::R2Se),
        "r2_max" => Ok(Selector::R2Max),
        "bic" => Ok(Selector::Bic),
        _ => Err(PlsKitException::new_err(format!("unknown selector: {s}"))),
    }
}

fn parse_rotation_method(
    method: &str,
    args: Option<&Bound<'_, PyDict>>,
) -> PyResult<RotationMethod> {
    match method {
        "varimax" => {
            let allowed: &[&str] = &["max_iter", "tol", "kaiser_normalize"];
            if let Some(a) = args {
                validate_keys("varimax", a, allowed)?;
            }
            let defaults = VarimaxArgs::default();
            let max_iter = match args.and_then(|a| a.get_item("max_iter").ok().flatten()) {
                Some(v) => v.extract::<usize>().map_err(|_| {
                    invalid_args_err(
                        "args['max_iter'] for method='varimax' must be a non-negative int",
                    )
                })?,
                None => defaults.max_iter,
            };
            let tol = match args.and_then(|a| a.get_item("tol").ok().flatten()) {
                Some(v) => v.extract::<f64>().map_err(|_| {
                    invalid_args_err("args['tol'] for method='varimax' must be a float")
                })?,
                None => defaults.tol,
            };
            let kaiser_normalize = match args
                .and_then(|a| a.get_item("kaiser_normalize").ok().flatten())
            {
                Some(v) => v.extract::<bool>().map_err(|_| {
                    invalid_args_err("args['kaiser_normalize'] for method='varimax' must be a bool")
                })?,
                None => defaults.kaiser_normalize,
            };
            Ok(RotationMethod::Varimax(VarimaxArgs {
                max_iter,
                tol,
                kaiser_normalize,
            }))
        }
        other => {
            // Stamp the typed code manually since this branch bypasses
            // `map_res` (which is the usual path that sets `code`).
            // PyO3 0.26: use `Python::attach` and `.value(py)`.
            let py_err = PlsKitException::new_err(format!(
                "rotation method '{other}' is not implemented in this version"
            ));
            Python::attach(|py| {
                let _ = py_err
                    .value(py)
                    .setattr("code", "rotation_method_not_implemented");
            });
            Err(py_err)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, k, *, pre_standardized=false, tol=1e-9, max_iter=500, weights=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
fn pls1_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k: usize,
    pre_standardized: bool,
    tol: f64,
    max_iter: usize,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let kspec = KSpec::Fixed(k);
    let opts = FitOpts {
        pre_standardized,
        tol,
        max_iter,
        check_n_eff: true,
    };
    // Bridge numpy → faer at the entry seam.
    let xf = np_mat_to_faer(x);
    let yf = np_col_to_faer(y);
    let wf = weights.map(np_col_to_faer);
    let wref = wf.as_ref().map(Col::as_ref);
    let m = map_res(core_pls1_fit(xf.as_ref(), yf.as_ref(), kspec, wref, opts))?;
    Ok(pls1_model_to_dict(py, m))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn pls1_predict<'py>(
    py: Python<'py>,
    model: Bound<'_, PyDict>,
    x_new: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Reconstruct a Pls1Model from the dict's fields — bridge ndarray view → faer Mat.
    let m = pls1_model_from_dict(py, &model)?;
    let x_faer = np_mat_to_faer(x_new);
    let yhat = map_res(core_pls1_predict(&m, x_faer.as_ref()))?;
    Ok(faer_col_to_np(py, yhat))
}

fn pls1_model_from_dict(py: Python<'_>, d: &Bound<'_, PyDict>) -> PyResult<Pls1Model> {
    let get_mat = |k: &str| -> PyResult<Mat<f64>> {
        let v: PyReadonlyArray2<'_, f64> = d
            .get_item(k)?
            .ok_or_else(|| PlsKitException::new_err(format!("missing field {k}")))?
            .extract()?;
        Ok(np_mat_to_faer(v))
    };
    let get_col = |k: &str| -> PyResult<Col<f64>> {
        let v: PyReadonlyArray1<'_, f64> = d
            .get_item(k)?
            .ok_or_else(|| PlsKitException::new_err(format!("missing field {k}")))?
            .extract()?;
        Ok(np_col_to_faer(v))
    };
    let intercept: f64 = d.get_item("intercept")?.unwrap().extract()?;
    let k_used: usize = d.get_item("k_used")?.unwrap().extract()?;
    let pre_standardized: bool = d
        .get_item("pre_standardized")?
        .ok_or_else(|| PlsKitException::new_err("missing field pre_standardized"))?
        .extract()?;
    let t_scores = get_mat("T")?;
    let n_samples = t_scores.nrows();
    // "n_eff" — present in dicts written by Task 8+; default to n_samples for
    // back-compat with old dicts that lack the key.
    #[allow(clippy::cast_precision_loss)]
    let n_eff: f64 = match d.get_item("n_eff")? {
        Some(v) => v.extract()?,
        None => n_samples as f64,
    };
    // "weights" — present in dicts written by Task 8+; default to None.
    let weights: Option<Col<f64>> = match d.get_item("weights")? {
        Some(v) if !v.is_none() => {
            let arr: PyReadonlyArray1<'_, f64> = v.extract()?;
            Some(np_col_to_faer(arr))
        }
        _ => None,
    };
    let _ = py;
    Ok(Pls1Model {
        // Dict keys are short (Python-facing); Rust fields are long snake_case.
        t_scores,
        p_loadings: get_mat("P")?,
        w_star: get_mat("W")?,
        q_loadings: get_col("Q")?,
        coef: get_col("coef")?,
        beta: get_col("beta")?,
        intercept,
        k_used,
        pre_standardized,
        weights,
        n_eff,
    })
}

#[pyfunction]
#[pyo3(signature = (w, *, method, args=None, l=None))]
#[allow(clippy::needless_pass_by_value)]
fn rotate<'py>(
    py: Python<'py>,
    w: PyReadonlyArray2<'_, f64>,
    method: &str,
    args: Option<Bound<'_, PyDict>>,
    l: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let rot_method = parse_rotation_method(method, args.as_ref())?;
    let wf = np_mat_to_faer(w);
    let lf_storage = l.map(np_mat_to_faer);
    let lf_ref = lf_storage.as_ref().map(faer::Mat::as_ref);
    let out: RotateOutput = map_res(core_rotate(wf.as_ref(), rot_method, lf_ref))?;
    // PyO3 0.26: `PyDict::new` (was `new_bound` in 0.22).
    let d = PyDict::new(py);
    d.set_item("w_rot", faer_mat_to_np(py, out.w_rot))?;
    d.set_item("r", faer_mat_to_np(py, out.r))?;
    d.set_item("sweeps", out.sweeps)?;
    d.set_item("v_converged", out.v_converged)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (x, y, k, *, method, args=None,
                    ci=false, n_boot=1000, m_rate=0.7, level=0.95,
                    max_failure_rate=0.01,
                    pre_standardized=false, seed=None,
                    disable_parallelism=false, verbose=false,
                    weights=None, max_skip_rate=0.01))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::many_single_char_names)]
fn pls1_confirmatory_test_raw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k: usize,
    method: &str,
    args: Option<Bound<'_, PyDict>>,
    ci: bool,
    n_boot: usize,
    m_rate: f64,
    level: f64,
    max_failure_rate: f64,
    pre_standardized: bool,
    seed: Option<u64>,
    disable_parallelism: bool,
    verbose: bool,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    max_skip_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let ci_opts = if ci {
        Some(plskit::CIOpts {
            n_boot,
            m_rate,
            level,
            max_failure_rate,
        })
    } else {
        None
    };
    let opts = ConfirmatoryTestOpts {
        args: parse_confirmatory_args(method, args.as_ref())?,
        pre_standardized,
        seed,
        disable_parallelism,
        verbose,
        ci: ci_opts,
        max_skip_rate,
    };
    let xf = np_mat_to_faer(x);
    let yf = np_col_to_faer(y);
    let wf = weights.map(np_col_to_faer);
    let wref = wf.as_ref().map(Col::as_ref);
    let r = map_res(core_pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: xf.as_ref(),
            y: yf.as_ref(),
            k,
            weights: wref,
        },
        opts,
    ))?;
    let d = PyDict::new(py);
    d.set_item("pvalue", r.pvalue)?;
    d.set_item("statistic", r.statistic)?;
    d.set_item("method", r.method.clone())?;
    d.set_item("k", r.k)?;
    d.set_item("n_perm", r.n_perm)?;
    d.set_item("n_splits", r.n_splits)?;
    d.set_item("seed", r.seed)?;
    d.set_item("n_eff", r.n_eff)?;
    let ci_py: Option<Bound<'_, PyDict>> = r.ci.map(|c| confirmatory_ci_to_dict(py, c));
    d.set_item("ci", ci_py)?;
    Ok(d)
}

fn parse_varimax_args(args: Option<&Bound<'_, PyDict>>) -> PyResult<plskit::VarimaxArgs> {
    use plskit::VarimaxArgs;
    let allowed: &[&str] = &["max_iter", "tol", "kaiser_normalize"];
    if let Some(a) = args {
        validate_keys("varimax", a, allowed)?;
    }
    let max_iter = match args.and_then(|a| a.get_item("max_iter").ok().flatten()) {
        Some(v) => v.extract::<usize>()?,
        None => 50,
    };
    let tol = match args.and_then(|a| a.get_item("tol").ok().flatten()) {
        Some(v) => v.extract::<f64>()?,
        None => 1e-8,
    };
    let kaiser_normalize = match args.and_then(|a| a.get_item("kaiser_normalize").ok().flatten()) {
        Some(v) => v.extract::<bool>()?,
        None => true,
    };
    Ok(VarimaxArgs {
        max_iter,
        tol,
        kaiser_normalize,
    })
}

#[pyfunction]
#[pyo3(signature = (
    x, y, k, *,
    rotation_method = "varimax",
    rotation_args = None,
    l = None,
    n_boot = 1000, m_rate = 0.7, level = 0.95,
    pre_standardized = false,
    seed = None,
    disable_parallelism = false,
    verbose = false,
    weights = None,
    max_skip_rate = 0.01,
))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
fn pls1_rotation_stability_raw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k: usize,
    rotation_method: &str,
    rotation_args: Option<&Bound<'_, PyDict>>,
    l: Option<PyReadonlyArray2<'_, f64>>,
    n_boot: usize,
    m_rate: f64,
    level: f64,
    pre_standardized: bool,
    seed: Option<u64>,
    disable_parallelism: bool,
    verbose: bool,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    max_skip_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let xm = np_mat_to_faer(x);
    let yc = np_col_to_faer(y);

    let method = match rotation_method {
        "varimax" => {
            let args = parse_varimax_args(rotation_args)?;
            plskit::RotationStabilityMethod::Varimax(args)
        }
        other => {
            // Stamp the typed code manually since this branch bypasses
            // `map_res` (which is the usual path that sets `code`).
            let py_err = PlsKitException::new_err(format!(
                "rotation_method '{other}' not implemented in v0.x"
            ));
            Python::attach(|py| {
                let _ = py_err
                    .value(py)
                    .setattr("code", "rotation_method_not_implemented");
            });
            return Err(py_err);
        }
    };

    let l_mat: Option<Mat<f64>> = l.map(np_mat_to_faer);
    let wf = weights.map(np_col_to_faer);
    let wref = wf.as_ref().map(Col::as_ref);

    let opts = plskit::RotationStabilityOpts {
        n_boot,
        m_rate,
        level,
        pre_standardized,
        seed,
        disable_parallelism,
        verbose,
        max_skip_rate,
    };

    let out = map_res(core_pls1_rotation_stability(
        xm.as_ref(),
        yc.as_ref(),
        k,
        method,
        l_mat.as_ref().map(Mat::as_ref),
        wref,
        opts,
    ))?;

    Ok(rotation_stability_to_dict(py, out))
}

#[pyfunction]
#[pyo3(signature = (x, y, k_max, *, selector="r2_se", diagnostic=None,
                    args=None,
                    pre_standardized=false, seed=None,
                    disable_parallelism=false, verbose=false,
                    weights=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::fn_params_excessive_bools)]
fn pls1_find_k_optimal<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k_max: usize,
    selector: &str,
    diagnostic: Option<&str>,
    args: Option<Bound<'_, PyDict>>,
    pre_standardized: bool,
    seed: Option<u64>,
    disable_parallelism: bool,
    verbose: bool,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let sel = parse_optimal_selector(selector)?;
    let diag_method: Option<ConfirmatoryMethod> = match diagnostic {
        Some(s) => Some(parse_confirmatory_method(s)?),
        None => None,
    };
    let allowed: &[&str] = &["n_folds", "n_perm", "n_splits"];
    if let Some(a) = args.as_ref() {
        validate_keys("optimal", a, allowed)?;
    }
    let n_folds = match args
        .as_ref()
        .and_then(|a| a.get_item("n_folds").ok().flatten())
    {
        Some(v) => v.extract::<usize>()?,
        None => 5,
    };
    // Reject n_folds with bic.
    if matches!(sel, Selector::Bic)
        && args
            .as_ref()
            .and_then(|a| a.get_item("n_folds").ok().flatten())
            .is_some()
    {
        return Err(invalid_args_err(
            "selector='bic' does not accept arg 'n_folds'",
        ));
    }
    let n_perm = match args
        .as_ref()
        .and_then(|a| a.get_item("n_perm").ok().flatten())
    {
        Some(v) => {
            let Some(dm) = diag_method else {
                return Err(invalid_args_err(
                    "args['n_perm'] requires diagnostic to be set",
                ));
            };
            if !matches!(
                dm,
                ConfirmatoryMethod::RawPerm | ConfirmatoryMethod::SplitPerm
            ) {
                return Err(invalid_args_err(
                    "args['n_perm'] only valid for diagnostic in {raw_perm, split_perm}",
                ));
            }
            v.extract::<usize>()?
        }
        None => 1000,
    };
    let n_splits = match args
        .as_ref()
        .and_then(|a| a.get_item("n_splits").ok().flatten())
    {
        Some(v) => {
            let Some(dm) = diag_method else {
                return Err(invalid_args_err(
                    "args['n_splits'] requires diagnostic to be set",
                ));
            };
            if !matches!(
                dm,
                ConfirmatoryMethod::SplitNb | ConfirmatoryMethod::SplitPerm
            ) {
                return Err(invalid_args_err(
                    "args['n_splits'] only valid for diagnostic in {split_nb, split_perm}",
                ));
            }
            v.extract::<usize>()?
        }
        None => 50,
    };
    let opts = FindKOptimalOpts {
        selector: sel,
        n_folds,
        diagnostic: diag_method,
        n_perm,
        n_splits,
        pre_standardized,
        seed,
        disable_parallelism,
        verbose,
    };
    let xf = np_mat_to_faer(x);
    let yf = np_col_to_faer(y);
    let wf = weights.map(np_col_to_faer);
    let wref = wf.as_ref().map(Col::as_ref);
    let r: FindKOptimalOutput = map_res(core_pls1_find_k_optimal(
        xf.as_ref(),
        yf.as_ref(),
        k_max,
        wref,
        opts,
    ))?;
    let d = PyDict::new(py);
    d.set_item("k_star", r.k_star)?;
    d.set_item("selector", r.selector)?;
    d.set_item(
        "cv_scores",
        r.cv_scores.as_ref().map(|m| btreemap_to_dict(py, m)),
    )?;
    d.set_item(
        "cv_scores_se",
        r.cv_scores_se.as_ref().map(|m| btreemap_to_dict(py, m)),
    )?;
    d.set_item(
        "bic_scores",
        r.bic_scores.as_ref().map(|m| btreemap_to_dict(py, m)),
    )?;
    d.set_item("pvalues", r.pvalues.map(|c| faer_col_to_np(py, c)))?;
    d.set_item("diagnostic", r.diagnostic)?;
    d.set_item("seed", r.seed)?;
    d.set_item("n_eff", r.n_eff)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (x, y, k_max, *, test_method="split_nb", alpha=0.05,
                    args=None,
                    pre_standardized=false, seed=None,
                    disable_parallelism=false, verbose=false,
                    weights=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::fn_params_excessive_bools)]
fn pls1_find_k_sequence<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k_max: usize,
    test_method: &str,
    alpha: f64,
    args: Option<Bound<'_, PyDict>>,
    pre_standardized: bool,
    seed: Option<u64>,
    disable_parallelism: bool,
    verbose: bool,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let tm = parse_confirmatory_method(test_method)?;
    let allowed: &[&str] = match tm {
        ConfirmatoryMethod::RawPerm => &["n_perm"],
        ConfirmatoryMethod::SplitNb => &["n_splits"],
        ConfirmatoryMethod::SplitPerm => &["n_perm", "n_splits"],
        ConfirmatoryMethod::E => &[],
        ConfirmatoryMethod::Score => {
            return Err(invalid_args_err(
                "test_method='score' has no sequential variant",
            ));
        }
    };
    if let Some(a) = args.as_ref() {
        validate_keys("sequence", a, allowed)?;
    }
    let n_perm = match args
        .as_ref()
        .and_then(|a| a.get_item("n_perm").ok().flatten())
    {
        Some(v) => v.extract::<usize>()?,
        None => 1000,
    };
    let n_splits = match args
        .as_ref()
        .and_then(|a| a.get_item("n_splits").ok().flatten())
    {
        Some(v) => v.extract::<usize>()?,
        None => 50,
    };
    let opts = FindKSequenceOpts {
        test_method: tm,
        alpha,
        n_perm,
        n_splits,
        pre_standardized,
        seed,
        disable_parallelism,
        verbose,
    };
    let xf = np_mat_to_faer(x);
    let yf = np_col_to_faer(y);
    let wf = weights.map(np_col_to_faer);
    let wref = wf.as_ref().map(Col::as_ref);
    let r: FindKSequenceOutput = map_res(core_pls1_find_k_sequence(
        xf.as_ref(),
        yf.as_ref(),
        k_max,
        wref,
        opts,
    ))?;
    let d = PyDict::new(py);
    d.set_item("k_star", r.k_star)?;
    d.set_item("pvalues", faer_col_to_np(py, r.pvalues))?;
    d.set_item("test_method", r.test_method)?;
    d.set_item("alpha", r.alpha)?;
    d.set_item("seed", r.seed)?;
    d.set_item("n_eff", r.n_eff)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (x, y, k, *, n_perm=1000, return_perm_matrix=false,
                    pre_standardized=false, seed=None,
                    disable_parallelism=false, verbose=false,
                    weights=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::many_single_char_names)]
fn pls1_perm_null_raw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k: usize,
    n_perm: usize,
    return_perm_matrix: bool,
    pre_standardized: bool,
    seed: Option<u64>,
    disable_parallelism: bool,
    verbose: bool,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = PermNullOpts {
        n_perm,
        return_perm_matrix,
        pre_standardized,
        disable_parallelism,
        verbose,
    };
    let xf = np_mat_to_faer(x);
    let yf = np_col_to_faer(y);
    let wf = weights.map(np_col_to_faer);
    let wref = wf.as_ref().map(Col::as_ref);
    let out: PermNullOutput = map_res(core_pls1_perm_null(
        xf.as_ref(),
        yf.as_ref(),
        k,
        wref,
        opts,
        seed,
    ))?;

    let d = PyDict::new(py);
    d.set_item("n_perm", out.n_perm)?;
    d.set_item("k", out.k)?;
    d.set_item("seed", out.seed)?;
    d.set_item("n_eff", out.n_eff)?;
    d.set_item(
        "beta_ref",
        Array1::<f64>::from_vec(out.beta_ref).into_pyarray(py),
    )?;
    d.set_item(
        "beta_perm_mean",
        Array1::<f64>::from_vec(out.beta_perm_mean).into_pyarray(py),
    )?;
    d.set_item(
        "beta_perm_sd",
        Array1::<f64>::from_vec(out.beta_perm_sd).into_pyarray(py),
    )?;
    d.set_item(
        "beta_perm_z",
        Array1::<f64>::from_vec(out.beta_perm_z).into_pyarray(py),
    )?;
    let mat_py = out.beta_perm_matrix.map(|flat| {
        let n = out.n_perm;
        let dd = flat.len() / n.max(1);
        Array2::<f64>::from_shape_vec((n, dd), flat)
            .expect("perm matrix shape invariant: len == n_perm * d")
            .into_pyarray(py)
    });
    d.set_item("beta_perm_matrix", mat_py)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (x=None, y=None, weights=None))]
fn preprocess<'py>(
    py: Python<'py>,
    x: Option<PyReadonlyArray2<'_, f64>>,
    y: Option<Bound<'_, PyAny>>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    // Bridge X (2-D, optional).
    let xf = x.map(np_mat_to_faer);
    let xref = xf.as_ref().map(faer::Mat::as_ref);

    // Y is shape-polymorphic (spec §5.1). Detect 1-D vs 2-D up front.
    let mut y_was_2d = false;
    let yf_1d: Option<faer::Col<f64>>;
    let yf_2d: Option<faer::Mat<f64>>;
    if let Some(yobj) = y {
        if let Ok(arr1) = yobj.extract::<PyReadonlyArray1<'_, f64>>() {
            yf_1d = Some(np_col_to_faer(arr1));
            yf_2d = None;
        } else {
            let arr2: PyReadonlyArray2<'_, f64> = yobj.extract()?;
            yf_2d = Some(np_mat_to_faer(arr2));
            yf_1d = None;
            y_was_2d = true;
        }
    } else {
        yf_1d = None;
        yf_2d = None;
    }
    let y_ref_1d = yf_1d.as_ref().map(faer::Col::as_ref);

    // Bridge weights.
    let wf = weights.map(np_col_to_faer);
    let wref = wf.as_ref().map(faer::Col::as_ref);

    let d = PyDict::new(py);
    if y_was_2d {
        // Run core preprocess WITHOUT y so it normalizes weights and standardizes X.
        let core_in = PreprocessInput {
            x: xref,
            y: None,
            weights: wref,
        };
        let core_r = map_res(core_preprocess(core_in))?;
        // Apply the SAME normalized weights to standardize the 2-D Y.
        let yref2d = yf_2d.as_ref().unwrap().as_ref();
        let wn_ref = core_r.weights_normalized.as_ref().map(faer::Col::as_ref);
        let (y_std, y_mean, y_scale) = plskit::linalg::standardize_weighted(yref2d, wn_ref);

        if let Some((xs, x_mean, x_scale)) = core_r.x_std {
            d.set_item("X_std", faer_mat_to_np(py, xs))?;
            d.set_item("X_mean", faer_col_to_np(py, x_mean))?;
            d.set_item("X_scale", faer_col_to_np(py, x_scale))?;
        } else {
            d.set_item("X_std", py.None())?;
            d.set_item("X_mean", py.None())?;
            d.set_item("X_scale", py.None())?;
        }
        d.set_item("Y_std", faer_mat_to_np(py, y_std))?;
        d.set_item("Y_mean", faer_col_to_np(py, y_mean))?;
        d.set_item("Y_scale", faer_col_to_np(py, y_scale))?;
        if let Some(wn) = core_r.weights_normalized {
            d.set_item("weights_normalized", faer_col_to_np(py, wn))?;
        } else {
            d.set_item("weights_normalized", py.None())?;
        }
        d.set_item("n_eff", core_r.n_eff)?;
    } else {
        let core_in = PreprocessInput {
            x: xref,
            y: y_ref_1d,
            weights: wref,
        };
        let r = map_res(core_preprocess(core_in))?;
        if let Some((xs, x_mean, x_scale)) = r.x_std {
            d.set_item("X_std", faer_mat_to_np(py, xs))?;
            d.set_item("X_mean", faer_col_to_np(py, x_mean))?;
            d.set_item("X_scale", faer_col_to_np(py, x_scale))?;
        } else {
            d.set_item("X_std", py.None())?;
            d.set_item("X_mean", py.None())?;
            d.set_item("X_scale", py.None())?;
        }
        if let Some((ys, y_mean, y_scale)) = r.y_std {
            d.set_item("Y_std", faer_col_to_np(py, ys))?;
            d.set_item("Y_mean", y_mean)?;
            d.set_item("Y_scale", y_scale)?;
        } else {
            d.set_item("Y_std", py.None())?;
            d.set_item("Y_mean", py.None())?;
            d.set_item("Y_scale", py.None())?;
        }
        if let Some(wn) = r.weights_normalized {
            d.set_item("weights_normalized", faer_col_to_np(py, wn))?;
        } else {
            d.set_item("weights_normalized", py.None())?;
        }
        d.set_item("n_eff", r.n_eff)?;
    }
    Ok(d)
}

#[pymodule]
fn _plskit(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PlsKitException", py.get_type::<PlsKitException>())?;
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(pls1_fit, m)?)?;
    m.add_function(wrap_pyfunction!(pls1_predict, m)?)?;
    m.add_function(wrap_pyfunction!(pls1_confirmatory_test_raw, m)?)?;
    m.add_function(wrap_pyfunction!(pls1_rotation_stability_raw, m)?)?;
    m.add_function(wrap_pyfunction!(pls1_find_k_optimal, m)?)?;
    m.add_function(wrap_pyfunction!(pls1_find_k_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(pls1_perm_null_raw, m)?)?;
    m.add_function(wrap_pyfunction!(rotate, m)?)?;
    Ok(())
}
