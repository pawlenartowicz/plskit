//! PLS1 / sPLS1 K selection, and `pls1_fit`'s `k = "optimal" | "sequence"`.

use faer::{ColRef, MatRef};
use plskit::{FindKOptimalOutput, FindKSequenceOutput};

use crate::coerce;
use crate::convert;
use crate::error::BindError;
use crate::fmt::{py_list, py_repr};
use crate::fns::{done, engine, with_warning};
use crate::inputs::Inputs;
use crate::methods::{self, Common};
use crate::value::{Outcome, Record, Value, VecF64};
use crate::warn;

/// The data every K-selection call runs on.
#[derive(Clone, Copy)]
pub(crate) struct Data<'d> {
    pub(crate) x: MatRef<'d, f64>,
    pub(crate) y: ColRef<'d, f64>,
    pub(crate) weights: Option<ColRef<'d, f64>>,
}

/// Body shared by `pls1_find_k_optimal`, `spls1_find_k_optimal` and
/// `pls1_fit(k = "optimal")`. With `for_fit`, a selection with no K to
/// fit raises the engine's `optimal_no_component` before any warning.
#[allow(clippy::too_many_arguments)]
fn optimal_core(
    d: Data<'_>,
    k_max: usize,
    keep: Option<usize>,
    selector: &str,
    diagnostic: Option<&str>,
    args: Option<&Record<'_>>,
    c: &Common,
    for_fit: bool,
) -> Result<(FindKOptimalOutput, Option<Record<'static>>), BindError> {
    let opts = methods::find_k_optimal_opts(selector, diagnostic, args, c)?;
    let r = match keep {
        None => engine(plskit::pls1_find_k_optimal(
            d.x, d.y, k_max, d.weights, opts,
        ))?,
        Some(kp) => engine(plskit::spls1_find_k_optimal(
            d.x, d.y, k_max, kp, d.weights, opts,
        ))?,
    };
    if for_fit {
        engine(r.k_to_fit())?;
    }
    let warning = warn::rerouted(
        diagnostic,
        r.diagnostic.as_deref(),
        Some(plskit::SPLIT_NB_REROUTE_N_PERM),
        r.stable_rank,
        Some(r.n_eff),
    );
    Ok((r, warning))
}

/// Body shared by `pls1_find_k_sequence`, `spls1_find_k_sequence` and
/// `pls1_fit(k = "sequence")`.
#[allow(clippy::too_many_arguments)]
fn sequence_core(
    d: Data<'_>,
    k_max: usize,
    keep: Option<usize>,
    test_method: &str,
    alpha: Option<f64>,
    args: Option<&Record<'_>>,
    c: &Common,
    for_fit: bool,
) -> Result<(FindKSequenceOutput, Option<Record<'static>>), BindError> {
    let opts = methods::find_k_sequence_opts(test_method, alpha, args, c)?;
    let r = match keep {
        None => engine(plskit::pls1_find_k_sequence(
            d.x, d.y, k_max, d.weights, opts,
        ))?,
        Some(kp) => engine(plskit::spls1_find_k_sequence(
            d.x, d.y, k_max, kp, d.weights, opts,
        ))?,
    };
    if for_fit {
        engine(r.k_to_fit())?;
    }
    let warning = warn::rerouted(
        Some(test_method),
        Some(&r.test_method),
        Some(plskit::SPLIT_NB_REROUTE_N_PERM),
        r.stable_rank,
        Some(r.n_eff),
    );
    Ok((r, warning))
}

fn common(inp: &mut Inputs<'_>) -> Result<Common, BindError> {
    Ok(Common {
        pre_standardized: inp.bool("pre_standardized")?,
        seed: inp.seed()?,
        verbose: inp.bool("verbose")?,
    })
}

fn optimal_entry(inp: &mut Inputs<'_>, sparse: bool) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k_max = inp.usize("k_max")?;
    let keep = if sparse {
        Some(inp.usize("keep")?)
    } else {
        None
    };
    let selector = inp.string("selector")?;
    let diagnostic = inp.opt_string("diagnostic")?;
    let args = inp.opt_record("args")?;
    let c = common(inp)?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let d = Data {
        x: x.as_mat(),
        y: y.as_col(),
        weights: w.as_ref().map(VecF64::as_col),
    };
    let (r, warning) = optimal_core(
        d,
        k_max,
        keep,
        &selector,
        diagnostic.as_deref(),
        args.as_ref(),
        &c,
        false,
    )?;
    Ok(with_warning(convert::find_k_optimal(&r), warning))
}

fn sequence_entry(inp: &mut Inputs<'_>, sparse: bool) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k_max = inp.usize("k_max")?;
    let keep = if sparse {
        Some(inp.usize("keep")?)
    } else {
        None
    };
    let test_method = inp.string("test_method")?;
    let alpha = inp.opt_f64("alpha")?;
    let args = inp.opt_record("args")?;
    let c = common(inp)?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let d = Data {
        x: x.as_mat(),
        y: y.as_col(),
        weights: w.as_ref().map(VecF64::as_col),
    };
    let (r, warning) = sequence_core(
        d,
        k_max,
        keep,
        &test_method,
        alpha,
        args.as_ref(),
        &c,
        false,
    )?;
    Ok(with_warning(convert::find_k_sequence(&r), warning))
}

/// `pls1_find_k_optimal(X, y, k_max, *, selector="r2_se", diagnostic=None,
/// args=None, pre_standardized=False, seed=None,
/// verbose=False, weights=None)`.
pub(crate) fn pls1_find_k_optimal(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    optimal_entry(inp, false)
}

/// `spls1_find_k_optimal(X, y, k_max, keep, *, ...)`: as above at a fixed keep.
pub(crate) fn spls1_find_k_optimal(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    optimal_entry(inp, true)
}

/// `pls1_find_k_sequence(X, y, k_max, *, test_method="auto", alpha=None,
/// args=None, pre_standardized=False, seed=None,
/// verbose=False, weights=None)`.
pub(crate) fn pls1_find_k_sequence(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    sequence_entry(inp, false)
}

/// `spls1_find_k_sequence(X, y, k_max, keep, *, ...)`.
pub(crate) fn spls1_find_k_sequence(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    sequence_entry(inp, true)
}

/// `spls1_find_keep_optimal(X, y, k, *, args=None, seed=None,
/// verbose=False, weights=None)`.
pub(crate) fn spls1_find_keep_optimal(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.usize("k")?;
    let args = inp.opt_record("args")?;
    let seed = inp.seed()?;
    let verbose = inp.bool("verbose")?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let opts = methods::keep_optimal_opts(args.as_ref(), seed, verbose)?;
    let r = engine(plskit::spls1_find_keep_optimal(
        x.as_mat(),
        y.as_col(),
        k,
        w.as_ref().map(VecF64::as_col),
        opts,
    ))?;
    Ok(done(convert::find_keep_optimal(&r)))
}

fn check_find_k_args(fk: &Record<'_>, allowed: &[&str]) -> Result<(), BindError> {
    for key in fk.keys() {
        if !allowed.contains(&key) {
            return Err(BindError::invalid_args(format!(
                "find_k_args does not accept arg {}; allowed: {}",
                py_repr(key),
                py_list(allowed)
            )));
        }
    }
    Ok(())
}

fn fk_str(fk: &Record<'_>, key: &str) -> Result<Option<String>, BindError> {
    methods::arg(Some(fk), key)
        .map(|v| {
            coerce::to_str(v)
                .map(str::to_owned)
                .map_err(|w| BindError::invalid_args(format!("find_k_args['{key}'] {w}")))
        })
        .transpose()
}

fn fk_f64(fk: &Record<'_>, key: &str) -> Result<Option<f64>, BindError> {
    methods::arg(Some(fk), key)
        .map(|v| {
            coerce::to_f64(v)
                .map_err(|w| BindError::invalid_args(format!("find_k_args['{key}'] {w}")))
        })
        .transpose()
}

fn fk_args<'r, 'a>(fk: &'r Record<'a>) -> Result<Option<&'r Record<'a>>, BindError> {
    match methods::arg(Some(fk), "args") {
        None => Ok(None),
        Some(Value::Record(r)) => Ok(Some(r)),
        Some(v) => Err(BindError::invalid_args(format!(
            "find_k_args['args'] must be a record of named values, got {}",
            coerce::describe(v)
        ))),
    }
}

/// `pls1_fit`'s string `k`: `find_k_args` validated, the selection run with
/// `for_fit`, and the K to fit returned with the selection record and any
/// reroute warning.
pub(crate) fn select_k(
    d: Data<'_>,
    mode: &str,
    k_max: Option<usize>,
    find_k_args: Option<Record<'_>>,
    pre_standardized: bool,
    seed: Option<u64>,
) -> Result<(usize, Value<'static>, Vec<Record<'static>>), BindError> {
    let Some(k_max) = k_max else {
        return Err(BindError::invalid_argument(format!(
            "k={} requires k_max",
            py_repr(mode)
        )));
    };
    let fk = find_k_args.unwrap_or_default();
    let c = Common {
        pre_standardized,
        seed,
        verbose: false,
    };
    match mode {
        "optimal" => {
            check_find_k_args(&fk, &["selector", "diagnostic", "args"])?;
            let selector = fk_str(&fk, "selector")?.unwrap_or_else(|| "r2_se".to_owned());
            let diagnostic = fk_str(&fk, "diagnostic")?;
            let (r, w) = optimal_core(
                d,
                k_max,
                None,
                &selector,
                diagnostic.as_deref(),
                fk_args(&fk)?,
                &c,
                true,
            )?;
            Ok((
                r.k_star,
                Value::Record(convert::find_k_optimal(&r)),
                w.into_iter().collect(),
            ))
        }
        "sequence" => {
            check_find_k_args(&fk, &["test_method", "alpha", "args"])?;
            let test_method = fk_str(&fk, "test_method")?.unwrap_or_else(|| "auto".to_owned());
            let alpha = fk_f64(&fk, "alpha")?;
            let (r, w) =
                sequence_core(d, k_max, None, &test_method, alpha, fk_args(&fk)?, &c, true)?;
            Ok((
                r.k_star,
                Value::Record(convert::find_k_sequence(&r)),
                w.into_iter().collect(),
            ))
        }
        _ => Err(BindError::invalid_argument(format!(
            "unknown k mode {}; use int, 'optimal', or 'sequence'",
            py_repr(mode)
        ))),
    }
}
