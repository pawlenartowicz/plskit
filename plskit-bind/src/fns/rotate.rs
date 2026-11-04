//! `rotate`: array, `FactorsResult` and `PLS3Result` overloads.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use plskit::{FactorsOutput, PreprocessInput, RotationMethod};

use crate::coerce;
use crate::convert::{self, count, mat};
use crate::error::BindError;
use crate::fns::{done, engine, with_warning};
use crate::inputs::Inputs;
use crate::methods;
use crate::value::{MatF64, Outcome, Record, Value};
use crate::warn;

struct Rotated {
    w_rot: Mat<f64>,
    r: Mat<f64>,
    spec: Record<'static>,
}

pub(crate) fn product(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut c = Mat::<f64>::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, 1.0, Par::Seq);
    c
}

/// `x` standardized, or `None` when `pre_standardized` says to use it as given.
pub(crate) fn standardized(
    x: MatRef<'_, f64>,
    pre_standardized: bool,
) -> Result<Option<Mat<f64>>, BindError> {
    if pre_standardized {
        return Ok(None);
    }
    let r = engine(plskit::preprocess(PreprocessInput {
        x: Some(x),
        y: None,
        weights: None,
    }))?;
    Ok(Some(r.x_std.expect("X was passed").0))
}

fn rotate_array(
    w: MatRef<'_, f64>,
    method: &str,
    l: Option<&MatF64<'_>>,
    args: Option<&Record<'_>>,
) -> Result<Rotated, BindError> {
    let va = methods::rotation_method(method, args)?;
    let out = engine(plskit::rotate(
        w,
        RotationMethod::Varimax(va),
        l.map(MatF64::as_mat),
    ))?;
    let resolved = Record::new()
        .field("max_iter", count(va.max_iter))
        .field("tol", Value::F64(va.tol))
        .field("kaiser_normalize", Value::Bool(va.kaiser_normalize));
    let spec = Record::typed("RotationSpec")
        .field("method", Value::text(method))
        .field("args", Value::Record(resolved))
        .field("R", mat(&out.r))
        .field("sweeps", count(out.sweeps))
        .field("V_converged", Value::F64(out.v_converged))
        .field("L_was_provided", Value::Bool(l.is_some()));
    Ok(Rotated {
        w_rot: out.w_rot,
        r: out.r,
        spec,
    })
}

/// Varimax-rotate the unit directions of `f`, then sort, sign-pin and
/// re-project through [`FactorsOutput::rotated`]. The returned spec holds
/// the `R` after sorting and sign pinning, so `W_after = W_before · R`.
pub(crate) fn rotate_factors(
    f: &FactorsOutput,
    method: &str,
    l: Option<&MatF64<'_>>,
    args: Option<&Record<'_>>,
) -> Result<(FactorsOutput, Record<'static>), BindError> {
    let mut rot = rotate_array(f.w.as_ref(), method, l, args)?;
    let (out, r) = engine(f.rotated(rot.r.as_ref()))?;
    rot.spec.set("R", mat(&r));
    Ok((out, rot.spec))
}

// `rotation_spec` exists only on the wrapper records, not on engine results,
// so `already_rotated` is raised here and has no `PlsKitError` variant.
fn check_not_rotated(rec: &Record<'_>, what: &str) -> Result<(), BindError> {
    if rec.get("rotation_spec").is_some_and(|v| !v.is_null()) {
        return Err(BindError::new(
            "already_rotated",
            format!("{what} already has a rotation_spec; re-rotation is not supported"),
        ));
    }
    Ok(())
}

/// `Xs W` as a rotation target. `Xs` is `x` standardized unless
/// `pre_standardized`.
fn row_target(
    x: MatRef<'_, f64>,
    w: MatRef<'_, f64>,
    pre_standardized: bool,
) -> Result<MatF64<'static>, BindError> {
    if x.ncols() != w.nrows() {
        return Err(BindError::new(
            "shape_mismatch",
            format!(
                "X has {} columns, the weights have {} rows",
                x.ncols(),
                w.nrows()
            ),
        ));
    }
    if !x.is_all_finite() {
        return Err(BindError::new(
            "invalid_input",
            "X contains non-finite values",
        ));
    }
    let xs_owned = standardized(x, pre_standardized)?;
    let xs = xs_owned.as_ref().map_or(x, Mat::as_ref);
    Ok(MatF64::from_mat(&product(xs, w)))
}

/// A field of `rec` copied as it is, or null.
pub(crate) fn carried(rec: &Record<'_>, key: &str) -> Value<'static> {
    rec.get(key).map_or(Value::Null, |v| v.clone().into_owned())
}

/// `rotate(model_or_W, X=None, *, method="varimax", L=None, args=None,
/// pre_standardized=None)`.
///
/// `X` sets the target to `Xs W`. `pre_standardized` defaults to the
/// result's own value, and to false for a matrix. For a `PLS3Result` the
/// target is its stored `x_scores` (`X` is checked for its row count only),
/// so `pre_standardized` is refused there.
pub(crate) fn rotate(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let target = inp.take("model_or_W");
    let x = inp.opt_mat("X")?;
    let method = inp.string("method")?;
    let l = inp.opt_mat("L")?;
    let args = inp.opt_record("args")?;
    let set = inp.opt_bool("pre_standardized")?;
    inp.finish()?;
    if x.is_some() && l.is_some() {
        return Err(BindError::invalid_argument(
            "X and L both set the rotation target: pass one of them",
        ));
    }
    if x.is_none() && set.is_some() {
        return Err(BindError::invalid_argument(
            "pre_standardized needs X: it says how to standardize X",
        ));
    }
    match target {
        Value::Mat(w) => {
            let l = match &x {
                Some(x) => Some(row_target(x.as_mat(), w.as_mat(), set.unwrap_or(false))?),
                None => l,
            };
            let rot = rotate_array(w.as_mat(), &method, l.as_ref(), args.as_ref())?;
            Ok(done(
                Record::typed("RotateResult")
                    .field("W_rot", mat(&rot.w_rot))
                    .field("spec", Value::Record(rot.spec)),
            ))
        }
        Value::Record(r) if r.type_name() == Some("FactorsResult") => {
            check_not_rotated(&r, "factors")?;
            let f = convert::factors_from_record(&r)?;
            let l = match &x {
                Some(x) => Some(row_target(
                    x.as_mat(),
                    f.w.as_ref(),
                    set.unwrap_or(f.pre_standardized),
                )?),
                None => l,
            };
            let (out, spec) = rotate_factors(&f, &method, l.as_ref(), args.as_ref())?;
            Ok(done(convert::factors(&out, Value::Record(spec))))
        }
        Value::Record(r) if r.type_name() == Some("PLS3Result") => {
            if set.is_some() {
                return Err(BindError::invalid_argument(
                    "pre_standardized is not taken for a PLS3Result: X gives its stored x_scores as the target",
                ));
            }
            check_not_rotated(&r, "model")?;
            let m = convert::pls3_model_from_record(&r)?;
            let l = match &x {
                Some(x) => {
                    if x.ncols() != m.u_saliences.nrows() {
                        return Err(BindError::new(
                            "shape_mismatch",
                            format!(
                                "X has {} columns, the weights have {} rows",
                                x.ncols(),
                                m.u_saliences.nrows()
                            ),
                        ));
                    }
                    if x.nrows() != m.x_scores.nrows() {
                        return Err(BindError::new(
                            "invalid_input",
                            format!(
                                "X has {} rows, the model was fit on {}",
                                x.nrows(),
                                m.x_scores.nrows()
                            ),
                        ));
                    }
                    Some(MatF64::from_mat(&m.x_scores))
                }
                None => l,
            };
            let mut rot = rotate_array(m.u_saliences.as_ref(), &method, l.as_ref(), args.as_ref())?;
            let (out, r_final) = engine(m.rotated(rot.r.as_ref()))?;
            rot.spec.set("R", mat(&r_final));
            let mut rec = convert::pls3_model(&out, Value::Record(rot.spec));
            rec.set("converged", carried(&r, "converged"));
            rec.set("n_iter", carried(&r, "n_iter"));
            let sparse = m.keep_x.is_some() || m.keep_y.is_some();
            Ok(with_warning(rec, sparse.then(warn::rotated_sparse)))
        }
        Value::Record(r) if r.type_name() == Some("PLS1Result") => {
            Err(BindError::invalid_argument(
                "rotate() does not take a PLS1Result: use profiles(m, X, n), or rotate(m.W, ...) for the weights alone",
            ))
        }
        other => Err(BindError::invalid_argument(format!(
            "rotate() first arg must be a FactorsResult, a PLS3Result or a matrix, got {}",
            coerce::describe(&other)
        ))),
    }
}
