//! `rotate`: array and model overloads. The model overload composes
//! `T·R`, `P·R`, `Rᵀ·Q` here in faer (Python composes them in numpy
//! today; P5 moves Python onto this path).

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use plskit::RotationMethod;

use crate::coerce;
use crate::convert::{self, count, mat};
use crate::error::BindError;
use crate::fns::{done, engine};
use crate::inputs::Inputs;
use crate::methods;
use crate::types;
use crate::value::{MatF64, Outcome, Record, Value, VecF64};

struct Rotated {
    w_rot: Mat<f64>,
    r: Mat<f64>,
    spec: Record<'static>,
}

fn product(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut c = Mat::<f64>::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, 1.0, Par::Seq);
    c
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

/// A hand-edited `PLS1Result` (Decision 6 allows untagged records) can
/// carry `T`, `P` or `Q` that no longer agree with `W`'s column count.
/// Composing them would panic inside faer's `matmul`, so check shapes
/// here and raise `invalid_argument` instead.
fn check_model_shape(model: &plskit::Pls1Model) -> Result<(), BindError> {
    let k = model.w_star.ncols();
    if model.t_scores.ncols() != k {
        return Err(BindError::invalid_argument(format!(
            "model field 'T' has {} columns, W has {k}",
            model.t_scores.ncols()
        )));
    }
    if model.p_loadings.ncols() != k {
        return Err(BindError::invalid_argument(format!(
            "model field 'P' has {} columns, W has {k}",
            model.p_loadings.ncols()
        )));
    }
    if model.p_loadings.nrows() != model.w_star.nrows() {
        return Err(BindError::invalid_argument(format!(
            "model field 'P' has {} rows, W has {} rows",
            model.p_loadings.nrows(),
            model.w_star.nrows()
        )));
    }
    if model.q_loadings.nrows() != k {
        return Err(BindError::invalid_argument(format!(
            "model field 'Q' has length {}, W has {k}",
            model.q_loadings.nrows()
        )));
    }
    Ok(())
}

fn rotate_model(
    rec: Record<'_>,
    method: &str,
    l: Option<&MatF64<'_>>,
    args: Option<&Record<'_>>,
) -> Result<Outcome, BindError> {
    if rec.get("rotation_spec").is_some_and(|v| !v.is_null()) {
        return Err(BindError::new(
            "already_rotated",
            "model already has a rotation_spec; re-rotation is not supported",
        ));
    }
    let model = convert::pls1_model_from_record(&rec)?;
    check_model_shape(&model)?;
    let rot = rotate_array(model.w_star.as_ref(), method, l, args)?;
    let t = product(model.t_scores.as_ref(), rot.r.as_ref());
    let p = product(model.p_loadings.as_ref(), rot.r.as_ref());
    let q = product(rot.r.transpose(), model.q_loadings.as_ref().as_mat());
    // Every field the rotation does not touch (keep, selection_result,
    // weights, ...) is carried through, as Python's dataclasses.replace does.
    let mut updated = rec.into_owned();
    updated.set_type_name("PLS1Result");
    updated.set("T", mat(&t));
    updated.set("P", mat(&p));
    updated.set("W", mat(&rot.w_rot));
    updated.set(
        "Q",
        Value::Vec(VecF64::Owned((0..q.nrows()).map(|i| q[(i, 0)]).collect())),
    );
    updated.set("rotation_spec", Value::Record(rot.spec));
    // Rebuild strictly to the PLS1Result field list (Python's
    // dataclasses.replace semantics), so an untagged or partial input
    // record does not leave stray or missing fields in the output.
    let spec = types::result_type("PLS1Result").expect("PLS1Result is a declared result type");
    let mut out = Record::typed("PLS1Result");
    for f in spec.fields {
        let v = updated.get(f.name).cloned().unwrap_or(Value::Null);
        out.set(f.name, v);
    }
    Ok(done(out))
}

/// `rotate(model_or_W, *, method="varimax", L=None, args=None)`.
pub(crate) fn rotate(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let target = inp.take("model_or_W");
    let method = inp.string("method")?;
    let l = inp.opt_mat("L")?;
    let args = inp.opt_record("args")?;
    inp.finish()?;
    match target {
        Value::Mat(w) => {
            let rot = rotate_array(w.as_mat(), &method, l.as_ref(), args.as_ref())?;
            Ok(done(
                Record::typed("RotateResult")
                    .field("W_rot", mat(&rot.w_rot))
                    .field("spec", Value::Record(rot.spec)),
            ))
        }
        v @ Value::Record(_) => {
            let rec = convert::model_record(v, "PLS1Result", "rotate() first arg")?;
            rotate_model(rec, &method, l.as_ref(), args.as_ref())
        }
        other => Err(BindError::invalid_argument(format!(
            "rotate() first arg must be a PLS1Result or a matrix, got {}",
            coerce::describe(&other)
        ))),
    }
}
