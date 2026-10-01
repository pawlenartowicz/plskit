//! preprocess, `pls1_fit`, `pls1_predict`, `spls1_fit`.

use faer::Col;
use plskit::{FitOpts, KSpec, PreprocessInput};

use crate::coerce;
use crate::convert::{self, col, mat, opt};
use crate::error::BindError;
use crate::fns::find_k;
use crate::fns::{done, engine};
use crate::inputs::Inputs;
use crate::value::{MatF64, Outcome, Record, Value, VecF64};

fn put_x(
    rec: Record<'static>,
    x_std: Option<(faer::Mat<f64>, Col<f64>, Col<f64>)>,
) -> Record<'static> {
    match x_std {
        Some((xs, mean, scale)) => rec
            .field("X_std", mat(&xs))
            .field("X_mean", col(&mean))
            .field("X_scale", col(&scale)),
        None => rec
            .field("X_std", Value::Null)
            .field("X_mean", Value::Null)
            .field("X_scale", Value::Null),
    }
}

/// `preprocess(X=None, Y=None, weights=None)`; `Y` may be 1-D or 2-D.
pub(crate) fn preprocess(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.opt_mat("X")?;
    let y = inp.take("Y");
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let xref = x.as_ref().map(MatF64::as_mat);
    let wref = w.as_ref().map(VecF64::as_col);
    let rec = Record::typed("PreprocessResult");
    if let Value::Mat(ym) = &y {
        let core = engine(plskit::preprocess::preprocess_block(
            plskit::preprocess::PreprocessBlockInput {
                x: xref,
                y: Some(ym.as_mat()),
                weights: wref,
            },
        ))?;
        let (y_std, y_mean, y_scale) = core.y_std.expect("Y was passed");
        let rec = put_x(rec, core.x_std)
            .field("Y_std", mat(&y_std))
            .field("Y_mean", col(&y_mean))
            .field("Y_scale", col(&y_scale))
            .field(
                "weights_normalized",
                opt(core.weights_normalized.as_ref(), col),
            )
            .field("n_eff", opt(core.n_eff, Value::F64));
        return Ok(done(rec));
    }
    let y1 = match y {
        Value::Null => None,
        v => Some(coerce::to_vec(v).map_err(|w| BindError::invalid_argument(format!("Y {w}")))?),
    };
    let r = engine(plskit::preprocess(PreprocessInput {
        x: xref,
        y: y1.as_ref().map(VecF64::as_col),
        weights: wref,
    }))?;
    let rec = put_x(rec, r.x_std);
    let rec = match r.y_std {
        Some((ys, mean, scale)) => rec
            .field("Y_std", col(&ys))
            .field("Y_mean", Value::F64(mean))
            .field("Y_scale", Value::F64(scale)),
        None => rec
            .field("Y_std", Value::Null)
            .field("Y_mean", Value::Null)
            .field("Y_scale", Value::Null),
    };
    Ok(done(
        rec.field(
            "weights_normalized",
            opt(r.weights_normalized.as_ref(), col),
        )
        .field("n_eff", opt(r.n_eff, Value::F64)),
    ))
}

/// `pls1_fit(X, y, k=1, *, k_max=None, find_k_args=None,
/// pre_standardized=False, seed=None, weights=None)`.
///
/// `k` is an integer, or `"optimal"` / `"sequence"` (then `k_max` is
/// required and `find_k_args` is forwarded to the selection). With an
/// integer `k`, a non-null `k_max` or `find_k_args` raises
/// `invalid_argument`, and `seed` is accepted and ignored.
pub(crate) fn pls1_fit(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.take("k");
    let k_max = inp.opt_usize("k_max")?;
    let find_k_args = inp.opt_record("find_k_args")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let seed = inp.seed()?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let d = find_k::Data {
        x: x.as_mat(),
        y: y.as_col(),
        weights: w.as_ref().map(VecF64::as_col),
    };
    let (k_int, selection, warnings) = match k {
        Value::Str(mode) => find_k::select_k(d, &mode, k_max, find_k_args, pre_standardized, seed)?,
        other => {
            if k_max.is_some() || find_k_args.is_some() {
                return Err(BindError::invalid_argument(
                    "k_max and find_k_args apply only when k is 'optimal' or 'sequence'",
                ));
            }
            let k = coerce::to_usize(&other)
                .map_err(|w| BindError::invalid_argument(format!("k {w}")))?;
            (k, Value::Null, Vec::new())
        }
    };
    let model = engine(plskit::pls1_fit(
        d.x,
        d.y,
        KSpec::Fixed(k_int),
        d.weights,
        FitOpts {
            pre_standardized,
            ..FitOpts::default()
        },
    ))?;
    Ok(Outcome {
        result: Value::Record(convert::pls1_model(&model, Value::Null, selection)),
        warnings,
    })
}

/// `pls1_predict(model, X_new)`; returns a plain vector.
pub(crate) fn pls1_predict(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let model = inp.take("model");
    let x_new = inp.mat("X_new")?;
    inp.finish()?;
    let rec = convert::model_record(model, "PLS1Result", "model")?;
    let m = convert::pls1_model_from_record(&rec)?;
    let yhat = engine(plskit::pls1_predict(&m, x_new.as_mat()))?;
    Ok(Outcome::new(col(&yhat)))
}

/// `spls1_fit(X, y, k, keep, *, pre_standardized=False, weights=None)`.
pub(crate) fn spls1_fit(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.usize("k")?;
    let keep = inp.usize("keep")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let m = engine(plskit::spls1_fit(
        x.as_mat(),
        y.as_col(),
        KSpec::Fixed(k),
        keep,
        w.as_ref().map(VecF64::as_col),
        FitOpts {
            pre_standardized,
            ..FitOpts::default()
        },
    ))?;
    Ok(done(convert::pls1_model(&m, Value::Null, Value::Null)))
}
