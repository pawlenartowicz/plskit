//! PLS3 / PLSSVD and sparse PLS3.

use plskit::Pls3FitOpts;

use crate::convert;
use crate::error::BindError;
use crate::fns::{done, engine, pls3_y};
use crate::inputs::Inputs;
use crate::methods;
use crate::value::{MatF64, Outcome, VecF64};

/// `pls3_fit` / `plssvd_fit(X, Y, k=1, *, pre_standardized_X=False,
/// pre_standardized_Y=False, weights=None)`.
pub(crate) fn pls3_fit(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = pls3_y(inp)?;
    let k = inp.usize("k")?;
    let pre_standardized_x = inp.bool("pre_standardized_X")?;
    let pre_standardized_y = inp.bool("pre_standardized_Y")?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let opts = Pls3FitOpts {
        pre_standardized_x,
        pre_standardized_y,
        ..Pls3FitOpts::default()
    };
    let m = engine(plskit::pls3_fit(
        x.as_mat(),
        y.as_mat(),
        k,
        w.as_ref().map(VecF64::as_col),
        opts,
    ))?;
    Ok(done(convert::pls3_model(&m)))
}

/// `spls3_fit(X, Y, k, keep_X, keep_Y, *, pre_standardized_X=False,
/// pre_standardized_Y=False, max_iter=None, tol=None, weights=None)`.
pub(crate) fn spls3_fit(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = pls3_y(inp)?;
    let k = inp.usize("k")?;
    let keep_x = inp.usize("keep_X")?;
    let keep_y = inp.usize("keep_Y")?;
    let pre_standardized_x = inp.bool("pre_standardized_X")?;
    let pre_standardized_y = inp.bool("pre_standardized_Y")?;
    let max_iter = inp.opt_usize("max_iter")?;
    let tol = inp.opt_f64("tol")?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let d = Pls3FitOpts::default();
    let opts = Pls3FitOpts {
        pre_standardized_x,
        pre_standardized_y,
        max_iter: max_iter.unwrap_or(d.max_iter),
        tol: tol.unwrap_or(d.tol),
        ..d
    };
    let m = engine(plskit::spls3_fit(
        x.as_mat(),
        y.as_mat(),
        k,
        keep_x,
        keep_y,
        w.as_ref().map(VecF64::as_col),
        opts,
    ))?;
    Ok(done(convert::pls3_model(&m)))
}

/// `pls3_transform` / `plssvd_transform(model, X_new=None, Y_new=None, *,
/// which="both")`.
pub(crate) fn pls3_transform(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let model = inp.take("model");
    let x_new = inp.opt_mat("X_new")?;
    let y_new = inp.opt_mat("Y_new")?;
    let which = inp.string("which")?;
    inp.finish()?;
    let rec = convert::model_record(model, "PLS3Result", "model")?;
    let m = convert::pls3_model_from_record(&rec)?;
    let s = engine(plskit::pls3_transform(
        &m,
        x_new.as_ref().map(MatF64::as_mat),
        y_new.as_ref().map(MatF64::as_mat),
        methods::which(&which)?,
    ))?;
    Ok(done(convert::pls3_scores(&s)))
}
