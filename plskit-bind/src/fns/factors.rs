//! `factors`, `pca_basis`, `profiles_find_n`, `profiles` and `profiles_search`.

use faer::{Col, ColRef, Mat, MatRef};
use plskit::Pls1Model;

use crate::coerce;
use crate::convert::{self, mat};
use crate::error::BindError;
use crate::fns::rotate::{carried, product, rotate_factors, standardized};
use crate::fns::{done, engine};
use crate::inputs::Inputs;
use crate::methods;
use crate::value::{MatF64, Outcome, Record, Value};

/// A count argument. A negative whole number is `invalid_input` (the engine's
/// range error), not `invalid_argument`; every other bad value stays
/// `invalid_argument`.
fn count_value(name: &str, v: &Value<'_>) -> Result<usize, BindError> {
    let negative = match *v {
        Value::I64(n) => n < 0,
        Value::F64(x) => x.is_finite() && x.fract() == 0.0 && x < 0.0,
        _ => false,
    };
    if negative {
        return Err(BindError::new(
            "invalid_input",
            format!("{name} must be at least 1, got {}", coerce::describe(v)),
        ));
    }
    coerce::to_usize(v).map_err(|w| BindError::invalid_argument(format!("{name} {w}")))
}

fn count(inp: &mut Inputs<'_>, name: &str) -> Result<usize, BindError> {
    let v = inp.take(name);
    count_value(name, &v)
}

/// Read `model_or_coef`: a `PLS1Result` record is checked with
/// [`splittable_model`], anything else is read as a coefficient vector.
/// Returns the coefficient, the fit's own `pre_standardized` (false for a
/// vector), and, when `with_model`, the `PLS1Result` value (null otherwise
/// and for a vector).
fn first_arg(
    v: Value<'_>,
    x: MatRef<'_, f64>,
    with_model: bool,
) -> Result<(Col<f64>, bool, Value<'static>), BindError> {
    if matches!(v, Value::Record(_)) {
        let rec = convert::model_record(v, "PLS1Result", "model_or_coef")?;
        let m = splittable_model(&rec, x)?;
        let value = if with_model {
            model_value(&m, &rec)
        } else {
            Value::Null
        };
        return Ok((m.coef, m.pre_standardized, value));
    }
    let c =
        coerce::to_vec(v).map_err(|w| BindError::invalid_argument(format!("model_or_coef {w}")))?;
    Ok((c.as_col().to_owned(), false, Value::Null))
}

/// `factors(model_or_coef, X, n, *, pre_standardized=None)`.
pub(crate) fn factors(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let first = inp.take("model_or_coef");
    let x = inp.mat("X")?;
    let n = count(inp, "n")?;
    let set = inp.opt_bool("pre_standardized")?;
    inp.finish()?;
    let (coef, fit_pre, _) = first_arg(first, x.as_mat(), false)?;
    let out = engine(plskit::factors(
        coef.as_ref(),
        x.as_mat(),
        n,
        set.unwrap_or(fit_pre),
    ))?;
    Ok(done(convert::factors(&out, Value::Null)))
}

/// `pca_basis(X, *, pre_standardized=False)`.
pub(crate) fn pca_basis(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    inp.finish()?;
    let out = engine(plskit::pca_basis(x.as_mat(), pre_standardized))?;
    Ok(done(convert::pca_basis(&out)))
}

/// `profiles_find_n(X, y, k, n_max=9, *, args=None, pre_standardized=False,
/// seed=None)`.
pub(crate) fn profiles_find_n(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.usize("k")?;
    let n_max = count(inp, "n_max")?;
    let args = inp.opt_record("args")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let seed = inp.seed()?;
    inp.finish()?;
    let opts = methods::profiles_find_n_opts(args.as_ref(), pre_standardized, seed)?;
    let out = engine(plskit::profiles_find_n(
        x.as_mat(),
        y.as_col(),
        k,
        n_max,
        opts,
    ))?;
    Ok(done(convert::profiles_find_n(&out)))
}

/// The fit in `rec`, checked to be one `factors` can split on `x`.
///
/// `Xs` here is `x` standardized without weights, so a weighted `coef` would
/// be split in the wrong space. The row count is the one check that `x` is
/// the data the model was fit on.
fn splittable_model(rec: &Record<'_>, x: MatRef<'_, f64>) -> Result<Pls1Model, BindError> {
    let m = convert::pls1_model_from_record(rec)?;
    if m.weights.is_some() {
        return Err(BindError::new(
            "invalid_input",
            "model must be an unweighted fit: it has weights",
        ));
    }
    if m.t_scores.nrows() != x.nrows() {
        return Err(BindError::new(
            "invalid_input",
            format!(
                "X has {} rows, the model was fit on {}",
                x.nrows(),
                m.t_scores.nrows()
            ),
        ));
    }
    if m.k_used == 0 {
        return Err(BindError::new(
            "invalid_input",
            "coef is zero: the fit kept no component, so there is nothing to split",
        ));
    }
    Ok(m)
}

/// `m` as a `PLS1Result`, with the wrapper-level `selection_result` of `rec`.
fn model_value(m: &Pls1Model, rec: &Record<'_>) -> Value<'static> {
    Value::Record(convert::pls1_model(m, carried(rec, "selection_result")))
}

/// The `FactorsResult` of `coef` at `n`, rotated, and its scores `Xs F`.
///
/// Runs `factors` and `rotate` with `L = Xs W_f`.
fn split_and_rotate(
    x: MatRef<'_, f64>,
    coef: ColRef<'_, f64>,
    n: usize,
    pre_standardized: bool,
    rotation: &str,
    args: Option<&Record<'_>>,
) -> Result<(Value<'static>, Value<'static>), BindError> {
    let f = engine(plskit::factors(coef, x, n, pre_standardized))?;
    let xs_owned = standardized(x, pre_standardized)?;
    let xs = xs_owned.as_ref().map_or(x, Mat::as_ref);
    let l = MatF64::from_mat(&product(xs, f.w.as_ref()));
    let (rotated, spec) = rotate_factors(&f, rotation, Some(&l), args)?;
    let scores = product(xs, rotated.f.as_ref());
    Ok((
        Value::Record(convert::factors(&rotated, Value::Record(spec))),
        mat(&scores),
    ))
}

/// `profiles(model_or_coef, X, n, *, rotation="varimax", args=None,
/// pre_standardized=None)`.
///
/// `X` is the data the fit came from. Standardization follows
/// `pre_standardized`, else the model, else is on for a vector.
pub(crate) fn profiles(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let first = inp.take("model_or_coef");
    let x = inp.mat("X")?;
    let n = count(inp, "n")?;
    let rotation = inp.string("rotation")?;
    let args = inp.opt_record("args")?;
    let set = inp.opt_bool("pre_standardized")?;
    inp.finish()?;
    methods::rotation_method(&rotation, args.as_ref())?;
    let (coef, fit_pre, model) = first_arg(first, x.as_mat(), true)?;
    let (factors, scores) = split_and_rotate(
        x.as_mat(),
        coef.as_ref(),
        n,
        set.unwrap_or(fit_pre),
        &rotation,
        args.as_ref(),
    )?;
    Ok(done(
        Record::typed("ProfilesResult")
            .field("model", model)
            .field("factors", factors)
            .field("scores", scores),
    ))
}

/// `profiles_search(model, X, y, *, n_max=9, rotation="varimax", args=None,
/// find_n_args=None, seed=None)`.
///
/// `profiles` at the `n` that `profiles_find_n` picks, with that search on
/// the result. Fold fits use `model.k_used` components. `n_star = 0` runs
/// `n = 1`.
pub(crate) fn profiles_search(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let model = inp.take("model");
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let n_max = count(inp, "n_max")?;
    let rotation = inp.string("rotation")?;
    let args = inp.opt_record("args")?;
    let find_n_args = inp.opt_record("find_n_args")?;
    let seed = inp.seed()?;
    inp.finish()?;
    methods::rotation_method(&rotation, args.as_ref())?;
    let rec = convert::model_record(model, "PLS1Result", "model")?;
    let m = splittable_model(&rec, x.as_mat())?;
    if m.keep.is_some() {
        return Err(BindError::new(
            "invalid_input",
            "model must be a dense fit: it has keep, and the search fits dense folds",
        ));
    }
    let opts = methods::profiles_find_n_opts(find_n_args.as_ref(), m.pre_standardized, seed)?;
    let found = engine(plskit::profiles_find_n(
        x.as_mat(),
        y.as_col(),
        m.k_used,
        n_max,
        opts,
    ))?;
    let n = found.n_star.max(1);
    let (factors, scores) = split_and_rotate(
        x.as_mat(),
        m.coef.as_ref(),
        n,
        m.pre_standardized,
        &rotation,
        args.as_ref(),
    )?;
    Ok(done(
        Record::typed("ProfilesSearchResult")
            .field("model", model_value(&m, &rec))
            .field("factors", factors)
            .field("scores", scores)
            .field("n_search", Value::Record(convert::profiles_find_n(&found)))
            .field("seed", Value::U64(found.seed)),
    ))
}
