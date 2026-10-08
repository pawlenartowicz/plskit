//! Engine outputs to records (`to_record`), and model records back to
//! engine models (`from_record`).
//!
//! `from_record` reads only the fields the engine model needs and ignores
//! the rest (a result also carries `selection_result`, `rotation_spec`,
//! ...). A missing or mistyped needed field raises `invalid_argument`.

use std::collections::BTreeMap;

use faer::{Col, Mat};
use plskit::{
    CIScalar, ConfirmatoryCI, ConfirmatoryTestOutput, FindKOptimalOutput, FindKSequenceOutput,
    FindKeepOptimalOutput, PermNullOutput, Pls1Model, Pls3Model, Pls3Scores,
    RotationStabilityOutput,
};

use crate::coerce;
use crate::error::BindError;
use crate::value::{MatF64, Record, Value, VecF64};

pub(crate) fn mat(m: &Mat<f64>) -> Value<'static> {
    Value::Mat(MatF64::from_mat(m))
}

pub(crate) fn col(c: &Col<f64>) -> Value<'static> {
    Value::Vec(VecF64::from_col(c))
}

pub(crate) fn count(n: usize) -> Value<'static> {
    Value::I64(i64::try_from(n).unwrap_or(i64::MAX))
}

pub(crate) fn opt<T>(o: Option<T>, f: impl FnOnce(T) -> Value<'static>) -> Value<'static> {
    o.map_or(Value::Null, f)
}

/// `PLS1Result` from an engine model plus the two wrapper-level fields.
pub(crate) fn pls1_model(
    m: &Pls1Model,
    rotation_spec: Value<'static>,
    selection_result: Value<'static>,
) -> Record<'static> {
    Record::typed("PLS1Result")
        .field("T", mat(&m.t_scores))
        .field("P", mat(&m.p_loadings))
        .field("W", mat(&m.w_star))
        .field("Q", col(&m.q_loadings))
        .field("coef", col(&m.coef))
        .field("beta", col(&m.beta))
        .field("intercept", Value::F64(m.intercept))
        .field("k_used", count(m.k_used))
        .field("pre_standardized", Value::Bool(m.pre_standardized))
        .field("weights", opt(m.weights.as_ref(), col))
        .field("n_eff", Value::F64(m.n_eff))
        .field("rotation_spec", rotation_spec)
        .field("selection_result", selection_result)
        .field("keep", opt(m.keep, count))
}

/// The record passed as `name`, checked to be untagged or tagged `ty`.
pub(crate) fn model_record<'a>(
    v: Value<'a>,
    ty: &str,
    name: &str,
) -> Result<Record<'a>, BindError> {
    match v {
        Value::Record(r) => match r.type_name() {
            None => Ok(r),
            Some(t) if t == ty => Ok(r),
            Some(t) => Err(BindError::invalid_argument(format!(
                "{name} must be a {ty}, got a {t}"
            ))),
        },
        other => Err(BindError::invalid_argument(format!(
            "{name} must be a {ty}, got {}",
            coerce::describe(&other)
        ))),
    }
}

/// A needed, non-null field.
pub(crate) fn need<'r, 'a>(r: &'r Record<'a>, key: &str) -> Result<&'r Value<'a>, BindError> {
    match r.get(key) {
        None => Err(BindError::invalid_argument(format!(
            "model is missing field '{key}'"
        ))),
        Some(Value::Null) => Err(BindError::invalid_argument(format!(
            "model field '{key}' is null"
        ))),
        Some(v) => Ok(v),
    }
}

fn bad(key: &str, why: &str) -> BindError {
    BindError::invalid_argument(format!("model field '{key}' {why}"))
}

pub(crate) fn need_mat(r: &Record<'_>, key: &str) -> Result<Mat<f64>, BindError> {
    match need(r, key)? {
        // Read through the reference: `to_mat` takes its value whole, and
        // cloning an owned matrix to hand it over would copy it twice.
        Value::Mat(m) => Ok(m.as_mat().to_owned()),
        other => coerce::to_mat(other.clone())
            .map(|m| m.as_mat().to_owned())
            .map_err(|w| bad(key, &w)),
    }
}

pub(crate) fn need_col(r: &Record<'_>, key: &str) -> Result<Col<f64>, BindError> {
    let v = coerce::to_vec(need(r, key)?.clone()).map_err(|w| bad(key, &w))?;
    Ok(v.as_col().to_owned())
}

pub(crate) fn need_f64(r: &Record<'_>, key: &str) -> Result<f64, BindError> {
    coerce::to_f64(need(r, key)?).map_err(|w| bad(key, &w))
}

pub(crate) fn need_usize(r: &Record<'_>, key: &str) -> Result<usize, BindError> {
    coerce::to_usize(need(r, key)?).map_err(|w| bad(key, &w))
}

pub(crate) fn need_bool(r: &Record<'_>, key: &str) -> Result<bool, BindError> {
    coerce::to_bool(need(r, key)?).map_err(|w| bad(key, &w))
}

/// A nullable field: absent and null both read as `None`.
pub(crate) fn opt_field<'r, 'a>(r: &'r Record<'a>, key: &str) -> Option<&'r Value<'a>> {
    r.get(key).filter(|v| !v.is_null())
}

pub(crate) fn opt_col(r: &Record<'_>, key: &str) -> Result<Option<Col<f64>>, BindError> {
    opt_field(r, key)
        .map(|v| {
            coerce::to_vec(v.clone())
                .map(|c| c.as_col().to_owned())
                .map_err(|w| bad(key, &w))
        })
        .transpose()
}

pub(crate) fn opt_usize(r: &Record<'_>, key: &str) -> Result<Option<usize>, BindError> {
    opt_field(r, key)
        .map(|v| coerce::to_usize(v).map_err(|w| bad(key, &w)))
        .transpose()
}

/// `Pls1Model` from a `PLS1Result` record.
pub(crate) fn pls1_model_from_record(r: &Record<'_>) -> Result<Pls1Model, BindError> {
    Ok(Pls1Model {
        t_scores: need_mat(r, "T")?,
        p_loadings: need_mat(r, "P")?,
        w_star: need_mat(r, "W")?,
        q_loadings: need_col(r, "Q")?,
        coef: need_col(r, "coef")?,
        beta: need_col(r, "beta")?,
        intercept: need_f64(r, "intercept")?,
        k_used: need_usize(r, "k_used")?,
        pre_standardized: need_bool(r, "pre_standardized")?,
        weights: opt_col(r, "weights")?,
        n_eff: need_f64(r, "n_eff")?,
        keep: opt_usize(r, "keep")?,
    })
}

pub(crate) fn floats(v: Vec<f64>) -> Value<'static> {
    Value::Vec(VecF64::Owned(v))
}

pub(crate) fn ci_scalar(c: &CIScalar) -> Value<'static> {
    Value::Record(
        Record::typed("CIScalar")
            .field("point", Value::F64(c.point))
            .field("lower", Value::F64(c.lower))
            .field("upper", Value::F64(c.upper))
            .field("sd", Value::F64(c.sd)),
    )
}

fn confirmatory_ci(ci: ConfirmatoryCI) -> Value<'static> {
    Value::Record(
        Record::typed("ConfirmatoryCI")
            .field("n_boot", count(ci.n_boot))
            .field("m", count(ci.m))
            .field("m_rate", Value::F64(ci.m_rate))
            .field("level", Value::F64(ci.level))
            .field("beta_sign_z", floats(ci.beta_sign_z))
            .field("beta_sign_z_signed", floats(ci.beta_sign_z_signed))
            .field("leverage_ci_lower", floats(ci.leverage_ci_lower))
            .field("leverage_ci_upper", floats(ci.leverage_ci_upper))
            .field("leverage_se", floats(ci.leverage_se))
            .field("beta_ci_lower", floats(ci.beta_ci_lower))
            .field("beta_ci_upper", floats(ci.beta_ci_upper))
            .field("beta_se", floats(ci.beta_se))
            .field("holdout_corr", ci_scalar(&ci.holdout_corr))
            .field("n_boot_finite", count(ci.n_boot_finite))
            .field(
                "n_boot_finite_holdout_corr",
                count(ci.n_boot_finite_holdout_corr),
            ),
    )
}

/// `ConfirmatoryTestResult`, fields in `_docs/python/results.md` order.
pub(crate) fn confirmatory_test(r: ConfirmatoryTestOutput) -> Record<'static> {
    Record::typed("ConfirmatoryTestResult")
        .field("pvalue", Value::F64(r.pvalue))
        .field("statistic", Value::F64(r.statistic))
        .field("test_method", Value::text(&r.test_method))
        .field("k", count(r.k))
        .field("n_perm", opt(r.n_perm, count))
        .field("n_splits", opt(r.n_splits, count))
        .field("seed", Value::U64(r.seed))
        .field("n_eff", Value::F64(r.n_eff))
        .field("rho_hat", opt(r.rho_hat, Value::F64))
        .field("stable_rank", opt(r.stable_rank, Value::F64))
        .field("ci", opt(r.ci, confirmatory_ci))
}

/// `PermNullResult`. The engine's permutation matrix is row-major
/// `(n_perm, d)`, and is moved into the result without a copy.
pub(crate) fn perm_null(out: PermNullOutput) -> Record<'static> {
    let PermNullOutput {
        n_perm,
        k,
        seed,
        n_eff,
        beta_ref,
        beta_perm_mean,
        beta_perm_sd,
        beta_perm_z,
        beta_perm_matrix,
    } = out;
    let matrix = opt(beta_perm_matrix, |flat| {
        let d = flat.len() / n_perm.max(1);
        Value::Mat(MatF64::OwnedRowMajor {
            data: flat,
            nrows: n_perm,
            ncols: d,
        })
    });
    Record::typed("PermNullResult")
        .field("n_perm", count(n_perm))
        .field("k", count(k))
        .field("seed", Value::U64(seed))
        .field("beta_ref", floats(beta_ref))
        .field("beta_perm_mean", floats(beta_perm_mean))
        .field("beta_perm_sd", floats(beta_perm_sd))
        .field("beta_perm_z", floats(beta_perm_z))
        .field("beta_perm_matrix", matrix)
        .field("n_eff", Value::F64(n_eff))
}

/// `RotationStabilityResult`.
pub(crate) fn rotation_stability(out: RotationStabilityOutput) -> Record<'static> {
    Record::typed("RotationStabilityResult")
        .field("method", Value::text(&out.method))
        .field("n_boot", count(out.n_boot))
        .field("m", count(out.m))
        .field("m_rate", Value::F64(out.m_rate))
        .field("level", Value::F64(out.level))
        .field("seed", Value::U64(out.seed))
        .field("variance_ratio", ci_scalar(&out.variance_ratio))
        .field(
            "variance_ratio_per_axis",
            Value::List(out.variance_ratio_per_axis.iter().map(ci_scalar).collect()),
        )
        .field("variance_unrot", Value::F64(out.variance_unrot))
        .field("variance_rot", Value::F64(out.variance_rot))
        .field(
            "variance_unrot_per_axis",
            floats(out.variance_unrot_per_axis),
        )
        .field("variance_rot_per_axis", floats(out.variance_rot_per_axis))
        .field("degenerate_baseline", Value::Bool(out.degenerate_baseline))
        .field("n_boot_finite", count(out.n_boot_finite))
        .field("n_eff", Value::F64(out.n_eff))
}

pub(crate) fn score_map(m: &BTreeMap<usize, f64>) -> Value<'static> {
    Value::IntMap(
        m.iter()
            .map(|(&k, &v)| (i64::try_from(k).unwrap_or(i64::MAX), v))
            .collect(),
    )
}

/// `FindKOptimalResult`.
pub(crate) fn find_k_optimal(r: &FindKOptimalOutput) -> Record<'static> {
    Record::typed("FindKOptimalResult")
        .field("k_star", count(r.k_star))
        .field("selector", Value::text(&r.selector))
        .field("cv_scores", opt(r.cv_scores.as_ref(), score_map))
        .field("cv_scores_se", opt(r.cv_scores_se.as_ref(), score_map))
        .field("bic_scores", opt(r.bic_scores.as_ref(), score_map))
        .field("pvalues", opt(r.pvalues.as_ref(), col))
        .field("diagnostic", opt(r.diagnostic.as_deref(), Value::text))
        .field("seed", Value::U64(r.seed))
        .field("n_eff", Value::F64(r.n_eff))
        .field("stable_rank", opt(r.stable_rank, Value::F64))
}

/// `FindKSequenceResult`.
pub(crate) fn find_k_sequence(r: &FindKSequenceOutput) -> Record<'static> {
    Record::typed("FindKSequenceResult")
        .field("k_star", count(r.k_star))
        .field("pvalues", col(&r.pvalues))
        .field("test_method", Value::text(&r.test_method))
        .field("alpha", Value::F64(r.alpha))
        .field("seed", Value::U64(r.seed))
        .field("n_eff", Value::F64(r.n_eff))
        .field("stable_rank", opt(r.stable_rank, Value::F64))
}

/// `FindKeepOptimalResult`.
pub(crate) fn find_keep_optimal(r: &FindKeepOptimalOutput) -> Record<'static> {
    Record::typed("FindKeepOptimalResult")
        .field("keep_star", count(r.keep_star))
        .field("k", count(r.k))
        .field("cv_scores", score_map(&r.cv_scores))
        .field("cv_scores_se", score_map(&r.cv_scores_se))
        .field(
            "keep_grid",
            Value::IntVec(
                r.keep_grid
                    .iter()
                    .map(|&g| i64::try_from(g).unwrap_or(i64::MAX))
                    .collect(),
            ),
        )
        .field("seed", Value::U64(r.seed))
        .field("n_eff", Value::F64(r.n_eff))
}

/// `PLS3Result`.
pub(crate) fn pls3_model(m: &Pls3Model) -> Record<'static> {
    Record::typed("PLS3Result")
        .field("U", mat(&m.u_saliences))
        .field("V", mat(&m.v_saliences))
        .field("singular_values", col(&m.singular_values))
        .field("x_scores", mat(&m.x_scores))
        .field("y_scores", mat(&m.y_scores))
        .field("X_mean", col(&m.x_mean))
        .field("X_scale", col(&m.x_scale))
        .field("Y_mean", col(&m.y_mean))
        .field("Y_scale", col(&m.y_scale))
        .field("k_used", count(m.k_used))
        .field("pre_standardized_X", Value::Bool(m.pre_standardized_x))
        .field("pre_standardized_Y", Value::Bool(m.pre_standardized_y))
        .field("keep_X", opt(m.keep_x, count))
        .field("keep_Y", opt(m.keep_y, count))
        .field("converged", opt(m.converged.clone(), Value::BoolVec))
        .field(
            "n_iter",
            opt(m.n_iter.clone(), |v| {
                Value::IntVec(
                    v.into_iter()
                        .map(|n| i64::try_from(n).unwrap_or(i64::MAX))
                        .collect(),
                )
            }),
        )
}

/// `Pls3Model` from a `PLS3Result` record. `converged` / `n_iter` are not
/// read: `pls3_transform`, the only consumer, never uses them.
pub(crate) fn pls3_model_from_record(r: &Record<'_>) -> Result<Pls3Model, BindError> {
    Ok(Pls3Model {
        u_saliences: need_mat(r, "U")?,
        v_saliences: need_mat(r, "V")?,
        singular_values: need_col(r, "singular_values")?,
        x_scores: need_mat(r, "x_scores")?,
        y_scores: need_mat(r, "y_scores")?,
        x_mean: need_col(r, "X_mean")?,
        x_scale: need_col(r, "X_scale")?,
        y_mean: need_col(r, "Y_mean")?,
        y_scale: need_col(r, "Y_scale")?,
        k_used: need_usize(r, "k_used")?,
        pre_standardized_x: need_bool(r, "pre_standardized_X")?,
        pre_standardized_y: need_bool(r, "pre_standardized_Y")?,
        keep_x: opt_usize(r, "keep_X")?,
        keep_y: opt_usize(r, "keep_Y")?,
        converged: None,
        n_iter: None,
    })
}

/// `PLS3Scores`.
pub(crate) fn pls3_scores(s: &Pls3Scores) -> Record<'static> {
    Record::typed("PLS3Scores")
        .field("x_scores", opt(s.x_scores.as_ref(), mat))
        .field("y_scores", opt(s.y_scores.as_ref(), mat))
}
