//! Confirmatory tests, the `split_nb` gate, permutation null, dual
//! coefficients from a Gram matrix (`pls1_perm_alpha`), and the held-out
//! Gram matrix itself (`gram_matrix`).

use plskit::{
    CIOpts, ConfirmatoryTestInput, ConfirmatoryTestOpts, GramMatrixOutput, PermNullOpts,
    Pls3ConfirmatoryTestOpts,
};

use crate::coerce;
use crate::convert;
use crate::error::BindError;
use crate::fns::{done, engine, pls3_y_value, with_warning};
use crate::inputs::Inputs;
use crate::methods;
use crate::value::{MatF64, Outcome, Record, Value, VecF64};
use crate::warn;

/// The `gram` argument of a `gram=` call, in one of its two forms.
enum GramArg<'a> {
    /// A `GramMatrixResult` passed back.
    Object(GramMatrixOutput),
    /// A raw matrix with its `n_features`.
    Raw(MatF64<'a>, usize),
}

/// Read `gram`: a `GramMatrixResult`, which records `n_features` and how X
/// was scaled, so `n_features` and a set `flag` (the scaling flag's name in
/// this function) are refused with it; or a matrix, which needs
/// `n_features`.
fn gram_arg<'a>(
    gram: Value<'a>,
    n_features: Option<usize>,
    pre_standardized: bool,
    flag: &str,
) -> Result<GramArg<'a>, BindError> {
    match gram {
        Value::Record(r) if r.type_name() == Some("GramMatrixResult") => {
            if n_features.is_some() {
                return Err(BindError::invalid_argument(
                    "n_features is not taken with a GramMatrixResult, which records it",
                ));
            }
            if pre_standardized {
                return Err(BindError::invalid_argument(format!(
                    "{flag} is not taken with a GramMatrixResult, which records how X was scaled"
                )));
            }
            Ok(GramArg::Object(convert::gram_matrix_from_record(&r)?))
        }
        Value::Record(r) => Err(BindError::invalid_argument(format!(
            "gram must be a GramMatrixResult or a matrix, got a {} record",
            r.type_name().unwrap_or("untyped")
        ))),
        v => {
            let m =
                coerce::to_mat(v).map_err(|w| BindError::invalid_argument(format!("gram {w}")))?;
            let n_features = n_features.ok_or_else(|| {
                BindError::invalid_argument(
                    "n_features is required with a gram matrix (a GramMatrixResult records it)",
                )
            })?;
            Ok(GramArg::Raw(m, n_features))
        }
    }
}

/// The engine's record of a `gram=` argument of the tests: the object as
/// passed, or the raw matrix copied with its `n_features` and
/// `pre_standardized`.
fn gram_output(arg: GramArg<'_>, pre_standardized: bool) -> GramMatrixOutput {
    match arg {
        GramArg::Object(g) => g,
        GramArg::Raw(m, n_features) => GramMatrixOutput {
            gram: m.as_mat().to_owned(),
            n_features,
            pre_standardized,
        },
    }
}

/// The `X` / `gram` choice of a test: exactly one is given, and
/// `n_features` and `weights` only with `gram`'s raw form and with `X`.
fn check_x_or_gram(
    has_x: bool,
    gram: &Value<'_>,
    n_features: Option<usize>,
    weights: bool,
) -> Result<(), BindError> {
    match (has_x, gram.is_null()) {
        (true, false) => Err(BindError::invalid_argument("pass X or gram, not both")),
        (false, true) => Err(BindError::invalid_argument("pass X or gram")),
        (true, true) if n_features.is_some() => Err(BindError::invalid_argument(
            "n_features is taken only with a gram matrix, not with X",
        )),
        (false, false) if weights => Err(BindError::invalid_argument(
            "weights do not combine with gram: pass X for a weighted test",
        )),
        _ => Ok(()),
    }
}

/// `fn_name() missing required argument 'name'`, as the registry words it.
fn missing(fn_name: &str, name: &str) -> BindError {
    BindError::invalid_argument(format!("{fn_name}() missing required argument '{name}'"))
}

pub(crate) fn split_nb_gate(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let w = inp.opt_vec("weights")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    inp.finish()?;
    let r = engine(plskit::split_nb_gate(
        x.as_mat(),
        w.as_ref().map(VecF64::as_col),
        pre_standardized,
    ))?;
    Ok(done(
        Record::typed("SplitNbGateResult")
            .field("fires", Value::Bool(r.fires))
            .field("stable_rank", Value::F64(r.stable_rank))
            .field("n_eff", Value::F64(r.n_eff)),
    ))
}

pub(crate) fn gram_matrix(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    inp.finish()?;
    let out = engine(plskit::gram_matrix(x.as_mat(), pre_standardized))?;
    Ok(done(convert::gram_matrix(&out)))
}

/// Refuses the CI knobs passed without `ci`, naming each one passed.
fn refuse_ci_knobs(knobs: &[(&str, bool)]) -> Result<(), BindError> {
    let passed: Vec<&str> = knobs
        .iter()
        .filter(|(_, set)| *set)
        .map(|(name, _)| *name)
        .collect();
    if passed.is_empty() {
        return Ok(());
    }
    let verb = if passed.len() == 1 {
        "applies"
    } else {
        "apply"
    };
    Err(BindError::invalid_argument(format!(
        "{} {verb} only with ci",
        passed.join(", ")
    )))
}

/// `y` is required, and exactly one of `X` and `gram`.
/// `k` is one count or a set of them; `None` is the engine's default set.
/// With `null_y`, `args` takes no `n_perm`: the columns of `null_y` set it.
pub(crate) fn pls1_confirmatory_test(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.opt_mat("X")?;
    let y = inp.opt_vec("y")?;
    let k = inp.opt_usize_set("k")?;
    let test_method = inp.string("test_method")?;
    let args = inp.opt_record("args")?;
    let ci = inp.bool("ci")?;
    let n_boot = inp.opt_usize("n_boot")?;
    let m_rate = inp.opt_f64("m_rate")?;
    let level = inp.opt_f64("level")?;
    let max_failure_rate = inp.opt_f64("max_failure_rate")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let seed = inp.seed()?;
    let w = inp.opt_vec("weights")?;
    let z = inp.opt_covariates()?;
    let null_y = inp.opt_mat("null_y")?;
    let max_skip_rate = inp.opt_f64("max_skip_rate")?;
    let gram = inp.take("gram");
    let n_features = inp.opt_usize("n_features")?;
    inp.finish()?;
    if !ci {
        refuse_ci_knobs(&[
            ("n_boot", n_boot.is_some()),
            ("m_rate", m_rate.is_some()),
            ("level", level.is_some()),
            ("max_failure_rate", max_failure_rate.is_some()),
            ("max_skip_rate", max_skip_rate.is_some()),
        ])?;
    }
    let y = y.ok_or_else(|| missing("pls1_confirmatory_test", "y"))?;
    check_x_or_gram(x.is_some(), &gram, n_features, w.is_some())?;
    if null_y.is_some() && methods::arg(args.as_ref(), "n_perm").is_some() {
        return Err(BindError::invalid_args(
            "args['n_perm'] is not taken with null_y: its columns are the null replicates",
        ));
    }
    let z = z.as_ref().map(MatF64::as_mat);
    let null_y = null_y.as_ref().map(MatF64::as_mat);
    let cd = CIOpts::default();
    let ci_opts = ci.then(|| CIOpts {
        n_boot: n_boot.unwrap_or(cd.n_boot),
        m_rate: m_rate.unwrap_or(cd.m_rate),
        level: level.unwrap_or(cd.level),
        max_failure_rate: max_failure_rate.unwrap_or(cd.max_failure_rate),
    });
    let od = ConfirmatoryTestOpts::default();
    let opts = ConfirmatoryTestOpts {
        args: methods::confirmatory_args(&test_method, args.as_ref())?,
        pre_standardized,
        seed,
        ci: ci_opts,
        max_skip_rate: max_skip_rate.unwrap_or(od.max_skip_rate),
        keep: None,
    };
    let r = if let Some(x) = &x {
        engine(plskit::pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_mat(),
                y: y.as_col(),
                k: k.as_deref(),
                weights: w.as_ref().map(VecF64::as_col),
                covariates: z,
                null_y,
            },
            opts,
        ))?
    } else {
        let arg = gram_arg(gram, n_features, pre_standardized, "pre_standardized")?;
        let g = gram_output(arg, pre_standardized);
        engine(plskit::pls1_confirmatory_test(
            ConfirmatoryTestInput::Gram {
                gram: &g,
                y: y.as_col(),
                k: k.as_deref(),
                covariates: z,
                null_y,
            },
            ConfirmatoryTestOpts {
                pre_standardized: false,
                ..opts
            },
        ))?
    };
    let warning = warn::rerouted(
        Some(&test_method),
        Some(&r.test_method),
        r.n_perm,
        r.stable_rank,
        Some(r.n_eff),
    );
    Ok(with_warning(convert::confirmatory_test(r), warning))
}

/// `Y` is required, and exactly one of `X` and `gram`.
pub(crate) fn pls3_confirmatory_test(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.opt_mat("X")?;
    let y = match inp.take("Y") {
        Value::Null => None,
        v => Some(pls3_y_value(v)?),
    };
    let k = inp.usize("k")?;
    let test_method = inp.string("test_method")?;
    let args = inp.opt_record("args")?;
    let pre_standardized_x = inp.bool("pre_standardized_X")?;
    let pre_standardized_y = inp.bool("pre_standardized_Y")?;
    let seed = inp.seed()?;
    let gram = inp.take("gram");
    let n_features = inp.opt_usize("n_features")?;
    inp.finish()?;
    let y = y.ok_or_else(|| missing("pls3_confirmatory_test", "Y"))?;
    check_x_or_gram(x.is_some(), &gram, n_features, false)?;
    let opts = Pls3ConfirmatoryTestOpts {
        args: methods::pls3_confirmatory_args(&test_method, args.as_ref())?,
        pre_standardized_x,
        pre_standardized_y,
        seed,
        ..Pls3ConfirmatoryTestOpts::default()
    };
    let r = if let Some(x) = &x {
        engine(plskit::pls3_confirmatory_test(
            x.as_mat(),
            y.as_mat(),
            k,
            opts,
        ))?
    } else {
        let arg = gram_arg(gram, n_features, pre_standardized_x, "pre_standardized_X")?;
        let g = gram_output(arg, pre_standardized_x);
        engine(plskit::pls3_confirmatory_test_gram(
            &g,
            y.as_mat(),
            k,
            Pls3ConfirmatoryTestOpts {
                pre_standardized_x: false,
                ..opts
            },
        ))?
    };
    let warning = warn::rerouted(
        Some(&test_method),
        Some(&r.test_method),
        r.n_perm,
        r.stable_rank,
        Some(r.n_eff),
    );
    Ok(with_warning(convert::confirmatory_test(r), warning))
}

pub(crate) fn pls1_perm_null(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.usize("k")?;
    let n_perm = inp.opt_usize("n_perm")?;
    let return_perm_matrix = inp.bool("return_perm_matrix")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let seed = inp.seed()?;
    let w = inp.opt_vec("weights")?;
    let z = inp.opt_covariates()?;
    inp.finish()?;
    let d = PermNullOpts::default();
    let opts = PermNullOpts {
        n_perm: n_perm.unwrap_or(d.n_perm),
        return_perm_matrix,
        pre_standardized,
    };
    let out = engine(plskit::pls1_perm_null(
        x.as_mat(),
        y.as_col(),
        k,
        w.as_ref().map(VecF64::as_col),
        z.as_ref().map(MatF64::as_mat),
        opts,
        seed,
    ))?;
    Ok(done(convert::perm_null(out)))
}

/// `gram` is a `GramMatrixResult` or a matrix with `n_features`.
pub(crate) fn pls1_perm_alpha(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let gram = inp.take("gram");
    let y = inp.mat("Y")?;
    let k = inp.usize("k")?;
    let n_features = inp.opt_usize("n_features")?;
    inp.finish()?;
    // No scaling flag here: the fit does not read how X̃ was scaled.
    let out = match gram_arg(gram, n_features, false, "")? {
        GramArg::Object(g) => engine(plskit::pls1_perm_alpha_gram(&g, y.as_mat(), k))?,
        GramArg::Raw(g, n_features) => engine(plskit::pls1_perm_alpha(
            g.as_mat(),
            y.as_mat(),
            k,
            n_features,
        ))?,
    };
    Ok(done(convert::perm_alpha(out)))
}
