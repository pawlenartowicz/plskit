//! Confirmatory tests, the `split_nb` gate, permutation null, rotation stability.

use plskit::{
    CIOpts, ConfirmatoryTestInput, ConfirmatoryTestOpts, PermNullOpts, Pls3ConfirmatoryTestOpts,
    RotationStabilityMethod, RotationStabilityOpts,
};

use crate::convert;
use crate::error::BindError;
use crate::fns::{done, engine, pls3_y, with_warning};
use crate::inputs::Inputs;
use crate::methods;
use crate::value::{MatF64, Outcome, Record, Value, VecF64};
use crate::warn;

/// `split_nb_gate(X, *, weights=None)`.
pub(crate) fn split_nb_gate(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let r = engine(plskit::split_nb_gate(
        x.as_mat(),
        w.as_ref().map(VecF64::as_col),
    ))?;
    Ok(done(
        Record::typed("SplitNbGateResult")
            .field("fires", Value::Bool(r.fires))
            .field("stable_rank", Value::F64(r.stable_rank))
            .field("n_eff", Value::F64(r.n_eff)),
    ))
}

/// `pls1_confirmatory_test(X, y, k=1, *, test_method="auto", args=None, ci=False,
/// n_boot=None, m_rate=None, level=None, max_failure_rate=None,
/// pre_standardized=False, seed=None,
/// verbose=False, weights=None, max_skip_rate=None)`.
pub(crate) fn pls1_confirmatory_test(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.usize("k")?;
    let test_method = inp.string("test_method")?;
    let args = inp.opt_record("args")?;
    let ci = inp.bool("ci")?;
    let n_boot = inp.opt_usize("n_boot")?;
    let m_rate = inp.opt_f64("m_rate")?;
    let level = inp.opt_f64("level")?;
    let max_failure_rate = inp.opt_f64("max_failure_rate")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let seed = inp.seed()?;
    let verbose = inp.bool("verbose")?;
    let w = inp.opt_vec("weights")?;
    let max_skip_rate = inp.opt_f64("max_skip_rate")?;
    inp.finish()?;
    // With ci = FALSE the four CI knobs are accepted and ignored, as in Python.
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
        verbose,
        ci: ci_opts,
        max_skip_rate: max_skip_rate.unwrap_or(od.max_skip_rate),
        keep: None,
    };
    let r = engine(plskit::pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_mat(),
            y: y.as_col(),
            k,
            weights: w.as_ref().map(VecF64::as_col),
        },
        opts,
    ))?;
    let warning = warn::rerouted(
        Some(&test_method),
        Some(&r.test_method),
        r.n_perm,
        r.stable_rank,
        Some(r.n_eff),
    );
    Ok(with_warning(convert::confirmatory_test(r), warning))
}

/// `pls3_confirmatory_test(X, Y, k=1, *, test_method="auto", args=None,
/// pre_standardized_X=False, pre_standardized_Y=False, seed=None,
/// verbose=False)`.
pub(crate) fn pls3_confirmatory_test(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = pls3_y(inp)?;
    let k = inp.usize("k")?;
    let test_method = inp.string("test_method")?;
    let args = inp.opt_record("args")?;
    let pre_standardized_x = inp.bool("pre_standardized_X")?;
    let pre_standardized_y = inp.bool("pre_standardized_Y")?;
    let seed = inp.seed()?;
    let verbose = inp.bool("verbose")?;
    inp.finish()?;
    let opts = Pls3ConfirmatoryTestOpts {
        args: methods::pls3_confirmatory_args(&test_method, args.as_ref())?,
        pre_standardized_x,
        pre_standardized_y,
        seed,
        verbose,
        ..Pls3ConfirmatoryTestOpts::default()
    };
    let r = engine(plskit::pls3_confirmatory_test(
        x.as_mat(),
        y.as_mat(),
        k,
        opts,
    ))?;
    let warning = warn::rerouted(
        Some(&test_method),
        Some(&r.test_method),
        r.n_perm,
        r.stable_rank,
        Some(r.n_eff),
    );
    let mut rec = convert::confirmatory_test(r);
    rec.set("ci", Value::Null); // this family has no CI, as in _api.py
    Ok(with_warning(rec, warning))
}

/// `pls1_perm_null(X, y, k, *, n_perm=None, return_perm_matrix=False,
/// pre_standardized=False, seed=None,
/// verbose=False, weights=None)`.
pub(crate) fn pls1_perm_null(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.usize("k")?;
    let n_perm = inp.opt_usize("n_perm")?;
    let return_perm_matrix = inp.bool("return_perm_matrix")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let seed = inp.seed()?;
    let verbose = inp.bool("verbose")?;
    let w = inp.opt_vec("weights")?;
    inp.finish()?;
    let d = PermNullOpts::default();
    let opts = PermNullOpts {
        n_perm: n_perm.unwrap_or(d.n_perm),
        return_perm_matrix,
        pre_standardized,
        verbose,
    };
    let out = engine(plskit::pls1_perm_null(
        x.as_mat(),
        y.as_col(),
        k,
        w.as_ref().map(VecF64::as_col),
        opts,
        seed,
    ))?;
    Ok(done(convert::perm_null(out)))
}

/// `pls1_rotation_stability(X, y, k, *, rotation_method="varimax",
/// rotation_args=None, L=None, n_boot=None, m_rate=None, level=None,
/// pre_standardized=False, seed=None,
/// verbose=False, weights=None, max_skip_rate=None)`.
pub(crate) fn pls1_rotation_stability(inp: &mut Inputs<'_>) -> Result<Outcome, BindError> {
    let x = inp.mat("X")?;
    let y = inp.vec("y")?;
    let k = inp.usize("k")?;
    let rotation_method = inp.string("rotation_method")?;
    let rotation_args = inp.opt_record("rotation_args")?;
    let l = inp.opt_mat("L")?;
    let n_boot = inp.opt_usize("n_boot")?;
    let m_rate = inp.opt_f64("m_rate")?;
    let level = inp.opt_f64("level")?;
    let pre_standardized = inp.bool("pre_standardized")?;
    let seed = inp.seed()?;
    let verbose = inp.bool("verbose")?;
    let w = inp.opt_vec("weights")?;
    let max_skip_rate = inp.opt_f64("max_skip_rate")?;
    inp.finish()?;
    if rotation_method != "varimax" {
        return Err(BindError::new(
            "rotation_method_not_implemented",
            format!("rotation_method '{rotation_method}' not implemented in v0.x"),
        ));
    }
    let method = RotationStabilityMethod::Varimax(methods::varimax_args(rotation_args.as_ref())?);
    let d = RotationStabilityOpts::default();
    let opts = RotationStabilityOpts {
        n_boot: n_boot.unwrap_or(d.n_boot),
        m_rate: m_rate.unwrap_or(d.m_rate),
        level: level.unwrap_or(d.level),
        pre_standardized,
        seed,
        verbose,
        max_skip_rate: max_skip_rate.unwrap_or(d.max_skip_rate),
    };
    let out = engine(plskit::pls1_rotation_stability(
        x.as_mat(),
        y.as_col(),
        k,
        method,
        l.as_ref().map(MatF64::as_mat),
        w.as_ref().map(VecF64::as_col),
        opts,
    ))?;
    Ok(done(convert::rotation_stability(out)))
}
