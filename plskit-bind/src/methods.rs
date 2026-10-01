//! Method strings and `args` records, ported from `plskit-py/src/lib.rs`
//! with the same allowed keys, defaults and messages. Absent keys, and
//! keys whose value is null, take the engine's own defaults; no wrapper
//! keeps a copy of them.

use plskit::{
    ConfirmatoryArgs, ConfirmatoryMethod, FindKOptimalOpts, FindKSequenceOpts, FindKeepOptimalOpts,
    Selector, TransformWhich, VarimaxArgs,
};

use crate::coerce;
use crate::error::BindError;
use crate::value::{Record, Value};

/// Reject keys outside `allowed`, with plskit-py's message.
pub(crate) fn validate_keys(
    method: &str,
    args: Option<&Record<'_>>,
    allowed: &[&str],
) -> Result<(), BindError> {
    if let Some(a) = args {
        for key in a.keys() {
            if !allowed.contains(&key) {
                return Err(BindError::invalid_args(format!(
                    "method='{method}' does not accept arg '{key}'; allowed: {allowed:?}"
                )));
            }
        }
    }
    Ok(())
}

/// A non-null value in an `args` record.
pub(crate) fn arg<'r, 'a>(args: Option<&'r Record<'a>>, key: &str) -> Option<&'r Value<'a>> {
    args.and_then(|a| a.get(key)).filter(|v| !v.is_null())
}

fn value_err(key: &str, label: &str, why: &str) -> BindError {
    BindError::invalid_args(format!("args['{key}'] for {label} {why}"))
}

pub(crate) fn arg_usize(
    args: Option<&Record<'_>>,
    label: &str,
    key: &str,
    default: usize,
) -> Result<usize, BindError> {
    arg(args, key).map_or(Ok(default), |v| {
        coerce::to_usize(v).map_err(|w| value_err(key, label, &w))
    })
}

pub(crate) fn arg_f64(
    args: Option<&Record<'_>>,
    label: &str,
    key: &str,
    default: f64,
) -> Result<f64, BindError> {
    arg(args, key).map_or(Ok(default), |v| {
        coerce::to_f64(v).map_err(|w| value_err(key, label, &w))
    })
}

pub(crate) fn arg_bool(
    args: Option<&Record<'_>>,
    label: &str,
    key: &str,
    default: bool,
) -> Result<bool, BindError> {
    arg(args, key).map_or(Ok(default), |v| {
        coerce::to_bool(v).map_err(|w| value_err(key, label, &w))
    })
}

/// A confirmatory method name.
pub(crate) fn parse_method(s: &str) -> Result<ConfirmatoryMethod, BindError> {
    match s {
        "raw_perm" => Ok(ConfirmatoryMethod::RawPerm),
        "split_nb" => Ok(ConfirmatoryMethod::SplitNb),
        "split_exact" => Ok(ConfirmatoryMethod::SplitExact),
        "score" => Ok(ConfirmatoryMethod::Score),
        "e" => Ok(ConfirmatoryMethod::E),
        _ => Err(BindError::invalid_args(format!("unknown method: {s}"))),
    }
}

/// `method` + `args` for `pls1_confirmatory_test`.
pub(crate) fn confirmatory_args(
    method: &str,
    args: Option<&Record<'_>>,
) -> Result<ConfirmatoryArgs, BindError> {
    let m = parse_method(method)?;
    let label = format!("method='{method}'");
    Ok(match (m, ConfirmatoryArgs::defaults_for(m)) {
        (ConfirmatoryMethod::RawPerm, ConfirmatoryArgs::RawPerm { n_perm, n_folds }) => {
            validate_keys(method, args, &["n_perm", "n_folds"])?;
            ConfirmatoryArgs::RawPerm {
                n_perm: arg_usize(args, &label, "n_perm", n_perm)?,
                n_folds: arg_usize(args, &label, "n_folds", n_folds)?,
            }
        }
        (ConfirmatoryMethod::SplitNb, ConfirmatoryArgs::SplitNb { n_splits, force }) => {
            validate_keys(method, args, &["n_splits", "force"])?;
            ConfirmatoryArgs::SplitNb {
                n_splits: arg_usize(args, &label, "n_splits", n_splits)?,
                force: arg_bool(args, &label, "force", force)?,
            }
        }
        (ConfirmatoryMethod::SplitExact, ConfirmatoryArgs::SplitExact { n_perm, n_splits }) => {
            validate_keys(method, args, &["n_perm", "n_splits"])?;
            ConfirmatoryArgs::SplitExact {
                n_perm: arg_usize(args, &label, "n_perm", n_perm)?,
                n_splits: arg_usize(args, &label, "n_splits", n_splits)?,
            }
        }
        (ConfirmatoryMethod::Score, _) => {
            validate_keys(method, args, &[])?;
            ConfirmatoryArgs::Score
        }
        (ConfirmatoryMethod::E, _) => {
            validate_keys(method, args, &[])?;
            ConfirmatoryArgs::E
        }
        (m, d) => {
            return Err(BindError::internal(format!(
                "ConfirmatoryArgs::defaults_for({m:?}) returned {d:?}"
            )))
        }
    })
}

/// `method` + `args` for `pls3_confirmatory_test`: two of the five methods.
pub(crate) fn pls3_confirmatory_args(
    method: &str,
    args: Option<&Record<'_>>,
) -> Result<ConfirmatoryArgs, BindError> {
    if method != "split_exact" && method != "split_nb" {
        return Err(BindError::invalid_args(format!(
            "method='{method}' is not available for pls3_confirmatory_test; \
             allowed: [\"split_exact\", \"split_nb\"]"
        )));
    }
    confirmatory_args(method, args)
}

/// Varimax `args` / `rotation_args`.
pub(crate) fn varimax_args(args: Option<&Record<'_>>) -> Result<VarimaxArgs, BindError> {
    validate_keys("varimax", args, &["max_iter", "tol", "kaiser_normalize"])?;
    let d = VarimaxArgs::default();
    let label = "method='varimax'";
    Ok(VarimaxArgs {
        max_iter: arg_usize(args, label, "max_iter", d.max_iter)?,
        tol: arg_f64(args, label, "tol", d.tol)?,
        kaiser_normalize: arg_bool(args, label, "kaiser_normalize", d.kaiser_normalize)?,
    })
}

/// Flags shared by the resampling entry points.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Common {
    pub(crate) pre_standardized: bool,
    pub(crate) seed: Option<u64>,
    pub(crate) disable_parallelism: bool,
    pub(crate) verbose: bool,
}

/// A K-selection criterion name.
pub(crate) fn selector(s: &str) -> Result<Selector, BindError> {
    match s {
        "r2_se" => Ok(Selector::R2Se),
        "r2_max" => Ok(Selector::R2Max),
        "bic" => Ok(Selector::Bic),
        _ => Err(BindError::invalid_args(format!("unknown selector: {s}"))),
    }
}

/// `selector` / `diagnostic` / `args` for the `*_find_k_optimal` family,
/// with plskit-py's cross-key rules.
pub(crate) fn find_k_optimal_opts(
    selector_name: &str,
    diagnostic: Option<&str>,
    args: Option<&Record<'_>>,
    c: &Common,
) -> Result<FindKOptimalOpts, BindError> {
    let d = FindKOptimalOpts::default();
    let sel = selector(selector_name)?;
    let diag = diagnostic.map(parse_method).transpose()?;
    validate_keys("optimal", args, &["n_folds", "n_perm", "n_splits", "force"])?;
    let label = "method='optimal'";
    let n_folds = arg_usize(args, label, "n_folds", d.n_folds)?;
    if matches!(sel, Selector::Bic) && arg(args, "n_folds").is_some() {
        return Err(BindError::invalid_args(
            "selector='bic' does not accept arg 'n_folds'",
        ));
    }
    // A diagnostic key needs a diagnostic, and one that reads it.
    let needs = |key: &str, ok: &[ConfirmatoryMethod], only: &str| -> Result<(), BindError> {
        if arg(args, key).is_none() {
            return Ok(());
        }
        match diag {
            None => Err(BindError::invalid_args(format!(
                "args['{key}'] requires diagnostic to be set"
            ))),
            Some(m) if !ok.contains(&m) => Err(BindError::invalid_args(format!(
                "args['{key}'] only valid for {only}"
            ))),
            Some(_) => Ok(()),
        }
    };
    needs(
        "n_perm",
        &[ConfirmatoryMethod::RawPerm, ConfirmatoryMethod::SplitExact],
        "diagnostic in {raw_perm, split_exact}",
    )?;
    needs(
        "n_splits",
        &[ConfirmatoryMethod::SplitNb, ConfirmatoryMethod::SplitExact],
        "diagnostic in {split_nb, split_exact}",
    )?;
    needs(
        "force",
        &[ConfirmatoryMethod::SplitNb],
        "diagnostic='split_nb'",
    )?;
    Ok(FindKOptimalOpts {
        selector: sel,
        n_folds,
        diagnostic: diag,
        n_perm: arg_usize(args, label, "n_perm", d.n_perm)?,
        n_splits: arg_usize(args, label, "n_splits", d.n_splits)?,
        force: arg_bool(args, "diagnostic='split_nb'", "force", d.force)?,
        pre_standardized: c.pre_standardized,
        seed: c.seed,
        disable_parallelism: c.disable_parallelism,
        verbose: c.verbose,
    })
}

/// `test_method` / `alpha` / `args` for the `*_find_k_sequence` family.
pub(crate) fn find_k_sequence_opts(
    test_method: &str,
    alpha: Option<f64>,
    args: Option<&Record<'_>>,
    c: &Common,
) -> Result<FindKSequenceOpts, BindError> {
    let d = FindKSequenceOpts::default();
    let tm = parse_method(test_method)?;
    let allowed: &[&str] = match tm {
        ConfirmatoryMethod::RawPerm => &["n_perm"],
        ConfirmatoryMethod::SplitNb => &["n_splits", "force"],
        ConfirmatoryMethod::SplitExact => &["n_perm", "n_splits"],
        ConfirmatoryMethod::E => &[],
        // Score has no per-component reading, so no sequential variant.
        ConfirmatoryMethod::Score => {
            return Err(BindError::invalid_args(format!(
                "test_method='{}' has no sequential variant",
                tm.as_str()
            )));
        }
    };
    validate_keys("sequence", args, allowed)?;
    let label = format!("test_method='{test_method}'");
    Ok(FindKSequenceOpts {
        test_method: tm,
        alpha: alpha.unwrap_or(d.alpha),
        n_perm: arg_usize(args, &label, "n_perm", d.n_perm)?,
        n_splits: arg_usize(args, &label, "n_splits", d.n_splits)?,
        force: arg_bool(args, &label, "force", d.force)?,
        pre_standardized: c.pre_standardized,
        seed: c.seed,
        disable_parallelism: c.disable_parallelism,
        verbose: c.verbose,
    })
}

/// `args` for `spls1_find_keep_optimal`.
pub(crate) fn keep_optimal_opts(
    args: Option<&Record<'_>>,
    seed: Option<u64>,
    disable_parallelism: bool,
    verbose: bool,
) -> Result<FindKeepOptimalOpts, BindError> {
    validate_keys("keep_optimal", args, &["n_folds"])?;
    let d = FindKeepOptimalOpts::default();
    Ok(FindKeepOptimalOpts {
        n_folds: arg_usize(args, "method='keep_optimal'", "n_folds", d.n_folds)?,
        seed,
        disable_parallelism,
        verbose,
    })
}

/// `pls3_transform`'s `which`.
pub(crate) fn which(s: &str) -> Result<TransformWhich, BindError> {
    match s {
        "x_scores" => Ok(TransformWhich::XScores),
        "y_scores" => Ok(TransformWhich::YScores),
        "both" => Ok(TransformWhich::Both),
        _ => Err(BindError::invalid_args(format!(
            "unknown which: {s}; allowed: [\"x_scores\", \"y_scores\", \"both\"]"
        ))),
    }
}

/// `rotate`'s `method` + `args`: varimax is the only method today.
pub(crate) fn rotation_method(
    method: &str,
    args: Option<&Record<'_>>,
) -> Result<VarimaxArgs, BindError> {
    if method != "varimax" {
        return Err(BindError::new(
            "rotation_method_not_implemented",
            format!("rotation method '{method}' is not implemented in this version"),
        ));
    }
    varimax_args(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(fields: &[(&str, Value<'static>)]) -> Record<'static> {
        let mut r = Record::new();
        for (k, v) in fields {
            r.push(k, v.clone()).unwrap();
        }
        r
    }

    #[test]
    fn unknown_key_message_matches_plskit_py() {
        let a = args(&[("n_splits", Value::I64(3))]);
        let e = confirmatory_args("raw_perm", Some(&a)).unwrap_err();
        assert_eq!(e.code, "invalid_args");
        assert_eq!(
            e.message,
            "method='raw_perm' does not accept arg 'n_splits'; allowed: [\"n_perm\", \"n_folds\"]"
        );
    }

    #[test]
    fn absent_and_null_keys_take_engine_defaults() {
        let a = args(&[("n_perm", Value::Null)]);
        let got = confirmatory_args("split_exact", Some(&a)).unwrap();
        let want = ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::SplitExact);
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
    }

    #[test]
    fn values_are_checked() {
        let a = args(&[("n_perm", Value::F64(2.5))]);
        assert_eq!(
            confirmatory_args("split_exact", Some(&a)).unwrap_err().code,
            "invalid_args"
        );
        let a = args(&[("force", Value::I64(1))]);
        assert_eq!(
            confirmatory_args("split_nb", Some(&a)).unwrap_err().code,
            "invalid_args"
        );
        let e = confirmatory_args("split_perm", None).unwrap_err();
        assert_eq!(
            (e.code, e.message.as_str()),
            ("invalid_args", "unknown method: split_perm")
        );
    }

    #[test]
    fn pls3_takes_two_methods() {
        assert!(pls3_confirmatory_args("split_nb", None).is_ok());
        let e = pls3_confirmatory_args("raw_perm", None).unwrap_err();
        assert_eq!(e.code, "invalid_args");
    }

    #[test]
    fn varimax_defaults_are_the_engines() {
        let got = varimax_args(None).unwrap();
        let want = VarimaxArgs::default();
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
        let a = args(&[("bogus", Value::I64(1))]);
        assert_eq!(varimax_args(Some(&a)).unwrap_err().code, "invalid_args");
    }
}
