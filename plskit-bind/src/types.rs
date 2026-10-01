//! Declared shape of every result type: field names in `to_record()`
//! order (which is `_docs/python/results.md` order), a type code per
//! field, and nullability. The registry dump publishes this table;
//! `scripts/render-jl-stubs.py` generates the Julia result structs from it.

use crate::value::{Record, Value};

/// One field of a result type.
#[derive(Debug, Clone, Copy)]
pub struct FieldSpec {
    /// Field name, identical across languages.
    pub name: &'static str,
    /// Type code: `mat`, `vec`, `f64`, `i64`, `u64`, `bool`, `str`,
    /// `intmap`, `intvec`, `boolvec`, `vec_or_mat`, `f64_or_vec`, `args`,
    /// `record:<Type>[|<Type>]`, or `list:<Type>`.
    pub ty: &'static str,
    /// Whether the field may be null.
    pub nullable: bool,
}

/// A result type.
#[derive(Debug, Clone, Copy)]
pub struct ResultTypeSpec {
    /// Python type name, e.g. `PLS1Result`.
    pub name: &'static str,
    /// R S3 class (`snake_case` of the name), e.g. `pls1_result`.
    pub r_class: &'static str,
    /// Fields in `to_record()` order.
    pub fields: &'static [FieldSpec],
}

const fn req(name: &'static str, ty: &'static str) -> FieldSpec {
    FieldSpec {
        name,
        ty,
        nullable: false,
    }
}

const fn opt(name: &'static str, ty: &'static str) -> FieldSpec {
    FieldSpec {
        name,
        ty,
        nullable: true,
    }
}

static RESULT_TYPES: &[ResultTypeSpec] = &[
    ResultTypeSpec {
        name: "PreprocessResult",
        r_class: "preprocess_result",
        fields: &[
            opt("X_std", "mat"),
            opt("X_mean", "vec"),
            opt("X_scale", "vec"),
            opt("Y_std", "vec_or_mat"),
            opt("Y_mean", "f64_or_vec"),
            opt("Y_scale", "f64_or_vec"),
            opt("weights_normalized", "vec"),
            opt("n_eff", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "PLS1Result",
        r_class: "pls1_result",
        fields: &[
            req("T", "mat"),
            req("P", "mat"),
            req("W", "mat"),
            req("Q", "vec"),
            req("coef", "vec"),
            req("beta", "vec"),
            req("intercept", "f64"),
            req("k_used", "i64"),
            req("pre_standardized", "bool"),
            opt("weights", "vec"),
            req("n_eff", "f64"),
            opt("rotation_spec", "record:RotationSpec"),
            opt(
                "selection_result",
                "record:FindKOptimalResult|FindKSequenceResult",
            ),
            opt("keep", "i64"),
        ],
    },
    ResultTypeSpec {
        name: "PLS3Result",
        r_class: "pls3_result",
        fields: &[
            req("U", "mat"),
            req("V", "mat"),
            req("singular_values", "vec"),
            req("x_scores", "mat"),
            req("y_scores", "mat"),
            req("X_mean", "vec"),
            req("X_scale", "vec"),
            req("Y_mean", "vec"),
            req("Y_scale", "vec"),
            req("k_used", "i64"),
            req("pre_standardized_X", "bool"),
            req("pre_standardized_Y", "bool"),
            opt("keep_X", "i64"),
            opt("keep_Y", "i64"),
            opt("converged", "boolvec"),
            opt("n_iter", "intvec"),
        ],
    },
    ResultTypeSpec {
        name: "PLS3Scores",
        r_class: "pls3_scores",
        fields: &[opt("x_scores", "mat"), opt("y_scores", "mat")],
    },
    ResultTypeSpec {
        name: "FindKOptimalResult",
        r_class: "find_k_optimal_result",
        fields: &[
            req("k_star", "i64"),
            req("selector", "str"),
            opt("cv_scores", "intmap"),
            opt("cv_scores_se", "intmap"),
            opt("bic_scores", "intmap"),
            opt("pvalues", "vec"),
            opt("diagnostic", "str"),
            req("seed", "u64"),
            req("n_eff", "f64"),
            opt("stable_rank", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "FindKSequenceResult",
        r_class: "find_k_sequence_result",
        fields: &[
            req("k_star", "i64"),
            req("pvalues", "vec"),
            req("test_method", "str"),
            req("alpha", "f64"),
            req("seed", "u64"),
            req("n_eff", "f64"),
            opt("stable_rank", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "FindKeepOptimalResult",
        r_class: "find_keep_optimal_result",
        fields: &[
            req("keep_star", "i64"),
            req("k", "i64"),
            req("cv_scores", "intmap"),
            req("cv_scores_se", "intmap"),
            req("keep_grid", "intvec"),
            req("seed", "u64"),
            req("n_eff", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "ConfirmatoryTestResult",
        r_class: "confirmatory_test_result",
        fields: &[
            req("pvalue", "f64"),
            req("statistic", "f64"),
            req("test_method", "str"),
            req("k", "i64"),
            opt("n_perm", "i64"),
            opt("n_splits", "i64"),
            req("seed", "u64"),
            req("n_eff", "f64"),
            opt("rho_hat", "f64"),
            opt("stable_rank", "f64"),
            opt("ci", "record:ConfirmatoryCI"),
        ],
    },
    ResultTypeSpec {
        name: "SplitNbGateResult",
        r_class: "split_nb_gate_result",
        fields: &[
            req("fires", "bool"),
            req("stable_rank", "f64"),
            req("n_eff", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "PermNullResult",
        r_class: "perm_null_result",
        fields: &[
            req("n_perm", "i64"),
            req("k", "i64"),
            req("seed", "u64"),
            req("beta_ref", "vec"),
            req("beta_perm_mean", "vec"),
            req("beta_perm_sd", "vec"),
            req("beta_perm_z", "vec"),
            opt("beta_perm_matrix", "mat"),
            req("n_eff", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "CIScalar",
        r_class: "ci_scalar",
        fields: &[
            req("point", "f64"),
            req("lower", "f64"),
            req("upper", "f64"),
            req("sd", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "ConfirmatoryCI",
        r_class: "confirmatory_ci",
        fields: &[
            req("n_boot", "i64"),
            req("m", "i64"),
            req("m_rate", "f64"),
            req("level", "f64"),
            req("beta_sign_z", "vec"),
            req("beta_sign_z_signed", "vec"),
            req("leverage_ci_lower", "vec"),
            req("leverage_ci_upper", "vec"),
            req("leverage_se", "vec"),
            req("beta_ci_lower", "vec"),
            req("beta_ci_upper", "vec"),
            req("beta_se", "vec"),
            req("holdout_corr", "record:CIScalar"),
            req("n_boot_finite", "i64"),
            req("n_boot_finite_holdout_corr", "i64"),
        ],
    },
    ResultTypeSpec {
        name: "RotateResult",
        r_class: "rotate_result",
        fields: &[req("W_rot", "mat"), req("spec", "record:RotationSpec")],
    },
    ResultTypeSpec {
        name: "RotationStabilityResult",
        r_class: "rotation_stability_result",
        fields: &[
            req("method", "str"),
            req("n_boot", "i64"),
            req("m", "i64"),
            req("m_rate", "f64"),
            req("level", "f64"),
            req("seed", "u64"),
            req("variance_ratio", "record:CIScalar"),
            req("variance_ratio_per_axis", "list:CIScalar"),
            req("variance_unrot", "f64"),
            req("variance_rot", "f64"),
            req("variance_unrot_per_axis", "vec"),
            req("variance_rot_per_axis", "vec"),
            req("degenerate_baseline", "bool"),
            req("n_boot_finite", "i64"),
            req("n_eff", "f64"),
        ],
    },
    ResultTypeSpec {
        name: "RotationSpec",
        r_class: "rotation_spec",
        fields: &[
            req("method", "str"),
            req("args", "args"),
            req("R", "mat"),
            req("sweeps", "i64"),
            req("V_converged", "f64"),
            req("L_was_provided", "bool"),
        ],
    },
];

/// Every result type, in a fixed order.
#[must_use]
pub fn result_types() -> &'static [ResultTypeSpec] {
    RESULT_TYPES
}

/// A result type by its Python name.
#[must_use]
pub fn result_type(name: &str) -> Option<&'static ResultTypeSpec> {
    RESULT_TYPES.iter().find(|t| t.name == name)
}

fn conforms(v: &Value<'_>, ty: &str) -> Result<(), String> {
    let ok = match (ty, v) {
        ("mat", Value::Mat(_))
        | ("vec", Value::Vec(_))
        | ("f64", Value::F64(_))
        | ("i64", Value::I64(_))
        | ("u64", Value::U64(_))
        | ("bool", Value::Bool(_))
        | ("str", Value::Str(_))
        | ("intmap", Value::IntMap(_))
        | ("intvec", Value::IntVec(_))
        | ("boolvec", Value::BoolVec(_))
        | ("vec_or_mat", Value::Vec(_) | Value::Mat(_))
        | ("f64_or_vec", Value::F64(_) | Value::Vec(_)) => true,
        ("args", Value::Record(r)) => r.type_name().is_none(),
        (t, Value::Record(r)) if t.starts_with("record:") => {
            let allowed: Vec<&str> = t["record:".len()..].split('|').collect();
            match r.type_name() {
                Some(name) if allowed.contains(&name) => {
                    check_record(r)?;
                    true
                }
                _ => false,
            }
        }
        (t, Value::List(items)) if t.starts_with("list:") => {
            let inner = format!("record:{}", &t["list:".len()..]);
            for item in items {
                conforms(item, &inner)?;
            }
            true
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("expected {ty}, got {}", v.kind_name()))
    }
}

/// Check a result record against its declared shape: a known type tag,
/// exactly the declared fields in order, and each value of its declared
/// type (nested results checked recursively).
pub fn check_record(rec: &Record<'_>) -> Result<(), String> {
    let name = rec.type_name().ok_or("record has no type tag")?;
    let spec = result_type(name).ok_or_else(|| format!("unknown result type {name}"))?;
    let got: Vec<&str> = rec.keys().collect();
    let want: Vec<&str> = spec.fields.iter().map(|f| f.name).collect();
    if got != want {
        return Err(format!("{name}: fields {got:?}, expected {want:?}"));
    }
    for (f, (_, v)) in spec.fields.iter().zip(rec.iter()) {
        if v.is_null() {
            if f.nullable {
                continue;
            }
            return Err(format!("{name}.{}: null but not nullable", f.name));
        }
        conforms(v, f.ty).map_err(|e| format!("{name}.{}: {e}", f.name))?;
    }
    Ok(())
}
