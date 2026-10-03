//! The function registry and the single entry point.

use std::any::Any;
use std::fmt::Write as _;
use std::panic::{self, AssertUnwindSafe};

use crate::error::BindError;
use crate::fns;
use crate::inputs::Inputs;
use crate::types::result_types;
use crate::value::{Outcome, Record, Value};

/// How `call` coerces a parameter's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// 2-D float array.
    Mat,
    /// 1-D float array (a scalar widens).
    Vec,
    /// 1-D or 2-D float array (`preprocess`'s `Y`).
    VecOrMat,
    /// Non-negative whole number.
    Int,
    /// A seed: `u64`, or a decimal string.
    Seed,
    /// A float.
    Float,
    /// A bool.
    Bool,
    /// A string (method names, option values).
    Str,
    /// An integer, or `"optimal"` / `"sequence"` (`pls1_fit`'s `k`).
    KOrMode,
    /// A record of named values (`args`, `rotation_args`, `find_k_args`).
    Args,
    /// A result record passed back (`model`).
    Model,
    /// A `PLS1Result` record or a matrix (`rotate`'s first argument).
    ModelOrMat,
}

impl ParamKind {
    /// Snake-case name used in the registry dump.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mat => "mat",
            Self::Vec => "vec",
            Self::VecOrMat => "vec_or_mat",
            Self::Int => "int",
            Self::Seed => "seed",
            Self::Float => "float",
            Self::Bool => "bool",
            Self::Str => "str",
            Self::KOrMode => "k_or_mode",
            Self::Args => "args",
            Self::Model => "model",
            Self::ModelOrMat => "model_or_mat",
        }
    }
}

/// A parameter's default in the public signature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultValue {
    /// No default: the caller must pass it.
    Required,
    /// `None` / `NULL` / `nothing`.
    Null,
    /// A bool literal.
    Bool(bool),
    /// An integer literal.
    Int(i64),
    /// A string literal.
    Str(&'static str),
}

impl DefaultValue {
    fn to_value(self) -> Value<'static> {
        match self {
            Self::Required | Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(b),
            Self::Int(n) => Value::I64(n),
            Self::Str(s) => Value::text(s),
        }
    }
}

/// One parameter of a public function.
#[derive(Debug, Clone, Copy)]
pub struct Param {
    /// Name, identical across languages.
    pub name: &'static str,
    /// How the value is coerced.
    pub kind: ParamKind,
    /// Default, or [`DefaultValue::Required`].
    pub default: DefaultValue,
    /// Python's keyword-only boundary (`*`): true for parameters after it.
    pub keyword_only: bool,
}

impl Param {
    /// True when the parameter has no default.
    #[must_use]
    pub fn required(&self) -> bool {
        matches!(self.default, DefaultValue::Required)
    }
}

type RunFn = fn(&mut Inputs<'_>) -> Result<Outcome, BindError>;

/// One public function.
pub struct FnSpec {
    /// Name, identical across languages.
    pub name: &'static str,
    /// Parameters in signature order.
    pub params: &'static [Param],
    /// Result type names it can return; empty for a plain array
    /// (`pls1_predict`); two for `rotate`.
    pub result_types: &'static [&'static str],
    run: RunFn,
}

const fn pos(name: &'static str, kind: ParamKind) -> Param {
    Param {
        name,
        kind,
        default: DefaultValue::Required,
        keyword_only: false,
    }
}

const fn pos_d(name: &'static str, kind: ParamKind, default: DefaultValue) -> Param {
    Param {
        name,
        kind,
        default,
        keyword_only: false,
    }
}

const fn kw(name: &'static str, kind: ParamKind, default: DefaultValue) -> Param {
    Param {
        name,
        kind,
        default,
        keyword_only: true,
    }
}

// Kinds are written `K::Mat` rather than imported: a bare `Vec` variant
// would shadow the prelude's `Vec` in this file.
use DefaultValue::{Bool as B, Int as I, Null as N, Str as S};
use ParamKind as K;

static REGISTRY: &[FnSpec] = &[
    FnSpec {
        name: "preprocess",
        params: &[
            pos_d("X", K::Mat, N),
            pos_d("Y", K::VecOrMat, N),
            pos_d("weights", K::Vec, N),
        ],
        result_types: &["PreprocessResult"],
        run: fns::pls1::preprocess,
    },
    FnSpec {
        name: "pls1_fit",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos_d("k", K::KOrMode, I(1)),
            kw("k_max", K::Int, N),
            kw("find_k_args", K::Args, N),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("weights", K::Vec, N),
        ],
        result_types: &["PLS1Result"],
        run: fns::pls1::pls1_fit,
    },
    FnSpec {
        name: "pls1_predict",
        params: &[pos("model", K::Model), pos("X_new", K::Mat)],
        result_types: &[],
        run: fns::pls1::pls1_predict,
    },
    FnSpec {
        name: "pls1_find_k_optimal",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k_max", K::Int),
            kw("selector", K::Str, S("r2_se")),
            kw("diagnostic", K::Str, N),
            kw("args", K::Args, N),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["FindKOptimalResult"],
        run: fns::find_k::pls1_find_k_optimal,
    },
    FnSpec {
        name: "pls1_find_k_sequence",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k_max", K::Int),
            kw("test_method", K::Str, S("auto")),
            kw("alpha", K::Float, N),
            kw("args", K::Args, N),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["FindKSequenceResult"],
        run: fns::find_k::pls1_find_k_sequence,
    },
    FnSpec {
        name: "spls1_fit",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k", K::Int),
            pos("keep", K::Int),
            kw("pre_standardized", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["PLS1Result"],
        run: fns::pls1::spls1_fit,
    },
    FnSpec {
        name: "spls1_find_keep_optimal",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k", K::Int),
            kw("args", K::Args, N),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["FindKeepOptimalResult"],
        run: fns::find_k::spls1_find_keep_optimal,
    },
    FnSpec {
        name: "spls1_find_k_optimal",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k_max", K::Int),
            pos("keep", K::Int),
            kw("selector", K::Str, S("r2_se")),
            kw("diagnostic", K::Str, N),
            kw("args", K::Args, N),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["FindKOptimalResult"],
        run: fns::find_k::spls1_find_k_optimal,
    },
    FnSpec {
        name: "spls1_find_k_sequence",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k_max", K::Int),
            pos("keep", K::Int),
            kw("test_method", K::Str, S("auto")),
            kw("alpha", K::Float, N),
            kw("args", K::Args, N),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["FindKSequenceResult"],
        run: fns::find_k::spls1_find_k_sequence,
    },
    FnSpec {
        name: "pls3_fit",
        params: &[
            pos("X", K::Mat),
            pos("Y", K::Mat),
            pos_d("k", K::Int, I(1)),
            kw("pre_standardized_X", K::Bool, B(false)),
            kw("pre_standardized_Y", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["PLS3Result"],
        run: fns::pls3::pls3_fit,
    },
    FnSpec {
        name: "plssvd_fit",
        params: &[
            pos("X", K::Mat),
            pos("Y", K::Mat),
            pos_d("k", K::Int, I(1)),
            kw("pre_standardized_X", K::Bool, B(false)),
            kw("pre_standardized_Y", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["PLS3Result"],
        run: fns::pls3::pls3_fit,
    },
    FnSpec {
        name: "pls3_transform",
        params: &[
            pos("model", K::Model),
            pos_d("X_new", K::Mat, N),
            pos_d("Y_new", K::Mat, N),
            kw("which", K::Str, S("both")),
        ],
        result_types: &["PLS3Scores"],
        run: fns::pls3::pls3_transform,
    },
    FnSpec {
        name: "plssvd_transform",
        params: &[
            pos("model", K::Model),
            pos_d("X_new", K::Mat, N),
            pos_d("Y_new", K::Mat, N),
            kw("which", K::Str, S("both")),
        ],
        result_types: &["PLS3Scores"],
        run: fns::pls3::pls3_transform,
    },
    FnSpec {
        name: "spls3_fit",
        params: &[
            pos("X", K::Mat),
            pos("Y", K::Mat),
            pos("k", K::Int),
            pos("keep_X", K::Int),
            pos("keep_Y", K::Int),
            kw("pre_standardized_X", K::Bool, B(false)),
            kw("pre_standardized_Y", K::Bool, B(false)),
            kw("max_iter", K::Int, N),
            kw("tol", K::Float, N),
            kw("weights", K::Vec, N),
        ],
        result_types: &["PLS3Result"],
        run: fns::pls3::spls3_fit,
    },
    FnSpec {
        name: "pls1_confirmatory_test",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos_d("k", K::Int, I(1)),
            kw("test_method", K::Str, S("auto")),
            kw("args", K::Args, N),
            kw("ci", K::Bool, B(false)),
            kw("n_boot", K::Int, N),
            kw("m_rate", K::Float, N),
            kw("level", K::Float, N),
            kw("max_failure_rate", K::Float, N),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
            kw("max_skip_rate", K::Float, N),
        ],
        result_types: &["ConfirmatoryTestResult"],
        run: fns::inference::pls1_confirmatory_test,
    },
    FnSpec {
        name: "split_nb_gate",
        params: &[pos("X", K::Mat), kw("weights", K::Vec, N)],
        result_types: &["SplitNbGateResult"],
        run: fns::inference::split_nb_gate,
    },
    FnSpec {
        name: "pls1_perm_null",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k", K::Int),
            kw("n_perm", K::Int, N),
            kw("return_perm_matrix", K::Bool, B(false)),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
        ],
        result_types: &["PermNullResult"],
        run: fns::inference::pls1_perm_null,
    },
    FnSpec {
        name: "pls3_confirmatory_test",
        params: &[
            pos("X", K::Mat),
            pos("Y", K::Mat),
            pos_d("k", K::Int, I(1)),
            kw("test_method", K::Str, S("auto")),
            kw("args", K::Args, N),
            kw("pre_standardized_X", K::Bool, B(false)),
            kw("pre_standardized_Y", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
        ],
        result_types: &["ConfirmatoryTestResult"],
        run: fns::inference::pls3_confirmatory_test,
    },
    FnSpec {
        name: "rotate",
        params: &[
            pos("model_or_W", K::ModelOrMat),
            kw("method", K::Str, S("varimax")),
            kw("L", K::Mat, N),
            kw("args", K::Args, N),
        ],
        result_types: &["RotateResult", "PLS1Result"],
        run: fns::rotate::rotate,
    },
    FnSpec {
        name: "pls1_rotation_stability",
        params: &[
            pos("X", K::Mat),
            pos("y", K::Vec),
            pos("k", K::Int),
            kw("rotation_method", K::Str, S("varimax")),
            kw("rotation_args", K::Args, N),
            kw("L", K::Mat, N),
            kw("n_boot", K::Int, N),
            kw("m_rate", K::Float, N),
            kw("level", K::Float, N),
            kw("pre_standardized", K::Bool, B(false)),
            kw("seed", K::Seed, N),
            kw("verbose", K::Bool, B(false)),
            kw("weights", K::Vec, N),
            kw("max_skip_rate", K::Float, N),
        ],
        result_types: &["RotationStabilityResult"],
        run: fns::inference::pls1_rotation_stability,
    },
];

/// Every public function, in `_docs/python/api.md` order.
#[must_use]
pub fn registry() -> &'static [FnSpec] {
    REGISTRY
}

/// Run a public function. Undeclared keys raise `invalid_argument`,
/// absent optional arguments take their defaults, and a panic anywhere
/// below comes back as an `internal` error.
pub fn call(name: &str, inputs: Record<'_>) -> Result<Outcome, BindError> {
    let spec = REGISTRY
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| BindError::invalid_argument(format!("unknown function '{name}'")))?;
    call_spec(spec, inputs)
}

fn call_spec(spec: &FnSpec, mut inputs: Record<'_>) -> Result<Outcome, BindError> {
    for key in inputs.keys() {
        if !spec.params.iter().any(|p| p.name == key) {
            return Err(BindError::invalid_argument(format!(
                "{}() got an unexpected argument '{key}'",
                spec.name
            )));
        }
    }
    for p in spec.params {
        let present = inputs.get(p.name).map(|v| !v.is_null());
        match (present, p.required()) {
            (Some(true), _) => {}
            (None | Some(false), true) => {
                return Err(BindError::invalid_argument(format!(
                    "{}() missing required argument '{}'",
                    spec.name, p.name
                )));
            }
            (None, false) => inputs.push(p.name, p.default.to_value())?,
            // An explicit Null for an optional parameter means "absent",
            // the same as a missing key: substitute the default in place
            // (a Null default leaves the field Null).
            (Some(false), false) => inputs.set(p.name, p.default.to_value()),
        }
    }
    let mut inp = Inputs::new(spec.name, inputs);
    match panic::catch_unwind(AssertUnwindSafe(|| (spec.run)(&mut inp))) {
        Ok(r) => r,
        Err(payload) => Err(BindError::internal(format!(
            "panic in {}: {}",
            spec.name,
            panic_message(payload.as_ref())
        ))),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", u32::from(c));
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_default(d: DefaultValue) -> String {
    match d {
        DefaultValue::Required | DefaultValue::Null => "null".to_owned(),
        DefaultValue::Bool(b) => b.to_string(),
        DefaultValue::Int(n) => n.to_string(),
        DefaultValue::Str(s) => json_str(s),
    }
}

fn json_list<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
    items.iter().map(f).collect::<Vec<_>>().join(", ")
}

/// The registry and the result types as JSON: the input
/// of the R / Julia stub renderers and of `scripts/check-bind-registry.py`.
/// Hand-written so the crate needs no serde dependency.
#[must_use]
pub fn registry_json() -> String {
    let functions: Vec<String> = REGISTRY
        .iter()
        .map(|f| {
            let params: Vec<String> = f
                .params
                .iter()
                .map(|p| {
                    format!(
                        "        {{\"name\": {}, \"kind\": {}, \"required\": {}, \"keyword_only\": {}, \"default\": {}}}",
                        json_str(p.name),
                        json_str(p.kind.as_str()),
                        p.required(),
                        p.keyword_only,
                        json_default(p.default)
                    )
                })
                .collect();
            format!(
                "    {{\"name\": {}, \"result_types\": [{}], \"params\": [\n{}\n    ]}}",
                json_str(f.name),
                json_list(f.result_types, |t| json_str(t)),
                params.join(",\n")
            )
        })
        .collect();
    let types: Vec<String> = result_types()
        .iter()
        .map(|t| {
            let fields: Vec<String> = t
                .fields
                .iter()
                .map(|f| {
                    format!(
                        "        {{\"name\": {}, \"type\": {}, \"nullable\": {}}}",
                        json_str(f.name),
                        json_str(f.ty),
                        f.nullable
                    )
                })
                .collect();
            format!(
                "    {{\"name\": {}, \"r_class\": {}, \"fields\": [\n{}\n    ]}}",
                json_str(t.name),
                json_str(t.r_class),
                fields.join(",\n")
            )
        })
        .collect();
    format!(
        "{{\n  \"engine_version\": {},\n  \"functions\": [\n{}\n  ],\n  \"result_types\": [\n{}\n  ]\n}}\n",
        json_str(plskit::version()),
        functions.join(",\n"),
        types.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boom(_: &mut Inputs<'_>) -> Result<Outcome, BindError> {
        panic!("kaboom")
    }

    // A panic under call() is an `internal` error.
    #[test]
    fn a_panic_becomes_an_internal_error() {
        let spec = FnSpec {
            name: "boom",
            params: &[],
            result_types: &[],
            run: boom,
        };
        let e = call_spec(&spec, Record::new()).unwrap_err();
        assert_eq!(e.code, "internal");
        assert!(e.message.contains("kaboom"), "{}", e.message);
    }
}
