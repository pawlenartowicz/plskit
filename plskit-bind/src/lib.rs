//! plskit-bind: the language-neutral half of every plskit wrapper.
//!
//! A wrapper converts its native values into a [`Record`] of [`Value`]s,
//! calls [`call`] with a public function name, and converts the
//! [`Outcome`] (or [`BindError`]) back. Everything else a wrapper used to
//! do lives here: method-string parsing, strict `args` validation, engine
//! default resolution, result records, the `split_nb` reroute warning
//! text, `rotate(model)` composition and the error-code mapping.

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![forbid(unsafe_code)]
// Counts cross the boundary as i64 and f64; the casts are range-checked
// where it matters (coerce.rs), and float equality is used only for exact
// whole-number tests.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::many_single_char_names
)]

mod coerce;
mod convert;
mod error;
mod fmt;
mod fns;
mod inputs;
mod methods;
mod registry;
mod types;
mod value;
mod warn;

pub use error::{BindError, ERROR_CODES};
pub use registry::{call, registry, registry_json, DefaultValue, FnSpec, Param, ParamKind};
pub use types::{check_record, result_type, result_types, FieldSpec, ResultTypeSpec};
pub use value::{MatF64, Outcome, Record, Value, VecF64};

/// Version of the `plskit` engine this layer wraps.
#[must_use]
pub fn engine_version() -> &'static str {
    plskit::version()
}
