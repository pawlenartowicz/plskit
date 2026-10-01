//! The R seam of plskit: `Robj` to `plskit_bind::Value` and back, plus the
//! three `.Call` entry points the R package uses. Everything else (argument
//! validation, method dispatch, result records, error codes) lives in
//! `plskit-bind`.

#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
// `#[extendr]` takes arguments by value and expands to code that binds
// `_`-prefixed names; both lints fire on the macro output, not on ours.
#![allow(clippy::needless_pass_by_value, clippy::used_underscore_binding)]

mod from_r;
mod to_r;

use std::panic::{self, AssertUnwindSafe};

use extendr_api::prelude::*;
use plskit_bind::BindError;

/// Run one public function. Returns `list(ok = TRUE, result, warnings)` or
/// `list(ok = FALSE, code, message, details)`; the R side signals the
/// condition, so no R error ever unwinds through Rust frames.
#[extendr]
fn plskit_call_impl(name: &str, inputs: List) -> Robj {
    let run = panic::catch_unwind(AssertUnwindSafe(|| call_inner(name, &inputs)));
    match run {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => to_r::error(e),
        Err(_) => to_r::error(BindError::internal(format!(
            "panic in the R seam while running {name}"
        ))),
    }
}

fn call_inner(name: &str, inputs: &List) -> std::result::Result<Robj, BindError> {
    let tree = from_r::Inputs::collect(inputs)?;
    let record = tree.to_record(name)?;
    let outcome = plskit_bind::call(name, record)?;
    Ok(to_r::outcome(outcome))
}

/// The registry, error codes and result types as an R list (for tests,
/// `print` and condition classes).
#[extendr]
fn plskit_registry_impl() -> Robj {
    to_r::registry_list()
}

/// Version of the Rust engine compiled into this package.
#[extendr]
fn plskit_engine_version_impl() -> &'static str {
    plskit::version()
}

extendr_module! {
    mod plskit;
    fn plskit_call_impl;
    fn plskit_registry_impl;
    fn plskit_engine_version_impl;
}
