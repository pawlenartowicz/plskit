//! One `run` function per public function. Each reads every declared
//! parameter through [`Inputs`], calls `finish`, builds the engine
//! options, calls the engine and converts the output.

pub(crate) mod find_k;
pub(crate) mod inference;
pub(crate) mod pls1;
pub(crate) mod pls3;
pub(crate) mod rotate;

use plskit::PlsKitResult;

use crate::coerce;
use crate::error::BindError;
use crate::inputs::Inputs;
use crate::value::{MatF64, Outcome, Record, Value};

/// An engine result with its error mapped.
pub(crate) fn engine<T>(r: PlsKitResult<T>) -> Result<T, BindError> {
    r.map_err(BindError::from)
}

/// A record result with no warnings.
pub(crate) fn done(rec: Record<'static>) -> Outcome {
    Outcome::new(Value::Record(rec))
}

/// A record result with at most one warning.
pub(crate) fn with_warning(rec: Record<'static>, warning: Option<Record<'static>>) -> Outcome {
    Outcome {
        result: Value::Record(rec),
        warnings: warning.into_iter().collect(),
    }
}

/// PLS3's `Y`: 2-D only, with `_api.py`'s hint when a vector arrives.
pub(crate) fn pls3_y<'a>(inp: &mut Inputs<'a>) -> Result<MatF64<'a>, BindError> {
    match inp.take("Y") {
        Value::Vec(_) | Value::IntVec(_) => Err(BindError::invalid_argument(
            "Y must be 2-D, got 1-D. A single-column outcome is a PLS1 problem — use pls1_fit.",
        )),
        v => coerce::to_mat(v).map_err(|w| BindError::invalid_argument(format!("Y {w}"))),
    }
}
