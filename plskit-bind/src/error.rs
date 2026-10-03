//! The error every wrapper renders: a Python-compatible code, a message,
//! and structured details.

use std::borrow::Cow;
use std::fmt;

use plskit::PlsKitError;

use crate::value::{Record, Value};

/// A failed call.
#[derive(Debug, Clone)]
pub struct BindError {
    /// A `PlsKitError::code()` value, or `invalid_argument` / `invalid_args` /
    /// `internal` from bind itself.
    pub code: &'static str,
    /// Human-readable message (Python's `str(e)`).
    pub message: String,
    /// `reason` for `invalid_weights`; `skipped` / `total` / `skip_rate` /
    /// `threshold` for `resampling_degenerate`; empty otherwise.
    pub details: Record<'static>,
}

impl BindError {
    /// An error with no details.
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: Record::new(),
        }
    }

    /// A bad top-level argument.
    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new("invalid_argument", message)
    }

    /// A bad key or value inside an `args`-style record.
    #[must_use]
    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::new("invalid_args", message)
    }

    /// A bug: a panic or a broken invariant.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("internal", message)
    }
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BindError {}

impl From<PlsKitError> for BindError {
    fn from(e: PlsKitError) -> Self {
        let code = e.code();
        let message = e.to_string();
        let details = match e {
            PlsKitError::InvalidWeights { reason } => {
                Record::new().field("reason", Value::Str(Cow::Borrowed(reason)))
            }
            PlsKitError::ResamplingDegenerate {
                skipped,
                total,
                skip_rate,
                threshold,
            } => Record::new()
                .field(
                    "skipped",
                    Value::I64(i64::try_from(skipped).unwrap_or(i64::MAX)),
                )
                .field(
                    "total",
                    Value::I64(i64::try_from(total).unwrap_or(i64::MAX)),
                )
                .field("skip_rate", Value::F64(skip_rate))
                .field("threshold", Value::F64(threshold)),
            _ => Record::new(),
        };
        Self {
            code,
            message,
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_degenerate_carries_its_counts() {
        let e = BindError::from(PlsKitError::ResamplingDegenerate {
            skipped: 3,
            total: 100,
            skip_rate: 0.03,
            threshold: 0.01,
        });
        assert_eq!(e.code, "resampling_degenerate");
        let keys: Vec<&str> = e.details.keys().collect();
        assert_eq!(keys, ["skipped", "total", "skip_rate", "threshold"]);
        assert!(matches!(e.details.get("skipped"), Some(Value::I64(3))));
    }
}
