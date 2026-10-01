//! The error every wrapper renders: a Python-compatible code, a message,
//! and structured details.

use std::borrow::Cow;
use std::fmt;

use plskit::PlsKitError;

use crate::value::{Record, Value};

/// Every code a [`BindError`] can carry: the engine's `PlsKitError::code()`
/// set, which already includes the three bind-level codes
/// (`invalid_argument`, `invalid_args`, `internal`). Identical to the
/// spellings of Python's `PlsKitError.code`.
pub const ERROR_CODES: &[&str] = &[
    "dimension_mismatch",
    "k_exceeds_max",
    "non_finite_input",
    "convergence_failure",
    "invalid_argument",
    "internal",
    "rotation_method_not_implemented",
    "invalid_args",
    "invalid_input",
    "shape_mismatch",
    "already_rotated",
    "invalid_weights",
    "resampling_degenerate",
    "resample_failure_rate_exceeded",
    "perm_null_degenerate",
    "optimal_no_component",
    "sequence_no_rejection",
];

/// A failed call.
#[derive(Debug, Clone)]
pub struct BindError {
    /// One of [`ERROR_CODES`].
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

    /// `{ code, message, details }`, the shape the C ABI hands out.
    #[must_use]
    pub fn into_record(self) -> Record<'static> {
        Record::new()
            .field("code", Value::Str(Cow::Borrowed(self.code)))
            .field("message", Value::Str(Cow::Owned(self.message)))
            .field("details", Value::Record(self.details))
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
    fn invalid_weights_carries_its_reason() {
        let e = BindError::from(PlsKitError::InvalidWeights { reason: "negative" });
        assert_eq!(e.code, "invalid_weights");
        assert!(matches!(e.details.get("reason"), Some(Value::Str(s)) if s == "negative"));
    }

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

    #[test]
    fn every_engine_error_code_is_listed() {
        let all = [
            PlsKitError::DimensionMismatch { x: (1, 1), y: 2 },
            PlsKitError::KExceedsMax { k: 2, k_max: 1 },
            PlsKitError::NonFiniteInput,
            PlsKitError::ConvergenceFailure { iter: 1, tol: 1e-8 },
            PlsKitError::InvalidArgument(String::new()),
            PlsKitError::Internal(String::new()),
            PlsKitError::RotationMethodNotImplemented {
                name: String::new(),
            },
            PlsKitError::InvalidArgs {
                method: String::new(),
                detail: String::new(),
            },
            PlsKitError::InvalidInput(String::new()),
            PlsKitError::ShapeMismatch(String::new()),
            PlsKitError::AlreadyRotated,
            PlsKitError::InvalidWeights { reason: "negative" },
            PlsKitError::ResamplingDegenerate {
                skipped: 0,
                total: 0,
                skip_rate: 0.0,
                threshold: 0.0,
            },
            PlsKitError::ResampleFailureRateExceeded {
                max_failure_rate: 0.0,
                observed_worker: 0.0,
                observed_holdout_corr: 0.0,
                n_worker_failed: 0,
                n_holdout_corr_failed: 0,
                n_boot: 0,
            },
            PlsKitError::PermNullDegenerate {
                failed: 0,
                total: 0,
            },
            PlsKitError::OptimalNoComponent,
            PlsKitError::SequenceNoRejection { alpha: 0.05 },
        ];
        for e in all {
            let code = e.code();
            assert!(
                ERROR_CODES.contains(&code),
                "{code} missing from ERROR_CODES"
            );
        }
        assert_eq!(ERROR_CODES.len(), 17);
    }

    #[test]
    fn error_record_shape() {
        let r = BindError::invalid_args("bad").into_record();
        assert_eq!(r.keys().collect::<Vec<_>>(), ["code", "message", "details"]);
    }
}
