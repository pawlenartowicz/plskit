//! Error type for plskit. All public functions return PlsKitResult<T>.

use thiserror::Error;

/// Convenience alias: every fallible public function returns this.
pub type PlsKitResult<T> = Result<T, PlsKitError>;

/// All errors returned by plskit public functions.
#[derive(Debug, Error)]
pub enum PlsKitError {
    /// X and y have incompatible row counts.
    #[error("dimension mismatch: X has shape {x:?}, y has length {y}")]
    DimensionMismatch {
        /// Shape of X as (`n_rows`, `n_cols`).
        x: (usize, usize),
        /// Length of y.
        y: usize,
    },

    /// Requested k exceeds the maximum supported for this dataset.
    #[error("k={k} exceeds maximum {k_max}")]
    KExceedsMax {
        /// Requested number of components.
        k: usize,
        /// Maximum allowed number of components.
        k_max: usize,
    },

    /// Input matrix or vector contains NaN or infinity.
    #[error("non-finite values in input")]
    NonFiniteInput,

    /// NIPALS loop did not converge within the iteration budget.
    #[error("NIPALS did not converge after {iter} iterations (tol={tol})")]
    ConvergenceFailure {
        /// Number of iterations attempted.
        iter: usize,
        /// Convergence tolerance that was not reached.
        tol: f64,
    },

    /// Generic invalid-argument error (message describes the problem).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Internal bug — should never reach users in normal usage.
    #[error("internal error: {0}")]
    Internal(String),

    /// Requested rotation method is not implemented in this version.
    #[error("rotation method '{name}' is not implemented in this version")]
    RotationMethodNotImplemented {
        /// Method name as received from the caller (e.g. "promax", "geomin").
        name: String,
    },

    /// Method-specific args dict had unknown keys, missing keys, or wrong-typed values.
    #[error("invalid args for method '{method}': {detail}")]
    InvalidArgs {
        /// Method name (e.g. "varimax").
        method: String,
        /// Human-readable explanation of which key/value was wrong.
        detail: String,
    },

    /// Generic invalid-input error (shape, finiteness, K=0, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Two arrays had incompatible shapes.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Caller tried to rotate an already-rotated `PLS1Result`.
    #[error("model already has a rotation_spec; v0.1.1 does not support re-rotation")]
    AlreadyRotated,

    /// Observation weights vector is invalid (negative, all-zero, or too
    /// concentrated for the requested resampling).
    #[error("invalid weights: {reason}")]
    InvalidWeights {
        /// Short `camel_snake` token describing the problem
        /// (e.g. `"negative"`, `"all_zero"`, `"insufficient_effective_n"`).
        reason: &'static str,
    },

    /// Too many resamples failed per-subsample validation. Carries the raw
    /// counts and the `max_skip_rate` threshold so callers can report
    /// context-appropriate messages.
    #[error(
        "Resampling degenerate: {skipped}/{total} draws (skip_rate={skip_rate:.3}) failed \
         per-subsample validation, exceeding `max_skip_rate` (threshold={threshold:.3}). \
         The resampling loop cannot produce an unbiased CI under these conditions \
         — surviving draws over-represent rows with high positive weight. Likely \
         causes: requested `k` is too large for the effective sample size, the \
         weight vector is concentrated on too few rows, or `m_rate` is too low. \
         Remediation: lower `k`, soften the weight distribution, raise `m_rate`, \
         or raise `max_skip_rate` to accept the truncation knowingly."
    )]
    ResamplingDegenerate {
        /// Number of resamples that failed per-subsample validation.
        skipped: usize,
        /// Total number of resamples attempted (== `n_boot`).
        total: usize,
        /// `skipped / total` as f64 for downstream filtering.
        skip_rate: f64,
        /// The `max_skip_rate` threshold that was exceeded.
        threshold: f64,
    },

    /// Per-resample failure rate exceeded `max_failure_rate` for the
    /// confirmatory CI engine. Carries both worker-only and combined
    /// (worker + holdout-NaN) rates so callers can distinguish numerical
    /// failure from data pathology.
    #[error(
        "resample failure rate exceeded: observed_holdout_corr={observed_holdout_corr:.3} \
         (= {n_holdout_corr_failed}/{n_boot}), observed_worker={observed_worker:.3} \
         (= {n_worker_failed}/{n_boot}), max_failure_rate={max_failure_rate:.3}"
    )]
    ResampleFailureRateExceeded {
        /// Caller's threshold (`SubsampleOpts.max_failure_rate`).
        max_failure_rate: f64,
        /// `n_worker_failed / n_boot` — fit / numerical failures only.
        observed_worker: f64,
        /// `n_holdout_corr_failed / n_boot` — combined: workers that failed
        /// plus workers that succeeded but produced NaN `holdout_corr`.
        observed_holdout_corr: f64,
        /// Number of resamples whose worker (full fit + alignment + composite
        /// readouts) returned `Err`.
        n_worker_failed: usize,
        /// Worker failures plus successful-worker-with-NaN-holdout_corr.
        n_holdout_corr_failed: usize,
        /// Total resamples attempted (== `SubsampleOpts.n_boot`).
        n_boot: usize,
    },
}

impl PlsKitError {
    /// Stable string code used by the Python wrapper to populate
    /// `PlsKitError.code`. Variant names in `snake_case`.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::DimensionMismatch { .. } => "dimension_mismatch",
            Self::KExceedsMax { .. } => "k_exceeds_max",
            Self::NonFiniteInput => "non_finite_input",
            Self::ConvergenceFailure { .. } => "convergence_failure",
            Self::InvalidArgument(_) => "invalid_argument",
            Self::Internal(_) => "internal",
            Self::RotationMethodNotImplemented { .. } => "rotation_method_not_implemented",
            Self::InvalidArgs { .. } => "invalid_args",
            Self::InvalidInput(_) => "invalid_input",
            Self::ShapeMismatch(_) => "shape_mismatch",
            Self::AlreadyRotated => "already_rotated",
            Self::InvalidWeights { .. } => "invalid_weights",
            Self::ResamplingDegenerate { .. } => "resampling_degenerate",
            Self::ResampleFailureRateExceeded { .. } => "resample_failure_rate_exceeded",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_weights_variants_render() {
        let e = PlsKitError::InvalidWeights { reason: "negative" };
        assert_eq!(format!("{e}"), "invalid weights: negative");
        let e = PlsKitError::InvalidWeights { reason: "all_zero" };
        assert_eq!(format!("{e}"), "invalid weights: all_zero");
        let e = PlsKitError::InvalidWeights {
            reason: "insufficient_effective_n",
        };
        assert_eq!(format!("{e}"), "invalid weights: insufficient_effective_n");
    }

    #[test]
    fn resampling_degenerate_carries_threshold() {
        let e = PlsKitError::ResamplingDegenerate {
            skipped: 50,
            total: 1000,
            skip_rate: 0.050,
            threshold: 0.250,
        };
        let s = format!("{e}");
        assert!(s.contains("50/1000"), "missing skipped/total");
        assert!(s.contains("0.050"), "missing skip_rate");
        assert!(s.contains("0.250"), "missing threshold");
    }

    #[test]
    fn error_displays_dimension_mismatch() {
        let e = PlsKitError::DimensionMismatch { x: (10, 5), y: 9 };
        let s = format!("{e}");
        assert!(s.contains("10"));
        assert!(s.contains('9'));
    }

    #[test]
    fn rotation_method_not_implemented_code() {
        let e = PlsKitError::RotationMethodNotImplemented {
            name: "promax".into(),
        };
        assert_eq!(e.code(), "rotation_method_not_implemented");
        assert!(format!("{e}").contains("promax"));
    }

    #[test]
    fn already_rotated_code() {
        let e = PlsKitError::AlreadyRotated;
        assert_eq!(e.code(), "already_rotated");
    }

    #[test]
    fn resample_failure_rate_exceeded_code() {
        let e = PlsKitError::ResampleFailureRateExceeded {
            max_failure_rate: 0.01,
            observed_worker: 0.0,
            observed_holdout_corr: 0.5,
            n_worker_failed: 0,
            n_holdout_corr_failed: 50,
            n_boot: 100,
        };
        assert_eq!(e.code(), "resample_failure_rate_exceeded");
        let s = format!("{e}");
        assert!(
            s.contains("0.5") || s.contains("50/100"),
            "missing observed rate in: {s}"
        );
        assert!(s.contains("0.01"), "missing max_failure_rate in: {s}");
    }
}
