//! Error type for plskit. All public functions return `PlsKitResult<T>`.

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
    ///
    /// Reserved — never emitted by any current production path. Kept so
    /// wrapper error-code enumerations stay stable if a strict-convergence
    /// mode lands later.
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
    #[error("model already has a rotation_spec; re-rotation is not supported")]
    AlreadyRotated,

    /// Observation weights vector is invalid (negative, all-zero, or too
    /// concentrated for the requested resampling).
    #[error("invalid weights: {reason}")]
    InvalidWeights {
        /// Short `camel_snake` token describing the problem
        /// (e.g. `"negative"`, `"all_zero"`, `"insufficient_effective_n"`,
        /// `"length_mismatch"`).
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

    /// More than half the permutation null fits failed, so the NaN-as-exceedance
    /// fail-soft rule (`raw_perm`) would saturate p toward 1 instead of measuring
    /// anything. Distinct from `ResamplingDegenerate`, whose message diagnoses
    /// weighted-subsample skips.
    #[error(
        "permutation null degenerate: {failed}/{total} null fits failed (NaN). \
         The p-value would be dominated by the conservative NaN-as-exceedance \
         rule rather than the data. Likely causes: `k` too large for `n`, or a \
         pathological weight vector. Lower `k` or check the inputs."
    )]
    PermNullDegenerate {
        /// Number of permutation null fits that returned NaN.
        failed: usize,
        /// Total permutations attempted (== `n_perm`).
        total: usize,
    },

    /// A fit at the K `pls1_find_k_optimal` selected was asked for, and the
    /// selection returned `k_star = 0`: the full-data fit cannot extract a
    /// first component. Returned by `FindKOptimalOutput::k_to_fit`, which a
    /// wrapper's `pls1_fit(k = "optimal")` calls.
    #[error(
        "no component to fit: pls1_find_k_optimal returned k_star = 0 because the \
         full-data fit cannot extract a first component (y is constant, or orthogonal \
         to X up to rounding). Pass an explicit integer k to get the k_used = 0 zero \
         model anyway."
    )]
    OptimalNoComponent,

    /// A fit at the K `pls1_find_k_sequence` selected was asked for, and the
    /// sequence rejected no component (`k_star = 0`). Returned by
    /// `FindKSequenceOutput::k_to_fit`, which a wrapper's
    /// `pls1_fit(k = "sequence")` calls.
    #[error(
        "no component to fit: pls1_find_k_sequence rejected no component at alpha \
         {alpha} (all p-values >= alpha). Call pls1_find_k_sequence directly and pass \
         an explicit integer k to fit anyway."
    )]
    SequenceNoRejection {
        /// The significance threshold the sequence ran at.
        alpha: f64,
    },
}

impl From<procrustes::ProcrustesError> for PlsKitError {
    /// Caught one frame up by the resampling workers (Rayon handler maps
    /// `Err → NaN row / Failed`), so this conversion is in practice not
    /// user-visible. The mapping preserves the procrustes message for the
    /// rare cases where it does surface (e.g. direct callers of internal
    /// alignment paths).
    fn from(e: procrustes::ProcrustesError) -> Self {
        match e {
            procrustes::ProcrustesError::NonFinite => Self::NonFiniteInput,
            other => Self::Internal(format!("procrustes alignment failed: {other}")),
        }
    }
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
            Self::PermNullDegenerate { .. } => "perm_null_degenerate",
            Self::OptimalNoComponent => "optimal_no_component",
            Self::SequenceNoRejection { .. } => "sequence_no_rejection",
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests {
    use super::*;

    /// Every variant's `code()` is the string documented in `_docs/python/api.md`
    /// ("All codes"); the Python wrapper stamps it on `PlsKitError.code`, so the
    /// strings are a cross-language contract. The index match has no wildcard:
    /// a new variant does not compile until it is given an index, and then fails
    /// until it has a row here (and, by the same review, in the docs).
    #[test]
    fn every_variant_maps_to_its_documented_code() {
        use PlsKitError as E;
        let s = || String::from("x");
        let table: Vec<(E, &str)> = vec![
            (
                E::DimensionMismatch { x: (1, 1), y: 2 },
                "dimension_mismatch",
            ),
            (E::KExceedsMax { k: 2, k_max: 1 }, "k_exceeds_max"),
            (E::NonFiniteInput, "non_finite_input"),
            (
                E::ConvergenceFailure { iter: 1, tol: 0.1 },
                "convergence_failure",
            ),
            (E::InvalidArgument(s()), "invalid_argument"),
            (E::Internal(s()), "internal"),
            (
                E::RotationMethodNotImplemented { name: s() },
                "rotation_method_not_implemented",
            ),
            (
                E::InvalidArgs {
                    method: s(),
                    detail: s(),
                },
                "invalid_args",
            ),
            (E::InvalidInput(s()), "invalid_input"),
            (E::ShapeMismatch(s()), "shape_mismatch"),
            (E::AlreadyRotated, "already_rotated"),
            (E::InvalidWeights { reason: "negative" }, "invalid_weights"),
            (
                E::ResamplingDegenerate {
                    skipped: 1,
                    total: 2,
                    skip_rate: 0.5,
                    threshold: 0.25,
                },
                "resampling_degenerate",
            ),
            (
                E::ResampleFailureRateExceeded {
                    max_failure_rate: 0.01,
                    observed_worker: 0.0,
                    observed_holdout_corr: 0.5,
                    n_worker_failed: 0,
                    n_holdout_corr_failed: 50,
                    n_boot: 100,
                },
                "resample_failure_rate_exceeded",
            ),
            (
                E::PermNullDegenerate {
                    failed: 1,
                    total: 2,
                },
                "perm_null_degenerate",
            ),
            (E::OptimalNoComponent, "optimal_no_component"),
            (
                E::SequenceNoRejection { alpha: 0.05 },
                "sequence_no_rejection",
            ),
        ];
        let mut seen: Vec<usize> = table
            .iter()
            .map(|(e, code)| {
                assert_eq!(e.code(), *code, "{e:?}");
                match e {
                    E::DimensionMismatch { .. } => 0,
                    E::KExceedsMax { .. } => 1,
                    E::NonFiniteInput => 2,
                    E::ConvergenceFailure { .. } => 3,
                    E::InvalidArgument(_) => 4,
                    E::Internal(_) => 5,
                    E::RotationMethodNotImplemented { .. } => 6,
                    E::InvalidArgs { .. } => 7,
                    E::InvalidInput(_) => 8,
                    E::ShapeMismatch(_) => 9,
                    E::AlreadyRotated => 10,
                    E::InvalidWeights { .. } => 11,
                    E::ResamplingDegenerate { .. } => 12,
                    E::ResampleFailureRateExceeded { .. } => 13,
                    E::PermNullDegenerate { .. } => 14,
                    E::OptimalNoComponent => 15,
                    E::SequenceNoRejection { .. } => 16,
                }
            })
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..17).collect::<Vec<_>>(), "one row per variant");
    }

    #[test]
    fn procrustes_dimension_mismatch_converts_to_internal() {
        // Regression for review-finding R4 (ticket #1, 2026-05-10):
        // procrustes returns DimensionMismatch when a NIPALS short-circuit
        // truncates `w_b`; the From impl must convert (so `?` works at the
        // call sites in subsample.rs / rotation_stability.rs).
        let a = faer::Mat::<f64>::zeros(6, 1);
        let r = faer::Mat::<f64>::from_fn(6, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let pe: PlsKitError = procrustes::orthogonal(a.as_ref(), r.as_ref(), false)
            .unwrap_err()
            .into();
        assert!(matches!(pe, PlsKitError::Internal(_)));
    }

    #[test]
    fn resample_failure_rate_exceeded_message_carries_the_rates() {
        let e = PlsKitError::ResampleFailureRateExceeded {
            max_failure_rate: 0.01,
            observed_worker: 0.0,
            observed_holdout_corr: 0.5,
            n_worker_failed: 0,
            n_holdout_corr_failed: 50,
            n_boot: 100,
        };
        let s = format!("{e}");
        assert!(
            s.contains("0.5") || s.contains("50/100"),
            "missing observed rate in: {s}"
        );
        assert!(s.contains("0.01"), "missing max_failure_rate in: {s}");
    }
}
