//! Error type for plskit. All public functions return `PlsKitResult<T>`.

use thiserror::Error;

/// Convenience alias: every fallible public function returns this.
pub type PlsKitResult<T> = Result<T, PlsKitError>;

/// All errors returned by plskit public functions.
#[derive(Debug, Error)]
#[non_exhaustive]
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

    /// Generic invalid-argument error (message describes the problem).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Internal bug — should never reach users in normal usage.
    #[error("internal error: {0}")]
    Internal(String),

    /// Generic invalid-input error (shape, K=0, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Two arrays had incompatible shapes.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Observation weights vector is invalid (negative, all-zero, or too
    /// concentrated for the requested resampling).
    #[error("invalid weights: {reason}")]
    InvalidWeights {
        /// Short `camel_snake` token describing the problem
        /// (e.g. `"negative"`, `"all_zero"`, `"insufficient_effective_n"`,
        /// `"length_mismatch"`).
        reason: &'static str,
    },

    /// Too many resamples failed per-subsample validation, or no draw is left
    /// to reduce after skips and failures, or every draw left has a NaN
    /// holdout correlation (whatever `max_skip_rate` and
    /// `max_failure_rate` are). Carries the raw counts and the
    /// `max_skip_rate` threshold so callers can report context-appropriate
    /// messages.
    #[error(
        "Resampling degenerate: {opening} (`max_skip_rate` threshold={threshold:.3}). Either the \
         skip rate exceeds `max_skip_rate`, or no draw with a holdout correlation is left \
         after skips and failures (every draw left failed or has a NaN holdout \
         correlation), so there is nothing to reduce. When draws do survive many skips, they \
         over-represent rows with high positive weight, so the loop cannot produce \
         an unbiased CI. Likely causes: requested `k` is too large for the \
         effective sample size, the weight vector is concentrated on too few rows, \
         or `m_rate` is too low. Remediation: lower `k`, soften the weight \
         distribution, raise `m_rate`, or, when the skip rate exceeds `max_skip_rate`, \
         raise `max_skip_rate` to accept the truncation knowingly.",
        opening = if *.skipped == 0 {
            format!("no draw with a holdout correlation is left ({} draws, none skipped)", .total)
        } else {
            format!(
                "{}/{} draws (skip_rate={:.3}) failed per-subsample validation",
                .skipped, .total, .skip_rate
            )
        }
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
        /// readouts) returned `Err` other than a weight skip.
        n_worker_failed: usize,
        /// Worker failures plus successful-worker-with-NaN-holdout_corr.
        n_holdout_corr_failed: usize,
        /// Total resamples attempted (== `SubsampleOpts.n_boot`).
        n_boot: usize,
    },

    /// More than half the permutation null statistics are not finite, so the
    /// non-finite-as-exceedance fail-soft rule (`raw_perm`) would saturate p
    /// toward 1 instead of measuring anything. Distinct from
    /// `ResamplingDegenerate`, whose message diagnoses weighted-subsample skips.
    #[error(
        "permutation null degenerate: {failed}/{total} null statistics are not finite. \
         The p-value would be dominated by the conservative non-finite-as-exceedance \
         rule rather than the data. Likely causes: `k` too large for `n`, or a \
         pathological weight vector. Lower `k` or check the inputs."
    )]
    PermNullDegenerate {
        /// Number of permutation null statistics that are not finite.
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

    /// Too few residual degrees of freedom after removing the covariates:
    /// `n − rank([1, Z]) < k + 1`.
    #[error(
        "insufficient residual degrees of freedom: n={n} rows minus rank([1, Z])={rank} \
         leaves fewer than k + 1 = {} for k={k} components. Lower k or pass fewer \
         covariates.",
        k + 1
    )]
    InsufficientResidualDf {
        /// Rows with positive weight (all rows unweighted).
        n: usize,
        /// Rank of `[1, Z]`, the intercept included.
        rank: usize,
        /// Requested number of components.
        k: usize,
    },

    /// A loop of the call does not fit `PLSKIT_MEMORY_BUDGET`: the bytes it
    /// holds across its units plus one unit are above the budget, or a buffer
    /// the call checks alone does not fit. Raised before the loop or that
    /// buffer is allocated, at a nonzero budget only.
    #[error(
        "{env} is {budget} bytes; this call needs at least {needed} bytes. Set it to at \
         least {needed}, or to 0 to run with the least memory.",
        env = crate::fit::MEMORY_BUDGET_ENV
    )]
    MemoryBudget {
        /// The budget in bytes.
        budget: u64,
        /// The bytes the loop needs to run one unit (`u64::MAX` when the count
        /// overflows). A later loop of the same call may need more.
        needed: u64,
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
            Self::InvalidArgument(_) => "invalid_argument",
            Self::Internal(_) => "internal",
            Self::InvalidInput(_) => "invalid_input",
            Self::ShapeMismatch(_) => "shape_mismatch",
            Self::InvalidWeights { .. } => "invalid_weights",
            Self::ResamplingDegenerate { .. } => "resampling_degenerate",
            Self::ResampleFailureRateExceeded { .. } => "resample_failure_rate_exceeded",
            Self::PermNullDegenerate { .. } => "perm_null_degenerate",
            Self::OptimalNoComponent => "optimal_no_component",
            Self::SequenceNoRejection { .. } => "sequence_no_rejection",
            Self::InsufficientResidualDf { .. } => "insufficient_residual_df",
            Self::MemoryBudget { .. } => "memory_budget",
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests {
    use super::*;

    #[test]
    fn resampling_degenerate_opening_names_the_case() {
        let none_skipped = PlsKitError::ResamplingDegenerate {
            skipped: 0,
            total: 100,
            skip_rate: 0.0,
            threshold: 1.0,
        }
        .to_string();
        assert!(
            none_skipped
                .contains("no draw with a holdout correlation is left (100 draws, none skipped)")
        );
        let some_skipped = PlsKitError::ResamplingDegenerate {
            skipped: 40,
            total: 100,
            skip_rate: 0.4,
            threshold: 0.1,
        }
        .to_string();
        assert!(
            some_skipped.contains("40/100 draws (skip_rate=0.400) failed per-subsample validation")
        );
    }

    /// Every variant's `code()` is the string documented in `docs/python/api.md`
    /// ("All codes"); the Python wrapper stamps it on `PlsKitError.code`, so the
    /// strings are a cross-language contract. The index match has no wildcard:
    /// a new variant does not compile until it is given an index, and then fails
    /// until it has a row here (and, by the same review, in the docs).
    #[test]
    #[allow(clippy::too_many_lines)] // one table row per variant
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
            (E::InvalidArgument(s()), "invalid_argument"),
            (E::Internal(s()), "internal"),
            (E::InvalidInput(s()), "invalid_input"),
            (E::ShapeMismatch(s()), "shape_mismatch"),
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
            (
                E::InsufficientResidualDf {
                    n: 3,
                    rank: 2,
                    k: 1,
                },
                "insufficient_residual_df",
            ),
            (
                E::MemoryBudget {
                    budget: 1,
                    needed: 2,
                },
                "memory_budget",
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
                    E::InvalidArgument(_) => 3,
                    E::Internal(_) => 4,
                    E::InvalidInput(_) => 5,
                    E::ShapeMismatch(_) => 6,
                    E::InvalidWeights { .. } => 7,
                    E::ResamplingDegenerate { .. } => 8,
                    E::ResampleFailureRateExceeded { .. } => 9,
                    E::PermNullDegenerate { .. } => 10,
                    E::OptimalNoComponent => 11,
                    E::SequenceNoRejection { .. } => 12,
                    E::InsufficientResidualDf { .. } => 13,
                    E::MemoryBudget { .. } => 14,
                }
            })
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..15).collect::<Vec<_>>(), "one row per variant");
    }

    #[test]
    fn memory_budget_message_names_the_variable_the_budget_and_the_need() {
        let e = PlsKitError::MemoryBudget {
            budget: 1,
            needed: 2,
        };
        assert_eq!(
            e.to_string(),
            "PLSKIT_MEMORY_BUDGET is 1 bytes; this call needs at least 2 bytes. \
             Set it to at least 2, or to 0 to run with the least memory."
        );
    }

    /// The message is all a wrapper's exception carries of this variant's
    /// fields: both rates with their counts, and the threshold, three
    /// decimals each.
    #[test]
    fn resample_failure_rate_message_reports_both_rates() {
        let e = PlsKitError::ResampleFailureRateExceeded {
            max_failure_rate: 0.01,
            observed_worker: 0.0,
            observed_holdout_corr: 0.5,
            n_worker_failed: 0,
            n_holdout_corr_failed: 50,
            n_boot: 100,
        };
        assert_eq!(
            e.to_string(),
            "resample failure rate exceeded: observed_holdout_corr=0.500 (= 50/100), \
             observed_worker=0.000 (= 0/100), max_failure_rate=0.010"
        );
    }
}
