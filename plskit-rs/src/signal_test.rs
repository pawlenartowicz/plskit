//! Confirmatory PLS1 omnibus test at fixed K: five methods.

use faer::{Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::Pls1Model;
// The split-half correlation's degeneracy guard is the same rule
// standardization uses to classify a column as constant.
use crate::linalg::{scaled_moments, ScaledMoments};

/// Which test statistic / resampling method to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmatoryMethod {
    /// Raw permutation CV R² test (`raw_perm`).
    RawPerm,
    /// Split-half NB test with Fisher-z correction (`split_nb`).
    SplitNb,
    /// Permutation-calibrated split-half test (`split_exact`). One test with
    /// two internal routes — a no-refit batched route at K = 1 and an honest
    /// per-permutation refit route otherwise — chosen by the engine from the
    /// input, never by the caller. Both routes report `tanh(z̄)` against a
    /// fixed-split permutation reference.
    SplitExact,
    /// Score test (closed-form, Welch-Satterthwaite χ² approximation).
    Score,
    /// Universal-inference split-LR e-value.
    E,
    /// Pick `split_exact` or `split_nb` from X and the weights (never from
    /// y), once per call, before any randomness is drawn. `split_exact` runs
    /// when the `split_nb` auto-gate fires, when `n_eff` < 250, or when X
    /// has more than 100 columns per row (p > 100·n) on a PLS1 request at
    /// k = 1 without `keep`; `split_nb` runs otherwise. The thresholds may change between versions. A result never reports
    /// `"auto"`: its `test_method` names the method that ran.
    Auto,
}

impl ConfirmatoryMethod {
    /// Public string identifier (`snake_case`) used in result objects and wrapper APIs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ConfirmatoryMethod::RawPerm => "raw_perm",
            ConfirmatoryMethod::SplitNb => "split_nb",
            ConfirmatoryMethod::SplitExact => "split_exact",
            ConfirmatoryMethod::Score => "score",
            ConfirmatoryMethod::E => "e",
            ConfirmatoryMethod::Auto => "auto",
        }
    }
}

/// Input to `pls1_confirmatory_test`: either raw `(X, y, k)` or a pre-fitted model.
pub enum ConfirmatoryTestInput<'a> {
    // TODO: implement; until then, callers must use Raw
    /// Pre-fitted model. Core validates preconditions then returns `Internal`
    /// (wrappers must reassemble raw X+y before calling core).
    #[doc(hidden)]
    Model(&'a Pls1Model),
    /// Raw data with an explicit component count.
    Raw {
        /// Feature matrix `(n_samples, n_features)`.
        x: MatRef<'a, f64>,
        /// Target vector `(n_samples,)`.
        y: ColRef<'a, f64>,
        /// Number of components to test.
        k: usize,
        /// Optional per-observation weights. `None` means uniform weights.
        /// Validated (finite, non-negative, Σw > 0) and normalized to mean 1 before use.
        weights: Option<ColRef<'a, f64>>,
    },
}

/// Method-specific arguments for `pls1_confirmatory_test`.
///
/// The variant chosen *is* the method; per-method knobs live inside the variant.
/// Cross-cutting kwargs (`seed`, `pre_standardized`, …) live on
/// [`ConfirmatoryTestOpts`] alongside this enum.
#[derive(Debug, Clone, Copy)]
pub enum ConfirmatoryArgs {
    /// Raw permutation CV R² test.
    RawPerm {
        /// Number of permutations.
        n_perm: usize,
        /// Number of CV folds. Must be `>= 2` and `< n`: `n_folds >= n` is
        /// leave-one-out (an `n_folds` above `n` leaves empty folds plus
        /// one-row folds, still leave-one-out), where every validation fold
        /// is a single row with zero spread, so the pooled CV R² is
        /// undefined and is rejected with `InvalidArgument` rather than
        /// returned as the degenerate statistic `0` / `p = 1`. `n / 2 <
        /// n_folds < n` is not rejected: some folds hold a single row,
        /// which weakens power but keeps the permutation p-value valid
        /// since the same statistic is used on the nulls.
        n_folds: usize,
    },
    /// Split-half NB test with Fisher-z correction.
    SplitNb {
        /// Number of split-half repetitions.
        n_splits: usize,
        /// Run NB even on a design the auto-gate flags (see
        /// `SPLIT_NB_GATE_MIN_N_EFF`). Default `false`: a flagged design is
        /// rerouted to `split_exact` and the result reports that method.
        force: bool,
    },
    /// Permutation-calibrated split-half test. The engine picks the no-refit
    /// or refit route from `(k, keep)`; there is no route knob.
    SplitExact {
        /// Number of permutations.
        n_perm: usize,
        /// Number of split-half repetitions, drawn once and held fixed across
        /// all permutations.
        n_splits: usize,
    },
    /// Closed-form score test (Welch-Satterthwaite generalized χ²).
    Score,
    /// Universal-inference split-LR e-value.
    E,
    /// `split_exact` or `split_nb`, resolved per design (see
    /// [`ConfirmatoryMethod::Auto`]). Both counts are checked against
    /// `split_exact`'s floors on every design; `n_perm` is unused when the
    /// call resolves to `split_nb`.
    Auto {
        /// Number of permutations if `split_exact` runs.
        n_perm: usize,
        /// Number of split-half repetitions.
        n_splits: usize,
    },
}

impl ConfirmatoryArgs {
    /// The method tag this variant represents (e.g. `"raw_perm"`).
    #[must_use]
    pub fn method(&self) -> ConfirmatoryMethod {
        match self {
            ConfirmatoryArgs::RawPerm { .. } => ConfirmatoryMethod::RawPerm,
            ConfirmatoryArgs::SplitNb { .. } => ConfirmatoryMethod::SplitNb,
            ConfirmatoryArgs::SplitExact { .. } => ConfirmatoryMethod::SplitExact,
            ConfirmatoryArgs::Score => ConfirmatoryMethod::Score,
            ConfirmatoryArgs::E => ConfirmatoryMethod::E,
            ConfirmatoryArgs::Auto { .. } => ConfirmatoryMethod::Auto,
        }
    }

    /// Default args for a given method (used when the caller passes no
    /// method-specific kwargs).
    #[must_use]
    pub fn defaults_for(method: ConfirmatoryMethod) -> Self {
        match method {
            ConfirmatoryMethod::RawPerm => ConfirmatoryArgs::RawPerm {
                n_perm: 1000,
                n_folds: 5,
            },
            ConfirmatoryMethod::SplitNb => ConfirmatoryArgs::SplitNb {
                n_splits: 50,
                force: false,
            },
            ConfirmatoryMethod::SplitExact => ConfirmatoryArgs::SplitExact {
                n_perm: 1000,
                n_splits: 50,
            },
            ConfirmatoryMethod::Score => ConfirmatoryArgs::Score,
            ConfirmatoryMethod::E => ConfirmatoryArgs::E,
            ConfirmatoryMethod::Auto => ConfirmatoryArgs::Auto {
                n_perm: 1000,
                n_splits: 50,
            },
        }
    }
}

/// Sample-size floor of the `split_nb` auto-gate: a design with fewer
/// effective observations than this is rerouted to `split_exact`.
///
/// This and [`SPLIT_NB_GATE_MIN_STABLE_RANK`] are calibrated constants, not
/// round numbers picked by taste. The authors calibrated them on 5,000 null
/// replicates per cell across iid Gaussian, decaying-spectrum,
/// single-factor, NIR gasoline and GloVe-300 designs. The NB correction is
/// exact at ρ = ½ and loses its level when the sample is small or X's
/// spectrum is concentrated on few directions; the pair of thresholds is the
/// smallest rule separating the level-holding cells from the rest, and it
/// deliberately errs toward flagging.
const SPLIT_NB_GATE_MIN_N_EFF: f64 = 25.0;

/// Spectrum floor of the `split_nb` auto-gate, on `linalg::stable_rank` of the
/// standardized X. See [`SPLIT_NB_GATE_MIN_N_EFF`] for the shared provenance.
const SPLIT_NB_GATE_MIN_STABLE_RANK: f64 = 3.0;

/// Column-count precheck of the same gate, derived from
/// [`SPLIT_NB_GATE_MIN_STABLE_RANK`] rather than tuned separately.
/// `stable_rank(A) ≤ A.ncols()` always, so a design with exactly
/// `SPLIT_NB_GATE_MIN_STABLE_RANK` columns can only land ON the floor or
/// under it — the computed value carries no information there — and one
/// column above that it is close to a coin flip on real data. At or below
/// this many columns we stop trusting the computed rank and always reroute.
/// Hence `= SPLIT_NB_GATE_MIN_STABLE_RANK + 1`.
// The cast is exact and const-evaluated: the floor is a small positive whole
// number written as f64 because `stable_rank` returns f64.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const SPLIT_NB_GATE_MAX_COLS_PRECHECK: usize = SPLIT_NB_GATE_MIN_STABLE_RANK as usize + 1;

/// Sample-size floor of `test_method = "auto"`: below this Kish `n_eff` it
/// runs `split_exact`, because `split_nb` gives up power on small samples.
/// The value is also stated in `ConfirmatoryMethod::Auto`'s doc comment and in
/// every doc, doc comment and docstring that states the thresholds (search
/// for `100·n`, `100*n`, `250`, `AUTO_MIN_N_EFF`) — change together.
const AUTO_MIN_N_EFF: f64 = 250.0;

/// Width ceiling of `test_method = "auto"`: with more than this many columns
/// per row it runs `split_exact`, because on very wide X `split_exact`'s
/// no-refit route (K = 1, dense) costs about as much as `split_nb`. Only
/// requests that take that route use the ceiling: on the refit route
/// `split_exact` costs about `n_perm + 1` times `split_nb`. The
/// value is also stated in `ConfirmatoryMethod::Auto`'s doc comment and in
/// every doc, doc comment and docstring that states the thresholds (search
/// for `100·n`, `100*n`, `250`, `AUTO_MAX_P_PER_N`) — change together.
const AUTO_MAX_P_PER_N: usize = 100;

/// Permutation budget of the `split_nb` -> `split_exact` reroute: `split_exact`'s
/// own default, written once. Every reroute site reads it from here; the
/// requested `n_splits` always carries over untouched. Public so the Python
/// wrapper's reroute warning can read the same value through the seam
/// instead of keeping its own copy.
pub const SPLIT_NB_REROUTE_N_PERM: usize = 1000;

/// What the `split_nb` auto-gate decided on one design. Returned by
/// [`resolve_split_nb`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitNbResolution {
    /// The design is flagged (whatever `force` says).
    pub(crate) fires: bool,
    /// `fires && !force`: the caller must run `split_exact` with
    /// [`SPLIT_NB_REROUTE_N_PERM`] permutations instead of `split_nb`.
    pub(crate) reroute: bool,
    /// Stable rank of the standardized X, reported whether or not the gate
    /// fired (and under `force` too).
    pub(crate) stable_rank: f64,
}

impl SplitNbResolution {
    /// The method that runs on a `split_nb` request.
    pub(crate) fn method(self) -> ConfirmatoryMethod {
        if self.reroute {
            ConfirmatoryMethod::SplitExact
        } else {
            ConfirmatoryMethod::SplitNb
        }
    }
}

/// The `split_nb` auto-gate, in the one place it is written. Every site that
/// evaluates it calls this: the public [`split_nb_gate`] query,
/// `pls1_confirmatory_test`, the hoisted sequence gate in
/// `sequential::run_incremental_sequence` (which the `find_k` diagnostic also
/// goes through, including its `k_star = 0` branch), the PLS3 test, and
/// [`resolve_auto`], which every `test_method = "auto"` site calls.
///
/// Inputs are the validated weights, exactly as
/// `fit::validate_and_normalize_weights` returns them: `w_norm` the
/// normalized weights (`None` when absent or all-equal) and `n_eff` its Kish
/// effective sample size (computed from the weights as the caller handed them,
/// and exactly the row count when weights are absent or all-equal). That one `n_eff` is the
/// number every result type reports, so the gate's size test and the
/// reported `n_eff` can never disagree by a rounding step.
///
/// Standardizes its own copy of `x` with `w_norm`, unconditionally, including
/// under `pre_standardized`: re-standardizing an already-standardized matrix
/// is the identity up to fp rounding, and stable rank is scale-invariant on
/// top of that, so the extra sweep cannot move the decision.
pub(crate) fn resolve_split_nb(
    x: MatRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    n_eff: f64,
    force: bool,
) -> SplitNbResolution {
    let (xs, _, _) = crate::linalg::standardize_weighted(x, w_norm);
    // `sr` is computed unconditionally even when the column precheck already
    // decided: it is reported as the gate's `stable_rank` diagnostic either way.
    let sr = crate::linalg::stable_rank(xs.as_ref());
    let narrow = xs.ncols() <= SPLIT_NB_GATE_MAX_COLS_PRECHECK;
    let fires = narrow || n_eff < SPLIT_NB_GATE_MIN_N_EFF || sr < SPLIT_NB_GATE_MIN_STABLE_RANK;
    SplitNbResolution {
        fires,
        reroute: fires && !force,
        stable_rank: sr,
    }
}

/// Resolve `test_method = "auto"` to `split_exact` or `split_nb`, in the one
/// place the rule is written. Returns the method and, when the `split_nb`
/// auto-gate was evaluated, the stable rank it saw.
///
/// `w_norm` and `n_eff` are the validated pair `resolve_split_nb` takes.
/// `no_refit` says whether `split_exact` would take its no-refit route on
/// this request ([`split_exact_no_refit_route`]; always `false` for PLS3,
/// which has none). The clauses that need no spectrum run first: a narrow X
/// (the gate's column precheck), `n_eff < AUTO_MIN_N_EFF` (which contains
/// the gate's own `n_eff` floor) and, when `no_refit`,
/// `p > AUTO_MAX_P_PER_N · n` decide `split_exact` without standardizing X,
/// and the stable rank stays `None`. Skipping the
/// standardized copy and its SVD matters most on the widest designs, where
/// that copy is as large as X. Every other design goes through
/// `resolve_split_nb`, and a fired gate means `split_exact`. Draws no
/// randomness.
pub(crate) fn resolve_auto(
    x: MatRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    n_eff: f64,
    no_refit: bool,
) -> (ConfirmatoryMethod, Option<f64>) {
    if x.ncols() <= SPLIT_NB_GATE_MAX_COLS_PRECHECK
        || n_eff < AUTO_MIN_N_EFF
        || (no_refit && x.ncols() > AUTO_MAX_P_PER_N * x.nrows())
    {
        return (ConfirmatoryMethod::SplitExact, None);
    }
    // `force` only decides the reroute flag, which is not read here.
    let gate = resolve_split_nb(x, w_norm, n_eff, false);
    let method = if gate.fires {
        ConfirmatoryMethod::SplitExact
    } else {
        ConfirmatoryMethod::SplitNb
    };
    (method, Some(gate.stable_rank))
}

/// What the `split_nb` auto-gate sees on a design. Returned by
/// [`split_nb_gate`].
#[derive(Debug, Clone, Copy)]
pub struct SplitNbGateOutput {
    /// `true` when the design is flagged: a `split_nb` request on it reroutes
    /// to `split_exact` unless the caller forces it.
    pub fires: bool,
    /// Stable rank of the standardized X.
    pub stable_rank: f64,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
}

/// Ask whether the `split_nb` auto-gate flags a design, without running a test.
///
/// Reports the same decision `pls1_confirmatory_test` and the `find_k`
/// entry points make internally — this evaluates the one rule, it does not
/// restate it. Standardizes its own copy of `x` (weighted moments when
/// `weights` is given), exactly as the embedded gates do.
///
/// # Errors
/// - `PlsKitError::NonFiniteInput` when X or weights contain NaN/inf
/// - `PlsKitError::InvalidWeights` for length-mismatched, negative, or all-zero weights
///
/// # Panics
/// Never — the finiteness check runs ahead of the SVD in
/// `linalg::stable_rank`, which is the only fallible step.
pub fn split_nb_gate(
    x: MatRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
) -> PlsKitResult<SplitNbGateOutput> {
    crate::fit::with_thread_limit(|| split_nb_gate_impl(x, weights))
}

/// Body of [`split_nb_gate`], on the caller's pool.
pub(crate) fn split_nb_gate_impl(
    x: MatRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
) -> PlsKitResult<SplitNbGateOutput> {
    // Entry-point validation is repeated here rather than inherited: this
    // function runs ahead of any dispatch, and `linalg::stable_rank` expects
    // its SVD to converge. A NaN reaching it either panics or yields a NaN
    // rank that loses the threshold comparison silently — either way the
    // caller would lose the clean `NonFiniteInput`.
    crate::fit::check_finite_mat(x)?;
    // k_requested is only used for a check this function doesn't make (there
    // is no component count in play), so any value passes; 0 is the honest one.
    let (w_norm, n_eff) = crate::fit::validate_and_normalize_weights(weights, x.nrows(), 0)?;
    // `force` only decides whether a flagged design is rerouted; this query
    // reports `fires` alone, so its value is irrelevant here.
    let gate = resolve_split_nb(x, w_norm.as_ref().map(Col::as_ref), n_eff, false);
    Ok(SplitNbGateOutput {
        fires: gate.fires,
        stable_rank: gate.stable_rank,
        n_eff,
    })
}

/// Knobs for the optional CI branch on `pls1_confirmatory_test`. When
/// `ConfirmatoryTestOpts.ci` is `Some`, the function runs an independent
/// subsampling pass (separate child-seed branch from the test pass) and
/// populates `ConfirmatoryTestOutput.ci`.
#[derive(Debug, Clone, Copy)]
pub struct CIOpts {
    /// Number of subsampling resamples. Must be ≥ 100.
    pub n_boot: usize,
    /// Subsample rate: `m = ceil(n^m_rate)`. Must satisfy `0.5 < m_rate < 0.95`.
    pub m_rate: f64,
    /// Nominal CI level (e.g. 0.95). Must satisfy `0.5 ≤ level ≤ 0.99`.
    pub level: f64,
    /// Maximum tolerable combined per-resample failure rate. Default `0.01`.
    /// Range `[0.0, 1.0]`. See `subsample::SubsampleOpts::max_failure_rate`.
    pub max_failure_rate: f64,
}

impl Default for CIOpts {
    fn default() -> Self {
        Self {
            n_boot: 1000,
            m_rate: 0.7,
            level: 0.95,
            max_failure_rate: 0.01,
        }
    }
}

/// Cross-cutting tuning knobs for `pls1_confirmatory_test`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct ConfirmatoryTestOpts {
    /// Method dispatch + per-method args.
    pub args: ConfirmatoryArgs,
    /// Caller asserts X and y are already standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
    /// Optional CI bundle. When `Some`, runs an independent subsampling pass
    /// after the test and populates `ConfirmatoryTestOutput.ci`.
    pub ci: Option<CIOpts>,
    /// Subsample-loop skip threshold for the `ci` branch (resamples skipped by weight validation).
    /// Default `0.01`. The CI loop fails with `ResamplingDegenerate`
    /// if `skipped/total > max_skip_rate`.
    pub max_skip_rate: f64,
    /// Sparse keep-count plumbing for the `spls1` family: every inner fit
    /// (CV folds for `raw_perm`, split halves for `split_nb`/`split_exact`,
    /// the train half for `e`) runs sparse at this keep. `Score` ignores it —
    /// the score statistic `T = ‖X'y‖²` is fit-free. `split_exact`'s no-refit
    /// route also ignores it — it performs no inner fits at all (see "Why no
    /// refits" on `split_perm_nr_zbars`); the route function itself still
    /// guards and errors rather than silently running dense, but a set `keep`
    /// always sends `split_exact`'s dispatch to the refit route instead, so a
    /// caller never observes that guard. `split_exact`'s refit route needs no
    /// such guard: a set `keep` simply routes it to `run_split_perm`, which
    /// honors the keep. `None` (default) = dense. Wrapper surfaces do not
    /// expose this; it exists for `spls1_find_k_sequence`'s per-component
    /// tests (and Rust-level use).
    pub keep: Option<usize>,
}

impl Default for ConfirmatoryTestOpts {
    /// `args` defaults to `Auto` (`split_exact` or `split_nb`, chosen per
    /// design; see [`ConfirmatoryMethod::Auto`]), the same default as
    /// [`Pls3ConfirmatoryTestOpts`](crate::Pls3ConfirmatoryTestOpts) and every
    /// wrapper.
    fn default() -> Self {
        Self {
            args: ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::Auto),
            pre_standardized: false,
            seed: None,
            verbose: false,
            ci: None,
            max_skip_rate: 0.01,
            keep: None,
        }
    }
}

/// Result of `pls1_confirmatory_test`.
#[derive(Debug, Clone)]
pub struct ConfirmatoryTestOutput {
    /// p-value (or `min(1, 1/e)` for the `e` method).
    pub pvalue: f64,
    /// Observed test statistic (CV R² for `raw_perm`; mean Fisher-z back-transformed
    /// (`tanh(z̄)`) for `split_exact` and `split_nb` — one statistic across
    /// both, generally different splits; `||X'y||²` for `score`; log-e for
    /// `e`).
    pub statistic: f64,
    /// Method name as a lowercase string (e.g. `"raw_perm"`, `"split_nb"`, `"e"`).
    pub test_method: String,
    /// Resolved number of components actually tested.
    pub k: usize,
    /// Number of `raw_perm` / `split_exact` iterations used. `None` when the method has no permutation count.
    pub n_perm: Option<usize>,
    /// Number of split-half repetitions used. `None` when the method has no split count.
    pub n_splits: Option<usize>,
    /// RNG seed actually used.
    pub seed: u64,
    /// CI bundle. `Some` when the caller passed `ConfirmatoryTestOpts.ci = Some(...)`.
    pub ci: Option<crate::subsample::ConfirmatoryCI>,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
    /// Estimated split correlation ρ̂, `Some` only on `split_nb` when its ruler
    /// is valid (unweighted, which includes all-equal weights, and
    /// `n_test >= 4`); `None` for every other method, including
    /// `split_exact` (no z-scatter interpretation to offer), and `None` on
    /// `split_nb` itself under non-uniform weights or `n_test < 4`. Unrelated to
    /// `n_eff` (that field is the weights effective-n).
    pub rho_hat: Option<f64>,
    /// Stable rank `‖X_std‖²_F / ‖X_std‖²₂` of the standardized X, as seen by
    /// the `split_nb` auto-gate. `Some` on an explicit `split_nb` request, and
    /// on an `"auto"` request that reached the stable-rank check (p > 4,
    /// `n_eff` ≥ 250, and, on a PLS1 request at k = 1 without `keep`,
    /// p ≤ 100·n); `None` otherwise. On a `split_nb` request it
    /// is set whether the gate fired or not, and also under `force` (the
    /// point of the field is to show what the gate saw).
    pub stable_rank: Option<f64>,
}

/// Confirmatory PLS1 omnibus test at fixed K.
///
/// # Shapes
/// - `input` (Raw form): `x: (n_samples, n_features)`, `y: (n_samples,)`, `k: 1..=k_max`
///
/// # Errors
/// - `PlsKitError::Internal` for the `Model` input form (wrappers must reassemble raw X+y)
/// - `PlsKitError::DimensionMismatch` when row counts disagree
/// - `PlsKitError::KExceedsMax` when k > `d`
/// - `PlsKitError::InvalidArgument` when `k = 0` (every method, `score`
///   included although its statistic does not depend on k)
/// - `PlsKitError::InvalidArgument` for `test_method = "raw_perm"` with
///   `n_folds < 2` or `n_folds >= n` (leave-one-out; the pooled CV R² is
///   undefined when every validation fold is a single row)
/// - `PlsKitError::NonFiniteInput` when X, y, or weights contain NaN/inf
/// - `PlsKitError::InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
pub fn pls1_confirmatory_test(
    input: ConfirmatoryTestInput<'_>,
    opts: ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    crate::fit::with_thread_limit(|| confirmatory_test_impl(input, opts, GateMode::Owned))
}

/// Who owns the `split_nb` auto-gate decision for one confirmatory call.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateMode {
    /// This call evaluates the rule and reroutes if it fires. Every public
    /// caller owns the decision.
    Owned,
    /// The caller already decided, on the undeflated X. Run the requested
    /// method as-is. Only `sequential::run_incremental_sequence` passes this:
    /// a step sees the deflated residual, whose spectrum is not X's, so
    /// re-evaluating there could flip methods mid-chain — which closed testing
    /// cannot use.
    Decided,
}

/// Count floors of `split_exact` (and of `split_nb`'s `n_splits`), shared by
/// every entry point that can run `split_exact` (PLS1 and PLS3 confirmatory
/// tests and the sequence, explicit or through `auto`). `n_perm` is `None`
/// for a `split_nb` request, which has no permutation count.
pub(crate) fn check_split_exact_counts(n_perm: Option<usize>, n_splits: usize) -> PlsKitResult<()> {
    if n_splits < 2 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n_splits must be ≥ 2, got {n_splits}"
        )));
    }
    if let Some(b) = n_perm {
        if b < 1 {
            return Err(PlsKitError::InvalidArgument(format!(
                "n_perm must be ≥ 1, got {b}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_pass_by_value)] // ConfirmatoryTestInput wraps non-Copy Pls1Model ref
pub(crate) fn confirmatory_test_impl(
    input: ConfirmatoryTestInput<'_>,
    opts: ConfirmatoryTestOpts,
    gate: GateMode,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let (x_ref, y_ref, k_resolved, weights_in) = match &input {
        ConfirmatoryTestInput::Raw { x, y, k, weights } => (*x, *y, *k, *weights),
        ConfirmatoryTestInput::Model(_) => {
            // Model form: wrappers must call us with Raw (they hold the original X reference).
            return Err(PlsKitError::Internal(
                "Model form not yet supported in core; wrapper must pass Raw".into(),
            ));
        }
    };

    let n = x_ref.nrows();
    if y_ref.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x_ref.ncols()),
            y: y_ref.nrows(),
        });
    }
    crate::fit::check_finite_mat(x_ref)?;
    crate::fit::check_finite_col(y_ref)?;

    // Per-method count floors. Each method validates only the counts it uses:
    // resampling needs ≥1 permutation, split/CV calibration needs ≥2 of each.
    match opts.args {
        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => {
            if n_folds < 2 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_folds must be ≥ 2, got {n_folds}"
                )));
            }
            // n_folds >= n is leave-one-out: every validation fold is a
            // single row (n_folds > n adds empty folds on top, via
            // linalg::fold_split; an empty fold contributes (0, 0) to the
            // pooled sums, so it does not change this), so that fold's
            // ss_tot (spread of one point about its own mean) is always 0
            // and the pooled CV R² is undefined. Reject rather than return
            // the degenerate statistic 0 / p 1.
            if n_folds >= n {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_folds must be < n (got n_folds={n_folds}, n={n}): n_folds >= n is \
                     leave-one-out (n_folds > n leaves empty folds plus one-row folds, which \
                     is still leave-one-out), where every validation fold is a single row \
                     with zero spread, so the pooled CV R² is undefined"
                )));
            }
            if n_perm < 1 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_perm must be ≥ 1, got {n_perm}"
                )));
            }
        }
        ConfirmatoryArgs::SplitNb { n_splits, .. } => check_split_exact_counts(None, n_splits)?,
        // `Auto` takes split_exact's floors before it resolves, so a bad
        // `n_perm` errors on every design, not only on those that resolve to
        // split_exact.
        ConfirmatoryArgs::SplitExact { n_perm, n_splits }
        | ConfirmatoryArgs::Auto { n_perm, n_splits } => {
            check_split_exact_counts(Some(n_perm), n_splits)?;
        }
        ConfirmatoryArgs::Score | ConfirmatoryArgs::E => {}
    }
    // k = 0 is rejected here, before dispatch, for every method: if it were
    // left to the runners, `raw_perm` and `e` would reject it through
    // `pls1_fit` while `split_exact` / `split_nb` would return a constant p
    // and `score` (which never reads k) a real one. Same error `pls1_fit`
    // raises.
    if k_resolved == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    let k_max = x_ref.ncols();
    if k_resolved > k_max {
        return Err(PlsKitError::KExceedsMax {
            k: k_resolved,
            k_max,
        });
    }

    // Validate + normalize weights. Row-scaling pattern: materialize X̃ = √w' · X_std
    // and ỹ = √w' · y_std and run the unweighted statistics on (X̃, ỹ).
    let (w_norm, n_eff_val) = crate::fit::validate_weights_for_k(weights_in, n, k_resolved)?;

    if let Some(kp) = opts.keep {
        crate::fit::validate_keep(kp, x_ref.ncols())?;
        if opts.ci.is_some() {
            return Err(PlsKitError::InvalidArgument(
                "keep does not combine with ci: per-coordinate subsample CIs under \
                 selection are post-selection inference, which plskit does not implement"
                    .into(),
            ));
        }
    }

    // ── `split_nb` auto-gate ────────────────────────────────────────────────
    // NB's Fisher-z correction is exact at ρ = ½ and drifts off level when the
    // sample is small or X's spectrum is concentrated on few directions. Those
    // designs get rerouted to `split_exact`, which calibrates by permutation
    // and holds its level either way.
    //
    // Resolution happens here, BEFORE dispatch, not inside it: `test_method`,
    // `n_perm` and `n_splits` on the output are read off `args_resolved`, so
    // rewriting the args is exactly what makes `result.test_method` say
    // `"split_exact"` when the gate fired.
    //
    // Under `GateMode::Decided` the whole block is skipped: the caller settled
    // the method already and would only be re-paying for a standardize plus a
    // full SVD (`linalg::stable_rank`) whose answer it discards.
    //
    // `Auto` resolves in the same place and for the same reason: the output
    // reports the method that ran, never `"auto"`.
    let mut args_resolved = opts.args;
    let mut stable_rank_out = None;
    match (gate, opts.args) {
        (GateMode::Owned, ConfirmatoryArgs::SplitNb { n_splits, force }) => {
            let gate = resolve_split_nb(x_ref, w_norm.as_ref().map(Col::as_ref), n_eff_val, force);
            stable_rank_out = Some(gate.stable_rank);
            if gate.reroute {
                args_resolved = ConfirmatoryArgs::SplitExact {
                    n_perm: SPLIT_NB_REROUTE_N_PERM,
                    n_splits,
                };
            }
        }
        (GateMode::Owned, ConfirmatoryArgs::Auto { n_perm, n_splits }) => {
            let (method, sr) = resolve_auto(
                x_ref,
                w_norm.as_ref().map(Col::as_ref),
                n_eff_val,
                split_exact_no_refit_route(k_resolved, opts.keep),
            );
            stable_rank_out = sr;
            args_resolved = if method == ConfirmatoryMethod::SplitNb {
                ConfirmatoryArgs::SplitNb {
                    n_splits,
                    force: false,
                }
            } else {
                ConfirmatoryArgs::SplitExact { n_perm, n_splits }
            };
        }
        _ => {}
    }

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;

    let (result, n_perm_out, n_splits_out) = match args_resolved {
        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => (
            run_raw_perm(
                x_ref,
                y_ref,
                k_resolved,
                n_perm,
                n_folds,
                w_norm.as_ref().map(Col::as_ref),
                &opts,
                &mut rng,
            )?,
            Some(n_perm),
            None,
        ),
        ConfirmatoryArgs::SplitNb { n_splits, .. } => (
            run_split_nb(
                x_ref,
                y_ref,
                k_resolved,
                n_splits,
                w_norm.as_ref().map(Col::as_ref),
                &opts,
                &mut rng,
            )?,
            None,
            Some(n_splits),
        ),
        ConfirmatoryArgs::SplitExact { n_perm, n_splits } => {
            // One test, two internal routes; the caller never picks. The
            // no-refit route is an exact shortcut only under the K = 1
            // identity: `pls1_fit` returns `w ∝ +X̃_tr'y_tr` with `p'w = 1`, so
            // the test-half score is a fixed linear map of y determined by X
            // and the split alone, and permuting y only re-feeds that map.
            // K ≥ 2 deflation breaks it (component 2's weights depend on
            // component 1's y-dependent scores), and sparse `keep` breaks it
            // even at K = 1 (the selected column set moves with every
            // permutation). Weights do not break it: under the √w row-scaling
            // convention (`linalg::sqrt_col`) the map just carries a
            // `diag(√w_tr)` inside it and stays fixed and linear in y, so
            // weighted K = 1 dense input takes the no-refit route too (see
            // `split_perm_nr_zbars`, "Under weights").
            //
            // Route decided here, before any draw from `rng`: the no-refit
            // route consumes the parent rng through its own sequential draws
            // and so must receive it untouched; the refit runner draws its
            // splits, then its seeds, from it.
            let route = split_exact_refit_route(
                n,
                x_ref.ncols(),
                n_perm,
                k_resolved,
                opts.keep,
                w_norm.is_some(),
            );
            let wr = w_norm.as_ref().map(Col::as_ref);
            let result = match route {
                ReplicateRoute::Special => run_split_perm_nr(
                    x_ref, y_ref, k_resolved, n_perm, n_splits, wr, &opts, &mut rng,
                )?,
                route @ (ReplicateRoute::Primal
                | ReplicateRoute::Nspace
                | ReplicateRoute::GramP) => run_split_perm(
                    route, x_ref, y_ref, k_resolved, n_perm, n_splits, wr, &opts, &mut rng,
                )?,
            };
            (result, Some(n_perm), Some(n_splits))
        }
        ConfirmatoryArgs::Score => (
            run_score(x_ref, y_ref, w_norm.as_ref().map(Col::as_ref), &opts)?,
            None,
            None,
        ),
        ConfirmatoryArgs::E => (
            run_e(
                x_ref,
                y_ref,
                k_resolved,
                w_norm.as_ref().map(Col::as_ref),
                &opts,
                &mut rng,
            )?,
            None,
            None,
        ),
        // Resolved above under `GateMode::Owned`; a `Decided` caller passes
        // the method it already resolved.
        ConfirmatoryArgs::Auto { .. } => {
            return Err(PlsKitError::Internal(
                "test_method='auto' reached dispatch unresolved".into(),
            ))
        }
    };

    let ci_payload = if let Some(ci_opts) = opts.ci {
        let sub_opts = crate::subsample::SubsampleOpts {
            n_boot: ci_opts.n_boot,
            m_rate: ci_opts.m_rate,
            level: ci_opts.level,
            pre_standardized: opts.pre_standardized,
            max_failure_rate: ci_opts.max_failure_rate,
            max_skip_rate: opts.max_skip_rate,
        };
        sub_opts.validate()?;

        // Independent child-seed branch — derive a second child RNG from the
        // post-test parent state. This guarantees stream non-interference
        // between test path and CI path while keeping a single user-facing seed.
        let mut ci_rng = {
            use rand::Rng;
            crate::rng::child_rng(rng.next_u64())
        };

        // Reference fit on full data.
        let fit_ref = {
            use crate::fit::{pls1_fit_impl, FitOpts, KSpec};
            pls1_fit_impl(
                x_ref,
                y_ref,
                KSpec::Fixed(k_resolved),
                w_norm.as_ref().map(Col::as_ref),
                FitOpts {
                    pre_standardized: opts.pre_standardized,
                    ..FitOpts::default()
                },
            )?
        };

        // leverage_ref[j] = diag(W_ref (W_ref' W_ref)^-1 W_ref').
        let leverage_ref = crate::linalg::leverage_diag(fit_ref.w_star.as_ref());
        Some(crate::subsample::pls1_subsample_inference_confirmatory(
            x_ref,
            y_ref,
            k_resolved,
            fit_ref.w_star.as_ref(),
            fit_ref.beta.as_ref(),
            &leverage_ref,
            sub_opts,
            w_norm.as_ref().map(Col::as_ref),
            &mut ci_rng,
        )?)
    } else {
        None
    };

    Ok(ConfirmatoryTestOutput {
        pvalue: result.pvalue,
        statistic: result.statistic,
        test_method: args_resolved.method().as_str().to_owned(),
        k: k_resolved,
        n_perm: n_perm_out,
        n_splits: n_splits_out,
        seed: seed_used,
        ci: ci_payload,
        n_eff: n_eff_val,
        rho_hat: result.rho_hat,
        stable_rank: stable_rank_out,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal result carrier (not pub)
// ──────────────────────────────────────────────────────────────────────────────

struct RunResult {
    pvalue: f64,
    statistic: f64,
    /// Estimated split correlation ρ̂. `Some` only on the `split_nb` path,
    /// and only when its ruler is valid (unweighted, `n_test >= 4`).
    rho_hat: Option<f64>,
}

/// `run_raw_perm`'s route rule. The K = 1 Gram closed form
/// (`dual_route::pls1_cv_r2_columns`) needs a fixed training set reused
/// across replicates, which K-fold gives, plus the K = 1 coefficient
/// identity: deflation forecloses K ≥ 2, a sparse `keep` moves the selected
/// column set every replicate, and the weighted case is unverified, so all
/// of those stay primal. `n_tr_max` is the largest training fold
/// (`fold_split`'s groups differ by at most one), so the memory side of the
/// rule sees the worst case and the whole statistic runs on one route.
/// Decided from shape alone; internal and silent.
fn raw_perm_k1_gram_route(
    n: usize,
    n_folds: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    dense: bool,
    weighted: bool,
) -> bool {
    let n_tr_max = n - n / n_folds;
    k == 1 && dense && !weighted && crate::dual_route::use_dual_route(n_tr_max, p, n_perm + 1, 1)
}

/// `split_exact`'s route rule: the no-refit route is an exact shortcut only
/// under the K = 1 identity on a dense fit (see the dispatch in
/// `confirmatory_test_impl`); everything else refits.
pub(crate) fn split_exact_no_refit_route(k: usize, keep: Option<usize>) -> bool {
    k == 1 && keep.is_none()
}

thread_local! {
    /// Set by `with_gram_routes_disabled` (tests); read through `gram_routes_disabled`.
    static GRAM_ROUTES_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the replicate-loop route selectors are restricted to
/// `ReplicateRoute::Special` and `ReplicateRoute::Primal`, which takes the
/// Gram routes `ReplicateRoute::Nspace` and `ReplicateRoute::GramP` out of
/// the choice. Always `false` outside the test-only
/// `with_gram_routes_disabled`. Thread-local: a selector runs on the
/// calling thread before any parallel work, so a test's override reaches
/// exactly the calls that test makes. The override never forces a route
/// onto an input the route's rule rejects; it only removes `Nspace` and
/// `GramP` from the choice. With `PLSKIT_NUM_THREADS` set to a cap, a
/// public call runs on a plskit pool worker, which this thread-local does
/// not reach, so `with_gram_routes_disabled` panics when the variable sets
/// a cap.
pub(crate) fn gram_routes_disabled() -> bool {
    GRAM_ROUTES_DISABLED.with(std::cell::Cell::get)
}

/// Run `f` with [`gram_routes_disabled`] true on this thread, restoring the
/// previous value afterwards, also when `f` panics, so nested and failing
/// tests cannot leak the override. Route-invisibility tests and the layout
/// table run primal-only arms inside it.
#[cfg(test)]
pub(crate) fn with_gram_routes_disabled<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            GRAM_ROUTES_DISABLED.with(|c| c.set(self.0));
        }
    }
    assert!(
        std::env::var_os(crate::fit::NUM_THREADS_ENV).is_none_or(|v| v == "0"),
        "unset PLSKIT_NUM_THREADS or set it to 0: the override does not reach plskit's pool"
    );
    let _restore = Restore(GRAM_ROUTES_DISABLED.with(|c| c.replace(true)));
    f()
}

/// Route of one replicate loop, decided once per call by the site's
/// selector on the calling thread, before any parallel work, from the
/// input's shape alone. Internal and silent: every route computes the same
/// statistic, and the choice is not observable on the public surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplicateRoute {
    /// The K = 1 Gram shortcuts that run outside the drivers: `raw_perm`'s
    /// closed form (`dual_route::pls1_cv_r2_columns`) and `split_exact`'s
    /// no-refit route (`run_split_perm_nr`). The runner runs it itself; a
    /// driver handed it computes the primal statistic.
    Special,
    /// The primal (X-side) per-unit bodies.
    Primal,
    /// n-space Gram route (`dual_route::pls1_nspace_kernel`) for dense,
    /// unweighted fits of `1 ≤ k ≤ K_DUAL_MAX` components (`k ≥ 2` at
    /// `raw_perm` and `split_exact`). A unit the kernel cannot certify runs the
    /// Primal unit body.
    Nspace,
    /// The p-space Gram backend (`gram_p`): tall blocks
    /// (`p < 1.54·n_tr`), dense, weighted or sparse, `1 ≤ k ≤ K_GRAM_MAX`.
    GramP,
}

/// `raw_perm`'s selector: `Special` when the K = 1 Gram closed form claims
/// the input (`raw_perm_k1_gram_route` owns that rule). After the K = 1
/// closed form, dense and unweighted `2 ≤ k ≤ K_DUAL_MAX` input the n-space
/// rule admits (`dual_route::nspace_eligible_raw_perm`) takes `Nspace`,
/// unless `gram_routes_disabled()`. `GramP` when `gram_p::gram_p_eligible`
/// admits the block (tried after `Nspace`: in the overlap band
/// `n_tr < p < 1.54·n_tr` the n-space route is cheaper), unless
/// `gram_routes_disabled()`. Everything else is `Primal`.
pub(crate) fn raw_perm_route(
    n: usize,
    n_folds: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    keep: Option<usize>,
    weighted: bool,
) -> ReplicateRoute {
    if raw_perm_k1_gram_route(n, n_folds, p, n_perm, k, keep.is_none(), weighted) {
        return ReplicateRoute::Special;
    }
    if gram_routes_disabled() {
        return ReplicateRoute::Primal;
    }
    let dense_unweighted = keep.is_none() && !weighted;
    if crate::dual_route::nspace_eligible_raw_perm(n, n_folds, p, n_perm, k, dense_unweighted) {
        ReplicateRoute::Nspace
    } else if crate::gram_p::gram_p_eligible(n - n / n_folds, p, n_perm + 1, k, keep) {
        ReplicateRoute::GramP
    } else {
        ReplicateRoute::Primal
    }
}

/// `split_exact`'s selector, called by `confirmatory_test_impl` before any
/// draw from the parent rng: `Special` (the no-refit route,
/// `run_split_perm_nr`) exactly when `split_exact_no_refit_route(k, keep)`
/// holds; otherwise the refit route through [`split_zbars_columns`]. After
/// the K = 1 no-refit route, dense and unweighted `2 ≤ k ≤ K_DUAL_MAX`
/// input the n-space rule admits (`dual_route::nspace_eligible_split_exact`
/// on `split_sizes(n, k).0`) takes `Nspace`, unless `gram_routes_disabled()`.
/// `GramP` when `gram_p::gram_p_eligible` admits the block (tried after
/// `Nspace`: in the overlap band `n_tr < p < 1.54·n_tr` the n-space route is
/// cheaper), unless `gram_routes_disabled()`. Everything else is `Primal`.
pub(crate) fn split_exact_refit_route(
    n: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    keep: Option<usize>,
    weighted: bool,
) -> ReplicateRoute {
    if split_exact_no_refit_route(k, keep) {
        return ReplicateRoute::Special;
    }
    if gram_routes_disabled() {
        return ReplicateRoute::Primal;
    }
    let (n_train, _) = crate::resample::split_sizes(n, k);
    let dense_unweighted = keep.is_none() && !weighted;
    if crate::dual_route::nspace_eligible_split_exact(n_train, p, n_perm, k, dense_unweighted) {
        ReplicateRoute::Nspace
    } else if crate::gram_p::gram_p_eligible(n_train, p, n_perm + 1, k, keep) {
        ReplicateRoute::GramP
    } else {
        ReplicateRoute::Primal
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// `raw_perm` CV R² test
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
fn run_raw_perm(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_perm: usize,
    n_folds: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use rand::seq::SliceRandom;

    let n = x.nrows();

    // Fixed fold indices: shuffle once, then split.
    // Weights are passed through to the fold driver, which re-normalizes per fold.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    let folds = crate::linalg::fold_split(&indices, n_folds);

    // Seeds are `child_seeds(rng, n_perm)`, drawn right after the fold
    // shuffle and before the route is chosen: column c >= 1 of `cols` is
    // `permutation_from_seed(n, seeds[c - 1])` applied to y, so every route,
    // and any loop order, sees the same permutations at a given seed.
    let seeds = crate::rng::child_seeds(rng, n_perm);
    let cols = crate::resample::Columns { y, seeds: &seeds };

    // Route choice: `raw_perm_route` owns the rule. Decided here, on the
    // calling thread, before any parallel work.
    let route = raw_perm_route(
        n,
        n_folds,
        x.ncols(),
        n_perm,
        k,
        opts.keep,
        w_norm.is_some(),
    );
    let r2: Vec<f64> = match route {
        ReplicateRoute::Special => {
            // Build y_mat one column at a time. `y_mat` is not covered by
            // the dual route's `n_tr` cap (see `DUAL_ROUTE_MAX_N_TR`), which
            // bounds only the Gram matrix.
            let mut y_mat = Mat::<f64>::zeros(n, cols.len());
            for c in 0..cols.len() {
                let yc = cols.column(c);
                for i in 0..n {
                    y_mat[(i, c)] = yc[i];
                }
            }
            crate::dual_route::pls1_cv_r2_columns(x, y_mat.as_ref(), &folds)
        }
        // Folds outer, replicate columns inner: each fold's X side (gather,
        // standardize, √w) is built once instead of once per replicate.
        // Parallelism stays over the n_perm + 1 columns (there are only
        // n_folds folds). Permuted y rows; weights stay tied to row indices
        // (not permuted). Null column c regenerates its permutation from
        // seeds[c - 1] inside each unit, so no n_perm·n buffer is held.
        route @ (ReplicateRoute::Primal | ReplicateRoute::Nspace | ReplicateRoute::GramP) => {
            pooled_cv_r2_columns(route, x, &folds, w_norm, k, opts.keep, cols.len(), &|c| {
                cols.column(c)
            })?
        }
    };
    let cv_r2_obs = r2[0];
    let nulls_vec = &r2[1..];

    // A failed null fit surfaces as NaN (the fold driver makes a failed null
    // column NaN); count it as an exceedance so it biases p upward, never
    // downward. That fail-soft rule is only sound for occasional failures:
    // past half the nulls, p saturates toward 1 and the test silently stops
    // measuring anything: error out instead (hardcoded 1/2; not worth a knob).
    let nan_nulls = nulls_vec.iter().filter(|v| v.is_nan()).count();
    if nan_nulls * 2 > n_perm {
        return Err(PlsKitError::PermNullDegenerate {
            failed: nan_nulls,
            total: n_perm,
        });
    }
    let exceedances = nulls_vec
        .iter()
        .filter(|v| v.is_nan() || **v >= cv_r2_obs)
        .count();
    let p = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);

    Ok(RunResult {
        pvalue: p,
        statistic: cv_r2_obs,
        rho_hat: None,
    })
}

/// Single-column [`pooled_cv_r2_columns`] on the primal route: the primal
/// fold loop that the Gram routes and the route tests compare against.
#[cfg(test)]
fn pls1_cv_r2(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    folds: &[Vec<usize>],
    weights: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
) -> PlsKitResult<f64> {
    let r2 = pooled_cv_r2_columns(
        ReplicateRoute::Primal,
        x,
        folds,
        weights,
        k,
        keep,
        1,
        &|_| y.to_owned(),
    )?;
    Ok(r2[0])
}

/// The caller's mean-one weights sliced to rows `idx` and renormalized to
/// mean one within them: the per-fold / per-half weights of `raw_perm` and
/// the split routes. An all-zero or all-equal slice is no weights (`None`),
/// as `pls1_fit` drops all-equal weights.
fn renormalized_slice_weights(w: Option<ColRef<'_, f64>>, idx: &[usize]) -> Option<Col<f64>> {
    use crate::linalg::{col_row_subset, normalize_weights};
    w.and_then(|w| normalize_weights(col_row_subset(w, idx).as_ref()))
        .filter(|w| !crate::fit::weights_all_equal(w.as_ref()))
}

/// One `raw_perm` fold, prepared once: the fold's X side of
/// `pls1_fit(pre_standardized, !check_n_eff, Seq, w_tr)`, standardized
/// training and validation blocks plus the fold's renormalized weights and
/// the √w'' scaling `pls1_fit` applies, everything that depends on X, the
/// folds and the weights but not on the outcome.
pub(crate) struct CvFold {
    /// Training rows (every fold but this one, in fold order).
    pub(crate) train_idx: Vec<usize>,
    /// This fold's rows.
    pub(crate) val_idx: Vec<usize>,
    /// Standardized training block (weighted moments when weighted), unscaled.
    pub(crate) xs_tr: Mat<f64>,
    /// Validation block standardized with the training moments, unscaled.
    pub(crate) xs_val: Mat<f64>,
    /// The fold's renormalized weights (`renormalized_slice_weights` on the
    /// fold's training slice), `None` when that slice is all zero or all equal.
    pub(crate) w_tr: Option<Col<f64>>,
    /// `fit_row_scale(w_tr)`, the √w'' `pls1_fit` would apply.
    pub(crate) sqw_fit: Option<Col<f64>>,
    /// `scale_rows(xs_tr, sqw_fit)` when weighted, else `None` (use `xs_tr`).
    pub(crate) xs_tr_fit: Option<Mat<f64>>,
    /// `‖·‖_F` (`norm_l2`) of the block the fit runs on (`xs_tr_fit`, else
    /// `xs_tr`), taken once per fold rather than once per column: the same
    /// call on the same view, so the same bits.
    pub(crate) x_fro: f64,
}

/// Prepare fold `fi` of `folds` once. `weights` are the caller's mean-one
/// weights; the fold slices them to its training rows and renormalizes
/// (no weights when the slice is all zero or all equal), and `pls1_fit` would renormalize
/// that again before its √w scaling, which `sqw_fit` reproduces. The X-side
/// check of `pls1_fit` (a finite standardized training block) runs here,
/// once per fold on every route, instead of once per column.
///
/// # Errors
/// `NonFiniteInput` when the standardized training block is not finite:
/// the error every column's fit on this fold met first (`pls1_fit` checks X
/// before y), so the driver returns the same error a per-column fit of
/// column 0 would.
pub(crate) fn prepare_cv_fold(
    x: MatRef<'_, f64>,
    folds: &[Vec<usize>],
    fi: usize,
    weights: Option<ColRef<'_, f64>>,
) -> PlsKitResult<CvFold> {
    use crate::fit::{check_finite_mat, fit_row_scale, scale_rows};
    use crate::linalg::{standardize_apply_rows, standardize_rows};
    let val_idx = folds[fi].clone();
    let train_idx: Vec<usize> = folds
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != fi)
        .flat_map(|(_, f)| f.iter().copied())
        .collect();
    let w_tr = renormalized_slice_weights(weights, &train_idx);
    let (xs_tr, x_mean, x_scale) =
        standardize_rows(x, &train_idx, w_tr.as_ref().map(Col::as_ref), None);
    check_finite_mat(xs_tr.as_ref())?;
    let xs_val = standardize_apply_rows(x, &val_idx, x_mean.as_ref(), x_scale.as_ref(), None);
    let sqw_fit = w_tr.as_ref().map(|w| fit_row_scale(w.as_ref()));
    let xs_tr_fit = sqw_fit
        .as_ref()
        .map(|s| scale_rows(xs_tr.as_ref(), s.as_ref()));
    let x_fro = xs_tr_fit.as_ref().unwrap_or(&xs_tr).norm_l2();
    Ok(CvFold {
        train_idx,
        val_idx,
        xs_tr,
        xs_val,
        w_tr,
        sqw_fit,
        xs_tr_fit,
        x_fro,
    })
}

/// This fold's `(ss_res, ss_tot)` for one outcome column, on a prepared
/// fold. `y_of(i)` is the raw outcome of full-data row `i`. The outcome is
/// standardized per fold per column (the per-fold y scale does not cancel
/// out of the pooled ratio, so it is not hoisted). The fit is what
/// `pls1_fit(xs_tr, ys_tr, k, w_tr, pre_standardized, !check_n_eff, Seq,
/// keep)` computes, with its √w'' X scaling taken from the fold's own
/// `xs_tr_fit` / `sqw_fit`. Its X check ran once in `prepare_cv_fold`;
/// the per-column checks run here in `pls1_fit`'s order, so a fold fails
/// where and how `pls1_fit` would.
///
/// # Errors
/// What `pls1_fit` returns for this fold's arrays.
#[allow(clippy::similar_names)]
pub(crate) fn cv_fold_contribution(
    fold: &CvFold,
    y_of: &dyn Fn(usize) -> f64,
    k: usize,
    keep: Option<usize>,
) -> PlsKitResult<(f64, f64)> {
    use crate::fit::{check_fit_y_and_k, pls1_fit_prepared_fro, scale_col, ParChoice};
    let (ys_tr, ys_val) = cv_fold_targets(fold, y_of);

    check_fit_y_and_k(fold.xs_tr.ncols(), ys_tr.as_ref(), k, keep)?;
    // Seq inside the per-fold worker: outer Rayon owns the threadpool.
    let fit = match (&fold.xs_tr_fit, &fold.sqw_fit) {
        (Some(xs_fit), Some(sqw)) => {
            let ys_fit = scale_col(ys_tr.as_ref(), sqw.as_ref());
            pls1_fit_prepared_fro(
                xs_fit.as_ref(),
                ys_fit.as_ref(),
                k,
                keep,
                ParChoice::Seq,
                fold.x_fro,
            )?
        }
        _ => pls1_fit_prepared_fro(
            fold.xs_tr.as_ref(),
            ys_tr.as_ref(),
            k,
            keep,
            ParChoice::Seq,
            fold.x_fro,
        )?,
    };

    let y_pred = crate::linalg::mat_vec(fold.xs_val.as_ref(), fit.coef.as_ref(), faer::Par::Seq);
    Ok(cv_fold_ss(&y_pred, &ys_val))
}

/// The fold's standardized training target and validation target for one
/// outcome column: `standardize1_weighted` (with the fold's `w_tr` when
/// weighted) of the training y, and the validation y standardized with the
/// training moments. The y side of [`cv_fold_contribution`], shared with the
/// Gram-p arm so both fit and score the same bits.
#[allow(clippy::similar_names)]
pub(crate) fn cv_fold_targets(fold: &CvFold, y_of: &dyn Fn(usize) -> f64) -> (Col<f64>, Col<f64>) {
    let n_tr = fold.train_idx.len();
    let n_val = fold.val_idx.len();
    let y_tr = Col::<f64>::from_fn(n_tr, |i| y_of(fold.train_idx[i]));
    let (ys_tr, y_mean, y_scale) =
        crate::linalg::standardize1_weighted(y_tr.as_ref(), fold.w_tr.as_ref().map(Col::as_ref));
    let ys_val = Col::<f64>::from_fn(n_val, |i| (y_of(fold.val_idx[i]) - y_mean) / y_scale);
    (ys_tr, ys_val)
}

/// This fold's `(ss_res, ss_tot)` from its validation predictions: the
/// unweighted pooled-sum arithmetic of the CV-R² convention (see
/// [`pooled_cv_r2_columns`]). `ss_tot` reads `ys_val` only.
pub(crate) fn cv_fold_ss(y_pred: &Col<f64>, ys_val: &Col<f64>) -> (f64, f64) {
    let n_val = ys_val.nrows();
    let mean_val: f64 = (0..n_val).map(|i| ys_val[i]).sum::<f64>() / n_val as f64;
    let res = (0..n_val)
        .map(|i| (y_pred[i] - ys_val[i]).powi(2))
        .sum::<f64>();
    let tot = (0..n_val)
        .map(|i| (ys_val[i] - mean_val).powi(2))
        .sum::<f64>();
    (res, tot)
}

/// The matrix the fold's training fit runs on: `xs_tr_fit` (`√w''`-scaled)
/// when weighted, `xs_tr` otherwise. The fold's Gram block is built on it.
pub(crate) fn fold_fit_matrix(fold: &CvFold) -> MatRef<'_, f64> {
    fold.xs_tr_fit
        .as_ref()
        .map_or(fold.xs_tr.as_ref(), Mat::as_ref)
}

/// The Gram-p arm of [`fold_unit`]: one (fold, column) on the fold's
/// p-space Gram block, else [`cv_fold_contribution`] (the Primal arm),
/// whose value or error is then the Primal route's to the bit. `gram` is
/// built on [`fold_fit_matrix`]; the fit target is the fold's standardized
/// training y, `scale_col`-scaled by `sqw_fit` when weighted, the arrays the
/// Primal arm fits. A non-finite target fails the `tt` gate (every gate is
/// an `a >= b` conjunction, so NaN fails it) and reaches the Primal arm,
/// which reports it; invalid `k` or `keep` never reach this arm
/// (`gram_p::gram_p_eligible`). The validation predictions are a sequential
/// product, so the unit's bits do not depend on the thread count when the
/// Gram fit resolves; the fallback is the Primal arm's.
pub(crate) fn raw_perm_unit(
    gram: &crate::gram_p::GramPBlock<'_>,
    fold: &CvFold,
    y_of: &dyn Fn(usize) -> f64,
    k: usize,
    keep: Option<usize>,
) -> PlsKitResult<(f64, f64)> {
    use faer::linalg::matmul::matmul;
    debug_assert!(k >= 1);
    let (ys_tr, ys_val) = cv_fold_targets(fold, y_of);
    let ys_fit = match fold.sqw_fit.as_ref() {
        Some(s) => crate::fit::scale_col(ys_tr.as_ref(), s.as_ref()),
        None => ys_tr,
    };
    if let Some((coef, _)) = gram.fit_replicate(ys_fit.as_ref(), k, keep) {
        let mut y_pred = Col::<f64>::zeros(fold.xs_val.nrows());
        matmul(
            y_pred.as_mut().as_mat_mut(),
            faer::Accum::Replace,
            fold.xs_val.as_ref(),
            coef.as_ref().as_mat(),
            1.0,
            Par::Seq,
        );
        return Ok(cv_fold_ss(&y_pred, &ys_val));
    }
    cv_fold_contribution(fold, y_of, k, keep)
}

/// Per-fold state of a route, built once per fold by
/// [`pooled_cv_r2_columns`] after the fold's preparation and shared
/// read-only by that fold's columns. The primal route needs nothing beyond
/// the prepared fold.
pub(crate) enum FoldBlock<'f> {
    /// No per-fold state.
    Primal(std::marker::PhantomData<&'f ()>),
    /// n-space Gram route: `G`, its norms and `M = X̃_val X̃_tr'` of this fold.
    Nspace(crate::dual_route::NspaceFold),
    /// The fold's p-space Gram block on [`fold_fit_matrix`].
    GramP(crate::gram_p::GramPBlock<'f>),
}

/// The block of `route` for one prepared fold, its precompute run under
/// `par` ([`crate::fit::par_fixed`]). `Special` never reaches a
/// driver; handed one, it gets the primal block. `Nspace` builds `G`, its
/// norms and `M` once per fold; `GramP` builds `C` on [`fold_fit_matrix`]
/// and its norm estimate once per fold.
pub(crate) fn fold_block(route: ReplicateRoute, fold: &CvFold, par: Par) -> FoldBlock<'_> {
    match route {
        ReplicateRoute::Nspace => FoldBlock::Nspace(crate::dual_route::NspaceFold::new(fold, par)),
        ReplicateRoute::GramP => {
            FoldBlock::GramP(crate::gram_p::GramPBlock::new(fold_fit_matrix(fold), par))
        }
        ReplicateRoute::Special | ReplicateRoute::Primal => {
            FoldBlock::Primal(std::marker::PhantomData)
        }
    }
}

/// One (fold, column) unit: this fold's `(ss_res, ss_tot)` for the outcome
/// `y_of`. The primal arm is [`cv_fold_contribution`]; a route that cannot
/// certify a unit falls back to it.
///
/// # Errors
/// What `cv_fold_contribution` returns.
pub(crate) fn fold_unit(
    block: &FoldBlock<'_>,
    fold: &CvFold,
    y_of: &dyn Fn(usize) -> f64,
    k: usize,
    keep: Option<usize>,
) -> PlsKitResult<(f64, f64)> {
    match block {
        FoldBlock::Primal(_) => cv_fold_contribution(fold, y_of, k, keep),
        FoldBlock::Nspace(nf) => {
            debug_assert!(k >= 1 && keep.is_none() && fold.w_tr.is_none());
            crate::dual_route::fold_column_nspace(fold, nf, y_of, k)
        }
        FoldBlock::GramP(gram) => raw_perm_unit(gram, fold, y_of, k, keep),
    }
}

/// Pooled K-fold CV R² for PLS1, for `n_cols` outcome columns at once: the
/// fold driver of `raw_perm`, on `route`.
///
/// CV-R² convention (anchor: `find_k::select_cv` cites this): this function
/// pools `SS_res` and `SS_tot` across all folds and forms a single R² from
/// the pooled sums, with **unweighted** validation residuals even when
/// training is weighted. `select_cv` in `find_k.rs` deliberately uses the
/// other convention (per-fold **weighted-validation** R² averaged across
/// folds) because k-selection wants each fold weighted on its own scale. The
/// two are not interchangeable; each method owns the convention appropriate
/// to its statistic.
///
/// Folds outer and columns inner: each fold is prepared once
/// ([`prepare_cv_fold`], where the X check runs), its block is built once
/// ([`fold_block`] under [`crate::fit::par_fixed`]), and the columns
/// map through [`fold_unit`] in parallel, so no `B·n`
/// outcome buffer is held; `column_y(c)` builds column `c` (length n, raw)
/// inside the unit that needs it. `ss_res` and `ss_tot` accumulate per
/// column in fold order, from `0.0`, exactly as the single-column loop
/// does. A failed fold preparation, or a failed fit of column 0, returns
/// that error (the first in fold order); a failed fit of any other column
/// makes that column NaN and skips it in the remaining folds. `k <= p` is
/// the caller's precondition (`confirmatory_test_impl` rejects `k > p`);
/// `k = 0` is reachable and fails in the unit with `pls1_fit`'s error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pooled_cv_r2_columns(
    route: ReplicateRoute,
    x: MatRef<'_, f64>,
    folds: &[Vec<usize>],
    weights: Option<ColRef<'_, f64>>,
    k: usize,
    keep: Option<usize>,
    n_cols: usize,
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
) -> PlsKitResult<Vec<f64>> {
    debug_assert!(k <= x.ncols(), "k <= p is the caller's precondition");
    let par = crate::fit::par_fixed();
    let mut ss_res = vec![0.0_f64; n_cols];
    let mut ss_tot = vec![0.0_f64; n_cols];
    let mut failed = vec![false; n_cols];
    for fi in 0..folds.len() {
        let fold = prepare_cv_fold(x, folds, fi, weights)?;
        let block = fold_block(route, &fold, par);
        let contrib: Vec<Option<PlsKitResult<(f64, f64)>>> =
            crate::resample::map_indexed(n_cols, |c| {
                if failed[c] {
                    return None;
                }
                let yc = column_y(c);
                Some(fold_unit(&block, &fold, &|i| yc[i], k, keep))
            });
        for (c, r) in contrib.into_iter().enumerate() {
            match r {
                None => {}
                Some(Ok((res, tot))) => {
                    ss_res[c] += res;
                    ss_tot[c] += tot;
                }
                Some(Err(e)) if c == 0 => return Err(e),
                Some(Err(_)) => failed[c] = true,
            }
        }
    }
    Ok((0..n_cols)
        .map(|c| {
            if failed[c] {
                f64::NAN
            } else if ss_tot[c] > 0.0 {
                1.0 - ss_res[c] / ss_tot[c]
            } else {
                0.0
            }
        })
        .collect())
}

// ──────────────────────────────────────────────────────────────────────────────
// `split_nb` and `split_exact`'s refit route (`run_split_perm`)
// ──────────────────────────────────────────────────────────────────────────────

/// One split-half partition of the row indices `0..n`.
pub(crate) struct SplitIdx {
    pub(crate) tr: Vec<usize>,
    pub(crate) te: Vec<usize>,
}

/// Per-column `z̄` over the J splits: `per_split` maps one split to its
/// `n_cols` clamped `atanh(r)` values, and column `c` of the result is
/// their mean over the splits.
///
/// The one accumulation behind every splits-outer `split_exact` route
/// (PLS1's `split_perm_nr_zbars` and `split_zbars_columns`, PLS3's
/// `pls3_split_zbars_columns_primal` and
/// `dual_route::pls3_split_zbars_columns`), shared so that those routes
/// cannot drift apart in the last bits. The splits may be mapped in
/// parallel, but the sum runs over them **ascending** and divides by J
/// exactly once at the end: `mean_fisher_z`'s left-to-right
/// sum-then-divide, column by column. Averaging `r` first,
/// dividing inside the loop, or summing in completion order would not
/// reproduce it.
///
/// `parallel_splits`: `true` maps the splits with Rayon, `false` one at a time
/// (one prepared split alive; the parallelism is over columns inside `per_split`).
pub(crate) fn zbars_over_splits<F>(
    splits: &[SplitIdx],
    n_cols: usize,
    parallel_splits: bool,
    per_split: F,
) -> Vec<f64>
where
    F: Fn(&SplitIdx) -> Vec<f64> + Sync,
{
    let per_split_z: Vec<Vec<f64>> = if parallel_splits {
        use rayon::prelude::*;
        splits.par_iter().map(&per_split).collect()
    } else {
        splits.iter().map(&per_split).collect()
    };
    let mut z_sum = vec![0.0_f64; n_cols];
    for per in &per_split_z {
        for (col, v) in per.iter().enumerate() {
            z_sum[col] += v;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let j = splits.len() as f64;
    z_sum.into_iter().map(|s| s / j).collect()
}

/// Draw the J split-half partitions used by `split_nb` and by `split_exact`'s
/// refit route.
///
/// Split fraction is hardcoded 50/50: NB calibration assumes balanced halves
/// and there is no scientific reason to vary it.
///
/// Drawn through `parallel_for_each_seeded` rather than a plain sequential loop:
/// the parent state gives J child seeds, and `one_split` is the first draw off
/// each child. This per-split seeded stream fixes `split_nb`'s splits, and so
/// its results, at a given seed; a different draw order would move them.
///
/// Also drawn by `pls3_signal_test::pls3_confirmatory_test`, which needs the
/// same fixed-across-replicates split set on the PLS3 statistic.
pub(crate) fn draw_splits(
    n: usize,
    k: usize,
    n_splits: usize,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<Vec<SplitIdx>> {
    use crate::resample::{one_split, split_sizes};

    // n-3 test-floor in split_sizes can drop n_train below k+2; below that the
    // per-half fit degrades to a silent r=0. Reject up front. See split_sizes.
    // Mirrored in split_perm_nr_zbars and run_e — change together.
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, _) = split_sizes(n, k);

    Ok(crate::resample::parallel_for_each_seeded(
        rng,
        n_splits,
        |_, child| {
            let (tr, te) = one_split(n, n_train, child);
            SplitIdx { tr, te }
        },
    ))
}

/// Compute the split-half Pearson r on each supplied split.
///
/// Takes the splits rather than drawing them so a caller can hold one set of
/// splits fixed across many outcome vectors — what `split_exact`'s permutation
/// loop needs. Nothing here consumes randomness: given `(x, y, split)` the fit
/// and the r are deterministic, so the J splits map in parallel directly
/// instead of through `parallel_for_each_seeded`.
///
/// Its only production caller is `run_split_nb`; `split_exact`'s refit route
/// (`run_split_perm`) reaches the same per-split arithmetic through
/// `split_zbars_columns` instead.
///
/// `x`/`y` are the **raw** (un-√w-scaled) inputs; `w_norm` carries the
/// mean-1-normalized weights when present. Each half is standardized with
/// **weighted** moments and then √w' row-scaled (the √w row-scaling
/// convention of `pls1_fit`, see `linalg::sqrt_col`). The reported
/// statistic is the unweighted Pearson r on the resulting √w-scaled
/// test-half scores and test-half y — *not* the weighted Pearson r on the
/// original data, because the test-half centering subtracts
/// the plain mean of the √w-scaled values rather than the weighted mean. The
/// identical transform is applied to observed and permuted data, and the
/// per-half weights stay tied to their rows (never permuted), so permutation
/// calibration still holds.
#[allow(clippy::too_many_arguments)]
fn split_half_correlations(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    splits: &[SplitIdx],
    w_norm: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
) -> Col<f64> {
    let per_split = |sp: &SplitIdx| split_half_r(x, y, k, sp, w_norm, keep);

    // Same shape as split_perm_nr_zbars' per-split dispatch: collect preserves
    // split order, so the result does not depend on which worker runs which split.
    let r_vec: Vec<f64> = {
        use rayon::prelude::*;
        splits.par_iter().map(per_split).collect()
    };

    Col::<f64>::from_fn(r_vec.len(), |i| r_vec[i])
}

/// One split of `split_exact`'s refit route and of `split_nb`, prepared
/// once: the X side of [`split_half_r`], which depends on X, the split and
/// the weights but not on the outcome, so a permutation loop can build it
/// once per split.
pub(crate) struct PreparedSplit {
    /// Standardized training half, √w_tr-scaled when weighted.
    pub(crate) xs_tr: Mat<f64>,
    /// Test half with training moments, √w_te-scaled when weighted.
    pub(crate) xs_te: Mat<f64>,
    /// Per-half renormalized weights (`split_half_r`'s `w_tr`).
    pub(crate) w_tr: Option<Col<f64>>,
    /// Per-half renormalized weights (`split_half_r`'s `w_te`).
    pub(crate) w_te: Option<Col<f64>>,
    /// Whether `xs_tr` is finite: the X check `pls1_fit` makes, done once
    /// per split instead of once per column. When false, every column's `r`
    /// is 0, as the failed per-half fit gave.
    pub(crate) x_finite: bool,
    /// `xs_tr.norm_l2()`, taken once per split rather than once per column:
    /// the same call on the same view, so the same bits.
    pub(crate) x_fro: f64,
}

/// The X side of [`split_half_r`] for split `sp`. Per-half weights are the
/// caller's `w_norm` sliced to the half and renormalized to mean one
/// (mirrors `signal_test::pooled_cv_r2_columns`'s per-fold renormalization),
/// with no weights for an all-zero or all-equal half. Each half is standardized with the
/// training half's weighted moments and then √w-scaled with its own
/// half's weights as given (the √w row-scaling convention of `pls1_fit`), in one pass
/// per half (`linalg::standardize_rows`).
#[allow(clippy::similar_names)]
pub(crate) fn prepare_split(
    x: MatRef<'_, f64>,
    sp: &SplitIdx,
    w_norm: Option<ColRef<'_, f64>>,
) -> PreparedSplit {
    use crate::linalg::{sqrt_col, standardize_apply_rows, standardize_rows};
    let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
    let w_tr = renormalized_slice_weights(w_norm, tr);
    let w_te = renormalized_slice_weights(w_norm, te);
    let sw_tr = w_tr.as_ref().map(|w| sqrt_col(w.as_ref()));
    let sw_te = w_te.as_ref().map(|w| sqrt_col(w.as_ref()));
    let (xs_tr, x_mean, x_scale) = standardize_rows(
        x,
        tr,
        w_tr.as_ref().map(Col::as_ref),
        sw_tr.as_ref().map(Col::as_ref),
    );
    let xs_te = standardize_apply_rows(
        x,
        te,
        x_mean.as_ref(),
        x_scale.as_ref(),
        sw_te.as_ref().map(Col::as_ref),
    );
    let x_finite = crate::fit::check_finite_mat(xs_tr.as_ref()).is_ok();
    let x_fro = xs_tr.norm_l2();
    PreparedSplit {
        xs_tr,
        xs_te,
        w_tr,
        w_te,
        x_finite,
        x_fro,
    }
}

/// [`split_half_r`] after its X side: standardize the training outcome
/// (weighted moments when weighted), √w_tr-scale it, fit on the prepared
/// training half, score the prepared test half, and correlate with the
/// √w_te-scaled test outcome through `guarded_pearson`. `y_of(i)` is the raw
/// outcome of full-data row `i`. The fit is `pls1_fit(xs_tr, ys_tr, k, None,
/// pre_standardized, !check_n_eff, Seq, keep)` with its X check done once
/// in `prepare_split` (`x_finite`): the per-column checks in `pls1_fit`'s
/// order, then the kernel tail, which is all `pls1_fit` runs on unweighted
/// pre-standardized input. A failed per-half fit gives `r = 0`: a truncated
/// model still yields a valid `r` at `k_used`, which beats silently
/// recording 0 through the error arm, so there is no `n_eff` truncation check.
#[allow(clippy::similar_names)]
pub(crate) fn split_column_r(
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
    k: usize,
    keep: Option<usize>,
) -> f64 {
    use crate::fit::{check_fit_y_and_k, pls1_fit_prepared_fro, ParChoice};
    if !prep.x_finite {
        return 0.0;
    }
    let ys_tr = split_train_target(prep, sp, y_of);
    if check_fit_y_and_k(prep.xs_tr.ncols(), ys_tr.as_ref(), k, keep).is_err() {
        return 0.0;
    }
    // Seq inside the per-split worker: outer Rayon owns the threadpool.
    let Ok(m) = pls1_fit_prepared_fro(
        prep.xs_tr.as_ref(),
        ys_tr.as_ref(),
        k,
        keep,
        ParChoice::Seq,
        prep.x_fro,
    ) else {
        return 0.0;
    };
    // Both the scores (through the √w_te-scaled X̃_te) and the test outcome
    // carry √w_te, so the Pearson r is taken on √w-scaled data (the √w
    // row-scaling convention).
    let scores_te = crate::linalg::mat_vec(prep.xs_te.as_ref(), m.coef.as_ref(), faer::Par::Seq);
    let n_te = scores_te.nrows();
    let y_te = split_test_target(prep, sp, y_of);
    guarded_pearson(n_te, |i| scores_te[i], |i| y_te[i])
}

/// The training target `split_column_r` fits on: the half's y standardized
/// (weighted with the half's `w_tr` when weighted), then `√w_tr`-scaled as
/// `split_half_r` scales it. Shared with the Gram-p arm so both fit the
/// same bits.
#[allow(clippy::similar_names)]
pub(crate) fn split_train_target(
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
) -> Col<f64> {
    let tr = sp.tr.as_slice();
    let y_tr = Col::<f64>::from_fn(tr.len(), |i| y_of(tr[i]));
    let w_tr_ref = prep.w_tr.as_ref().map(Col::as_ref);
    let (ys_tr, _, _) = crate::linalg::standardize1_weighted(y_tr.as_ref(), w_tr_ref);
    match w_tr_ref {
        Some(w) => Col::<f64>::from_fn(ys_tr.nrows(), |i| ys_tr[i] * w[i].sqrt()),
        None => ys_tr,
    }
}

/// The held-out target `split_column_r` correlates with: the test half's
/// raw y, `√w_te`-scaled when weighted (the √w row-scaling convention, see
/// `split_half_correlations`).
pub(crate) fn split_test_target(
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
) -> Col<f64> {
    let te = sp.te.as_slice();
    let n_te = te.len();
    match prep.w_te.as_ref() {
        Some(w) => Col::<f64>::from_fn(n_te, |i| y_of(te[i]) * w[i].sqrt()),
        None => Col::<f64>::from_fn(n_te, |i| y_of(te[i])),
    }
}

/// The p-space Gram block of one split's training half (`prep.xs_tr`,
/// `√w_tr`-scaled with the half's own weights when weighted), with the test
/// half's Frobenius norm the score gate needs. Built once per split by
/// `split_block`.
pub(crate) struct SplitGram<'a> {
    block: crate::gram_p::GramPBlock<'a>,
    xte_fro: f64,
}

impl<'a> SplitGram<'a> {
    /// Build the block for `prep` (the driver passes
    /// `fit::par_fixed()` as `par`).
    pub(crate) fn new(prep: &'a PreparedSplit, par: Par) -> Self {
        Self {
            block: crate::gram_p::GramPBlock::new(prep.xs_tr.as_ref(), par),
            xte_fro: prep.xs_te.norm_l2(),
        }
    }
}

/// The Gram-p arm of [`split_unit`]: the Gram fit, then the test-half
/// scores through a score-degeneracy gate before the correlation, else
/// [`split_column_r`] (the Primal arm), whose `r` is then the Primal
/// route's to the bit. The fit target is [`split_train_target`], the array
/// the Primal arm fits. A non-finite target fails the Gram gates (every
/// gate is an `a >= b` conjunction, so NaN fails it) and reaches the Primal
/// arm, which answers it; invalid `k` or `keep` never reach this arm
/// (`gram_p::gram_p_eligible`). The test-half scores are a sequential
/// product, so the unit's bits do not depend on the thread count when the
/// Gram scores are kept; the fallback is the Primal arm's.
///
/// # Score gate (a decision gate)
/// `guarded_pearson`'s `constant_to_rounding` test is a discontinuity, and
/// the two routes' scores differ by about
/// `gram_p::score_discrepancy_bound` (`D`, first order). The Gram scores
/// are kept only when they are not constant to rounding, their centered
/// norm `t_c` is at least `dual_route::SCORE_BAND` times their norm (the
/// one shared constant), and `t_c` clears `RESOLVE_BAND` times both
/// `D` and `constant_to_rounding`'s threshold `2·n_te·ε·‖s_te‖`, so the
/// Primal route's scores are not constant either. Two cases are decided
/// directly because they are exact on both routes: a resolved `k_used = 0`
/// (a zero `coef`, zero scores, `r = 0`) and a held-out y constant to
/// rounding (the same bits on both routes). `split_unit` answers a
/// non-finite X half before any arm runs.
pub(crate) fn split_column_r_gram_p(
    gram: &SplitGram<'_>,
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
    k: usize,
    keep: Option<usize>,
) -> f64 {
    use faer::linalg::matmul::matmul;
    debug_assert!(k >= 1);
    let ys_tr = split_train_target(prep, sp, y_of);
    let Some(fit) = gram.block.fit_replicate_full(ys_tr.as_ref(), k, keep) else {
        return split_column_r(prep, sp, y_of, k, keep);
    };
    if fit.k_used == 0 {
        return 0.0;
    }
    let y_te = split_test_target(prep, sp, y_of);
    let n_te = y_te.nrows();
    let mut scores = Col::<f64>::zeros(n_te);
    matmul(
        scores.as_mut().as_mat_mut(),
        faer::Accum::Replace,
        prep.xs_te.as_ref(),
        fit.coef.as_ref().as_mat(),
        1.0,
        Par::Seq,
    );
    let ms = scaled_moments(n_te, |i| scores[i], None);
    let my = scaled_moments(n_te, |i| y_te[i], None);
    if my.is_constant(n_te) {
        return 0.0;
    }
    let t_norm = ms.sq.sqrt() * ms.s;
    let t_c = ms.ss.sqrt() * ms.s;
    let d = crate::gram_p::score_discrepancy_bound(
        gram.xte_fro,
        fit.coef_err,
        fit.coef.norm_l2(),
        prep.xs_te.ncols(),
        n_te,
        t_norm,
    );
    let const_threshold = 2.0 * n_te as f64 * f64::EPSILON * t_norm;
    let band = crate::dual_route::RESOLVE_BAND;
    // `a >= b` conjunctions, so a NaN fails the gate.
    let resolved = !ms.is_constant(n_te)
        && t_c >= crate::dual_route::SCORE_BAND * t_norm
        && t_c >= band * d
        && t_c >= band * const_threshold;
    if resolved {
        pearson_scaled(n_te, |i| scores[i], |i| y_te[i], &ms, &my)
    } else {
        split_column_r(prep, sp, y_of, k, keep)
    }
}

/// The split-half Pearson r on one split: `split_half_correlations`'s
/// per-split computation, which `run_split_nb` uses. Also the primal
/// fallback of `split_perm_nr_zbars` for a column that route cannot decide
/// on its own, so that column's `r` is this function's to the bit.
/// `split_exact`'s refit route (`run_split_perm`) reaches the same
/// computation through `split_column_r` via `split_unit`, not through this
/// function.
fn split_half_r(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    sp: &SplitIdx,
    w_norm: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
) -> f64 {
    split_column_r(&prepare_split(x, sp, w_norm), sp, &|i| y[i], k, keep)
}

/// Per-split state of a route, built once per split by
/// [`split_zbars_columns`] after the split's preparation and shared
/// read-only by that split's columns. The primal route needs nothing beyond
/// the prepared split.
pub(crate) enum SplitBlock<'s> {
    /// No per-split state.
    Primal(std::marker::PhantomData<&'s ()>),
    /// n-space Gram route: `G`, its norms and `M = X̃_te X̃_tr'` of this split.
    Nspace(crate::dual_route::NspaceSplit),
    /// The split's p-space Gram block on the training half.
    GramP(SplitGram<'s>),
}

/// The block of `route` for one prepared split, its precompute run under
/// `par` ([`crate::fit::par_fixed`]). `Special` never reaches a
/// driver; handed one, it gets the primal block. `Nspace` builds `G`, its
/// norms and `M` once per split; `GramP` builds `C = X̃_tr'X̃_tr` of the
/// training half ([`SplitGram`]) once per split.
pub(crate) fn split_block(route: ReplicateRoute, prep: &PreparedSplit, par: Par) -> SplitBlock<'_> {
    match route {
        ReplicateRoute::Nspace => {
            SplitBlock::Nspace(crate::dual_route::NspaceSplit::new(prep, par))
        }
        ReplicateRoute::GramP => SplitBlock::GramP(SplitGram::new(prep, par)),
        ReplicateRoute::Special | ReplicateRoute::Primal => {
            SplitBlock::Primal(std::marker::PhantomData)
        }
    }
}

/// One (split, column) unit: the split-half `r` for the outcome `y_of`,
/// before the clamp. A split whose prepared training half is not finite
/// gives `r = 0` on every route (the failed per-half fit's value). The
/// primal arm is [`split_column_r`]; a route that cannot certify a unit
/// falls back to it.
pub(crate) fn split_unit(
    block: &SplitBlock<'_>,
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
    k: usize,
    keep: Option<usize>,
) -> f64 {
    if !prep.x_finite {
        return 0.0;
    }
    match block {
        SplitBlock::Primal(_) => split_column_r(prep, sp, y_of, k, keep),
        SplitBlock::Nspace(ns) => {
            debug_assert!(k >= 1 && keep.is_none() && prep.w_tr.is_none());
            crate::dual_route::split_columns_r_nspace(sp, ns, 1, &|_, i| y_of(i), k)[0]
                .unwrap_or_else(|| split_column_r(prep, sp, y_of, k, None))
        }
        SplitBlock::GramP(gram) => split_column_r_gram_p(gram, prep, sp, y_of, k, keep),
    }
}

/// The `n_cols` split-half `r` values of one prepared split on the
/// n-space route, before the clamp and Fisher z. The columns go through
/// `dual_route::split_columns_r_nspace` in runs of
/// `dual_route::NSPACE_BATCH` consecutive columns from column 0, one call
/// per run, so the per-replicate products with `G` and `M` become one
/// product per run; the runs are fixed by `n_cols` alone, so the bits do
/// not depend on the thread count. The columns a run hands back undecided
/// then run the Primal arm (`split_column_r`) in a second indexed map over
/// just those columns, so a fallback-heavy input keeps one task per column
/// rather than one per run. Each column's value is computed from its own
/// inputs alone, whichever worker computes it. `prep.x_finite` is the
/// caller's precondition.
pub(crate) fn split_columns_nspace(
    prep: &PreparedSplit,
    sp: &SplitIdx,
    ns: &crate::dual_route::NspaceSplit,
    n_cols: usize,
    k: usize,
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
) -> Vec<f64> {
    let batch = crate::dual_route::NSPACE_BATCH;
    let runs = crate::resample::map_indexed(n_cols.div_ceil(batch), |run| {
        let c0 = run * batch;
        let ys: Vec<Col<f64>> = (c0..n_cols.min(c0 + batch)).map(column_y).collect();
        crate::dual_route::split_columns_r_nspace(sp, ns, ys.len(), &|j, i| ys[j][i], k)
    });
    let decided: Vec<Option<f64>> = runs.into_iter().flatten().collect();
    let undecided: Vec<usize> = (0..n_cols).filter(|&c| decided[c].is_none()).collect();
    let fallbacks = crate::resample::map_indexed(undecided.len(), |u| {
        let yc = column_y(undecided[u]);
        split_column_r(prep, sp, &|i| yc[i], k, None)
    });
    let mut out: Vec<f64> = decided.into_iter().map(|r| r.unwrap_or(0.0)).collect();
    for (c, r) in undecided.into_iter().zip(fallbacks) {
        out[c] = r;
    }
    out
}

/// Per-column `z̄` of `split_exact`'s refit route on `route`: the split
/// driver, splits outer and replicate columns inner (the PLS3 primal
/// route's shape, with one difference): the splits run one at a time, so
/// exactly one prepared split and its block ([`split_block`] under
/// [`crate::fit::par_fixed`]) are alive, and the `n_cols` columns of
/// that split map in parallel through [`split_unit`], the
/// same width the replicate-outer loop had. On the `Nspace` route the
/// columns go through [`split_columns_nspace`] instead, in fixed runs with
/// the fallbacks mapped per column. `column_y(c)` builds column `c` (raw,
/// length n) inside the unit (or run) that needs it, so no `B·n` buffer is
/// held.
/// `k <= p` is the caller's precondition; `k = 0` is reachable and gives
/// `r = 0` in every unit.
///
/// On the Primal route, byte-identical to `mean_fisher_z` over
/// `split_half_correlations` per column: each `(split, column)` value is
/// `split_half_r`'s on the same inputs, clamped and `atanh`-ed where
/// `mean_fisher_z` does it, and `zbars_over_splits` sums splits ascending
/// and divides by J once. One exception: `mean_fisher_z` sums from -0.0 and
/// `zbars_over_splits` from +0.0, so a statistic whose every term is -0.0
/// comes out +0.0. No p-value can change.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_zbars_columns(
    route: ReplicateRoute,
    x: MatRef<'_, f64>,
    splits: &[SplitIdx],
    w_norm: Option<ColRef<'_, f64>>,
    k: usize,
    keep: Option<usize>,
    n_cols: usize,
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
) -> Vec<f64> {
    debug_assert!(k <= x.ncols(), "k <= p is the caller's precondition");
    let par = crate::fit::par_fixed();
    // `false`: splits run one at a time; the parallelism is over columns.
    zbars_over_splits(splits, n_cols, false, |sp: &SplitIdx| {
        let prep = prepare_split(x, sp, w_norm);
        let block = split_block(route, &prep, par);
        // ±0.9999 pre-atanh clamp mirrored from nb_test (change together).
        let fisher_z = |r: f64| r.clamp(-0.9999, 0.9999).atanh();
        match &block {
            SplitBlock::Nspace(ns) if prep.x_finite => {
                split_columns_nspace(&prep, sp, ns, n_cols, k, column_y)
                    .into_iter()
                    .map(fisher_z)
                    .collect()
            }
            _ => crate::resample::map_indexed(n_cols, |c| {
                let yc = column_y(c);
                fisher_z(split_unit(&block, &prep, sp, &|i| yc[i], k, keep))
            }),
        }
    })
}

/// Pearson r of `a(0)..a(n−1)` and `b(0)..b(n−1)` with the crate's
/// degenerate-input convention: exactly `0.0` when either vector is
/// constant to rounding (`constant_to_rounding` on its [`scaled_moments`]),
/// otherwise the correlation clamped to `[-1, 1]`. The split-half statistic
/// of `split_exact`'s refit route and of `split_nb` (`split_half_r`), and
/// PLS3's `pls3_signal_test::pearson_r_guarded`. `n` must be positive.
pub(crate) fn guarded_pearson(n: usize, a: impl Fn(usize) -> f64, b: impl Fn(usize) -> f64) -> f64 {
    let ma = scaled_moments(n, &a, None);
    let mb = scaled_moments(n, &b, None);
    if ma.is_constant(n) || mb.is_constant(n) {
        return 0.0;
    }
    pearson_scaled(n, a, b, &ma, &mb)
}

/// Pearson r of `a` and `b` from their [`scaled_moments`], clamped to
/// `[-1, 1]`: [`pearson_from_moments`] on `a/s_a` and `b/s_b`. The caller
/// has already applied the constant test.
pub(crate) fn pearson_scaled(
    n: usize,
    a: impl Fn(usize) -> f64,
    b: impl Fn(usize) -> f64,
    ma: &ScaledMoments,
    mb: &ScaledMoments,
) -> f64 {
    pearson_from_moments(
        n,
        |i| a(i) * ma.inv,
        |i| b(i) * mb.inv,
        (ma.mean, ma.ss),
        (mb.mean, mb.ss),
    )
}

/// Pearson r from the moments `centered_moments` returned, clamped to
/// `[-1, 1]`. The caller has already applied `constant_to_rounding`.
/// Callers reach it through [`pearson_scaled`].
pub(crate) fn pearson_from_moments(
    n: usize,
    s: impl Fn(usize) -> f64,
    y: impl Fn(usize) -> f64,
    (s_mean, ss_s): (f64, f64),
    (y_mean, ss_y): (f64, f64),
) -> f64 {
    let cross: f64 = (0..n).map(|i| (s(i) - s_mean) * (y(i) - y_mean)).sum();
    (cross / (ss_s * ss_y).sqrt()).clamp(-1.0, 1.0)
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn run_split_nb(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use crate::resample::split_sizes;
    let n = x.nrows();
    let (n_train, n_test) = split_sizes(n, k);

    // Raw (X, y) and weights flow into split_half_correlations, which does the
    // per-half weighted-standardize-then-√w (the √w row-scaling convention)
    // internally.
    let splits = draw_splits(n, k, n_splits, rng)?;
    let r_splits = split_half_correlations(x, y, k, &splits, w_norm, opts.keep);
    let (p, mean_r, _t_stat, _df) = nb_test(&r_splits, n_train, n_test);

    // ρ̂ additionally needs unweighted input; the n_test floor is nb_rho_hat's.
    let rho_hat = if w_norm.is_none() {
        nb_rho_hat(&r_splits, n_test)
    } else {
        None
    };
    Ok(RunResult {
        pvalue: p,
        statistic: mean_r,
        rho_hat,
    })
}

/// NB t-test on Fisher-z transforms: the corrected resampled t-test of
/// Nadeau & Bengio (2003), "Inference for the Generalization Error",
/// Machine Learning 52:239–281.
///
/// Shared with `pls3_signal_test::pls3_confirmatory_test`, whose `split_nb`
/// runs the identical statistic on PLS3's held-out LV correlations — the two
/// families differ in how `r` is produced, not in how it is tested.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub(crate) fn nb_test(stats: &Col<f64>, n_train: usize, n_test: usize) -> (f64, f64, f64, f64) {
    let j = stats.nrows() as f64;
    // Fisher-z transform. The ±0.9999 pre-atanh clamp is mirrored in
    // split_perm_nr_zbars, in split_zbars_columns, in run_split_nb's
    // rho_hat block below, in dual_route::pls3_split_zbars_columns, and in
    // the test-only mean_fisher_z: change together. This site owns the
    // explanation (keeps z̄ finite at |r| = 1).
    let z_vec: Vec<f64> = (0..stats.nrows())
        .map(|i| stats[i].clamp(-0.9999, 0.9999).atanh())
        .collect();
    let z_mean: f64 = z_vec.iter().sum::<f64>() / j;
    let z_var: f64 = z_vec.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / (j - 1.0);
    let z_std = z_var.sqrt();
    // Clamp se to the degeneracy floor rather than branching to an exact p=0:
    // a vanishing spread (all split-r identical) still yields a finite, tiny p
    // through the t survival function instead of an unattainable exact zero.
    // The z_mean ≤ 0 side stays conservative — t ≤ 0 maps to p ≥ 0.5.
    let se = (z_std * (1.0 / j + n_test as f64 / n_train as f64).sqrt()).max(1e-15);
    let t = z_mean / se;
    let p = crate::linalg::t_sf(t, j - 1.0);
    (p, z_mean.tanh(), t, j - 1.0)
}

/// ρ̂, the split-stability estimate reported alongside a `split_nb` p-value.
///
/// `clip(1 − s²/σ₀², 0, 1)` with `σ₀² = 1/(n_test−3)` and `s²` the ddof-1
/// variance of the Fisher-z values. `None` when `n_test < 4`, where the ruler
/// is undefined. Callers add their own eligibility rules on top: PLS1 returns
/// `None` under weights, since the ruler assumes unweighted input.
///
/// `s²` is the same quantity `nb_test` computes internally for its own SE,
/// recomputed here rather than threaded out: it is cheap next to the
/// resampling loop that produced `stats`, and it keeps `nb_test`'s signature
/// free of a quantity `nb_test` itself does not need.
pub(crate) fn nb_rho_hat(stats: &Col<f64>, n_test: usize) -> Option<f64> {
    if n_test < 4 {
        return None;
    }
    let j = stats.nrows() as f64;
    // ±0.9999 pre-atanh clamp mirrored from nb_test above (change together) —
    // nb_test owns the explanation.
    let z_vec: Vec<f64> = (0..stats.nrows())
        .map(|i| stats[i].clamp(-0.9999, 0.9999).atanh())
        .collect();
    let z_mean: f64 = z_vec.iter().sum::<f64>() / j;
    let s2: f64 = z_vec.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / (j - 1.0);
    let sigma0_sq = 1.0 / (n_test as f64 - 3.0);
    Some((1.0 - s2 / sigma0_sq).clamp(0.0, 1.0))
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn run_split_perm(
    route: ReplicateRoute,
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_perm: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    let n = x.nrows();
    // The J splits are drawn once and held fixed across all B permutation
    // replicates. Redrawing them per replicate would fold split-to-split
    // scatter into the null, so the reference distribution would stop
    // isolating the y–X association the observed statistic measures.
    // split_perm_nr_zbars holds its splits fixed the same way.
    let splits = draw_splits(n, k, n_splits, rng)?;

    // One child seed per null column, drawn right after the splits, in
    // `parallel_for_each_seeded`'s order, so every route sees the same
    // permutations at a given seed.
    let seeds = crate::rng::child_seeds(rng, n_perm);
    let cols = crate::resample::Columns { y, seeds: &seeds };

    // Raw (X, y, weights) flow into the per-split preparation (the √w
    // row-scaling convention: weighted-standardize-then-√w); permuted y
    // rows, weights tied to row positions (w[i] always pairs with
    // destination row i).
    let z = split_zbars_columns(route, x, &splits, w_norm, k, opts.keep, cols.len(), &|c| {
        cols.column(c)
    });
    let (z_bar_obs, null_zbars) = (z[0], &z[1..]);

    // A non-finite null statistic counts as an exceedance so it biases p
    // upward, never downward (mirrors run_raw_perm and run_split_perm_nr;
    // change together). A failed per-half fit already degrades to r = 0
    // inside split_unit rather than surfacing here.
    let exceedances = null_zbars
        .iter()
        .filter(|v| !v.is_finite() || **v >= z_bar_obs)
        .count();
    let p = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);

    Ok(RunResult {
        pvalue: p,
        statistic: z_bar_obs.tanh(),
        rho_hat: None,
    })
}

/// `z̄`, the mean Fisher-z of a vector of split-half correlations, summed
/// left to right and divided once: the replicate-outer reference arithmetic
/// that `zbars_over_splits` reproduces per column (up to the sign of an
/// all-zero sum). `tanh` of this is the statistic `split_nb` and
/// `split_exact` both report, and on the permutation routes it is also the
/// scale the null comparison happens on:
/// mean-of-z is not a monotone function of mean-of-r, so comparing on one
/// scale and reporting the other would give a p-value for a different test.
#[cfg(test)]
pub(crate) fn mean_fisher_z(r: &Col<f64>) -> f64 {
    let j = r.nrows();
    if j == 0 {
        return 0.0;
    }
    // ±0.9999 pre-atanh clamp mirrored from nb_test (change together) —
    // nb_test owns the explanation.
    (0..j)
        .map(|i| r[i].clamp(-0.9999, 0.9999).atanh())
        .sum::<f64>()
        / j as f64
}

// ──────────────────────────────────────────────────────────────────────────────
// `split_exact`'s no-refit route (`run_split_perm_nr` / `split_perm_nr_zbars`)
// ──────────────────────────────────────────────────────────────────────────────

/// Multiple of the worst-case rounding bound and of the NIPALS relative floor
/// that `split_perm_nr_zbars` requires of a column before it trusts its own
/// truncation decision (see "Truncation and degenerate halves" there).
const NR_RESOLVE_BAND: f64 = 8.0;

/// Multiple of the NIPALS `1e-14` absolute exits that `split_perm_nr_zbars`
/// requires of its lower bounds on `‖X̃_tr'z‖` and `t't`.
const NR_ABS_BAND: f64 = 100.0;

/// Per-column z̄ values for `split_perm_nr` (length `n_perm + 1`; column 0 is
/// the observed y, columns `1..=n_perm` are permutation nulls). Factored out of
/// `run_split_perm_nr` so the equivalence test (below) can compare every
/// column against an honest per-split, per-column refit, not just the final
/// p-value — see the equivalence test below.
///
/// # Why no refits — the identity this function exploits
/// At K = 1, `pls1_fit` returns `w ∝ +X̃_tr'y_tr` with `p'w = 1`, so the
/// fitted coefficient is a positive scalar times `X̃_tr'y_tr`. The reported
/// statistic is `corr(X̃_te·coef, y_te)`, and correlation is invariant to
/// positive scaling of either argument — including y's own standardization,
/// since centering y does not change `X̃_tr'y_tr` (`X̃_tr` columns are exactly
/// mean-zero on the training half, so the centering term is annihilated) and
/// scaling y by a positive constant only rescales `coef` by that same
/// constant. So `t_te ∝ X̃_te·X̃_tr'·y_tr` is a fixed linear map of y,
/// determined by X and the split alone — permuting y changes only the vector
/// the map is applied to, and no per-column fit is needed. K ≥ 2 breaks this
/// (deflation makes component 2 depend on component 1's y-dependent scores),
/// which is why this method is K = 1 only.
///
/// # Under weights
/// The identity survives the √w row-scaling convention intact, but the map is not the same
/// map. `X̃_tr` is now `diag(√w_tr)·X_std,tr` with `X_std,tr` standardized on
/// *weighted* train-half moments, and the train y that `pls1_fit` sees is
/// `diag(√w_tr)·y_std,tr`. So the fixed map carries a second `diag(√w_tr)`
/// *inside* it — `X̃_tr'·diag(√w_tr)·y_tr` — which, unlike the y scale, is
/// not a positive scalar and cannot be pulled out front. It is still linear
/// in y and still independent of y, which is all the batching needs.
///
/// Centering of y still dies, for the weighted reason: the columns of
/// `X_std,tr` are weighted-mean-zero, so the centering term is
/// `Σᵢ wᵢ (xᵢⱼ − mean_w,ⱼ)/scaleⱼ = (Σw·mean_w,ⱼ − mean_w,ⱼ·Σw)/scaleⱼ = 0`
/// straight from `mean_w,ⱼ = Σᵢ wᵢ xᵢⱼ / Σᵢ wᵢ` — exactly the annihilation
/// the unweighted argument uses. Note this needs only that the same `w`
/// defines the mean and weights the sum; the mean-1 renormalization cancels
/// out and is not load-bearing here (it matters for matching `pls1_fit`'s
/// scale, not for this identity).
///
/// The test half gets its OWN renormalized weights: both `X_std,te` (train
/// moments, as always) and the raw test y are `√w_te`-scaled, so the Pearson
/// r is taken on `√w_te`-scaled data. That is what `split_half_correlations`
/// reports, and matching it is the point — a `√w_tr`-only implementation
/// would compute a correlation no other method in the crate reports.
/// Weights stay tied to their row positions and are never permuted, so the
/// permutation reference is unaffected.
///
/// # Truncation and degenerate halves
/// The identity says nothing about the refit route's early exits.
/// `fit::pls1_kernel` keeps no component when `‖X̃_tr'z‖` (`z` the standardized,
/// `√w_tr`-scaled training `y`) falls under
/// `fit::w_rel_floor(n_tr, p, ‖X̃_tr‖_F, ‖z‖)` or under `1e-14`, or when
/// `t't < 1e-14`; the test-half scores are then exactly zero and `r = 0`.
/// Here `t_te` is a map of the *raw* `y`, so when the training `y` is
/// orthogonal to `X̃_tr` up to rounding, `t_te` is rounding noise times
/// `‖y‖`, and no guard on the size of `t_te` can tell that noise from
/// signal: an absolute one reads it as signal once `y` is large. This route
/// cannot form `‖X̃_tr'z‖` (the route-B association order never forms
/// `X̃_tr'y` at all), so, like `dual_route::pls1_cv_r2_columns`, it decides
/// only the columns it can resolve and hands every other column of that
/// split to `split_half_r`, the refit route's own per-split computation,
/// whose `r` is then the refit route's to the bit.
///
/// In exact arithmetic on the stored inputs, `√w_tr ⊙ y_tr = ȳ·u + σ·z`
/// with `ȳ`, `σ` the mean and scale the refit route's `standardize1` call
/// returns and `u = √w_tr` (all ones unweighted), so
/// `t_te = σ·X̃_te g + ȳ·X̃_te h` with `g = X̃_tr'z` and `h = X̃_tr'u` (zero
/// only up to rounding). The computed, centered `t_te` departs from
/// `σ·P X̃_te g` (`P` the centering projection) by at most `err`, the sum of
/// the two GEMMs' and the standardization's rounding,
/// `(p + n_tr + 4)·ε·‖X̃_te‖_F·‖X̃_tr‖_F·(‖y_tr‖ + |ȳ|·‖u‖)`, the centering's,
/// `(n_te + 2)·ε·‖t_te‖`, and the mean term `|ȳ|·‖X̃_te‖_F·‖h‖` (`‖h‖` bounded
/// by its computed value plus `n_tr·ε·‖X̃_tr‖_F·‖u‖`). Hence
/// `‖g‖ ≥ (‖t_te,c‖ − err) / (σ·‖X̃_te‖_F)`, and `t't ≥ ‖g‖²/‖z‖²` (Cauchy-Schwarz
/// on `‖g‖² = z'X̃_tr g`). A column keeps the batched formula only when
/// `‖t_te,c‖ ≥ NR_RESOLVE_BAND·err`, that lower bound on `‖g‖` clears
/// `NR_RESOLVE_BAND` times the floor and `NR_ABS_BAND·1e-14`, and the one on
/// `t't` clears `NR_ABS_BAND·1e-14`. The refit route's computed `‖X̃_tr'z‖`
/// is within `n_tr·ε·‖X̃_tr‖_F·‖z‖` (at most the floor) of `‖g‖`, so past
/// the gate it keeps the component. Its scores are then a positive multiple
/// of `X̃_te g`, computed with an error that, relative to that multiple, is of
/// the order `(p + n_tr)·ε·‖X̃_te‖_F·‖X̃_tr‖_F·σ‖z‖` already inside `err`; with
/// the band at 8, the centered scores on both routes stay well above what
/// `constant_to_rounding` treats as constant, and both routes compute the
/// correlation. Two cases are decided directly because they are exact on
/// both routes: a degenerate test-half `y` (`constant_to_rounding` sees the
/// same bits on both routes) and a constant training `y` (`z` is exactly
/// zero, so the refit route's `X̃_tr'z` is an exact zero).
///
/// Every quantity the gate compares scales with `y` together (`t_te`,
/// `err`, `σ`) or not at all (`‖z‖`, the floor's ratio), so rescaling `y`
/// cannot move a column across it; `y` enters divided by a power of two
/// near its largest entry, so none of them overflows or underflows at any
/// magnitude of `y` either. On ordinary data the gate never fires:
/// an unstructured `y`'s `‖X̃_tr'z‖` sits many orders of magnitude above the
/// floor (see `fit::w_rel_floor`), and the lower bound gives up at most about
/// a factor `√p` of that. It fires on a training half whose `y` is
/// orthogonal, or nearly so, to `X̃_tr` (residualized or constructed
/// outcomes), on a test half where `X̃_te g` is nearly constant, and on a
/// `y` whose mean exceeds its spread by a factor near `1/((p + n)·ε)`,
/// where the batched map itself has lost the precision. Each such column
/// costs one refit of that split.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::items_after_statements)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)] // per-half √w row-scaling setup inflates the per-split closure
fn split_perm_nr_zbars(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_perm: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<Vec<f64>> {
    use crate::linalg::{
        row_subset, standardize1, standardize1_weighted, standardize_apply_rows, standardize_rows,
    };
    use crate::resample::{one_split, permute_indices, split_sizes};

    // Scope limits, checked here (this route's own runner), never in
    // split_half_correlations: that function backs only run_split_nb,
    // not run_split_perm, and knows nothing of this formula anyway. Guard,
    // not fallback: ineligible input errors rather than silently running
    // the run_split_perm route. Weights are *not* a scope limit: see
    // "Under weights" above.
    if k != 1 {
        return Err(PlsKitError::InvalidArgument(format!(
            "run_split_perm_nr requires k = 1 (got k={k}); use run_split_perm \
             (split_exact's refit route) for k > 1"
        )));
    }
    if opts.keep.is_some() {
        return Err(PlsKitError::InvalidArgument(
            "run_split_perm_nr does not support sparse keep; use run_split_perm \
             (split_exact's refit route) instead"
                .into(),
        ));
    }

    let n = x.nrows();
    // Mirror of draw_splits' n ≥ k+5 guard — change together.
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, n_test) = split_sizes(n, k);
    let p_features = x.ncols();
    let n_cols = n_perm + 1;

    // Draw the J splits once, fixed across all B permutation draws: the fixed
    // linear map in this function's "Why no refits" doc depends only on X and
    // the split, so every permutation reuses it. Drawn sequentially off the
    // parent RNG here rather than through draw_splits'
    // parallel_for_each_seeded: the two split_exact
    // routes are deliberately not stream-unified, so this route keeps the draw
    // order its equivalence tests are anchored to. Only the (tr, te) index
    // pairs are drawn here; standardization is deferred to per_split_z below so
    // peak memory stays O(np) plus the (B+1)-column blocks. By contrast,
    // materializing xs_tr/xs_te for all J splits up front would be O(J·n·p)
    // (≈2.1 GB at n=13365, p=400, J=50).
    // standardize_rows/standardize_apply_rows are pure functions of x[tr]/x[te] with no
    // RNG, so moving them into the (possibly parallel) per-split closure
    // changes nothing about determinism or the parallel-order guarantee.
    let splits: Vec<SplitIdx> = (0..n_splits)
        .map(|_| {
            let (tr, te) = one_split(n, n_train, rng);
            SplitIdx { tr, te }
        })
        .collect();

    // Outcome matrix Y (n x n_cols): column 0 is the observed y, columns
    // 1..=n_perm are independent permutations drawn exactly as run_split_perm
    // draws its null replicates (resample::permute_indices). No y
    // standardization anywhere — see the identity note above; it would only
    // rescale coef by a positive constant that the correlation divides out.
    //
    // Every column is `y` divided by one power of two near its largest
    // absolute value (`linalg::pow2_scale`, the standardizers' kernel), so
    // that the scores `t_te` and the gate's `‖y_tr‖`, `ȳ`, `σ` and `err`
    // below stay finite and normal at any magnitude of `y` (formed on the
    // raw `y`, `‖y_tr‖` is `inf` once `|y|` exceeds about `1e154`, and `0`
    // below about `1e-162`). The division is exact, every one of those
    // quantities is linear in `y`, and the gate and the correlation read
    // only their ratios, so wherever the raw quantities were in range this
    // changes no bit of the output. The fallback `split_half_r` gets the
    // same divided column; its `r` does not depend on the scale of `y`
    // either (the training `y` is standardized, the test one enters a
    // correlation through `scaled_moments`).
    let perms: Vec<Vec<usize>> = (0..n_perm).map(|_| permute_indices(n, rng)).collect();
    let y_max_abs = (0..n).map(|i| y[i].abs()).fold(0.0_f64, f64::max);
    let (_, y_inv) = crate::linalg::pow2_scale(y_max_abs);
    let y_mat = Mat::<f64>::from_fn(n, n_cols, |i, col| {
        if col == 0 {
            y[i] * y_inv
        } else {
            y[perms[col - 1][i]] * y_inv
        }
    });
    // The B·n index vectors are dead once y_mat is built.
    drop(perms);

    // Choose the association order once, outside the per-split loop: all
    // splits share (n_train, n_test) since split_sizes depends only on
    // (n, k). Route B wins when n_te·n_tr·(p+B) < n·p·B (flop counts of the
    // two GEMM association orders). No caller-facing knob: the cost model decides.
    let b_f = n_cols as f64;
    let route_b = (n_test as f64) * (n_train as f64) * (p_features as f64 + b_f)
        < (n as f64) * (p_features as f64) * b_f;

    // Per-split z contribution (unsummed over J): standardize the half
    // (train moments only, matching split_half_correlations), then the
    // batched two-GEMM product, then per-column
    // center/correlate/guard/clamp/atanh. Splits are independent given the
    // (tr, te) index pairs drawn above, so this parallelizes over splits with
    // no RNG involved — unlike split_half_correlations, which needs
    // parallel_for_each_seeded's seeded-per-iteration RNG because each split
    // there draws a fresh fit.
    let per_split_z = |sp: &SplitIdx| -> Vec<f64> {
        // Per-half weights renormalized to mean 1 within the half, weighted
        // moments, then √w row-scaling — every step mirrored from
        // split_half_correlations (change together; that function owns the
        // explanation of the weighted setup), including `renormalized_slice_weights`
        // for an all-zero or all-equal half.
        let w_tr = renormalized_slice_weights(w_norm, &sp.tr);
        let w_te = renormalized_slice_weights(w_norm, &sp.te);
        // Root taken once per row here, not once per (row, column) — with
        // n_cols = B+1 columns of Y the inline form would repeat it B times.
        let root = |w: &Col<f64>| Col::<f64>::from_fn(w.nrows(), |i| w[i].sqrt());
        let sw_tr = w_tr.as_ref().map(root);
        let sw_te = w_te.as_ref().map(root);

        // Gather, standardize and √w-scale each half in one pass: the same
        // moments, element expression and row factor as standardizing the
        // row subset and then scaling it (`linalg::standardize_rows`).
        let (xs_tr, mean, scale) = standardize_rows(
            x,
            &sp.tr,
            w_tr.as_ref().map(Col::as_ref),
            sw_tr.as_ref().map(Col::as_ref),
        );
        let xs_te = standardize_apply_rows(
            x,
            &sp.te,
            mean.as_ref(),
            scale.as_ref(),
            sw_te.as_ref().map(Col::as_ref),
        );

        let y_tr = row_subset(y_mat.as_ref(), &sp.tr);
        let y_te = row_subset(y_mat.as_ref(), &sp.te);
        let n_te = sp.te.len();

        // √w row-scaling of the outcomes. X̃_tr and X̃_te already
        // carry √w_tr and √w_te from `standardize_rows`. On the train side
        // the √w_tr on Y is the `diag(√w_tr)` that sits *inside* the linear
        // map (see "Under weights"). On the test side both the scores
        // (through X̃_te) and the raw test y carry √w_te, so the Pearson r
        // below is taken on √w_te-scaled data, matching what
        // split_half_correlations reports.
        let scale_rows = |m: Mat<f64>, sw: Option<&Col<f64>>| match sw {
            Some(sw) => Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * sw[i]),
            None => m,
        };
        let y_tr = scale_rows(y_tr, sw_tr.as_ref());
        let y_te = scale_rows(y_te, sw_te.as_ref());

        // Same two GEMM calls either way — only the operand grouping differs.
        // Seq inside the per-split worker: outer Rayon owns the threadpool.
        let seq = faer::Par::Seq;
        let t_te: Mat<f64> = if route_b {
            let m = crate::linalg::mat_mul(xs_te.as_ref(), xs_tr.transpose(), seq); // n_te x n_tr
            crate::linalg::mat_mul(m.as_ref(), y_tr.as_ref(), seq) // n_te x n_cols
        } else {
            let g = crate::linalg::mat_mul(xs_tr.transpose(), y_tr.as_ref(), seq); // p x n_cols
            crate::linalg::mat_mul(xs_te.as_ref(), g.as_ref(), seq) // n_te x n_cols
        };

        // Once per split: the inputs of the truncation gate (see
        // "Truncation and degenerate halves" above). `x_tr_fro` is the
        // `‖X̃_tr‖_F` that `fit::pls1_kernel` computes for its floor, from the
        // same matrix. `h = X̃_tr'u` is where the training `y`'s mean goes:
        // it is exactly zero only in exact arithmetic. Plain index-order
        // loops, so the gate's inputs cannot depend on the thread count.
        let n_tr = sp.tr.len();
        let x_tr_fro = xs_tr.norm_l2();
        let x_te_fro = xs_te.norm_l2();
        let u = |i: usize| sw_tr.as_ref().map_or(1.0, |sw| sw[i]);
        let u_norm = (0..n_tr).map(|i| u(i) * u(i)).sum::<f64>().sqrt();
        let h_norm = (0..p_features)
            .map(|j| {
                let h_j: f64 = (0..n_tr).map(|i| xs_tr[(i, j)] * u(i)).sum();
                h_j * h_j
            })
            .sum::<f64>()
            .sqrt();
        let eps = f64::EPSILON;
        let gemm_err = (p_features + n_tr + 4) as f64 * eps * x_te_fro * x_tr_fro;
        let h_bound = x_te_fro * (h_norm + n_tr as f64 * eps * x_tr_fro * u_norm);

        (0..n_cols)
            .map(|col| {
                let s = |i: usize| t_te[(i, col)];
                let yv = |i: usize| y_te[(i, col)];
                let ms = scaled_moments(n_te, s, None);
                let my = scaled_moments(n_te, yv, None);

                // A degenerate test-half y gives r = 0.0 on both routes: its
                // moments are computed here from the same bits by the same
                // helper as in split_half_r, so this decision is the refit
                // route's exactly. Never skipped or NaN.
                let r = if my.is_constant(n_te) {
                    0.0
                } else {
                    // The refit route's training-y standardization, call for
                    // call (split_half_r; change together).
                    let y_tr_raw = Col::<f64>::from_fn(n_tr, |i| y_mat[(sp.tr[i], col)]);
                    let (zs, y_bar, y_scale) = match w_tr.as_ref() {
                        Some(w) => standardize1_weighted(y_tr_raw.as_ref(), Some(w.as_ref())),
                        None => standardize1(y_tr_raw.as_ref()),
                    };
                    let z_norm = match sw_tr.as_ref() {
                        Some(sw) => Col::<f64>::from_fn(n_tr, |i| zs[i] * sw[i]).norm_l2(),
                        None => zs.norm_l2(),
                    };
                    if z_norm == 0.0 {
                        // Constant training y: X̃'z is an exact zero on the
                        // refit route, its fit keeps no component, and its
                        // scores are exactly zero.
                        0.0
                    } else {
                        let y_tr_norm = (0..n_tr)
                            .map(|i| y_tr[(i, col)] * y_tr[(i, col)])
                            .sum::<f64>()
                            .sqrt();
                        // `‖t_te‖` and `‖t_te,c‖` back in the units of
                        // `err`: `ms` holds the sums of `t_te / ms.s`, and
                        // `sqrt(sum)·ms.s` is exact.
                        let t_norm = ms.sq.sqrt() * ms.s;
                        let t_c = ms.ss.sqrt() * ms.s;
                        let err = gemm_err * (y_tr_norm + y_bar.abs() * u_norm)
                            + (n_te + 2) as f64 * eps * t_norm
                            + y_bar.abs() * h_bound;
                        let g_lower = (t_c - err) / (y_scale * x_te_fro);
                        let w_floor = crate::fit::w_rel_floor(n_tr, p_features, x_tr_fro, z_norm);
                        // `a >= b` conjunctions, so a NaN anywhere fails the
                        // gate and takes the fallback.
                        let resolved = t_c >= NR_RESOLVE_BAND * err
                            && g_lower >= NR_RESOLVE_BAND * w_floor
                            && g_lower >= NR_ABS_BAND * crate::fit::NIPALS_ABS_FLOOR
                            && (g_lower / z_norm).powi(2)
                                >= NR_ABS_BAND * crate::fit::NIPALS_ABS_FLOOR
                            && !ms.is_constant(n_te);
                        if resolved {
                            pearson_scaled(n_te, s, yv, &ms, &my)
                        } else {
                            // The refit route's own computation for this
                            // split and column.
                            let y_col = Col::<f64>::from_fn(n, |i| y_mat[(i, col)]);
                            split_half_r(x, y_col.as_ref(), 1, sp, w_norm, None)
                        }
                    }
                };
                // ±0.9999 pre-atanh clamp mirrored from nb_test (change
                // together): keeps the statistic identical to split_nb's and
                // z̄ finite at |r| = 1.
                r.clamp(-0.9999, 0.9999).atanh()
            })
            .collect()
    };

    Ok(zbars_over_splits(&splits, n_cols, true, per_split_z))
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn run_split_perm_nr(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_perm: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    let z_bar = split_perm_nr_zbars(x, y, k, n_perm, n_splits, w_norm, opts, rng)?;
    let z_bar_obs = z_bar[0];
    // Tie handling mirrors split_perm (signal_test.rs run_split_perm): >=,
    // and a non-finite null statistic counts as an exceedance so p is biased
    // upward, never downward.
    let exceedances = z_bar[1..]
        .iter()
        .filter(|zb| !zb.is_finite() || **zb >= z_bar_obs)
        .count();
    let p = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);

    Ok(RunResult {
        pvalue: p,
        statistic: z_bar_obs.tanh(),
        rho_hat: None,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// score test (Welch-Satterthwaite generalized χ²)
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::many_single_char_names)]
#[allow(clippy::unnecessary_wraps)] // signature must match other run_* helpers returning PlsKitResult
fn run_score(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
) -> PlsKitResult<RunResult> {
    use crate::linalg::{standardize, standardize1, standardize1_weighted, standardize_weighted};

    let n = x.nrows();

    // Standardize with weighted moments when weights are present (matching
    // pls1_fit's path); the √w row-scaling below then sits on top.
    // Pre-standardized input is used as given, copied only when it is not
    // column-major (`linalg::col_major_or_copy`).
    let x_copy = if opts.pre_standardized {
        crate::linalg::col_major_or_copy(x)
    } else {
        None
    };
    let xs_std: Option<Mat<f64>> = if opts.pre_standardized {
        None
    } else if w_norm.is_some() {
        Some(standardize_weighted(x, w_norm).0)
    } else {
        Some(standardize(x).0)
    };
    let xs: MatRef<'_, f64> = match (&xs_std, &x_copy) {
        (Some(m), _) | (None, Some(m)) => m.as_ref(),
        (None, None) => x,
    };

    let ys: Col<f64> = if opts.pre_standardized {
        Col::<f64>::from_fn(n, |i| y[i])
    } else if w_norm.is_some() {
        standardize1_weighted(y, w_norm).0
    } else {
        standardize1(y).0
    };

    // When weights are present, further row-scale the standardized data by √w'.
    // T_w = ||X̃'ỹ||² where X̃ = diag(√w')·X_std, ỹ = diag(√w')·y_std.
    // This equals the unweighted T on (X̃, ỹ).
    let (xs_w, ys_eff): (Option<Mat<f64>>, Col<f64>) = if let Some(w) = w_norm {
        (
            Some(Mat::<f64>::from_fn(n, xs.ncols(), |i, j| {
                xs[(i, j)] * w[i].sqrt()
            })),
            Col::<f64>::from_fn(n, |i| ys[i] * w[i].sqrt()),
        )
    } else {
        (None, ys)
    };
    let xs_eff: MatRef<'_, f64> = xs_w.as_ref().map_or(xs, Mat::as_ref);

    // T_obs = ||X'y||² = y'XX'y
    let par = crate::fit::par_fixed();
    let xy = crate::linalg::mat_vec(xs_eff.transpose(), ys_eff.as_ref(), par);
    let t_obs: f64 = (0..xy.nrows()).map(|i| xy[i].powi(2)).sum::<f64>();

    // Eigenvalues of the smaller Gram matrix (X'X for d≤n, XX' otherwise).
    let nn = xs_eff.nrows();
    let d = xs_eff.ncols();
    let lambdas: Col<f64> = if d <= nn {
        let gram = crate::linalg::mat_mul(xs_eff.transpose(), xs_eff, par);
        eigenvalues_symmetric(gram.as_ref(), par)
    } else {
        let gram = crate::linalg::mat_mul(xs_eff, xs_eff.transpose(), par);
        eigenvalues_symmetric(gram.as_ref(), par)
    };

    // Welch-Satterthwaite: T ~ a·χ²(df) approximately.
    let s1: f64 = (0..lambdas.nrows()).map(|i| lambdas[i]).sum();
    let s2: f64 = (0..lambdas.nrows()).map(|i| lambdas[i].powi(2)).sum();

    if s1.abs() < 1e-15 || s2 < 1e-30 {
        return Ok(RunResult {
            pvalue: 1.0,
            statistic: t_obs,
            rho_hat: None,
        });
    }

    let scale = s2 / s1;
    let df = s1 * s1 / s2;
    let p = chi2_sf(t_obs / scale, df);

    Ok(RunResult {
        pvalue: p,
        statistic: t_obs,
        rho_hat: None,
    })
}

/// Symmetric eigenvalues via faer's self-adjoint EVD on the lower triangle
/// (`linalg::self_adjoint_eigen`, pinned for byte-parity stability), with
/// an explicit `par`. Returns eigenvalues ascending.
fn eigenvalues_symmetric(a: MatRef<'_, f64>, par: faer::Par) -> Col<f64> {
    // An `Err` is non-convergence, not expected on PD/PSD Gram matrices. If
    // the matrix is degenerate (n=0 or all-zero), return a zero Col: the
    // caller guards s1 < 1e-15.
    match crate::linalg::self_adjoint_eigen(a, par) {
        Ok((lambda, _)) => lambda,
        Err(_) => Col::<f64>::zeros(a.nrows()),
    }
}

/// Survival function of `χ²(df)` at `x`. Uses the regularized upper incomplete
/// gamma Q(a, z) directly so extreme tails are not lost to a `1-(1-Q)` round trip.
#[allow(clippy::many_single_char_names)]
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let a = df / 2.0;
    let z = x / 2.0;
    gammainc_upper(a, z)
}

/// Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x). Numerical Recipes
/// §6.2. The continued-fraction branch returns Q directly (no `1−` round trip),
/// so χ² survival values below ~1e-16 stay representable instead of collapsing
/// to 0.0. The series branch evaluates P and returns its complement.
#[allow(clippy::many_single_char_names)]
fn gammainc_upper(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    let log_pref = a * x.ln() - x - crate::linalg::lgamma(a);
    if x < a + 1.0 {
        // Series expansion for the lower tail P(a, x); returns Q = 1 − P.
        let mut term = 1.0 / a;
        let mut sum = term;
        for i in 1_i32..200 {
            term *= x / (a + f64::from(i));
            sum += term;
            if term.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        1.0 - sum * log_pref.exp()
    } else {
        // Continued fraction evaluates Q(a, x) directly.
        let tiny = 1e-30;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1_i32..200 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-14 {
                break;
            }
        }
        h * log_pref.exp()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// universal-inference split-LR e-value
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn run_e(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use crate::fit::{pls1_fit_impl, FitOpts, KSpec};
    use crate::linalg::{col_row_subset, standardize1, standardize1_weighted};
    use crate::resample::{one_split, split_sizes};

    let n = x.nrows();
    // Mirror of split_half_correlations' guard — change together.
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, _) = split_sizes(n, k);
    let (tr, te) = one_split(n, n_train, rng);

    let y_tr = col_row_subset(y, &tr);
    let y_te = col_row_subset(y, &te);

    let prep = prepare_split(x, &SplitIdx { tr, te }, w_norm);
    let w_tr_ref = prep.w_tr.as_ref().map(Col::as_ref);

    let (ys_tr, y_mean, y_scale) = if let Some(w) = w_tr_ref {
        standardize1_weighted(y_tr.as_ref(), Some(w))
    } else {
        standardize1(y_tr.as_ref())
    };
    let n_te = y_te.nrows();
    let ys_te = Col::<f64>::from_fn(n_te, |i| (y_te[i] - y_mean) / y_scale);

    // √w' row-scaling of the outcomes on top of weighted standardization.
    let ys_tr = match w_tr_ref {
        Some(w) => Col::<f64>::from_fn(ys_tr.nrows(), |i| ys_tr[i] * w[i].sqrt()),
        None => ys_tr,
    };
    let ys_te = match prep.w_te.as_ref() {
        Some(w) => Col::<f64>::from_fn(n_te, |i| ys_te[i] * w[i].sqrt()),
        None => ys_te,
    };
    let (xs_tr, xs_te) = (prep.xs_tr, prep.xs_te);

    let m = pls1_fit_impl(
        xs_tr.as_ref(),
        ys_tr.as_ref(),
        KSpec::Fixed(k),
        None,
        FitOpts {
            pre_standardized: true,
            // check_n_eff: false — train-half refit; n_eff was validated at the
            // top-level entry, and the e-value remains valid at a truncated k_used.
            check_n_eff: false,
            keep: opts.keep,
            ..FitOpts::default()
        },
    )?;

    let par = crate::fit::par_fixed();
    let y_pred = crate::linalg::mat_vec(xs_te.as_ref(), m.coef.as_ref(), par);

    // Universal inference fixes the numerator density on the training half:
    // σ²_alt is the residual MLE from predicting the training X with the fitted
    // model, evaluated against training y. Computing it on the test half would
    // make the likelihood ratio data-dependent and break the e-value guarantee.
    let n_tr = ys_tr.nrows();
    let y_pred_tr = crate::linalg::mat_vec(xs_tr.as_ref(), m.coef.as_ref(), par);
    let sigma2_alt: f64 = (0..n_tr)
        .map(|i| (ys_tr[i] - y_pred_tr[i]).powi(2))
        .sum::<f64>()
        / n_tr as f64;

    // σ² under null: variance of test y
    let mean_te: f64 = (0..n_te).map(|i| ys_te[i]).sum::<f64>() / n_te as f64;
    let sigma2_null: f64 =
        (0..n_te).map(|i| (ys_te[i] - mean_te).powi(2)).sum::<f64>() / n_te as f64;

    let n_te_f = n_te as f64;
    // Gaussian log-likelihoods with MLE variance
    let ll = |sigma2: f64, residuals_sq_sum: f64| -> f64 {
        let s = sigma2.max(1e-30);
        -0.5 * n_te_f * (2.0 * std::f64::consts::PI * s).ln() - 0.5 * residuals_sq_sum / s
    };

    let resid_alt_ss: f64 = (0..n_te).map(|i| (ys_te[i] - y_pred[i]).powi(2)).sum();
    let resid_null_ss: f64 = (0..n_te).map(|i| (ys_te[i] - mean_te).powi(2)).sum();

    let ll_alt = ll(sigma2_alt, resid_alt_ss);
    let ll_null = ll(sigma2_null, resid_null_ss);

    let log_e = ll_alt - ll_null;
    // Clip e below 1 so that p = 1/e ≤ 1.
    let e = log_e.exp().max(1.0);
    let p = (1.0 / e).min(1.0);

    // opts unused by e method; pre_standardized has no effect (re-standardizes each half by
    // design).
    let _ = opts;

    Ok(RunResult {
        pvalue: p,
        statistic: log_e,
        rho_hat: None,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{centered_moments, constant_to_rounding, normalize_weights};
    use crate::test_support::{orthonormal_basis, project_off, signal_data, synth};

    /// Each method rejects on a strong-signal design, and reports its own
    /// name. `split_exact` at k = 2 takes the refit route; its p-values lie
    /// on the grid (c + 1) / 101, so `< 0.05` means at most four null
    /// exceedances.
    /// (n, d, snr, data seed), k, args, seed, method, strict p bound.
    type PowerRow = (
        (usize, usize, f64, u64),
        usize,
        ConfirmatoryArgs,
        u64,
        &'static str,
        f64,
    );

    #[test]
    fn each_method_rejects_under_signal() {
        let rows: [PowerRow; 4] = [
            (
                (80, 6, 5.0, 11),
                3,
                ConfirmatoryArgs::RawPerm {
                    n_perm: 200,
                    n_folds: 5,
                },
                7,
                "raw_perm",
                0.05,
            ),
            (
                (60, 5, 4.0, 17),
                2,
                ConfirmatoryArgs::SplitNb {
                    n_splits: 30,
                    force: false,
                },
                2,
                "split_nb",
                0.1,
            ),
            (
                (60, 5, 4.0, 23),
                2,
                ConfirmatoryArgs::SplitExact {
                    n_perm: 100,
                    n_splits: 20,
                },
                3,
                "split_exact",
                0.05,
            ),
            // Universal inference bounds P(reject | H0) by alpha exactly, so
            // under signal p < 0.5 is the modest claim.
            ((80, 5, 3.0, 41), 2, ConfirmatoryArgs::E, 5, "e", 0.5),
        ];
        for ((n, d, snr, data_seed), k, args, seed, method, bound) in rows {
            let (x, y) = synth(n, d, 3, snr, data_seed);
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(r.test_method, method);
            assert!(r.pvalue < bound, "{method}: p={}", r.pvalue);
        }
    }

    /// `rho_hat` is `Some` in [0, 1] only on an unweighted `split_nb` run
    /// with `n_test >= 4`; weights, a too-small test half, and `split_exact`
    /// give `None`.
    /// x, y, k, weights, args, reported method, `rho_hat.is_some()`.
    type RhoRow<'a> = (
        &'a Mat<f64>,
        &'a Col<f64>,
        usize,
        Option<&'a Col<f64>>,
        ConfirmatoryArgs,
        &'static str,
        bool,
    );

    #[test]
    fn rho_hat_follows_the_ruler() {
        let (x, y) = synth(60, 5, 3, 4.0, 17);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 1.5 } else { 0.5 });
        let (x10, y10) = synth(10, 5, 0, 0.0, 9);
        let nb = |n_splits, force| ConfirmatoryArgs::SplitNb { n_splits, force };
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 49,
            n_splits: 5,
        };
        // split_sizes(10, 5) -> (7, 3): n_test = 3 is below the ruler's
        // floor. `force`, because n = 10 trips the auto-gate's n floor and
        // would otherwise reroute to split_exact, which reports `None` for
        // its own reasons and would pass vacuously.
        let rows: [RhoRow<'_>; 4] = [
            (&x, &y, 2, None, nb(30, false), "split_nb", true),
            (&x, &y, 2, Some(&w), nb(30, false), "split_nb", false),
            (&x10, &y10, 5, None, nb(10, true), "split_nb", false),
            (&x, &y, 1, None, exact, "split_exact", false),
        ];
        for (x, y, k, w, args, method, some) in rows {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k,
                    weights: w.map(Col::as_ref),
                },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(2),
                    ..Default::default()
                },
            )
            .unwrap();
            let what = format!("{method} k={k} weighted={}", w.is_some());
            assert_eq!(r.test_method, method, "{what}");
            assert_eq!(r.rho_hat.is_some(), some, "{what}: {:?}", r.rho_hat);
            if let Some(rho) = r.rho_hat {
                assert!((0.0..=1.0).contains(&rho), "{what}: {rho}");
            }
        }
    }

    // ── raw_perm tests ───────────────────────────────────────────────────────

    #[test]
    fn raw_perm_rejects_leave_one_out_folds() {
        // n_folds == n makes every validation fold a single row: ss_tot for
        // that fold (spread of one point about its own mean) is always 0,
        // so the pooled CV R² is undefined. Must be rejected, not silently
        // returned as the degenerate statistic 0 / p 1.
        // n_folds > n gives n one-row folds plus (n_folds - n) empty folds
        // (linalg::fold_split). An empty fold contributes (0, 0) to the
        // pooled sums, so every non-empty validation fold is still a single
        // row: the same degeneracy, rejected the same way.
        let (x, y) = synth(10, 3, 0, 0.0, 5);
        for n_folds in [10, 11] {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: 1,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::RawPerm {
                        n_perm: 50,
                        n_folds,
                    },
                    seed: Some(3),
                    ..Default::default()
                },
            );
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "expected InvalidArgument for n_folds={n_folds} with n = 10, got {r:?}"
            );
        }
    }

    // ── raw_perm dual (Gram) route tests ────────────────────────────────

    /// The equivalence test the dual route rests on. Route A is the shipped
    /// primal path — one honest `pls1_cv_r2` per replicate column. Route B
    /// is `dual_route::pls1_cv_r2_columns`. Both are handed the *same*
    /// folds and the *same* permutation columns, so they evaluate the same
    /// quantity two ways and must agree on every column, not just on the
    /// final p-value. Small B and n keep the refitting route affordable.
    ///
    /// The fixture's shape must put `raw_perm_route` on the K = 1 Gram
    /// route, which is asserted first. The check then calls the public
    /// `pls1_confirmatory_test` (not just the isolated `dual_route` module)
    /// and compares its aggregate output against the reference built from
    /// primitives, so a bug in the production permutation loop cannot hide
    /// behind an in-test reimplementation. Replaying the fold draw and
    /// child seeds from the same seed also pins that no RNG draw sits
    /// between `resolve_seed` and the dispatch, and that the runner's
    /// child-seed stream is the one `parallel_for_each_seeded` draws.
    #[allow(clippy::many_single_char_names)]
    fn assert_raw_perm_dual_route_matches_primal(
        n: usize,
        p: usize,
        snr: f64,
        data_seed: u64,
        n_perm: usize,
        n_folds: usize,
        seed: u64,
    ) {
        use crate::resample::permute_indices;
        use rand::seq::SliceRandom;

        let (x, y) = synth(n, p, 3, snr, data_seed);
        let n_cols = n_perm + 1;

        // Rebuild the runner's fold draw off the same seed, then its child
        // seeds — `parallel_for_each_seeded` draws exactly this sequence.
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let folds = crate::linalg::fold_split(&indices, n_folds);
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let perms: Vec<Vec<usize>> = seeds
            .iter()
            .map(|s| permute_indices(n, &mut crate::rng::child_rng(*s)))
            .collect();
        let y_mat = Mat::<f64>::from_fn(n, n_cols, |i, col| {
            if col == 0 {
                y[i]
            } else {
                y[perms[col - 1][i]]
            }
        });

        assert_eq!(
            raw_perm_route(n, n_folds, p, n_perm, 1, None, false),
            ReplicateRoute::Special,
            "test premise: the shape must take the K = 1 Gram route \
             (n={n}, n_folds={n_folds}, p={p}, n_perm={n_perm})"
        );

        // Route A: the shipped primal path, one refit per column.
        let a: Vec<f64> = (0..n_cols)
            .map(|col| {
                let y_col = Col::<f64>::from_fn(n, |i| y_mat[(i, col)]);
                pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap()
            })
            .collect();

        // Call the public entry at the same seed: it consumes the identical
        // shuffle/child-seed sequence only if no draw sits between
        // `resolve_seed` and the dispatch to `run_raw_perm`, and it lands on
        // the K = 1 Gram route (asserted above).
        let prod = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm { n_perm, n_folds },
                seed: Some(seed),
                ..Default::default()
            },
        )
        .unwrap();
        let stat_scale = prod.statistic.abs().max(a[0].abs());
        let stat_rel = if stat_scale == 0.0 {
            0.0
        } else {
            (prod.statistic - a[0]).abs() / stat_scale
        };
        assert!(
            stat_rel < 1e-10,
            "production statistic diverges from reference: prod={} ref={} rel={stat_rel}",
            prod.statistic,
            a[0]
        );
        let ref_count = a[1..].iter().filter(|r| r.is_nan() || **r >= a[0]).count();
        let ref_pvalue = (ref_count as f64 + 1.0) / (n_perm as f64 + 1.0);
        // Both sides evaluate the identical `(exceedances + 1) / (n_perm + 1)`
        // formula on the same n_perm, so exact equality — not a tolerance —
        // is the right check here.
        #[allow(clippy::float_cmp)]
        let pvalues_match = ref_pvalue == prod.pvalue;
        assert!(
            pvalues_match,
            "production pvalue diverges from reference: prod={} ref={ref_pvalue}",
            prod.pvalue
        );

        // Route B: the Gram path, all columns at once.
        let b = crate::dual_route::pls1_cv_r2_columns(x.as_ref(), y_mat.as_ref(), &folds);

        assert_eq!(a.len(), b.len());
        // Pure relative tolerance — no absolute escape hatch. Scale by the
        // larger magnitude so the ratio stays defined when both are 0.
        for (col, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            let scale = av.abs().max(bv.abs());
            let rel = if scale == 0.0 {
                0.0
            } else {
                (av - bv).abs() / scale
            };
            assert!(rel < 1e-10, "col {col}: primal={av} dual={bv} rel={rel}");
        }

        // The p-value is a count of exceedances, so a near-tie could in
        // principle split even when every column agrees to 1e-10. Assert it
        // does not on these fixtures; a failure here is a report-and-stop,
        // not a tolerance to widen.
        let count = |v: &[f64]| v[1..].iter().filter(|r| r.is_nan() || **r >= v[0]).count();
        assert_eq!(count(&a), count(&b), "exceedance counts split on a tie");
    }

    #[test]
    fn raw_perm_dual_route_matches_primal_when_selected() {
        // n=60, p=3000, n_folds=5 ⇒ n_tr_max=48, B+1=50:
        // 48·(50+3000) = 146,400 < 3000·50 = 150,000 ⇒ dual is live.
        assert_raw_perm_dual_route_matches_primal(60, 3000, 4.0, 7, 49, 5, 11);
    }

    #[test]
    fn raw_perm_dual_route_degenerate_x_is_finite_and_matches_primal() {
        // Every X row identical ⇒ standardize's zero-variance branch makes
        // X̃_tr exactly zero ⇒ z'Gz = 0 ⇒ the NIPALS norm guard fires ⇒ the
        // prediction is zero. The pooled R² that follows is finite but not
        // 0: ss_res = Σ ys_val² while ss_tot = Σ(ys_val − mean_val)², and
        // ys_val carries the *training* fold's y moments, so its validation
        // mean is not 0. The claim under test is finiteness (never NaN) and
        // agreement with the primal route, not a particular value.
        let x = Mat::<f64>::from_fn(40, 8, |_, _| 3.0);
        let (_, y) = synth(40, 8, 3, 4.0, 5);
        let folds = crate::linalg::fold_split(&(0..40).collect::<Vec<_>>(), 4);
        let y_mat = Mat::<f64>::from_fn(40, 3, |i, _| y[i]);
        let dual = crate::dual_route::pls1_cv_r2_columns(x.as_ref(), y_mat.as_ref(), &folds);
        for (col, v) in dual.iter().enumerate() {
            assert!(v.is_finite(), "col {col} is not finite: {v}");
            let primal = {
                let y_col = Col::<f64>::from_fn(40, |i| y_mat[(i, col)]);
                pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap()
            };
            assert!(
                (v - primal).abs() < 1e-12,
                "col {col}: primal={primal} dual={v}"
            );
        }
    }

    /// A training fold whose standardized `y` is orthogonal to `X̃_tr` up to
    /// rounding: `fit::pls1_kernel` drops the only component (relative floor) and
    /// that fold predicts zero, while the Gram quantities `z'Gz` and `z'G²z`
    /// are pure rounding and cannot make the same call. The dual route must
    /// hand such a column to the primal kernel and agree with it. `X` has
    /// rank 3 at `p = 200`, so the orthogonal complement of the training
    /// fold's columns is large; `y` on the other folds' rows stays random.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn raw_perm_dual_route_matches_primal_when_a_fold_y_is_orthogonal_to_x() {
        use rand::RngExt;
        use rand::SeedableRng;
        let (n, p) = (40_usize, 200_usize);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(91);
        let a = Mat::<f64>::from_fn(n, 3, |_, _| rng.random_range(-1.0..1.0));
        let b = Mat::<f64>::from_fn(3, p, |_, _| rng.random_range(-1.0..1.0));
        let x: Mat<f64> = &a * &b;
        let folds = crate::linalg::fold_split(&(0..n).collect::<Vec<_>>(), 4);
        let train0: Vec<usize> = folds[1..].iter().flatten().copied().collect();

        // Orthonormal basis of span{1, X̃_tr} on fold 0's training rows
        // (Gram-Schmidt, each projection applied twice).
        let (xs_tr, _, _) =
            crate::linalg::standardize(crate::linalg::row_subset(x.as_ref(), &train0).as_ref());
        let m = train0.len();
        let basis = orthonormal_basis(
            Col::<f64>::from_fn(m, |_| 1.0).as_ref(),
            xs_tr.as_ref(),
            1e-8,
        );
        assert!(basis.len() <= 4, "test premise: X̃_tr has rank 3");
        // Eight columns, each with its own orthogonal training part. Which
        // sign rounding gives `z'Gz` is a coin flip per column; a negative
        // one gives the primal's zero prediction by luck, a positive one an
        // order-one prediction.
        let n_cols = 8;
        let y_val = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let mut y_mat = Mat::<f64>::from_fn(n, n_cols, |i, _| y_val[i]);
        for col in 0..n_cols {
            let mut u = Col::<f64>::from_fn(m, |_| rng.random_range(-1.0..1.0));
            project_off(&basis, &mut u);
            for (i, &r) in train0.iter().enumerate() {
                y_mat[(r, col)] = u[i];
            }

            // Test premise: the primal fold-0 fit truncates to zero components.
            let y_tr = Col::<f64>::from_fn(m, |i| y_mat[(train0[i], col)]);
            let (z, _, _) = crate::linalg::standardize1(y_tr.as_ref());
            let fit0 = pls1_fit(
                xs_tr.as_ref(),
                z.as_ref(),
                KSpec::Fixed(1),
                None,
                FitOpts {
                    pre_standardized: true,
                    check_n_eff: false,
                    ..FitOpts::default()
                },
            )
            .unwrap();
            assert_eq!(
                fit0.k_used, 0,
                "test premise: col {col}'s fold-0 fit must truncate"
            );
        }

        let dual = crate::dual_route::pls1_cv_r2_columns(x.as_ref(), y_mat.as_ref(), &folds);
        for (col, v) in dual.iter().enumerate() {
            let y_col = Col::<f64>::from_fn(n, |i| y_mat[(i, col)]);
            let primal = pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap();
            let scale = primal.abs().max(v.abs());
            let rel = if scale == 0.0 {
                0.0
            } else {
                (v - primal).abs() / scale
            };
            assert!(rel < 1e-10, "col {col}: primal={primal} dual={v} rel={rel}");
        }
    }

    // ── split_exact tests ────────────────────────────────────────────────────

    // Route selection is by input shape, not by a caller knob. Each case is
    // pinned bit-for-bit against the route it must land on, so a future edit
    // to the predicate fails here rather than silently switching routes.
    // The routes are not caller-selectable, so the comparator side calls
    // `run_split_perm_nr` / `run_split_perm` directly. `run_route`
    // reproduces exactly the preprocessing `pls1_confirmatory_test` performs
    // ahead of dispatch — weight normalization and `resolve_seed` — so the comparator sees the same
    // `(w_norm, rng)` state dispatch would hand the route. `SplitExact` args
    // never trip the `split_nb` auto-gate (the only other RNG consumer ahead
    // of dispatch), so no extra RNG draws sit between `resolve_seed` and the
    // route call on either side.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn split_exact_selects_route_from_input_shape() {
        let (x, y) = synth(60, 5, 3, 4.0, 17);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 1.5 } else { 0.5 });
        let (n_perm, n_splits) = (49_usize, 5_usize);
        let n = 60_usize;

        let run_exact = |k: usize, weights: Option<ColRef<'_, f64>>, keep, seed: u64| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k,
                    weights,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
                    seed: Some(seed),
                    keep,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let run_route = |k: usize, weights: Option<ColRef<'_, f64>>, keep, refit: bool| {
            let (w_norm, _) = crate::fit::validate_and_normalize_weights(weights, n, k).unwrap();
            let (_, mut rng) = crate::rng::resolve_seed(Some(2)).unwrap();
            let opts = ConfirmatoryTestOpts {
                keep,
                ..Default::default()
            };
            if refit {
                run_split_perm(
                    ReplicateRoute::Primal,
                    x.as_ref(),
                    y.as_ref(),
                    k,
                    n_perm,
                    n_splits,
                    w_norm.as_ref().map(Col::as_ref),
                    &opts,
                    &mut rng,
                )
                .unwrap()
            } else {
                run_split_perm_nr(
                    x.as_ref(),
                    y.as_ref(),
                    k,
                    n_perm,
                    n_splits,
                    w_norm.as_ref().map(Col::as_ref),
                    &opts,
                    &mut rng,
                )
                .unwrap()
            }
        };

        // k=1, dense ⇒ no-refit route, weighted or not.
        for weights in [None, Some(w.as_ref())] {
            let a = run_exact(1, weights, None, 2);
            let b = run_route(1, weights, None, false);
            assert_eq!(a.test_method, "split_exact");
            assert_eq!(
                a.pvalue.to_bits(),
                b.pvalue.to_bits(),
                "weighted={} took the wrong route",
                weights.is_some()
            );
            assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
        }
        // The seed reaches the no-refit route: another seed draws other
        // splits, and the statistic (tanh of the observed column's z̄) only
        // matches when the splits do.
        assert_ne!(
            run_exact(1, None, None, 2).statistic.to_bits(),
            run_exact(1, None, None, 3).statistic.to_bits(),
            "different seed should (generally) draw different splits"
        );

        // k>1 and sparse keep each independently force the refit route,
        // weighted or not.
        for weights in [None, Some(w.as_ref())] {
            for (k, keep) in [(2, None), (1, Some(3))] {
                let e = run_exact(k, weights, keep, 2);
                let s = run_route(k, weights, keep, true);
                let what = format!("weighted={} k={k} keep={keep:?}", weights.is_some());
                assert_eq!(e.test_method, "split_exact");
                assert_eq!(
                    e.pvalue.to_bits(),
                    s.pvalue.to_bits(),
                    "{what} took the wrong route"
                );
                assert_eq!(e.statistic.to_bits(), s.statistic.to_bits(), "{what}");
            }
        }
    }

    // ── split_exact's no-refit route (split_perm_nr) tests ──────────────────

    // Test 1: the equivalence test. Route A is an
    // honest per-split, per-column pls1_fit refit; route B is the shipped
    // batched path (split_perm_nr_zbars). Same seed ⇒ same splits and same
    // permuted Y columns for both routes (each draws the identical sequence
    // of one_split/permute_indices calls from the same RNG stream), so they
    // evaluate the same quantity two ways and must agree — this is the
    // property the method rests on (see "Why no refits" on `split_perm_nr_zbars`).
    // Small B=49, J=5 so the refitting route is affordable in a unit test.
    //
    // Shared by two call sites below (not parameterized #[test] — the crate
    // has no test-matrix macro in use elsewhere) so BOTH association orders
    // from `split_perm_nr_zbars`' cost model are exercised: (n=60, p=5) takes
    // route A (route_b cost 30·30·55=49,500 > route A cost 60·5·50=15,000),
    // (n=60, p=40) takes route B (route_b cost 30·30·90=81,000 < route A cost
    // 60·40·50=120,000). `expect_route_b` re-derives the same cost expression
    // split_perm_nr_zbars uses internally and asserts which branch is live,
    // so the coverage claim is enforced rather than assumed.
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    #[allow(clippy::items_after_statements)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn assert_split_perm_nr_route_matches_honest_refit(
        n: usize,
        p: usize,
        snr: f64,
        data_seed: u64,
        n_perm: usize,
        n_splits: usize,
        seed: u64,
        expect_route_b: bool,
        weighted: bool,
    ) {
        use crate::linalg::{
            col_row_subset, normalize_weights, row_subset, standardize, standardize1,
            standardize1_weighted, standardize_apply, standardize_weighted,
        };
        use crate::resample::{one_split, permute_indices, split_sizes};

        let k = 1_usize;
        let (x, y) = synth(n, p, 3, snr, data_seed);
        let n_cols = n_perm + 1;

        // Globally mean-1-normalized weights, exactly what
        // pls1_confirmatory_test hands the runner. Deliberately unequal and
        // not aligned to the split boundary, so per-half renormalization
        // actually changes the numbers (a uniform w would pass even if the
        // implementation ignored weights entirely).
        let w_all: Option<Col<f64>> = weighted.then(|| degenerate_half_weights(n));
        let w_norm = w_all.as_ref().map(Col::as_ref);

        // Cost-model check (mirrors split_perm_nr_zbars' own comparison):
        // fails loudly if a future edit to the thresholds moves this
        // configuration to the other branch, so the coverage claim below
        // stays true rather than assumed.
        let (n_train, n_test) = split_sizes(n, k);
        let b_f = n_cols as f64;
        let route_b =
            (n_test as f64) * (n_train as f64) * (p as f64 + b_f) < (n as f64) * (p as f64) * b_f;
        assert_eq!(
            route_b, expect_route_b,
            "cost-model check: expected route_b={expect_route_b}, computed={route_b} \
             (n_train={n_train}, n_test={n_test}, p={p}, B+1={n_cols})"
        );

        // Route B (shipped): draws splits + Y from its own RNG stream.
        let (_, mut rng_b) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let z_bar_b = split_perm_nr_zbars(
            x.as_ref(),
            y.as_ref(),
            k,
            n_perm,
            n_splits,
            w_norm,
            &ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
                seed: Some(seed),
                ..Default::default()
            },
            &mut rng_b,
        )
        .unwrap();

        // Route A (reference, test-only): same seed ⇒ resolve_seed +
        // one_split/permute_indices draw identically, so re-deriving them
        // here reproduces route B's exact splits and Y columns.
        let (_, mut rng_a) = crate::rng::resolve_seed(Some(seed)).unwrap();
        struct Split {
            tr: Vec<usize>,
            te: Vec<usize>,
            xs_tr: Mat<f64>,
            xs_te: Mat<f64>,
            /// Per-half mean-1-renormalized weights; `None` when unweighted.
            w_tr: Option<Col<f64>>,
            w_te: Option<Col<f64>>,
        }
        let splits: Vec<Split> = (0..n_splits)
            .map(|_| {
                let (tr, te) = one_split(n, n_train, &mut rng_a);
                let x_tr = row_subset(x.as_ref(), &tr);
                let x_te = row_subset(x.as_ref(), &te);
                // The √w row-scaling convention, spelled out the long way (this is the
                // reference — it deliberately does not call the production
                // helper): renormalize each half's weights to mean 1,
                // standardize with weighted moments, then √w row-scale.
                let w_tr =
                    w_norm.map(|w| normalize_weights(col_row_subset(w, &tr).as_ref()).unwrap());
                let w_te =
                    w_norm.map(|w| normalize_weights(col_row_subset(w, &te).as_ref()).unwrap());
                let (xs_tr, mean, scale) = if let Some(w) = w_tr.as_ref() {
                    standardize_weighted(x_tr.as_ref(), Some(w.as_ref()))
                } else {
                    standardize(x_tr.as_ref())
                };
                let xs_te = standardize_apply(x_te.as_ref(), mean.as_ref(), scale.as_ref());
                let scale_rows = |m: Mat<f64>, w: Option<&Col<f64>>| match w {
                    Some(w) => {
                        Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * w[i].sqrt())
                    }
                    None => m,
                };
                let xs_tr = scale_rows(xs_tr, w_tr.as_ref());
                let xs_te = scale_rows(xs_te, w_te.as_ref());
                Split {
                    tr,
                    te,
                    xs_tr,
                    xs_te,
                    w_tr,
                    w_te,
                }
            })
            .collect();
        let perms: Vec<Vec<usize>> = (0..n_perm)
            .map(|_| permute_indices(n, &mut rng_a))
            .collect();
        let y_mat = Mat::<f64>::from_fn(n, n_cols, |i, col| {
            if col == 0 {
                y[i]
            } else {
                y[perms[col - 1][i]]
            }
        });

        let mut z_sum_a = vec![0.0_f64; n_cols];
        for sp in &splits {
            let n_te = sp.te.len();
            for col in 0..n_cols {
                let y_tr_col = Col::<f64>::from_fn(sp.tr.len(), |i| y_mat[(sp.tr[i], col)]);
                let (ys_tr, _, _) = if let Some(w) = sp.w_tr.as_ref() {
                    standardize1_weighted(y_tr_col.as_ref(), Some(w.as_ref()))
                } else {
                    standardize1(y_tr_col.as_ref())
                };
                // √w_tr on the standardized train y, matching the √w_tr
                // already baked into sp.xs_tr (the √w row-scaling convention).
                let ys_tr = match sp.w_tr.as_ref() {
                    Some(w) => Col::<f64>::from_fn(ys_tr.nrows(), |i| ys_tr[i] * w[i].sqrt()),
                    None => ys_tr,
                };
                let m = pls1_fit(
                    sp.xs_tr.as_ref(),
                    ys_tr.as_ref(),
                    KSpec::Fixed(1),
                    None,
                    FitOpts {
                        pre_standardized: true,
                        check_n_eff: false,
                        ..Default::default()
                    },
                )
                .unwrap();
                let scores_te: Col<f64> = &sp.xs_te * &m.coef;
                // Test-half y is raw-but-√w_te-scaled (never standardized) —
                // the correlation is taken on √w-scaled data on both sides.
                let y_te_col = Col::<f64>::from_fn(n_te, |i| match sp.w_te.as_ref() {
                    Some(w) => y_mat[(sp.te[i], col)] * w[i].sqrt(),
                    None => y_mat[(sp.te[i], col)],
                });
                let s_mean: f64 = (0..n_te).map(|i| scores_te[i]).sum::<f64>() / n_te as f64;
                let y_mean: f64 = (0..n_te).map(|i| y_te_col[i]).sum::<f64>() / n_te as f64;
                let ss_s: f64 = (0..n_te).map(|i| (scores_te[i] - s_mean).powi(2)).sum();
                let ss_y: f64 = (0..n_te).map(|i| (y_te_col[i] - y_mean).powi(2)).sum();
                // Degeneracy guard relative to each vector's own magnitude,
                // as in production (`constant_to_rounding`).
                let sq_s: f64 = (0..n_te).map(|i| scores_te[i].powi(2)).sum();
                let sq_y: f64 = (0..n_te).map(|i| y_te_col[i].powi(2)).sum();
                let tol = (2.0 * n_te as f64 * f64::EPSILON).powi(2);
                let r = if ss_s <= tol * sq_s || ss_y <= tol * sq_y {
                    0.0
                } else {
                    let cross: f64 = (0..n_te)
                        .map(|i| (scores_te[i] - s_mean) * (y_te_col[i] - y_mean))
                        .sum();
                    (cross / (ss_s * ss_y).sqrt()).clamp(-1.0, 1.0)
                };
                z_sum_a[col] += r.clamp(-0.9999, 0.9999).atanh();
            }
        }
        let z_bar_a: Vec<f64> = z_sum_a.iter().map(|s| s / n_splits as f64).collect();

        assert_eq!(z_bar_a.len(), z_bar_b.len());
        // Pure relative tolerance at 1e-10; do not silently relax it. No
        // absolute-tolerance escape hatch. Scale by the larger
        // magnitude so the ratio stays defined if both columns are exactly 0.
        for (col, (a, b)) in z_bar_a.iter().zip(z_bar_b.iter()).enumerate() {
            let scale = a.abs().max(b.abs());
            let rel = if scale == 0.0 {
                0.0
            } else {
                (a - b).abs() / scale
            };
            assert!(rel < 1e-10, "col {col}: route A={a} route B={b} rel={rel}");
        }
    }

    /// Both association orders, unweighted and weighted. n = 60, B+1 = 50:
    /// at p = 5 route B costs 30·30·55 = 49,500 > route A's 60·5·50 = 15,000,
    /// so route A is live; at p = 40 route B costs 30·30·90 = 81,000 < route
    /// A's 60·40·50 = 120,000, so route B is live (the branch production
    /// takes at n ≤ 320, p = 400, B = 1000). The
    /// weighted rows license the weighted derivation: the batched map folds
    /// `diag(√w_tr)` into a fixed linear map, the reference refits honestly
    /// under the √w row-scaling convention, and every one of the B+1 columns
    /// must agree.
    #[test]
    fn split_perm_nr_route_matches_honest_refit() {
        for (p, expect_route_b, weighted) in [
            (5, false, false),
            (5, false, true),
            (40, true, false),
            (40, true, true),
        ] {
            assert_split_perm_nr_route_matches_honest_refit(
                60,
                p,
                4.0,
                7,
                49,
                5,
                11,
                expect_route_b,
                weighted,
            );
        }
    }

    // Same case as above, degenerate: every X row identical ⇒ standardize's
    // zero-variance branch makes xs_tr exactly the zero matrix, so the
    // no-refit route's test-half scores are constant (zero) for every
    // split/column — the guard must fire and return r = 0.0, never NaN.
    #[test]
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::float_cmp)] // exact 0.0 is the point: the guard must never emit NaN
    fn split_perm_nr_guard_constant_scores_give_zero_correlation() {
        let n = 20;
        let d = 3;
        let x = Mat::<f64>::from_fn(n, d, |_, j| (j + 1) as f64); // every row identical
        let y = Col::<f64>::from_fn(n, |i| i as f64);
        let (n_perm, n_splits, k, seed) = (9_usize, 4_usize, 1_usize, 3_u64);

        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let z_bar = split_perm_nr_zbars(
            x.as_ref(),
            y.as_ref(),
            k,
            n_perm,
            n_splits,
            None,
            &ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
                seed: Some(seed),
                ..Default::default()
            },
            &mut rng,
        )
        .unwrap();
        for zb in &z_bar {
            assert_eq!(
                *zb, 0.0,
                "the no-refit route must return exactly 0, not NaN"
            );
        }
    }

    // ── split_exact routes on degenerate training halves and rescaled y ─────

    /// Replays `split_perm_nr_zbars`' draws off `seed` (the J splits, then the
    /// B permutations, all sequentially off the parent stream) and evaluates
    /// every column through the refit route's own per-split code,
    /// `split_half_correlations`. The result is the refit route's `z̄` on
    /// exactly the no-refit route's splits and columns. Also returns the splits
    /// and the refit route's per-split `r` for the observed column.
    #[allow(clippy::type_complexity)]
    fn refit_zbars_on_nr_draws(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        n_perm: usize,
        n_splits: usize,
        w_norm: Option<ColRef<'_, f64>>,
        seed: u64,
    ) -> (Vec<SplitIdx>, Vec<f64>, Col<f64>) {
        use crate::resample::{one_split, permute_indices, split_sizes};
        let n = x.nrows();
        let (n_train, _) = split_sizes(n, 1);
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits: Vec<SplitIdx> = (0..n_splits)
            .map(|_| {
                let (tr, te) = one_split(n, n_train, &mut rng);
                SplitIdx { tr, te }
            })
            .collect();
        let perms: Vec<Vec<usize>> = (0..n_perm).map(|_| permute_indices(n, &mut rng)).collect();
        let r_obs = split_half_correlations(x, y, 1, &splits, w_norm, None);
        let mut zbars = vec![mean_fisher_z(&r_obs)];
        for perm in &perms {
            let y_perm = Col::<f64>::from_fn(n, |i| y[perm[i]]);
            let r = split_half_correlations(x, y_perm.as_ref(), 1, &splits, w_norm, None);
            zbars.push(mean_fisher_z(&r));
        }
        (splits, zbars, r_obs)
    }

    /// The splits `split_perm_nr_zbars` draws off `seed`.
    fn nr_splits(n: usize, n_splits: usize, seed: u64) -> Vec<SplitIdx> {
        use crate::resample::{one_split, split_sizes};
        let (n_train, _) = split_sizes(n, 1);
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        (0..n_splits)
            .map(|_| {
                let (tr, te) = one_split(n, n_train, &mut rng);
                SplitIdx { tr, te }
            })
            .collect()
    }

    /// Rank-3 `X` (n × p) and a `y` whose rows in `tr` are orthogonal, up to
    /// rounding, to `span{√w_tr, X̃_tr}` (the √w-row-scaled training matrix the
    /// refit route builds for that half), so the refit route's
    /// training-half fit truncates to zero components. The other rows of `y`
    /// are random. `y` is then multiplied by `scale`.
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    fn design_with_orthogonal_train_half(
        n: usize,
        p: usize,
        tr: &[usize],
        w_norm: Option<ColRef<'_, f64>>,
        scale: f64,
        seed: u64,
    ) -> (Mat<f64>, Col<f64>) {
        use crate::linalg::{
            col_row_subset, normalize_weights, row_subset, standardize, standardize_weighted,
        };
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let a = Mat::<f64>::from_fn(n, 3, |_, _| rng.random_range(-1.0..1.0));
        let b = Mat::<f64>::from_fn(3, p, |_, _| rng.random_range(-1.0..1.0));
        let x: Mat<f64> = &a * &b;

        let m = tr.len();
        let w_tr: Option<Col<f64>> =
            w_norm.map(|w| normalize_weights(col_row_subset(w, tr).as_ref()).unwrap());
        let sw = Col::<f64>::from_fn(m, |i| w_tr.as_ref().map_or(1.0, |w| w[i].sqrt()));
        let x_tr = row_subset(x.as_ref(), tr);
        let (xs_tr, _, _) = match w_tr.as_ref() {
            Some(w) => standardize_weighted(x_tr.as_ref(), Some(w.as_ref())),
            None => standardize(x_tr.as_ref()),
        };
        // Orthonormal basis of span{√w, √w ⊙ X_std}.
        let sw_xs = Mat::<f64>::from_fn(m, p, |i, j| sw[i] * xs_tr[(i, j)]);
        let basis = orthonormal_basis(sw.as_ref(), sw_xs.as_ref(), 1e-8);
        assert!(basis.len() <= 4, "test premise: X̃_tr has rank 3");
        let mut u = Col::<f64>::from_fn(m, |_| rng.random_range(-1.0..1.0));
        project_off(&basis, &mut u);

        let mut y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0) * scale);
        for (i, &row) in tr.iter().enumerate() {
            // √w ⊙ y ⟂ the basis ⇔ y ⟂ it in the w-weighted inner product.
            y[row] = u[i] / sw[i] * scale;
        }
        (x, y)
    }

    fn assert_zbars_close(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len());
        for (col, (u, v)) in a.iter().zip(b.iter()).enumerate() {
            let scale = u.abs().max(v.abs());
            let rel = if scale == 0.0 {
                0.0
            } else {
                (u - v).abs() / scale
            };
            assert!(rel < 1e-10, "{what}: col {col}: {u} vs {v} rel={rel}");
        }
    }

    fn nr_zbars(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        n_perm: usize,
        n_splits: usize,
        w_norm: Option<ColRef<'_, f64>>,
        seed: u64,
    ) -> Vec<f64> {
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        split_perm_nr_zbars(
            x,
            y,
            1,
            n_perm,
            n_splits,
            w_norm,
            &ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
                seed: Some(seed),
                ..Default::default()
            },
            &mut rng,
        )
        .unwrap()
    }

    /// The `(p, weighted, route_b)` cases the degenerate-half tests cover:
    /// both association orders of `split_perm_nr_zbars`, with and without
    /// weights, at n = 40, B = 20, J = 4.
    const DEGENERATE_HALF_CASES: [(usize, bool, bool); 4] = [
        (5, false, false),
        (5, true, false),
        (60, false, true),
        (60, true, true),
    ];

    fn degenerate_half_weights(n: usize) -> Col<f64> {
        crate::linalg::normalize_weights(
            Col::<f64>::from_fn(n, |i| 0.25 + ((i * 7) % 5) as f64 * 0.5).as_ref(),
        )
        .unwrap()
    }

    /// A training half whose `y` is orthogonal to `X̃_tr`, with `y` on a large
    /// scale. The refit route's NIPALS fit drops the only component on that
    /// half (relative floor), so its `r` there is 0. The no-refit route's
    /// scores are a fixed linear map of the raw `y`, which on that half is
    /// rounding noise times `‖y‖`; a guard on their absolute size cannot tell
    /// that apart from signal once `y` is large, and the route would report
    /// the correlation of that noise with the test-half `y`.
    #[test]
    fn split_exact_no_refit_route_matches_refit_when_a_train_half_y_is_orthogonal_to_x() {
        let (n, n_perm, n_splits, seed) = (40_usize, 20_usize, 4_usize, 5_u64);
        let splits = nr_splits(n, n_splits, seed);
        let w = degenerate_half_weights(n);
        for (p, weighted, expect_route_b) in DEGENERATE_HALF_CASES {
            let w_norm = weighted.then_some(w.as_ref());
            let (x, y) = design_with_orthogonal_train_half(n, p, &splits[0].tr, w_norm, 1e8, 17);

            let (n_train, n_test) = crate::resample::split_sizes(n, 1);
            let b_f = (n_perm + 1) as f64;
            let route_b = (n_test as f64) * (n_train as f64) * (p as f64 + b_f)
                < (n as f64) * (p as f64) * b_f;
            assert_eq!(route_b, expect_route_b, "cost-model premise at p={p}");

            let (_, z_re, r_obs) =
                refit_zbars_on_nr_draws(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed);
            assert!(
                r_obs[0] == 0.0,
                "test premise: the refit route's fit on split 0 truncates (r = {})",
                r_obs[0]
            );
            let z_nr = nr_zbars(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed);
            assert_zbars_close(
                &z_nr,
                &z_re,
                &format!("p={p} weighted={weighted}: no-refit vs refit"),
            );
        }
    }

    /// Multiplying `y` by a positive constant changes no output of either
    /// `split_exact` route beyond rounding: the refit route standardizes the
    /// training `y`, and the no-refit route's scores scale with `y` but its
    /// statistic is a correlation. Checked on the degenerate design above
    /// (where the truncation decision is in play) and on an ordinary one.
    #[test]
    fn split_exact_routes_are_invariant_to_the_scale_of_y() {
        let (n, n_perm, n_splits, seed) = (40_usize, 20_usize, 4_usize, 5_u64);
        let splits = nr_splits(n, n_splits, seed);
        let w = degenerate_half_weights(n);
        for (p, weighted, _) in DEGENERATE_HALF_CASES {
            let w_norm = weighted.then_some(w.as_ref());
            let (x, y_orth) =
                design_with_orthogonal_train_half(n, p, &splits[0].tr, w_norm, 1.0, 17);
            let (_, y_plain) = synth(n, p, 3, 0.3, 23);
            for (label, y) in [("orthogonal half", &y_orth), ("ordinary", &y_plain)] {
                let z_nr = nr_zbars(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed);
                let (_, z_re, _) =
                    refit_zbars_on_nr_draws(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed);
                let refit_run = |y: ColRef<'_, f64>| {
                    let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
                    run_split_perm(
                        ReplicateRoute::Primal,
                        x.as_ref(),
                        y,
                        1,
                        n_perm,
                        n_splits,
                        w_norm,
                        &ConfirmatoryTestOpts::default(),
                        &mut rng,
                    )
                    .unwrap()
                };
                let base_run = refit_run(y.as_ref());
                for factor in [1e8, 1e-8] {
                    let what = format!("p={p} weighted={weighted} {label} y×{factor:e}");
                    let ys = Col::<f64>::from_fn(n, |i| y[i] * factor);
                    let z_nr_s = nr_zbars(x.as_ref(), ys.as_ref(), n_perm, n_splits, w_norm, seed);
                    assert_zbars_close(&z_nr, &z_nr_s, &format!("{what}, no-refit"));
                    let (_, z_re_s, _) = refit_zbars_on_nr_draws(
                        x.as_ref(),
                        ys.as_ref(),
                        n_perm,
                        n_splits,
                        w_norm,
                        seed,
                    );
                    assert_zbars_close(&z_re, &z_re_s, &format!("{what}, refit"));
                    let scaled_run = refit_run(ys.as_ref());
                    assert_zbars_close(
                        &[base_run.statistic],
                        &[scaled_run.statistic],
                        &format!("{what}, run_split_perm statistic"),
                    );
                    assert_eq!(
                        base_run.pvalue.to_bits(),
                        scaled_run.pvalue.to_bits(),
                        "{what}, run_split_perm p-value"
                    );
                }
            }
        }
    }

    /// Guarded correlation from moments of the raw (unscaled) vectors: the
    /// reference for the bit-identity test.
    #[allow(clippy::similar_names)]
    fn raw_guarded_pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len();
        let (av, bv) = (|i: usize| a[i], |i: usize| b[i]);
        let (a_mean, ss_a, sq_a) = centered_moments(n, av);
        let (b_mean, ss_b, sq_b) = centered_moments(n, bv);
        if constant_to_rounding(ss_a, sq_a, n) || constant_to_rounding(ss_b, sq_b, n) {
            return 0.0;
        }
        pearson_from_moments(n, av, bv, (a_mean, ss_a), (b_mean, ss_b))
    }

    /// Wherever the raw sums of squares are in range, forming them on each
    /// vector divided by a power of two changes no bit of the guarded
    /// correlation or of the constant decision, and the `ScaledMoments`
    /// sums are the raw ones times `1/s` or `1/s²` exactly. Outside that
    /// range the raw formulas read a varying vector as constant (`r = 0`);
    /// the scaled ones do not.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn guarded_pearson_matches_the_raw_formulas_bit_for_bit_in_range() {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(41);
        // In range means every raw sum and the product `ss_a·ss_b` under
        // the square root in `pearson_from_moments` are normal numbers.
        let mags = [1e-70, 3.1e-7, 0.5, 1.0, 1.7, 3.0, 1e3, 6.02e23, 1e70];
        for n in [5_usize, 17, 30, 200] {
            let base_a: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let noise: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let base_b: Vec<f64> = (0..n).map(|i| 0.6 * base_a[i] + noise[i]).collect();
            // Offsets exercise the mean; the last pair is constant to
            // rounding (a large constant plus sub-ulp jitter), and the
            // all-zero vector is constant too.
            let shapes: Vec<(Vec<f64>, Vec<f64>)> = vec![
                (base_a.clone(), base_b.clone()),
                (
                    base_a.iter().map(|v| v + 40.0).collect(),
                    base_b.iter().map(|v| v - 3.0).collect(),
                ),
                (
                    base_a.iter().map(|v| 1.0 + v * 1e-17).collect(),
                    base_b.clone(),
                ),
                (vec![0.0; n], base_b.clone()),
            ];
            for (a0, b0) in &shapes {
                for &fa in &mags {
                    for &fb in &mags {
                        let a: Vec<f64> = a0.iter().map(|v| v * fa).collect();
                        let b: Vec<f64> = b0.iter().map(|v| v * fb).collect();
                        let want = raw_guarded_pearson(&a, &b);
                        let got = guarded_pearson(n, |i| a[i], |i| b[i]);
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "n={n} fa={fa:e} fb={fb:e}: {got} vs {want}"
                        );
                        let m = scaled_moments(n, |i| a[i], None);
                        let (mean, ss, sq) = centered_moments(n, |i| a[i]);
                        assert_eq!((m.mean * m.s).to_bits(), mean.to_bits());
                        assert_eq!((m.ss.sqrt() * m.s).to_bits(), ss.sqrt().to_bits());
                        assert_eq!((m.sq.sqrt() * m.s).to_bits(), sq.sqrt().to_bits());
                        assert_eq!(m.is_constant(n), constant_to_rounding(ss, sq, n));
                    }
                }
            }
            // Out of range: the raw sums overflow (or underflow) and read a
            // varying vector as constant; the scaled ones give the in-range r.
            let r = guarded_pearson(n, |i| base_a[i], |i| base_b[i]);
            assert!(r != 0.0, "test premise: r = {r}");
            for f in [1e200, 1e-200, 1e300, 1e-300] {
                let b: Vec<f64> = base_b.iter().map(|v| v * f).collect();
                assert_eq!(
                    raw_guarded_pearson(&base_a, &b).to_bits(),
                    0.0_f64.to_bits(),
                    "test premise: the raw formulas read y×{f:e} as constant"
                );
                let got = guarded_pearson(n, |i| base_a[i], |i| b[i]);
                assert!((got - r).abs() < 1e-12, "n={n} y×{f:e}: {got} vs {r}");
            }
            // Either argument alone, from 1e±150 (the raw sums still in
            // range) out to where they overflow or underflow: the same r.
            for f in [1e-150, 1e150, 1e-200, 1e200, 1e-300, 1e300] {
                let a: Vec<f64> = base_a.iter().map(|v| v * f).collect();
                let b: Vec<f64> = base_b.iter().map(|v| v * f).collect();
                let got_a = guarded_pearson(n, |i| a[i], |i| base_b[i]);
                let got_b = guarded_pearson(n, |i| base_a[i], |i| b[i]);
                assert!((got_a - r).abs() < 1e-12, "n={n} x×{f:e}: {got_a} vs {r}");
                assert!((got_b - r).abs() < 1e-12, "n={n} y×{f:e}: {got_b} vs {r}");
            }
            // Two small vectors whose own sums are normal but whose product
            // `ss_a·ss_b` underflows: the raw formula divides by a zero
            // root and clamps to ±1.
            let (a, b): (Vec<f64>, Vec<f64>) = (
                base_a.iter().map(|v| v * 1e-140).collect(),
                base_b.iter().map(|v| v * 1e-140).collect(),
            );
            assert_eq!(
                raw_guarded_pearson(&a, &b).abs().to_bits(),
                1.0_f64.to_bits(),
                "test premise"
            );
            let got = guarded_pearson(n, |i| a[i], |i| b[i]);
            assert!((got - r).abs() < 1e-12, "n={n} both ×1e-140: {got} vs {r}");
        }
    }

    /// `split_exact` (both routes) and `split_nb` give the same statistic and
    /// p-value on `(X·a, y·b)` as on `(X, y)` for any positive `a`, `b`,
    /// including magnitudes where `Σy²` overflows or underflows. Power-of-two
    /// factors must leave every bit unchanged (every rescaling inside is
    /// exact); decimal ones only perturb rounding.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn split_statistics_are_invariant_to_extreme_scales_of_x_and_y() {
        let n = 60_usize;
        let (x, y) = synth(n, 5, 3, 1.0, 29);
        let w = Col::<f64>::from_fn(n, |i| if i % 3 == 0 { 2.0 } else { 0.5 });
        let run = |x: MatRef<'_, f64>, y: ColRef<'_, f64>, k, weights, args| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw { x, y, k, weights },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(8),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 99,
            n_splits: 10,
        };
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: 20,
            force: true,
        };
        let two = |e: i32| 2.0_f64.powi(e);
        for (k, args) in [(1_usize, exact), (2, exact), (1, nb), (2, nb)] {
            for weights in [None, Some(w.as_ref())] {
                let what = format!("k={k} {args:?} weighted={}", weights.is_some());
                let base = run(x.as_ref(), y.as_ref(), k, weights, args);
                assert!(
                    base.statistic.abs() > 0.05,
                    "{what}: test premise, statistic = {}",
                    base.statistic
                );
                for (fx, fy) in [
                    (1.0, 1e200),
                    (1.0, 1e-200),
                    (1e200, 1.0),
                    (1e-200, 1.0),
                    (1e-200, 1e200),
                    (1e200, 1e-200),
                    (1.0, two(665)),
                    (1.0, two(-665)),
                    (two(700), two(-700)),
                ] {
                    let xs = Mat::<f64>::from_fn(n, x.ncols(), |i, j| x[(i, j)] * fx);
                    let ys = Col::<f64>::from_fn(n, |i| y[i] * fy);
                    let got = run(xs.as_ref(), ys.as_ref(), k, weights, args);
                    let what = format!("{what} X×{fx:e} y×{fy:e}");
                    if fx.log2().fract() == 0.0 && fy.log2().fract() == 0.0 {
                        assert_eq!(
                            got.statistic.to_bits(),
                            base.statistic.to_bits(),
                            "{what}: statistic {} vs {}",
                            got.statistic,
                            base.statistic
                        );
                        assert_eq!(got.pvalue.to_bits(), base.pvalue.to_bits(), "{what}: p");
                    } else {
                        let tol = 1e-10 * base.statistic.abs();
                        assert!(
                            (got.statistic - base.statistic).abs() <= tol,
                            "{what}: statistic {} vs {}",
                            got.statistic,
                            base.statistic
                        );
                        assert!(
                            (got.pvalue - base.pvalue).abs() <= 1e-10 * base.pvalue.max(1e-300),
                            "{what}: p {} vs {}",
                            got.pvalue,
                            base.pvalue
                        );
                    }
                }
            }
        }
    }

    // ── split_nb auto-gate ───────────────────────────────────────────────────

    /// Single dominant factor: every column is the same latent `f` plus a tiny
    /// jitter, so σ₁ carries nearly all the energy and the stable rank sits
    /// just above 1 — the concentrated-spectrum half of the gate.
    #[allow(clippy::many_single_char_names)]
    fn synth_one_factor(n: usize, d: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let f = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let x = Mat::<f64>::from_fn(n, d, |i, _| f[i] + 0.01 * rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        (x, y)
    }

    fn gate_run(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        weights: Option<ColRef<'_, f64>>,
        n_splits: usize,
        force: bool,
    ) -> ConfirmatoryTestOutput {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x,
                y,
                k: 1,
                weights,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits, force },
                seed: Some(4242),
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn gate_reroutes_when_n_below_floor() {
        // n = 20 < 25, spectrum flat (iid) — the size clause fires. d = 10 so
        // neither the rank clause nor the column precheck can be the reason.
        let (x, y) = synth(20, 10, 0, 0.0, 5);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(r.test_method, "split_exact");
        let sr = r.stable_rank.expect("gate was evaluated");
        assert!(
            sr >= SPLIT_NB_GATE_MIN_STABLE_RANK,
            "n, not the spectrum, must be what fired here (stable_rank={sr})"
        );
    }

    /// The column precheck: `stable_rank ≤ ncols`, so at four columns the
    /// computed rank is not trustworthy even when it clears the floor. n = 60
    /// keeps the size clause out of it, and the rank assertion proves the
    /// precheck — not the rank clause — is what fired.
    #[test]
    fn gate_reroutes_on_narrow_x_despite_adequate_rank() {
        let (x, y) = synth(60, 4, 0, 0.0, 11);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(r.test_method, "split_exact");
        let sr = r.stable_rank.expect("gate was evaluated");
        assert!(
            sr >= SPLIT_NB_GATE_MIN_STABLE_RANK,
            "column count, not the spectrum, must be what fired here (stable_rank={sr})"
        );
    }

    #[test]
    fn gate_reroutes_when_stable_rank_below_floor() {
        let (x, y) = synth_one_factor(40, 5, 6);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 12, false);
        assert_eq!(r.test_method, "split_exact");
        let sr = r.stable_rank.expect("gate was evaluated");
        assert!(sr < SPLIT_NB_GATE_MIN_STABLE_RANK, "stable_rank={sr}");
        // n_perm is split_exact's own default; n_splits is what the caller asked for.
        assert_eq!(r.n_perm, Some(1000));
        assert_eq!(r.n_splits, Some(12));
    }

    #[test]
    fn gate_passes_flat_spectrum_at_adequate_n() {
        // d = 10 rather than the file's usual 5: at n = 60, d = 5 the stable
        // rank lands around 3.1, so the negative case would clear the floor by
        // a few percent and turn into a coin flip under any reseeding. Ten iid
        // columns put it near 7.
        let (x, y) = synth(60, 10, 0, 0.0, 7);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(r.test_method, "split_nb");
        // No rank assertion — `test_method == "split_nb"` at n = 60 already implies
        // the rank cleared the floor. What is worth pinning is that the
        // diagnostic is populated on the pass path too.
        assert!(r.stable_rank.is_some());
        // Reported counts still come from the (unrewritten) split_nb args.
        assert_eq!(r.n_perm, None);
        assert_eq!(r.n_splits, Some(20));
    }

    /// `force` overrides the rank clause and the column precheck alike (the
    /// precheck is one more OR term, not a hard block), and the rank is
    /// still reported.
    #[test]
    fn gate_force_runs_nb_on_flagged_design_and_still_reports_rank() {
        for (name, (x, y)) in [
            ("one factor", synth_one_factor(40, 5, 6)),
            ("narrow x", synth(60, 4, 0, 0.0, 11)),
        ] {
            let r = gate_run(x.as_ref(), y.as_ref(), None, 20, true);
            assert_eq!(r.test_method, "split_nb", "{name}");
            let sr = r.stable_rank.expect(
                "rank is computed even under force — it is how a caller sees what the gate saw",
            );
            if name == "one factor" {
                assert!(sr < SPLIT_NB_GATE_MIN_STABLE_RANK, "stable_rank={sr}");
            }
        }
    }

    #[test]
    fn gate_uses_n_eff_under_weights() {
        // Raw n = 40 clears the floor; the weights pull Kish n_eff under it.
        // Σw = 22, Σw² = 20.2 ⇒ n_eff = 22²/20.2 ≈ 23.96 < 25.
        let (x, y) = synth(40, 5, 0, 0.0, 8);
        let w = Col::<f64>::from_fn(40, |i| if i % 2 == 0 { 1.0 } else { 0.1 });

        let unweighted = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(
            unweighted.test_method, "split_nb",
            "the same X at raw n = 40 must clear the gate, so the weighted \
             case below isolates n_eff"
        );

        let r = gate_run(x.as_ref(), y.as_ref(), Some(w.as_ref()), 20, false);
        assert_eq!(r.test_method, "split_exact");
        let sr = r.stable_rank.expect("gate was evaluated");
        assert!(
            sr >= SPLIT_NB_GATE_MIN_STABLE_RANK,
            "n_eff, not the spectrum, must be what fired here (stable_rank={sr})"
        );

        // At the floor itself, n = 25: equal weights are no weights. Kish's
        // ratio of 25 × 0.3 rounds to 24.999999999999975, which would fire
        // the gate on a design that clears it unweighted.
        let x25 = synth(25, 10, 0, 0.0, 8).0;
        let equal = Col::<f64>::from_fn(25, |_| 0.3);
        let absent = split_nb_gate(x25.as_ref(), None).unwrap();
        assert!(!absent.fires, "n = 25 must clear the gate unweighted");
        let got = split_nb_gate(x25.as_ref(), Some(equal.as_ref())).unwrap();
        assert_eq!(
            (got.fires, got.n_eff.to_bits()),
            (false, 25.0_f64.to_bits())
        );
    }

    /// Weights change the gate's rank through the standardization constants,
    /// not through any row scaling: `standardize_weighted` divides each column
    /// by its *weighted* sd, so down-weighting the rows that carry a column's
    /// spread shrinks that divisor and inflates the column's share of the
    /// energy. Here column 0 is 50× wider in the first half of the rows, and
    /// the weights all but ignore that half — so under weighted standardization
    /// column 0 swamps the spectrum (rank ≈ 1.5) while plain standardization
    /// leaves it in scale with the other nine (rank ≈ 6).
    ///
    /// This is the one test that fails if the gate standardizes with `None`
    /// where it should pass `w_norm`.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn gate_weighted_standardization_drives_the_rank_clause() {
        use rand::RngExt;
        use rand::SeedableRng;
        let (n, d) = (60, 10);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let x = Mat::<f64>::from_fn(n, d, |i, j| {
            let u = rng.random_range(-1.0..1.0);
            if j == 0 && i < n / 2 {
                50.0 * u
            } else {
                u
            }
        });
        let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        // Kish n_eff = 60²/(30·0.05² + 30·1.95²) ≈ 31.5 — comfortably over the
        // size floor, so only the rank clause can fire.
        let w = Col::<f64>::from_fn(n, |i| if i < n / 2 { 0.05 } else { 1.95 });

        let unweighted = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(unweighted.test_method, "split_nb");

        let weighted = gate_run(x.as_ref(), y.as_ref(), Some(w.as_ref()), 20, false);
        assert_eq!(weighted.test_method, "split_exact");
        assert!(
            weighted.n_eff >= SPLIT_NB_GATE_MIN_N_EFF,
            "{}",
            weighted.n_eff
        );
        let (sr_u, sr_w) = (
            unweighted.stable_rank.expect("gate was evaluated"),
            weighted.stable_rank.expect("gate was evaluated"),
        );
        assert!(
            sr_u >= SPLIT_NB_GATE_MIN_STABLE_RANK,
            "unweighted sr={sr_u}"
        );
        assert!(sr_w < SPLIT_NB_GATE_MIN_STABLE_RANK, "weighted sr={sr_w}");
    }

    // ── the public gate query ────────────────────────────────────────────────
    //
    // These are the whole contract of `split_nb_gate`: it must answer exactly
    // what the embedded gate decides, and it must reject bad input with the
    // same errors the test entry points do rather than reaching the SVD.

    /// One flagged design, one clean one, and a weighted one (the `n_eff`
    /// clause and the weighted-moment standardization behind the rank
    /// clause have to travel with the weights), checked against what
    /// `pls1_confirmatory_test` actually did with the same X.
    #[test]
    #[allow(clippy::float_cmp)] // same rule on the same standardized X — bit-exact or it's a bug
    fn public_gate_answers_what_the_embedded_gate_decided() {
        let half = Col::<f64>::from_fn(40, |i| if i % 2 == 0 { 1.0 } else { 0.1 });
        for ((x, y), w, expect_fires) in [
            (synth_one_factor(40, 5, 6), None, true),
            (synth(60, 10, 0, 0.0, 7), None, false),
            (synth(40, 5, 0, 0.0, 8), Some(half.as_ref()), true),
        ] {
            let embedded = gate_run(x.as_ref(), y.as_ref(), w, 20, false);
            let q = split_nb_gate(x.as_ref(), w).unwrap();
            assert_eq!(q.fires, expect_fires);
            assert_eq!(q.fires, embedded.test_method == "split_exact");
            assert_eq!(q.stable_rank, embedded.stable_rank.unwrap());
            assert_eq!(q.n_eff, embedded.n_eff);
            if w.is_some() {
                assert!(q.n_eff < 40.0, "weights must reach n_eff: {}", q.n_eff);
            }
        }
    }

    /// The reason the function repeats entry-point validation: without it a
    /// NaN reaches `linalg::stable_rank`'s SVD instead of this error.
    #[test]
    fn public_gate_rejects_non_finite_x() {
        let mut x = synth(60, 10, 0, 0.0, 7).0;
        x[(3, 2)] = f64::NAN;
        assert!(matches!(
            split_nb_gate(x.as_ref(), None),
            Err(PlsKitError::NonFiniteInput)
        ));
    }

    #[test]
    fn public_gate_rejects_bad_weights() {
        let x = synth(60, 10, 0, 0.0, 7).0;
        let neg = Col::<f64>::from_fn(60, |i| if i == 0 { -1.0 } else { 1.0 });
        assert!(matches!(
            split_nb_gate(x.as_ref(), Some(neg.as_ref())),
            Err(PlsKitError::InvalidWeights { reason: "negative" })
        ));
        let short = Col::<f64>::from_fn(59, |_| 1.0);
        assert!(matches!(
            split_nb_gate(x.as_ref(), Some(short.as_ref())),
            Err(PlsKitError::InvalidWeights {
                reason: "length_mismatch"
            })
        ));
    }

    // ── `test_method = "auto"` ───────────────────────────────────────────────

    /// The sample-size clause at its floor, on unweighted X where `n_eff` is
    /// the row count. Ten iid columns clear the gate and the width clause, so
    /// one row below the floor is the only reason for `split_exact`, and that
    /// decision is made without the stable-rank check.
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn auto_n_eff_floor() {
        let floor = AUTO_MIN_N_EFF as usize;
        for (n, expect, checked) in [
            (floor - 1, ConfirmatoryMethod::SplitExact, false),
            (floor, ConfirmatoryMethod::SplitNb, true),
        ] {
            let x = synth(n, 10, 0, 0.0, 21).0;
            let (method, sr) = resolve_auto(x.as_ref(), None, n as f64, true);
            assert_eq!(method, expect, "n={n}");
            assert_eq!(sr.is_some(), checked, "n={n}");
        }
    }

    /// The width clause at its ceiling. `n_eff` is passed apart from the
    /// row count, so a 10-row X with `n_eff` above the floor tests the
    /// clause without a 250-row, 25 000-column SVD: `p = AUTO_MAX_P_PER_N · n`
    /// goes on to the stable-rank check (iid columns clear it), one column
    /// more decides `split_exact` without it. Off the no-refit route (k ≥ 2,
    /// a set `keep`, PLS3) the ceiling does not apply.
    #[test]
    fn auto_p_per_n_ceiling() {
        let n = 10;
        let n_eff = AUTO_MIN_N_EFF;
        let ceiling = AUTO_MAX_P_PER_N * n;
        for (p, no_refit, expect, checked) in [
            (ceiling, true, ConfirmatoryMethod::SplitNb, true),
            (ceiling + 1, true, ConfirmatoryMethod::SplitExact, false),
            (ceiling + 1, false, ConfirmatoryMethod::SplitNb, true),
        ] {
            let x = synth(n, p, 0, 0.0, 22).0;
            let (method, sr) = resolve_auto(x.as_ref(), None, n_eff, no_refit);
            assert_eq!(method, expect, "p={p} no_refit={no_refit}");
            assert_eq!(sr.is_some(), checked, "p={p} no_refit={no_refit}");
        }
    }

    /// The sample-size clause reads Kish `n_eff`, the width clause reads
    /// rows. 300 rows weighted 1 / 0.1 alternately give
    /// `n_eff` = 165² / 151.5 ≈ 180, so `split_exact`; the same X unweighted
    /// runs `split_nb`.
    #[test]
    fn auto_n_eff_clause_reads_weights() {
        let x = synth(300, 10, 0, 0.0, 23).0;
        let w = Col::<f64>::from_fn(300, |i| if i % 2 == 0 { 1.0 } else { 0.1 });
        let (w_norm, n_eff) =
            crate::fit::validate_and_normalize_weights(Some(w.as_ref()), 300, 0).unwrap();
        assert!(n_eff < AUTO_MIN_N_EFF, "n_eff={n_eff}");
        let weighted = resolve_auto(x.as_ref(), w_norm.as_ref().map(Col::as_ref), n_eff, true);
        assert_eq!(weighted, (ConfirmatoryMethod::SplitExact, None));
        let (method, sr) = resolve_auto(x.as_ref(), None, 300.0, true);
        assert_eq!(method, ConfirmatoryMethod::SplitNb);
        assert!(sr.is_some());
    }

    /// Above both size clauses the `split_nb` gate still decides: a
    /// one-factor X (stable rank near 1) runs `split_exact` and reports the
    /// rank, and a 4-column X runs `split_exact` on the column precheck
    /// without computing it.
    #[test]
    fn auto_defers_to_the_split_nb_gate() {
        let x = synth_one_factor(300, 10, 6).0;
        let (method, sr) = resolve_auto(x.as_ref(), None, 300.0, true);
        assert_eq!(method, ConfirmatoryMethod::SplitExact);
        let sr = sr.expect("the stable-rank check ran");
        assert!(sr < SPLIT_NB_GATE_MIN_STABLE_RANK, "stable_rank={sr}");

        let x = synth(300, 4, 0, 0.0, 11).0;
        assert_eq!(
            resolve_auto(x.as_ref(), None, 300.0, true),
            (ConfirmatoryMethod::SplitExact, None)
        );
    }

    /// Resolution draws no randomness, so at one seed `Auto` returns what an
    /// explicit call of the resolved method returns, field by field except
    /// `stable_rank`. One design per resolved method, and the CI branch on
    /// the `split_exact` one.
    #[test]
    fn auto_matches_the_resolved_method_at_the_same_seed() {
        let run = |(x, y): &(Mat<f64>, Col<f64>), args, ci| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: 1,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(31),
                    ci,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let small = synth(60, 8, 3, 1.0, 12);
        let large = synth(300, 10, 3, 0.3, 13);
        let auto = ConfirmatoryArgs::Auto {
            n_perm: 49,
            n_splits: 6,
        };
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 49,
            n_splits: 6,
        };
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: 6,
            force: false,
        };
        let ci = Some(CIOpts {
            n_boot: 100,
            ..CIOpts::default()
        });
        for (data, explicit, ci, has_rank) in [
            (&small, exact, None, false),
            (&small, exact, ci, false),
            (&large, nb, None, true),
        ] {
            let tag = format!("{explicit:?} ci={}", ci.is_some());
            let mut got = run(data, auto, ci);
            let mut want = run(data, explicit, ci);
            assert_eq!(got.test_method, explicit.method().as_str(), "{tag}");
            assert_eq!(got.stable_rank.is_some(), has_rank, "{tag}");
            got.stable_rank = None;
            want.stable_rank = None;
            assert_eq!(format!("{got:?}"), format!("{want:?}"), "{tag}");
        }
    }

    /// `Auto` checks `split_exact`'s count floors before it resolves, so
    /// they hold on a design that resolves to `split_nb`, where `n_perm` is
    /// otherwise unused.
    #[test]
    fn auto_count_floors_hold_on_a_split_nb_design() {
        let (x, y) = synth(300, 10, 3, 0.3, 13);
        assert_eq!(
            resolve_auto(x.as_ref(), None, 300.0, true).0,
            ConfirmatoryMethod::SplitNb
        );
        for (n_perm, n_splits) in [(0, 6), (49, 1)] {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: 1,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::Auto { n_perm, n_splits },
                    seed: Some(31),
                    ..Default::default()
                },
            );
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "n_perm={n_perm} n_splits={n_splits}: {r:?}"
            );
        }
    }

    // ── input validation ─────────────────────────────────────────────────────

    #[test]
    fn confirmatory_rejects_keep_with_ci() {
        let (x, y) = synth(80, 5, 3, 4.0, 99);
        let e = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb {
                    n_splits: 30,
                    force: false,
                },
                seed: Some(7),
                keep: Some(2),
                ci: Some(CIOpts::default()),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(e.code(), "invalid_argument");
    }

    #[test]
    fn confirmatory_rejects_bad_keep() {
        let (x, y) = synth(60, 5, 3, 4.0, 17);
        let e = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                keep: Some(99),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(e.code(), "invalid_argument");
    }

    #[test]
    fn ci_branch_rejects_invalid_m_rate() {
        let (x, y) = synth(80, 5, 3, 4.0, 11);
        let err = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb {
                    n_splits: 30,
                    force: false,
                },
                seed: Some(7),
                ci: Some(CIOpts {
                    n_boot: 200,
                    m_rate: 0.4,
                    level: 0.95,
                    max_failure_rate: 0.01,
                }),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
    }

    // ── internals without a public-surface proof ─────────────────────────────

    #[test]
    fn pooled_columns_fail_soft_on_a_null_and_hard_on_the_observed_column() {
        let (x, y0) = signal_data(48, 7, 101);
        let n = x.nrows();
        let folds = crate::linalg::fold_split(&(0..n).collect::<Vec<_>>(), 4);
        let column = |c: usize, bad: Option<usize>| -> Col<f64> {
            let mut y = Col::<f64>::from_fn(n, |i| y0[(i + 3 * c) % n]);
            if bad == Some(c) {
                y[5] = f64::NAN;
            }
            y
        };
        let primal = ReplicateRoute::Primal;
        let clean = pooled_cv_r2_columns(primal, x.as_ref(), &folds, None, 2, None, 4, &|c| {
            column(c, None)
        })
        .unwrap();
        let hit = pooled_cv_r2_columns(primal, x.as_ref(), &folds, None, 2, None, 4, &|c| {
            column(c, Some(2))
        })
        .unwrap();
        assert!(hit[2].is_nan(), "a failed null column is NaN");
        for c in [0, 1, 3] {
            assert_eq!(hit[c].to_bits(), clean[c].to_bits(), "column {c} untouched");
        }
        let err = pooled_cv_r2_columns(primal, x.as_ref(), &folds, None, 2, None, 4, &|c| {
            column(c, Some(0))
        })
        .unwrap_err();
        assert_eq!(
            err.code(),
            "non_finite_input",
            "column 0 propagates its error"
        );
        // A non-finite X fails once, in the preparation of the first fold
        // that trains on it (row 7 sits in fold 0, so fold 1 fails).
        let mut bad_x = x.clone();
        bad_x[(7, 2)] = f64::NAN;
        let a = pls1_cv_r2(bad_x.as_ref(), y0.as_ref(), 2, &folds, None, None).unwrap_err();
        assert_eq!(a.code(), "non_finite_input");
    }

    /// A CV fold or a split half whose training weights are all zero or all
    /// equal trains unweighted, as `pls1_fit` does with all-equal weights.
    #[test]
    fn all_equal_training_weights_train_unweighted() {
        // fold 0 holds rows 0..12; its training rows are 12..48.
        let (x, _) = signal_data(48, 7, 61);
        let folds = crate::linalg::fold_split(&(0..48).collect::<Vec<_>>(), 4);
        for rest in [0.0, 3.0] {
            let w = Col::<f64>::from_fn(48, |i| if i < 12 { 1.0 + i as f64 } else { rest });
            let wn = normalize_weights(w.as_ref()).unwrap();
            let fold = prepare_cv_fold(x.as_ref(), &folds, 0, Some(wn.as_ref())).unwrap();
            assert!(
                fold.w_tr.is_none() && fold.sqw_fit.is_none(),
                "fold, rest = {rest}"
            );
        }

        // This split trains on zero-weight rows only; its test half is weighted.
        let (x, _) = signal_data(40, 6, 81);
        let w = Col::<f64>::from_fn(40, |i| if i < 5 { 1.0 + i as f64 } else { 0.0 });
        let wn = normalize_weights(w.as_ref()).unwrap();
        let split = SplitIdx {
            tr: (5..25).collect(),
            te: (0..5).chain(25..40).collect(),
        };
        let prep = prepare_split(x.as_ref(), &split, Some(wn.as_ref()));
        assert!(prep.w_tr.is_none(), "training half: unweighted");
        assert!(prep.w_te.is_some(), "test half keeps its weights");
    }

    /// The X finiteness check sits in `prepare_split`, not in every per-half
    /// fit; a split whose training half is not finite gives r = +0.
    #[test]
    fn a_non_finite_training_half_gives_r_zero() {
        let (mut x, y) = signal_data(40, 6, 83);
        x[(3, 1)] = f64::INFINITY;
        let sp = SplitIdx {
            tr: (0..20).collect(),
            te: (20..40).collect(),
        };
        assert!(!prepare_split(x.as_ref(), &sp, None).x_finite);
        for k in [1_usize, 2] {
            let r = split_half_r(x.as_ref(), y.as_ref(), k, &sp, None, None);
            assert_eq!(r.to_bits(), 0.0_f64.to_bits(), "k={k}");
        }
    }

    /// Every confirmatory runner reads X through owned row gathers
    /// (`prepare_split`, `prepare_cv_fold`, `standardize_rows`), a
    /// standardized owned copy (`score`, the `split_nb` gate) or
    /// `linalg::col_major_or_copy` (`score` with `pre_standardized`), so a
    /// padded submatrix, a row-major view and a negative-column-stride view
    /// give the owned matrix's whole output to the bit. Run on the default
    /// route set and again primal-only. No CI: its reference fit is
    /// `pls1_fit` on the input view, layout-invariant only to rounding
    /// since 0.6.1.
    #[test]
    fn confirmatory_output_is_bit_identical_across_x_layouts() {
        use crate::test_support::{assert_layout_invariant, copy_free_families};
        // n_perm 49 puts the unweighted wide family's k = 1 raw_perm on the
        // K = 1 Gram route.
        assert_eq!(
            raw_perm_route(30, 5, 400, 49, 1, None, false),
            ReplicateRoute::Special
        );
        let raw_perm = ConfirmatoryArgs::RawPerm {
            n_perm: 49,
            n_folds: 5,
        };
        let split_exact = ConfirmatoryArgs::SplitExact {
            n_perm: 9,
            n_splits: 4,
        };
        let split_nb = ConfirmatoryArgs::SplitNb {
            n_splits: 4,
            force: true,
        };
        let four = [(1, None), (1, Some(3)), (2, None), (2, Some(3))];
        let mut rows: Vec<(ConfirmatoryArgs, usize, Option<usize>)> = Vec::new();
        for args in [raw_perm, split_nb, ConfirmatoryArgs::E] {
            rows.extend(four.iter().map(|&(k, keep)| (args, k, keep)));
        }
        // A single null column: the reachable edge of leave-one-out folds.
        let one_null = ConfirmatoryArgs::RawPerm {
            n_perm: 1,
            n_folds: 5,
        };
        rows.push((one_null, 1, None));
        rows.extend(four.iter().map(|&(k, keep)| (split_exact, k, keep)));
        rows.push((split_exact, 3, None));

        let table = || {
            for f in copy_free_families() {
                let w = f.w.as_ref().map(Col::as_ref);
                for &(args, k, keep) in &rows {
                    let what = format!("{} {args:?} k={k} keep={keep:?}", f.name);
                    assert_layout_invariant(&f.x, &what, |xv| {
                        pls1_confirmatory_test(
                            ConfirmatoryTestInput::Raw {
                                x: xv,
                                y: f.y.as_ref(),
                                k,
                                weights: w,
                            },
                            ConfirmatoryTestOpts {
                                args,
                                seed: Some(21),
                                keep,
                                ..Default::default()
                            },
                        )
                        .unwrap()
                    });
                }
                for pre in [false, true] {
                    let (x0, y0) = f.inputs(pre);
                    let what = format!("{} score pre={pre}", f.name);
                    assert_layout_invariant(&x0, &what, |xv| {
                        pls1_confirmatory_test(
                            ConfirmatoryTestInput::Raw {
                                x: xv,
                                y: y0.as_ref(),
                                k: 1,
                                weights: w,
                            },
                            ConfirmatoryTestOpts {
                                args: ConfirmatoryArgs::Score,
                                pre_standardized: pre,
                                seed: Some(21),
                                ..Default::default()
                            },
                        )
                        .unwrap()
                    });
                }
            }
        };
        table();
        with_gram_routes_disabled(table);
    }
}

#[cfg(test)]
mod tests_gram_p {
    use super::*;
    use crate::resample::Columns;
    use crate::test_support::{gram_p_weights, linear_y, unif};
    use faer::Par;

    fn seeds(n_perm: usize, seed: u64) -> Vec<u64> {
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).expect("seed");
        crate::rng::child_seeds(&mut rng, n_perm)
    }

    fn normalized(w: Option<&Col<f64>>) -> Option<Col<f64>> {
        w.map(|w| crate::linalg::normalize_weights(w.as_ref()).expect("nonzero"))
    }

    fn folds5(n: usize) -> Vec<Vec<usize>> {
        let idx: Vec<usize> = (0..n).collect();
        crate::linalg::fold_split(&idx, 5)
    }

    /// Route-invisibility rule on statistics: `1e-10` absolute, NaN in the
    /// same places.
    fn assert_columns_close(g: &[f64], x: &[f64], label: &str) {
        assert_eq!(g.len(), x.len(), "{label}: length");
        for (c, (a, b)) in g.iter().zip(x).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-10,
                "{label}, column {c}: {a} vs {b}"
            );
        }
    }

    /// Exceedance counts (column 0 observed; NaN or non-finite counts as an
    /// exceedance) agree unless a null lies within `1e-10` of the observed
    /// value on the Primal route, where `>=` may resolve either way.
    fn assert_same_exceedances(g: &[f64], x: &[f64], label: &str) {
        let count = |v: &[f64]| {
            v[1..]
                .iter()
                .filter(|z| !z.is_finite() || **z >= v[0])
                .count()
        };
        let near_tie = x[1..].iter().any(|z| (z - x[0]).abs() <= 1e-10);
        if !near_tie {
            assert_eq!(count(g), count(x), "{label}: exceedance counts differ");
        }
    }

    /// Pooled CV R² per column through the `raw_perm` driver on `route`:
    /// column 0 is `y`, column c the permutation of `seeds[c - 1]`.
    #[allow(clippy::too_many_arguments)]
    fn raw_perm_columns(
        route: ReplicateRoute,
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        folds: &[Vec<usize>],
        seeds: &[u64],
        k: usize,
        keep: Option<usize>,
        w: Option<ColRef<'_, f64>>,
    ) -> Vec<f64> {
        let cols = Columns { y, seeds };
        pooled_cv_r2_columns(route, x, folds, w, k, keep, cols.len(), &|c| cols.column(c))
            .expect("columns")
    }

    const CASES: [(&str, usize, Option<usize>, bool); 5] = [
        ("dense_k1", 1, None, false),
        ("dense_k2", 2, None, false),
        ("keep10_k2", 2, Some(10), false),
        ("weighted_k2", 2, None, true),
        ("weighted_keep10_k2", 2, Some(10), true),
    ];

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn raw_perm_gram_fold_units_match_primal_units() {
        // Dense K = 1 and K = 2, sparse, weighted, weighted sparse: the
        // weighted and sparse cells have no corpus fixture, so this test and
        // the route-invisibility test below carry them. Per fold: ss_tot bit
        // for bit (it depends only on y), |Δss_res| ≤ 1e-10·max(1, ss_tot);
        // then the pooled columns.
        let x = unif(2000, 50, 81);
        let y = linear_y(&x, 3.0, 82);
        let w = gram_p_weights(2000);
        let folds = folds5(2000);
        let seeds = seeds(20, 83);
        let cols = Columns {
            y: y.as_ref(),
            seeds: &seeds,
        };
        for (label, k, keep, weighted) in CASES {
            let wn = normalized(weighted.then_some(&w));
            let wref = wn.as_ref().map(Col::as_ref);
            let (mut units, mut resolved) = (0usize, 0usize);
            // Closeness alone would pass if the GramP arm always fell back
            // to the Primal arm, so units whose `ss_res` bits differ from
            // the Primal arm's are counted too: the Gram kernel must run.
            let mut differing = 0usize;
            for fi in 0..folds.len() {
                let fold = prepare_cv_fold(x.as_ref(), &folds, fi, wref).expect("fold");
                let g_block = fold_block(ReplicateRoute::GramP, &fold, Par::Seq);
                let p_block = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
                let FoldBlock::GramP(gram) = &g_block else {
                    panic!("{label}: fold_block(GramP) built another block")
                };
                for c in 0..cols.len() {
                    let yc = cols.column(c);
                    let y_of = |i: usize| yc[i];
                    let (res_g, tot_g) = fold_unit(&g_block, &fold, &y_of, k, keep).expect("unit");
                    let (res_p, tot_p) = fold_unit(&p_block, &fold, &y_of, k, keep).expect("unit");
                    assert_eq!(
                        tot_g.to_bits(),
                        tot_p.to_bits(),
                        "{label}, fold {fi}, column {c}: ss_tot"
                    );
                    assert!(
                        (res_g - res_p).abs() <= 1e-10 * tot_p.max(1.0),
                        "{label}, fold {fi}, column {c}: ss_res {res_g} vs {res_p}"
                    );
                    let (ys_tr, _) = cv_fold_targets(&fold, &y_of);
                    let ys_fit = match fold.sqw_fit.as_ref() {
                        Some(s) => crate::fit::scale_col(ys_tr.as_ref(), s.as_ref()),
                        None => ys_tr,
                    };
                    units += 1;
                    if gram.fit_replicate(ys_fit.as_ref(), k, keep).is_some() {
                        resolved += 1;
                    }
                    if res_g.to_bits() != res_p.to_bits() {
                        differing += 1;
                    }
                }
            }
            assert!(
                resolved * 100 >= units * 99,
                "{label}: only {resolved}/{units} units resolved on ordinary data"
            );
            assert!(
                differing > 0,
                "{label}: no unit's ss_res bits differed from the Primal arm: the GramP arm \
                 may be falling back on every unit"
            );
            let p = raw_perm_columns(
                ReplicateRoute::Primal,
                x.as_ref(),
                y.as_ref(),
                &folds,
                &seeds,
                k,
                keep,
                wref,
            );
            let g = raw_perm_columns(
                ReplicateRoute::GramP,
                x.as_ref(),
                y.as_ref(),
                &folds,
                &seeds,
                k,
                keep,
                wref,
            );
            assert_columns_close(&g, &p, label);
            assert_same_exceedances(&g, &p, label);
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn raw_perm_gram_matches_primal_with_many_zero_weights() {
        // Rows 0..400 (all of fold 0) and every third row after carry weight
        // 0: training blocks with long runs of zero rows.
        let x = unif(2000, 50, 87);
        let y = linear_y(&x, 1.0, 88);
        let w = Col::<f64>::from_fn(2000, |i| {
            if i < 400 || i % 3 == 0 {
                0.0
            } else {
                1.0 + (i % 4) as f64
            }
        });
        let wn = normalized(Some(&w));
        let wref = wn.as_ref().map(Col::as_ref);
        let folds = folds5(2000);
        let seeds = seeds(40, 89);
        for (label, keep) in [
            ("zero_weights_k2", None),
            ("zero_weights_keep5_k2", Some(5)),
        ] {
            let g = raw_perm_columns(
                ReplicateRoute::GramP,
                x.as_ref(),
                y.as_ref(),
                &folds,
                &seeds,
                2,
                keep,
                wref,
            );
            let p = raw_perm_columns(
                ReplicateRoute::Primal,
                x.as_ref(),
                y.as_ref(),
                &folds,
                &seeds,
                2,
                keep,
                wref,
            );
            assert_columns_close(&g, &p, label);
            assert_same_exceedances(&g, &p, label);
        }
    }

    #[test]
    fn raw_perm_gram_keep_equal_to_p_is_the_dense_fit_to_the_bit() {
        let x = unif(2000, 50, 90);
        let y = linear_y(&x, 1.0, 91);
        let folds = folds5(2000);
        let seeds = seeds(30, 92);
        let run = |keep| {
            raw_perm_columns(
                ReplicateRoute::GramP,
                x.as_ref(),
                y.as_ref(),
                &folds,
                &seeds,
                2,
                keep,
                None,
            )
        };
        let (dense, keep_p) = (run(None), run(Some(50)));
        assert!(dense
            .iter()
            .zip(&keep_p)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn raw_perm_gram_p_arm_keeps_the_primal_errors_and_nan_columns() {
        // Row 5 is a validation row of fold 0 and a training row of every
        // other fold. A NaN there fails the Gram gates in folds 1..5, so the
        // GramP arm hands the unit to the Primal arm, which reports it: a
        // null column goes NaN, column 0 returns the Primal route's error.
        let x = unif(2000, 50, 93);
        let y = linear_y(&x, 1.0, 94);
        let folds = folds5(2000);
        let seeds = seeds(6, 95);
        let cols = Columns {
            y: y.as_ref(),
            seeds: &seeds,
        };
        let column = |c: usize, bad: Option<usize>| -> Col<f64> {
            let mut yc = cols.column(c);
            if bad == Some(c) {
                yc[5] = f64::NAN;
            }
            yc
        };
        let run = |route: ReplicateRoute, bad: Option<usize>| {
            pooled_cv_r2_columns(route, x.as_ref(), &folds, None, 2, None, cols.len(), &|c| {
                column(c, bad)
            })
        };
        let clean = run(ReplicateRoute::GramP, None).expect("clean");
        let hit = run(ReplicateRoute::GramP, Some(2)).expect("a failed null column fails soft");
        assert!(hit[2].is_nan(), "a failed null column is NaN");
        for c in (0..cols.len()).filter(|&c| c != 2) {
            assert_eq!(hit[c].to_bits(), clean[c].to_bits(), "column {c} untouched");
        }
        let e_g = run(ReplicateRoute::GramP, Some(0)).expect_err("column 0 fails hard");
        let e_p = run(ReplicateRoute::Primal, Some(0)).expect_err("column 0 fails hard");
        assert_eq!(e_g.code(), e_p.code());
        assert_eq!(e_g.to_string(), e_p.to_string());
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn raw_perm_gram_route_is_invisible() {
        let x = unif(2000, 50, 96);
        let y = linear_y(&x, 30.0, 97);
        let w = gram_p_weights(2000);
        for (label, wopt, keep) in [
            ("dense", None, None),
            ("weighted", Some(&w), None),
            ("keep10", None, Some(10)),
        ] {
            assert_eq!(
                raw_perm_route(2000, 5, 50, 300, 2, keep, wopt.is_some()),
                ReplicateRoute::GramP,
                "{label}: premise"
            );
            let opts = ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 300,
                    n_folds: 5,
                },
                seed: Some(98),
                keep,
                ..Default::default()
            };
            let call = || {
                pls1_confirmatory_test(
                    ConfirmatoryTestInput::Raw {
                        x: x.as_ref(),
                        y: y.as_ref(),
                        k: 2,
                        weights: wopt.map(Col::as_ref),
                    },
                    opts,
                )
                .expect("raw_perm")
            };
            let g = call();
            let p = with_gram_routes_disabled(call);
            // No null lies within 1e-10 of the observed statistic at this
            // seed (the driver-level tests above check the near-tie rule),
            // so the p-values are equal.
            assert_eq!(g.pvalue.to_bits(), p.pvalue.to_bits(), "{label}: pvalue");
            assert!(
                (g.statistic - p.statistic).abs() <= 1e-10,
                "{label}: statistic {} vs {}",
                g.statistic,
                p.statistic
            );
        }
    }

    /// Per-column z̄ through the `split_exact` driver on `route`.
    #[allow(clippy::too_many_arguments)]
    fn split_exact_zbars(
        route: ReplicateRoute,
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        splits: &[SplitIdx],
        seeds: &[u64],
        k: usize,
        keep: Option<usize>,
        w: Option<ColRef<'_, f64>>,
    ) -> Vec<f64> {
        let cols = Columns { y, seeds };
        split_zbars_columns(route, x, splits, w, k, keep, cols.len(), &|c| {
            cols.column(c)
        })
    }

    const SPLIT_CASES: [(&str, usize, Option<usize>, bool); 5] = [
        ("dense_k2", 2, None, false),
        ("weighted_k2", 2, None, true),
        ("keep10_k1", 1, Some(10), false),
        ("keep10_k2", 2, Some(10), false),
        ("weighted_keep10_k2", 2, Some(10), true),
    ];

    #[test]
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    fn split_exact_gram_units_match_primal_units() {
        // Dense, weighted, sparse and weighted sparse: the dense and
        // weighted refit cells have no corpus fixture, so this test and the
        // route-invisibility tests below carry them. Per split r within
        // 1e-10 absolute, then the z̄ statistics.
        let x = unif(2000, 50, 99);
        let y = linear_y(&x, 2.0, 100);
        let w = gram_p_weights(2000);
        let (_, mut rng) = crate::rng::resolve_seed(Some(101)).expect("seed");
        let splits = draw_splits(2000, 2, 3, &mut rng).expect("splits");
        let seeds = crate::rng::child_seeds(&mut rng, 40);
        let cols = Columns {
            y: y.as_ref(),
            seeds: &seeds,
        };
        for (label, k, keep, weighted) in SPLIT_CASES {
            let wn = normalized(weighted.then_some(&w));
            let wref = wn.as_ref().map(Col::as_ref);
            let (mut units, mut resolved) = (0usize, 0usize);
            // Closeness alone would pass if the GramP arm always fell back
            // to the Primal arm, so units whose `r` bits differ from the
            // Primal arm's are counted too: the Gram kernel and its scores
            // must be used.
            let mut differing = 0usize;
            for (si, sp) in splits.iter().enumerate() {
                let prep = prepare_split(x.as_ref(), sp, wref);
                let g_block = split_block(ReplicateRoute::GramP, &prep, Par::Seq);
                let p_block = split_block(ReplicateRoute::Primal, &prep, Par::Seq);
                let SplitBlock::GramP(gram) = &g_block else {
                    panic!("{label}: split_block(GramP) built another block")
                };
                for c in 0..cols.len() {
                    let yc = cols.column(c);
                    let y_of = |i: usize| yc[i];
                    let r_g = split_unit(&g_block, &prep, sp, &y_of, k, keep);
                    let r_p = split_unit(&p_block, &prep, sp, &y_of, k, keep);
                    assert!(
                        (r_g - r_p).abs() <= 1e-10,
                        "{label}, split {si}, column {c}: {r_g} vs {r_p}"
                    );
                    let ys_tr = split_train_target(&prep, sp, &y_of);
                    units += 1;
                    if gram
                        .block
                        .fit_replicate_full(ys_tr.as_ref(), k, keep)
                        .is_some()
                    {
                        resolved += 1;
                    }
                    if r_g.to_bits() != r_p.to_bits() {
                        differing += 1;
                    }
                }
            }
            assert!(
                resolved * 100 >= units * 99,
                "{label}: only {resolved}/{units} units resolved on ordinary data"
            );
            assert!(
                differing > 0,
                "{label}: no unit's r bits differed from the Primal arm: the GramP arm may be \
                 falling back on every unit"
            );
            let p = split_exact_zbars(
                ReplicateRoute::Primal,
                x.as_ref(),
                y.as_ref(),
                &splits,
                &seeds,
                k,
                keep,
                wref,
            );
            let g = split_exact_zbars(
                ReplicateRoute::GramP,
                x.as_ref(),
                y.as_ref(),
                &splits,
                &seeds,
                k,
                keep,
                wref,
            );
            assert_columns_close(&g, &p, label);
            assert_same_exceedances(&g, &p, label);
        }
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn split_exact_gram_zero_model_gives_r_zero() {
        // A y orthogonal to every training half's columns is not
        // constructible across random splits; a constant y is: both arms
        // keep no component and report r = 0.
        let x = unif(2000, 50, 102);
        let y = Col::<f64>::from_fn(2000, |_| 3.0);
        let (_, mut rng) = crate::rng::resolve_seed(Some(103)).expect("seed");
        let splits = draw_splits(2000, 2, 2, &mut rng).expect("splits");
        for sp in &splits {
            let prep = prepare_split(x.as_ref(), sp, None);
            let g_block = split_block(ReplicateRoute::GramP, &prep, Par::Seq);
            let p_block = split_block(ReplicateRoute::Primal, &prep, Par::Seq);
            let y_of = |i: usize| y[i];
            let r_g = split_unit(&g_block, &prep, sp, &y_of, 2, None);
            let r_p = split_unit(&p_block, &prep, sp, &y_of, 2, None);
            assert_eq!(r_g.to_bits(), r_p.to_bits());
            assert_eq!(r_p.to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn score_band_covers_the_gram_p_score_discrepancy() {
        // dual_route::SCORE_BAND = 10·η_max/1e-10 was set from the
        // n-space route's measured test-score discrepancy η = ‖Δs‖/‖s‖.
        // The Gram-p route shares the constant, so its own η_max must
        // fit under the same 10× margin, over every k it serves
        // (1..=K_GRAM_MAX), dense and sparse, unweighted and weighted.
        let x = unif(2000, 50, 104);
        let y = linear_y(&x, 2.0, 105);
        let w = gram_p_weights(2000);
        let (_, mut rng) = crate::rng::resolve_seed(Some(106)).expect("seed");
        let splits = draw_splits(2000, 2, 3, &mut rng).expect("splits");
        let seeds = crate::rng::child_seeds(&mut rng, 40);
        let cols = Columns {
            y: y.as_ref(),
            seeds: &seeds,
        };
        let (mut eta_max, mut measured) = (0.0_f64, 0usize);
        let cases = [
            (1usize, Some(10usize)),
            (2, None),
            (2, Some(10)),
            (3, None),
            (3, Some(10)),
        ];
        assert_eq!(
            crate::gram_p::K_GRAM_MAX,
            3,
            "cases cover k = 1..=K_GRAM_MAX"
        );
        for ((k, keep), weighted) in cases
            .into_iter()
            .flat_map(|case| [(case, false), (case, true)])
        {
            let wn = normalized(weighted.then_some(&w));
            let wref = wn.as_ref().map(Col::as_ref);
            for sp in &splits {
                let prep = prepare_split(x.as_ref(), sp, wref);
                let gram = SplitGram::new(&prep, Par::Seq);
                for c in 0..cols.len() {
                    let yc = cols.column(c);
                    let y_of = |i: usize| yc[i];
                    let ys_tr = split_train_target(&prep, sp, &y_of);
                    let Some(fit) = gram.block.fit_replicate_full(ys_tr.as_ref(), k, keep) else {
                        continue;
                    };
                    let xf = crate::fit::pls1_fit_prepared_fro(
                        prep.xs_tr.as_ref(),
                        ys_tr.as_ref(),
                        k,
                        keep,
                        crate::fit::ParChoice::Seq,
                        prep.x_fro,
                    )
                    .expect("X backend");
                    let s_g: Col<f64> = &prep.xs_te * &fit.coef;
                    let s_x: Col<f64> = &prep.xs_te * &xf.coef;
                    let norm = s_x.norm_l2();
                    if norm > 0.0 {
                        let diff = (0..s_x.nrows())
                            .map(|i| (s_g[i] - s_x[i]).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        eta_max = eta_max.max(diff / norm);
                        measured += 1;
                    }
                }
            }
        }
        assert!(measured > 0, "no resolved unit was measured");
        let required = 10.0 * eta_max / 1e-10;
        eprintln!(
            "gram_p test-score discrepancy over {measured} units: eta_max = {eta_max:e}; \
             SCORE_BAND must be at least {required:e} (it is {:e})",
            crate::dual_route::SCORE_BAND
        );
        assert!(
            crate::dual_route::SCORE_BAND >= required,
            "SCORE_BAND {:e} is below 10·eta_max/1e-10 = {required:e} for the Gram-p route: \
             see signal_test::tests_gram_p::score_band_covers_the_gram_p_score_discrepancy",
            crate::dual_route::SCORE_BAND
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn split_exact_gram_route_is_invisible_with_a_large_outcome_offset() {
        let x = unif(2000, 50, 107);
        let e = linear_y(&x, 20.0, 108);
        let y = Col::<f64>::from_fn(2000, |i| 1e8 + e[i]);
        let w = gram_p_weights(2000);
        for (label, wopt) in [("dense", None), ("weighted", Some(&w))] {
            assert_eq!(
                split_exact_refit_route(2000, 50, 600, 2, None, wopt.is_some()),
                ReplicateRoute::GramP,
                "{label}: premise"
            );
            let opts = ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 600,
                    n_splits: 5,
                },
                seed: Some(109),
                ..Default::default()
            };
            let call = || {
                pls1_confirmatory_test(
                    ConfirmatoryTestInput::Raw {
                        x: x.as_ref(),
                        y: y.as_ref(),
                        k: 2,
                        weights: wopt.map(Col::as_ref),
                    },
                    opts,
                )
                .expect("split_exact")
            };
            let g = call();
            let p = with_gram_routes_disabled(call);
            assert_eq!(g.pvalue.to_bits(), p.pvalue.to_bits(), "{label}: pvalue");
            assert!(
                (g.statistic - p.statistic).abs() <= 1e-10,
                "{label}: statistic {} vs {}",
                g.statistic,
                p.statistic
            );
        }
    }

    #[test]
    fn spls1_sequence_split_exact_gram_route_is_invisible() {
        // The public path to the sparse refit route: steps test at
        // k = 1 with keep.
        let x = unif(2000, 200, 110);
        let y = linear_y(&x, 20.0, 111);
        assert_eq!(
            split_exact_refit_route(2000, 200, 400, 1, Some(10), false),
            ReplicateRoute::GramP,
            "premise"
        );
        let opts = crate::find_k::FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitExact,
            n_perm: 400,
            n_splits: 5,
            seed: Some(112),
            ..Default::default()
        };
        let call = || {
            crate::find_k::spls1_find_k_sequence(x.as_ref(), y.as_ref(), 2, 10, None, opts)
                .expect("sequence")
        };
        let g = call();
        let p = with_gram_routes_disabled(call);
        assert_eq!(g.k_star, p.k_star, "k_star");
        assert_eq!(g.pvalues.nrows(), p.pvalues.nrows());
        for i in 0..g.pvalues.nrows() {
            assert_eq!(
                g.pvalues[i].to_bits(),
                p.pvalues[i].to_bits(),
                "pvalues[{i}]"
            );
        }
    }

    /// A 300-row design split into training rows `0..200` and test rows
    /// `200..300`. Training rows are uniform with a linear outcome; every
    /// test row is one fixed row plus `spread` times uniform noise, so the
    /// standardized test half is constant up to a relative `spread`. The
    /// test-half outcome is `y_te(i)`.
    fn near_constant_test_half(
        spread: f64,
        y_te: impl Fn(usize) -> f64,
    ) -> (Mat<f64>, Col<f64>, SplitIdx) {
        let base = unif(200, 10, 113);
        let noise = unif(100, 10, 114);
        let x = Mat::<f64>::from_fn(300, 10, |i, j| {
            if i < 200 {
                base[(i, j)]
            } else {
                0.5 - 0.05 * j as f64 + spread * noise[(i - 200, j)]
            }
        });
        let y_base = linear_y(&base, 0.5, 115);
        let y = Col::<f64>::from_fn(300, |i| if i < 200 { y_base[i] } else { y_te(i - 200) });
        let sp = SplitIdx {
            tr: (0..200).collect(),
            te: (200..300).collect(),
        };
        (x, y, sp)
    }

    #[test]
    fn split_exact_gram_score_gate_falls_back_to_the_primal_arm() {
        // Test-half scores with centered norm about 1e-6 of their norm:
        // below SCORE_BAND, yet far from constant to rounding, so the
        // Primal arm reports a nonzero r. The GramP arm must hand the unit
        // to it and return its bits.
        let e = crate::test_support::unif_col(100, 116);
        let (x, y, sp) = near_constant_test_half(1e-6, |i| e[i]);
        let prep = prepare_split(x.as_ref(), &sp, None);
        let y_of = |i: usize| y[i];
        for k in 1..=2 {
            let SplitBlock::GramP(gram) = split_block(ReplicateRoute::GramP, &prep, Par::Seq)
            else {
                panic!("split_block(GramP) built another block")
            };
            let ys_tr = split_train_target(&prep, &sp, &y_of);
            let fit = gram
                .block
                .fit_replicate_full(ys_tr.as_ref(), k, None)
                .expect("premise: the Gram fit resolves");
            assert!(fit.k_used == k, "premise: k_used = {}", fit.k_used);
            let s: Col<f64> = &prep.xs_te * &fit.coef;
            let ms = scaled_moments(100, |i| s[i], None);
            assert!(
                !ms.is_constant(100) && ms.ss.sqrt() < crate::dual_route::SCORE_BAND * ms.sq.sqrt(),
                "premise: the scores are below the score band but not constant"
            );
            let r_g = split_column_r_gram_p(&gram, &prep, &sp, &y_of, k, None);
            let r_p = split_column_r(&prep, &sp, &y_of, k, None);
            assert!(r_p != 0.0, "premise: the Primal arm reports a nonzero r");
            assert_eq!(r_g.to_bits(), r_p.to_bits(), "k = {k}");
        }
    }

    #[test]
    fn split_exact_gram_constant_held_out_y_gives_r_zero() {
        // A resolved Gram fit with ordinary test-half scores, but a
        // held-out outcome constant to rounding: r = 0 on both arms.
        let (x, y, sp) = near_constant_test_half(1.0, |_| 7.0);
        let prep = prepare_split(x.as_ref(), &sp, None);
        let y_of = |i: usize| y[i];
        for k in 1..=2 {
            let SplitBlock::GramP(gram) = split_block(ReplicateRoute::GramP, &prep, Par::Seq)
            else {
                panic!("split_block(GramP) built another block")
            };
            let ys_tr = split_train_target(&prep, &sp, &y_of);
            let fit = gram
                .block
                .fit_replicate_full(ys_tr.as_ref(), k, None)
                .expect("premise: the Gram fit resolves");
            assert!(fit.k_used == k, "premise: k_used = {}", fit.k_used);
            let r_g = split_column_r_gram_p(&gram, &prep, &sp, &y_of, k, None);
            let r_p = split_column_r(&prep, &sp, &y_of, k, None);
            assert_eq!(r_g.to_bits(), 0.0_f64.to_bits(), "k = {k}");
            assert_eq!(r_p.to_bits(), 0.0_f64.to_bits(), "k = {k}");
        }
    }
}
