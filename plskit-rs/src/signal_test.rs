//! Confirmatory PLS1 omnibus test at fixed K: five methods.

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::Pls1Model;

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
        /// Normalized to mean 1 before use (spec §3.3–3.4).
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
        /// Number of CV folds.
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
        }
    }
}

/// Sample-size floor of the `split_nb` auto-gate: a design with fewer
/// effective observations than this is rerouted to `split_exact`.
///
/// This and [`SPLIT_NB_GATE_MIN_STABLE_RANK`] are calibrated constants, not
/// round numbers picked by taste: 5,000 null replicates per cell across iid
/// Gaussian, decaying-spectrum, single-factor, NIR gasoline and GloVe-300
/// designs. The NB correction is exact at ρ = ½ and loses its level when the
/// sample is small or X's spectrum is concentrated on few directions; the
/// pair of thresholds is the smallest rule separating the level-holding cells
/// from the rest, and it deliberately errs toward flagging.
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

/// The `split_nb` auto-gate rule, in the one place it is written. Returns
/// `(fires, stable_rank)`; `fires` means the design is flagged and the caller
/// should reroute to `split_exact`.
///
/// `xs` must ALREADY be standardized (weighted moments when weights are in
/// play) — standardization policy is the caller's. All three call sites
/// standardize unconditionally before calling this: `pls1_confirmatory_test`
/// always standardizes a fresh copy, `sequential::run_incremental_sequence`
/// does too (ignoring its `pre_standardized` flag, so the two can't disagree
/// under weighted input), and the public [`split_nb_gate`] query standardizes
/// whatever it is handed. `n_gate` is Kish `n_eff` under weights and the raw
/// row count without.
pub(crate) fn split_nb_gate_rule(xs: MatRef<'_, f64>, n_gate: f64) -> (bool, f64) {
    // `sr` is computed unconditionally even when the column precheck already
    // decided: it is reported as the gate's `stable_rank` diagnostic either way.
    let sr = crate::linalg::stable_rank(xs);
    let narrow = xs.ncols() <= SPLIT_NB_GATE_MAX_COLS_PRECHECK;
    (
        narrow || n_gate < SPLIT_NB_GATE_MIN_N_EFF || sr < SPLIT_NB_GATE_MIN_STABLE_RANK,
        sr,
    )
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
    // Entry-point validation is repeated here rather than inherited: this
    // function runs ahead of any dispatch, and `linalg::stable_rank` expects
    // its SVD to converge. A NaN reaching it either panics or yields a NaN
    // rank that loses the threshold comparison silently — either way the
    // caller would lose the clean `NonFiniteInput`.
    crate::fit::check_finite_mat(x)?;
    // k_requested is only used for a check this function doesn't make (there
    // is no component count in play), so any value passes; 0 is the honest one.
    let (w_norm, n_eff, _all_uniform) =
        crate::fit::validate_and_normalize_weights(weights, x.nrows(), 0)?;
    let (xs, _, _) = crate::linalg::standardize_weighted(x, w_norm.as_ref().map(Col::as_ref));
    let (fires, stable_rank) = split_nb_gate_rule(xs.as_ref(), n_eff);
    Ok(SplitNbGateOutput {
        fires,
        stable_rank,
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
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
    pub disable_parallelism: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
    /// Optional CI bundle. When `Some`, runs an independent subsampling pass
    /// after the test and populates `ConfirmatoryTestOutput.ci`.
    pub ci: Option<CIOpts>,
    /// Subsample-loop skip threshold for the `ci` branch (spec §6.3).
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
    fn default() -> Self {
        Self {
            args: ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::SplitNb),
            pre_standardized: false,
            seed: None,
            disable_parallelism: false,
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
    pub method: String,
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
    /// is valid (unweighted, `n_test >= 4`); `None` for every other method,
    /// including `split_exact` (no z-scatter interpretation to offer), and
    /// `None` on `split_nb` itself when weighted or `n_test < 4`. Unrelated to
    /// `n_eff` (that field is the weights effective-n).
    pub rho_hat: Option<f64>,
    /// Stable rank `‖X_std‖²_F / ‖X_std‖²₂` of the standardized X, as seen by
    /// the `split_nb` auto-gate. `Some` whenever `split_nb` was the requested
    /// method — whether the gate fired or not, and also under `force` (the
    /// point of the field is to show what the gate saw); `None` for every
    /// other requested method, which never evaluates the gate.
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
/// - `PlsKitError::NonFiniteInput` when X, y, or weights contain NaN/inf
/// - `PlsKitError::InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
pub fn pls1_confirmatory_test(
    input: ConfirmatoryTestInput<'_>,
    opts: ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    confirmatory_test_impl(input, opts, GateMode::Owned)
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
            if n_perm < 1 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_perm must be ≥ 1, got {n_perm}"
                )));
            }
        }
        ConfirmatoryArgs::SplitNb { n_splits, .. } => {
            if n_splits < 2 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_splits must be ≥ 2, got {n_splits}"
                )));
            }
        }
        ConfirmatoryArgs::SplitExact { n_perm, n_splits } => {
            if n_splits < 2 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_splits must be ≥ 2, got {n_splits}"
                )));
            }
            if n_perm < 1 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_perm must be ≥ 1, got {n_perm}"
                )));
            }
        }
        ConfirmatoryArgs::Score | ConfirmatoryArgs::E => {}
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
    let (w_norm, n_eff_val, _all_uniform) =
        crate::fit::validate_and_normalize_weights(weights_in, n, k_resolved)?;
    crate::fit::check_n_eff_for_k(n_eff_val, k_resolved, weights_in.is_some())?;

    if let Some(kp) = opts.keep {
        crate::fit::validate_keep(kp, x_ref.ncols())?;
        if opts.ci.is_some() {
            return Err(PlsKitError::InvalidArgument(
                "keep does not combine with ci: per-coordinate subsample CIs under \
                 selection are post-selection inference (see the spls1 spec)"
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
    // Resolution happens here, BEFORE dispatch, not inside it: `method`,
    // `n_perm` and `n_splits` on the output are read off `args_resolved`, so
    // rewriting the args is exactly what makes `result.method` say
    // `"split_exact"` when the gate fired.
    //
    // Under `GateMode::Decided` the whole block is skipped: the caller settled
    // the method already and would only be re-paying for a standardize plus a
    // full SVD (`linalg::stable_rank`) whose answer it discards.
    let mut args_resolved = opts.args;
    let mut stable_rank_out = None;
    if let (GateMode::Owned, ConfirmatoryArgs::SplitNb { n_splits, force }) = (gate, opts.args) {
        // The gate standardizes its own copy of X, unconditionally — including
        // under `pre_standardized`. Re-standardizing an already-standardized
        // matrix is the identity up to fp rounding, and stable rank is
        // scale-invariant on top of that, so the extra pass cannot move the
        // decision; skipping it would only save one O(n·d) sweep.
        let (xs, _, _) =
            crate::linalg::standardize_weighted(x_ref, w_norm.as_ref().map(Col::as_ref));
        // `n_eff_val` is Kish `n_eff` under weights and exactly `n` when
        // weights are absent, so it feeds `split_nb_gate`'s `n_gate` directly.
        let (fires, sr) = split_nb_gate_rule(xs.as_ref(), n_eff_val);
        stable_rank_out = Some(sr);
        if !force && fires {
            // `n_perm` is split_exact's own default; the requested `n_splits`
            // carries over untouched. Mirrored by the hoisted sequence-level
            // reroute in `sequential::run_incremental_sequence` and by
            // `_REROUTE_FALLBACK_N_PERM` in plskit-py/python/plskit/_api.py —
            // all three change together.
            args_resolved = ConfirmatoryArgs::SplitExact {
                n_perm: 1000,
                n_splits,
            };
        }
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
            // permutation). Weights do not break it: under Convention A the
            // map just carries a `diag(√w_tr)` inside it and stays fixed and
            // linear in y, so weighted K = 1 dense input takes the no-refit
            // route too (see `split_perm_nr_zbars`, "Under weights").
            let no_refit = k_resolved == 1 && opts.keep.is_none();
            let run = if no_refit {
                run_split_perm_nr
            } else {
                run_split_perm
            };
            (
                run(
                    x_ref,
                    y_ref,
                    k_resolved,
                    n_perm,
                    n_splits,
                    w_norm.as_ref().map(Col::as_ref),
                    &opts,
                    &mut rng,
                )?,
                Some(n_perm),
                Some(n_splits),
            )
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
    };

    let ci_payload = if let Some(ci_opts) = opts.ci {
        let sub_opts = crate::subsample::SubsampleOpts {
            n_boot: ci_opts.n_boot,
            m_rate: ci_opts.m_rate,
            level: ci_opts.level,
            pre_standardized: opts.pre_standardized,
            disable_parallelism: opts.disable_parallelism,
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
            use crate::fit::{pls1_fit, FitOpts, KSpec};
            pls1_fit(
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
        method: args_resolved.method().as_str().to_owned(),
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

// Debug is only needed so test-code assert_...!/unwrap_err() calls on
// PlsKitResult<RunResult> can format the Ok side; production code never
// prints a RunResult.
#[derive(Debug)]
struct RunResult {
    pvalue: f64,
    statistic: f64,
    /// Estimated split correlation ρ̂. `Some` only on the `split_nb` path,
    /// and only when its ruler is valid (unweighted, `n_test >= 4`).
    rho_hat: Option<f64>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 4: `raw_perm` CV R² test
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
    // Weights are passed through to pls1_cv_r2 which re-normalizes per fold.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    let folds = crate::linalg::fold_split(&indices, n_folds);

    let cv_r2_obs = pls1_cv_r2(x, y, k, &folds, w_norm, opts.keep)?;

    let nulls_vec = crate::resample::parallel_for_each_seeded(
        rng,
        n_perm,
        opts.disable_parallelism,
        |_, child| {
            // Permute y rows; weights stay tied to row indices (not permuted).
            let perm = crate::resample::permute_indices(n, child);
            let y_perm = Col::<f64>::from_fn(n, |i| y[perm[i]]);
            pls1_cv_r2(x, y_perm.as_ref(), k, &folds, w_norm, opts.keep).unwrap_or(f64::NAN)
        },
    );

    // A failed null fit surfaces as NaN (parallel_for_each_seeded has no error
    // channel); count it as an exceedance so it biases p upward, never downward.
    // That fail-soft rule is only sound for occasional failures: past half the
    // nulls, p saturates toward 1 and the test silently stops measuring
    // anything — error out instead (hardcoded 1/2; not worth a knob).
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

/// K-fold cross-validated R² for PLS1.
///
/// CV-R² convention (anchor — `find_k::select_cv` cites this): this function
/// pools `SS_res` and `SS_tot` across all folds and forms a single R² from the
/// pooled sums, with **unweighted** validation residuals even when training is
/// weighted. `select_cv` in `find_k.rs` deliberately uses the other convention —
/// per-fold **weighted-validation** R² averaged across folds — because k-selection
/// wants each fold weighted on its own scale. The two are not interchangeable;
/// each method owns the convention appropriate to its statistic.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
fn pls1_cv_r2(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    folds: &[Vec<usize>],
    weights: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
) -> PlsKitResult<f64> {
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{
        col_row_subset, normalize_weights, row_subset, standardize, standardize1,
        standardize1_weighted, standardize_apply, standardize_weighted,
    };

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for (fi, val_idx) in folds.iter().enumerate() {
        let train_idx: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != fi)
            .flat_map(|(_, f)| f.iter().copied())
            .collect();

        let x_tr = row_subset(x, &train_idx);
        let y_tr = col_row_subset(y, &train_idx);
        let x_val = row_subset(x, val_idx);
        let y_val = col_row_subset(y, val_idx);

        // Slice and re-normalize weights for the training fold.
        let w_tr_norm: Option<Col<f64>> = weights.map(|w| {
            let w_slice = col_row_subset(w, &train_idx);
            // Re-normalize so weights mean = 1 within this fold.
            normalize_weights(w_slice.as_ref())
                .unwrap_or_else(|| Col::from_fn(train_idx.len(), |_| 1.0))
        });
        let w_tr_ref: Option<ColRef<'_, f64>> = w_tr_norm.as_ref().map(Col::as_ref);

        let (xs_tr, x_mean, x_scale) = if let Some(w) = w_tr_ref {
            standardize_weighted(x_tr.as_ref(), Some(w))
        } else {
            standardize(x_tr.as_ref())
        };
        let xs_val = standardize_apply(x_val.as_ref(), x_mean.as_ref(), x_scale.as_ref());
        let (ys_tr, y_mean, y_scale) = if let Some(w) = w_tr_ref {
            standardize1_weighted(y_tr.as_ref(), Some(w))
        } else {
            standardize1(y_tr.as_ref())
        };
        let ys_val = Col::<f64>::from_fn(y_val.nrows(), |i| (y_val[i] - y_mean) / y_scale);

        let m = pls1_fit(
            xs_tr.as_ref(),
            ys_tr.as_ref(),
            KSpec::Fixed(k),
            w_tr_ref,
            FitOpts {
                pre_standardized: true,
                // check_n_eff: false — per-fold slice may have low n_eff; let the math degrade
                // and rely on the parent statistic to absorb noise (see Option B contract)
                check_n_eff: false,
                // Seq inside the per-fold worker — outer Rayon owns the threadpool.
                par: crate::fit::ParChoice::Seq,
                keep,
            },
        )?;

        let y_pred: Col<f64> = &xs_val * &m.coef;

        let n_val = ys_val.nrows();
        let mean_val: f64 = (0..n_val).map(|i| ys_val[i]).sum::<f64>() / n_val as f64;

        ss_res += (0..n_val)
            .map(|i| (y_pred[i] - ys_val[i]).powi(2))
            .sum::<f64>();
        ss_tot += (0..n_val)
            .map(|i| (ys_val[i] - mean_val).powi(2))
            .sum::<f64>();
    }

    Ok(if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 5: `split_nb` and `split_exact`'s refit route (`run_split_perm`)
// ──────────────────────────────────────────────────────────────────────────────

/// One split-half partition of the row indices `0..n`.
struct SplitIdx {
    tr: Vec<usize>,
    te: Vec<usize>,
}

/// Draw the J split-half partitions used by `split_nb` and by `split_exact`'s
/// refit route.
///
/// Split fraction is hardcoded 50/50: NB calibration assumes balanced halves
/// and there is no scientific reason to vary it.
///
/// Drawn through `parallel_for_each_seeded` rather than a plain sequential loop
/// because that is exactly how `split_half_correlations` used to draw them from
/// inside its per-split worker: same parent state in, same J child seeds, and
/// `one_split` is still the first draw off each child. So lifting the draw out
/// to this function leaves `split_nb` seed-for-seed identical.
fn draw_splits(
    n: usize,
    k: usize,
    n_splits: usize,
    disable_parallelism: bool,
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
        disable_parallelism,
        |_, child| {
            let (tr, te) = one_split(n, n_train, child);
            SplitIdx { tr, te }
        },
    ))
}

/// Compute the split-half Pearson r on each supplied split. Port of
/// `_tests.py:161-186`.
///
/// Takes the splits rather than drawing them so a caller can hold one set of
/// splits fixed across many outcome vectors — what `split_exact`'s permutation
/// loop needs. Nothing here consumes randomness: given `(x, y, split)` the fit
/// and the r are deterministic, so the J splits map in parallel directly
/// instead of through `parallel_for_each_seeded`.
///
/// `x`/`y` are the **raw** (un-√w-scaled) inputs; `w_norm` carries the
/// mean-1-normalized weights when present. Each half is standardized with
/// **weighted** moments and then √w' row-scaled (Convention A, matching
/// `pls1_fit`). The reported statistic is the unweighted Pearson r on the
/// resulting √w-scaled test-half scores and test-half y — *not* the weighted
/// Pearson r on the original data, because the test-half centering subtracts
/// the plain mean of the √w-scaled values rather than the weighted mean. The
/// identical transform is applied to observed and permuted data, and the
/// per-half weights stay tied to their rows (never permuted), so permutation
/// calibration still holds.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
fn split_half_correlations(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    splits: &[SplitIdx],
    w_norm: Option<ColRef<'_, f64>>,
    disable_parallelism: bool,
    keep: Option<usize>,
) -> Col<f64> {
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{
        col_row_subset, normalize_weights, row_subset, standardize, standardize1,
        standardize1_weighted, standardize_apply, standardize_weighted,
    };

    let per_split = |sp: &SplitIdx| -> f64 {
        let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
        let x_tr = row_subset(x, tr);
        let y_tr = col_row_subset(y, tr);
        let x_te = row_subset(x, te);
        let y_te = col_row_subset(y, te);

        // Per-half weights, re-normalized to mean 1 within the half (mirrors
        // pls1_cv_r2's per-fold renormalization). √w of each half is applied
        // *after* weighted standardization so the per-half fit matches what
        // pls1_fit would compute on that half's (X, y, w).
        let w_tr: Option<Col<f64>> = w_norm.map(|w| {
            let s = col_row_subset(w, tr);
            normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(tr.len(), |_| 1.0))
        });
        let w_te: Option<Col<f64>> = w_norm.map(|w| {
            let s = col_row_subset(w, te);
            normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(te.len(), |_| 1.0))
        });
        let w_tr_ref = w_tr.as_ref().map(Col::as_ref);

        let (xs_tr, x_mean, x_scale) = if let Some(w) = w_tr_ref {
            standardize_weighted(x_tr.as_ref(), Some(w))
        } else {
            standardize(x_tr.as_ref())
        };
        let xs_te = standardize_apply(x_te.as_ref(), x_mean.as_ref(), x_scale.as_ref());
        let (ys_tr, _, _) = if let Some(w) = w_tr_ref {
            standardize1_weighted(y_tr.as_ref(), Some(w))
        } else {
            standardize1(y_tr.as_ref())
        };

        // √w' row-scaling on top of weighted standardization (Convention A).
        let (xs_tr, ys_tr) = match w_tr_ref {
            Some(w) => (
                Mat::<f64>::from_fn(xs_tr.nrows(), xs_tr.ncols(), |i, j| {
                    xs_tr[(i, j)] * w[i].sqrt()
                }),
                Col::<f64>::from_fn(ys_tr.nrows(), |i| ys_tr[i] * w[i].sqrt()),
            ),
            None => (xs_tr, ys_tr),
        };
        let xs_te = match w_te.as_ref() {
            Some(w) => Mat::<f64>::from_fn(xs_te.nrows(), xs_te.ncols(), |i, j| {
                xs_te[(i, j)] * w[i].sqrt()
            }),
            None => xs_te,
        };

        let Ok(m) = pls1_fit(
            xs_tr.as_ref(),
            ys_tr.as_ref(),
            KSpec::Fixed(k),
            None,
            FitOpts {
                pre_standardized: true,
                // check_n_eff: false — per-half fit may truncate (small half,
                // sparse keep); a truncated model still yields a valid r at
                // k_used, which beats silently recording r=0 via the Err arm.
                check_n_eff: false,
                // Seq inside the per-half worker — outer Rayon owns the threadpool.
                par: crate::fit::ParChoice::Seq,
                keep,
            },
        ) else {
            return 0.0;
        };

        // scores on test half = X_te * coef. Both scores (via √w-scaled
        // xs_te) and y_te carry √w' so the Pearson r is taken on √w-scaled
        // data, matching Convention A.
        let scores_te: Col<f64> = &xs_te * &m.coef;
        let n_te = scores_te.nrows();
        let y_te: Col<f64> = match w_te.as_ref() {
            Some(w) => Col::<f64>::from_fn(n_te, |i| y_te[i] * w[i].sqrt()),
            None => y_te,
        };

        let s_mean: f64 = (0..n_te).map(|i| scores_te[i]).sum::<f64>() / n_te as f64;
        let y_mean: f64 = (0..n_te).map(|i| y_te[i]).sum::<f64>() / n_te as f64;

        let scores_c = Col::<f64>::from_fn(n_te, |i| scores_te[i] - s_mean);
        let y_c = Col::<f64>::from_fn(n_te, |i| y_te[i] - y_mean);

        let ss_s: f64 = (0..n_te).map(|i| scores_c[i] * scores_c[i]).sum();
        let ss_y: f64 = (0..n_te).map(|i| y_c[i] * y_c[i]).sum();

        // Mirrored in split_perm_nr_zbars — change together; the mirror
        // explanation lives there (split_perm_nr duplicates this
        // arithmetic deliberately, since it removes the fit this
        // function performs above).
        if ss_s < 1e-15 || ss_y < 1e-15 {
            return 0.0;
        }

        let cross: f64 = (0..n_te).map(|i| scores_c[i] * y_c[i]).sum();
        let r = cross / (ss_s * ss_y).sqrt();
        r.clamp(-1.0, 1.0)
    };

    // Same shape as split_perm_nr_zbars' per-split dispatch: collect preserves
    // split order in both arms, so serial and parallel results are byte-equal.
    let r_vec: Vec<f64> = if disable_parallelism {
        splits.iter().map(per_split).collect()
    } else {
        use rayon::prelude::*;
        splits.par_iter().map(per_split).collect()
    };

    Col::<f64>::from_fn(r_vec.len(), |i| r_vec[i])
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
    // per-half weighted-standardize-then-√w (Convention A) internally.
    let splits = draw_splits(n, k, n_splits, opts.disable_parallelism, rng)?;
    let r_splits = split_half_correlations(
        x,
        y,
        k,
        &splits,
        w_norm,
        opts.disable_parallelism,
        opts.keep,
    );
    let (p, mean_r, _t_stat, _df) = nb_test(&r_splits, n_train, n_test);

    // ρ̂ carve-out (spec "Salvage from the working tree" — moved here from the
    // retired split_j_eff arm, since the paper's Lemma 2 depends on it). The
    // ruler σ₀² = 1/(n_test−3) needs unweighted input and n_test ≥ 4; None
    // otherwise. ρ̂ itself is clip(1 − s²·(n_test−3), 0, 1), s² = var(z_j) with
    // ddof=1 — the same s² nb_test already computes internally for its own SE,
    // recomputed here rather than threaded out, since it is cheap relative to
    // the resampling loop above and keeps nb_test's signature untouched
    // (nb_test is shared history with the committed split_nb path — do not
    // reshape it for a caller-side quantity it doesn't need itself).
    let rho_hat = if w_norm.is_none() && n_test >= 4 {
        let j = r_splits.nrows() as f64;
        // ±0.9999 pre-atanh clamp mirrored from nb_test above (change
        // together) — nb_test owns the explanation.
        let z_vec: Vec<f64> = (0..r_splits.nrows())
            .map(|i| r_splits[i].clamp(-0.9999, 0.9999).atanh())
            .collect();
        let z_mean: f64 = z_vec.iter().sum::<f64>() / j;
        let s2: f64 = z_vec.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / (j - 1.0);
        let sigma0_sq = 1.0 / (n_test as f64 - 3.0);
        Some((1.0 - s2 / sigma0_sq).clamp(0.0, 1.0))
    } else {
        None
    };
    Ok(RunResult {
        pvalue: p,
        statistic: mean_r,
        rho_hat,
    })
}

/// NB t-test on Fisher-z transforms. Port of `_core.py:32-89`.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn nb_test(stats: &Col<f64>, n_train: usize, n_test: usize) -> (f64, f64, f64, f64) {
    let j = stats.nrows() as f64;
    // Fisher-z transform. The ±0.9999 pre-atanh clamp is mirrored in
    // split_perm_nr_zbars, in mean_fisher_z, and in run_split_nb's rho_hat
    // block below — change together; this site owns the explanation (keeps z̄
    // finite at |r| = 1).
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

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn run_split_perm(
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
    // replicates. Redrawing them per replicate (what this function used to do)
    // folds split-to-split scatter into the null, so the reference
    // distribution stops isolating the y–X association the observed statistic
    // measures. split_perm_nr_zbars has always worked this way; this is the
    // route that had to move.
    let splits = draw_splits(n, k, n_splits, opts.disable_parallelism, rng)?;

    // Raw (X, y, weights) flow into split_half_correlations (Convention A
    // weighted-standardize-then-√w internally); the permutation loop reuses the
    // raw x and permutes raw y rows.
    let r_obs = split_half_correlations(
        x,
        y,
        k,
        &splits,
        w_norm,
        opts.disable_parallelism,
        opts.keep,
    );
    let z_bar_obs = mean_fisher_z(&r_obs);

    let null_zbars = crate::resample::parallel_for_each_seeded(
        rng,
        n_perm,
        opts.disable_parallelism,
        |_, outer_rng| {
            let perm = crate::resample::permute_indices(n, outer_rng);
            // Permute raw y rows; weights stay tied to row positions (w_norm is
            // passed unpermuted), so w[i] always pairs with destination row i.
            let y_perm = Col::<f64>::from_fn(n, |i| y[perm[i]]);
            let r_null = split_half_correlations(
                x,
                y_perm.as_ref(),
                k,
                &splits,
                w_norm,
                opts.disable_parallelism,
                opts.keep,
            );
            mean_fisher_z(&r_null)
        },
    );

    // A non-finite null statistic counts as an exceedance so it biases p
    // upward, never downward (mirrors run_raw_perm and run_split_perm_nr —
    // change together). A failed per-half fit already degrades to r = 0 inside
    // split_half_correlations rather than surfacing here.
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

/// `z̄`: the mean Fisher-z of a vector of split-half correlations. `tanh` of
/// this is the statistic `split_nb` and `split_exact` both report, and on the
/// permutation routes it is also the scale the null comparison happens on —
/// mean-of-z is not a monotone function of mean-of-r, so comparing on one
/// scale and reporting the other would give a p-value for a different test.
fn mean_fisher_z(r: &Col<f64>) -> f64 {
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
// Step 5b: `split_exact`'s no-refit route (`run_split_perm_nr` / `split_perm_nr_zbars`)
// ──────────────────────────────────────────────────────────────────────────────

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
/// The identity survives Convention A intact, but the map is not the same
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
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::items_after_statements)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)] // per-half Convention A setup inflates the per-split closure
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
        col_row_subset, normalize_weights, row_subset, standardize, standardize_apply,
        standardize_weighted,
    };
    use crate::resample::{one_split, permute_indices, split_sizes};

    // Scope limits, checked here (this route's own runner), never in
    // split_half_correlations — that function is shared with run_split_perm and
    // knows nothing of this formula. Guard, not fallback: ineligible input
    // errors rather than silently running the run_split_perm route. Weights
    // are *not* a scope limit: see "Under weights" above.
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

    // Draw the J splits once, fixed across all B permutation draws (spec "The
    // algorithm", step 1). Drawn sequentially off the parent RNG here rather
    // than through draw_splits' parallel_for_each_seeded — the two split_exact
    // routes are deliberately not stream-unified, so this route keeps the draw
    // order its equivalence tests are anchored to. Only the (tr, te) index
    // pairs are drawn here; standardization is deferred to per_split_z below so
    // peak memory stays O(np) plus the (B+1)-column blocks, per spec "Cost" —
    // materializing xs_tr/xs_te for all J splits up front would be O(J·n·p)
    // (≈2.1 GB at the embedding-scale n=13365, p=400, J=50 case).
    // standardize/standardize_apply are pure functions of x[tr]/x[te] with no
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
    let perms: Vec<Vec<usize>> = (0..n_perm).map(|_| permute_indices(n, rng)).collect();
    let y_mat = Mat::<f64>::from_fn(n, n_cols, |i, col| {
        if col == 0 {
            y[i]
        } else {
            y[perms[col - 1][i]]
        }
    });
    // The B·n index vectors are dead once y_mat is built.
    drop(perms);

    // Choose the association order once, outside the per-split loop: all
    // splits share (n_train, n_test) since split_sizes depends only on
    // (n, k). Route B wins when n_te·n_tr·(p+B) < n·p·B (spec "Choose the
    // association order"). No caller-facing knob — the cost model decides.
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
        let x_tr = row_subset(x, &sp.tr);
        let x_te = row_subset(x, &sp.te);

        // Per-half weights renormalized to mean 1 within the half, weighted
        // moments, then √w row-scaling — every step mirrored from
        // split_half_correlations (change together; that function owns the
        // Convention A explanation). `normalize_weights` only returns None on
        // an all-zero half, which the parent's weight validation makes
        // unreachable; fall back to uniform there exactly as that function
        // does rather than growing an error path this route cannot hit.
        let half_weights = |idx: &[usize]| -> Option<Col<f64>> {
            w_norm.map(|w| {
                let s = col_row_subset(w, idx);
                normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(idx.len(), |_| 1.0))
            })
        };
        let w_tr = half_weights(&sp.tr);
        let w_te = half_weights(&sp.te);
        // Root taken once per row here, not once per (row, column) — with
        // n_cols = B+1 columns of Y the inline form would repeat it B times.
        let root = |w: &Col<f64>| Col::<f64>::from_fn(w.nrows(), |i| w[i].sqrt());
        let sw_tr = w_tr.as_ref().map(root);
        let sw_te = w_te.as_ref().map(root);

        let (xs_tr, mean, scale) = match w_tr.as_ref() {
            Some(w) => standardize_weighted(x_tr.as_ref(), Some(w.as_ref())),
            None => standardize(x_tr.as_ref()),
        };
        let xs_te = standardize_apply(x_te.as_ref(), mean.as_ref(), scale.as_ref());

        let y_tr = row_subset(y_mat.as_ref(), &sp.tr);
        let y_te = row_subset(y_mat.as_ref(), &sp.te);
        let n_te = sp.te.len();

        // Convention A row-scaling. On the train side the √w_tr on Y is the
        // `diag(√w_tr)` that sits *inside* the linear map (see "Under
        // weights"): X̃_tr already carries one √w_tr, and the raw y needs the
        // other. On the test side both the scores (through X̃_te) and the raw
        // test y carry √w_te, so the Pearson r below is taken on √w_te-scaled
        // data — matching what split_half_correlations reports.
        let scale_rows = |m: Mat<f64>, sw: Option<&Col<f64>>| match sw {
            Some(sw) => Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * sw[i]),
            None => m,
        };
        let xs_tr = scale_rows(xs_tr, sw_tr.as_ref());
        let xs_te = scale_rows(xs_te, sw_te.as_ref());
        let y_tr = scale_rows(y_tr, sw_tr.as_ref());
        let y_te = scale_rows(y_te, sw_te.as_ref());

        // Same two GEMM calls either way — only the operand grouping differs.
        let t_te: Mat<f64> = if route_b {
            let m = xs_te.as_ref() * xs_tr.transpose(); // n_te x n_tr
            m.as_ref() * y_tr.as_ref() // n_te x n_cols
        } else {
            let g = xs_tr.transpose() * y_tr.as_ref(); // p x n_cols
            xs_te.as_ref() * g.as_ref() // n_te x n_cols
        };

        (0..n_cols)
            .map(|col| {
                let s_mean: f64 = (0..n_te).map(|i| t_te[(i, col)]).sum::<f64>() / n_te as f64;
                let y_mean: f64 = (0..n_te).map(|i| y_te[(i, col)]).sum::<f64>() / n_te as f64;
                let ss_s: f64 = (0..n_te).map(|i| (t_te[(i, col)] - s_mean).powi(2)).sum();
                let ss_y: f64 = (0..n_te).map(|i| (y_te[(i, col)] - y_mean).powi(2)).sum();

                // Guard mirrored from split_half_correlations (change
                // together): a degenerate column returns r = 0.0, never
                // skipped or NaN, so the equivalence test agrees on the rare
                // draw that hits it.
                let r = if ss_s < 1e-15 || ss_y < 1e-15 {
                    0.0
                } else {
                    let cross: f64 = (0..n_te)
                        .map(|i| (t_te[(i, col)] - s_mean) * (y_te[(i, col)] - y_mean))
                        .sum();
                    (cross / (ss_s * ss_y).sqrt()).clamp(-1.0, 1.0)
                };
                // ±0.9999 pre-atanh clamp mirrored from nb_test (change
                // together): keeps the statistic identical to split_nb's and
                // z̄ finite at |r| = 1.
                r.clamp(-0.9999, 0.9999).atanh()
            })
            .collect()
    };

    let per_split: Vec<Vec<f64>> = if opts.disable_parallelism {
        splits.iter().map(per_split_z).collect()
    } else {
        use rayon::prelude::*;
        splits.par_iter().map(per_split_z).collect()
    };

    let j = n_splits as f64;
    let mut z_sum = vec![0.0_f64; n_cols];
    for per in &per_split {
        for (col, v) in per.iter().enumerate() {
            z_sum[col] += v;
        }
    }
    Ok(z_sum.into_iter().map(|s| s / j).collect())
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
// Step 6: score test (Welch-Satterthwaite generalized χ²)
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
    let (xs, _, _) = if opts.pre_standardized {
        let d = x.ncols();
        (
            Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)]),
            Col::<f64>::zeros(d),
            Col::<f64>::from_fn(d, |_| 1.0),
        )
    } else if w_norm.is_some() {
        standardize_weighted(x, w_norm)
    } else {
        standardize(x)
    };

    let (ys, _, _) = if opts.pre_standardized {
        (Col::<f64>::from_fn(n, |i| y[i]), 0.0_f64, 1.0_f64)
    } else if w_norm.is_some() {
        standardize1_weighted(y, w_norm)
    } else {
        standardize1(y)
    };

    // When weights are present, further row-scale the standardized data by √w'.
    // T_w = ||X̃'ỹ||² where X̃ = diag(√w')·X_std, ỹ = diag(√w')·y_std.
    // This equals the unweighted T on (X̃, ỹ).
    let (xs_eff, ys_eff) = if let Some(w) = w_norm {
        let xs_w = Mat::<f64>::from_fn(n, xs.ncols(), |i, j| xs[(i, j)] * w[i].sqrt());
        let ys_w = Col::<f64>::from_fn(n, |i| ys[i] * w[i].sqrt());
        (xs_w, ys_w)
    } else {
        (xs, ys)
    };

    // T_obs = ||X'y||² = y'XX'y
    let xy: Col<f64> = xs_eff.transpose() * &ys_eff;
    let t_obs: f64 = (0..xy.nrows()).map(|i| xy[i].powi(2)).sum::<f64>();

    // Eigenvalues of the smaller Gram matrix (X'X for d≤n, XX' otherwise).
    let nn = xs_eff.nrows();
    let d = xs_eff.ncols();
    let lambdas: Col<f64> = if d <= nn {
        let gram: Mat<f64> = xs_eff.transpose() * xs_eff.as_ref();
        eigenvalues_symmetric(gram.as_ref())
    } else {
        let gram: Mat<f64> = xs_eff.as_ref() * xs_eff.transpose();
        eigenvalues_symmetric(gram.as_ref())
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

/// Symmetric eigenvalues via faer's `self_adjoint_eigen`. Returns eigenvalues ascending.
/// `Side::Lower` is pinned for byte-parity stability.
fn eigenvalues_symmetric(a: MatRef<'_, f64>) -> Col<f64> {
    // Returns Result<SelfAdjointEigen<f64>, EvdError>; unwrap is safe for
    // PD/PSD Gram matrices. If the matrix is degenerate (n=0 or all-zero),
    // return a zero Col — the caller guards s1 < 1e-15.
    match a.self_adjoint_eigen(faer::Side::Lower) {
        Ok(eig) => eig.S().column_vector().to_owned(),
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
        // Series expansion for the lower tail P(a, x); Q = 1 − P. Identical to
        // the prior series branch — only the final complement differs.
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
// Step 7: universal-inference split-LR e-value
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
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{
        col_row_subset, normalize_weights, row_subset, standardize, standardize1,
        standardize1_weighted, standardize_apply, standardize_weighted,
    };
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

    // Split the raw data, then per-half weighted-standardize-then-√w
    // (Convention A, matching split_half_correlations / pls1_fit). Per-half
    // weights are re-normalized to mean 1 within the half (mirrors pls1_cv_r2).
    let x_tr = row_subset(x, &tr);
    let y_tr = col_row_subset(y, &tr);
    let x_te = row_subset(x, &te);
    let y_te = col_row_subset(y, &te);

    let w_tr: Option<Col<f64>> = w_norm.map(|w| {
        let s = col_row_subset(w, &tr);
        normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(tr.len(), |_| 1.0))
    });
    let w_te: Option<Col<f64>> = w_norm.map(|w| {
        let s = col_row_subset(w, &te);
        normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(te.len(), |_| 1.0))
    });
    let w_tr_ref = w_tr.as_ref().map(Col::as_ref);

    let (xs_tr, x_mean, x_scale) = if let Some(w) = w_tr_ref {
        standardize_weighted(x_tr.as_ref(), Some(w))
    } else {
        standardize(x_tr.as_ref())
    };
    let xs_te = standardize_apply(x_te.as_ref(), x_mean.as_ref(), x_scale.as_ref());
    let (ys_tr, y_mean, y_scale) = if let Some(w) = w_tr_ref {
        standardize1_weighted(y_tr.as_ref(), Some(w))
    } else {
        standardize1(y_tr.as_ref())
    };
    let n_te = y_te.nrows();
    let ys_te = Col::<f64>::from_fn(n_te, |i| (y_te[i] - y_mean) / y_scale);

    // √w' row-scaling on top of weighted standardization, train and test halves.
    let (xs_tr, ys_tr) = match w_tr_ref {
        Some(w) => (
            Mat::<f64>::from_fn(xs_tr.nrows(), xs_tr.ncols(), |i, j| {
                xs_tr[(i, j)] * w[i].sqrt()
            }),
            Col::<f64>::from_fn(ys_tr.nrows(), |i| ys_tr[i] * w[i].sqrt()),
        ),
        None => (xs_tr, ys_tr),
    };
    let (xs_te, ys_te) = match w_te.as_ref() {
        Some(w) => (
            Mat::<f64>::from_fn(xs_te.nrows(), xs_te.ncols(), |i, j| {
                xs_te[(i, j)] * w[i].sqrt()
            }),
            Col::<f64>::from_fn(n_te, |i| ys_te[i] * w[i].sqrt()),
        ),
        None => (xs_te, ys_te),
    };

    let m = pls1_fit(
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

    let y_pred: Col<f64> = &xs_te * &m.coef;

    // Universal inference fixes the numerator density on the training half:
    // σ²_alt is the residual MLE from predicting the training X with the fitted
    // model, evaluated against training y. Computing it on the test half would
    // make the likelihood ratio data-dependent and break the e-value guarantee.
    let n_tr = ys_tr.nrows();
    let y_pred_tr: Col<f64> = &xs_tr * &m.coef;
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
    // design); disable_parallelism moot (single split, no inner loop).
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

    fn synth_with_signal(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < 3 { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
        (x, y)
    }

    fn synth_no_signal(n: usize, d: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        (x, y)
    }

    // Smoke: ensure we can fit and immediately confirm.
    #[test]
    fn fit_then_confirm_smoke() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 1);
        let _ = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let r = pls1_confirmatory_test(
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
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!((0.0..=1.0).contains(&r.pvalue));
    }

    // ── raw_perm tests ───────────────────────────────────────────────────────

    #[test]
    fn raw_perm_calibration_under_h0() {
        let (x, y) = synth_no_signal(40, 5, 99);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 200,
                    n_folds: 5,
                },
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "raw_perm");
        assert!((0.0..=1.0).contains(&r.pvalue));
        // No assertion on p — calibration tested in plskit-py statistical suite.
    }

    #[test]
    fn raw_perm_rejects_under_signal() {
        let (x, y) = synth_with_signal(80, 6, 5.0, 11);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 3,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 200,
                    n_folds: 5,
                },
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.pvalue < 0.05, "p={}", r.pvalue);
    }

    // ── split_nb and split_exact tests ──────────────────────────────────────

    #[test]
    fn split_nb_rejects_under_signal() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let r = pls1_confirmatory_test(
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
                seed: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "split_nb");
        assert!(r.pvalue < 0.1, "p={}", r.pvalue);
    }

    // k=2 forces split_exact's refit route (run_split_perm) — moved from the
    // old standalone split_perm method test, which this replaces.
    #[test]
    fn split_exact_refit_route_runs_with_signal() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 23);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 100,
                    n_splits: 20,
                },
                seed: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "split_exact");
        assert!((0.0..=1.0).contains(&r.pvalue));
        // The statistic is tanh(z̄) on this path too, so it stays a correlation
        // and remains in (-1, 1) — the clamp before atanh bounds |z̄| well
        // short of infinity.
        assert!(r.statistic.abs() < 1.0, "statistic={}", r.statistic);
    }

    // ── split_exact tests ────────────────────────────────────────────────────

    // §7 route agreement. The two routes draw splits from different RNG
    // streams (no-refit sequentially off the parent, refit through
    // parallel_for_each_seeded child RNGs), so they never agree seed-for-seed.
    // Injecting the same splits into both removes that difference and turns
    // the check into the exact statement the method rests on: at K = 1 the
    // batched linear map and an honest per-half pls1_fit produce the same
    // test-half correlation, hence the same tanh(z̄).
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn split_exact_routes_agree_on_identical_splits() {
        let n = 60_usize;
        // Weighted case: the same identity has to survive Convention A
        // (per-half weighted moments + √w row-scaling), which is the whole
        // claim the no-refit route makes about weighted input.
        let w = crate::linalg::normalize_weights(
            Col::<f64>::from_fn(n, |i| if i % 3 == 0 { 2.0 } else { 0.5 }).as_ref(),
        )
        .unwrap();
        assert_split_exact_routes_agree(n, None);
        assert_split_exact_routes_agree(n, Some(w.as_ref()));
    }

    #[allow(clippy::many_single_char_names)]
    fn assert_split_exact_routes_agree(n: usize, w_norm: Option<ColRef<'_, f64>>) {
        use crate::resample::{one_split, split_sizes};

        let (p, k) = (8_usize, 1_usize);
        let (n_perm, n_splits, seed) = (9_usize, 6_usize, 31_u64);
        let (x, y) = synth_with_signal(n, p, 3.0, 13);
        let opts = ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            ..Default::default()
        };

        let (_, mut rng_nr) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let z_bar_nr = split_perm_nr_zbars(
            x.as_ref(),
            y.as_ref(),
            k,
            n_perm,
            n_splits,
            w_norm,
            &opts,
            &mut rng_nr,
        )
        .unwrap();

        // Replay the no-refit route's split draw exactly (same seed, same
        // sequential one_split calls), then hand those splits to the refit
        // route. Only the observed column exists on this side — the null
        // columns depend on permutation draws the two routes do not share.
        let (_, mut rng_re) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let (n_train, _) = split_sizes(n, k);
        let splits: Vec<SplitIdx> = (0..n_splits)
            .map(|_| {
                let (tr, te) = one_split(n, n_train, &mut rng_re);
                SplitIdx { tr, te }
            })
            .collect();
        let z_bar_re = mean_fisher_z(&split_half_correlations(
            x.as_ref(),
            y.as_ref(),
            k,
            &splits,
            w_norm,
            false,
            None,
        ));

        let scale = z_bar_nr[0].abs().max(z_bar_re.abs());
        let rel = if scale == 0.0 {
            0.0
        } else {
            (z_bar_nr[0] - z_bar_re).abs() / scale
        };
        assert!(
            rel < 1e-10,
            "weighted={}: no-refit z̄={} refit z̄={z_bar_re} rel={rel}",
            w_norm.is_some(),
            z_bar_nr[0]
        );
    }

    // Route selection is by input shape, not by a caller knob. Each case is
    // pinned bit-for-bit against the route it must land on, so a future edit
    // to the predicate fails here rather than silently switching routes.
    // Task 8 parked this rewrite at the test's creation: `SplitPermNr` and
    // `SplitPerm` are no longer selectable methods, so the comparator side
    // now calls `run_split_perm_nr` / `run_split_perm` directly instead of
    // going through `pls1_confirmatory_test` with those (now-deleted) args
    // variants. `run_route` reproduces exactly the preprocessing
    // `pls1_confirmatory_test` performs ahead of dispatch — weight
    // normalization and `resolve_seed` — so the comparator sees the same
    // `(w_norm, rng)` state dispatch would hand the route. `SplitExact` args
    // never trip the `split_nb` auto-gate (the only other RNG consumer ahead
    // of dispatch), so no extra RNG draws sit between `resolve_seed` and the
    // route call on either side.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn split_exact_selects_route_from_input_shape() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 1.5 } else { 0.5 });
        let (n_perm, n_splits) = (49_usize, 5_usize);
        let n = 60_usize;

        let run_exact = |k: usize, weights: Option<ColRef<'_, f64>>, keep| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k,
                    weights,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
                    seed: Some(2),
                    keep,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let run_route = |k: usize, weights: Option<ColRef<'_, f64>>, keep, refit: bool| {
            let (w_norm, _, _) = crate::fit::validate_and_normalize_weights(weights, n, k).unwrap();
            let (_, mut rng) = crate::rng::resolve_seed(Some(2)).unwrap();
            let opts = ConfirmatoryTestOpts {
                keep,
                ..Default::default()
            };
            if refit {
                run_split_perm(
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
            let a = run_exact(1, weights, None);
            let b = run_route(1, weights, None, false);
            assert_eq!(a.method, "split_exact");
            assert_eq!(
                a.pvalue.to_bits(),
                b.pvalue.to_bits(),
                "weighted={} took the wrong route",
                weights.is_some()
            );
            assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
        }

        // k>1 and sparse keep each independently force the refit route.
        // Both cases are unweighted: weights no longer affect route choice,
        // so pairing them with these two would test nothing extra.
        for (k, keep) in [(2, None), (1, Some(3))] {
            let e = run_exact(k, None, keep);
            let s = run_route(k, None, keep, true);
            assert_eq!(e.method, "split_exact");
            assert_eq!(
                e.pvalue.to_bits(),
                s.pvalue.to_bits(),
                "k={k} keep={keep:?} took the wrong route"
            );
            assert_eq!(e.statistic.to_bits(), s.statistic.to_bits());
        }
    }

    // ── split_exact's no-refit route (split_perm_nr) tests ──────────────────

    // Test 1 (write first, per spec): the equivalence test. Route A is an
    // honest per-split, per-column pls1_fit refit; route B is the shipped
    // batched path (split_perm_nr_zbars). Same seed ⇒ same splits and same
    // permuted Y columns for both routes (each draws the identical sequence
    // of one_split/permute_indices calls from the same RNG stream), so they
    // evaluate the same quantity two ways and must agree — this is the
    // property the method rests on (spec "Why no refits — the identity").
    // Small B=49, J=5 so the refitting route is affordable in a unit test.
    //
    // Shared by two call sites below (not parameterized #[test] — the crate
    // has no test-matrix macro in use elsewhere) so BOTH association orders
    // from "Choose the association order" are exercised: (n=60, p=5) takes
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
        let (x, y) = synth_with_signal(n, p, snr, data_seed);
        let n_cols = n_perm + 1;

        // Globally mean-1-normalized weights, exactly what
        // pls1_confirmatory_test hands the runner. Deliberately unequal and
        // not aligned to the split boundary, so per-half renormalization
        // actually changes the numbers (a uniform w would pass even if the
        // implementation ignored weights entirely).
        let w_all: Option<Col<f64>> = weighted.then(|| {
            normalize_weights(
                Col::<f64>::from_fn(n, |i| 0.25 + ((i * 7) % 5) as f64 * 0.5).as_ref(),
            )
            .unwrap()
        });
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
                // Convention A, spelled out the long way (this is the
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
                // already baked into sp.xs_tr (Convention A).
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
                let r = if ss_s < 1e-15 || ss_y < 1e-15 {
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
        // Pure relative tolerance, per spec ("do not silently relax the
        // 1e-10") — no absolute-tolerance escape hatch. Scale by the larger
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

    #[test]
    fn split_perm_nr_route_a_matches_honest_refit_reference() {
        // n=60, p=5, B+1=50: route_b cost 30·30·55=49,500 > route A cost
        // 60·5·50=15,000 ⇒ route A is the active branch here.
        assert_split_perm_nr_route_matches_honest_refit(60, 5, 4.0, 7, 49, 5, 11, false, false);
    }

    // Same two association orders again, weighted. This is the check that
    // licenses the weighted derivation: the batched map folds diag(√w_tr)
    // into a fixed linear map, the reference refits honestly under
    // Convention A, and every one of the B+1 columns must still agree.
    #[test]
    fn split_perm_nr_route_a_matches_honest_refit_weighted() {
        assert_split_perm_nr_route_matches_honest_refit(60, 5, 4.0, 7, 49, 5, 11, false, true);
    }

    #[test]
    fn split_perm_nr_route_b_matches_honest_refit_weighted() {
        assert_split_perm_nr_route_matches_honest_refit(60, 40, 4.0, 7, 49, 5, 11, true, true);
    }

    #[test]
    fn split_perm_nr_route_b_matches_honest_refit_reference() {
        // n=60, p=40, B+1=50: route_b cost 30·30·90=81,000 < route A cost
        // 60·40·50=120,000 ⇒ route B is the active branch here — the branch
        // production actually uses on the near-singular grid (n≤320, p=400,
        // B=1000).
        assert_split_perm_nr_route_matches_honest_refit(60, 40, 4.0, 7, 49, 5, 11, true, false);
    }

    // Same case as above, degenerate: every X row identical ⇒ standardize's
    // zero-variance branch makes xs_tr exactly the zero matrix, so both
    // routes' test-half scores are constant (zero) for every split/column —
    // the guard must fire and return r = 0.0 for both, never NaN.
    #[test]
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    #[allow(clippy::float_cmp)] // exact 0.0 is the point: the guard must never emit NaN
    fn split_perm_nr_guard_constant_scores_give_zero_correlation() {
        use crate::linalg::{row_subset, standardize, standardize1, standardize_apply};
        use crate::resample::{one_split, split_sizes};

        let n = 20;
        let d = 3;
        let x = Mat::<f64>::from_fn(n, d, |_, j| (j + 1) as f64); // every row identical
        let y = Col::<f64>::from_fn(n, |i| i as f64);
        let (n_perm, n_splits, k, seed) = (9_usize, 4_usize, 1_usize, 3_u64);

        let (_, mut rng_b) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let z_bar_b = split_perm_nr_zbars(
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
            &mut rng_b,
        )
        .unwrap();
        for zb in &z_bar_b {
            assert_eq!(*zb, 0.0, "route B must return exactly 0, not NaN");
        }

        // Route A: honest refit hits the same degeneracy (xs_tr = 0 ⇒
        // pls1_fit's Err arm or a zero coef; either way scores_te is
        // constant), and the guard above returns 0.0 rather than propagating
        // pls1_fit's failure into a skipped/NaN entry.
        let (_, mut rng_a) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let (n_train, _) = split_sizes(n, k);
        for _ in 0..n_splits {
            let (tr, te) = one_split(n, n_train, &mut rng_a);
            let x_tr = row_subset(x.as_ref(), &tr);
            let x_te = row_subset(x.as_ref(), &te);
            let (xs_tr, mean, scale) = standardize(x_tr.as_ref());
            let xs_te = standardize_apply(x_te.as_ref(), mean.as_ref(), scale.as_ref());
            let y_tr_col = crate::linalg::col_row_subset(y.as_ref(), &tr);
            let (ys_tr, _, _) = standardize1(y_tr_col.as_ref());
            let n_te = te.len();
            let (s_mean, ss_s) = match pls1_fit(
                xs_tr.as_ref(),
                ys_tr.as_ref(),
                KSpec::Fixed(1),
                None,
                FitOpts {
                    pre_standardized: true,
                    check_n_eff: false,
                    ..Default::default()
                },
            ) {
                Ok(m) => {
                    let scores_te: Col<f64> = &xs_te * &m.coef;
                    let s_mean: f64 = (0..n_te).map(|i| scores_te[i]).sum::<f64>() / n_te as f64;
                    let ss_s: f64 = (0..n_te).map(|i| (scores_te[i] - s_mean).powi(2)).sum();
                    (s_mean, ss_s)
                }
                Err(_) => (0.0, 0.0),
            };
            assert!(
                ss_s < 1e-15,
                "expected the honest-refit route to also hit the zero-variance guard: ss_s={ss_s}, s_mean={s_mean}"
            );
        }
    }

    // Test 3: determinism. Same seed ⇒ identical p; different seed ⇒
    // generally different p. Splits are drawn once, so same seed ⇒ same
    // splits too — checked here via the identical statistic (tanh(z̄[0])),
    // which only matches if both the splits and the observed column agree.
    // k=1, dense, unweighted ⇒ split_exact's no-refit route (SplitPermNr's
    // former public identity) via the public API.
    #[test]
    fn split_exact_no_refit_route_deterministic_under_same_seed() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 5);
        let mk = |seed| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: 1,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 100,
                        n_splits: 10,
                    },
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let a1 = mk(42);
        let a2 = mk(42);
        assert_eq!(a1.pvalue.to_bits(), a2.pvalue.to_bits());
        assert_eq!(a1.statistic.to_bits(), a2.statistic.to_bits());

        let b = mk(43);
        assert_ne!(
            a1.statistic.to_bits(),
            b.statistic.to_bits(),
            "different seed should (generally) draw different splits"
        );
    }

    // Test 4 (#[ignore], slow): exactness under the null. y drawn
    // independently of X ⇒ p is uniform to within Monte Carlo error over a
    // few hundred reps. Run deliberately (`cargo test -- --ignored`), not in
    // the fast suite.
    #[test]
    #[ignore = "slow MC null-uniformity check; run deliberately with --ignored"]
    fn split_exact_no_refit_route_null_p_is_uniform() {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut seed_rng = rand_chacha::ChaCha8Rng::seed_from_u64(2026);
        let n_reps = 300;
        let mut below_05 = 0;
        for _ in 0..n_reps {
            let seed: u64 = seed_rng.random();
            let (x, y) = synth_no_signal(40, 4, seed);
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: 1,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 199,
                        n_splits: 20,
                    },
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap();
            if r.pvalue < 0.5 {
                below_05 += 1;
            }
        }
        // Under H0, p ~ Uniform(0,1) ⇒ P(p<0.5) = 0.5. Binomial(300, 0.5)
        // std ≈ 8.7; allow a generous ±5 std Monte Carlo band.
        let frac = f64::from(below_05) / f64::from(n_reps);
        assert!(
            (0.35..=0.65).contains(&frac),
            "P(p<0.5) = {frac}, expected ≈ 0.5"
        );
    }

    // Test 5: guards. k=2 and sparse keep both error, each message naming
    // run_split_perm as the alternative route. Both guards live inside
    // split_perm_nr_zbars/run_split_perm_nr and are unreachable through the
    // public API now: split_exact's dispatch never calls this route for k!=1
    // or with keep set (it picks run_split_perm instead), so these tests call
    // run_split_perm_nr directly. k=1 dense succeeds (weighted included — see
    // split_exact_no_refit_route_accepts_weighted_input).
    #[test]
    fn split_exact_no_refit_route_guard_rejects_k_not_one() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let (_, mut rng) = crate::rng::resolve_seed(Some(2)).unwrap();
        let e = run_split_perm_nr(
            x.as_ref(),
            y.as_ref(),
            2,
            49,
            5,
            None,
            &ConfirmatoryTestOpts::default(),
            &mut rng,
        )
        .unwrap_err();
        assert_eq!(e.code(), "invalid_argument");
        assert!(e.to_string().contains("split_exact"), "msg={e}");
    }

    // Inverse of the guard this test used to assert: weighted k=1 dense input
    // is now in scope for the no-refit route, and the weights must actually
    // reach the statistic (a run that silently dropped them would return the
    // unweighted numbers bit-for-bit).
    #[test]
    fn split_exact_no_refit_route_accepts_weighted_input() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 1.5 } else { 0.5 });
        let run = |weights| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: 1,
                    weights,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 49,
                        n_splits: 5,
                    },
                    seed: Some(2),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let weighted = run(Some(w.as_ref()));
        let unweighted = run(None);
        assert_eq!(weighted.method, "split_exact");
        assert!((0.0..=1.0).contains(&weighted.pvalue));
        assert_ne!(
            weighted.statistic.to_bits(),
            unweighted.statistic.to_bits(),
            "weights were ignored"
        );
    }

    // Guard, not fallback: opts.keep is live sparse-fit plumbing honored by
    // raw_perm/split_nb/split_exact's refit route/e, but split_exact's
    // no-refit route performs no inner fits at all (see "Why no refits"), so
    // a caller-set keep must error rather than silently running dense with no
    // signal that keep was ignored. Calls run_split_perm_nr directly — see
    // the comment on split_exact_no_refit_route_guard_rejects_k_not_one for
    // why the public API cannot reach this guard.
    #[test]
    fn split_exact_no_refit_route_guard_rejects_sparse_keep() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let (_, mut rng) = crate::rng::resolve_seed(Some(2)).unwrap();
        let opts = ConfirmatoryTestOpts {
            keep: Some(2),
            ..Default::default()
        };
        let e =
            run_split_perm_nr(x.as_ref(), y.as_ref(), 1, 49, 5, None, &opts, &mut rng).unwrap_err();
        assert_eq!(e.code(), "invalid_argument");
        assert!(e.to_string().contains("split_exact"), "msg={e}");
    }

    #[test]
    fn split_exact_no_refit_route_succeeds_unweighted_k_one() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 49,
                    n_splits: 5,
                },
                seed: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "split_exact");
        assert!((0.0..=1.0).contains(&r.pvalue));
        assert!(r.pvalue >= 1.0 / 50.0 - 1e-12, "p={}", r.pvalue);
        assert!(r.rho_hat.is_none(), "split_exact has no rho_hat");
    }

    // Test 6: rho_hat relocation (Rust side). split_nb, unweighted, n_te >= 4
    // ⇒ Some in [0, 1]. Weighted or n_te = 3 ⇒ None. split_exact and every
    // other method ⇒ None (checked above and in the other methods' tests).
    #[test]
    fn split_nb_rho_hat_populated_when_ruler_valid() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let r = pls1_confirmatory_test(
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
                seed: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let rho = r
            .rho_hat
            .expect("split_nb must populate rho_hat when unweighted, n_test>=4");
        assert!((0.0..=1.0).contains(&rho));
    }

    #[test]
    fn split_nb_rho_hat_none_when_weighted() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 1.5 } else { 0.5 });
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: Some(w.as_ref()),
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb {
                    n_splits: 30,
                    force: false,
                },
                seed: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.rho_hat.is_none());
    }

    #[test]
    fn split_nb_rho_hat_none_when_n_test_below_four() {
        // split_sizes(10, 5) -> (n_train=7, n_test=3): below the n_test >= 4
        // floor for the rho_hat ruler (same floor split_j_eff used to check).
        let (x, y) = synth_no_signal(10, 5, 9);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 5,
                weights: None,
            },
            ConfirmatoryTestOpts {
                // `force` — this test is about the NB ruler's n_test floor, so
                // NB has to actually run; n = 10 trips the auto-gate's n floor
                // and would otherwise reroute to split_exact, which reports
                // `rho_hat: None` for its own reasons and would pass vacuously.
                args: ConfirmatoryArgs::SplitNb {
                    n_splits: 10,
                    force: true,
                },
                seed: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.rho_hat.is_none());
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
        let (x, y) = synth_no_signal(20, 10, 5);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(r.method, "split_exact");
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
        let (x, y) = synth_no_signal(60, 4, 11);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(r.method, "split_exact");
        let sr = r.stable_rank.expect("gate was evaluated");
        assert!(
            sr >= SPLIT_NB_GATE_MIN_STABLE_RANK,
            "column count, not the spectrum, must be what fired here (stable_rank={sr})"
        );
    }

    /// `force` overrides the column precheck exactly as it overrides the other
    /// two clauses — the precheck is one more OR term, not a hard block.
    #[test]
    fn gate_force_runs_nb_on_narrow_x() {
        let (x, y) = synth_no_signal(60, 4, 11);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, true);
        assert_eq!(r.method, "split_nb");
    }

    #[test]
    fn gate_reroutes_when_stable_rank_below_floor() {
        let (x, y) = synth_one_factor(40, 5, 6);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(r.method, "split_exact");
        let sr = r.stable_rank.expect("gate was evaluated");
        assert!(sr < SPLIT_NB_GATE_MIN_STABLE_RANK, "stable_rank={sr}");
    }

    #[test]
    fn gate_passes_flat_spectrum_at_adequate_n() {
        // d = 10 rather than the file's usual 5: at n = 60, d = 5 the stable
        // rank lands around 3.1, so the negative case would clear the floor by
        // a few percent and turn into a coin flip under any reseeding. Ten iid
        // columns put it near 7.
        let (x, y) = synth_no_signal(60, 10, 7);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(r.method, "split_nb");
        // No rank assertion — `method == "split_nb"` at n = 60 already implies
        // the rank cleared the floor. What is worth pinning is that the
        // diagnostic is populated on the pass path too.
        assert!(r.stable_rank.is_some());
        // Reported counts still come from the (unrewritten) split_nb args.
        assert_eq!(r.n_perm, None);
        assert_eq!(r.n_splits, Some(20));
    }

    #[test]
    fn gate_force_runs_nb_on_flagged_design_and_still_reports_rank() {
        let (x, y) = synth_one_factor(40, 5, 6);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 20, true);
        assert_eq!(r.method, "split_nb");
        let sr = r.stable_rank.expect(
            "rank is computed even under force — it is how a caller sees what the gate saw",
        );
        assert!(sr < SPLIT_NB_GATE_MIN_STABLE_RANK, "stable_rank={sr}");
    }

    #[test]
    fn gate_uses_n_eff_under_weights() {
        // Raw n = 40 clears the floor; the weights pull Kish n_eff under it.
        // Σw = 22, Σw² = 20.2 ⇒ n_eff = 22²/20.2 ≈ 23.96 < 25.
        let (x, y) = synth_no_signal(40, 5, 8);
        let w = Col::<f64>::from_fn(40, |i| if i % 2 == 0 { 1.0 } else { 0.1 });

        let unweighted = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
        assert_eq!(
            unweighted.method, "split_nb",
            "the same X at raw n = 40 must clear the gate, so the weighted \
             case below isolates n_eff"
        );

        let r = gate_run(x.as_ref(), y.as_ref(), Some(w.as_ref()), 20, false);
        assert_eq!(r.method, "split_exact");
        let sr = r.stable_rank.expect("gate was evaluated");
        assert!(
            sr >= SPLIT_NB_GATE_MIN_STABLE_RANK,
            "n_eff, not the spectrum, must be what fired here (stable_rank={sr})"
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
        assert_eq!(unweighted.method, "split_nb");

        let weighted = gate_run(x.as_ref(), y.as_ref(), Some(w.as_ref()), 20, false);
        assert_eq!(weighted.method, "split_exact");
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

    #[test]
    fn gate_result_counts_read_off_resolved_args() {
        let (x, y) = synth_one_factor(40, 5, 6);
        let r = gate_run(x.as_ref(), y.as_ref(), None, 12, false);
        assert_eq!(r.method, "split_exact");
        // n_perm is split_exact's own default; n_splits is what the caller asked for.
        assert_eq!(r.n_perm, Some(1000));
        assert_eq!(r.n_splits, Some(12));
    }

    #[test]
    fn gate_not_evaluated_for_other_methods() {
        let (x, y) = synth_no_signal(20, 4, 5);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 100,
                    n_splits: 10,
                },
                seed: Some(4242),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.stable_rank.is_none());
    }

    // ── the public gate query ────────────────────────────────────────────────
    //
    // These are the whole contract of `split_nb_gate`: it must answer exactly
    // what the embedded gate decides, and it must reject bad input with the
    // same errors the test entry points do rather than reaching the SVD.

    /// One flagged design and one clean one, checked against what
    /// `pls1_confirmatory_test` actually did with the same X.
    #[test]
    #[allow(clippy::float_cmp)] // same rule on the same standardized X — bit-exact or it's a bug
    fn public_gate_answers_what_the_embedded_gate_decided() {
        for (x, y, expect_fires) in [
            (
                synth_one_factor(40, 5, 6).0,
                synth_one_factor(40, 5, 6).1,
                true,
            ),
            (
                synth_no_signal(60, 10, 7).0,
                synth_no_signal(60, 10, 7).1,
                false,
            ),
        ] {
            let embedded = gate_run(x.as_ref(), y.as_ref(), None, 20, false);
            let q = split_nb_gate(x.as_ref(), None).unwrap();
            assert_eq!(q.fires, expect_fires);
            assert_eq!(q.fires, embedded.method == "split_exact");
            assert_eq!(q.stable_rank, embedded.stable_rank.unwrap());
            assert_eq!(q.n_eff, embedded.n_eff);
        }
    }

    /// The weighted path: both the `n_eff` clause and the weighted-moment
    /// standardization behind the rank clause have to travel with it.
    #[test]
    #[allow(clippy::float_cmp)]
    fn public_gate_answers_under_weights() {
        let (x, y) = synth_no_signal(40, 5, 8);
        let w = Col::<f64>::from_fn(40, |i| if i % 2 == 0 { 1.0 } else { 0.1 });
        let embedded = gate_run(x.as_ref(), y.as_ref(), Some(w.as_ref()), 20, false);
        let q = split_nb_gate(x.as_ref(), Some(w.as_ref())).unwrap();
        assert!(q.fires);
        assert_eq!(q.fires, embedded.method == "split_exact");
        assert_eq!(q.stable_rank, embedded.stable_rank.unwrap());
        assert_eq!(q.n_eff, embedded.n_eff);
        assert!(q.n_eff < 40.0, "weights must reach n_eff: {}", q.n_eff);
    }

    /// The reason the function repeats entry-point validation: without it a
    /// NaN reaches `linalg::stable_rank`'s SVD instead of this error.
    #[test]
    fn public_gate_rejects_non_finite_x() {
        let mut x = synth_no_signal(60, 10, 7).0;
        x[(3, 2)] = f64::NAN;
        assert!(matches!(
            split_nb_gate(x.as_ref(), None),
            Err(PlsKitError::NonFiniteInput)
        ));
    }

    #[test]
    fn public_gate_rejects_bad_weights() {
        let x = synth_no_signal(60, 10, 7).0;
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

    // ── Score test ───────────────────────────────────────────────────────────

    #[test]
    fn score_returns_bounded_p() {
        let (x, y) = synth_no_signal(50, 6, 31);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::Score,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "score");
        assert!(r.pvalue >= 0.0 && r.pvalue <= 1.0);
    }

    // ── E-value test ─────────────────────────────────────────────────────────

    #[test]
    fn e_returns_bounded_p() {
        let (x, y) = synth_with_signal(80, 5, 3.0, 41);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::E,
                seed: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "e");
        assert!(r.pvalue >= 0.0 && r.pvalue <= 1.0);
        // Universal inference always satisfies P(reject | H0) ≤ α exactly,
        // so under signal we expect p < 0.5 typically.
        assert!(r.pvalue < 0.5, "p={}", r.pvalue);
    }

    #[test]
    fn ci_branch_populates_ci_field_when_requested() {
        let (x, y) = synth_with_signal(80, 5, 4.0, 99);
        let r = pls1_confirmatory_test(
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
                    m_rate: 0.7,
                    level: 0.95,
                    max_failure_rate: 0.01,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let ci = r.ci.expect("ci should be populated");
        assert_eq!(ci.n_boot, 200);
        assert_eq!(ci.beta_sign_z.len(), 5);
    }

    #[test]
    fn ci_none_keeps_ci_field_none() {
        let (x, y) = synth_with_signal(80, 5, 4.0, 99);
        let r = pls1_confirmatory_test(
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
                ci: None,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.ci.is_none());
    }

    #[test]
    fn confirmatory_keep_dense_endpoint_bit_parity() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let mk = |keep: Option<usize>| {
            pls1_confirmatory_test(
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
                    seed: Some(2),
                    keep,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let dense = mk(None);
        let endpoint = mk(Some(5));
        assert_eq!(dense.pvalue.to_bits(), endpoint.pvalue.to_bits());
        assert_eq!(dense.statistic.to_bits(), endpoint.statistic.to_bits());
    }

    #[test]
    fn confirmatory_sparse_split_nb_runs() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let r = pls1_confirmatory_test(
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
                seed: Some(2),
                keep: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert!((0.0..=1.0).contains(&r.pvalue));
    }

    #[test]
    fn confirmatory_rejects_keep_with_ci() {
        let (x, y) = synth_with_signal(80, 5, 4.0, 99);
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
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
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
        let (x, y) = synth_with_signal(80, 5, 4.0, 11);
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
}
