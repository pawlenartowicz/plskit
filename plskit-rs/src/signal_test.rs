//! Confirmatory PLS1 omnibus test at one k or over a set of k: five methods.
//!
//! A set of k is combined by min-p (Westfall & Young 1993, *Resampling-Based
//! Multiple Testing*, Wiley).

use std::collections::BTreeMap;

use faer::{Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::Pls1Model;
use crate::held_out_gram::HeldOutGram;
// The split-half correlation's degeneracy guard is the same rule
// standardization uses to classify a column as constant.
use crate::linalg::{ScaledMoments, scaled_moments};

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
    /// y), once per call, before any randomness is drawn. A request with
    /// covariates, one over a set of more than one k, and a PLS3 request
    /// always run `split_exact`. Otherwise
    /// `split_exact` runs when the `split_nb` auto-gate fires, when `n_eff`
    /// ≤ 1000 at k = 1 without `keep` or `n_eff` < 250 otherwise, or when X
    /// has more than 100 columns per row (p > 100·n) at k = 1 without
    /// `keep`; `split_nb` runs otherwise. The thresholds may change between
    /// versions. A result never reports
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

/// Input to `pls1_confirmatory_test`: raw `(X, y, k)`, the Gram matrix of X
/// with `y` and `k`, or a pre-fitted model.
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
        /// The numbers of components to test, in any order. `None` is the
        /// default set, `1..=DEFAULT_K_SET_MAX` cut by the design (see
        /// [`pls1_confirmatory_test`]).
        k: Option<&'a [usize]>,
        /// Optional per-observation weights. `None` means uniform weights.
        /// Validated (finite, non-negative, Σw > 0) and normalized to mean 1 before use.
        weights: Option<ColRef<'a, f64>>,
        /// Optional covariates `Z`, `(n_samples, p_z)`. X is standardized,
        /// then X and y are replaced by their (weighted) residuals on
        /// `[1, Z]`, and the test runs on those residuals as given (the
        /// `pre_standardized` semantics). Permutation null columns are
        /// Freedman–Lane: null column c is `M_Z·π_c(ỹ)`, the residual of the
        /// permuted residual `ỹ`, which keeps the test approximately exact,
        /// not exact. Accepted by `split_exact`, `raw_perm` and `auto` (which
        /// then runs `split_exact`).
        covariates: Option<MatRef<'a, f64>>,
        /// Optional caller-built null outcomes, `(n_samples, B)`. Column c
        /// replaces permutation null column c + 1 of `split_exact`: the
        /// statistic is computed on each column with the same splits as on
        /// `y`, and `p = (#{null ≥ obs} + 1) / (B + 1)`. The columns must be
        /// on the scale of `y`. `n_perm` in `opts.args` is ignored and the
        /// output's `n_perm` is `B`. Works with `weights`.
        ///
        /// # Errors
        /// `InvalidArgument` for every `test_method` but `split_exact`
        /// (`auto` is refused too), when `covariates` is set, or when `B = 0`;
        /// `ShapeMismatch` when the rows differ from `n_samples`;
        /// `NonFiniteInput` for NaN/inf.
        null_y: Option<MatRef<'a, f64>>,
    },
    /// The Gram matrix of X in place of X, unweighted. Every method runs
    /// from it alone: the call centres `gram.gram` over its rows (so a row
    /// subset from `GramMatrixOutput::subset` is accepted) and checks its
    /// entries, and `pre_standardized` is read from `gram`. See
    /// [`pls1_confirmatory_test`] for what this form refuses.
    Gram {
        /// The Gram matrix, as [`crate::gram_matrix`] returns it or built
        /// by the caller with the same three fields.
        gram: &'a crate::held_out_gram::GramMatrixOutput,
        /// Target vector `(n_samples,)`, one entry per row of `gram`.
        y: ColRef<'a, f64>,
        /// The numbers of components to test, as in the Raw form.
        k: Option<&'a [usize]>,
        /// Optional covariates `Z`, `(n_samples, p_z)`, one row per row of
        /// `gram` (a `GramMatrixOutput::subset` takes `Z[rows]`). `gram.gram`
        /// is replaced by `M_Z·G·M_Z`, the Gram matrix of X with `[1, Z]`
        /// removed, and y by its residual; the permutation null columns are
        /// Freedman–Lane as in the Raw form. Accepted by `split_exact`,
        /// `raw_perm` and `auto`.
        covariates: Option<MatRef<'a, f64>>,
        /// Optional caller-built null outcomes, `(n_samples, B)`, as in the
        /// Raw form: `split_exact` only, refused with `covariates`, `n_perm`
        /// ignored, the output's `n_perm` is `B`.
        ///
        /// # Errors
        /// `InvalidArgument` for every `test_method` but `split_exact`
        /// (`auto` is refused too), when `covariates` is set, or when `B = 0`;
        /// `ShapeMismatch` when the rows differ from the rows of `gram`;
        /// `NonFiniteInput` for NaN/inf.
        null_y: Option<MatRef<'a, f64>>,
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
                n_splits: 32,
                force: false,
            },
            ConfirmatoryMethod::SplitExact => ConfirmatoryArgs::SplitExact {
                n_perm: 1000,
                n_splits: 32,
            },
            ConfirmatoryMethod::Score => ConfirmatoryArgs::Score,
            ConfirmatoryMethod::E => ConfirmatoryArgs::E,
            ConfirmatoryMethod::Auto => ConfirmatoryArgs::Auto {
                n_perm: 1000,
                n_splits: 32,
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

/// Spectrum floor of the `split_nb` auto-gate, on `linalg::stable_rank` of X
/// as the gate reads it (standardized once over all rows, or only centred
/// under `pre_standardized`). See [`SPLIT_NB_GATE_MIN_N_EFF`] for the shared
/// provenance.
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

/// Sample-size floor of `test_method = "auto"` on the no-refit route: at or
/// below this Kish `n_eff` it runs `split_exact`, which there is cheap and
/// has the larger power. The value is also stated in
/// `ConfirmatoryMethod::Auto`'s doc comment and in every doc, doc comment
/// and docstring that states the thresholds (search for `100·n`, `100*n`,
/// `250`, `1000`, `AUTO_MIN_N_EFF_NO_REFIT`, `AUTO_MIN_N_EFF_REFIT`);
/// change together.
const AUTO_MIN_N_EFF_NO_REFIT: f64 = 1000.0;

/// Sample-size floor of `test_method = "auto"` on the refit route: below this
/// Kish `n_eff` it runs `split_exact`, because `split_nb` gives up power on
/// small samples. Stated in the same places as [`AUTO_MIN_N_EFF_NO_REFIT`];
/// change together.
const AUTO_MIN_N_EFF_REFIT: f64 = 250.0;

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
#[derive(Debug, Clone)]
pub(crate) struct SplitNbResolution {
    /// The design is flagged (whatever `force` says).
    pub(crate) fires: bool,
    /// `fires && !force`: the caller must run `split_exact` with
    /// [`SPLIT_NB_REROUTE_N_PERM`] permutations instead of `split_nb`.
    pub(crate) reroute: bool,
    /// Stable rank of X as the gate read it (standardized once over all
    /// rows, or only centred under `pre_standardized`), reported whether or
    /// not the gate fired (and under `force` too).
    pub(crate) stable_rank: f64,
    /// On wide X, the Gram matrix the stable rank comes from
    /// (`HeldOutGram::from_x` on the gate's X, weights and scale), which
    /// the test that runs next reads instead of building its own. `None`
    /// on tall X.
    pub(crate) gram: Option<HeldOutGram>,
}

impl SplitNbResolution {
    /// The method that runs on a `split_nb` request.
    pub(crate) fn method(&self) -> ConfirmatoryMethod {
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
/// [`resolve_auto_on_x`], which every `test_method = "auto"` site calls.
///
/// Inputs are the validated weights, exactly as
/// `fit::validate_and_normalize_weights` returns them: `w_norm` the
/// normalized weights (`None` when absent or all-equal) and `n_eff` its Kish
/// effective sample size (computed from the weights as the caller handed them,
/// and exactly the row count when weights are absent or all-equal). That one `n_eff` is the
/// number every result type reports, so the gate's size test and the
/// reported `n_eff` can never disagree by a rounding step.
///
/// Reads X as the held-out tests do: centred over all rows (weighted means
/// under `w_norm`), divided by `scale`, the column scale the caller
/// computed once (`linalg::held_out_scale`: the fit's scale, or ones under
/// `pre_standardized`, where X is only centred), and its rows scaled by
/// `√w_norm`, so a row of weight 0 does not enter the rank. Block by block
/// on wide `x`, into its row-side Gram matrix (`HeldOutGram::from_x`),
/// which the resolution keeps without the row scaling; into a copy on tall
/// `x`.
pub(crate) fn resolve_split_nb(
    x: MatRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    n_eff: f64,
    force: bool,
    scale: ColRef<'_, f64>,
) -> SplitNbResolution {
    let (n, p) = (x.nrows(), x.ncols());
    // Wide X: the Gram matrix on the row side, summed over column blocks of
    // the scaled X, so no copy of X is written.
    let gram = (n <= p).then(|| HeldOutGram::from_x(x, w_norm, scale));
    // `sr` is computed unconditionally even when the column precheck already
    // decided: it is reported as the gate's `stable_rank` diagnostic either way.
    let sr = if let Some(g) = &gram {
        w_norm.map_or_else(|| g.stable_rank(), |w| g.with_row_weights(w).stable_rank())
    } else {
        // Tall X: the Gram matrix is on the column side and does not sum
        // over column blocks.
        let rows: Vec<usize> = (0..n).collect();
        let sw = w_norm.map(crate::linalg::sqrt_col);
        let (xs, _, _) =
            crate::linalg::center_rows_fro(x, &rows, w_norm, sw.as_ref().map(Col::as_ref), scale);
        crate::linalg::stable_rank(xs.as_ref())
    };
    let fires = split_nb_gate_fires(p, n_eff, sr);
    SplitNbResolution {
        fires,
        reroute: fires && !force,
        stable_rank: sr,
        gram,
    }
}

/// The `split_nb` gate's rule on a design of `p` columns, Kish `n_eff` and
/// stable rank `sr`: at most [`SPLIT_NB_GATE_MAX_COLS_PRECHECK`] columns,
/// `n_eff` below [`SPLIT_NB_GATE_MIN_N_EFF`], or `sr` below
/// [`SPLIT_NB_GATE_MIN_STABLE_RANK`]. A NaN stable rank (a non-finite Gram
/// matrix) fires it: the spectrum the rule reads is unknown.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(sr >= floor)` fires on a NaN rank
pub(crate) fn split_nb_gate_fires(p: usize, n_eff: f64, sr: f64) -> bool {
    p <= SPLIT_NB_GATE_MAX_COLS_PRECHECK
        || n_eff < SPLIT_NB_GATE_MIN_N_EFF
        || !(sr >= SPLIT_NB_GATE_MIN_STABLE_RANK)
}

/// What the rules of `test_method = "auto"` read on one request. X is not
/// here: only the rule that needs its spectrum calls `stable_rank`, so a
/// request an earlier rule decides never pays for the pass over X and its
/// Gram matrix, which matters most on the widest designs.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct AutoInput<'a> {
    /// Rows of X.
    pub(crate) n: usize,
    /// Columns of X.
    pub(crate) p: usize,
    /// Kish effective sample size of the weights (the row count when there
    /// are none), as `resolve_split_nb` takes it.
    pub(crate) n_eff: f64,
    /// `split_exact` would take its no-refit route on this request
    /// ([`split_exact_no_refit_route`]; always `false` for PLS3, which has
    /// none).
    pub(crate) no_refit: bool,
    /// The request is a `pls3_confirmatory_test` one.
    pub(crate) pls3: bool,
    /// The request has covariates.
    pub(crate) covariates: bool,
    /// The request tests a set of more than one k.
    pub(crate) multi_k: bool,
    /// Stable rank of X as the held-out tests read it: centred over all
    /// rows and divided by the call's column scale.
    pub(crate) stable_rank: &'a dyn Fn() -> f64,
}

/// One rule of `test_method = "auto"`: its name and its decision, `None`
/// when it does not apply. A decision is the method and the stable rank the
/// rule read, if it read one.
pub(crate) type AutoRule = (
    &'static str,
    fn(&AutoInput<'_>) -> Option<(ConfirmatoryMethod, Option<f64>)>,
);

/// `split_exact` without a stable rank: what every rule but the last decides.
const AUTO_EXACT: Option<(ConfirmatoryMethod, Option<f64>)> =
    Some((ConfirmatoryMethod::SplitExact, None));

/// The rules of `test_method = "auto"`, in order; the first that answers
/// decides.
///
/// - `covariates`: the request has covariates, so `split_exact`, the method
///   with Freedman–Lane null columns.
/// - `k_set`: the request tests more than one k, so `split_exact`, the
///   method that combines a set by min-p.
/// - `narrow_x`: X has at most [`SPLIT_NB_GATE_MAX_COLS_PRECHECK`] columns
///   (the gate's column precheck), so `split_exact`.
/// - `small_n_eff`: on the no-refit route `n_eff ≤ AUTO_MIN_N_EFF_NO_REFIT`,
///   otherwise `n_eff < AUTO_MIN_N_EFF_REFIT`; both contain the gate's own
///   `n_eff` floor, so `split_exact` (power).
/// - `too_wide_for_gate`: on the no-refit route only,
///   `p > AUTO_MAX_P_PER_N · n`, so `split_exact` (cost).
/// - `pls3_exact`: every PLS3 request runs `split_exact`.
/// - `split_nb_gate`: always answers. The only rule that reads the stable
///   rank: `split_exact` when the gate fires, `split_nb` otherwise.
pub(crate) const AUTO_RULES: &[AutoRule] = &[
    (
        "covariates",
        |a| if a.covariates { AUTO_EXACT } else { None },
    ),
    ("k_set", |a| if a.multi_k { AUTO_EXACT } else { None }),
    ("narrow_x", |a| {
        if a.p <= SPLIT_NB_GATE_MAX_COLS_PRECHECK {
            AUTO_EXACT
        } else {
            None
        }
    }),
    ("small_n_eff", |a| {
        let small = if a.no_refit {
            a.n_eff <= AUTO_MIN_N_EFF_NO_REFIT
        } else {
            a.n_eff < AUTO_MIN_N_EFF_REFIT
        };
        if small { AUTO_EXACT } else { None }
    }),
    ("too_wide_for_gate", |a| {
        if a.no_refit && a.p > AUTO_MAX_P_PER_N * a.n {
            AUTO_EXACT
        } else {
            None
        }
    }),
    ("pls3_exact", |a| if a.pls3 { AUTO_EXACT } else { None }),
    ("split_nb_gate", |a| {
        let sr = (a.stable_rank)();
        let method = if split_nb_gate_fires(a.p, a.n_eff, sr) {
            ConfirmatoryMethod::SplitExact
        } else {
            ConfirmatoryMethod::SplitNb
        };
        Some((method, Some(sr)))
    }),
];

/// Resolve `test_method = "auto"` to `split_exact` or `split_nb`: the
/// decision of the first rule in [`AUTO_RULES`] that answers, with its
/// name. The stable rank is `Some` only when that rule read it. Draws no
/// randomness.
pub(crate) fn resolve_auto(
    input: &AutoInput<'_>,
) -> (ConfirmatoryMethod, Option<f64>, &'static str) {
    AUTO_RULES
        .iter()
        .find_map(|&(name, rule)| rule(input).map(|(method, sr)| (method, sr, name)))
        .expect("the last rule always answers")
}

/// [`resolve_auto`] on X: the stable rank comes from the `split_nb` gate on
/// X (`resolve_split_nb` on these inputs), run only if the rule that reads
/// it is reached. Returns the method, that stable rank and the gate's Gram
/// matrix (`SplitNbResolution::gram`: wide X, and only when the gate ran),
/// which the k = 1 test that runs next reads instead of building its own.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub(crate) fn resolve_auto_on_x(
    x: MatRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    n_eff: f64,
    no_refit: bool,
    pls3: bool,
    covariates: bool,
    multi_k: bool,
    scale: ColRef<'_, f64>,
) -> (ConfirmatoryMethod, Option<f64>, Option<HeldOutGram>) {
    // `force` only decides the reroute flag, which is not read here.
    let gate = std::cell::OnceCell::new();
    let stable_rank = || {
        gate.get_or_init(|| resolve_split_nb(x, w_norm, n_eff, false, scale))
            .stable_rank
    };
    let (method, sr, _) = resolve_auto(&AutoInput {
        n: x.nrows(),
        p: x.ncols(),
        n_eff,
        no_refit,
        pls3,
        covariates,
        multi_k,
        stable_rank: &stable_rank,
    });
    (method, sr, gate.into_inner().and_then(|g| g.gram))
}

/// What the `split_nb` auto-gate sees on a design. Returned by
/// [`split_nb_gate`].
#[derive(Debug, Clone, Copy)]
pub struct SplitNbGateOutput {
    /// `true` when the design is flagged: a `split_nb` request on it reroutes
    /// to `split_exact` unless the caller forces it.
    pub fires: bool,
    /// Stable rank of X as the tests read it: standardized, or only centred
    /// under `pre_standardized`, with rows scaled by `√w` under weights (a
    /// row of weight 0 does not count). NaN when its Gram matrix is not
    /// finite (a product that overflows); the gate then fires.
    pub stable_rank: f64,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
}

/// Ask whether the `split_nb` auto-gate flags a design, without running a test.
///
/// Reports the decision `pls1_confirmatory_test` makes on a call with the
/// same `pre_standardized`: X standardized once over all rows (weighted
/// moments when `weights` is given), or with the flag used as given and
/// only centred. This evaluates the one rule, it does not restate it.
///
/// # Errors
/// - `PlsKitError::NonFiniteInput` when X or weights contain NaN/inf
/// - `PlsKitError::InvalidWeights` for length-mismatched, negative, or all-zero weights
///
/// # Panics
/// Only if the eigendecomposition of the Gram matrix of the standardized
/// `x` fails to converge (`linalg::stable_rank_from_gram`, the only
/// fallible step).
pub fn split_nb_gate(
    x: MatRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> PlsKitResult<SplitNbGateOutput> {
    crate::fit::with_thread_limit(|| split_nb_gate_impl(x, weights, pre_standardized))
}

/// Body of [`split_nb_gate`], on the caller's pool.
pub(crate) fn split_nb_gate_impl(
    x: MatRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> PlsKitResult<SplitNbGateOutput> {
    // Entry-point validation is repeated here rather than inherited: this
    // function runs ahead of any dispatch, and a non-finite X would reach
    // the gate as a NaN stable rank (`linalg::stable_rank_from_gram`),
    // which fires it, and the caller would lose the clean `NonFiniteInput`.
    crate::fit::check_finite_mat(x)?;
    // k_requested is only used for a check this function doesn't make (there
    // is no component count in play), so any value passes; 0 is the honest one.
    let (w_norm, n_eff) = crate::fit::validate_and_normalize_weights(weights, x.nrows(), 0)?;
    // `force` only decides whether a flagged design is rerouted; this query
    // reports `fires` alone, so its value is irrelevant here.
    let w_norm = w_norm.as_ref().map(Col::as_ref);
    let scale = crate::linalg::held_out_scale(x, w_norm, pre_standardized);
    let gate = resolve_split_nb(x, w_norm, n_eff, false, scale.as_ref());
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
///
/// The CI is beta, not to be trusted yet. It scales X and y once over all
/// rows, as the held-out tests do (not under `pre_standardized`). Its
/// `holdout_corr` half split centres the training half with its own means
/// and the held-out half with the training means, as the held-out tests do.
/// The β and leverage resamples are centred with their own means too, except
/// under `pre_standardized`, where they use X and y as given, as the
/// reference fit does.
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
    /// Caller asserts X and y are already standardized: skips the one
    /// standardization of X over all rows. `split_exact`, `split_nb` (and
    /// its gate) and `e` then use X as given and only centre each part with
    /// its own means. `raw_perm` likewise uses X as given and only centres
    /// each CV fold's training rows with their own means.
    pub pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
    /// Optional CI bundle. When `Some`, runs an independent subsampling pass
    /// after the test and populates `ConfirmatoryTestOutput.ci`. Beta, see
    /// [`CIOpts`].
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
    /// p-value (or `min(1, 1/e)` for the `e` method). Over a set of more
    /// than one k, the min-p combined p for the whole set.
    pub pvalue: f64,
    /// Observed test statistic (CV R² for `raw_perm`; mean Fisher-z back-transformed
    /// (`tanh(z̄)`) for `split_exact` and `split_nb`, one statistic across both,
    /// on the same splits at one seed and `n_splits`; `||X'y||²` for `score`;
    /// log-e for `e`). `None` over a set of more than one k.
    pub statistic: Option<f64>,
    /// Method name as a lowercase string (e.g. `"raw_perm"`, `"split_nb"`, `"e"`).
    pub test_method: String,
    /// The tested numbers of components, sorted ascending.
    pub k: Vec<usize>,
    /// Each tested k's own p (unadjusted).
    pub pvalues_by_k: BTreeMap<usize, f64>,
    /// Each tested k's own statistic.
    pub statistics_by_k: BTreeMap<usize, f64>,
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
    /// Stable rank `‖X_std‖²_F / ‖X_std‖²₂` of X standardized once over all
    /// rows (only centred under `pre_standardized`), as seen by the
    /// `split_nb` auto-gate. `Some` on an explicit `split_nb` request, and
    /// on a PLS1 `"auto"` request that reached the stable-rank check (one
    /// k, no covariates, p > 4, `n_eff` above 1000 at k = 1 without `keep` (at
    /// least 250 otherwise), and, at k = 1 without `keep`, p ≤ 100·n); `None`
    /// otherwise, and on every PLS3 `"auto"` request. On a `split_nb` request it
    /// is set whether the gate fired or not, and also under `force` (the
    /// point of the field is to show what the gate saw). NaN when the Gram
    /// matrix of X so read is not finite; the gate then fires.
    pub stable_rank: Option<f64>,
    /// `rank([1, Z]) − 1`, the covariates kept after aliased columns are
    /// dropped. `None` without covariates.
    pub n_covariates: Option<usize>,
}

/// Confirmatory PLS1 omnibus test at one k or over a set of k.
///
/// A set of more than one k answers "X predicts y at some k in the set":
/// `split_exact` and `raw_perm` draw their design once for the largest k,
/// compute every k's statistic on the observed and on each null column
/// from one fit per (split or fold, column), and combine the per-k
/// p-values by min-p, which keeps the permutation p exact. The set is fixed
/// before the test sees y; the result names no k to fit at.
///
/// The default set (`k = None`) is `1..=DEFAULT_K_SET_MAX` cut at
/// `min(DEFAULT_K_SET_MAX, p, ⌊n/2⌋ − 2)`, then below the weights' Kish
/// `n_eff` (`n_eff ≥ k + 1`) and the covariates' residual df
/// (`n_pos − rank([1, Z]) ≥ k + 1`), never below 1. It is `{1}` for
/// `split_nb`, `score` and `e`, with `ci` and with `keep`, which refuse a
/// set of more than one k. `auto` with more than one k runs `split_exact`.
///
/// The `Gram` input form runs every method from the Gram matrix alone. On
/// `gram_matrix(x, flag)` it gives the Raw form's result on `x` with
/// `pre_standardized = flag` and the same seed, to rounding; `score` there
/// reads the centred G, so it equals the Raw form on X centred over all
/// rows.
///
/// # Shapes
/// - `input` (Raw form): `x: (n_samples, n_features)`, `y: (n_samples,)`, every k in
///   `1..=k_max`, `covariates: (n_samples, p_z)`
/// - `input` (Gram form): `gram.gram: (n_samples, n_samples)`, `y: (n_samples,)`,
///   every k in `1..=min(gram.n_features, 10)`
///
/// # Errors
/// Every limit on k below applies to the largest k of the set.
/// - `PlsKitError::Internal` for the `Model` input form (wrappers must reassemble raw X+y)
/// - `PlsKitError::DimensionMismatch` when row counts disagree
/// - `PlsKitError::KExceedsMax` when k > `d` (`gram.n_features` in the Gram form)
/// - `PlsKitError::InvalidArgument` when a k is 0 (every method, `score`
///   included although its statistic does not depend on k), when the set
///   is empty or has a duplicate, or when it has more than one k with
///   `split_nb`, `score`, `e`, `ci` or `keep`
/// - `PlsKitError::InvalidArgument` for `test_method = "raw_perm"` with
///   `n_folds < 2` or `n_folds >= n` (leave-one-out; the pooled CV R² is
///   undefined when every validation fold is a single row)
/// - `PlsKitError::NonFiniteInput` when X, y, or weights contain NaN/inf
/// - `PlsKitError::InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
/// - Raw form with covariates: `PlsKitError::InvalidArgument` for
///   `test_method` `split_nb`, `score` or `e`; `ShapeMismatch` when
///   `covariates` does not have `n_samples` rows; `NonFiniteInput` when it
///   holds NaN/inf; `InsufficientResidualDf` when
///   `n_samples − rank([1, Z]) < k + 1`
/// - Gram form with covariates: `PlsKitError::InvalidArgument` for
///   `test_method` `split_nb`, `score` or `e`; `ShapeMismatch` when
///   `covariates` does not have one row per row of `gram.gram`;
///   `NonFiniteInput` when it holds NaN/inf; `InsufficientResidualDf` when
///   `n_samples − rank([1, Z]) < k + 1`
/// - With `null_y`: `PlsKitError::InvalidArgument` for any `test_method`
///   but `split_exact` (`auto` is refused too), with `covariates`, or with zero
///   columns; `ShapeMismatch` when its rows differ from those of y;
///   `NonFiniteInput` for NaN/inf. `n_perm` is then the column count.
/// - Gram form: `PlsKitError::InvalidArgument` when `gram.gram` has
///   another row count than `y`, `k > 10` (a fit with more components
///   needs X), `opts.keep` or `opts.ci` is set, or `opts.pre_standardized`
///   is set (it is read from `gram`); `ShapeMismatch` when `gram.gram` is
///   not square; `NonFiniteInput` when it is not finite; `InvalidInput`
///   when the centred matrix has a negative diagonal entry, is not
///   symmetric or breaks Cauchy-Schwarz, beyond rounding
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
pub fn pls1_confirmatory_test(
    input: ConfirmatoryTestInput<'_>,
    opts: ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    crate::fit::with_thread_limit(|| confirmatory_test_impl(input, opts, GateMode::Owned))
}

/// The largest k of the default set of `pls1_confirmatory_test`, before
/// the design cuts it.
pub const DEFAULT_K_SET_MAX: usize = 8;

/// The largest k up to `cap` that a single call allows on `p` columns,
/// Kish `n_eff` (`n_eff ≥ k + 1`, `validate_weights_for_k`) and covariate
/// residual df (`residual_df ≥ k + 1`, `CovariateBasis::check_df`); at
/// least 1.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn k_limit(
    cap: usize,
    p: usize,
    n_eff: Option<f64>,
    residual_df: Option<usize>,
) -> usize {
    let mut k = cap.min(p);
    if let Some(ne) = n_eff {
        k = k.min((ne.floor() as usize).saturating_sub(1));
    }
    if let Some(df) = residual_df {
        k = k.min(df.saturating_sub(1));
    }
    k.max(1)
}

/// The sorted set of k a call tests. `None` is the default set:
/// `1..=default_cut`, or `{1}` where a set of more than one k is refused
/// (`split_nb`, `score`, `e`, `ci`, `keep`). An explicit set is sorted and
/// otherwise taken as given; empty, duplicates, or more than one k where
/// refused is an error. Every other limit is checked at the largest k by
/// the caller.
fn resolve_k_set(
    k: Option<&[usize]>,
    method: ConfirmatoryMethod,
    ci: bool,
    keep: Option<usize>,
    default_cut: usize,
) -> PlsKitResult<Vec<usize>> {
    let single_only = match method {
        ConfirmatoryMethod::SplitNb | ConfirmatoryMethod::Score | ConfirmatoryMethod::E => {
            Some(method.as_str())
        }
        _ => None,
    }
    .or(ci.then_some("ci"))
    .or(keep.map(|_| "keep"));
    let Some(k) = k else {
        return Ok(if single_only.is_some() {
            vec![1]
        } else {
            (1..=default_cut).collect()
        });
    };
    if k.is_empty() {
        return Err(PlsKitError::InvalidArgument("k is empty".into()));
    }
    let mut ks = k.to_vec();
    ks.sort_unstable();
    if ks.windows(2).any(|w| w[0] == w[1]) {
        return Err(PlsKitError::InvalidArgument("k contains duplicates".into()));
    }
    if let (true, Some(what)) = (ks.len() > 1, single_only) {
        return Err(PlsKitError::InvalidArgument(format!(
            "a set of more than one k does not combine with {what}"
        )));
    }
    Ok(ks)
}

/// The default set's cut on `n` rows and `p` columns: [`k_limit`] at
/// [`DEFAULT_K_SET_MAX`], then `⌊n/2⌋ − 2`, the largest k whose split
/// halves equal those at k = 1 (`resample::split_sizes`); at least 1.
fn default_k_cut(n: usize, p: usize, n_eff: f64, residual_df: Option<usize>) -> usize {
    k_limit(DEFAULT_K_SET_MAX, p, Some(n_eff), residual_df)
        .min((n / 2).saturating_sub(2))
        .max(1)
}

/// The min-p p-value over a set of k. `table` is `|K| × C`, row i the
/// statistic at the i-th k, column 0 the observed `y` and columns `1..C`
/// the null columns, a non-finite null already `+∞`; `observed_p[i]` is
/// row i's single-k p. With `p_i(c) = #{c' : T_i(c') ≥ T_i(c)} / C` and
/// `m(c) = min_i p_i(c)`, `m(0) = min_i observed_p[i]`, the p is
/// `#{c : m(c) ≤ m(0)} / C` (Westfall & Young 1993).
fn min_p_combine(table: MatRef<'_, f64>, observed_p: &[f64]) -> f64 {
    let c_all = table.ncols();
    let m0 = observed_p.iter().copied().fold(f64::INFINITY, f64::min);
    let mut m = vec![f64::INFINITY; c_all];
    for i in 0..table.nrows() {
        // Ranks by sorting, O(C log C) per k; a NaN (an observed statistic
        // only) is never `≥` anything.
        let mut sorted: Vec<f64> = (0..c_all)
            .map(|c| table[(i, c)])
            .filter(|v| !v.is_nan())
            .collect();
        sorted.sort_unstable_by(f64::total_cmp);
        for (c, mc) in m.iter_mut().enumerate().skip(1) {
            let t = table[(i, c)];
            let at_least = sorted.len() - sorted.partition_point(|&v| v < t);
            *mc = mc.min(at_least as f64 / c_all as f64);
        }
    }
    // Configuration 0 is `m(0)` itself, so it always counts.
    let hits = 1 + m[1..].iter().filter(|&&v| v <= m0).count();
    hits as f64 / c_all as f64
}

/// Who owns the `split_nb` auto-gate decision for one confirmatory call.
#[derive(Clone, Copy)]
pub(crate) enum GateMode<'a> {
    /// This call evaluates the rule and reroutes if it fires. Every public
    /// caller owns the decision.
    Owned,
    /// The caller already decided, on the undeflated X. Run the requested
    /// method as-is. Only `sequential::run_incremental_sequence` passes this:
    /// a step sees the deflated residual, whose spectrum is not X's, so
    /// re-evaluating there could flip methods mid-chain — which closed testing
    /// cannot use. Carries the gate's Gram matrix when the caller's gate
    /// built it from this call's X, weights and scale (the sequence's first
    /// step), for the k = 1 test to read.
    Decided(Option<&'a HeldOutGram>),
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

#[allow(clippy::needless_pass_by_value)] // ConfirmatoryTestInput wraps non-Copy Pls1Model ref
#[allow(clippy::many_single_char_names)]
pub(crate) fn confirmatory_test_impl(
    input: ConfirmatoryTestInput<'_>,
    opts: ConfirmatoryTestOpts,
    gate: GateMode<'_>,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let (x, y, k, weights, covariates, null_y) = match &input {
        ConfirmatoryTestInput::Raw {
            x,
            y,
            k,
            weights,
            covariates,
            null_y,
        } => (*x, *y, *k, *weights, *covariates, *null_y),
        ConfirmatoryTestInput::Gram {
            gram,
            y,
            k,
            covariates,
            null_y,
        } => {
            return confirmatory_test_gram(gram, *y, *k, *covariates, *null_y, opts);
        }
        ConfirmatoryTestInput::Model(_) => {
            // Model form: wrappers must call us with Raw (they hold the original X reference).
            return Err(PlsKitError::Internal(
                "Model form not yet supported in core; wrapper must pass Raw".into(),
            ));
        }
    };
    let opts = apply_null_y(opts, null_y, covariates.is_some(), x.nrows())?;
    let Some(z) = covariates else {
        return confirmatory_test_run(x, y, k, weights, opts, gate, None, null_y, None);
    };
    check_covariate_method(opts.args.method(), "split_exact, raw_perm and auto")?;
    let n = x.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x.ncols()),
            y: y.nrows(),
        });
    }
    crate::fit::check_finite_mat(x)?;
    crate::fit::check_finite_col(y)?;
    // The n_eff floor and the residual df are checked at the largest k
    // in `confirmatory_test_run`, once the set is resolved.
    let (w_norm, _) = crate::fit::validate_and_normalize_weights(weights, n, 0)?;
    let adj = crate::covariates::adjust(
        x,
        y,
        z,
        w_norm.as_ref().map(Col::as_ref),
        opts.pre_standardized,
    )?;
    confirmatory_test_run(
        adj.x.as_ref(),
        adj.y.as_ref(),
        k,
        weights,
        ConfirmatoryTestOpts {
            pre_standardized: true,
            ..opts
        },
        gate,
        Some(&adj.basis),
        None,
        Some(CiInput {
            x,
            y,
            z: Some(z),
            pre_standardized: opts.pre_standardized,
        }),
    )
}

/// The `null_map` of `null_y`: null column c (c ≥ 1) is column c − 1 of
/// `null_y`, overwriting the permuted outcome, so the runner's seed draws
/// and every route stay as they are for `n_perm = B`.
fn null_y_map(ny: MatRef<'_, f64>) -> impl Fn(usize, faer::MatMut<'_, f64>) + Sync + '_ {
    move |c, m| m.col_mut(0).copy_from(ny.col(c - 1))
}

/// `y` minus its all-row (weighted) mean, and each `null_y` column minus
/// its own weighted mean under weights (minus `y`'s mean unweighted, where
/// the statistic does not read a column's location). The split tests
/// correlate held-out predictions with `√w·y` on `y` as given, so without
/// this shift a column's location enters their statistic under weights, and
/// costs the no-refit routes precision when it is far from 0. Permutation
/// nulls permute the centred `y`, so they keep `y`'s constant.
fn centre_targets(
    y: ColRef<'_, f64>,
    w: Option<ColRef<'_, f64>>,
    null_y: Option<MatRef<'_, f64>>,
) -> (Col<f64>, Option<Mat<f64>>) {
    let mean = |v: ColRef<'_, f64>| crate::linalg::standardize1_weighted(v, w).1;
    let c = mean(y);
    (
        Col::from_fn(y.nrows(), |i| y[i] - c),
        null_y.map(|m| {
            let cs: Vec<f64> = (0..m.ncols())
                .map(|j| w.map_or(c, |_| mean(m.col(j))))
                .collect();
            Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] - cs[j])
        }),
    )
}

/// The refusal of `split_nb`, `score` and `e` with covariates: they have no
/// permutation null to correct.
pub(crate) fn check_covariate_method(
    method: ConfirmatoryMethod,
    accepted: &str,
) -> PlsKitResult<()> {
    match method {
        ConfirmatoryMethod::SplitNb | ConfirmatoryMethod::Score | ConfirmatoryMethod::E => {
            Err(PlsKitError::InvalidArgument(format!(
                "covariates are accepted by test_method {accepted}, not {}",
                method.as_str()
            )))
        }
        _ => Ok(()),
    }
}

/// The checks of `null_y` on `n` rows, and the options the test then runs
/// with: `n_perm` is the number of columns `B`, so the count floors and the
/// seed draws are those of a permutation test with `B` permutations. A
/// no-op without `null_y`.
fn apply_null_y(
    opts: ConfirmatoryTestOpts,
    null_y: Option<MatRef<'_, f64>>,
    has_covariates: bool,
    n: usize,
) -> PlsKitResult<ConfirmatoryTestOpts> {
    let Some(ny) = null_y else {
        return Ok(opts);
    };
    let ConfirmatoryArgs::SplitExact { n_splits, .. } = opts.args else {
        return Err(PlsKitError::InvalidArgument(format!(
            "null_y is accepted by test_method split_exact, not {}",
            opts.args.method().as_str()
        )));
    };
    if has_covariates {
        return Err(PlsKitError::InvalidArgument(
            "null_y does not combine with covariates: adjust X and y before passing your own null"
                .into(),
        ));
    }
    if ny.nrows() != n {
        return Err(PlsKitError::ShapeMismatch(format!(
            "null_y has {} rows, expected {n}",
            ny.nrows()
        )));
    }
    if ny.ncols() == 0 {
        return Err(PlsKitError::InvalidArgument(
            "null_y needs at least one column".into(),
        ));
    }
    crate::fit::check_finite_mat(ny)?;
    Ok(ConfirmatoryTestOpts {
        args: ConfirmatoryArgs::SplitExact {
            n_perm: ny.ncols(),
            n_splits,
        },
        ..opts
    })
}

/// The raw data the `ci` branch refits on when the test runs on adjusted
/// data: each subsample redoes the adjustment on its own rows.
#[derive(Clone, Copy)]
struct CiInput<'a> {
    x: MatRef<'a, f64>,
    y: ColRef<'a, f64>,
    z: Option<MatRef<'a, f64>>,
    pre_standardized: bool,
}

/// `pls1_confirmatory_test` on `(x, y)` as given: the checks, the method
/// resolution, the test and the `ci` branch. With `cov`, `(x, y)` are the
/// residuals on `[1, Z]` of the standardized data, `opts.pre_standardized`
/// is set, every permutation null column `π_c(ỹ)` is replaced by its
/// residual on `[1, Z]` (Freedman–Lane) and `auto` resolves to
/// `split_exact`. The `ci` branch refits on `(x, y)` as given under
/// `opts.pre_standardized`.
pub(crate) fn confirmatory_test_core(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: ConfirmatoryTestOpts,
    gate: GateMode<'_>,
    cov: Option<&crate::covariates::CovariateBasis>,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    if cov.is_some() {
        check_covariate_method(opts.args.method(), "split_exact, raw_perm and auto")?;
        // The `ci` branch with covariates refits on the raw data, which
        // only `confirmatory_test_impl` holds.
        debug_assert!(
            opts.ci.is_none(),
            "ci with covariates goes through confirmatory_test_impl"
        );
    }
    confirmatory_test_run(
        x,
        y,
        Some(std::slice::from_ref(&k)),
        weights,
        opts,
        gate,
        cov,
        None,
        None,
    )
}

/// [`confirmatory_test_core`], with the `ci` branch's raw data and
/// covariates (`ci_input`) when the test runs on adjusted data; `None`
/// refits on `(x_ref, y_ref)` under `opts.pre_standardized`.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn confirmatory_test_run(
    x_ref: MatRef<'_, f64>,
    y_ref: ColRef<'_, f64>,
    k: Option<&[usize]>,
    weights_in: Option<ColRef<'_, f64>>,
    opts: ConfirmatoryTestOpts,
    gate: GateMode<'_>,
    cov: Option<&crate::covariates::CovariateBasis>,
    null_y: Option<MatRef<'_, f64>>,
    ci_input: Option<CiInput<'_>>,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let n = x_ref.nrows();
    if y_ref.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x_ref.ncols()),
            y: y_ref.nrows(),
        });
    }
    crate::fit::check_finite_mat(x_ref)?;
    crate::fit::check_finite_col(y_ref)?;
    check_method_counts(opts.args, n)?;
    let p = x_ref.ncols();
    let (_, n_eff_all) = crate::fit::validate_and_normalize_weights(weights_in, n, 0)?;
    let ks = resolve_k_set(
        k,
        opts.args.method(),
        opts.ci.is_some(),
        opts.keep,
        default_k_cut(
            n,
            p,
            n_eff_all,
            cov.map(crate::covariates::CovariateBasis::residual_df),
        ),
    )?;
    // The single-k methods and options run at this k: their set is `{k}`.
    let k_max = *ks.last().expect("a resolved set is non-empty");
    // k = 0 is rejected here, before dispatch, for every method: if it were
    // left to the runners, `raw_perm` and `e` would reject it through
    // `pls1_fit` while `split_exact` / `split_nb` would return a constant p
    // and `score` (which never reads k) a real one. Same error `pls1_fit`
    // raises.
    if ks[0] == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    if k_max > p {
        return Err(PlsKitError::KExceedsMax { k: k_max, k_max: p });
    }

    // Validate + normalize weights. Row-scaling pattern: materialize X̃ = √w' · X_std
    // and ỹ = √w' · y_std and run the unweighted statistics on (X̃, ỹ).
    let (w_norm, n_eff_val) = crate::fit::validate_weights_for_k(weights_in, n, k_max)?;
    if let Some(b) = cov {
        b.check_df(k_max)?;
    }

    // The one column scale of X the held-out methods (`split_exact`,
    // `split_nb` and its gate, `e`, `raw_perm`) divide by, taken over all
    // rows: the fit's (weighted moments when weighted), or ones under
    // `pre_standardized`. `score` scales X itself and never reads it; its
    // empty scale would trip the helpers' length checks if it did.
    let x_scale = if matches!(opts.args, ConfirmatoryArgs::Score) {
        Col::<f64>::zeros(0)
    } else {
        crate::linalg::held_out_scale(
            x_ref,
            w_norm.as_ref().map(Col::as_ref),
            opts.pre_standardized,
        )
    };

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

    // The CI knobs, and the subsample size they give on this data, are
    // checked here, before the test runs, so a bad one costs no resampling.
    let ci_input = ci_input.unwrap_or(CiInput {
        x: x_ref,
        y: y_ref,
        z: None,
        pre_standardized: opts.pre_standardized,
    });
    let ci_sub_opts = opts
        .ci
        .as_ref()
        .map(|ci_opts| crate::subsample::SubsampleOpts {
            n_boot: ci_opts.n_boot,
            m_rate: ci_opts.m_rate,
            level: ci_opts.level,
            pre_standardized: ci_input.pre_standardized,
            max_failure_rate: ci_opts.max_failure_rate,
            max_skip_rate: opts.max_skip_rate,
        });
    if let Some(sub_opts) = &ci_sub_opts {
        sub_opts.validate()?;
        let rank = cov.map_or(1, crate::covariates::CovariateBasis::rank);
        crate::subsample::resolve_ci_m(n, k_max, rank, sub_opts.m_rate)?;
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
    // the method already and would only be re-paying for a pass over X plus
    // the eigendecomposition of its short-side Gram matrix
    // (`resolve_split_nb`), whose answer it discards.
    //
    // `Auto` resolves in the same place and for the same reason: the output
    // reports the method that ran, never `"auto"`.
    let mut args_resolved = opts.args;
    let mut stable_rank_out = None;
    // The gate's Gram matrix on wide X: the k = 1 test that runs next,
    // `split_nb` or the `split_exact` it reroutes to, reads it.
    let mut owned_gram = None;
    match (gate, opts.args) {
        (GateMode::Owned, ConfirmatoryArgs::SplitNb { n_splits, force }) => {
            let gate = resolve_split_nb(
                x_ref,
                w_norm.as_ref().map(Col::as_ref),
                n_eff_val,
                force,
                x_scale.as_ref(),
            );
            stable_rank_out = Some(gate.stable_rank);
            if gate.reroute {
                args_resolved = ConfirmatoryArgs::SplitExact {
                    n_perm: SPLIT_NB_REROUTE_N_PERM,
                    n_splits,
                };
            }
            owned_gram = gate.gram;
        }
        (GateMode::Owned, ConfirmatoryArgs::Auto { n_perm, n_splits }) => {
            let (method, sr, gram) = resolve_auto_on_x(
                x_ref,
                w_norm.as_ref().map(Col::as_ref),
                n_eff_val,
                split_exact_no_refit_route(k_max, opts.keep),
                false,
                cov.is_some(),
                ks.len() > 1,
                x_scale.as_ref(),
            );
            stable_rank_out = sr;
            owned_gram = gram;
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
    let gate_gram = match gate {
        GateMode::Owned => owned_gram.as_ref(),
        GateMode::Decided(g) => g,
    };

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;

    // The split tests read `y` as given (see `centre_targets`); covariate
    // residuals are centred already.
    let split = matches!(
        args_resolved,
        ConfirmatoryArgs::SplitNb { .. } | ConfirmatoryArgs::SplitExact { .. }
    );
    let centred = (split && cov.is_none())
        .then(|| centre_targets(y_ref, w_norm.as_ref().map(Col::as_ref), null_y));
    let (y_ref, null_y) = match &centred {
        Some((y, ny)) => (y.as_ref(), ny.as_ref().map(Mat::as_ref)),
        None => (y_ref, null_y),
    };

    // Freedman–Lane: null column c is the residual on `[1, Z]` of the
    // permuted residual, `M_Z·π_c(ỹ)`.
    let fl = cov.map(|b| move |_: usize, m: faer::MatMut<'_, f64>| b.residualize(m));
    let ny = null_y.map(null_y_map);
    let null_map: Option<crate::resample::NullMap<'_>> = fl
        .as_ref()
        .map(|f| f as crate::resample::NullMap<'_>)
        .or(ny.as_ref().map(|f| f as crate::resample::NullMap<'_>));

    let (result, n_perm_out, n_splits_out): (SetResult, _, _) = match args_resolved {
        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => (
            run_raw_perm(
                x_ref,
                y_ref,
                &ks,
                n_perm,
                n_folds,
                w_norm.as_ref().map(Col::as_ref),
                x_scale.as_ref(),
                &opts,
                null_map,
                &mut rng,
            )?,
            Some(n_perm),
            None,
        ),
        ConfirmatoryArgs::SplitNb { n_splits, .. } => (
            run_split_nb(
                x_ref,
                y_ref,
                k_max,
                n_splits,
                w_norm.as_ref().map(Col::as_ref),
                x_scale.as_ref(),
                &opts,
                gate_gram,
                &mut rng,
            )?
            .into(),
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
            // Route decided here, before any draw from `rng`: both routes
            // first draw their splits through `draw_splits`, then one child
            // seed per null column for `resample::Columns`, so every route
            // sees the same splits and the same null columns at a given seed.
            let route = split_exact_refit_route(
                n,
                x_ref.ncols(),
                n_perm,
                n_splits,
                k_max,
                opts.keep,
                w_norm.is_some(),
                gate_gram.is_some(),
            );
            let wr = w_norm.as_ref().map(Col::as_ref);
            let sc = x_scale.as_ref();
            let result = match route {
                ReplicateRoute::Special => run_split_perm_nr(
                    x_ref, y_ref, k_max, n_perm, n_splits, wr, sc, &opts, null_map, gate_gram,
                    &mut rng,
                )?
                .into(),
                route @ (ReplicateRoute::Primal
                | ReplicateRoute::Nspace
                | ReplicateRoute::GramP) => run_split_perm(
                    route, x_ref, y_ref, &ks, n_perm, n_splits, wr, sc, &opts, null_map, gate_gram,
                    &mut rng,
                )?,
            };
            (result, Some(n_perm), Some(n_splits))
        }
        ConfirmatoryArgs::Score => (
            run_score(x_ref, y_ref, w_norm.as_ref().map(Col::as_ref), &opts)?.into(),
            None,
            None,
        ),
        ConfirmatoryArgs::E => (
            run_e(
                x_ref,
                y_ref,
                k_max,
                w_norm.as_ref().map(Col::as_ref),
                x_scale.as_ref(),
                &opts,
                &mut rng,
            )?
            .into(),
            None,
            None,
        ),
        // Resolved above under `GateMode::Owned`; a `Decided` caller passes
        // the method it already resolved.
        ConfirmatoryArgs::Auto { .. } => {
            return Err(PlsKitError::Internal(
                "test_method='auto' reached dispatch unresolved".into(),
            ));
        }
    };

    let ci_payload = if let Some(sub_opts) = ci_sub_opts {
        // Independent child-seed branch — derive a second child RNG from the
        // post-test parent state. This guarantees stream non-interference
        // between test path and CI path while keeping a single user-facing seed.
        let mut ci_rng = {
            use rand::Rng;
            crate::rng::child_rng(rng.next_u64())
        };

        // Reference fit on full data.
        let fit_ref = {
            use crate::fit::{FitOpts, KSpec, pls1_fit_impl};
            pls1_fit_impl(
                ci_input.x,
                ci_input.y,
                KSpec::Fixed(k_max),
                w_norm.as_ref().map(Col::as_ref),
                ci_input.z,
                FitOpts {
                    pre_standardized: ci_input.pre_standardized,
                    ..FitOpts::default()
                },
            )?
        };

        // leverage_ref[j] = diag(W_ref (W_ref' W_ref)^-1 W_ref').
        let leverage_ref = crate::linalg::leverage_diag(fit_ref.w_star.as_ref());
        Some(crate::subsample::pls1_subsample_inference_confirmatory(
            ci_input.x,
            ci_input.y,
            k_max,
            fit_ref.beta.as_ref(),
            &leverage_ref,
            sub_opts,
            w_norm.as_ref().map(Col::as_ref),
            ci_input.z,
            &mut ci_rng,
        )?)
    } else {
        None
    };

    let (statistic, pvalues_by_k, statistics_by_k) = result.by_k(&ks);
    Ok(ConfirmatoryTestOutput {
        pvalue: result.pvalue,
        statistic,
        test_method: args_resolved.method().as_str().to_owned(),
        k: ks,
        pvalues_by_k,
        statistics_by_k,
        n_perm: n_perm_out,
        n_splits: n_splits_out,
        seed: seed_used,
        ci: ci_payload,
        n_eff: n_eff_val,
        rho_hat: result.per_k[0].rho_hat,
        stable_rank: stable_rank_out,
        n_covariates: cov.map(crate::covariates::CovariateBasis::n_covariates),
    })
}

/// Per-method count floors of `pls1_confirmatory_test` on `n` rows. Each
/// method validates only the counts it uses: resampling needs ≥1
/// permutation, split/CV calibration needs ≥2 of each.
fn check_method_counts(args: ConfirmatoryArgs, n: usize) -> PlsKitResult<()> {
    match args {
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
    Ok(())
}

/// The `gram` form's refusal of `pre_standardized` set in the options,
/// shared by `pls1_confirmatory_test` and `pls3_confirmatory_test_gram`:
/// the Gram matrix records how it was scaled.
pub(crate) fn check_gram_flag(pre_standardized: bool) -> PlsKitResult<()> {
    if pre_standardized {
        return Err(PlsKitError::InvalidArgument(
            "pre_standardized is read from the Gram matrix with gram: set it on the \
             GramMatrixOutput, not in the options"
                .into(),
        ));
    }
    Ok(())
}

/// `pls1_confirmatory_test` on the `Gram` input form. `gram.gram` is
/// centred over its rows and its entries checked (`HeldOutGram::from_gram`),
/// and every method then runs from that matrix alone: `raw_perm` at `k = 1`
/// by `dual_route::pls1_cv_r2_columns` and otherwise per fold by
/// [`fold_columns_nspace`]; `split_exact` and `split_nb` at `k = 1` by the
/// no-refit body's Gram route (`nr_split_r_gram`) and otherwise per split by
/// [`split_columns_nspace`]; `e` on one split's blocks; `score` from
/// `y'Gy` and the eigenvalues of `G`; and the `split_nb` gate and `auto`
/// from `G`'s stable rank. The draws from the seed are the X call's, so on
/// `gram_matrix(x, flag)` every method equals the X call on `x` with the
/// flag to rounding, `score` on X centred over all rows (it reads the
/// centred `G`, while the X call reads X as given under the flag). Under
/// the flag, on X whose largest entry is outside `[2^-256, 2^256]`, `G` is
/// that of `X/c` (`c` a power of two `gram_matrix` does not store), so the
/// `score` statistic is the one of `X/c` (the X call's divided by `c²`);
/// its p-value is the X call's.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
fn confirmatory_test_gram(
    gram: &crate::held_out_gram::GramMatrixOutput,
    y: ColRef<'_, f64>,
    k: Option<&[usize]>,
    covariates: Option<MatRef<'_, f64>>,
    null_y: Option<MatRef<'_, f64>>,
    opts: ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let n = gram.gram.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::InvalidArgument(format!(
            "gram is {n} x {}, y has {} rows: gram needs one row and column per row of y",
            gram.gram.ncols(),
            y.nrows()
        )));
    }
    crate::fit::check_finite_col(y)?;
    let opts = apply_null_y(opts, null_y, covariates.is_some(), n)?;
    check_method_counts(opts.args, n)?;
    let p = gram.n_features;
    // The basis is built before the set resolves: the default set's cut
    // reads its residual df.
    let basis = covariates
        .map(|z| {
            check_covariate_method(opts.args.method(), "split_exact, raw_perm and auto")?;
            if gram.gram.ncols() != n {
                return Err(PlsKitError::ShapeMismatch(format!(
                    "gram must be square, got {n} x {}",
                    gram.gram.ncols()
                )));
            }
            if z.nrows() != n {
                return Err(PlsKitError::ShapeMismatch(format!(
                    "covariates have {} rows, gram has {n}: one row of covariates per row of gram",
                    z.nrows()
                )));
            }
            crate::covariates::CovariateBasis::new(z, None)
        })
        .transpose()?;
    // No weights with gram, so Kish n_eff is exactly the row count.
    let n_eff = n as f64;
    let ks = resolve_k_set(
        k,
        opts.args.method(),
        opts.ci.is_some(),
        opts.keep,
        default_k_cut(
            n,
            p,
            n_eff,
            basis
                .as_ref()
                .map(crate::covariates::CovariateBasis::residual_df),
        ),
    )?;
    if ks[0] == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    // Every check below is a single call's, at the largest k.
    let k = *ks.last().expect("a resolved set is non-empty");
    if k > p {
        return Err(PlsKitError::KExceedsMax { k, k_max: p });
    }
    // The X call's `n >= k + 1` check, with no weights.
    crate::fit::validate_weights_for_k(None, n, k)?;
    let k_dual_max = crate::dual_route::K_DUAL_MAX;
    if k > k_dual_max {
        return Err(PlsKitError::InvalidArgument(format!(
            "gram serves k <= {k_dual_max}, got k = {k}: a fit with more components needs X"
        )));
    }
    check_gram_flag(opts.pre_standardized)?;
    if opts.keep.is_some() {
        return Err(PlsKitError::InvalidArgument(
            "keep does not combine with gram: hard selection acts on the weight vector of \
             length n_features, which the Gram matrix does not hold"
                .into(),
        ));
    }
    if opts.ci.is_some() {
        return Err(PlsKitError::InvalidArgument(
            "ci does not combine with gram: the CI resamples the fit on X".into(),
        ));
    }
    let (g, y_adj) = if let Some(basis) = &basis {
        basis.check_df(k)?;
        // Non-finite entries of G reach `from_gram` through the projection.
        let g = HeldOutGram::from_gram(basis.project_gram(gram.gram.as_ref()), p)?;
        let mut y_adj = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
        basis.residualize(y_adj.as_mut());
        (g, Some(y_adj))
    } else {
        (HeldOutGram::from_gram(gram.gram.to_owned(), p)?, None)
    };
    let y = y_adj.as_ref().map_or(y, |m| m.col(0));

    let mut args = opts.args;
    let mut stable_rank_out = None;
    match opts.args {
        ConfirmatoryArgs::SplitNb { n_splits, force } => {
            let sr = g.stable_rank();
            stable_rank_out = Some(sr);
            if split_nb_gate_fires(p, n_eff, sr) && !force {
                args = ConfirmatoryArgs::SplitExact {
                    n_perm: SPLIT_NB_REROUTE_N_PERM,
                    n_splits,
                };
            }
        }
        ConfirmatoryArgs::Auto { n_perm, n_splits } => {
            let (method, sr, _) = resolve_auto(&AutoInput {
                n,
                p,
                n_eff,
                no_refit: split_exact_no_refit_route(k, None),
                pls3: false,
                covariates: basis.is_some(),
                multi_k: ks.len() > 1,
                stable_rank: &|| g.stable_rank(),
            });
            stable_rank_out = sr;
            args = if method == ConfirmatoryMethod::SplitNb {
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

    // As in the Raw form: the split tests get `y` and `null_y` centred.
    let split = matches!(
        args,
        ConfirmatoryArgs::SplitNb { .. } | ConfirmatoryArgs::SplitExact { .. }
    );
    let centred = (split && basis.is_none()).then(|| centre_targets(y, None, null_y));
    let (y, null_y) = match &centred {
        Some((y, ny)) => (y.as_ref(), ny.as_ref().map(Mat::as_ref)),
        None => (y, null_y),
    };
    // Freedman–Lane, as in the Raw form.
    let fl = basis
        .as_ref()
        .map(|b| move |_: usize, m: faer::MatMut<'_, f64>| b.residualize(m));
    let ny = null_y.map(null_y_map);
    let null_map: Option<crate::resample::NullMap<'_>> = fl
        .as_ref()
        .map(|f| f as crate::resample::NullMap<'_>)
        .or(ny.as_ref().map(|f| f as crate::resample::NullMap<'_>));

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;
    let (result, n_perm_out, n_splits_out): (SetResult, _, _) = match args {
        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => (
            run_raw_perm_gram(&g, y, &ks, n_perm, n_folds, null_map, &mut rng)?,
            Some(n_perm),
            None,
        ),
        ConfirmatoryArgs::SplitNb { n_splits, .. } => (
            run_split_nb_gram(&g, y, k, n_splits, &mut rng)?.into(),
            None,
            Some(n_splits),
        ),
        ConfirmatoryArgs::SplitExact { n_perm, n_splits } => (
            run_split_perm_gram(&g, y, &ks, n_perm, n_splits, null_map, &mut rng)?,
            Some(n_perm),
            Some(n_splits),
        ),
        ConfirmatoryArgs::Score => (
            run_score_gram(&g, y, gram.pre_standardized).into(),
            None,
            None,
        ),
        ConfirmatoryArgs::E => (run_e_gram(&g, y, k, &mut rng)?.into(), None, None),
        ConfirmatoryArgs::Auto { .. } => {
            return Err(PlsKitError::Internal(
                "test_method='auto' reached dispatch unresolved".into(),
            ));
        }
    };
    let (statistic, pvalues_by_k, statistics_by_k) = result.by_k(&ks);
    Ok(ConfirmatoryTestOutput {
        pvalue: result.pvalue,
        statistic,
        test_method: args.method().as_str().to_owned(),
        k: ks,
        pvalues_by_k,
        statistics_by_k,
        n_perm: n_perm_out,
        n_splits: n_splits_out,
        seed: seed_used,
        ci: None,
        n_eff,
        rho_hat: result.per_k[0].rho_hat,
        stable_rank: stable_rank_out,
        n_covariates: basis
            .as_ref()
            .map(crate::covariates::CovariateBasis::n_covariates),
    })
}

/// `raw_perm` from the centred all-row Gram matrix `g`: the folds and null
/// seeds [`run_raw_perm`] draws, then the pooled CV R² of every column from
/// `dual_route::pls1_cv_r2_columns` when `ks` is `{1}`, or from
/// [`fold_columns_nspace`] per fold at every k of `ks`, the folds in
/// batches of `resample::map_units`.
fn run_raw_perm_gram(
    g: &HeldOutGram,
    y: ColRef<'_, f64>,
    ks: &[usize],
    n_perm: usize,
    n_folds: usize,
    null_map: Option<crate::resample::NullMap<'_>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<SetResult> {
    let n = y.nrows();
    let (folds, seeds) = raw_perm_draws(n, n_folds, n_perm, rng);
    let cols = crate::resample::Columns {
        y: y.as_mat(),
        seeds: &seeds,
        null_map,
    };
    let r2 = if ks == [1] {
        let y_mat = cols.matrix(0, cols.len());
        one_row(&crate::dual_route::pls1_cv_r2_columns(
            g,
            y_mat.as_ref(),
            &folds,
        )?)
    } else {
        let n_tr = n - folds.iter().map(Vec::len).min().unwrap_or(0);
        let unit = crate::resample::fold_blocks_bytes(n, &folds, n_tr);
        let per_fold = crate::resample::map_units(folds.len(), unit, |fi| {
            Ok(fold_columns_nspace(&folds, fi, g, ks, cols.len(), &|c| {
                cols.column(c)
            }))
        });
        pool_fold_contributions(per_fold, cols.len(), ks.len())?
    };
    set_result(&r2, n_perm, raw_perm_result)
}

/// `split_nb` from the centred all-row Gram matrix `g`: the splits
/// [`run_split_nb`] draws, each split's held-out correlation from the
/// no-refit body's Gram route at `k = 1` and from [`split_columns_nspace`]
/// otherwise, then [`nb_test`].
#[allow(clippy::many_single_char_names)]
fn run_split_nb_gram(
    g: &HeldOutGram,
    y: ColRef<'_, f64>,
    k: usize,
    n_splits: usize,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    let n = y.nrows();
    let (n_train, n_test) = crate::resample::split_sizes(n, k);
    let splits = draw_splits(n, k, n_splits, rng)?;
    let r = if k == 1 {
        let y_mat = nr_outcome_matrix(&crate::resample::Columns {
            y: y.as_mat(),
            seeds: &[],
            null_map: None,
        });
        let unit = nr_split_unit_bytes(n_train, n_test, K1Route::G, false);
        crate::resample::map_units(splits.len(), unit, |j| {
            nr_split_r_gram(g, None, y_mat.as_ref(), &splits[j], None)[0]
        })
    } else {
        let unit = split_blocks_bytes(&splits, n_train);
        crate::resample::map_units(splits.len(), unit, |j| {
            split_columns_nspace(g, &splits[j], 1, &[k], &|_| y.to_owned())[0][0]
        })
    };
    let r_splits = Col::<f64>::from_fn(r.len(), |j| r[j]);
    let (pvalue, statistic, _, _) = nb_test(&r_splits, n_train, n_test);
    Ok(RunResult {
        pvalue,
        statistic,
        rho_hat: nb_rho_hat(&r_splits, n_test),
    })
}

/// `split_exact` from the centred all-row Gram matrix `g`: the splits and
/// null seeds [`run_split_perm`] draws for the largest k, then every
/// column's `z̄` from the no-refit body's Gram route when `ks` is `{1}`
/// and from [`split_columns_nspace`] at every k of `ks` otherwise.
#[allow(clippy::many_single_char_names)]
fn run_split_perm_gram(
    g: &HeldOutGram,
    y: ColRef<'_, f64>,
    ks: &[usize],
    n_perm: usize,
    n_splits: usize,
    null_map: Option<crate::resample::NullMap<'_>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<SetResult> {
    let n = y.nrows();
    let k = *ks.last().expect("ks is non-empty");
    let nk = ks.len();
    let design = draw_design(n, k, n_splits, rng)?;
    let splits = &design.splits;
    let seeds = crate::rng::child_seeds(rng, n_perm);
    let cols = crate::resample::Columns {
        y: y.as_mat(),
        seeds: &seeds,
        null_map,
    };
    let (n_train, n_test) = crate::resample::split_sizes(n, k);
    // ±0.9999 pre-atanh clamp mirrored from nb_test (change together).
    let fisher_z = |r: f64| r.clamp(-0.9999, 0.9999).atanh();
    let z = if ks == [1] {
        let y_mat = nr_outcome_matrix(&cols);
        let unit = nr_split_unit_bytes(n_train, n_test, K1Route::G, false);
        let rows: Vec<Vec<f64>> = crate::resample::map_units(splits.len(), unit, |j| {
            nr_split_r_gram(g, None, y_mat.as_ref(), &splits[j], None)
                .into_iter()
                .map(fisher_z)
                .collect()
        });
        one_row(&zbars_from_rows(cols.len(), &rows))
    } else {
        let unit = split_blocks_bytes(splits, n_train);
        // Flat as in `split_zbars_columns`: entry `c·nk + i` is column c at `ks[i]`.
        let flat = zbars_over_splits(splits, cols.len() * nk, unit, |sp: &SplitIdx| {
            split_columns_nspace(g, sp, cols.len(), ks, &|c| cols.column(c))
                .into_iter()
                .flatten()
                .map(fisher_z)
                .collect()
        });
        Mat::<f64>::from_fn(nk, cols.len(), |i, c| flat[c * nk + i])
    };
    set_result(&z, n_perm, |z, b| Ok(split_perm_result(z, b)))
}

/// `score` from the centred all-row Gram matrix `g`: `T = y'Gy` with `y`
/// standardized (as given under `pre_standardized`), and the
/// Welch-Satterthwaite reference from the eigenvalues of `G`.
fn run_score_gram(g: &HeldOutGram, y: ColRef<'_, f64>, pre_standardized: bool) -> RunResult {
    let ys = if pre_standardized {
        y.to_owned()
    } else {
        crate::linalg::standardize1(y).0
    };
    let par = crate::fit::par_fixed();
    let gy = crate::linalg::mat_vec(g.g(), ys.as_ref(), par);
    let t_obs: f64 = (0..ys.nrows()).map(|i| ys[i] * gy[i]).sum();
    score_result(t_obs, &eigenvalues_symmetric(g.g(), par))
}

/// `e` from the centred all-row Gram matrix `g`: the split [`run_e`]
/// draws, the training fit on its blocks (`NspaceSplit::from_gram`) by the
/// n-space kernel, and the likelihoods of [`run_e`] on its predictions.
#[allow(clippy::similar_names)]
fn run_e_gram(
    g: &HeldOutGram,
    y: ColRef<'_, f64>,
    k: usize,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use crate::linalg::{col_row_subset, standardize1};
    let n = y.nrows();
    // Mirror of draw_splits' n ≥ k+5 guard (change together).
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, _) = crate::resample::split_sizes(n, k);
    let (tr, te) = crate::resample::one_split(n, n_train, rng);
    let y_tr = col_row_subset(y, &tr);
    let y_te = col_row_subset(y, &te);
    let (ys_tr, y_mean, y_scale) = standardize1(y_tr.as_ref());
    crate::fit::check_finite_col(ys_tr.as_ref())?;
    let ys_te = Col::<f64>::from_fn(y_te.nrows(), |i| (y_te[i] - y_mean) / y_scale);
    let ns = crate::dual_route::NspaceSplit::from_gram(g, &SplitIdx { tr, te });
    let (y_pred_tr, y_pred) = ns.predictions(ys_tr.as_ref(), k);
    Ok(e_result(&ys_tr, &y_pred_tr, &ys_te, &y_pred))
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

/// A test's result over its set of k: `per_k[i]` is the single-k result at
/// the i-th k, `pvalue` the p for the whole set (`per_k[0]`'s for one k).
struct SetResult {
    pvalue: f64,
    per_k: Vec<RunResult>,
}

impl From<RunResult> for SetResult {
    fn from(r: RunResult) -> Self {
        Self {
            pvalue: r.pvalue,
            per_k: vec![r],
        }
    }
}

impl SetResult {
    /// The output's `statistic` (one k only) and per-k maps over `ks`.
    fn by_k(&self, ks: &[usize]) -> (Option<f64>, BTreeMap<usize, f64>, BTreeMap<usize, f64>) {
        let statistic = match self.per_k.as_slice() {
            [r] => Some(r.statistic),
            _ => None,
        };
        let map = |f: fn(&RunResult) -> f64| {
            ks.iter()
                .zip(&self.per_k)
                .map(|(&k, r)| (k, f(r)))
                .collect()
        };
        (statistic, map(|r| r.pvalue), map(|r| r.statistic))
    }
}

/// The result over a set of k from its statistic table (`|K| × C`, row i
/// at the i-th k, column 0 the observed `y`): row i's single-k result by
/// `one_k(row, n_perm)`, so each k keeps its own rules for a failed null
/// and its own errors, then min-p over the rows when there are several,
/// with a non-finite null ranked `+∞` (an exceedance, as in `one_k`).
fn set_result(
    table: &Mat<f64>,
    n_perm: usize,
    one_k: impl Fn(&[f64], usize) -> PlsKitResult<RunResult>,
) -> PlsKitResult<SetResult> {
    let per_k = (0..table.nrows())
        .map(|i| {
            let row: Vec<f64> = (0..table.ncols()).map(|c| table[(i, c)]).collect();
            one_k(&row, n_perm)
        })
        .collect::<PlsKitResult<Vec<_>>>()?;
    let pvalue = if let [one] = per_k.as_slice() {
        one.pvalue
    } else {
        let ranked = Mat::<f64>::from_fn(table.nrows(), table.ncols(), |i, c| {
            let v = table[(i, c)];
            if c > 0 && !v.is_finite() {
                f64::INFINITY
            } else {
                v
            }
        });
        let observed: Vec<f64> = per_k.iter().map(|r| r.pvalue).collect();
        min_p_combine(ranked.as_ref(), &observed)
    };
    Ok(SetResult { pvalue, per_k })
}

/// `v` as the one-row table of a one-element set.
fn one_row(v: &[f64]) -> Mat<f64> {
    Mat::<f64>::from_fn(1, v.len(), |_, c| v[c])
}

/// Row 0 of a per-k table, the statistics of a one-element `ks`.
#[cfg(test)]
pub(crate) fn first_row(m: &Mat<f64>) -> Vec<f64> {
    (0..m.ncols()).map(|c| m[(0, c)]).collect()
}

/// `run_raw_perm`'s route rule. The K = 1 Gram closed form
/// (`dual_route::pls1_cv_r2_columns`) needs a fixed training set reused
/// across replicates, which K-fold gives, plus the K = 1 coefficient
/// identity: deflation forecloses K ≥ 2, a sparse `keep` moves the selected
/// column set every replicate, and the weighted case is unverified, so all
/// of those stay primal. `n_tr_max` is the largest training fold
/// (`fold_split`'s groups differ by at most one), so the memory side of the
/// rule sees the worst case and the whole statistic runs on one route.
///
/// The closed form builds one all-row Gram matrix for the call and reads
/// each fold's blocks from it, so it is taken when its multiply-adds are
/// fewer than the primal route's, with `F = n_folds`, `B' = n_perm + 1`,
/// `n_tr = n_tr_max` and `n_val = n − n_tr`:
///
/// ```text
/// n²·p/2 + F·B'·(2·n_tr² + n_val·n_tr)  <  F·B'·(2·n_tr + n_val)·p
/// ```
///
/// Per column and fold that is `Gz`, `G(Gz)` and `Mz` against `X̃'z`, `X̃w`
/// and `X̃_val·coef`; the primal's loading pass `X̃'t` is not counted, so
/// the rule leans primal. Also `0 < n_tr < p` and
/// `n_tr ≤ DUAL_ROUTE_MAX_N_TR`. Decided from shape alone; internal and
/// silent.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn raw_perm_k1_gram_route(
    n: usize,
    n_folds: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    dense: bool,
    weighted: bool,
) -> bool {
    let n_tr = n - n / n_folds;
    if !(k == 1 && dense && !weighted)
        || n_tr == 0
        || n_tr >= p
        || n_tr > crate::dual_route::DUAL_ROUTE_MAX_N_TR
    {
        return false;
    }
    let (f, b) = (n_folds as f64, (n_perm + 1) as f64);
    let (t, v) = (n_tr as f64, (n - n_tr) as f64);
    let gram = crate::dual_route::one_gram_cost(n, p, false) + f * b * (2.0 * t * t + v * t);
    gram < f * b * (2.0 * t + v) * p as f64
}

/// The no-refit rule: the no-refit body is an exact shortcut only under the
/// K = 1 identity on a dense fit; everything else refits. It decides the
/// route of `split_exact` (see the dispatch in `confirmatory_test_impl`)
/// and of `split_nb` (`run_split_nb`).
pub(crate) fn split_exact_no_refit_route(k: usize, keep: Option<usize>) -> bool {
    k == 1 && keep.is_none()
}

#[cfg(test)]
thread_local! {
    /// Set by `with_gram_routes_disabled`; read through `gram_routes_disabled`.
    static GRAM_ROUTES_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the replicate-loop route selectors are restricted to
/// `ReplicateRoute::Special` and `ReplicateRoute::Primal`, which takes the
/// Gram routes `ReplicateRoute::Nspace` and `ReplicateRoute::GramP` out of
/// the choice, and `k1_route` to `K1Route::A` and `K1Route::B`, which
/// takes `K1Route::G` out of it, and `test_moments_route` and
/// `pls3_signal_test::cells_route_eligible` to false, which takes the moment
/// arms and PLS3's cell route out of it. Always `false` outside the test-only
/// `with_gram_routes_disabled`. Thread-local: a selector runs on the
/// calling thread before any parallel work, so a test's override reaches
/// exactly the calls that test makes. The override never forces a route
/// onto an input the route's rule rejects; it only removes `Nspace`,
/// `GramP`, `K1Route::G`, the moment arms and PLS3's cell route from the choice. With `PLSKIT_NUM_THREADS` set to a cap, a
/// public call runs on a plskit pool worker, which this thread-local does
/// not reach, so `with_gram_routes_disabled` panics when the variable sets
/// a cap.
#[cfg(test)]
pub(crate) fn gram_routes_disabled() -> bool {
    GRAM_ROUTES_DISABLED.with(std::cell::Cell::get)
}

/// Builds without unit tests have no override: the Gram routes are always
/// in the choice.
#[cfg(not(test))]
pub(crate) const fn gram_routes_disabled() -> bool {
    false
}

/// Run `f` with [`gram_routes_disabled`] true on this thread, restoring the
/// previous value afterwards, also when `f` panics, so nested and failing
/// tests cannot leak the override. Route-invisibility tests and the layout
/// table run primal-only arms inside it, and the k = 1 no-refit body then
/// streams X (`K1Route::A` or `K1Route::B`) instead of reading the all-row
/// Gram matrix (`K1Route::G`).
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
    /// `raw_perm` and `split_exact`), every unit decided from `G` alone.
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
/// input the n-space rule admits (`dual_route::nspace_eligible_split_exact`,
/// which counts the all-row Gram matrix as free when the call holds the
/// gate's, `gram_held`) takes `Nspace`, unless `gram_routes_disabled()`.
/// `GramP` when `gram_p::gram_p_eligible` admits the block (tried after
/// `Nspace`: in the overlap band `n_tr < p < 1.54·n_tr` the n-space route is
/// cheaper), unless `gram_routes_disabled()`. Everything else is `Primal`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_exact_refit_route(
    n: usize,
    p: usize,
    n_perm: usize,
    n_splits: usize,
    k: usize,
    keep: Option<usize>,
    weighted: bool,
    gram_held: bool,
) -> ReplicateRoute {
    if split_exact_no_refit_route(k, keep) {
        return ReplicateRoute::Special;
    }
    if gram_routes_disabled() {
        return ReplicateRoute::Primal;
    }
    let (n_train, _) = crate::resample::split_sizes(n, k);
    let dense_unweighted = keep.is_none() && !weighted;
    if crate::dual_route::nspace_eligible_split_exact(
        n,
        p,
        n_perm,
        n_splits,
        k,
        dense_unweighted,
        gram_held,
    ) {
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
    ks: &[usize],
    n_perm: usize,
    n_folds: usize,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    opts: &ConfirmatoryTestOpts,
    null_map: Option<crate::resample::NullMap<'_>>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<SetResult> {
    let n = x.nrows();

    // Weights are passed through to the fold driver, which re-normalizes
    // per fold. Column c >= 1 of `cols` is `permutation_from_seed(n,
    // seeds[c - 1])` applied to y, then `null_map`, so every route, and any
    // loop order, sees the same null columns at a given seed.
    let (folds, seeds) = raw_perm_draws(n, n_folds, n_perm, rng);
    let cols = crate::resample::Columns {
        y: y.as_mat(),
        seeds: &seeds,
        null_map,
    };

    // Route choice: `raw_perm_route` owns the rule, at the largest k, so a
    // set of more than one k never takes the K = 1 closed form. Decided
    // here, on the calling thread, before any parallel work.
    let route = raw_perm_route(
        n,
        n_folds,
        x.ncols(),
        n_perm,
        *ks.last().expect("ks is non-empty"),
        opts.keep,
        w_norm.is_some(),
    );
    let r2: Mat<f64> = match route {
        // The K = 1 closed form on the call's all-row Gram matrix.
        ReplicateRoute::Special => {
            // `y_mat` is not covered by the dual route's `n_tr` cap (see
            // `DUAL_ROUTE_MAX_N_TR`), which bounds only the Gram matrix.
            let y_mat = cols.matrix(0, cols.len());
            one_row(&crate::dual_route::pls1_cv_r2_columns(
                &block_gram(x, scale, None),
                y_mat.as_ref(),
                &folds,
            )?)
        }
        // Folds outer, replicate columns inner: each fold's X side (gather,
        // centre, scale, √w) is built once instead of once per replicate.
        // The folds run in batches the memory budget allows, the
        // n_perm + 1 columns of a fold in parallel. Permuted y rows;
        // weights stay tied to row indices (not permuted). Null column c
        // regenerates its permutation from seeds[c - 1] inside each unit,
        // so no n_perm·n buffer is held.
        route @ (ReplicateRoute::Primal | ReplicateRoute::Nspace | ReplicateRoute::GramP) => {
            pooled_cv_r2_columns(
                route,
                x,
                &folds,
                w_norm,
                scale,
                ks,
                opts.keep,
                cols.len(),
                &|c| cols.column(c),
            )?
        }
    };
    set_result(&r2, n_perm, raw_perm_result)
}

/// `raw_perm`'s draws, in order: the folds (`n` rows shuffled once, then
/// split) and one child seed per null column, drawn before any route is
/// chosen.
fn raw_perm_draws(
    n: usize,
    n_folds: usize,
    n_perm: usize,
    rng: &mut crate::rng::Rng,
) -> (Vec<Vec<usize>>, Vec<u64>) {
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    let folds = crate::linalg::fold_split(&indices, n_folds);
    (folds, crate::rng::child_seeds(rng, n_perm))
}

/// `raw_perm`'s p-value from the pooled CV R² of every column, column 0
/// the observed `y`.
fn raw_perm_result(r2: &[f64], n_perm: usize) -> PlsKitResult<RunResult> {
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

/// Single-column [`pooled_cv_r2_columns`] on the primal route, at the
/// all-row scale `run_raw_perm` divides by without `pre_standardized`: the
/// primal fold loop that the Gram routes and the route tests compare
/// against.
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
        crate::linalg::held_out_scale(x, weights, false).as_ref(),
        &[k],
        keep,
        1,
        &|_| y.to_owned(),
    )?;
    Ok(r2[(0, 0)])
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
/// `pls1_fit(pre_standardized, !check_n_eff, Seq, w_tr)`, centred and scaled
/// training and validation blocks plus the fold's renormalized weights and
/// the √w'' scaling `pls1_fit` applies, everything that depends on X, the
/// folds and the weights but not on the outcome.
pub(crate) struct CvFold {
    /// Training rows (every fold but this one, in fold order).
    pub(crate) train_idx: Vec<usize>,
    /// This fold's rows.
    pub(crate) val_idx: Vec<usize>,
    /// Training block centred with its own means (weighted when weighted)
    /// and divided by the one column scale over all rows; no √w.
    pub(crate) xs_tr: Mat<f64>,
    /// Validation block centred with the training means and divided by the
    /// same scale; no √w.
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

/// The training rows of fold `fi`: every fold but `fi`, in fold order.
pub(crate) fn fold_train_rows(folds: &[Vec<usize>], fi: usize) -> Vec<usize> {
    folds
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != fi)
        .flat_map(|(_, f)| f.iter().copied())
        .collect()
}

/// The unweighted all-row Gram matrix whose blocks the n-space route and
/// the K = 1 fold closed form read: `held` when the call holds it (the
/// `split_nb` gate's, `HeldOutGram::from_x` of `x` and `scale`), else one
/// built from `x` and `scale`. Unweighted, every half of a finite X on the
/// all-row scale is finite (`linalg::requarter` centres without
/// overflow, and `linalg::held_out_scale` keeps a flagged X within `2^±256`),
/// and so is `G`.
pub(crate) fn block_gram<'a>(
    x: MatRef<'_, f64>,
    scale: ColRef<'_, f64>,
    held: Option<&'a HeldOutGram>,
) -> std::borrow::Cow<'a, HeldOutGram> {
    use std::borrow::Cow;
    held.map_or_else(
        || Cow::Owned(HeldOutGram::from_x(x, None, scale)),
        Cow::Borrowed,
    )
}

/// Prepare fold `fi` of `folds` once. `weights` are the caller's mean-one
/// weights; the fold slices them to its training rows and renormalizes
/// (no weights when the slice is all zero or all equal), and `pls1_fit` would renormalize
/// that again before its √w scaling, which `sqw_fit` reproduces. Both
/// blocks are centred with the training rows' means (weighted with the
/// fold's weights when weighted) and divided by `scale`, the one column
/// scale over all rows (`linalg::held_out_scale`); nothing is scaled per
/// fold. The X-side check of `pls1_fit` (a finite training block) runs here,
/// once per fold on every route, instead of once per column, and so does
/// the same check on the validation block, whose non-finite entries would
/// make every prediction of the fold non-finite.
///
/// # Errors
/// `NonFiniteInput` when the centred and scaled training block is not finite:
/// the error every column's fit on this fold met first (`pls1_fit` checks X
/// before y), so the driver returns the same error a per-column fit of
/// column 0 would. `NonFiniteInput` too when the validation block is not
/// finite.
pub(crate) fn prepare_cv_fold(
    x: MatRef<'_, f64>,
    folds: &[Vec<usize>],
    fi: usize,
    weights: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
) -> PlsKitResult<CvFold> {
    use crate::fit::{check_finite_mat, fit_row_scale, scale_rows};
    use crate::linalg::{center_rows_fro, standardize_apply_rows};
    let val_idx = folds[fi].clone();
    let train_idx = fold_train_rows(folds, fi);
    let w_tr = renormalized_slice_weights(weights, &train_idx);
    // Without weights, `xs_tr`, `xs_val` and `x_fro` are mirrored by the
    // materialized fold of `dual_route::pls1_cv_r2_columns`, and the checks
    // of `xs_tr` and `xs_val` by that function's first pass (change
    // together).
    let (xs_tr, x_mean, _) =
        center_rows_fro(x, &train_idx, w_tr.as_ref().map(Col::as_ref), None, scale);
    check_finite_mat(xs_tr.as_ref())?;
    let xs_val = standardize_apply_rows(x, &val_idx, x_mean.as_ref(), scale, None);
    check_finite_mat(xs_val.as_ref())?;
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
/// where and how `pls1_fit` would. Per k of `ks` (sorted ascending,
/// non-empty): one fit at `ks.last()`, read at each k through [`per_k`];
/// element i is the pair at `ks[i]`.
///
/// # Errors
/// What `pls1_fit` returns for this fold's arrays at `ks.last()`.
#[allow(clippy::similar_names)]
pub(crate) fn cv_fold_contribution(
    fold: &CvFold,
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
    keep: Option<usize>,
) -> PlsKitResult<Vec<(f64, f64)>> {
    use crate::fit::{ParChoice, check_fit_y_and_k, pls1_fit_prepared_fro, scale_col};
    let k = *ks.last().expect("ks is non-empty");
    let (ys_tr, ys_val) = cv_fold_targets(fold, y_of);

    check_fit_y_and_k(fold.xs_tr.ncols(), ys_tr.as_ref(), k, keep)?;
    // Seq: a fold's columns run in parallel, one fit each. The Gram-p arm's
    // first product (`raw_perm_unit`) takes the same `ParChoice`.
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

    Ok(per_k(
        ks,
        (&fit.w_star, &fit.p_loadings, &fit.q_loadings, &fit.coef),
        Par::Seq,
        |coef| {
            let y_pred = crate::linalg::mat_vec(fold.xs_val.as_ref(), coef, Par::Seq);
            cv_fold_ss(&y_pred, &ys_val)
        },
    ))
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

/// The Gram-p arm of [`fold_unit`]: one (fold, column) from the fold's
/// p-space Gram block, with no read of X beyond the GEMV `s₁`. `gram` is
/// built on [`fold_fit_matrix`]; the fit target is the fold's standardized
/// training y, `scale_col`-scaled by `sqw_fit` when weighted, the arrays the
/// Primal arm fits, and the Primal arm's per-column checks run first, so a
/// target it refuses gives its error here too; invalid `k` or `keep` never
/// reach this arm (`gram_p::gram_p_eligible`). The validation predictions
/// are a sequential product, so the unit's bits do not depend on the
/// thread count. Per k of `ks` as [`cv_fold_contribution`]: one fit at
/// `ks.last()`, read at each k through [`per_k`].
///
/// # Errors
/// What `fit::check_fit_y_and_k` returns for the fold's training target.
pub(crate) fn raw_perm_unit(
    gram: &crate::gram_p::GramPBlock,
    fold: &CvFold,
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
    keep: Option<usize>,
) -> PlsKitResult<Vec<(f64, f64)>> {
    let k = *ks.last().expect("ks is non-empty");
    debug_assert!(ks[0] >= 1);
    let (ys_tr, ys_val) = cv_fold_targets(fold, y_of);
    crate::fit::check_fit_y_and_k(fold.xs_tr.ncols(), ys_tr.as_ref(), k, keep)?;
    let ys_fit = match fold.sqw_fit.as_ref() {
        Some(s) => crate::fit::scale_col(ys_tr.as_ref(), s.as_ref()),
        None => ys_tr,
    };
    // `Seq`: the `ParChoice` of the Primal arm's fit, `cv_fold_contribution`.
    let s1 = crate::gram_p::first_product(
        fold_fit_matrix(fold),
        ys_fit.as_ref(),
        crate::fit::ParChoice::Seq,
        k,
    );
    let fit = gram.fit(s1.as_ref(), ys_fit.norm_l2(), k, keep);
    Ok(per_k(
        ks,
        (&fit.w, &fit.p, &fit.q, &fit.coef),
        Par::Seq,
        |coef| {
            let y_pred = crate::linalg::mat_vec(fold.xs_val.as_ref(), coef, Par::Seq);
            cv_fold_ss(&y_pred, &ys_val)
        },
    ))
}

/// Per-fold state of a route on a prepared fold, built once per fold by
/// [`pooled_cv_r2_columns`] after the fold's preparation and shared
/// read-only by that fold's columns. The primal route needs nothing beyond
/// the prepared fold; the n-space route reads no prepared fold
/// ([`fold_columns_nspace`]).
pub(crate) enum FoldBlock<'f> {
    /// No per-fold state.
    Primal(std::marker::PhantomData<&'f ()>),
    /// The fold's p-space Gram block on [`fold_fit_matrix`].
    GramP(crate::gram_p::GramPBlock),
}

/// The block of `route` for one prepared fold, its precompute run under
/// `par` ([`crate::fit::par_fixed`]). `Special` and `Nspace` never reach
/// it; handed one, they get the primal block. `GramP` builds `C` on
/// [`fold_fit_matrix`] once per fold.
pub(crate) fn fold_block(route: ReplicateRoute, fold: &CvFold, par: Par) -> FoldBlock<'_> {
    match route {
        ReplicateRoute::GramP => {
            FoldBlock::GramP(crate::gram_p::GramPBlock::new(fold_fit_matrix(fold), par))
        }
        ReplicateRoute::Special | ReplicateRoute::Nspace | ReplicateRoute::Primal => {
            FoldBlock::Primal(std::marker::PhantomData)
        }
    }
}

/// One (fold, column) unit: this fold's `(ss_res, ss_tot)` for the outcome
/// `y_of` at each k of `ks`. The primal arm is [`cv_fold_contribution`],
/// the Gram-p arm [`raw_perm_unit`].
///
/// # Errors
/// What `cv_fold_contribution` returns.
pub(crate) fn fold_unit(
    block: &FoldBlock<'_>,
    fold: &CvFold,
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
    keep: Option<usize>,
) -> PlsKitResult<Vec<(f64, f64)>> {
    match block {
        FoldBlock::Primal(_) => cv_fold_contribution(fold, y_of, ks, keep),
        FoldBlock::GramP(gram) => raw_perm_unit(gram, fold, y_of, ks, keep),
    }
}

/// Fold `fi` of `folds` on the n-space route, for the `n_cols` outcome
/// columns `column_y(c)` (raw, length n): the fold's block from the
/// unweighted all-row Gram matrix `g`
/// ([`NspaceFold::from_gram`](crate::dual_route::NspaceFold::from_gram)),
/// then each column through `dual_route::fold_column_nspace`. No half of X
/// is written. The columns map in parallel. `g` is finite. Each column's
/// result holds one pair per k of `ks`, from one kernel run at `ks.last()`.
pub(crate) fn fold_columns_nspace(
    folds: &[Vec<usize>],
    fi: usize,
    g: &HeldOutGram,
    ks: &[usize],
    n_cols: usize,
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
) -> Vec<PlsKitResult<Vec<(f64, f64)>>> {
    let (train_idx, val_idx) = (fold_train_rows(folds, fi), folds[fi].as_slice());
    let nf = crate::dual_route::NspaceFold::from_gram(g, &train_idx, val_idx);
    crate::resample::map_indexed(n_cols, |c| {
        let yc = column_y(c);
        crate::dual_route::fold_column_nspace(&nf, &train_idx, val_idx, &|i| yc[i], ks)
    })
}

/// The large buffers one fold unit of [`pooled_cv_r2_columns`] holds on
/// `route`, for X of `n × p`. On `Nspace` only the fold's block, `G`
/// (`n_tr × n_tr`) and `M` (`n_val × n_tr`); the all-row Gram matrix it
/// reads is one buffer per call and is not counted. Otherwise the
/// standardized blocks `xs_tr` and `xs_val` (`resample::fold_blocks_bytes`),
/// the √w-scaled copy `xs_tr_fit` of the training block when `weighted`,
/// the fold's block (`C`, `p × p`, on `GramP`) and one PLS1 fit per column
/// that runs at once ([`columns_bytes`]). `n_tr` is the largest training
/// set.
fn fold_unit_bytes(
    route: ReplicateRoute,
    n: usize,
    p: usize,
    folds: &[Vec<usize>],
    weighted: bool,
    k: usize,
    n_cols: usize,
) -> usize {
    use crate::resample::{copy_bytes, fold_blocks_bytes};
    let n_tr = n - folds.iter().map(Vec::len).min().unwrap_or(0);
    let block = match route {
        ReplicateRoute::Nspace => return fold_blocks_bytes(n, folds, n_tr),
        ReplicateRoute::GramP => copy_bytes(p, p),
        ReplicateRoute::Special | ReplicateRoute::Primal => 0,
    };
    let fit_copy = if weighted { copy_bytes(n_tr, p) } else { 0 };
    let fits = match route {
        ReplicateRoute::Primal | ReplicateRoute::GramP => {
            columns_bytes(crate::fit::pls1_fit_bytes(p, k), n_cols)
        }
        ReplicateRoute::Special | ReplicateRoute::Nspace => 0,
    };
    fold_blocks_bytes(n, folds, p)
        .saturating_add(fit_copy)
        .saturating_add(block)
        .saturating_add(fits)
}

/// One fold unit's result in [`pooled_cv_r2_columns`]: the fold's
/// preparation error, or each column's `(ss_res, ss_tot)` per k, or error.
type FoldContributions = PlsKitResult<Vec<PlsKitResult<Vec<(f64, f64)>>>>;

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
/// Folds outer and columns inner: the folds run in batches of
/// `resample::map_units`, each counted by [`fold_unit_bytes`], so at most
/// `resample::units_at_once` prepared folds and their blocks are alive at
/// once. Each fold is prepared once ([`prepare_cv_fold`], where the X
/// check runs, centring the fold and dividing by `scale`, the one column
/// scale over all rows), its block is built once ([`fold_block`] under
/// [`crate::fit::par_fixed`]), and the columns map through [`fold_unit`]
/// in parallel, so no `B·n` outcome buffer is held. On `Nspace` the call
/// builds one all-row Gram matrix (`HeldOutGram::from_x` on `x` and
/// `scale`) and each fold runs [`fold_columns_nspace`], which prepares no
/// half of X; `column_y(c)` builds
/// column `c` (length n, raw) inside the unit that needs it. Every column
/// is computed in every fold; the results are then read in fold order and
/// `ss_res` and `ss_tot` accumulate per column from `0.0`, exactly as the
/// single-column loop does. A failed fold preparation, or a failed fit of
/// column 0, returns that error (the first in fold order, column 0 before
/// the later folds); a failed fit of any other column makes that column
/// NaN, and its results in the later folds are ignored.
///
/// Row i of the result is the pooled CV R² at `ks[i]`, column c is column
/// c's. `ks` is sorted ascending, non-empty, every entry at least 1, and
/// `ks.last() <= p`: the caller's preconditions (`confirmatory_test_run`
/// checks them). Each (fold, column) unit fits once at `ks.last()` and
/// reads every k from that fit; the buffers are counted at `ks.last()`,
/// and a failed fit fails the column at every k.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pooled_cv_r2_columns(
    route: ReplicateRoute,
    x: MatRef<'_, f64>,
    folds: &[Vec<usize>],
    weights: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    ks: &[usize],
    keep: Option<usize>,
    n_cols: usize,
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
) -> PlsKitResult<Mat<f64>> {
    let k = *ks.last().expect("ks is non-empty");
    debug_assert!(
        k <= x.ncols(),
        "ks.last() <= p is the caller's precondition"
    );
    let par = crate::fit::par_fixed();
    let gram = matches!(route, ReplicateRoute::Nspace).then(|| {
        debug_assert!(weights.is_none() && keep.is_none());
        block_gram(x, scale, None)
    });
    let unit_bytes = fold_unit_bytes(
        route,
        x.nrows(),
        x.ncols(),
        folds,
        weights.is_some(),
        k,
        n_cols,
    );
    // One unit per fold: the fold's preparation error, or one result per
    // column, every column computed.
    let per_fold: Vec<FoldContributions> =
        crate::resample::map_units(folds.len(), unit_bytes, |fi| {
            if let Some(g) = &gram {
                return Ok(fold_columns_nspace(folds, fi, g, ks, n_cols, column_y));
            }
            let fold = prepare_cv_fold(x, folds, fi, weights, scale)?;
            let block = fold_block(route, &fold, par);
            Ok(crate::resample::map_indexed(n_cols, |c| {
                let yc = column_y(c);
                fold_unit(&block, &fold, &|i| yc[i], ks, keep)
            }))
        });
    pool_fold_contributions(per_fold, n_cols, ks.len())
}

/// The pooled CV R² of `n_cols` columns at `nk` values of k (`nk × n_cols`)
/// from each fold's preparation result or per-column `(ss_res, ss_tot)`
/// per k, the rules of [`pooled_cv_r2_columns`] applied in fold order, then
/// column order within a fold.
fn pool_fold_contributions(
    per_fold: Vec<FoldContributions>,
    n_cols: usize,
    nk: usize,
) -> PlsKitResult<Mat<f64>> {
    // The rules applied in fold order, then column order within a fold, so
    // the first error is the one a fold-by-fold loop meets first.
    let mut ss_res = Mat::<f64>::zeros(nk, n_cols);
    let mut ss_tot = Mat::<f64>::zeros(nk, n_cols);
    let mut failed = vec![false; n_cols];
    for contrib in per_fold {
        for (c, r) in contrib?.into_iter().enumerate() {
            if failed[c] {
                continue;
            }
            match r {
                Ok(pairs) => {
                    debug_assert_eq!(pairs.len(), nk);
                    for (i, (res, tot)) in pairs.into_iter().enumerate() {
                        ss_res[(i, c)] += res;
                        ss_tot[(i, c)] += tot;
                    }
                }
                Err(e) if c == 0 => return Err(e),
                Err(_) => failed[c] = true,
            }
        }
    }
    Ok(Mat::<f64>::from_fn(nk, n_cols, |i, c| {
        if failed[c] {
            f64::NAN
        } else if ss_tot[(i, c)] > 0.0 {
            1.0 - ss_res[(i, c)] / ss_tot[(i, c)]
        } else {
            0.0
        }
    }))
}

// ──────────────────────────────────────────────────────────────────────────────
// `split_nb` and `split_exact`'s refit route (`run_split_perm`)
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) use crate::resample::SplitIdx;

/// `resample::pair_blocks_bytes` for the splits of a split loop: one block
/// on the training rows and one on the test rows (the standardized halves
/// of X at `cols = p`; `G` and `M` at `cols = n_tr`). Mirrored by the
/// `split_blocks` closure in `tests/memory_budget_parity.rs` (change
/// together).
pub(crate) fn split_blocks_bytes(splits: &[SplitIdx], cols: usize) -> usize {
    crate::resample::pair_blocks_bytes(splits.iter().map(|sp| (sp.tr.len(), sp.te.len())), cols)
}

/// One fit's buffers, `fit_bytes` (`fit::pls1_fit_bytes` for a PLS1 fit,
/// `pls3::lv1_bytes` for a PLS3 LV1 fit), per replicate column of a unit
/// that runs at once. The columns map in parallel, so at most one per thread
/// of the pool runs: a PLS3 LV1 fit runs sequentially and does not yield its
/// thread, and the PLS1 split columns, whose fit can use the pool itself,
/// map through [`map_split_columns`]. Counted per unit: a batch of several
/// units shares the threads and holds less. The thread count sets only how
/// many units run at once, never a result.
pub(crate) fn columns_bytes(fit_bytes: usize, n_cols: usize) -> usize {
    n_cols
        .min(rayon::current_num_threads())
        .saturating_mul(fit_bytes)
}

/// Map the `n_cols` replicate columns of one split whose per-column work
/// is [`split_column_r`] on a training half of `n_tr × p` to `k`
/// components, or the Gram-p arm, which forms its first product with the
/// same `Par`. Past the `ParChoice::Auto` threshold
/// that fit uses the pool, and a worker waiting inside it can start another
/// column of the same map, so the columns run in batches of the pool's
/// thread count, the bound [`columns_bytes`] counts. Below it the fit is
/// sequential and the columns run in one map. Element `c` is `f(c)` either
/// way.
pub(crate) fn map_split_columns<T: Send>(
    n_tr: usize,
    p: usize,
    k: usize,
    n_cols: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let batch = match crate::fit::resolve_par(SPLIT_FIT_PAR, n_tr, p, k) {
        Par::Seq => n_cols,
        Par::Rayon(_) => rayon::current_num_threads(),
    };
    crate::resample::map_batches(n_cols, batch, f)
}

/// Per-column `z̄` over the J splits: `per_split` maps one split to its
/// `n_cols` clamped `atanh(r)` values, and column `c` of the result is
/// their mean over the splits.
///
/// The one accumulation behind the splits-outer `split_exact` routes that
/// map their splits through it (PLS1's `split_zbars_columns`, PLS3's
/// `pls3_split_zbars_columns_primal` and `dual_route::pls3_split_zbars_columns`),
/// shared with the route that computes its rows in two passes
/// (`split_perm_nr_zbars`) so that none of them can drift
/// apart in the last bits. The splits may be mapped in
/// parallel, but the sum runs over them **ascending** and divides by J
/// exactly once at the end: `mean_fisher_z`'s left-to-right
/// sum-then-divide, column by column. Averaging `r` first,
/// dividing inside the loop, or summing in completion order would not
/// reproduce it. The sum itself is [`zbars_from_rows`], which the two-pass
/// routes call on their own rows.
///
/// The splits run in batches of `resample::map_units` with `unit_bytes`
/// (the large buffers `per_split` builds for one split), so at most
/// `resample::units_at_once` prepared splits are alive at once;
/// `per_split` maps its columns in parallel itself. The rows come back in
/// split order whatever the batch size.
pub(crate) fn zbars_over_splits<F>(
    splits: &[SplitIdx],
    n_cols: usize,
    unit_bytes: usize,
    per_split: F,
) -> Vec<f64>
where
    F: Fn(&SplitIdx) -> Vec<f64> + Sync,
{
    let rows = crate::resample::map_units(splits.len(), unit_bytes, |j| per_split(&splits[j]));
    zbars_from_rows(n_cols, &rows)
}

/// The accumulation of [`zbars_over_splits`] on rows already computed:
/// `per_split_z[j]` holds split `j`'s `n_cols` clamped `atanh(r)` values,
/// and column `c` of the result is their mean over the splits, summed
/// ascending and divided by J once.
pub(crate) fn zbars_from_rows(n_cols: usize, per_split_z: &[Vec<f64>]) -> Vec<f64> {
    let mut z_sum = vec![0.0_f64; n_cols];
    for per in per_split_z {
        for (col, v) in per.iter().enumerate() {
            z_sum[col] += v;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let j = per_split_z.len() as f64;
    z_sum.into_iter().map(|s| s / j).collect()
}

/// Draw the J split-half partitions used by `split_nb`, by both `split_exact`
/// routes and by `pls3_signal_test::pls3_confirmatory_test`.
///
/// Split fraction is hardcoded 50/50: NB calibration assumes balanced halves
/// and there is no scientific reason to vary it.
///
/// The splits come from the balanced design over random cells of rows
/// (`resample::balanced_splits`). When n ≥ H (m > 0; n ≥ 64 at the default
/// J = 32), every pair of splits overlaps in exact quarters on the cell
/// rows; below that there are no cells and the splits are independent
/// random halves. Either way each split is a uniformly random half.
/// The parent state gives one child seed and the whole design is drawn off
/// that child, so the parent advances by one draw whatever J is, and every
/// caller gets the same splits for the same parent state.
pub(crate) fn draw_splits(
    n: usize,
    k: usize,
    n_splits: usize,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<Vec<SplitIdx>> {
    Ok(draw_design(n, k, n_splits, rng)?.splits)
}

/// The balanced design whose splits [`draw_splits`] returns, from the same
/// draws: the routes that sum their training rows from the design's cells
/// (`crate::cell_sums`) hold the design itself.
pub(crate) fn draw_design(
    n: usize,
    k: usize,
    n_splits: usize,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<crate::resample::SplitDesign> {
    use crate::resample::{balanced_splits, split_sizes};

    // n-3 test-floor in split_sizes can drop n_train below k+2; below that the
    // per-half fit degrades to a silent r=0. Reject up front. See split_sizes.
    // Mirrored in run_e (change together).
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, _) = split_sizes(n, k);
    let seed = crate::rng::child_seeds(rng, 1)[0];
    Ok(balanced_splits(
        n,
        n_train,
        n_splits,
        &mut crate::rng::child_rng(seed),
    ))
}

/// Compute the split-half Pearson r on each supplied split.
///
/// Takes the splits rather than drawing them so a caller can hold one set of
/// splits fixed across many outcome vectors — what `split_exact`'s permutation
/// loop needs. Nothing here consumes randomness: given `(x, y, split)` the fit
/// and the r are deterministic. The J splits run in batches of
/// `resample::map_units` by the memory budget, one unit being the split's
/// prepared halves of X ([`prepare_split`] writes the √w scaling into
/// them, so weights add no copy) and one fit's buffers; the result does not
/// depend on the batch size.
///
/// Its only production caller is `run_split_nb`, for K ≥ 2 or a sparse
/// `keep` when the call does not run the n-space route from the gate's Gram
/// matrix; `split_exact`'s refit route (`run_split_perm`) reaches the same
/// per-split arithmetic through `split_zbars_columns` instead.
///
/// `x`/`y` are the **raw** (un-√w-scaled) inputs; `w_norm` carries the
/// mean-1-normalized weights when present. Each half is centred with the
/// training half's **weighted** means, divided by `scale` (the one column
/// scale over all rows, see [`prepare_split`]) and then √w' row-scaled (the
/// √w row-scaling convention of `pls1_fit`, see `linalg::sqrt_col`). The reported
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
    scale: ColRef<'_, f64>,
    keep: Option<usize>,
) -> Col<f64> {
    // The splits run in batches by the memory budget, collected in split
    // order; a split holds its halves of X and one fit.
    let unit = split_blocks_bytes(splits, x.ncols())
        .saturating_add(crate::fit::pls1_fit_bytes(x.ncols(), k));
    let r_vec = crate::resample::map_units(splits.len(), unit, |j| {
        split_half_r(x, y, k, &splits[j], w_norm, scale, keep)
    });

    Col::<f64>::from_fn(r_vec.len(), |i| r_vec[i])
}

/// One split of `split_exact`'s refit route and of `split_nb` at K ≥ 2 or
/// with a sparse `keep`, prepared once: the X side of
/// [`split_half_r`], which depends on X, the split and the weights but not
/// on the outcome, so a permutation loop can build it once per split.
pub(crate) struct PreparedSplit {
    /// Training half centred with its own means and divided by the all-row
    /// column scale, √w_tr-scaled when weighted.
    pub(crate) xs_tr: Mat<f64>,
    /// Test half centred with the training means and divided by the same
    /// scale, √w_te-scaled when weighted.
    pub(crate) xs_te: Mat<f64>,
    /// Per-half renormalized weights (`split_half_r`'s `w_tr`).
    pub(crate) w_tr: Option<Col<f64>>,
    /// Per-half renormalized weights (`split_half_r`'s `w_te`).
    pub(crate) w_te: Option<Col<f64>>,
    /// Whether `xs_tr` and `xs_te` are finite: the X check `pls1_fit` makes
    /// on the training half, and the same check on the held-out half, whose
    /// non-finite entries would make every score non-finite. Done once per
    /// split instead of once per column. When false, every column's `r` is
    /// 0, as the failed per-half fit gave. Only weights can make it false: a
    /// row of zero weight far from the rest is outside the all-row scale.
    pub(crate) x_finite: bool,
    /// `xs_tr.norm_l2()`, taken once per split rather than once per column:
    /// the same call on the same view, so the same bits.
    pub(crate) x_fro: f64,
}

/// The X side of [`split_half_r`] for split `sp`. Per-half weights are the
/// caller's `w_norm` sliced to the half and renormalized to mean one
/// (mirrors `signal_test::pooled_cv_r2_columns`'s per-fold renormalization),
/// with no weights for an all-zero or all-equal half. Each half is centred
/// with the training half's weighted means, divided by `scale` (the one
/// column scale over all rows, `linalg::held_out_scale`; nothing is scaled
/// per half) and then √w-scaled with its own half's weights as given (the
/// √w row-scaling convention of `pls1_fit`), in one pass per half
/// (`linalg::center_rows_fro`).
#[allow(clippy::similar_names)]
pub(crate) fn prepare_split(
    x: MatRef<'_, f64>,
    sp: &SplitIdx,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
) -> PreparedSplit {
    use crate::linalg::{center_rows_fro, sqrt_col, standardize_apply_rows};
    let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
    let w_tr = renormalized_slice_weights(w_norm, tr);
    let w_te = renormalized_slice_weights(w_norm, te);
    let sw_tr = w_tr.as_ref().map(|w| sqrt_col(w.as_ref()));
    let sw_te = w_te.as_ref().map(|w| sqrt_col(w.as_ref()));
    let (xs_tr, x_mean, _) = center_rows_fro(
        x,
        tr,
        w_tr.as_ref().map(Col::as_ref),
        sw_tr.as_ref().map(Col::as_ref),
        scale,
    );
    let xs_te = standardize_apply_rows(
        x,
        te,
        x_mean.as_ref(),
        scale,
        sw_te.as_ref().map(Col::as_ref),
    );
    let x_finite = crate::fit::check_finite_mat(xs_tr.as_ref()).is_ok()
        && crate::fit::check_finite_mat(xs_te.as_ref()).is_ok();
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

/// The `ParChoice` of [`split_column_r`]'s per-half fit, and of the first
/// product of the Gram-p arm that stands in for it
/// ([`split_column_r_gram_p`]): `Auto`, so
/// the fit picks its arm from its shape (`fit::resolve_par` on `n_tr`, `p`,
/// `k`) and the bits do not depend on the batch or the pool.
pub(crate) const SPLIT_FIT_PAR: crate::fit::ParChoice = crate::fit::ParChoice::Auto;

/// [`split_half_r`] after its X side: standardize the training outcome
/// (weighted moments when weighted), √w_tr-scale it, fit on the prepared
/// training half, score the prepared test half, and correlate with the
/// √w_te-scaled test outcome through `guarded_pearson`. `y_of(i)` is the raw
/// outcome of full-data row `i`. The fit is `pls1_fit(xs_tr, ys_tr, k, None,
/// pre_standardized, !check_n_eff, SPLIT_FIT_PAR, keep)` with the X checks done once
/// in `prepare_split` (`x_finite`): the per-column checks in `pls1_fit`'s
/// order, then the kernel tail, which is all `pls1_fit` runs on unweighted
/// pre-standardized input. A failed per-half fit gives `r = 0`: a truncated
/// model still yields a valid `r` at `k_used`, which beats silently
/// recording 0 through the error arm, so there is no `n_eff` truncation check.
///
/// Per k of `ks` (sorted ascending, non-empty): one fit at `ks.last()`,
/// read at each k through [`per_k`]; every check and every `r = 0` exit is
/// taken at `ks.last()` and covers every k.
#[allow(clippy::similar_names)]
pub(crate) fn split_column_r(
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
    keep: Option<usize>,
) -> Vec<f64> {
    use crate::fit::{check_fit_y_and_k, pls1_fit_prepared_fro};
    let k = *ks.last().expect("ks is non-empty");
    let zeros = || vec![0.0; ks.len()];
    if !prep.x_finite {
        return zeros();
    }
    let ys_tr = split_train_target(prep, sp, y_of);
    if check_fit_y_and_k(prep.xs_tr.ncols(), ys_tr.as_ref(), k, keep).is_err() {
        return zeros();
    }
    let Ok(m) = pls1_fit_prepared_fro(
        prep.xs_tr.as_ref(),
        ys_tr.as_ref(),
        k,
        keep,
        SPLIT_FIT_PAR,
        prep.x_fro,
    ) else {
        return zeros();
    };
    let y_te = split_test_target(prep, sp, y_of);
    let n_te = y_te.nrows();
    // The `Par` the fit resolved, for the coefficients below `k_used`.
    let par = crate::fit::resolve_par(SPLIT_FIT_PAR, prep.xs_tr.nrows(), prep.xs_tr.ncols(), k);
    per_k(
        ks,
        (&m.w_star, &m.p_loadings, &m.q_loadings, &m.coef),
        par,
        |coef| {
            // Both the scores (through the √w_te-scaled X̃_te) and the test outcome
            // carry √w_te, so the Pearson r is taken on √w-scaled data (the √w
            // row-scaling convention).
            let scores_te = crate::linalg::mat_vec(prep.xs_te.as_ref(), coef, faer::Par::Seq);
            guarded_pearson(n_te, |i| scores_te[i], |i| y_te[i])
        },
    )
}

/// `value(coef_k)` for each k of `ks` (sorted ascending), `coef_k` the
/// coefficient on the first `k.min(k_used)` components of one PLS1 fit
/// `(W, P, Q, coef)` with `k_used = W.ncols()`: `coef` itself from
/// `k_used` on (so a one-element `ks` reads the fit's own coefficient, and
/// the k past `k_used` share one value), `fit::pls1_coef_at_k` under `par`
/// below it. A fit's first k components are the fit at k, so `coef_k` is
/// the coefficient a fit at k gives.
fn per_k<T: Copy>(
    ks: &[usize],
    (w, p, q, coef): (&Mat<f64>, &Mat<f64>, &Col<f64>, &Col<f64>),
    par: Par,
    value: impl Fn(ColRef<'_, f64>) -> T,
) -> Vec<T> {
    let k_used = w.ncols();
    let mut out: Vec<T> = Vec::with_capacity(ks.len());
    for (i, &k) in ks.iter().enumerate() {
        let v = if k < k_used {
            value(crate::fit::pls1_coef_at_k(w, p, q, k, par).as_ref())
        } else if i > 0 && ks[i - 1] >= k_used {
            out[i - 1]
        } else {
            value(coef.as_ref())
        };
        out.push(v);
    }
    out
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

/// The Gram-p arm of [`split_unit`]: [`split_column_r`] with the fit taken
/// from `gram`, the p-space Gram block of the training half `prep.xs_tr`
/// (`√w_tr`-scaled with the half's own weights when weighted): the same
/// target ([`split_train_target`]) and per-column checks, `s₁` the X
/// backend's GEMV under [`SPLIT_FIT_PAR`], then the test-half scores
/// `X̃_te·coef` (a sequential product, so the bits do not depend on the
/// thread count) and `guarded_pearson`, which gives `r = 0` for the zero
/// model's zero scores. No refit on X. `split_unit` answers a non-finite X
/// half before any arm runs; invalid `k` or `keep` never reach this arm
/// (`gram_p::gram_p_eligible`). Per k of `ks` as [`split_column_r`]: one
/// fit at `ks.last()`, read at each k through [`per_k`].
pub(crate) fn split_column_r_gram_p(
    gram: &crate::gram_p::GramPBlock,
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
    keep: Option<usize>,
) -> Vec<f64> {
    let k = *ks.last().expect("ks is non-empty");
    debug_assert!(ks[0] >= 1);
    let ys_tr = split_train_target(prep, sp, y_of);
    if crate::fit::check_fit_y_and_k(prep.xs_tr.ncols(), ys_tr.as_ref(), k, keep).is_err() {
        return vec![0.0; ks.len()];
    }
    let s1 = crate::gram_p::first_product(prep.xs_tr.as_ref(), ys_tr.as_ref(), SPLIT_FIT_PAR, k);
    let fit = gram.fit(s1.as_ref(), ys_tr.norm_l2(), k, keep);
    let y_te = split_test_target(prep, sp, y_of);
    per_k(ks, (&fit.w, &fit.p, &fit.q, &fit.coef), Par::Seq, |coef| {
        let scores_te = crate::linalg::mat_vec(prep.xs_te.as_ref(), coef, Par::Seq);
        guarded_pearson(y_te.nrows(), |i| scores_te[i], |i| y_te[i])
    })
}

/// The test half of one split read through its moments, for a split loop
/// that scores every outcome column of the split without reading its test
/// rows: `S_B = X̃_B'C_B X̃_B` (`p × p`), kept as the `R` factor of the
/// Householder QR of `C_B X̃_B` (`S_B = R_B'R_B`), and `X̃_B'1`, with
/// `X̃ = (x − 1·m')/scale` the call's X centred by its all-row column means
/// `m` ([`AllRowMoments`]) and divided by the all-row scale, `X̃_B` its test
/// rows and `C_B` the centring over them. Built once per split
/// ([`TestMoments::new`], about `n_B·p²` multiply-adds); [`r_from_moments`] then takes
/// a column's held-out correlation from `p`-vectors. The factor, not
/// `S_B`, is kept because `‖R_B·coef‖` loses accuracy in proportion to the
/// conditioning of the scores, as `‖C_B X̃_B coef‖` formed from the test
/// rows does, where `coef'S_B coef` loses it in proportion to its square.
pub(crate) struct TestMoments {
    r_b: Mat<f64>,
    xb_sum: Col<f64>,
    n_b: usize,
}

/// The norms of one column's test-half scores `X̃_B·coef` that the moment
/// arms' gates read ([`TestMoments::score_norms`]).
pub(crate) struct MomentScores {
    /// `‖C_B X̃_B coef‖ = ‖R_B·coef‖`.
    pub(crate) t_c: f64,
    /// `‖s‖` for the scores the refit route forms, `X̃_te·coef` with the test
    /// half centred by the training means: `√(t_c² + n_B·(μ_d'coef)²)`,
    /// `μ_d = X̃_B'1/n_B − X̃_A'1/n_A` the test half's mean in that frame.
    pub(crate) s_ref: f64,
    /// `‖X̃_B coef‖ = √(t_c² + n_B·(μ_B'coef)²)`, `μ_B = X̃_B'1/n_B`: the norm
    /// [`r_from_moments`]'s constant test reads.
    pub(crate) s_all: f64,
    /// `F_te = ‖X̃_B‖_F + √n_B·‖X̃_A'1‖/n_A`, at least `‖X̃_te‖_F` of the
    /// test half in the refit route's frame.
    pub(crate) f_te: f64,
    /// `F_te·‖coef‖`, at least the norm of every score vector above:
    /// `F_te·‖coef‖/t_c` is the conditioning of the centred scores.
    pub(crate) f_coef: f64,
    /// Bound on `|t_c − ‖C_B X̃_te coef‖|` from the moment arithmetic (see
    /// [`TestMoments::score_norms`]).
    pub(crate) err: f64,
}

impl TestMoments {
    /// The moments of rows `te` of `x` centred by `mean` and divided by
    /// `scale`: the rows as [`crate::linalg::standardize_apply_rows`] writes
    /// them (the entries `linalg::centered_xt_vec` sums), their column sums
    /// `X̃_B'1` in row order, then the rows centred by those sums over `n_B`
    /// in place and `R_B` from their QR ([`crate::linalg::qr_r_seq`]).
    pub(crate) fn new(
        x: MatRef<'_, f64>,
        te: &[usize],
        mean: ColRef<'_, f64>,
        scale: ColRef<'_, f64>,
    ) -> Self {
        let mut z = crate::linalg::standardize_apply_rows(x, te, mean, scale, None);
        let n_b = te.len();
        let xb_sum = Col::<f64>::from_fn(z.ncols(), |j| (0..n_b).map(|i| z[(i, j)]).sum());
        #[allow(clippy::cast_precision_loss)]
        let nf = n_b as f64;
        for j in 0..z.ncols() {
            let mu = xb_sum[j] / nf;
            for i in 0..n_b {
                z[(i, j)] -= mu;
            }
        }
        let r_b = crate::linalg::qr_r_seq(z);
        Self { r_b, xb_sum, n_b }
    }

    /// `‖R_B·coef‖²`, which is `coef'S_B coef`: the centred test-half
    /// scores' sum of squares, summed in index order.
    fn quad(&self, coef: ColRef<'_, f64>) -> f64 {
        let rc = crate::linalg::mat_vec(self.r_b.as_ref(), coef, Par::Seq);
        (0..rc.nrows()).map(|j| rc[j] * rc[j]).sum()
    }

    /// The score norms of `coef` the moment arms' gates read, with
    /// `x_sum = X̃'1` over all `n_A + n_B` rows (so `X̃_A'1 = x_sum − X̃_B'1`).
    ///
    /// # Rounding of `t_c`
    /// `err` bounds `|t_c − ‖C_B X̃_te coef‖|`, `X̃_te` the test half the
    /// refit route forms (centred by the training means), `ε` the machine
    /// epsilon and `F = ‖X̃_B‖_F ≤ F_te`:
    /// - The centring: `Z` is centred by its computed column sums, which are
    ///   off by at most `n_B·ε` times the mean of `|X̃_B|`, a column shift
    ///   `e` with `√n_B·‖e‖ ≤ n_B·ε·F`; it moves `‖Z·coef‖` by at most
    ///   `√n_B·|e'coef| ≤ n_B·ε·F·‖coef‖`.
    /// - The QR: Householder QR gives the exact `R` of `Z + ΔZ` with
    ///   `‖Δzⱼ‖ ≤ γ·‖zⱼ‖` column by column, `γ = 4·n_B·p·ε` (Higham, Accuracy
    ///   and Stability of Numerical Algorithms, Theorem 19.4, with its
    ///   constant taken as 4), so `‖R̂·coef‖` is `‖(Z + ΔZ)·coef‖` and moves
    ///   by at most `γ·F·‖coef‖`; the product `R̂·coef`, its sum of squares
    ///   and the root add `(p + 2)·ε·F·‖coef‖`. Both are linear in
    ///   `F·‖coef‖`, as the test-row arm's own `(p + 2)·ε·F·‖coef‖` is.
    /// - The frame: `X̃_B` here and `X̃_te` there differ by a column shift,
    ///   which `C_B` removes exactly, and by the rounding of their entries,
    ///   at most `ε·|x − m|/scale + ε·|x − m_A|/scale` each, `2ε·F_te` in
    ///   Frobenius norm.
    ///
    /// Hence `err = (n_B·(4p + 1) + p + 4)·ε·F_te·‖coef‖`. The arms add the
    /// rounding of their `coef` and of the refit route's own scores. A NaN
    /// anywhere makes `err` NaN, and every gate is an `a >= b` conjunction,
    /// so the column goes to the refit route.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn score_norms(
        &self,
        coef: ColRef<'_, f64>,
        x_sum: ColRef<'_, f64>,
        n_a: usize,
    ) -> MomentScores {
        let (nb, na) = (self.n_b as f64, n_a as f64);
        let p = coef.nrows();
        let t_c2 = self.quad(coef);
        let dot = |v: &dyn Fn(usize) -> f64| (0..p).map(|j| coef[j] * v(j)).sum::<f64>();
        let mu_b = dot(&|j| self.xb_sum[j]) / nb;
        let mu_a = dot(&|j| x_sum[j] - self.xb_sum[j]) / na;
        let xa_norm = (0..p)
            .map(|j| (x_sum[j] - self.xb_sum[j]).powi(2))
            .sum::<f64>()
            .sqrt();
        let f_te = self.x_fro() + nb.sqrt() * xa_norm / na;
        let f_coef = f_te * coef.norm_l2();
        let pf = p as f64;
        let gamma = nb.mul_add(4.0f64.mul_add(pf, 1.0), pf + 4.0) * f64::EPSILON;
        MomentScores {
            t_c: t_c2.sqrt(),
            s_ref: nb.mul_add((mu_b - mu_a).powi(2), t_c2).sqrt(),
            s_all: nb.mul_add(mu_b * mu_b, t_c2).sqrt(),
            f_te,
            f_coef,
            err: gamma * f_coef,
        }
    }

    /// `‖X̃_B‖_F = √(‖R_B‖_F² + ‖X̃_B'1‖²/n_B)`, `‖R_B‖_F² = tr S_B`.
    #[allow(clippy::cast_precision_loss)]
    fn x_fro(&self) -> f64 {
        let r = &self.r_b;
        let tr: f64 = (0..r.ncols())
            .map(|j| (0..r.nrows()).map(|i| r[(i, j)] * r[(i, j)]).sum::<f64>())
            .sum();
        let sq: f64 = (0..self.xb_sum.nrows())
            .map(|j| self.xb_sum[j].powi(2))
            .sum();
        (tr + sq / self.n_b as f64).sqrt()
    }
}

/// The held-out Pearson r of the scores `X̃_B·coef` and `y_B` from the test
/// half's moments `tm`: `coef'(X̃_B'y_B − X̃_B'1·ȳ_B) / √(coef'S_B coef ·
/// ‖C_B y_B‖²)`, clamped to `[-1, 1]`, with `xb_y = X̃_B'y_B`,
/// `y_b_sum = 1'y_B`, `ȳ_B = y_b_sum/n_B` and `y_b_ss = ‖C_B y_B‖²`. The
/// result does not change when one constant is subtracted from `y`
/// throughout, and the callers subtract the column's all-row mean, so that
/// the numerator does not cancel by the outcome's mean over its spread.
/// `guarded_pearson`'s zero for a constant side: exactly `0.0` when
/// `constant_to_rounding` reads the scores (`coef'S_B coef` against
/// `‖X̃_B coef‖² = coef'S_B coef + n_B·(μ_B'coef)²`, `μ_B = X̃_B'1/n_B`) or
/// `y_B` (`y_b_ss` against `y_b_ss + n_B·ȳ_B²`) as constant.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn r_from_moments(
    coef: ColRef<'_, f64>,
    xb_y: ColRef<'_, f64>,
    y_b_sum: f64,
    y_b_ss: f64,
    tm: &TestMoments,
) -> f64 {
    use crate::linalg::constant_to_rounding;
    let nb = tm.n_b as f64;
    let y_bar = y_b_sum / nb;
    let p = coef.nrows();
    let t_c2 = tm.quad(coef);
    let mu = (0..p).map(|j| coef[j] * tm.xb_sum[j]).sum::<f64>() / nb;
    if constant_to_rounding(t_c2, nb.mul_add(mu * mu, t_c2), tm.n_b)
        || constant_to_rounding(y_b_ss, nb.mul_add(y_bar * y_bar, y_b_ss), tm.n_b)
    {
        return 0.0;
    }
    let cross: f64 = (0..p)
        .map(|j| coef[j] * tm.xb_sum[j].mul_add(-y_bar, xb_y[j]))
        .sum();
    (cross / (t_c2.sqrt() * y_b_ss.sqrt())).clamp(-1.0, 1.0)
}

/// `(1'v, ‖C_B v‖²)` of the `n_b` values `v`, as [`r_from_moments`] takes
/// them: [`scaled_moments`]'s sums times its power of two (exact).
#[allow(clippy::cast_precision_loss)]
fn y_b_sums(n_b: usize, v: impl Fn(usize) -> f64) -> (f64, f64) {
    let my = scaled_moments(n_b, v, None);
    (my.mean * n_b as f64 * my.s, my.ss * my.s * my.s)
}

/// The mean of `n` values in index order, `0` for none.
#[allow(clippy::cast_precision_loss)]
fn plain_mean(n: usize, v: impl Fn(usize) -> f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    (0..n).map(v).sum::<f64>() / n as f64
}

/// The part of a moment arm's gate that does not depend on how its
/// `coef` was formed: the scores are conditioned well enough, and both
/// sides of the correlation clear [`r_from_moments`]'s and the refit route's
/// constant tests by `band`.
///
/// - Conditioning: `t_c ≥ SCORE_BAND·F_te·‖coef‖`, that is
///   `κ = F_te·‖coef‖/t_c ≤ 1/SCORE_BAND`. Both factors of the moment `r`
///   (`‖R_B·coef‖` and `coef'(X̃_B'C_B y_B)`) lose accuracy in proportion to
///   `κ`, so the two arms' `r` differ by about `η·κ`, `η` their
///   discrepancy per unit of `κ`. Measured on test halves whose `coef`
///   points into a nearly null direction of `C_B X̃_B`, `κ` from `6e1` to
///   `6e5`, `η` stays below `1e-16`
///   (`tests::moment_gate_sends_ill_conditioned_scores_to_the_test_rows`),
///   so past this gate `r` moves by at most about `1e-13`, inside the
///   `1e-12` bit-near tolerance. A larger `κ` goes to the test-row arm,
///   the reference.
/// - Constants: `t_c ≥ band·2n_B·ε·max(‖s‖_ref, ‖s‖)` and
///   `√y_b_ss ≥ band·2n_B·ε·√(y_b_ss + n_B·ȳ_B²)`. Past it, neither route
///   reads either side as constant (the refit route's test on `y_B` is the
///   same call on the same bits, and its scores are within `err` of `t_c`).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn moments_resolved(
    sn: &MomentScores,
    y_b_sum: f64,
    y_b_ss: f64,
    n_b: usize,
    band: f64,
) -> bool {
    let nb = n_b as f64;
    let tol = band * 2.0 * nb * f64::EPSILON;
    let y_bar = y_b_sum / nb;
    sn.t_c >= crate::dual_route::SCORE_BAND * sn.f_coef
        && sn.t_c >= tol * sn.s_ref.max(sn.s_all)
        && y_b_ss.sqrt() >= tol * nb.mul_add(y_bar * y_bar, y_b_ss).sqrt()
}

/// The all-row side of the moment arms, built once per call: the column
/// means `m` of X over all rows (summed in row order), `X̃'1`, and for each
/// outcome column `c` its all-row mean `y_shift[c]` and column `c` of the
/// `p × n_cols` matrix `X̃'(Y − 1·y_shift')`, for `X̃ = (x − 1·m')/scale`
/// (`linalg::centered_xt_vec`, each column summed in row order whichever
/// worker computes it). Every moment-arm quantity of `y` is formed on
/// `y − y_shift[c]`, which leaves the correlation unchanged and keeps the
/// sums from cancelling by the outcome's mean over its spread. A split's
/// `X̃_B'y = X̃'y − X̃_A'y` then needs only its training rows.
pub(crate) struct AllRowMoments {
    mean: Col<f64>,
    x_sum: Col<f64>,
    y_shift: Vec<f64>,
    xty: Mat<f64>,
}

/// `(max, min)` of every column of `x` over all rows.
pub(crate) fn column_extremes(x: MatRef<'_, f64>) -> Vec<(f64, f64)> {
    (0..x.ncols())
        .map(|j| {
            (0..x.nrows()).fold((f64::MIN, f64::MAX), |(hi, lo), i| {
                (hi.max(x[(i, j)]), lo.min(x[(i, j)]))
            })
        })
        .collect()
}

/// `‖X̃_tr‖_F` of the training rows `tr` as [`prepare_split`] gives it,
/// from their means without writing a half
/// (`linalg::centered_part_moments`). Unweighted.
fn train_fro(x: MatRef<'_, f64>, tr: &[usize], scale: ColRef<'_, f64>) -> f64 {
    crate::linalg::centered_part_moments(x, tr, None, scale, Par::Seq).1
}

/// The all-row frame of the moment arms: the column means `m` of X over
/// all rows (summed in row order) and `X̃'1`, `X̃ = (x − 1·m')/scale`.
fn all_row_frame(x: MatRef<'_, f64>, scale: ColRef<'_, f64>) -> (Col<f64>, Col<f64>) {
    let n = x.nrows();
    let mean = Col::<f64>::from_fn(x.ncols(), |j| plain_mean(n, |i| x[(i, j)]));
    let x_sum = crate::linalg::centered_xt_vec(x, mean.as_ref(), scale, |_| 1.0);
    (mean, x_sum)
}

impl AllRowMoments {
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn new(
        x: MatRef<'_, f64>,
        scale: ColRef<'_, f64>,
        n_cols: usize,
        column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
    ) -> Self {
        use crate::linalg::centered_xt_vec;
        let n = x.nrows();
        let (mean, x_sum) = all_row_frame(x, scale);
        let cols = crate::resample::map_indexed(n_cols, |c| {
            let y = column_y(c);
            let shift = plain_mean(n, |i| y[i]);
            let xty = centered_xt_vec(x, mean.as_ref(), scale, |i| y[i] - shift);
            (shift, xty)
        });
        let xty = Mat::<f64>::from_fn(x.ncols(), n_cols, |j, c| cols[c].1[j]);
        let y_shift = cols.into_iter().map(|(s, _)| s).collect();
        Self {
            mean,
            x_sum,
            y_shift,
            xty,
        }
    }
}

/// The large buffers one split's [`TestMoments`] adds to a unit: the test
/// half of X̃ it centres and factors in place, and `R_B`.
fn test_moments_bytes(n_test: usize, p: usize) -> usize {
    use crate::resample::copy_bytes;
    copy_bytes(n_test, p).saturating_add(copy_bytes(p, p))
}

/// Whether a split loop over `n_splits` splits of `n_test` test rows and
/// `n_cols` outcome columns scores its test halves from moments
/// ([`TestMoments`], [`r_from_moments`]) instead of from the test rows. Only
/// unweighted (weighted calls score the test rows), only while
/// [`gram_routes_disabled`] is false, only when it takes fewer
/// multiply-adds (`J·n_B·p² + n·p·B' + J·p²·B'` for the QR factors of the
/// test halves, `X̃'Y` and the quadratic forms, against `J·n_B·p·B'` for the test-row products, `B'`
/// the column count), and only when the buffers of one split and of the
/// call, `(p² + p·n_cols)·8` bytes, fit `fit::memory_budget`. Every input is
/// a shape or the budget, never the thread count, so the arm does not
/// depend on the pool.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn test_moments_route(
    n: usize,
    n_test: usize,
    p: usize,
    n_cols: usize,
    n_splits: usize,
    weighted: bool,
) -> bool {
    if weighted || gram_routes_disabled() {
        return false;
    }
    let (nf, nb, pf, c, j) = (
        n as f64,
        n_test as f64,
        p as f64,
        n_cols as f64,
        n_splits as f64,
    );
    let cheaper = j * nb * pf * pf + nf * pf * c + j * pf * pf * c < j * nb * pf * c;
    let bytes = p
        .saturating_mul(p)
        .saturating_add(p.saturating_mul(n_cols))
        .saturating_mul(8);
    cheaper && u64::try_from(bytes).is_ok_and(|b| b <= crate::fit::memory_budget())
}

/// The call-level side of the tall moment arms when they sum every split's
/// training rows from the cells of the split design ([`crate::cell_sums`]),
/// unweighted: `X̃ = (x − 1·m')/scale` over all rows in [`AllRowMoments`]'s
/// frame ([`all_row_frame`]), `X̃'1`, `‖X̃‖_F`, and per split its
/// [`TestMoments`] and, from [`crate::cell_sums::cell_grams`], its training
/// half's `X̃_A'1` and, with `with_grams` (for [`CellFrame::take_grams`]),
/// its centred cross-product matrix `S_A`; and per split
/// [`train_fro`]: `‖X̃_tr‖_F` as [`prepare_split`] gives it (so the
/// truncation floor reads the bits the other arms read).
pub(crate) struct CellFrame {
    pub(crate) xs: Mat<f64>,
    pub(crate) x_sum: Col<f64>,
    pub(crate) x_fro: f64,
    pub(crate) tms: Vec<TestMoments>,
    s_a: Vec<Mat<f64>>,
    pub(crate) xa_1: Vec<Col<f64>>,
    pub(crate) tr_fro: Vec<f64>,
}

impl CellFrame {
    pub(crate) fn new(
        x: MatRef<'_, f64>,
        scale: ColRef<'_, f64>,
        d: &crate::resample::SplitDesign,
        with_grams: bool,
    ) -> Self {
        let (mean, x_sum) = all_row_frame(x, scale);
        let all: Vec<usize> = (0..x.nrows()).collect();
        let xs = crate::linalg::standardize_apply_rows(x, &all, mean.as_ref(), scale, None);
        let (s_a, xa_1) = crate::cell_sums::cell_grams(xs.as_ref(), d, with_grams);
        // One unit is one test half's copy while it is factored, the
        // per-split moment arm's unit (`cell_sums::cells_eligible` counts one).
        let n_b = d.splits.first().map_or(0, |sp| sp.te.len());
        let unit = test_moments_bytes(n_b, x.ncols());
        let per_split = crate::resample::map_units(d.splits.len(), unit, |j| {
            let sp = &d.splits[j];
            (
                TestMoments::new(x, &sp.te, mean.as_ref(), scale),
                train_fro(x, &sp.tr, scale),
            )
        });
        let (tms, tr_fro): (Vec<_>, Vec<_>) = per_split.into_iter().unzip();
        Self {
            x_fro: xs.norm_l2(),
            xs,
            x_sum,
            tms,
            s_a,
            xa_1,
            tr_fro,
        }
    }

    /// Every split's Gram-p block from its `S_A`, which moves out of the
    /// frame, with `n_tr` training rows.
    fn take_grams(&mut self, n_tr: usize) -> Vec<crate::gram_p::GramPBlock> {
        debug_assert_eq!(
            self.s_a.len(),
            self.tr_fro.len(),
            "frame built without grams"
        );
        std::mem::take(&mut self.s_a)
            .into_iter()
            .zip(&self.tr_fro)
            .map(|(s, &fro)| crate::gram_p::GramPBlock::from_gram(s, n_tr, fro))
            .collect()
    }
}

/// The `J × n_cols` values of a tall moment arm that sums its training rows
/// from the cells of `d`: the columns in fixed batches of
/// `cell_sums::cell_batch(n_cols)`, each an `n × g` block that `column_y(c, ..)`
/// writes, every column of it centred by its all-row mean (the shift
/// [`AllRowMoments`] applies) and summed over the rows once per batch
/// ([`crate::cell_sums::cell_sums`] in groups of `g`); `per(cs, j, c)` then
/// gives split `j`'s value of the batch's column `c`, or `None` for the
/// caller's test-row arm. The batches run through `resample::map_units`,
/// each holding its `n × b·g` columns and its sums, and a batch's splits map
/// in parallel; the batches depend on `n_cols` alone, so the bits do not
/// depend on the pool.
#[allow(clippy::many_single_char_names)]
pub(crate) fn cell_rows<T: Clone + Send>(
    frame: &CellFrame,
    d: &crate::resample::SplitDesign,
    n_cols: usize,
    g: usize,
    column_y: &(dyn Fn(usize, faer::MatMut<'_, f64>) + Sync),
    per: &(dyn Fn(&crate::cell_sums::CellSums, usize, usize) -> T + Sync),
) -> Vec<Vec<T>> {
    use crate::resample::copy_bytes;
    let (n, p, n_splits) = (frame.xs.nrows(), frame.xs.ncols(), d.splits.len());
    let b = crate::cell_sums::cell_batch(n_cols);
    let bytes = copy_bytes(n, b * g)
        .saturating_add(copy_bytes(d.cells(), (p + 1 + g) * b * g))
        .saturating_add(copy_bytes(p, n_splits * b * g));
    let batches = crate::resample::map_units(n_cols.div_ceil(b), bytes, |u| {
        let (c0, c1) = (u * b, n_cols.min(u * b + b));
        let mut ys = Mat::<f64>::zeros(n, (c1 - c0) * g);
        for c in c0..c1 {
            column_y(c, ys.as_mut().subcols_mut((c - c0) * g, g));
        }
        for e in 0..ys.ncols() {
            let shift = plain_mean(n, |i| ys[(i, e)]);
            for i in 0..n {
                ys[(i, e)] -= shift;
            }
        }
        let cs = crate::cell_sums::cell_sums(frame.xs.as_ref(), d, ys.as_ref(), g);
        crate::resample::map_indexed(n_splits, |j| {
            (0..c1 - c0).map(|c| per(&cs, j, c)).collect::<Vec<_>>()
        })
    });
    (0..n_splits)
        .map(|j| {
            batches
                .iter()
                .flat_map(|bt| bt[j].iter().cloned())
                .collect()
        })
        .collect()
}

/// Column `c` of a batch on split `j`'s halves from the cells, `y` centred
/// by its all-row mean: `(1'y_A, ‖C_A y_A‖², 1'y_B, ‖C_B y_B‖²)`, the test
/// half's sums as the totals minus the training half's. The centred sums of
/// squares come from `y'y` and `1'y` in one pass, which loses up to about
/// `n·ε·y'y` (`y'y` over all `n` rows) to cancellation; where either is not
/// above `band·8n·ε·y'y`, or is not finite, the result is `None` and the
/// caller's test-row arm decides the column, constant halves included.
fn cell_y_sums(
    cs: &crate::cell_sums::CellSums,
    j: usize,
    c: usize,
    n_a: usize,
    n_b: usize,
    band: f64,
) -> Option<(f64, f64, f64, f64)> {
    let (na, nb) = (n_a as f64, n_b as f64);
    let (sum_a, ss_a) = (cs.ya_sum[(j, c)], cs.ya_ss[(j, c)]);
    let (sum_b, ss_b) = (cs.y_sum[c] - sum_a, cs.y_ss[c] - ss_a);
    let css_a = sum_a.mul_add(-sum_a / na, ss_a);
    let css_b = sum_b.mul_add(-sum_b / nb, ss_b);
    let tol = band * 8.0 * (na + nb) * f64::EPSILON * cs.y_ss[c];
    (css_a > tol && css_b > tol).then_some((sum_a, css_a, sum_b, css_b))
}

/// The training side of a batch's column `c` on split `j` from the cells,
/// with `y` centred by its all-row mean: the sums of [`cell_y_sums`],
/// `ȳ_A` (the mean of `y_A − c`), the scale `σ` of the refit route's
/// standardization of `y_A`, `‖z‖ = ‖C_A y_A‖/σ` for its target `z`, and
/// `v = X̃_A'(y_A − c) − ȳ_A·X̃_A'1`, which is `X̃_tr'C_A y_tr = σ·X̃_tr'z`.
/// `None` where [`cell_y_sums`] is.
struct CellColumn {
    sum_b: f64,
    css_b: f64,
    y_bar: f64,
    y_scale: f64,
    z_norm: f64,
    v: Col<f64>,
}

#[allow(clippy::similar_names, clippy::many_single_char_names)]
fn cell_column(
    f: &CellFrame,
    cs: &crate::cell_sums::CellSums,
    j: usize,
    c: usize,
    (n_tr, n_te): (usize, usize),
    band: f64,
) -> Option<CellColumn> {
    let (sum_a, css_a, sum_b, css_b) = cell_y_sums(cs, j, c, n_tr, n_te, band)?;
    let b = cs.ya_sum.ncols();
    let y_bar = sum_a / n_tr as f64;
    let y_scale = (css_a / n_tr as f64).sqrt();
    let v = Col::<f64>::from_fn(f.xs.ncols(), |q| {
        y_bar.mul_add(-f.xa_1[j][q], cs.xa_y[(q, j * b + c)])
    });
    Some(CellColumn {
        sum_b,
        css_b,
        y_bar,
        y_scale,
        z_norm: css_a.sqrt() / y_scale,
        v,
    })
}

/// `X̃_B'(y_B − c)` of a batch's column `c` on split `j`: the all-row total
/// minus `X̃_A'(y_A − c)`.
fn cell_xb_y(cs: &crate::cell_sums::CellSums, j: usize, c: usize) -> Col<f64> {
    let b = cs.ya_sum.ncols();
    Col::<f64>::from_fn(cs.x_y.nrows(), |q| cs.x_y[(q, c)] - cs.xa_y[(q, j * b + c)])
}

/// [`nr_split_r_moments`] with its training sums from the cells: with
/// `ȳ_A` the mean of `y_A − c` (`c` the all-row mean) and `σ` the scale of
/// the refit route's standardization of `y_A`, `v = X̃_A'(y_A − c) − ȳ_A·X̃_A'1`
/// is `X̃_tr'C_A y_tr = σ·X̃_tr'z`, and `X̃_B'(y_B − c)` is the total minus
/// `X̃_A'(y_A − c)`. The same truncation rule on `‖v‖/σ` against the floor
/// of `‖X̃_tr‖_F`, the same correlation ([`r_from_moments`]) and
/// the same gate, with the product's rounding bound taken over all `n` rows
/// and the transform's `levels = log₂ H` stages:
/// `F_te·(p + n + levels + 4)·ε·‖X̃‖_F·(‖y − c‖ + 2√n·|ȳ_A|)`. A column the
/// sums cannot settle ([`cell_y_sums`]) or the gate leaves is `None`. A
/// split whose halves are not both finite gives `0`, as on route A's
/// test-row arm.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names
)]
fn nr_cell_r(
    f: &CellFrame,
    cs: &crate::cell_sums::CellSums,
    j: usize,
    c: usize,
    n_tr: usize,
    n_te: usize,
    levels: usize,
) -> Option<f64> {
    let CellColumn {
        sum_b,
        css_b,
        y_bar,
        y_scale,
        z_norm,
        v,
    } = cell_column(f, cs, j, c, (n_tr, n_te), NR_RESOLVE_BAND)?;
    let (n, p) = (f.xs.nrows(), f.xs.ncols());
    let wn = v.norm_l2() / y_scale;
    if !k1_keeps_w(wn, z_norm, n_tr, p, f.tr_fro[j]) {
        return Some(0.0);
    }
    let xb_y = cell_xb_y(cs, j, c);
    let tm = &f.tms[j];
    let r = r_from_moments(v.as_ref(), xb_y.as_ref(), sum_b, css_b, tm);
    let sn = tm.score_norms(v.as_ref(), f.x_sum.as_ref(), n_tr);
    let eps = f64::EPSILON;
    let prod_err = (p + n + levels + 4) as f64 * eps * f.x_fro;
    let y_norm = (2.0 * (n as f64).sqrt()).mul_add(y_bar.abs(), cs.y_ss[c].sqrt());
    let err =
        (sn.f_te * prod_err).mul_add(y_norm, ((n_te + 2) as f64 * eps).mul_add(sn.s_ref, sn.err));
    // `a >= b` conjunctions, so a NaN leaves the column to the test-row arm.
    let resolved = sn.t_c >= NR_RESOLVE_BAND * err
        && moments_resolved(&sn, sum_b, css_b, n_te, NR_RESOLVE_BAND)
        && (wn / z_norm).powi(2) >= crate::fit::NIPALS_ABS_FLOOR
        && r.is_finite();
    resolved.then_some(r)
}

/// [`split_column_r_gram_p_moments`] with its training sums from the
/// cells: the Gram fit on `gram` (the split's `S_A` from the cells) of
/// `s₁ = (X̃_A'(y_A − c) − ȳ_A·X̃_A'1)/σ = X̃_tr'z` with `‖z‖ = ‖C_A y_A‖/σ`,
/// then the correlation and the gate of the moment arm, `X̃_B'(y_B − c)` the
/// total minus `X̃_A'(y_A − c)`. A column the sums cannot settle
/// ([`cell_y_sums`]) or the gate leaves is `None`. A split whose halves are
/// not both finite gives `0`, as on [`split_unit`]. Per k of `ks`: one fit
/// at `ks.last()`, read and gated at each k through [`per_k`].
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names
)]
fn gram_p_cell_r(
    f: &CellFrame,
    gram: &crate::gram_p::GramPBlock,
    cs: &crate::cell_sums::CellSums,
    j: usize,
    c: usize,
    (n_tr, n_te): (usize, usize),
    ks: &[usize],
    keep: Option<usize>,
) -> Vec<Option<f64>> {
    let Some(cc) = cell_column(f, cs, j, c, (n_tr, n_te), crate::dual_route::RESOLVE_BAND) else {
        return vec![None; ks.len()];
    };
    let fit = gram_p_cell_fit(gram, &cc, *ks.last().expect("ks is non-empty"), keep);
    if fit.k_used == 0 {
        return vec![Some(0.0); ks.len()];
    }
    let (sum_b, css_b) = (cc.sum_b, cc.css_b);
    let xb_y = cell_xb_y(cs, j, c);
    let tm = &f.tms[j];
    per_k(ks, (&fit.w, &fit.p, &fit.q, &fit.coef), Par::Seq, |coef| {
        let r = r_from_moments(coef, xb_y.as_ref(), sum_b, css_b, tm);
        let sn = tm.score_norms(coef, f.x_sum.as_ref(), n_tr);
        let band = crate::dual_route::RESOLVE_BAND;
        // `a >= b` conjunctions, so a NaN fails the gate.
        let resolved = sn.t_c >= band * sn.err
            && moments_resolved(&sn, sum_b, css_b, n_te, band)
            && r.is_finite();
        resolved.then_some(r)
    })
}

/// The Gram fit of [`gram_p_cell_r`]: `s₁ = v/σ`, `‖z‖` from `cc`.
fn gram_p_cell_fit(
    gram: &crate::gram_p::GramPBlock,
    cc: &CellColumn,
    k: usize,
    keep: Option<usize>,
) -> crate::gram_p::GramFit {
    let s1 = Col::<f64>::from_fn(cc.v.nrows(), |q| cc.v[q] / cc.y_scale);
    gram.fit(s1.as_ref(), cc.z_norm, k, keep)
}

/// `split_zbars_columns` on the Gram-p route with every split's training
/// sums from the cells of `d`: per split `C = S_A` from the cells
/// ([`CellFrame`]), the columns through [`gram_p_cell_r`] in the batches of
/// [`cell_rows`], then the columns it leaves at any k on [`split_unit`]'s
/// Gram-p arm over the split's prepared halves, the splits that have one in
/// batches of `resample::map_units` (`unit_bytes` each); a k the cell arm
/// decided keeps the cell arm's value. Each z̄ is
/// [`split_zbars_columns`]'s flat entry `c·ks.len() + i`.
#[allow(clippy::too_many_arguments)]
fn gram_p_cell_zbars(
    x: MatRef<'_, f64>,
    d: &crate::resample::SplitDesign,
    scale: ColRef<'_, f64>,
    ks: &[usize],
    keep: Option<usize>,
    n_cols: usize,
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
    unit_bytes: usize,
) -> Vec<f64> {
    let mut frame = CellFrame::new(x, scale, d, true);
    let sizes = (d.splits[0].tr.len(), d.splits[0].te.len());
    let grams = frame.take_grams(sizes.0);
    let cell = cell_rows(
        &frame,
        d,
        n_cols,
        1,
        &|c, out| out.col_mut(0).copy_from(column_y(c)),
        &|cs, j, c| gram_p_cell_r(&frame, &grams[j], cs, j, c, sizes, ks, keep),
    );
    let decided: Vec<Vec<Option<Vec<f64>>>> = cell
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| v.iter().copied().collect::<Option<Vec<f64>>>())
                .collect()
        })
        .collect();
    let par = crate::fit::par_fixed();
    let rows = crate::resample::fill_pending(decided, unit_bytes, |j, pending| {
        let sp = &d.splits[j];
        let prep = prepare_split(x, sp, None, scale);
        let block = split_block(ReplicateRoute::GramP, &prep, par);
        pending
            .iter()
            .map(|&c| {
                let yc = column_y(c);
                split_unit(&block, &prep, sp, &|i| yc[i], ks, keep)
            })
            .collect()
    });
    // ±0.9999 pre-atanh clamp mirrored from nb_test (change together).
    let z_rows: Vec<Vec<f64>> = rows
        .iter()
        .zip(&cell)
        .map(|(row, cell_row)| {
            row.iter()
                .zip(cell_row)
                .flat_map(|(rs, cs)| rs.iter().zip(cs))
                .map(|(&r, &c)| c.unwrap_or(r).clamp(-0.9999, 0.9999).atanh())
                .collect()
        })
        .collect();
    zbars_from_rows(n_cols * ks.len(), &z_rows)
}

/// [`split_column_r_gram_p`] with the test half scored from moments: the
/// same Gram fit on the same target, then the correlation from `tm` and the
/// call's [`AllRowMoments`] `am` (column `col` of `am`'s `X̃'Y` is this
/// column's) instead of from `X̃_te·coef`. Unweighted only. With `ŝ₁`
/// the fit's first product, `σ_A` the scale of the training target's
/// standardization, `c` the column's `y_shift` and `ȳ'_A` the mean of
/// `y_A − c`, `X̃_A'C_A y_A = σ_A·ŝ₁` and
/// `X̃_B'(y_B − c) = X̃'(y − c) − σ_A·ŝ₁ − X̃_A'1·ȳ'_A`.
///
/// The gate guards this arm's arithmetic against the test-row arm
/// ([`split_column_r_gram_p`], the same `coef`) on the moment norms
/// ([`TestMoments::score_norms`]): `t_c ≥ RESOLVE_BAND·err` (`err` the
/// moment arithmetic's bound), [`moments_resolved`] at `RESOLVE_BAND` (the
/// conditioning of the scores and the constant tests) and a finite `r`;
/// otherwise `None`, and the caller runs the test-row arm. A non-finite
/// half (`!prep.x_finite`), a target the per-column checks refuse, the
/// zero model and a constant held-out y give `0`, as on [`split_unit`].
/// Per k of `ks`: one fit at `ks.last()`, read and gated at each k through
/// [`per_k`].
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn split_column_r_gram_p_moments(
    gram: &crate::gram_p::GramPBlock,
    tm: &TestMoments,
    am: &AllRowMoments,
    col: usize,
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
    keep: Option<usize>,
) -> Vec<Option<f64>> {
    debug_assert!(prep.w_tr.is_none() && prep.w_te.is_none());
    let k = *ks.last().expect("ks is non-empty");
    let zeros = || vec![Some(0.0); ks.len()];
    if !prep.x_finite {
        return zeros();
    }
    let tr = sp.tr.as_slice();
    let y_tr = Col::<f64>::from_fn(tr.len(), |i| y_of(tr[i]));
    // `split_train_target`'s array, unweighted, with its scale.
    let (ys_tr, _, y_scale) = crate::linalg::standardize1(y_tr.as_ref());
    if crate::fit::check_fit_y_and_k(prep.xs_tr.ncols(), ys_tr.as_ref(), k, keep).is_err() {
        return zeros();
    }
    let s1 = crate::gram_p::first_product(prep.xs_tr.as_ref(), ys_tr.as_ref(), SPLIT_FIT_PAR, k);
    let fit = gram.fit(s1.as_ref(), ys_tr.norm_l2(), k, keep);
    if fit.k_used == 0 {
        return zeros();
    }
    let y_te = split_test_target(prep, sp, y_of);
    let n_te = y_te.nrows();
    if scaled_moments(n_te, |i| y_te[i], None).is_constant(n_te) {
        return zeros();
    }
    let shift = am.y_shift[col];
    let y_bar_a = plain_mean(tr.len(), |i| y_tr[i] - shift);
    let p = fit.coef.nrows();
    let xb_y = Col::<f64>::from_fn(p, |j| {
        let xa_sum = am.x_sum[j] - tm.xb_sum[j];
        am.xty[(j, col)] - y_scale * s1[j] - xa_sum * y_bar_a
    });
    let (y_b_sum, y_b_ss) = y_b_sums(n_te, |i| y_te[i] - shift);
    per_k(ks, (&fit.w, &fit.p, &fit.q, &fit.coef), Par::Seq, |coef| {
        let r = r_from_moments(coef, xb_y.as_ref(), y_b_sum, y_b_ss, tm);
        let sn = tm.score_norms(coef, am.x_sum.as_ref(), tr.len());
        let band = crate::dual_route::RESOLVE_BAND;
        // `a >= b` conjunctions, so a NaN fails the gate.
        let resolved = sn.t_c >= band * sn.err
            && moments_resolved(&sn, y_b_sum, y_b_ss, n_te, band)
            && r.is_finite();
        resolved.then_some(r)
    })
}

/// The split-half Pearson r on one split: `split_half_correlations`'s
/// per-split computation, which `run_split_nb` uses for K ≥ 2 or a sparse
/// `keep`. `split_exact`'s refit route (`run_split_perm`) reaches the same
/// computation through `split_column_r` via `split_unit`, not through this
/// function.
fn split_half_r(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    sp: &SplitIdx,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    keep: Option<usize>,
) -> f64 {
    split_column_r(
        &prepare_split(x, sp, w_norm, scale),
        sp,
        &|i| y[i],
        &[k],
        keep,
    )[0]
}

/// Per-split state of a route on a prepared split, built once per split by
/// [`split_zbars_columns`] after the split's preparation and shared
/// read-only by that split's columns. The primal route needs nothing beyond
/// the prepared split; the n-space route reads no prepared split
/// ([`split_columns_nspace`]).
pub(crate) enum SplitBlock<'s> {
    /// No per-split state.
    Primal(std::marker::PhantomData<&'s ()>),
    /// The split's p-space Gram block on the training half.
    GramP(crate::gram_p::GramPBlock),
}

/// The block of `route` for one prepared split, its precompute run under
/// `par` ([`crate::fit::par_fixed`]). `Special` and `Nspace` never reach
/// it; handed one, they get the primal block. `GramP` builds
/// `C = X̃_tr'X̃_tr` of the training half once per split.
pub(crate) fn split_block(route: ReplicateRoute, prep: &PreparedSplit, par: Par) -> SplitBlock<'_> {
    match route {
        ReplicateRoute::GramP => {
            SplitBlock::GramP(crate::gram_p::GramPBlock::new(prep.xs_tr.as_ref(), par))
        }
        ReplicateRoute::Special | ReplicateRoute::Nspace | ReplicateRoute::Primal => {
            SplitBlock::Primal(std::marker::PhantomData)
        }
    }
}

/// One (split, column) unit: the split-half `r` for the outcome `y_of`,
/// before the clamp. A split whose prepared halves are not both finite
/// gives `r = 0` on every route (the failed per-half fit's value). The
/// primal arm is [`split_column_r`], the Gram-p arm
/// [`split_column_r_gram_p`]. One `r` per k of `ks` (sorted ascending,
/// non-empty) from one fit at `ks.last()`.
pub(crate) fn split_unit(
    block: &SplitBlock<'_>,
    prep: &PreparedSplit,
    sp: &SplitIdx,
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
    keep: Option<usize>,
) -> Vec<f64> {
    if !prep.x_finite {
        return vec![0.0; ks.len()];
    }
    match block {
        SplitBlock::Primal(_) => split_column_r(prep, sp, y_of, ks, keep),
        SplitBlock::GramP(gram) => split_column_r_gram_p(gram, prep, sp, y_of, ks, keep),
    }
}

/// The `n_cols` split-half `r` values of split `sp` on the n-space route,
/// before the clamp and Fisher z. The split's block comes from the
/// unweighted all-row Gram matrix `g`
/// ([`NspaceSplit::from_gram`](crate::dual_route::NspaceSplit::from_gram)).
/// The columns go through `dual_route::split_columns_r_nspace` in runs of
/// `dual_route::NSPACE_BATCH` consecutive columns from column 0, one call
/// per run, so the per-replicate products with `G` and `M` become one
/// product per run; the runs are fixed by `n_cols` alone, so the bits do
/// not depend on the thread count. No half of X is written. `g` is finite.
/// Element `c` holds column `c`'s `r` at each k of `ks` (sorted ascending,
/// non-empty), from one fit at `ks.last()`.
pub(crate) fn split_columns_nspace(
    g: &HeldOutGram,
    sp: &SplitIdx,
    n_cols: usize,
    ks: &[usize],
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
) -> Vec<Vec<f64>> {
    let ns = crate::dual_route::NspaceSplit::from_gram(g, sp);
    let batch = crate::dual_route::NSPACE_BATCH;
    let runs = crate::resample::map_indexed(n_cols.div_ceil(batch), |run| {
        let c0 = run * batch;
        let ys: Vec<Col<f64>> = (c0..n_cols.min(c0 + batch)).map(column_y).collect();
        crate::dual_route::split_columns_r_nspace(sp, &ns, ys.len(), &|j, i| ys[j][i], ks)
    });
    runs.into_iter().flatten().collect()
}

/// The large buffers one split unit of [`split_zbars_columns`] holds on
/// `route`, for X of `n × p`. On `Nspace` only the split's block, `G`
/// (`n_tr × n_tr`) and `M` (`n_te × n_tr`); the all-row Gram matrix it
/// reads is one buffer per call and is not counted. Otherwise the prepared
/// halves `xs_tr` and `xs_te` ([`prepare_split`] writes the √w scaling into
/// them, so weights add no copy), the split's block (`C`, `p × p`, on
/// `GramP`) and one PLS1 fit per column that runs at once
/// ([`columns_bytes`]). Every split has the same training size (the split
/// sizes depend on `n` and `k` only), so the first split's stands for all.
fn split_unit_bytes(
    route: ReplicateRoute,
    p: usize,
    splits: &[SplitIdx],
    k: usize,
    n_cols: usize,
) -> usize {
    let block = match route {
        ReplicateRoute::Nspace => {
            return split_blocks_bytes(splits, splits.first().map_or(0, |sp| sp.tr.len()));
        }
        ReplicateRoute::GramP => crate::resample::copy_bytes(p, p),
        ReplicateRoute::Special | ReplicateRoute::Primal => 0,
    };
    let fits = match route {
        ReplicateRoute::Primal | ReplicateRoute::GramP => {
            columns_bytes(crate::fit::pls1_fit_bytes(p, k), n_cols)
        }
        ReplicateRoute::Special | ReplicateRoute::Nspace => 0,
    };
    split_blocks_bytes(splits, p)
        .saturating_add(block)
        .saturating_add(fits)
}

/// Per-column `z̄` of `split_exact`'s refit route on `route`: the split
/// driver, splits outer and replicate columns inner (the PLS3 primal
/// route's shape): the splits run in batches of `resample::map_units`,
/// each counted by [`split_unit_bytes`], so at most
/// `resample::units_at_once` prepared splits and their blocks
/// ([`split_block`] under [`crate::fit::par_fixed`]) are alive at once,
/// and the `n_cols` columns of each split map in parallel through
/// [`split_unit`] ([`map_split_columns`]), the same width the
/// replicate-outer loop had. On the `Nspace` route each split runs
/// [`split_columns_nspace`] instead, in fixed runs, on the blocks of one all-row Gram matrix: `gram`, the
/// unweighted `HeldOutGram::from_x` of `x` and `scale` when the caller
/// holds it (the `split_nb` gate's), else one built here
/// ([`block_gram`]). `column_y(c)` builds
/// column `c` (raw, length n) inside the unit (or run) that needs it, so no
/// `B·n` buffer is held. On the `GramP` route, when [`test_moments_route`]
/// admits the loop, each split also builds its [`TestMoments`] and every
/// column runs [`split_column_r_gram_p_moments`] on them and on one
/// [`AllRowMoments`] built here (one more pass of `column_y` over the
/// columns), so the test rows are not read per column. When the caller also
/// hands over the split `design` (`splits` are its splits) and
/// `cell_sums::cells_eligible` admits it, the training rows are not read
/// per split either: [`gram_p_cell_zbars`] takes every split's sums from
/// the cells.
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
///
/// Row `i` of the result is the z̄ at `ks[i]`, column `c` configuration `c`.
/// `ks` is sorted ascending and non-empty; every unit fits once at
/// `ks.last()` and reads each k from that fit, and the route's shapes
/// (unit sizes, column batches) are taken at `ks.last()`, so a one-element
/// `ks` runs the single-k loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_zbars_columns(
    route: ReplicateRoute,
    x: MatRef<'_, f64>,
    splits: &[SplitIdx],
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    ks: &[usize],
    keep: Option<usize>,
    n_cols: usize,
    column_y: &(dyn Fn(usize) -> Col<f64> + Sync),
    gram: Option<&HeldOutGram>,
    design: Option<&crate::resample::SplitDesign>,
) -> Mat<f64> {
    debug_assert!(
        ks.windows(2).all(|w| w[0] < w[1]),
        "ks is sorted, no duplicates"
    );
    let k = *ks.last().expect("ks is non-empty");
    let nk = ks.len();
    debug_assert!(k <= x.ncols(), "k <= p is the caller's precondition");
    debug_assert!(design.is_none_or(|d| std::ptr::eq(d.splits.as_ptr(), splits.as_ptr())));
    let par = crate::fit::par_fixed();
    let all = matches!(route, ReplicateRoute::Nspace).then(|| {
        debug_assert!(w_norm.is_none() && keep.is_none());
        block_gram(x, scale, gram)
    });
    let unit_bytes = split_unit_bytes(route, x.ncols(), splits, k, n_cols);
    // The Gram-p arm scores the test halves from moments when
    // `test_moments_route` admits the loop.
    let n_test = splits.first().map_or(0, |sp| sp.te.len());
    let moment_bytes = unit_bytes.saturating_add(test_moments_bytes(n_test, x.ncols()));
    let moments = route == ReplicateRoute::GramP
        && test_moments_route(
            x.nrows(),
            n_test,
            x.ncols(),
            n_cols,
            splits.len(),
            w_norm.is_some(),
        );
    let batch = crate::cell_sums::cell_batch(n_cols);
    let cells = design.filter(|d| {
        moments && crate::cell_sums::cells_eligible(d, x.ncols(), batch, w_norm.is_some())
    });
    // The units' values come flat, column-major (entry `c·nk + i` is
    // column c at `ks[i]`), so the split accumulation is the one-row loop's.
    let table = |flat: Vec<f64>| Mat::<f64>::from_fn(nk, n_cols, |i, c| flat[c * nk + i]);
    if let Some(d) = cells {
        return table(gram_p_cell_zbars(
            x, d, scale, ks, keep, n_cols, column_y, unit_bytes,
        ));
    }
    let am = moments.then(|| AllRowMoments::new(x, scale, n_cols, column_y));
    let unit_bytes = if am.is_some() {
        moment_bytes
    } else {
        unit_bytes
    };
    // ±0.9999 pre-atanh clamp mirrored from nb_test (change together).
    let fisher_z = |r: f64| r.clamp(-0.9999, 0.9999).atanh();
    table(zbars_over_splits(
        splits,
        n_cols * nk,
        unit_bytes,
        |sp: &SplitIdx| {
            if let Some(g) = &all {
                return split_columns_nspace(g, sp, n_cols, ks, column_y)
                    .into_iter()
                    .flatten()
                    .map(fisher_z)
                    .collect();
            }
            let prep = prepare_split(x, sp, w_norm, scale);
            let block = split_block(route, &prep, par);
            let tm = am
                .as_ref()
                .map(|am| TestMoments::new(x, &sp.te, am.mean.as_ref(), scale));
            map_split_columns(sp.tr.len(), x.ncols(), k, n_cols, |c| {
                let yc = column_y(c);
                let y_of = |i: usize| yc[i];
                match (&block, &tm, &am) {
                    (SplitBlock::GramP(g), Some(tm), Some(am)) => {
                        let mut r =
                            split_column_r_gram_p_moments(g, tm, am, c, &prep, sp, &y_of, ks, keep);
                        // A column the moment arm leaves at any k is fitted again
                        // by the test-row arm, which fills the k it left; the arm
                        // leaves few.
                        if r.contains(&None) {
                            let rows = split_column_r_gram_p(g, &prep, sp, &y_of, ks, keep);
                            for (v, f) in r.iter_mut().zip(rows) {
                                v.get_or_insert(f);
                            }
                        }
                        r.into_iter().flatten().map(fisher_z).collect()
                    }
                    _ => split_unit(&block, &prep, sp, &y_of, ks, keep)
                        .into_iter()
                        .map(fisher_z)
                        .collect::<Vec<f64>>(),
                }
            })
            .into_iter()
            .flatten()
            .collect()
        },
    ))
}

/// Pearson r of `a(0)..a(n−1)` and `b(0)..b(n−1)` with the crate's
/// degenerate-input convention: exactly `0.0` when either vector is
/// constant to rounding (`constant_to_rounding` on its [`scaled_moments`]),
/// otherwise the correlation clamped to `[-1, 1]`. The split-half statistic
/// wherever a split refits (`split_column_r` and `split_column_r_gram_p`:
/// `split_exact`'s refit route, `split_nb` at K ≥ 2 or with a sparse
/// `keep`), and PLS3's
/// `pls3_signal_test::pearson_r_guarded`. `n` must be positive.
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

/// The split-half correlations `split_nb` tests, at K = 1 on dense input:
/// each split's `r` is column 0 of the per-split body of `split_exact`'s
/// no-refit route on the route `k1_route` picks (`nr_split_r`, or
/// `nr_split_r_gram` on route G; `split_perm_nr_zbars` owns the identity
/// and the truncation rule), so no split writes a half of X. Every route
/// decides every split, in batches of `resample::map_units`. `gram` is the
/// gate's Gram
/// matrix of `x` (`SplitNbResolution::gram`), which route G reads instead
/// of building one.
fn split_nb_correlations_nr(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    splits: &[SplitIdx],
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    gram: Option<&HeldOutGram>,
) -> Col<f64> {
    let n = x.nrows();
    let (n_train, n_test) = crate::resample::split_sizes(n, 1);
    let y_mat = nr_outcome_matrix(&crate::resample::Columns {
        y: y.as_mat(),
        seeds: &[],
        null_map: None,
    });
    let route = k1_route(
        n,
        n_train,
        n_test,
        x.ncols(),
        1,
        splits.len(),
        gram.is_some(),
    );
    let g = (route == K1Route::G).then(|| k1_route_gram(x, w_norm, scale, gram, splits));
    let rs = crate::resample::map_units(
        splits.len(),
        nr_split_unit_bytes(
            n_train,
            n_test,
            route,
            g.as_ref().is_some_and(|(_, plain)| plain.is_some()),
        ),
        |j| match &g {
            Some((g, plain)) => {
                nr_split_r_gram(g, plain.as_deref(), y_mat.as_ref(), &splits[j], w_norm)[0]
            }
            None => nr_split_r(
                x,
                y_mat.as_ref(),
                &splits[j],
                w_norm,
                scale,
                route == K1Route::B,
            )[0],
        },
    );
    Col::<f64>::from_fn(rs.len(), |j| rs[j])
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn run_split_nb(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    opts: &ConfirmatoryTestOpts,
    gram: Option<&HeldOutGram>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use crate::resample::split_sizes;
    let n = x.nrows();
    let (n_train, n_test) = split_sizes(n, k);

    // Raw (X, y), weights and the all-row column scale flow into
    // split_half_correlations or split_nb_correlations_nr, which centre
    // each half, divide by the scale and √w-scale it (the √w row-scaling
    // convention) internally.
    let splits = draw_splits(n, k, n_splits, rng)?;
    // K = 1 on dense input runs the no-refit body, from the gate's Gram
    // matrix when it has one. At K ≥ 2 a call that holds the gate's Gram
    // matrix (finite, unweighted, dense) refits each split on the n-space
    // route from its blocks; every other request refits on prepared halves.
    let nspace_gram = gram
        .filter(|_| {
            !gram_routes_disabled()
                && crate::dual_route::nspace_eligible_split_nb(
                    n_train,
                    k,
                    opts.keep.is_none() && w_norm.is_none(),
                )
        })
        .map(|g| block_gram(x, scale, Some(g)));
    let r_splits = if split_exact_no_refit_route(k, opts.keep) {
        split_nb_correlations_nr(x, y, &splits, w_norm, scale, gram)
    } else if let Some(g) = &nspace_gram {
        let unit = split_unit_bytes(ReplicateRoute::Nspace, x.ncols(), &splits, k, 1);
        let r = crate::resample::map_units(splits.len(), unit, |j| {
            split_columns_nspace(g, &splits[j], 1, &[k], &|_| y.to_owned())[0][0]
        });
        Col::<f64>::from_fn(r.len(), |j| r[j])
    } else {
        split_half_correlations(x, y, k, &splits, w_norm, scale, opts.keep)
    };
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
    ks: &[usize],
    n_perm: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    opts: &ConfirmatoryTestOpts,
    null_map: Option<crate::resample::NullMap<'_>>,
    gram: Option<&HeldOutGram>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<SetResult> {
    let n = x.nrows();
    // The J splits are drawn once, for the largest k, and held fixed
    // across every k and all B permutation replicates. Redrawing them per replicate would fold split-to-split
    // scatter into the null, so the reference distribution would stop
    // isolating the y–X association the observed statistic measures.
    // run_split_perm_nr holds its splits fixed the same way.
    let design = draw_design(n, *ks.last().expect("ks is non-empty"), n_splits, rng)?;
    let splits = &design.splits;

    // One child seed per null column, drawn right after the splits, as
    // run_split_perm_nr and run_raw_perm draw them: every route sees the
    // same null columns at a given seed.
    let seeds = crate::rng::child_seeds(rng, n_perm);
    let cols = crate::resample::Columns {
        y: y.as_mat(),
        seeds: &seeds,
        null_map,
    };

    // Raw (X, y, weights) and the all-row column scale flow into the
    // per-split preparation (centre, scale, then √w: the √w row-scaling
    // convention); permuted y rows, weights tied to row positions (w[i]
    // always pairs with destination row i). The n-space route reads the
    // gate's all-row Gram matrix `gram` when the call holds one.
    let z = split_zbars_columns(
        route,
        x,
        splits,
        w_norm,
        scale,
        ks,
        opts.keep,
        cols.len(),
        &|c| cols.column(c),
        gram,
        Some(&design),
    );
    set_result(&z, n_perm, |z, b| Ok(split_perm_result(z, b)))
}

/// `split_exact`'s p-value and statistic `tanh(z̄)` from every column's
/// `z̄`, column 0 the observed `y`. A non-finite null statistic counts as
/// an exceedance so it biases p upward, never downward (mirrors
/// `raw_perm_result`; change together). A failed per-half fit already
/// degrades to r = 0 inside the units rather than surfacing here.
fn split_perm_result(z: &[f64], n_perm: usize) -> RunResult {
    let (z_bar_obs, null_zbars) = (z[0], &z[1..]);
    let exceedances = null_zbars
        .iter()
        .filter(|v| !v.is_finite() || **v >= z_bar_obs)
        .count();
    let p = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);
    RunResult {
        pvalue: p,
        statistic: z_bar_obs.tanh(),
        rho_hat: None,
    }
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

/// Multiple of its rounding bound by which the k = 1 moment arm
/// (`nr_split_r_moments`) requires its centred test-half scores, and its
/// constant tests, to clear zero before it keeps its own correlation
/// instead of route A's test-row arm.
pub(crate) const NR_RESOLVE_BAND: f64 = 8.0;

/// The route of the no-refit body at k = 1, decided once per call by
/// [`k1_route`] from the shape alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum K1Route {
    /// `X̃_te·(X̃_tr'·Z)`, `Z` the standardized training targets, block by
    /// block over the columns of X (`nr_split_r`), or `X̃_tr'·Y` block by
    /// block with the test half scored from its moments when
    /// `test_moments_route` admits the loop (`nr_split_r_moments`); there,
    /// when `cell_sums::cells_eligible` admits the design too, every split's
    /// `X̃_tr'·Y` comes from the cells (`nr_cell_r`).
    A,
    /// `(X̃_te·X̃_tr')·Y`: `M = X̃_te·X̃_tr'` and `K_A = X̃_tr·X̃_tr'` summed
    /// over the column blocks of X, then `M·Y` (`nr_split_r`); every column
    /// is decided from `K_A`.
    B,
    /// `M·Y` with `M` and `K_A` taken from the blocks of one all-row Gram
    /// matrix (`nr_split_r_gram`); every column is decided from `K_A`.
    G,
}

/// The route of the no-refit body with the fewest multiply-adds over
/// `n_splits` splits of `n_cols` outcome columns (`B + 1` for
/// `split_exact`, one for `split_nb`): A costs `J·n·p·n_cols`, B
/// `J·(n_test·n_train·(p + n_cols) + n_train²·(p + n_cols))` (`M` and `K_A`
/// summed over the columns of X, then `M·Y` and each column's `K_A·z`) and
/// G `n²p/2 + J·(n_test·n_train + n_train²)·n_cols`, without the `n²p/2` of
/// building the Gram matrix when the caller holds it already
/// (`gram_built`). Route A's extra pass over the training half for a column
/// whose `t't` its bound leaves open is rare and not counted. G is in the choice unless
/// [`gram_routes_disabled`]; the memory budget does not gate it, since its
/// Gram matrix is one buffer per call, not per unit. No caller-facing knob:
/// the cost model decides.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub(crate) fn k1_route(
    n: usize,
    n_train: usize,
    n_test: usize,
    p: usize,
    n_cols: usize,
    n_splits: usize,
    gram_built: bool,
) -> K1Route {
    let (n, p, c, j) = (n as f64, p as f64, n_cols as f64, n_splits as f64);
    let pair = n_test as f64 * n_train as f64;
    let kernel = n_train as f64 * n_train as f64;
    let a = j * n * p * c;
    let b = j * (pair + kernel) * (p + c);
    let g = if gram_built { 0.0 } else { n * n * p / 2.0 } + j * (pair + kernel) * c;
    if !gram_routes_disabled() && g < a.min(b) {
        K1Route::G
    } else if b < a {
        K1Route::B
    } else {
        K1Route::A
    }
}

/// The Gram matrices route G reads. The first is `gram` when the caller
/// holds it (`HeldOutGram::from_x` on `x`, `w_norm` and `scale`), else one
/// built here, with the √w row scaling under weights (`with_row_weights`).
/// The second is that matrix without the row scaling, kept only under
/// weights when some split of `splits` has a half of zero total weight
/// ([`half_weight_sum`]): [`nr_split_r_gram`] forms that split's halves
/// from it.
type K1Grams<'a> = (
    std::borrow::Cow<'a, HeldOutGram>,
    Option<std::borrow::Cow<'a, HeldOutGram>>,
);

fn k1_route_gram<'a>(
    x: MatRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    gram: Option<&'a HeldOutGram>,
    splits: &[SplitIdx],
) -> K1Grams<'a> {
    use std::borrow::Cow;
    let Some(w) = w_norm else {
        let g = gram.map_or_else(
            || Cow::Owned(HeldOutGram::from_x(x, None, scale)),
            Cow::Borrowed,
        );
        return (g, None);
    };
    let plain = gram.map_or_else(
        || Cow::Owned(HeldOutGram::from_x(x, Some(w), scale)),
        Cow::Borrowed,
    );
    let g = Cow::Owned(plain.with_row_weights(w));
    let zero_half = splits
        .iter()
        .any(|sp| half_weight_sum(w, &sp.tr) == 0.0 || half_weight_sum(w, &sp.te) == 0.0);
    (g, zero_half.then_some(plain))
}

/// `Σ_{i ∈ idx} w_i`, in index order: the total weight of a half.
fn half_weight_sum(w: ColRef<'_, f64>, idx: &[usize]) -> f64 {
    idx.iter().map(|&i| w[i]).sum::<f64>()
}

/// The `unit_bytes` of one split of the no-refit body: on routes A and B
/// ([`nr_split_r`]) its two stream blocks and, on B, `M`
/// (`n_test × n_train`) and `K_A` (`n_train × n_train`); on route G
/// ([`nr_split_r_gram`]) `K_A` and `M`, and, when the call holds the Gram
/// matrix without the row scaling (`plain`), the row-weighted copy of it
/// that a split with a half of zero total weight forms (`n × n`,
/// `n = n_train + n_test`). The outcome buffers grow with B and are not
/// counted.
fn nr_split_unit_bytes(n_train: usize, n_test: usize, route: K1Route, plain: bool) -> usize {
    use crate::resample::copy_bytes;
    let blocks = crate::linalg::stream_blocks_bytes(n_train, n_test);
    match route {
        K1Route::A => blocks,
        K1Route::B => copy_bytes(n_test, n_train)
            .saturating_add(copy_bytes(n_train, n_train))
            .saturating_add(blocks),
        K1Route::G => {
            let n = n_train + n_test;
            let copy = if plain { copy_bytes(n, n) } else { 0 };
            copy_bytes(n_train, n_train)
                .saturating_add(copy_bytes(n_test, n_train))
                .saturating_add(copy)
        }
    }
}

/// The outcome matrix of the no-refit route, `n × cols.len()`: the columns
/// of `cols` side by side, column 0 the observed `y`.
///
/// Every column is divided by one power of two near the largest absolute
/// entry of the matrix (`linalg::pow2_scale`, the standardizers' kernel), so
/// that the scores `t_te` and the moment arm's `‖y_tr‖`, `ȳ`, `σ` and `err`
/// stay finite and normal at any magnitude of `y` (formed on the raw `y`,
/// `‖y_tr‖` is `inf` once `|y|` exceeds about `1e154`, and `0` below about
/// `1e-162`). The division is exact, every one of those quantities is
/// linear in `y`, and the gate and the correlation read only their ratios,
/// so wherever the raw quantities are in range this changes no bit of the
/// output. The training targets `z` are standardized, so they do not
/// depend on the scale of `y` at all.
fn nr_outcome_matrix(cols: &crate::resample::Columns<'_>) -> Mat<f64> {
    let mut y_mat = cols.matrix(0, cols.len());
    let y_max_abs = y_mat
        .col_iter()
        .flat_map(|c| c.iter().map(|v| v.abs()))
        .fold(0.0_f64, f64::max);
    let (_, y_inv) = crate::linalg::pow2_scale(y_max_abs);
    for c in y_mat.col_iter_mut() {
        for v in c.iter_mut() {
            *v *= y_inv;
        }
    }
    y_mat
}

/// The `y_mat.ncols()` split-half correlations of one split on the
/// no-refit route, before the clamp and `atanh`: each half is centred with
/// the training means and divided by the all-row column `scale` (and
/// √w-scaled when weighted), as [`prepare_split`] does, the scores of
/// every column come from one pass over the columns of `x`, and each column
/// is then centered and correlated. `split_perm_nr_zbars` owns the
/// identity and the truncation rule. On route A the pass forms
/// `X̃_tr'Z`, `Z` the refit route's training targets ([`nr_train_z`]), and
/// the scores `X̃_te·(X̃_tr'Z)`; each column is decided by
/// [`nr_column_r_primal`], and every column of a split whose halves are not
/// both finite is `0`. On route B the same pass sums
/// `M = X̃_te X̃_tr'` and `K_A = X̃_tr X̃_tr'`, and every column is decided
/// by [`nr_column_r_dual`]. `y_mat` must come from
/// `nr_outcome_matrix`, so the norms stay finite and normal at any
/// magnitude of `y`. Given `(x, y_mat, sp)` nothing here draws randomness,
/// so a caller maps the splits in batches of `resample::map_units`
/// ([`nr_split_unit_bytes`] per split).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
fn nr_split_r(
    x: MatRef<'_, f64>,
    y_mat: MatRef<'_, f64>,
    sp: &SplitIdx,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    route_b: bool,
) -> Vec<f64> {
    use crate::linalg::row_subset;
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let n_cols = y_mat.ncols();
    let p_features = x.ncols();

    // Per-half weights renormalized to mean 1 within the half, weighted
    // means, then √w row-scaling, every step mirrored from
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

    // √w row-scaling of the outcomes. The blocks of X̃_tr and X̃_te carry
    // √w_tr and √w_te from `linalg::centered_blocks`. On the train side
    // the √w_tr on Y is the `diag(√w_tr)` that sits *inside* the linear
    // map (see "Under weights" on `split_perm_nr_zbars`). On the test side
    // both the scores (through X̃_te) and the raw test y carry √w_te, so the
    // Pearson r below is taken on √w_te-scaled data, matching what
    // split_half_correlations reports.
    let scale_rows = |m: Mat<f64>, sw: Option<&Col<f64>>| match sw {
        Some(sw) => Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * sw[i]),
        None => m,
    };
    let y_te = scale_rows(row_subset(y_mat.as_ref(), &sp.te), sw_te.as_ref());
    // Route B maps the √w-scaled training y; route A maps `Z`, the refit
    // route's standardized, √w-scaled training targets, whose `X̃_tr'z` is
    // the refit route's first product.
    let (y_tr, z_norms) = if route_b {
        (
            scale_rows(row_subset(y_mat.as_ref(), &sp.tr), sw_tr.as_ref()),
            Vec::new(),
        )
    } else {
        let zs: Vec<Col<f64>> = (0..n_cols)
            .map(|c| nr_train_z(c, y_mat, sp, w_tr.as_ref(), sw_tr.as_ref()).0)
            .collect();
        (
            Mat::<f64>::from_fn(sp.tr.len(), n_cols, |i, c| zs[c][i]),
            zs.iter().map(Col::norm_l2).collect(),
        )
    };

    let n_tr = sp.tr.len();
    let n_te = sp.te.len();
    // Seq inside the per-split worker: outer Rayon owns the threadpool.
    let seq = faer::Par::Seq;

    // One pass over the columns of X, a block at a time: the two halves of
    // a block are centred and scaled, used and dropped, so this pass holds `M`
    // and `K_A` (route B) or the block's `X̃_tr'Z` (route A) and writes no
    // half of X. Route B: `M = X̃_te X̃_tr'`, then `t_te = M·Y`. Route A:
    // `t_te = X̃_te (X̃_tr'·Z)` and `‖X̃_tr'z‖²`, block by block.
    let mut m = Mat::<f64>::zeros(if route_b { n_te } else { 0 }, n_tr);
    let k_side = if route_b { n_tr } else { 0 };
    let mut k_a = Mat::<f64>::zeros(k_side, k_side);
    let mut t_te = Mat::<f64>::zeros(n_te, n_cols);
    let mut wn2 = vec![0.0_f64; if route_b { 0 } else { n_cols }];
    // Whether both halves are finite (`PreparedSplit::x_finite`'s check).
    let mut x_finite = true;
    let x_tr_fro = crate::linalg::centered_blocks(
        x,
        &sp.tr,
        &sp.te,
        w_tr.as_ref().map(Col::as_ref),
        sw_tr.as_ref().map(Col::as_ref),
        sw_te.as_ref().map(Col::as_ref),
        scale,
        // The width linalg::stream_blocks_bytes counts (change together).
        crate::linalg::stream_block_width(n_tr.max(n_te)),
        |_, a, c| {
            if route_b {
                matmul(m.as_mut(), Accum::Add, c, a.transpose(), 1.0, seq);
                matmul(k_a.as_mut(), Accum::Add, a, a.transpose(), 1.0, seq);
                return;
            }
            x_finite &=
                crate::fit::check_finite_mat(a).is_ok() && crate::fit::check_finite_mat(c).is_ok();
            let g = crate::linalg::mat_mul(a.transpose(), y_tr.as_ref(), seq);
            matmul(t_te.as_mut(), Accum::Add, c, g.as_ref(), 1.0, seq);
            for (col, s) in wn2.iter_mut().enumerate() {
                *s += (0..g.nrows())
                    .map(|j| g[(j, col)] * g[(j, col)])
                    .sum::<f64>();
            }
        },
    );
    if route_b {
        t_te = crate::linalg::mat_mul(m.as_ref(), y_tr.as_ref(), seq);
        return (0..n_cols)
            .map(|col| {
                nr_column_r_dual(
                    col,
                    y_mat,
                    sp,
                    w_tr.as_ref(),
                    sw_tr.as_ref(),
                    t_te.as_ref(),
                    y_te.as_ref(),
                    k_a.as_ref(),
                    p_features,
                    x_tr_fro,
                )
            })
            .collect();
    }
    if !x_finite {
        // The refit route's answer for a non-finite half (`split_column_r`).
        return vec![0.0; n_cols];
    }
    // `t't` of the columns whose lower bound `‖X̃_tr'z‖²/‖z‖²` does not
    // clear the kernel's absolute exit: one more pass over the training
    // half, for those columns only.
    let kept = |col: usize| {
        let (wn, z_norm) = (wn2[col].sqrt(), z_norms[col]);
        k1_keeps_w(wn, z_norm, n_tr, p_features, x_tr_fro)
    };
    let need_tt: Vec<usize> = (0..n_cols)
        .filter(|&col| {
            kept(col) && wn2[col] / (z_norms[col] * z_norms[col]) < crate::fit::NIPALS_ABS_FLOOR
        })
        .collect();
    let tt = nr_train_tt(
        x,
        sp,
        w_tr.as_ref(),
        sw_tr.as_ref(),
        scale,
        &need_tt,
        y_tr.as_ref(),
    );
    (0..n_cols)
        .map(|col| {
            let tt_ok = need_tt
                .iter()
                .position(|&c| c == col)
                .is_none_or(|u| tt[u] / wn2[col] >= crate::fit::NIPALS_ABS_FLOOR || tt[u].is_nan());
            nr_column_r_primal(
                n_te,
                |i| t_te[(i, col)],
                |i| y_te[(i, col)],
                kept(col) && tt_ok,
            )
        })
        .collect()
}

/// Whether the K = 1 fit keeps its component on the first of
/// `fit::pls1_kernel`'s exits: `wn = ‖X̃'z‖` (`‖z‖ = z_norm`) at least
/// `max(NIPALS_ABS_FLOOR, w_rel_floor(n, p, x_fro, z_norm))`. A NaN `wn`
/// passes, as in the kernel. The second exit, `t't` under
/// `NIPALS_ABS_FLOOR`, needs `t't = ‖X̃X̃'z‖²/wn²`, which is at least
/// `wn²/‖z‖²` (Cauchy-Schwarz on `wn² = z'X̃X̃'z`).
fn k1_keeps_w(wn: f64, z_norm: f64, n: usize, p: usize, x_fro: f64) -> bool {
    use crate::fit::{NIPALS_ABS_FLOOR, w_rel_floor};
    wn >= NIPALS_ABS_FLOOR.max(w_rel_floor(n, p, x_fro, z_norm)) || wn.is_nan()
}

/// `‖X̃_tr X̃_tr'z‖²` of the training targets `z_tr` (`n_tr × ·`) columns
/// `cols`, from one more pass over the column blocks of the training half
/// of `x` as [`nr_split_r`] centres and scales it (`X̃_tr X̃_tr' = Σ A A'`
/// over the blocks `A`). No pass when `cols` is empty.
fn nr_train_tt(
    x: MatRef<'_, f64>,
    sp: &SplitIdx,
    w_tr: Option<&Col<f64>>,
    sw_tr: Option<&Col<f64>>,
    scale: ColRef<'_, f64>,
    cols: &[usize],
    z_tr: MatRef<'_, f64>,
) -> Vec<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    if cols.is_empty() {
        return Vec::new();
    }
    let n_tr = sp.tr.len();
    let z = Mat::<f64>::from_fn(n_tr, cols.len(), |i, u| z_tr[(i, cols[u])]);
    let mut t = Mat::<f64>::zeros(n_tr, cols.len());
    crate::linalg::centered_blocks(
        x,
        &sp.tr,
        &[],
        w_tr.map(Col::as_ref),
        sw_tr.map(Col::as_ref),
        None,
        scale,
        // The width `linalg::stream_blocks_bytes(n_tr, 0)` counts (change
        // together).
        crate::linalg::stream_block_width(n_tr),
        |_, a, _| {
            let g = crate::linalg::mat_mul(a.transpose(), z.as_ref(), Par::Seq);
            matmul(t.as_mut(), Accum::Add, a, g.as_ref(), 1.0, Par::Seq);
        },
    );
    (0..cols.len())
        .map(|u| (0..n_tr).map(|i| t[(i, u)] * t[(i, u)]).sum())
        .collect()
}

/// One column of a split on route A of the no-refit body, from its
/// test-half scores `s` and √w-scaled test-half outcome `y`, both of
/// length `n_te`: `r = 0` on a degenerate test-half `y`, when the fit keeps
/// no component (`keeps` false: a constant training `y`, or one of
/// `fit::pls1_kernel`'s exits) and on constant scores; otherwise the
/// correlation, the refit route's answer up to rounding.
fn nr_column_r_primal(
    n_te: usize,
    s: impl Fn(usize) -> f64,
    y: impl Fn(usize) -> f64,
    keeps: bool,
) -> f64 {
    let ms = scaled_moments(n_te, &s, None);
    let my = scaled_moments(n_te, &y, None);
    if my.is_constant(n_te) || !keeps || ms.is_constant(n_te) {
        return 0.0;
    }
    pearson_scaled(n_te, s, y, &ms, &my)
}

/// [`nr_split_r`] on route A with the test half scored from moments,
/// unweighted: the training half is read block by block as on route A
/// (`linalg::centered_blocks` on the training rows only), giving
/// `T = X̃_tr'(Y_tr − 1·c')` and `h = X̃_tr'1`, `c` the columns' all-row
/// means (`am.y_shift`); the test half enters once, through its
/// [`TestMoments`]. With `ȳ'_A` the mean of `y_A − c`, column `c`'s
/// coefficient is `v = T_c − ȳ'_A·h = X̃_tr'C_A y_tr`, `σ·X̃_tr'z` for the
/// refit route's target `z` and the scale `σ` of its standardization, and
/// with `X̃_A'(y_A − c) = T_c + (X̃_A'1 − h)·ȳ'_A` (the two frames differ by a
/// column shift), `X̃_B'(y_B − c) = X̃'(y − c) − X̃_A'(y_A − c)` from the
/// call's [`AllRowMoments`] `am`. [`r_from_moments`] gives the correlation
/// on `y − c`, which is the correlation on `y`.
///
/// The truncation rule is route A's on `‖X̃_tr'z‖ = ‖v‖/σ` ([`k1_keeps_w`]):
/// a column it drops gives `r = 0`, as do a degenerate test-half `y` and a
/// constant training `y`. The rest of the gate guards this arm's
/// arithmetic against route A's test-row arm, which forms the scores from
/// the test rows: `err` bounds the distance of `t_c` from
/// `σ·‖C_B X̃_te g‖`, `g = X̃_tr'z`. It is
/// `F_te·((p + n_tr + 4)·ε·‖X̃_tr‖_F·(‖y_tr − c‖ + |ȳ'_A|·‖u‖) + n_tr·ε·‖X̃_tr‖_F·‖u‖·|ȳ'_A|)`
/// for `v` against `σ·g` (`F_te` in place of `‖X̃_te‖_F`, of which it is an
/// upper bound), plus the moment arithmetic's `err`
/// ([`TestMoments::score_norms`]), plus the test-row arm's centring of its
/// scores, `(n_te + 2)·ε·‖s‖_ref`. A column is decided here when
/// `t_c ≥ NR_RESOLVE_BAND·err`, [`moments_resolved`] holds at
/// `NR_RESOLVE_BAND` (the conditioning of the scores and the constant
/// tests), `r` is finite and the kept component's `t't` clears the
/// kernel's exit through its lower bound `‖X̃_tr'z‖²/‖z‖²`. Any other column
/// is `None`: the caller computes it on route A's test-row arm
/// ([`nr_split_r`]).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn nr_split_r_moments(
    x: MatRef<'_, f64>,
    y_mat: MatRef<'_, f64>,
    sp: &SplitIdx,
    scale: ColRef<'_, f64>,
    am: &AllRowMoments,
) -> Vec<Option<f64>> {
    use crate::linalg::row_subset;
    let (n_cols, p_features) = (y_mat.ncols(), x.ncols());
    let (n_tr, n_te) = (sp.tr.len(), sp.te.len());
    let y_te = row_subset(y_mat, &sp.te);
    let shifted = |idx: &[usize]| {
        Mat::<f64>::from_fn(idx.len(), n_cols, |i, c| y_mat[(idx[i], c)] - am.y_shift[c])
    };
    let y_tr = shifted(&sp.tr);
    let y_te_c = shifted(&sp.te);
    let mut t = Mat::<f64>::zeros(p_features, n_cols);
    let mut h = Col::<f64>::zeros(p_features);
    let x_tr_fro = crate::linalg::centered_blocks(
        x,
        &sp.tr,
        &[],
        None,
        None,
        None,
        scale,
        // The width `linalg::stream_blocks_bytes(n_tr, 0)` counts (change
        // together).
        crate::linalg::stream_block_width(n_tr),
        |j0, a, _| {
            let g = crate::linalg::mat_mul(a.transpose(), y_tr.as_ref(), Par::Seq);
            t.subrows_mut(j0, a.ncols()).copy_from(&g);
            for j in 0..a.ncols() {
                h[j0 + j] = (0..n_tr).map(|i| a[(i, j)]).sum();
            }
        },
    );
    let tm = TestMoments::new(x, &sp.te, am.mean.as_ref(), scale);
    let eps = f64::EPSILON;
    let u_norm = (n_tr as f64).sqrt();
    let prod_err = (p_features + n_tr + 4) as f64 * eps * x_tr_fro;
    let h_err = n_tr as f64 * eps * x_tr_fro * u_norm;

    (0..n_cols)
        .map(|col| {
            let my = scaled_moments(n_te, |i| y_te[(i, col)], None);
            if my.is_constant(n_te) {
                return Some(0.0);
            }
            let (z, _, y_scale) = nr_train_z(col, y_mat, sp, None, None);
            let z_norm = z.norm_l2();
            if z_norm == 0.0 {
                return Some(0.0);
            }
            let y_bar = plain_mean(n_tr, |i| y_tr[(i, col)]);
            let v = Col::<f64>::from_fn(p_features, |j| t[(j, col)] - y_bar * h[j]);
            let wn = v.norm_l2() / y_scale;
            if !k1_keeps_w(wn, z_norm, n_tr, p_features, x_tr_fro) {
                return Some(0.0);
            }
            let xb_y = Col::<f64>::from_fn(p_features, |j| {
                let xa_sum = am.x_sum[j] - tm.xb_sum[j];
                am.xty[(j, col)] - t[(j, col)] - (xa_sum - h[j]) * y_bar
            });
            let (y_b_sum, y_b_ss) = y_b_sums(n_te, |i| y_te_c[(i, col)]);
            let r = r_from_moments(v.as_ref(), xb_y.as_ref(), y_b_sum, y_b_ss, &tm);
            let sn = tm.score_norms(v.as_ref(), am.x_sum.as_ref(), n_tr);
            let y_tr_norm = (0..n_tr)
                .map(|i| y_tr[(i, col)] * y_tr[(i, col)])
                .sum::<f64>()
                .sqrt();
            let err = sn.f_te
                * (prod_err * (y_tr_norm + y_bar.abs() * u_norm) + h_err * y_bar.abs())
                + sn.err
                + (n_te + 2) as f64 * eps * sn.s_ref;
            // `a >= b` conjunctions, so a NaN leaves the column to the
            // test-row arm.
            let resolved = sn.t_c >= NR_RESOLVE_BAND * err
                && moments_resolved(&sn, y_b_sum, y_b_ss, n_te, NR_RESOLVE_BAND)
                && (wn / z_norm).powi(2) >= crate::fit::NIPALS_ABS_FLOOR
                && r.is_finite();
            resolved.then_some(r)
        })
        .collect()
}

/// The refit route's training target of column `col` (`z`, standardized
/// and √w-scaled), with the mean and scale of that standardization: the
/// refit route's call, call for call (`split_half_r`; change together).
#[allow(clippy::similar_names)]
fn nr_train_z(
    col: usize,
    y_mat: MatRef<'_, f64>,
    sp: &SplitIdx,
    w_tr: Option<&Col<f64>>,
    sw_tr: Option<&Col<f64>>,
) -> (Col<f64>, f64, f64) {
    use crate::linalg::{standardize1, standardize1_weighted};
    let n_tr = sp.tr.len();
    let y_tr_raw = Col::<f64>::from_fn(n_tr, |i| y_mat[(sp.tr[i], col)]);
    let (zs, y_bar, y_scale) = match w_tr {
        Some(w) => standardize1_weighted(y_tr_raw.as_ref(), Some(w.as_ref())),
        None => standardize1(y_tr_raw.as_ref()),
    };
    let z = match sw_tr {
        Some(sw) => Col::<f64>::from_fn(n_tr, |i| zs[i] * sw[i]),
        None => zs,
    };
    (z, y_bar, y_scale)
}

/// [`nr_split_r`] on route G: the same correlations, with the products of
/// the halves taken from the blocks of `g` ([`HeldOutGram::part_blocks`])
/// instead of from X. `g` is `HeldOutGram::from_x` on the call's X, weights
/// and scale, with `with_row_weights(w_norm)` under weights, so the scores
/// are `t_te = M·(√w_tr ⊙ y_tr)` with
/// `M = (G_BA − √w_B√w_A'G_AA/Σw_A)·P_A`, `P_A = I − √w_A√w_A'/Σw_A`
/// (`1` for `√w` unweighted), and the test half's `y` is `√w_te ⊙ y_te`.
/// `M` and `K_A = P_A G_AA P_A` are multiplied by the constants that
/// `nr_split_r`'s renormalization of each half's weights puts on its
/// products, so they are the products of the halves `nr_split_r` and the
/// refit route form. A half of zero total weight, which `nr_split_r` runs
/// unweighted, is formed from `plain`, the Gram matrix without the row
/// scaling ([`k1_route_gram`]), with each half's own renormalized weights
/// or none. Every column is decided by [`nr_column_r_dual`].
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn nr_split_r_gram(
    g: &HeldOutGram,
    plain: Option<&HeldOutGram>,
    y_mat: MatRef<'_, f64>,
    sp: &SplitIdx,
    w_norm: Option<ColRef<'_, f64>>,
) -> Vec<f64> {
    use crate::linalg::row_subset;
    let n_cols = y_mat.ncols();
    let p_features = g.n_features();
    let n_tr = sp.tr.len();
    // The renormalized weights of each half and their roots, as nr_split_r
    // forms them (change together).
    let w_tr = renormalized_slice_weights(w_norm, &sp.tr);
    let w_te = renormalized_slice_weights(w_norm, &sp.te);
    let root = |w: &Col<f64>| Col::<f64>::from_fn(w.nrows(), |i| w[i].sqrt());
    let sw_tr = w_tr.as_ref().map(root);
    let sw_te = w_te.as_ref().map(root);
    // `n_half / Σ_half w`: the factor by which a half's renormalized weights
    // exceed `w_norm` on it.
    let renorm = |idx: &[usize]| match w_norm {
        None => Some(1.0),
        Some(w) => {
            let sum = half_weight_sum(w, idx);
            (sum > 0.0).then(|| idx.len() as f64 / sum)
        }
    };
    let (k_a, m) = if let (Some(c_tr), Some(c_te)) = (renorm(&sp.tr), renorm(&sp.te)) {
        let sw_all = w_norm.map(crate::linalg::sqrt_col);
        let (mut k_a, mut m) = g.part_blocks(&sp.tr, &sp.te, sw_all.as_ref().map(Col::as_ref));
        if w_norm.is_some() {
            let times = |a: &mut Mat<f64>, c: f64| {
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        a[(i, j)] *= c;
                    }
                }
            };
            times(&mut k_a, c_tr);
            times(&mut m, (c_tr * c_te).sqrt());
        }
        (k_a, m)
    } else {
        // `k1_route_gram` keeps `plain` for every split with a half of zero
        // total weight.
        let plain = plain.expect("the Gram matrix without the row scaling");
        // Each row carries its half's renormalized weight, or one.
        let mut wp = Col::<f64>::from_fn(plain.g().nrows(), |_| 1.0);
        for (idx, wh) in [(&sp.tr, &w_tr), (&sp.te, &w_te)] {
            if let Some(wh) = wh {
                for (i, &row) in idx.iter().enumerate() {
                    wp[row] = wh[i];
                }
            }
        }
        let sw = crate::linalg::sqrt_col(wp.as_ref());
        plain
            .with_row_weights(wp.as_ref())
            .part_blocks(&sp.tr, &sp.te, Some(sw.as_ref()))
    };

    let scale_rows = |m: Mat<f64>, sw: Option<&Col<f64>>| match sw {
        Some(sw) => Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * sw[i]),
        None => m,
    };
    let y_tr = scale_rows(row_subset(y_mat, &sp.tr), sw_tr.as_ref());
    let y_te = scale_rows(row_subset(y_mat, &sp.te), sw_te.as_ref());
    let t_te = crate::linalg::mat_mul(m.as_ref(), y_tr.as_ref(), Par::Seq);

    // The trace can lose its relative accuracy when `‖X̃_tr‖_F ≪ ‖X̃_A‖_F`;
    // it enters only the relative floor.
    let x_tr_fro = (0..n_tr).map(|i| k_a[(i, i)]).sum::<f64>().max(0.0).sqrt();
    (0..n_cols)
        .map(|col| {
            nr_column_r_dual(
                col,
                y_mat,
                sp,
                w_tr.as_ref(),
                sw_tr.as_ref(),
                t_te.as_ref(),
                y_te.as_ref(),
                k_a.as_ref(),
                p_features,
                x_tr_fro,
            )
        })
        .collect()
}

/// One column `col` of a split on the no-refit routes that hold the
/// training kernel `k_a = X̃_tr X̃_tr'` (routes B and G), from the test-half
/// scores `t_te` and √w-scaled test-half outcomes `y_te`: `r = 0` on a
/// degenerate test-half `y`, a training target that is constant or not
/// finite, a fit that keeps no component ([`k1_keeps`]) and constant scores;
/// otherwise the correlation. Every one of those is the refit route's
/// answer for the same case, so every column is decided here.
#[allow(clippy::too_many_arguments)]
fn nr_column_r_dual(
    col: usize,
    y_mat: MatRef<'_, f64>,
    sp: &SplitIdx,
    w_tr: Option<&Col<f64>>,
    sw_tr: Option<&Col<f64>>,
    t_te: MatRef<'_, f64>,
    y_te: MatRef<'_, f64>,
    k_a: MatRef<'_, f64>,
    p_features: usize,
    x_tr_fro: f64,
) -> f64 {
    let n_te = sp.te.len();
    let s = |i: usize| t_te[(i, col)];
    let yv = |i: usize| y_te[(i, col)];
    let ms = scaled_moments(n_te, s, None);
    let my = scaled_moments(n_te, yv, None);
    if my.is_constant(n_te) {
        return 0.0;
    }
    let (z, _, _) = nr_train_z(col, y_mat, sp, w_tr, sw_tr);
    let z_norm = z.norm_l2();
    if z_norm == 0.0
        || !z_norm.is_finite()
        || !k1_keeps(k_a, &z, z_norm, p_features, x_tr_fro)
        || ms.is_constant(n_te)
    {
        return 0.0;
    }
    pearson_scaled(n_te, s, yv, &ms, &my)
}

/// Whether the K = 1 fit on a training block with kernel `k_a = X̃X̃'` keeps
/// its component for the target `z` (`‖z‖ = z_norm`): `fit::pls1_kernel`'s
/// exits on `‖X̃'z‖ = √(z'K z)` (under `max(NIPALS_ABS_FLOOR, w_rel_floor)`,
/// with `‖X̃‖_F = x_fro` and `p` columns) and on
/// `t't = ‖K z‖²/(z'K z)` (under `NIPALS_ABS_FLOOR`). `z'K z` is formed from
/// a rounded `K` and resolves `‖X̃'z‖` only to about `√(p·ε)·‖X̃‖_F·‖z‖`,
/// far above the relative floor, so on a training `y` orthogonal to `X̃`,
/// up to rounding, this can keep a component the refit route drops. A NaN
/// passes both tests, as in the kernel.
fn k1_keeps(k_a: MatRef<'_, f64>, z: &Col<f64>, z_norm: f64, p: usize, x_fro: f64) -> bool {
    let n = z.nrows();
    let kz = crate::linalg::mat_vec(k_a, z.as_ref(), Par::Seq);
    let wn2: f64 = (0..n).map(|i| z[i] * kz[i]).sum();
    let wn = wn2.max(0.0).sqrt();
    if !k1_keeps_w(wn, z_norm, n, p, x_fro) {
        return false;
    }
    let tt = kz.squared_norm_l2() / (wn * wn);
    tt >= crate::fit::NIPALS_ABS_FLOOR || tt.is_nan()
}

/// Per-column z̄ values for `split_perm_nr` over `splits` (length
/// `cols.len()`; column 0 is the observed y, the others the null columns of
/// `cols`). Factored out of
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
/// map. `X̃_tr` is now `diag(√w_tr)·X_c,tr` with `X_c,tr` centred on the
/// *weighted* train-half means (and divided by the all-row column scale),
/// and the train y that `pls1_fit` sees is
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
/// `max(1e-14, fit::w_rel_floor(n_tr, p, ‖X̃_tr‖_F, ‖z‖))`, or when
/// `t't < 1e-14`; the test-half scores are then exactly zero and `r = 0`.
/// Every route applies those exits itself, to its own `‖X̃_tr'z‖` and
/// `t't = ‖X̃_tr X̃_tr'z‖²/‖X̃_tr'z‖²`, and gives `r = 0` where they drop
/// the component, so every column is decided without a refit:
///
/// - Routes B and G hold the training kernel `K_A = X̃_tr X̃_tr'` and read
///   `‖X̃_tr'z‖ = √(z'K_A z)` and `‖K_A z‖` from it ([`k1_keeps`]).
/// - Route A forms `X̃_tr'z` itself and reads `t't` through its lower bound
///   `‖X̃_tr'z‖²/‖z‖²`; a column whose bound does not clear `1e-14` gets
///   its `t't` from one more pass over the training half ([`nr_train_tt`]).
/// - The moment arm reads `‖X̃_tr'z‖` from its coefficient; a column whose
///   `t't` bound does not clear `1e-14`, or whose moments it cannot trust,
///   goes to route A's test-row arm. With the split `design` handed over
///   (`splits` its splits) and admitted by `cell_sums::cells_eligible`, the
///   arm takes every split's training sums from the cells ([`nr_cell_r`]),
///   whose rounding differs from a split's own pass, so near the floors it
///   can decide differently.
///
/// Near the floors (a training `y` orthogonal to `X̃_tr` up to rounding)
/// each route's decision rests on its own rounding, so routes can then
/// differ by O(1) in `r`; elsewhere they agree to rounding. Two cases are
/// exact on every route: a degenerate test-half `y` (`constant_to_rounding`
/// sees the same bits) and a constant training `y` (`z` is exactly zero).
/// `y` enters divided by a power of two near its largest entry
/// (`nr_outcome_matrix`), so no norm overflows or underflows at any
/// magnitude of `y`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn split_perm_nr_zbars(
    x: MatRef<'_, f64>,
    cols: &crate::resample::Columns<'_>,
    splits: &[SplitIdx],
    k: usize,
    w_norm: Option<ColRef<'_, f64>>,
    scale: ColRef<'_, f64>,
    opts: &ConfirmatoryTestOpts,
    gram: Option<&HeldOutGram>,
    design: Option<&crate::resample::SplitDesign>,
) -> PlsKitResult<Vec<f64>> {
    use crate::resample::split_sizes;
    debug_assert!(design.is_none_or(|d| std::ptr::eq(d.splits.as_ptr(), splits.as_ptr())));

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
    let (n_train, n_test) = split_sizes(n, k);
    let p_features = x.ncols();
    let n_cols = cols.len();

    // The J splits are fixed across all B null columns: the fixed linear map
    // in this function's "Why no refits" doc depends only on X and the
    // split, so every null column reuses it. Only the (tr, te) index pairs
    // are held; each split then reads X block by block (`nr_split_r`) and
    // holds `M` and `K_A` (route B) or one block's `X̃_tr'Y` (route A), or
    // reads the blocks of one all-row Gram matrix (`nr_split_r_gram`, route
    // G) and holds `K_A` and `M`. No half of X is written; a column the
    // moment arm leaves undecided is computed afterwards on route A, the
    // splits that have one in batches of `resample::map_units`.

    // The null columns come from `cols`, the way every route takes them, so
    // every route sees the same null columns at a given seed.
    let y_mat = nr_outcome_matrix(cols);

    // Choose the route once, outside the per-split loop: all splits share
    // (n_train, n_test) since split_sizes depends only on (n, k). `k1_route`
    // owns the rule; `gram` is the `split_nb` gate's Gram matrix of `x` on
    // a rerouted call. Route G's Gram matrix is built here, once per call.
    let route = k1_route(
        n,
        n_train,
        n_test,
        p_features,
        n_cols,
        splits.len(),
        gram.is_some(),
    );
    let g = (route == K1Route::G).then(|| k1_route_gram(x, w_norm, scale, gram, splits));
    // Route A scores the test halves from moments when `test_moments_route`
    // admits the loop: the training half streamed, plus one split's
    // `TestMoments`.
    let moment_bytes = crate::linalg::stream_blocks_bytes(n_train, 0)
        .saturating_add(test_moments_bytes(n_test, p_features));
    let moments = route == K1Route::A
        && test_moments_route(
            n,
            n_test,
            p_features,
            n_cols,
            splits.len(),
            w_norm.is_some(),
        );
    // With cells, the moment arm sums every split's training rows from them.
    let batch = crate::cell_sums::cell_batch(n_cols);
    let cells = design.filter(|d| {
        moments && crate::cell_sums::cells_eligible(d, p_features, batch, w_norm.is_some())
    });
    let am = (moments && cells.is_none())
        .then(|| AllRowMoments::new(x, scale, n_cols, &|c| y_mat.col(c).to_owned()));
    let unit_bytes = if am.is_some() {
        moment_bytes
    } else {
        nr_split_unit_bytes(
            n_train,
            n_test,
            route,
            g.as_ref().is_some_and(|(_, plain)| plain.is_some()),
        )
    };

    // Pass 1, splits in batches of `resample::map_units`: every column the
    // route decides (all of them on routes A, B and G). A split reads X block by block (`nr_split_r`), or its
    // training half block by block and its test half once for its moments
    // (`nr_split_r_moments`), or the blocks of `g` (`nr_split_r_gram`), and
    // writes no half of X. With cells, the moment arm reads no split's rows
    // per column: one pass per batch of columns gives every split's sums
    // (`nr_cell_r`).
    let rows: Vec<Vec<Option<f64>>> = if let Some(d) = cells {
        let frame = CellFrame::new(x, scale, d, false);
        let levels = d.cells().trailing_zeros() as usize;
        cell_rows(
            &frame,
            d,
            n_cols,
            1,
            &|c, out| out.col_mut(0).copy_from(y_mat.col(c)),
            &|cs, j, c| nr_cell_r(&frame, cs, j, c, n_train, n_test, levels),
        )
    } else {
        crate::resample::map_units(splits.len(), unit_bytes, |j| match (&g, &am) {
            (Some((g, plain)), _) => {
                nr_split_r_gram(g, plain.as_deref(), y_mat.as_ref(), &splits[j], w_norm)
                    .into_iter()
                    .map(Some)
                    .collect()
            }
            (None, Some(am)) => nr_split_r_moments(x, y_mat.as_ref(), &splits[j], scale, am),
            (None, None) => nr_split_r(
                x,
                y_mat.as_ref(),
                &splits[j],
                w_norm,
                scale,
                route == K1Route::B,
            )
            .into_iter()
            .map(Some)
            .collect(),
        })
    };
    // Pass 2 (the moment arm only): the splits with undecided columns run
    // in batches of `resample::map_units`, route A's stream blocks counted
    // per split. The columns left undecided are route A's test-row arm
    // (`nr_split_r`) on the same split and columns.
    let rows = crate::resample::fill_pending(
        rows,
        nr_split_unit_bytes(n_train, n_test, K1Route::A, false),
        |j, pending| {
            let y_sub =
                Mat::<f64>::from_fn(y_mat.nrows(), pending.len(), |i, u| y_mat[(i, pending[u])]);
            nr_split_r(x, y_sub.as_ref(), &splits[j], w_norm, scale, false)
        },
    );
    let z_rows: Vec<Vec<f64>> = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                // ±0.9999 pre-atanh clamp mirrored from nb_test (change
                // together): keeps the statistic identical to split_nb's and
                // z̄ finite at |r| = 1.
                .map(|r| r.clamp(-0.9999, 0.9999).atanh())
                .collect()
        })
        .collect();
    Ok(zbars_from_rows(n_cols, &z_rows))
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
    scale: ColRef<'_, f64>,
    opts: &ConfirmatoryTestOpts,
    null_map: Option<crate::resample::NullMap<'_>>,
    gram: Option<&HeldOutGram>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    // The splits, then one child seed per null column: run_split_perm's
    // draws, so both routes test the same splits on the same null columns.
    // `draw_design` also rejects n < k+5.
    let design = draw_design(x.nrows(), k, n_splits, rng)?;
    let seeds = crate::rng::child_seeds(rng, n_perm);
    let cols = crate::resample::Columns {
        y: y.as_mat(),
        seeds: &seeds,
        null_map,
    };
    let z_bar = split_perm_nr_zbars(
        x,
        &cols,
        &design.splits,
        k,
        w_norm,
        scale,
        opts,
        gram,
        Some(&design),
    )?;
    Ok(split_perm_result(&z_bar, n_perm))
}

// ──────────────────────────────────────────────────────────────────────────────
// score test (Welch-Satterthwaite generalized χ²)
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::many_single_char_names)]
fn run_score(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
) -> PlsKitResult<RunResult> {
    use crate::linalg::{standardize, standardize_weighted, standardize1, standardize1_weighted};

    let n = x.nrows();

    // Standardize with weighted moments when weights are present (matching
    // pls1_fit's path); the √w row-scaling below then sits on top.
    // Pre-standardized input is used as given, copied only when it is not
    // column-major (`linalg::col_major_or_copy`), or divided by a power of
    // two `c` when far from unit scale (`linalg::pre_standardized_input`):
    // that leaves the p-value as it is and divides the statistic by `c²`,
    // multiplied back below.
    let (x_copy, c) = if opts.pre_standardized {
        crate::linalg::pre_standardized_input(x, w_norm)
    } else {
        (None, 1.0)
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
    // A non-finite standardized X is non-finite input, as `pls1_fit`
    // answers it, rather than an infinite statistic.
    if xs_std.is_some() {
        crate::fit::check_finite_mat(xs_eff)?;
    }

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

    let mut r = score_result(t_obs, &lambdas);
    // Twice by `c`: `c²` alone can leave the `f64` range.
    r.statistic *= c;
    r.statistic *= c;
    Ok(r)
}

/// The score test's p-value for `T = t_obs` against the Welch-Satterthwaite
/// approximation `T ~ a·χ²(df)` from the Gram matrix eigenvalues `lambdas`.
fn score_result(t_obs: f64, lambdas: &Col<f64>) -> RunResult {
    let s1: f64 = (0..lambdas.nrows()).map(|i| lambdas[i]).sum();
    let s2: f64 = (0..lambdas.nrows()).map(|i| lambdas[i].powi(2)).sum();
    let pvalue = if s1.abs() < 1e-15 || s2 < 1e-30 {
        1.0
    } else {
        chi2_sf(t_obs / (s2 / s1), s1 * s1 / s2)
    };
    RunResult {
        pvalue,
        statistic: t_obs,
        rho_hat: None,
    }
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
    scale: ColRef<'_, f64>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use crate::fit::{FitOpts, KSpec, pls1_fit_impl};
    use crate::linalg::{col_row_subset, standardize1, standardize1_weighted};
    use crate::resample::{one_split, split_sizes};

    let n = x.nrows();
    // Mirror of draw_splits' n ≥ k+5 guard (change together).
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, _) = split_sizes(n, k);
    let (tr, te) = one_split(n, n_train, rng);

    let y_tr = col_row_subset(y, &tr);
    let y_te = col_row_subset(y, &te);

    let prep = prepare_split(x, &SplitIdx { tr, te }, w_norm, scale);
    // A non-finite training half is the error the fit below would return;
    // a non-finite held-out half would make every prediction non-finite.
    // Mirrors `prepare_cv_fold`'s two block checks (change together).
    if !prep.x_finite {
        return Err(PlsKitError::NonFiniteInput);
    }
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
    let y_pred_tr = crate::linalg::mat_vec(xs_tr.as_ref(), m.coef.as_ref(), par);
    Ok(e_result(&ys_tr, &y_pred_tr, &ys_te, &y_pred))
}

/// The `e` statistic `log e` and `p = min(1, 1/e)` from the standardized
/// training and test outcomes and their predictions by the training fit.
#[allow(clippy::similar_names)]
fn e_result(
    ys_tr: &Col<f64>,
    y_pred_tr: &Col<f64>,
    ys_te: &Col<f64>,
    y_pred: &Col<f64>,
) -> RunResult {
    let n_te = ys_te.nrows();
    // Universal inference fixes the numerator density on the training half:
    // σ²_alt is the residual MLE from predicting the training X with the fitted
    // model, evaluated against training y. Computing it on the test half would
    // make the likelihood ratio data-dependent and break the e-value guarantee.
    let n_tr = ys_tr.nrows();
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

    RunResult {
        pvalue: p,
        statistic: log_e,
        rho_hat: None,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{FitOpts, KSpec, pls1_fit};
    use crate::linalg::{centered_moments, constant_to_rounding, normalize_weights};
    use crate::test_support::{orthonormal_basis, project_off, signal_data, synth};

    /// The wide Gram units, called on parts chosen to make each column
    /// degenerate, decide it from the all-row Gram matrix alone, with the
    /// primal route's answer where that answer is exact: route G and route
    /// B of the K = 1 no-refit body (`nr_split_r_gram`, `nr_split_r`), the
    /// n-space split and fold units at k = 2, and the K = 1 closed form.
    /// X has 20 rows each taken twice (rows `2m`, `2m + 1`), uncentred.
    /// - A constant training outcome: `r = 0` and zero predictions, the
    ///   primal's bits.
    /// - An outcome antisymmetric across the pairs of a training part made
    ///   of whole pairs, so `X̃_tr'z` and `K_A z` vanish: the zero model,
    ///   the primal's bits.
    /// - An outcome orthogonal to a rank-3 `X̃_tr` up to rounding: finite.
    /// - `k = 8` on a 6-row training part (rank 5): a fit truncated to at
    ///   most 5 components, on both routes, with finite coefficients.
    /// - A column constant over the training part: the primal's `r` to
    ///   `1e-10`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::similar_names
    )]
    fn degenerate_columns_are_decided_without_x() {
        use crate::dual_route::{NspaceGram, NspaceOutcome, pls1_nspace_kernel};
        use crate::resample::Columns;
        let (n, p) = (40_usize, 300_usize);
        let (base, _) = signal_data(20, p, 201);
        let x = Mat::<f64>::from_fn(n, p, |i, j| base[(i / 2, j)] + 0.3 * j as f64);
        let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
        let g = block_gram(x.as_ref(), scale.as_ref(), None);
        let sp = SplitIdx {
            tr: (0..20).collect(),
            te: (20..40).collect(),
        };
        let prep = prepare_split(x.as_ref(), &sp, None, scale.as_ref());
        let noise = Col::<f64>::from_fn(n, |i| ((i * 13) % 17) as f64 - 8.0);
        let a = |i: usize| 1.0 + ((i / 2) % 5) as f64;
        let constant = Col::<f64>::from_fn(n, |i| if i < 20 { 3.0 } else { noise[i] });
        let pairs = Col::<f64>::from_fn(n, |i| {
            if i >= 20 {
                noise[i]
            } else if i % 2 == 0 {
                a(i)
            } else {
                -a(i)
            }
        });
        // Every K = 1 and k = 2 split route on one outcome, and the primal.
        let split_routes = |y: &Col<f64>| {
            let y_mat = nr_outcome_matrix(&Columns {
                y: y.as_ref().as_mat(),
                seeds: &[],
                null_map: None,
            });
            let route_g = nr_split_r_gram(&g, None, y_mat.as_ref(), &sp, None)[0];
            let route_b =
                nr_split_r(x.as_ref(), y_mat.as_ref(), &sp, None, scale.as_ref(), true)[0];
            let nspace = split_columns_nspace(&g, &sp, 1, &[2], &|_| y.clone())[0][0];
            let primal_1 = split_column_r(&prep, &sp, &|i| y[i], &[1], None)[0];
            let primal_2 = split_column_r(&prep, &sp, &|i| y[i], &[2], None)[0];
            (route_g, route_b, nspace, primal_1, primal_2)
        };
        for (what, y) in [("constant", &constant), ("pairs", &pairs)] {
            let (rg, rb, rn, p1, p2) = split_routes(y);
            for (route, r, want) in [("G", rg, p1), ("B", rb, p1), ("n-space k=2", rn, p2)] {
                assert_eq!(r.to_bits(), 0.0_f64.to_bits(), "{what}, {route}");
                assert_eq!(r.to_bits(), want.to_bits(), "{what}, {route}: primal");
            }
        }

        // Folds of whole pairs: every training set is whole pairs, so the
        // antisymmetric outcome is orthogonal to X̃_tr on every fold.
        let folds: Vec<Vec<usize>> = (0..5).map(|f| (8 * f..8 * f + 8).collect()).collect();
        let y_all = Col::<f64>::from_fn(n, |i| if i % 2 == 0 { a(i) } else { -a(i) });
        let y_const = Col::<f64>::from_fn(n, |_| 3.0);
        for (what, y) in [("pairs", &y_all), ("constant", &y_const)] {
            let y_mat = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
            let closed = crate::dual_route::pls1_cv_r2_columns(&g, y_mat.as_ref(), &folds)
                .expect("closed form");
            let fold_routes = |route, k| {
                first_row(
                    &pooled_cv_r2_columns(
                        route,
                        x.as_ref(),
                        &folds,
                        None,
                        scale.as_ref(),
                        &[k],
                        None,
                        1,
                        &|_| y.clone(),
                    )
                    .expect("folds"),
                )
            };
            let primal_1 = fold_routes(ReplicateRoute::Primal, 1);
            assert_eq!(closed[0].to_bits(), primal_1[0].to_bits(), "{what}, K = 1");
            let nspace = fold_routes(ReplicateRoute::Nspace, 2);
            let primal_2 = fold_routes(ReplicateRoute::Primal, 2);
            assert_eq!(nspace[0].to_bits(), primal_2[0].to_bits(), "{what}, k = 2");
        }

        // An outcome orthogonal to a rank-3 X̃_tr up to rounding.
        let a3 = Mat::<f64>::from_fn(n, 3, |i, j| ((i * (j + 3)) % 7) as f64 - 3.0);
        let b3 = Mat::<f64>::from_fn(3, p, |i, j| (((i + 1) * (j + 2)) % 11) as f64 - 5.0);
        let x3 = crate::linalg::mat_mul(a3.as_ref(), b3.as_ref(), Par::Seq);
        let scale3 = crate::linalg::held_out_scale(x3.as_ref(), None, false);
        let g3 = block_gram(x3.as_ref(), scale3.as_ref(), None);
        let prep3 = prepare_split(x3.as_ref(), &sp, None, scale3.as_ref());
        let ones = Col::<f64>::from_fn(20, |_| 1.0);
        let basis = orthonormal_basis(ones.as_ref(), prep3.xs_tr.as_ref(), 1e-8);
        let mut z = Col::<f64>::from_fn(20, |i| noise[i]);
        project_off(&basis, &mut z);
        let orth = Col::<f64>::from_fn(n, |i| if i < 20 { z[i] } else { noise[i] });
        let y_mat = nr_outcome_matrix(&Columns {
            y: orth.as_ref().as_mat(),
            seeds: &[],
            null_map: None,
        });
        let rg = nr_split_r_gram(&g3, None, y_mat.as_ref(), &sp, None)[0];
        let rb = nr_split_r(
            x3.as_ref(),
            y_mat.as_ref(),
            &sp,
            None,
            scale3.as_ref(),
            true,
        )[0];
        let rn = split_columns_nspace(&g3, &sp, 1, &[2], &|_| orth.clone())[0][0];
        for (route, r) in [("G", rg), ("B", rb), ("n-space k=2", rn)] {
            assert!(
                r.is_finite() && r.abs() <= 1.0,
                "orthogonal, {route}: {r:e}"
            );
        }

        // k = 8 on a 6-row training part.
        let xs6 = crate::linalg::standardize(base.subrows(0, 6)).0;
        let z6 = crate::linalg::standardize1(Col::<f64>::from_fn(6, |i| noise[i]).as_ref()).0;
        let gram6 = NspaceGram::new(xs6.as_ref(), Par::Seq);
        match pls1_nspace_kernel(&gram6.block(), z6.as_ref(), 8) {
            NspaceOutcome::Resolved { alpha, k_used } => {
                assert!(k_used <= 5, "n-space keeps {k_used}");
                assert!((0..6).all(|i| alpha[i].is_finite()));
            }
            NspaceOutcome::ZeroModel => panic!("an ordinary outcome is not the zero model"),
        }
        let fit6 = crate::fit::pls1_fit_prepared_fro(
            xs6.as_ref(),
            z6.as_ref(),
            8,
            None,
            crate::fit::ParChoice::Seq,
            xs6.norm_l2(),
        )
        .expect("primal");
        assert!(fit6.k_used <= 5, "primal keeps {}", fit6.k_used);

        // A column constant over the training part.
        let xc = Mat::<f64>::from_fn(n, p + 1, |i, j| {
            if j < p {
                base[(i / 2, j)] + 0.01 * i as f64
            } else if i < 20 {
                1.0
            } else {
                i as f64
            }
        });
        let scalec = crate::linalg::held_out_scale(xc.as_ref(), None, false);
        let gc = block_gram(xc.as_ref(), scalec.as_ref(), None);
        let prepc = prepare_split(xc.as_ref(), &sp, None, scalec.as_ref());
        let yc = Col::<f64>::from_fn(n, |i| base[(i / 2, 0)] + 0.1 * noise[i]);
        let y_mat = nr_outcome_matrix(&Columns {
            y: yc.as_ref().as_mat(),
            seeds: &[],
            null_map: None,
        });
        let rg = nr_split_r_gram(&gc, None, y_mat.as_ref(), &sp, None)[0];
        let rb = nr_split_r(
            xc.as_ref(),
            y_mat.as_ref(),
            &sp,
            None,
            scalec.as_ref(),
            true,
        )[0];
        let rn = split_columns_nspace(&gc, &sp, 1, &[2], &|_| yc.clone())[0][0];
        let p1 = split_column_r(&prepc, &sp, &|i| yc[i], &[1], None)[0];
        let p2 = split_column_r(&prepc, &sp, &|i| yc[i], &[2], None)[0];
        assert!(
            (rg - p1).abs() <= 1e-10,
            "constant column, G: {rg:e} vs {p1:e}"
        );
        assert!(
            (rb - p1).abs() <= 1e-10,
            "constant column, B: {rb:e} vs {p1:e}"
        );
        assert!(
            (rn - p2).abs() <= 1e-10,
            "constant column, n-space: {rn:e} vs {p2:e}"
        );
    }

    /// The tall units, called on parts chosen to make each column
    /// degenerate, decide it without refitting on X: route A of the K = 1
    /// no-refit body (`nr_split_r`) and its moment arm
    /// (`nr_split_r_moments`, a `None` going to route A), the Gram-p split
    /// unit at k = 2 and its moment arm, and the Gram-p fold unit at k = 2.
    /// X (200 × 6) has 100 rows each taken twice (rows `2m`, `2m + 1`).
    /// - A constant training outcome and an outcome antisymmetric across the
    ///   pairs of a training part of whole pairs (so `X̃_tr'z` is rounding):
    ///   the zero model, `r = 0` and zero predictions, the primal's answer.
    /// - An outcome orthogonal to a rank-3 `X̃_tr` up to rounding: finite.
    /// - `k = 4` on a training block of rank 2: a fit truncated to at most 2
    ///   components with finite coefficients, and a finite `r`.
    /// - A column constant over the training part: the primal's `r` to
    ///   `1e-10`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::similar_names
    )]
    fn degenerate_columns_are_decided_on_tall_data() {
        use crate::resample::Columns;
        let (n, p) = (200_usize, 6_usize);
        let (base, _) = signal_data(100, p, 211);
        let x = Mat::<f64>::from_fn(n, p, |i, j| base[(i / 2, j)] + 0.3 * j as f64);
        let sp = SplitIdx {
            tr: (0..100).collect(),
            te: (100..200).collect(),
        };
        let noise = Col::<f64>::from_fn(n, |i| ((i * 13) % 17) as f64 - 8.0);
        let a = |i: usize| 1.0 + ((i / 2) % 5) as f64;
        let constant = Col::<f64>::from_fn(n, |i| if i < 100 { 3.0 } else { noise[i] });
        let pairs = Col::<f64>::from_fn(n, |i| {
            if i >= 100 {
                noise[i]
            } else if i % 2 == 0 {
                a(i)
            } else {
                -a(i)
            }
        });
        // Every tall split route on one outcome: route A, its moment arm
        // (route A where it leaves the column), the Gram-p unit at `k` and
        // its moment arm (the Gram-p unit where it leaves the column), and
        // the primal at 1 and `k`.
        let split_routes = |x: MatRef<'_, f64>, y: &Col<f64>, k: usize| {
            let scale = crate::linalg::held_out_scale(x, None, false);
            let prep = prepare_split(x, &sp, None, scale.as_ref());
            let y_mat = nr_outcome_matrix(&Columns {
                y: y.as_ref().as_mat(),
                seeds: &[],
                null_map: None,
            });
            let route_a = nr_split_r(x, y_mat.as_ref(), &sp, None, scale.as_ref(), false)[0];
            let am1 = AllRowMoments::new(x, scale.as_ref(), 1, &|_| y_mat.col(0).to_owned());
            let moment_1 = nr_split_r_moments(x, y_mat.as_ref(), &sp, scale.as_ref(), &am1)[0]
                .unwrap_or(route_a);
            let g_block = split_block(ReplicateRoute::GramP, &prep, Par::Seq);
            let y_of = |i: usize| y[i];
            let gram_k = split_unit(&g_block, &prep, &sp, &y_of, &[k], None)[0];
            let am = AllRowMoments::new(x, scale.as_ref(), 1, &|_| y.clone());
            let tm = TestMoments::new(x, &sp.te, am.mean.as_ref(), scale.as_ref());
            let SplitBlock::GramP(gram) = &g_block else {
                panic!("split_block(GramP) built another block")
            };
            let moment_k =
                split_column_r_gram_p_moments(gram, &tm, &am, 0, &prep, &sp, &y_of, &[k], None)[0]
                    .unwrap_or(gram_k);
            let primal_1 = split_column_r(&prep, &sp, &y_of, &[1], None)[0];
            let primal_k = split_column_r(&prep, &sp, &y_of, &[k], None)[0];
            (
                [
                    ("A", route_a, primal_1),
                    ("moments k=1", moment_1, primal_1),
                    ("Gram-p", gram_k, primal_k),
                    ("Gram-p moments", moment_k, primal_k),
                ],
                prep,
            )
        };
        for (what, y) in [("constant", &constant), ("pairs", &pairs)] {
            for (route, r, want) in split_routes(x.as_ref(), y, 2).0 {
                assert_eq!(r.to_bits(), 0.0_f64.to_bits(), "{what}, {route}");
                assert_eq!(r.to_bits(), want.to_bits(), "{what}, {route}: primal");
            }
        }

        // Folds of whole pairs: every training set is whole pairs, so the
        // antisymmetric outcome is orthogonal to X̃_tr on every fold.
        let folds: Vec<Vec<usize>> = (0..5).map(|f| (40 * f..40 * f + 40).collect()).collect();
        let y_all = Col::<f64>::from_fn(n, |i| if i % 2 == 0 { a(i) } else { -a(i) });
        let y_const = Col::<f64>::from_fn(n, |_| 3.0);
        let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
        for (what, y) in [("pairs", &y_all), ("constant", &y_const)] {
            for fi in 0..folds.len() {
                let fold =
                    prepare_cv_fold(x.as_ref(), &folds, fi, None, scale.as_ref()).expect("fold");
                let g_block = fold_block(ReplicateRoute::GramP, &fold, Par::Seq);
                let p_block = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
                let y_of = |i: usize| y[i];
                let g = fold_unit(&g_block, &fold, &y_of, &[2], None);
                let pr = fold_unit(&p_block, &fold, &y_of, &[2], None);
                match (g, pr) {
                    (Ok(g), Ok(pr)) => {
                        let ((res_g, tot_g), (res_p, tot_p)) = (g[0], pr[0]);
                        assert_eq!(res_g.to_bits(), res_p.to_bits(), "{what}, fold {fi}");
                        assert_eq!(tot_g.to_bits(), tot_p.to_bits(), "{what}, fold {fi}");
                    }
                    (Err(_), Err(_)) => {}
                    (g, pr) => panic!("{what}, fold {fi}: {g:?} vs {pr:?}"),
                }
            }
        }

        // An outcome orthogonal to a rank-3 X̃_tr up to rounding.
        let a3 = Mat::<f64>::from_fn(n, 3, |i, j| ((i * (j + 3)) % 7) as f64 - 3.0);
        let b3 = Mat::<f64>::from_fn(3, p, |i, j| (((i + 1) * (j + 2)) % 11) as f64 - 5.0);
        let x3 = crate::linalg::mat_mul(a3.as_ref(), b3.as_ref(), Par::Seq);
        let scale3 = crate::linalg::held_out_scale(x3.as_ref(), None, false);
        let prep3 = prepare_split(x3.as_ref(), &sp, None, scale3.as_ref());
        let ones = Col::<f64>::from_fn(100, |_| 1.0);
        let basis = orthonormal_basis(ones.as_ref(), prep3.xs_tr.as_ref(), 1e-8);
        let mut z = Col::<f64>::from_fn(100, |i| noise[i]);
        project_off(&basis, &mut z);
        let orth = Col::<f64>::from_fn(n, |i| if i < 100 { z[i] } else { noise[i] });
        for (route, r, _) in split_routes(x3.as_ref(), &orth, 2).0 {
            assert!(
                r.is_finite() && r.abs() <= 1.0,
                "orthogonal, {route}: {r:e}"
            );
        }

        // k = 4 on a training block of rank 2.
        let a2 = Mat::<f64>::from_fn(n, 2, |i, j| ((i * (j + 5)) % 9) as f64 - 4.0);
        let b2 = Mat::<f64>::from_fn(2, p, |i, j| (((i + 2) * (j + 1)) % 7) as f64 - 3.0);
        let x2 = crate::linalg::mat_mul(a2.as_ref(), b2.as_ref(), Par::Seq);
        let y2 = Col::<f64>::from_fn(n, |i| a2[(i, 0)] + 0.1 * noise[i]);
        let (routes, prep2) = split_routes(x2.as_ref(), &y2, 4);
        for (route, r, _) in routes {
            assert!(r.is_finite() && r.abs() <= 1.0, "rank 2, {route}: {r:e}");
        }
        let ys2 = split_train_target(&prep2, &sp, &|i| y2[i]);
        let gram2 = crate::gram_p::GramPBlock::new(prep2.xs_tr.as_ref(), Par::Seq);
        let s1 = crate::gram_p::first_product(
            prep2.xs_tr.as_ref(),
            ys2.as_ref(),
            crate::fit::ParChoice::Seq,
            4,
        );
        let fit2 = gram2.fit(s1.as_ref(), ys2.norm_l2(), 4, None);
        assert!(fit2.k_used <= 2, "Gram-p keeps {}", fit2.k_used);
        assert!((0..p).all(|j| fit2.coef[j].is_finite()));

        // A column constant over the training part.
        let xc = Mat::<f64>::from_fn(n, p + 1, |i, j| {
            if j < p {
                base[(i / 2, j)] + 0.01 * i as f64
            } else if i < 100 {
                1.0
            } else {
                i as f64
            }
        });
        let yc = Col::<f64>::from_fn(n, |i| base[(i / 2, 0)] + 0.1 * noise[i]);
        for (route, r, want) in split_routes(xc.as_ref(), &yc, 2).0 {
            assert!(
                (r - want).abs() <= 1e-10,
                "constant column, {route}: {r:e} vs {want:e}"
            );
        }
    }

    /// Every split method defaults to J = 32 splits: the PLS1 and PLS3
    /// confirmatory tests, the sequence and the `find_k` diagnostic.
    #[test]
    fn default_n_splits_is_32() {
        use crate::find_k::{FindKOptimalOpts, FindKSequenceOpts};
        use crate::sequential::SequentialArgs;
        let j = |a: ConfirmatoryArgs| match a {
            ConfirmatoryArgs::SplitNb { n_splits, .. }
            | ConfirmatoryArgs::SplitExact { n_splits, .. }
            | ConfirmatoryArgs::Auto { n_splits, .. } => n_splits,
            other => panic!("{other:?} has no n_splits"),
        };
        for m in [
            ConfirmatoryMethod::SplitNb,
            ConfirmatoryMethod::SplitExact,
            ConfirmatoryMethod::Auto,
        ] {
            assert_eq!(j(ConfirmatoryArgs::defaults_for(m)), 32, "{m:?}");
            let seq = SequentialArgs::defaults_for(m)
                .unwrap()
                .to_confirmatory_args();
            assert_eq!(j(seq), 32, "sequence {m:?}");
        }
        assert_eq!(j(ConfirmatoryTestOpts::default().args), 32);
        let pls3 = crate::pls3_signal_test::Pls3ConfirmatoryTestOpts::default();
        assert_eq!(j(pls3.args), 32, "pls3");
        assert_eq!(FindKSequenceOpts::default().n_splits, 32);
        assert_eq!(FindKOptimalOpts::default().n_splits, 32);
    }

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
                    k: Some(&[k]),
                    weights: None,
                    covariates: None,
                    null_y: None,
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
                    k: Some(&[k]),
                    weights: w.map(Col::as_ref),
                    covariates: None,
                    null_y: None,
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

    /// `nb_rho_hat` against its closed form on hand-made Fisher-z values:
    /// z = (0.1, 0.3, 0.2) has ddof-1 variance 0.01.
    #[test]
    fn nb_rho_hat_matches_the_closed_form() {
        let stats = Col::<f64>::from_fn(3, |i| [0.1_f64, 0.3, 0.2][i].tanh());
        // n_test = 53: the ruler is 1/50 = 0.02, so ρ̂ = 1 − 0.01/0.02.
        let rho = nb_rho_hat(&stats, 53).unwrap();
        assert!((rho - 0.5).abs() < 1e-12, "{rho}");
        // n_test = 203: the ruler 1/200 is below s², and ρ̂ clips to 0.
        assert_eq!(nb_rho_hat(&stats, 203), Some(0.0));
        // Identical statistics have no spread: ρ̂ = 1.
        let flat = Col::<f64>::from_fn(3, |_| 0.2_f64.tanh());
        assert_eq!(nb_rho_hat(&flat, 53), Some(1.0));
        // Below four test rows the ruler is undefined.
        assert_eq!(nb_rho_hat(&stats, 3), None);
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
                    k: Some(&[1]),
                    weights: None,
                    covariates: None,
                    null_y: None,
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

    /// More than half the null fits failing is an error, not a p-value.
    /// Rows 0 and 1 hold `±f64::MAX` under equal weights, so in every fold
    /// that trains on both, the observed y has a weighted mean of exactly 0
    /// and standardizes to finite values. Every other row has its own
    /// weight, so a permutation puts the two extremes on rows of unequal
    /// weight; a fold that trains on both (at least 3 of the 5) then has a
    /// weighted mean near `f64::MAX / n`, `y − mean` overflows on the
    /// extreme of the opposite sign, and that null's fit fails with
    /// `NonFiniteInput`.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn raw_perm_errors_when_most_null_fits_fail() {
        let n = 30;
        let (x, _) = signal_data(n, 4, 61);
        let mut y = Col::<f64>::zeros(n);
        y[0] = f64::MAX;
        y[1] = -f64::MAX;
        let w = Col::<f64>::from_fn(n, |i| if i < 2 { 1.0 } else { i as f64 + 1.0 });
        let n_perm = 19;
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[1]),
                weights: Some(w.as_ref()),
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm { n_perm, n_folds: 5 },
                seed: Some(3),
                ..Default::default()
            },
        );
        let Err(PlsKitError::PermNullDegenerate { failed, total }) = r else {
            panic!("expected PermNullDegenerate, got {r:?}");
        };
        assert_eq!(total, n_perm);
        assert!(failed * 2 > total, "failed={failed}, total={total}");
    }

    // ── raw_perm dual (Gram) route tests ────────────────────────────────

    /// A finite X whose plain centring overflows: rows 0 and 1 of column 0
    /// at `f64::MAX`, row 2 at `-f64::MAX`, so `x − mean` overflows on row 2
    /// for the full X and for every training half or fold holding all
    /// three rows (`linalg::requarter` rewrites those columns from their
    /// quarter).
    fn centering_overflow(x: &mut Mat<f64>) {
        x[(0, 0)] = f64::MAX;
        x[(1, 0)] = f64::MAX;
        x[(2, 0)] = -f64::MAX;
    }

    /// The gate on X far from unit scale, on the tall and the wide branch:
    /// X whose plain centring overflows ([`centering_overflow`]) gets the
    /// rank of X/8, and a zero-weight row holding 1e200 (whose product
    /// overflows before the √w row scaling zeroes it) gets a finite rank.
    #[test]
    fn split_nb_gate_reads_x_near_max_and_a_far_zero_weight_row() {
        for (n, p) in [(40, 6), (6, 40), (400, 30), (30, 400)] {
            let (mut x, _) = signal_data(n, p, 3);
            centering_overflow(&mut x);
            let x8 = Mat::<f64>::from_fn(n, p, |i, j| x[(i, j)] / 8.0);
            let g = split_nb_gate(x.as_ref(), None, false).expect("finite X");
            let g8 = split_nb_gate(x8.as_ref(), None, false).expect("finite X");
            assert!(
                (g.stable_rank - g8.stable_rank).abs() <= 1e-12 * g8.stable_rank,
                "{n}x{p}: {g:?} vs {g8:?}"
            );

            let (mut x, _) = signal_data(n, p, 3);
            x[(0, 0)] = 1e200;
            let w = Col::<f64>::from_fn(n, |i| if i == 0 { 0.0 } else { 1.0 });
            let g = split_nb_gate(x.as_ref(), Some(w.as_ref()), false).expect("finite X");
            assert!(g.stable_rank.is_finite(), "{n}x{p}, weighted: {g:?}");
        }
    }

    /// The gate reads X with the √w row scaling of the tests: a row of
    /// weight 0 shifted by +50 leaves the stable rank of X without that row,
    /// on the wide and the tall branch.
    #[test]
    fn split_nb_gate_ignores_a_zero_weight_row() {
        for (n, p) in [(40, 200), (400, 30)] {
            let (mut x, _) = signal_data(n, p, 5);
            for j in 0..p {
                x[(0, j)] += 50.0;
            }
            let w = Col::<f64>::from_fn(n, |i| if i == 0 { 0.0 } else { 1.0 });
            let g = split_nb_gate(x.as_ref(), Some(w.as_ref()), false).unwrap();
            let rest = Mat::<f64>::from_fn(n - 1, p, |i, j| x[(i + 1, j)]);
            let want = split_nb_gate(rest.as_ref(), None, false).unwrap();
            assert!(
                (g.stable_rank - want.stable_rank).abs() <= 1e-10 * want.stable_rank,
                "{n}x{p}: {g:?} vs {want:?}"
            );
        }
    }

    /// PLS1 `split_nb` meets a non-finite Gram matrix in its gate and returns
    /// a result rather than panic in the eigendecomposition: the NaN rank
    /// fires the gate, which reroutes to `split_exact`. Wide X holds 1e200
    /// on a row of subnormal weight, whose product overflows and which the
    /// √w row scaling does not zero (a weight of 0 would). PLS3 has no
    /// weights, and a finite X gives it a finite Gram matrix.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn split_nb_reroutes_on_a_non_finite_gram_matrix_instead_of_panicking() {
        let (n, p) = (40, 200);
        let (mut x, y) = signal_data(n, p, 3);
        let w = Col::<f64>::from_fn(n, |i| if i == 0 { 1e-320 } else { 1.0 });
        x[(0, 0)] = 1e200;
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[1]),
                weights: Some(w.as_ref()),
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb {
                    n_splits: 6,
                    force: false,
                },
                seed: Some(7),
                ..Default::default()
            },
        )
        .expect("a result");
        assert_eq!(r.test_method, "split_exact");
        assert!(
            r.stable_rank.is_some_and(f64::is_nan),
            "{:?}",
            r.stable_rank
        );
    }

    /// `|a − b| ≤ 1e-12·max|b|` entry by entry: what the same computation
    /// gives on X and on X divided by a power of two.
    fn assert_scaled_close(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        let tol = 1e-12 * b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (i, (u, v)) in a.iter().zip(b).enumerate() {
            assert!((u - v).abs() <= tol, "{what} [{i}]: {u} vs {v}");
        }
    }

    /// The statistic and p of two confirmatory results, for
    /// [`assert_scaled_close`].
    fn stat_p(r: &ConfirmatoryTestOutput) -> [f64; 2] {
        [r.statistic.unwrap_or(f64::NAN), r.pvalue]
    }

    /// A finite X whose plain centring overflows ([`centering_overflow`])
    /// gives the fits and the `score` test what X/8 gives: `pls1_fit` at
    /// k = 1 on a tall and a wide X (both take the implicit route),
    /// `pls3_fit` at k = 1 with three targets, and `score` on a design under
    /// and over its `d <= n` branch.
    #[test]
    fn fits_and_score_read_x_near_max_as_x_over_8() {
        let eighth =
            |x: &Mat<f64>| Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] / 8.0);
        for (n, p) in [(40, 6), (40, 200)] {
            let (mut x, y) = signal_data(n, p, 3);
            centering_overflow(&mut x);
            let coef = |x: &Mat<f64>| {
                let m = pls1_fit(
                    x.as_ref(),
                    y.as_ref(),
                    KSpec::Fixed(1),
                    None,
                    None,
                    FitOpts::default(),
                )
                .expect("a fit");
                (0..p).map(|j| m.coef[j]).collect::<Vec<_>>()
            };
            assert_scaled_close(&coef(&x), &coef(&eighth(&x)), &format!("pls1_fit {n}x{p}"));
        }
        let (mut x, _) = signal_data(40, 6, 3);
        centering_overflow(&mut x);
        let y3 = Mat::<f64>::from_fn(40, 3, |i, j| x[(i, j + 1)]);
        let sv = |x: &Mat<f64>| {
            let m = crate::pls3_fit(
                x.as_ref(),
                y3.as_ref(),
                1,
                None,
                crate::Pls3FitOpts::default(),
            )
            .expect("a fit");
            (0..m.singular_values.nrows())
                .map(|i| m.singular_values[i])
                .collect::<Vec<_>>()
        };
        assert_scaled_close(&sv(&x), &sv(&eighth(&x)), "pls3_fit");
        for (n, p) in [(40, 6), (600, 20), (20, 40)] {
            let (mut x, y) = signal_data(n, p, 3);
            centering_overflow(&mut x);
            let score = |x: &Mat<f64>| {
                stat_p(
                    &pls1_confirmatory_test(
                        ConfirmatoryTestInput::Raw {
                            x: x.as_ref(),
                            y: y.as_ref(),
                            k: Some(&[1]),
                            weights: None,
                            covariates: None,
                            null_y: None,
                        },
                        ConfirmatoryTestOpts {
                            args: ConfirmatoryArgs::Score,
                            seed: Some(3),
                            ..Default::default()
                        },
                    )
                    .expect("a result"),
                )
            };
            assert_scaled_close(&score(&x), &score(&eighth(&x)), &format!("score {n}x{p}"));
        }
    }

    /// A finite X whose plain centring overflows only where held-out rows
    /// are centred: column 0 with row `a` at `f64::MAX` and row `b` at
    /// `-f64::MAX`. A training part holding both, or neither, centres
    /// without overflow. One holding exactly one of them has a mean of that
    /// row's sign far from 0, so `x − mean` overflows on the other row when
    /// it is held out.
    fn overflow_rows(x: &mut Mat<f64>, a: usize, b: usize) {
        x[(a, 0)] = f64::MAX;
        x[(b, 0)] = -f64::MAX;
    }

    /// [`overflow_rows`] on rows 0 and 1.
    fn held_out_overflow(x: &mut Mat<f64>) {
        overflow_rows(x, 0, 1);
    }

    /// An X whose plain fold centring overflows gives what X/8 gives on
    /// every `raw_perm` route: in a training fold ([`centering_overflow`]: a
    /// fold holding rows 0, 1 and 2, as at least two of the five do) or only
    /// in a validation fold ([`overflow_rows`] on the first rows of folds 0
    /// and 1 as the seed draws them, so the folds holding either one meet
    /// it). The `e` test meets the second X in its held-out half and gives
    /// what X/8 gives too.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn raw_perm_reads_x_near_max_as_x_over_8_on_every_route() {
        // (n, p, n_perm, k, the route the shape takes)
        let rows = [
            (30, 5, 99, 1, ReplicateRoute::Primal),
            (30, 200, 99, 1, ReplicateRoute::Special),
            (30, 200, 99, 2, ReplicateRoute::Nspace),
            (2500, 20, 999, 1, ReplicateRoute::GramP),
        ];
        let eighth =
            |x: &Mat<f64>| Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] / 8.0);
        let run = |x: &Mat<f64>, y: &Col<f64>, k, args| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[k]),
                    weights: None,
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(3),
                    ..Default::default()
                },
            )
        };
        // The folds `run_raw_perm` draws first from the seed.
        let folds_at_seed = |n: usize| {
            use rand::seq::SliceRandom;
            let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            crate::linalg::fold_split(&idx, 5)
        };
        for held_out in [false, true] {
            for (n, p, n_perm, k, want) in rows {
                let route = raw_perm_route(n, 5, p, n_perm, k, None, false);
                assert_eq!(route, want, "n={n} p={p} k={k}");
                let (mut x, y) = signal_data(n, p, 61);
                if held_out {
                    let f = folds_at_seed(n);
                    overflow_rows(&mut x, f[0][0], f[1][0]);
                } else {
                    centering_overflow(&mut x);
                }
                let args = ConfirmatoryArgs::RawPerm { n_perm, n_folds: 5 };
                assert_scaled_close(
                    &stat_p(&run(&x, &y, k, args).expect("a result")),
                    &stat_p(&run(&eighth(&x), &y, k, args).expect("a result")),
                    &format!("{route:?} held_out={held_out}"),
                );
            }
        }
        // `e` holds out one half, drawn first from the seed: the two
        // `±f64::MAX` rows go to a training and a held-out row of that half.
        let (mut x, y) = signal_data(40, 6, 61);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let n_train = crate::resample::split_sizes(40, 1).0;
        let (tr, te) = crate::resample::one_split(40, n_train, &mut rng);
        overflow_rows(&mut x, tr[0], te[0]);
        assert_scaled_close(
            &stat_p(&run(&x, &y, 1, ConfirmatoryArgs::E).expect("a result")),
            &stat_p(&run(&eighth(&x), &y, 1, ConfirmatoryArgs::E).expect("a result")),
            "e",
        );
    }

    /// A held-out half whose plain centring overflows is finite and gives
    /// the `r` of X/8. Column 0 is `-1.7e308` on the training rows and
    /// `1.7e308` on the held-out ones, so the training mean is `-1.7e308`
    /// and `x − mean` overflows on every held-out row.
    #[test]
    fn split_column_r_reads_a_held_out_half_near_max_as_x_over_8() {
        let (mut x, y) = signal_data(12, 3, 5);
        for i in 0..12 {
            x[(i, 0)] = if i < 6 { -1.7e308 } else { 1.7e308 };
        }
        let sp = SplitIdx {
            tr: (0..6).collect(),
            te: (6..12).collect(),
        };
        let r = |x: &Mat<f64>| {
            let prep = prepare_split(
                x.as_ref(),
                &sp,
                None,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
            );
            assert!(prep.x_finite);
            split_column_r(&prep, &sp, &|i| y[i], &[1], None)[0]
        };
        let x8 = Mat::<f64>::from_fn(12, 3, |i, j| x[(i, j)] / 8.0);
        assert_scaled_close(&[r(&x)], &[r(&x8)], "r");
    }

    /// The held-out tests give on X near `±f64::MAX` ([`centering_overflow`])
    /// what they give on X/8, and under `pre_standardized` on `X·2^±600`
    /// what they give on X: PLS1 `split_exact` at k = 1 and 2, `split_nb`
    /// and `raw_perm`, and PLS3 `split_exact`, on a wide and a tall X.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn held_out_tests_read_x_at_any_magnitude() {
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 19,
            n_splits: 6,
        };
        let pls1 = [
            (1, exact),
            (2, exact),
            (
                1,
                ConfirmatoryArgs::SplitNb {
                    n_splits: 6,
                    force: true,
                },
            ),
            (
                1,
                ConfirmatoryArgs::RawPerm {
                    n_perm: 19,
                    n_folds: 5,
                },
            ),
        ];
        for (n, p) in [(40, 200), (600, 20)] {
            let (x, y) = signal_data(n, p, 11);
            let y3 = Mat::<f64>::from_fn(n, 3, |i, j| y[i] + x[(i, j + 1)]);
            let mut big = x.clone();
            centering_overflow(&mut big);
            let times = |x: &Mat<f64>, f: f64| Mat::<f64>::from_fn(n, p, |i, j| x[(i, j)] * f);
            let cases = [
                (times(&big, 1.0), times(&big, 0.125), false),
                (times(&x, 2f64.powi(600)), x.clone(), true),
                (times(&x, 2f64.powi(-600)), x.clone(), true),
            ];
            for (a, b, pre) in &cases {
                for (k, args) in pls1 {
                    let run = |x: &Mat<f64>| {
                        stat_p(
                            &pls1_confirmatory_test(
                                ConfirmatoryTestInput::Raw {
                                    x: x.as_ref(),
                                    y: y.as_ref(),
                                    k: Some(&[k]),
                                    weights: None,
                                    covariates: None,
                                    null_y: None,
                                },
                                ConfirmatoryTestOpts {
                                    args,
                                    pre_standardized: *pre,
                                    seed: Some(7),
                                    ..Default::default()
                                },
                            )
                            .expect("a result"),
                        )
                    };
                    assert_scaled_close(
                        &run(a),
                        &run(b),
                        &format!("{n}x{p} pre={pre} k={k} {args:?}"),
                    );
                }
                let run3 = |x: &Mat<f64>| {
                    stat_p(
                        &crate::pls3_confirmatory_test(
                            x.as_ref(),
                            y3.as_ref(),
                            1,
                            crate::Pls3ConfirmatoryTestOpts {
                                args: exact,
                                pre_standardized_x: *pre,
                                seed: Some(7),
                                ..Default::default()
                            },
                        )
                        .expect("a result"),
                    )
                };
                assert_scaled_close(&run3(a), &run3(b), &format!("PLS3 {n}x{p} pre={pre}"));
            }
        }
    }

    /// On tall X whose plain centring overflows only in held-out halves
    /// ([`held_out_overflow`]), `split_exact` at k = 1 gives the same
    /// statistic on every arm of route A: the cell arm (default budget), the
    /// moment arm (400 KiB, below the cell buffers) and the test-row arm
    /// (1 KiB). Premise: the budgets pick
    /// those arms.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn tall_k1_arms_agree_when_a_held_out_half_overflows() {
        let (n, p, n_perm) = (2000_usize, 50_usize, 600_usize);
        let (mut x, y) = signal_data(n, p, 11);
        held_out_overflow(&mut x);
        for n_splits in [5_usize, 32] {
            let (_, mut rng) = crate::rng::resolve_seed(Some(1)).unwrap();
            let d = draw_design(n, 1, n_splits, &mut rng).unwrap();
            let (n_train, n_test) = crate::resample::split_sizes(n, 1);
            let b = crate::cell_sums::cell_batch(n_perm + 1);
            assert_eq!(
                k1_route(n, n_train, n_test, p, n_perm + 1, n_splits, false),
                K1Route::A
            );
            for (budget, moments, cells) in [
                (u64::MAX, true, true),
                (400 << 10, true, false),
                (1024, false, false),
            ] {
                crate::fit::with_budget(budget, || {
                    assert_eq!(
                        test_moments_route(n, n_test, p, n_perm + 1, n_splits, false),
                        moments
                    );
                    assert_eq!(crate::cell_sums::cells_eligible(&d, p, b, false), cells);
                });
            }
            let call = || {
                pls1_confirmatory_test(
                    ConfirmatoryTestInput::Raw {
                        x: x.as_ref(),
                        y: y.as_ref(),
                        k: Some(&[1]),
                        weights: None,
                        covariates: None,
                        null_y: None,
                    },
                    ConfirmatoryTestOpts {
                        args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
                        seed: Some(7),
                        ..Default::default()
                    },
                )
                .expect("a result")
            };
            let rows = crate::fit::with_budget(1024, call);
            for (label, got) in [
                ("cells", crate::fit::with_budget(u64::MAX, call)),
                ("moments", crate::fit::with_budget(400 << 10, call)),
            ] {
                assert!(
                    (got.statistic.unwrap() - rows.statistic.unwrap()).abs() <= 1e-10,
                    "J={n_splits} {label}: statistic {} vs test rows {}",
                    got.statistic.unwrap(),
                    rows.statistic.unwrap()
                );
                assert_eq!(
                    got.pvalue.to_bits(),
                    rows.pvalue.to_bits(),
                    "J={n_splits} {label}"
                );
            }
        }
    }

    /// Every PLS1 split statistic stays finite on an X whose plain centring
    /// overflows only in held-out halves ([`held_out_overflow`]). Checked on `split_nb` at
    /// K = 1 (the no-refit body) and K = 2, on `split_exact` on its no-refit
    /// route and on each refit route (the two Gram routes agreeing with the
    /// primal; on Gram-p both the cell arm and, on a budget under the cell
    /// buffers, the per-split moment arm), on a weighted run, whose held-out
    /// half carries √w, and on forced `split_nb` over X whose plain
    /// centring overflows in the full X ([`centering_overflow`]), whose
    /// stable rank is finite.
    #[test]
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    #[allow(clippy::cast_precision_loss)]
    fn split_statistics_stay_finite_when_a_held_out_half_overflows() {
        // (n, p, k, args, the split_exact route, if any)
        let rows = [
            (
                40,
                6,
                1,
                ConfirmatoryArgs::SplitNb {
                    n_splits: 20,
                    force: true,
                },
                None,
            ),
            (
                40,
                6,
                2,
                ConfirmatoryArgs::SplitNb {
                    n_splits: 20,
                    force: true,
                },
                None,
            ),
            (
                40,
                6,
                1,
                ConfirmatoryArgs::SplitExact {
                    n_perm: 19,
                    n_splits: 6,
                },
                Some(ReplicateRoute::Special),
            ),
            (
                40,
                6,
                2,
                ConfirmatoryArgs::SplitExact {
                    n_perm: 19,
                    n_splits: 6,
                },
                Some(ReplicateRoute::Primal),
            ),
            (
                40,
                200,
                2,
                ConfirmatoryArgs::SplitExact {
                    n_perm: 19,
                    n_splits: 6,
                },
                Some(ReplicateRoute::Nspace),
            ),
            (
                2000,
                50,
                2,
                ConfirmatoryArgs::SplitExact {
                    n_perm: 600,
                    n_splits: 5,
                },
                Some(ReplicateRoute::GramP),
            ),
        ];
        for (n, p, k, args, route) in rows {
            if let (Some(want), ConfirmatoryArgs::SplitExact { n_perm, n_splits }) = (route, args) {
                assert_eq!(
                    split_exact_refit_route(n, p, n_perm, n_splits, k, None, false, false),
                    want,
                    "{n}x{p} k={k}: premise"
                );
            }
            let (mut x, y) = signal_data(n, p, 11);
            held_out_overflow(&mut x);
            let call = || {
                pls1_confirmatory_test(
                    ConfirmatoryTestInput::Raw {
                        x: x.as_ref(),
                        y: y.as_ref(),
                        k: Some(&[k]),
                        weights: None,
                        covariates: None,
                        null_y: None,
                    },
                    ConfirmatoryTestOpts {
                        args,
                        seed: Some(7),
                        ..Default::default()
                    },
                )
                .expect("a result")
            };
            let g = call();
            assert!(
                g.statistic.unwrap().is_finite(),
                "{n}x{p} k={k} {args:?}: {}",
                g.statistic.unwrap()
            );
            if matches!(route, Some(ReplicateRoute::Nspace | ReplicateRoute::GramP)) {
                let pr = with_gram_routes_disabled(call);
                // On Gram-p: the cell arm at the default budget, and the
                // per-split moment arm at 400 KiB, under the cell buffers.
                let mut runs = vec![("default", g)];
                if route == Some(ReplicateRoute::GramP) {
                    let ConfirmatoryArgs::SplitExact { n_perm, n_splits } = args else {
                        unreachable!()
                    };
                    let (_, mut rng) = crate::rng::resolve_seed(Some(1)).unwrap();
                    let d = draw_design(n, k, n_splits, &mut rng).unwrap();
                    let n_test = crate::resample::split_sizes(n, k).1;
                    let b = crate::cell_sums::cell_batch(n_perm + 1);
                    crate::fit::with_budget(400 << 10, || {
                        assert!(test_moments_route(
                            n,
                            n_test,
                            p,
                            n_perm + 1,
                            n_splits,
                            false
                        ));
                        assert!(!crate::cell_sums::cells_eligible(&d, p, b, false));
                    });
                    crate::fit::with_budget(u64::MAX, || {
                        assert!(crate::cell_sums::cells_eligible(&d, p, b, false));
                    });
                    runs = vec![
                        ("cells", crate::fit::with_budget(u64::MAX, call)),
                        ("moments", crate::fit::with_budget(400 << 10, call)),
                    ];
                }
                for (label, g) in runs {
                    assert_eq!(
                        g.pvalue.to_bits(),
                        pr.pvalue.to_bits(),
                        "{n}x{p} {label}: pvalue"
                    );
                    assert!(
                        (g.statistic.unwrap() - pr.statistic.unwrap()).abs() <= 1e-10,
                        "{n}x{p} {label}: statistic {} vs {}",
                        g.statistic.unwrap(),
                        pr.statistic.unwrap()
                    );
                }
            }
        }

        let (mut x, y) = signal_data(40, 6, 11);
        held_out_overflow(&mut x);
        let w = Col::<f64>::from_fn(40, |i| 1.0 + (i % 3) as f64);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[2]),
                weights: Some(w.as_ref()),
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 19,
                    n_splits: 6,
                },
                seed: Some(7),
                ..Default::default()
            },
        )
        .expect("a result");
        assert!(
            r.statistic.unwrap().is_finite(),
            "weighted: {}",
            r.statistic.unwrap()
        );

        let (mut x, y) = signal_data(40, 6, 11);
        centering_overflow(&mut x);
        for k in [1, 2] {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[k]),
                    weights: None,
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitNb {
                        n_splits: 20,
                        force: true,
                    },
                    seed: Some(7),
                    ..Default::default()
                },
            )
            .expect("a result");
            assert_eq!(r.test_method, "split_nb", "k={k}");
            assert!(r.stable_rank.is_some_and(f64::is_finite), "k={k}");
            assert!(
                r.statistic.unwrap().is_finite(),
                "forced, k={k}: {}",
                r.statistic.unwrap()
            );
        }
    }

    /// A finite y whose fold standardization overflows is non-finite input
    /// on every `raw_perm` route. Rows 0 and 1 hold `f64::MAX` and row 2
    /// `-f64::MAX`; a training fold that holds all three (at least two of
    /// the five do) has a mean far above 0, and `y − mean` overflows on row
    /// 2. The observed column fails, so the call is an error, not a p-value.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn raw_perm_rejects_y_whose_fold_standardization_overflows_on_every_route() {
        // (n, p, n_perm, k, the route the shape takes)
        let rows = [
            (30, 5, 99, 1, ReplicateRoute::Primal),
            (30, 200, 99, 1, ReplicateRoute::Special),
            (30, 200, 99, 2, ReplicateRoute::Nspace),
            (2500, 20, 999, 1, ReplicateRoute::GramP),
        ];
        for (n, p, n_perm, k, want) in rows {
            let route = raw_perm_route(n, 5, p, n_perm, k, None, false);
            assert_eq!(route, want, "n={n} p={p} k={k}");
            let (x, _) = signal_data(n, p, 61);
            let mut y = Col::<f64>::zeros(n);
            y[0] = f64::MAX;
            y[1] = f64::MAX;
            y[2] = -f64::MAX;
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[k]),
                    weights: None,
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::RawPerm { n_perm, n_folds: 5 },
                    seed: Some(3),
                    ..Default::default()
                },
            );
            assert!(
                matches!(r, Err(PlsKitError::NonFiniteInput)),
                "{route:?}: {r:?}"
            );
        }
    }

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
            if col == 0 { y[i] } else { y[perms[col - 1][i]] }
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
                k: Some(&[1]),
                weights: None,
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm { n_perm, n_folds },
                seed: Some(seed),
                ..Default::default()
            },
        )
        .unwrap();
        let stat_scale = prod.statistic.unwrap().abs().max(a[0].abs());
        let stat_rel = if stat_scale == 0.0 {
            0.0
        } else {
            (prod.statistic.unwrap() - a[0]).abs() / stat_scale
        };
        assert!(
            stat_rel < 1e-10,
            "production statistic diverges from reference: prod={} ref={} rel={stat_rel}",
            prod.statistic.unwrap(),
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

        // Route B: the Gram path, all columns at once, with its folds one at
        // a time (budget 0) and all in one batch (no limit): the same bits.
        let run = |budget| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(budget, || {
                    closed_form_cv_r2(
                        x.as_ref(),
                        y_mat.as_ref(),
                        &folds,
                        crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                    )
                    .unwrap()
                })
            })
        };
        let ((b, log_one), (b_all, log_all)) = (run(0), run(u64::MAX));
        crate::resample::assert_batches(&log_one, n_folds, 1, 1);
        crate::resample::assert_batches(&log_all, n_folds, n_folds, 1);
        assert_eq!(
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b_all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the Gram route's bits depend on the budget"
        );

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

    /// `dual_route::pls1_cv_r2_columns` on the all-row Gram matrix of `x`
    /// and `scale`, as `run_raw_perm` builds it (`block_gram`).
    fn closed_form_cv_r2(
        x: MatRef<'_, f64>,
        y_mat: MatRef<'_, f64>,
        folds: &[Vec<usize>],
        scale: ColRef<'_, f64>,
    ) -> PlsKitResult<Vec<f64>> {
        let g = block_gram(x, scale, None);
        crate::dual_route::pls1_cv_r2_columns(&g, y_mat, folds)
    }

    #[test]
    fn raw_perm_k1_takes_the_closed_form_on_wide_real_shapes() {
        // T1w_wap2 and wide_254x20000 at the reference script's n_perm 200
        // and the default 5 folds.
        for p in [677_949, 20_000] {
            assert_eq!(
                raw_perm_route(254, 5, p, 200, 1, None, false),
                ReplicateRoute::Special,
                "254x{p}"
            );
        }
    }

    #[test]
    fn raw_perm_k1_rule_crosses_over_at_the_one_gram_cost() {
        // 254 x 20,000, 5 folds (n_tr 204, n_val 50): the closed form pays
        // from B' = 15 (254²·20,000/2 + 5·B'·(2·204² + 50·204) against
        // 5·B'·(2·204 + 50)·20,000).
        assert_eq!(
            raw_perm_route(254, 5, 20_000, 13, 1, None, false),
            ReplicateRoute::Primal
        );
        assert_eq!(
            raw_perm_route(254, 5, 20_000, 14, 1, None, false),
            ReplicateRoute::Special
        );
    }

    /// 40 x 2000, 5 folds, `n_perm` 9: the per-fold rule kept it primal, the
    /// one-G rule takes the closed form, and every column's CV R² agrees
    /// with the primal route's to a relative 1e-12.
    #[test]
    fn raw_perm_k1_closed_form_matches_the_primal_on_a_shape_the_one_gram_rule_adds() {
        let (n, p, n_perm, n_folds) = (40, 2000, 9, 5);
        assert!(
            !crate::dual_route::use_dual_route(n - n / n_folds, p, n_perm + 1, 1),
            "premise: the per-fold rule rejects it"
        );
        assert_eq!(
            raw_perm_route(n, n_folds, p, n_perm, 1, None, false),
            ReplicateRoute::Special
        );
        let (x, y) = synth(n, p, 3, 4.0, 7);
        let (_, mut rng) = crate::rng::resolve_seed(Some(11)).unwrap();
        let folds = crate::linalg::fold_split(&(0..n).collect::<Vec<_>>(), n_folds);
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let cols = crate::resample::Columns {
            y: y.as_ref().as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
        let y_mat = cols.matrix(0, cols.len());
        let closed = closed_form_cv_r2(x.as_ref(), y_mat.as_ref(), &folds, scale.as_ref()).unwrap();
        let primal = first_row(
            &pooled_cv_r2_columns(
                ReplicateRoute::Primal,
                x.as_ref(),
                &folds,
                None,
                scale.as_ref(),
                &[1],
                None,
                cols.len(),
                &|c| cols.column(c),
            )
            .unwrap(),
        );
        for (c, (a, b)) in closed.iter().zip(&primal).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
                "column {c}: {a:e} vs {b:e}"
            );
        }
    }

    #[test]
    fn raw_perm_dual_route_matches_primal_when_selected() {
        // n=60, p=3000, n_folds=5 ⇒ n_tr_max=48, B+1=50:
        // 48·(50+3000) = 146,400 < 3000·50 = 150,000 ⇒ dual is live.
        assert_raw_perm_dual_route_matches_primal(60, 3000, 4.0, 7, 49, 5, 11);
    }

    #[test]
    fn raw_perm_dual_route_matches_primal_over_several_blocks() {
        // n=40, n_folds=5 ⇒ n_tr_max=32 and a block of 8,192 columns, so
        // p=20,000 is three blocks. B+1=50:
        // 32·(50+20000) = 641,600 < 20000·50 = 1,000,000 ⇒ dual is live.
        assert_eq!(crate::linalg::stream_block_width(32), 8192);
        assert_raw_perm_dual_route_matches_primal(40, 20_000, 4.0, 7, 49, 5, 11);
    }

    /// Folds one at a time (budget 0) and all in one batch (no limit), at
    /// p = 100 and p = 400: the same bits, and the pooled R² of the primal
    /// fold loop, column by column.
    #[test]
    fn raw_perm_dual_route_matches_primal_on_both_fold_arms() {
        for p in [100_usize, 400] {
            // 4 folds of 40 rows: 30 training rows each.
            let (x, y) = synth(40, p, 3, 4.0, 5);
            let folds = crate::linalg::fold_split(&(0..40).collect::<Vec<_>>(), 4);
            // Column 0 is y; columns 1 and 2 are y rotated by 7 and 14 rows.
            let y_mat = Mat::<f64>::from_fn(40, 3, |i, c| y[(i + 7 * c) % 40]);
            let run = |b| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(b, || {
                        closed_form_cv_r2(
                            x.as_ref(),
                            y_mat.as_ref(),
                            &folds,
                            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                        )
                        .unwrap()
                    })
                })
            };
            let ((dual, log_one), (all, log_all)) = (run(0), run(u64::MAX));
            crate::resample::assert_batches(&log_one, 4, 1, 1);
            crate::resample::assert_batches(&log_all, 4, 4, 1);
            assert_eq!(
                dual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "p={p}"
            );
            for (col, v) in dual.iter().enumerate() {
                let y_col = Col::<f64>::from_fn(40, |i| y_mat[(i, col)]);
                let primal = pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap();
                assert!(
                    (v - primal).abs() <= 1e-10 * primal.abs().max(1.0),
                    "p={p} col {col}: primal={primal} dual={v}"
                );
            }
        }
    }

    /// A fold of the K = 1 Gram route's first pass counts its `G` and `M`
    /// (`resample::fold_blocks_bytes` at the training size): a budget of two
    /// such units runs two of the four folds at once.
    #[test]
    fn raw_perm_dual_route_counts_its_blocks() {
        // 4 folds of 10 rows: 30 training rows each.
        let (x, y) = synth(40, 100, 3, 4.0, 5);
        let folds = crate::linalg::fold_split(&(0..40).collect::<Vec<_>>(), 4);
        let y_mat = Mat::<f64>::from_fn(40, 3, |i, c| y[(i + 7 * c) % 40]);
        let unit = crate::resample::fold_blocks_bytes(40, &folds, 30);
        let (_, log) = crate::resample::with_batch_log(|| {
            crate::fit::with_budget(u64::try_from(2 * unit).unwrap(), || {
                closed_form_cv_r2(
                    x.as_ref(),
                    y_mat.as_ref(),
                    &folds,
                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                )
                .unwrap()
            })
        });
        crate::resample::assert_batches(&log, 4, 2, 1);
    }

    #[test]
    fn raw_perm_dual_route_degenerate_x_is_finite_and_matches_primal() {
        // Every X row identical ⇒ standardize's zero-variance branch makes
        // X̃_tr exactly zero ⇒ z'Gz = 0 ⇒ the NIPALS norm guard fires ⇒ the
        // prediction is zero. The pooled R² that follows is finite but not
        // 0: ss_res = Σ ys_val² while ss_tot = Σ(ys_val − mean_val)², and
        // ys_val carries the *training* fold's y moments, so its validation
        // mean is not 0. The claim under test is finiteness (never NaN) and
        // agreement with the primal route to the bit, both routes
        // predicting zero. The folds run one at a time (budget 0) and all in
        // one batch (no limit), with the same bits.
        for p in [8_usize, 200] {
            let x = Mat::<f64>::from_fn(40, p, |_, _| 3.0);
            let (_, y) = synth(40, p, 3, 4.0, 5);
            let folds = crate::linalg::fold_split(&(0..40).collect::<Vec<_>>(), 4);
            let y_mat = Mat::<f64>::from_fn(40, 3, |i, _| y[i]);
            let run = |b| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(b, || {
                        closed_form_cv_r2(
                            x.as_ref(),
                            y_mat.as_ref(),
                            &folds,
                            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                        )
                        .unwrap()
                    })
                })
            };
            let ((dual, log_one), (all, log_all)) = (run(0), run(u64::MAX));
            crate::resample::assert_batches(&log_one, 4, 1, 1);
            crate::resample::assert_batches(&log_all, 4, 4, 1);
            assert_eq!(
                dual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "p={p}"
            );
            for (col, v) in dual.iter().enumerate() {
                assert!(v.is_finite(), "p={p} col {col} is not finite: {v}");
                let primal = {
                    let y_col = Col::<f64>::from_fn(40, |i| y_mat[(i, col)]);
                    pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap()
                };
                assert_eq!(
                    v.to_bits(),
                    primal.to_bits(),
                    "p={p} col {col}: primal={primal} dual={v}"
                );
            }
        }
    }

    /// A fold fit that keeps a weak component. `X = A·B` has rank 3 at
    /// `p = 200`. Within each fold's rows a `y` column is orthogonal to
    /// `span{1, A}`, hence orthogonal to `X̃_tr` on every training set, plus
    /// `δ·A[:, 0]`. At `δ = 1e-7` `‖X̃'z‖` sits near `δ·‖X̃_tr‖_F·‖z‖`, far
    /// above `fit::w_rel_floor`, so both routes keep the component; `z'Gz`
    /// is then near `δ²·‖X̃_tr‖_F²·‖z‖²`, within a few hundred `ε` of its
    /// rounding, so the closed form's coefficient carries that rounding.
    /// Every column is finite.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn raw_perm_closed_form_is_finite_on_a_weak_component() {
        use rand::RngExt;
        use rand::SeedableRng;
        let (n, p, delta) = (40_usize, 200_usize, 1e-7_f64);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(91);
        let a = Mat::<f64>::from_fn(n, 3, |_, _| rng.random_range(-1.0..1.0));
        let b = Mat::<f64>::from_fn(3, p, |_, _| rng.random_range(-1.0..1.0));
        let x: Mat<f64> = &a * &b;
        let folds = crate::linalg::fold_split(&(0..n).collect::<Vec<_>>(), 4);

        let n_cols = 3;
        let mut y_mat = Mat::<f64>::zeros(n, n_cols);
        for fold in &folds {
            // Orthonormal basis of span{1, A} on this fold's rows.
            let m = fold.len();
            let basis = orthonormal_basis(
                Col::<f64>::from_fn(m, |_| 1.0).as_ref(),
                crate::linalg::row_subset(a.as_ref(), fold).as_ref(),
                1e-8,
            );
            for col in 0..n_cols {
                let mut u = Col::<f64>::from_fn(m, |_| rng.random_range(-1.0..1.0));
                project_off(&basis, &mut u);
                for (i, &r) in fold.iter().enumerate() {
                    y_mat[(r, col)] = u[i] + delta * a[(r, 0)];
                }
            }
        }

        // A constant X predicts zero on every fold (see
        // `raw_perm_dual_route_degenerate_x_is_finite_and_matches_primal`),
        // so its R² is the R² of a zero prediction of the same `y`.
        let x_const = Mat::<f64>::from_fn(n, p, |_, _| 3.0);
        let dual = closed_form_cv_r2(
            x.as_ref(),
            y_mat.as_ref(),
            &folds,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        )
        .unwrap();
        for (col, v) in dual.iter().enumerate() {
            let y_col = Col::<f64>::from_fn(n, |i| y_mat[(i, col)]);
            let primal = pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap();
            let zero = pls1_cv_r2(x_const.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap();
            assert_ne!(
                primal.to_bits(),
                zero.to_bits(),
                "test premise: col {col}'s primal prediction must move R² off a zero prediction's"
            );
            assert!(v.is_finite(), "col {col}: {v}");
        }
    }

    /// A training fold whose standardized `y` is orthogonal to `X̃_tr` up to
    /// rounding: `fit::pls1_kernel` drops the only component (relative floor) and
    /// that fold predicts zero, while the Gram quantities `z'Gz` and `z'G²z`
    /// are pure rounding and cannot make the same call. The closed form
    /// decides such a column from `G` alone: a finite R², the primal's when
    /// `z'Gz` rounds to a negative value. The other columns agree with the
    /// primal. `X` has rank 3 at `p = 200`, so the orthogonal complement of
    /// the training fold's columns is large; `y` on the other folds' rows
    /// stays random.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn raw_perm_closed_form_decides_a_fold_y_orthogonal_to_x() {
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
        // Eight columns; only the odd ones get an orthogonal training part.
        // Which sign rounding gives `z'Gz` is a coin flip per column; a
        // negative one gives the primal's zero prediction, a positive one an
        // order-one prediction.
        let n_cols = 8;
        let mut y_mat = Mat::<f64>::from_fn(n, n_cols, |_, _| rng.random_range(-1.0..1.0));
        for col in (1..n_cols).step_by(2) {
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

        let dual = closed_form_cv_r2(
            x.as_ref(),
            y_mat.as_ref(),
            &folds,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        )
        .unwrap();
        for (col, v) in dual.iter().enumerate() {
            let y_col = Col::<f64>::from_fn(n, |i| y_mat[(i, col)]);
            let primal = pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None).unwrap();
            let scale = primal.abs().max(v.abs());
            let rel = if scale == 0.0 {
                0.0
            } else {
                (v - primal).abs() / scale
            };
            assert!(v.is_finite(), "col {col}: {v}");
            if col % 2 == 0 {
                assert!(rel < 1e-10, "col {col}: primal={primal} dual={v} rel={rel}");
            }
        }
    }

    /// The error and NaN contract of `pls1_cv_r2_columns`, as the `GramP` arm
    /// keeps it (`raw_perm_gram_p_arm_keeps_the_primal_errors_and_nan_columns`).
    /// Row 5 validates in one fold and trains in the others; a NaN there
    /// fails the y-side check of the fold fits that train on it. Column 0
    /// returns the primal route's error; a null column goes NaN while the
    /// others keep the primal route's value.
    #[test]
    fn raw_perm_dual_route_keeps_the_primal_errors_and_nan_columns() {
        use rand::RngExt;
        use rand::SeedableRng;
        let (n, p, n_cols) = (40_usize, 200_usize, 4_usize);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(97);
        let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
        let y_clean = Mat::<f64>::from_fn(n, n_cols, |_, _| rng.random_range(-1.0..1.0));
        let folds = crate::linalg::fold_split(&(0..n).collect::<Vec<_>>(), 4);
        let with_nan = |bad: usize| {
            let mut y = y_clean.clone();
            y[(5, bad)] = f64::NAN;
            y
        };
        let primal = |y: &Mat<f64>, col: usize| {
            let y_col = Col::<f64>::from_fn(n, |i| y[(i, col)]);
            pls1_cv_r2(x.as_ref(), y_col.as_ref(), 1, &folds, None, None)
        };

        let clean = closed_form_cv_r2(
            x.as_ref(),
            y_clean.as_ref(),
            &folds,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        )
        .expect("clean");
        let y2 = with_nan(2);
        let hit = closed_form_cv_r2(
            x.as_ref(),
            y2.as_ref(),
            &folds,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        )
        .expect("a failed null column fails soft");
        assert!(hit[2].is_nan(), "a failed null column is NaN");
        for col in (0..n_cols).filter(|&c| c != 2) {
            assert_eq!(
                hit[col].to_bits(),
                clean[col].to_bits(),
                "col {col} untouched"
            );
            let want = primal(&y2, col).expect("primal");
            let rel = (hit[col] - want).abs() / want.abs().max(hit[col].abs());
            assert!(rel < 1e-10, "col {col}: primal={want} dual={}", hit[col]);
        }
        assert!(
            primal(&y2, 2).is_err(),
            "test premise: the primal route fails column 2"
        );

        let y0 = with_nan(0);
        let e_d = closed_form_cv_r2(
            x.as_ref(),
            y0.as_ref(),
            &folds,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        )
        .expect_err("column 0 fails hard");
        let e_p = primal(&y0, 0).expect_err("column 0 fails hard");
        assert_eq!(e_d.code(), e_p.code());
        assert_eq!(e_d.to_string(), e_p.to_string());
    }

    /// The error and NaN contract of `pooled_cv_r2_columns` on the Primal
    /// route, on the inputs of
    /// `raw_perm_dual_route_keeps_the_primal_errors_and_nan_columns`, with
    /// the four folds one at a time (budget 0) and all in one batch (no
    /// limit): the same values to the bit, the same NaN column, and the same
    /// error, which is the single-column fold loop's.
    #[test]
    fn pooled_columns_keep_their_errors_and_nan_columns_at_every_budget() {
        use rand::RngExt;
        use rand::SeedableRng;
        let (n, p, n_cols) = (40_usize, 200_usize, 4_usize);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(97);
        let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
        let y_clean = Mat::<f64>::from_fn(n, n_cols, |_, _| rng.random_range(-1.0..1.0));
        let folds = crate::linalg::fold_split(&(0..n).collect::<Vec<_>>(), 4);
        let with_nan = |bad: usize| {
            let mut y = y_clean.clone();
            y[(5, bad)] = f64::NAN;
            y
        };
        let run = |y: &Mat<f64>, budget: u64| {
            crate::fit::with_budget(budget, || {
                pooled_cv_r2_columns(
                    ReplicateRoute::Primal,
                    x.as_ref(),
                    &folds,
                    None,
                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                    &[1],
                    None,
                    n_cols,
                    &|c| Col::<f64>::from_fn(n, |i| y[(i, c)]),
                )
                .map(|m| first_row(&m))
            })
        };
        let bits = |v: &[f64]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();

        let clean = run(&y_clean, 0).expect("clean");
        assert_eq!(bits(&clean), bits(&run(&y_clean, u64::MAX).expect("clean")));
        let y2 = with_nan(2);
        for budget in [0, u64::MAX] {
            let hit = run(&y2, budget).expect("a failed null column fails soft");
            assert!(
                hit[2].is_nan(),
                "budget {budget}: a failed null column is NaN"
            );
            for col in (0..n_cols).filter(|&c| c != 2) {
                assert_eq!(
                    hit[col].to_bits(),
                    clean[col].to_bits(),
                    "budget {budget}: col {col} untouched"
                );
            }
        }

        let y0 = with_nan(0);
        let y0_col = Col::<f64>::from_fn(n, |i| y0[(i, 0)]);
        let e_single = pls1_cv_r2(x.as_ref(), y0_col.as_ref(), 1, &folds, None, None)
            .expect_err("column 0 fails hard");
        for budget in [0, u64::MAX] {
            let e = run(&y0, budget).expect_err("column 0 fails hard");
            assert_eq!(e.code(), e_single.code(), "budget {budget}");
            assert_eq!(e.to_string(), e_single.to_string(), "budget {budget}");
        }
    }

    // ── split_exact tests ────────────────────────────────────────────────────

    // The split tests read `√w·y` as given, so a shift of `y` must not move
    // their statistic: weighted, on the no-refit (k = 1) and refit (k = 2)
    // routes and in `split_nb`.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn weighted_split_tests_ignore_the_location_of_y() {
        let (x, y) = synth(60, 5, 3, 4.0, 17);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 1.5 } else { 0.5 });
        let y_far = Col::<f64>::from_fn(60, |i| y[i] + 40.0);
        let run = |y: ColRef<'_, f64>, k: usize, args| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y,
                    k: Some(&[k]),
                    weights: Some(w.as_ref()),
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(3),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 49,
            n_splits: 5,
        };
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: 16,
            force: true,
        };
        for (k, args) in [(1, exact), (2, exact), (1, nb)] {
            let (a, b) = (run(y.as_ref(), k, args), run(y_far.as_ref(), k, args));
            assert!((a.statistic.unwrap() - b.statistic.unwrap()).abs() < 1e-10);
            assert!((a.pvalue - b.pvalue).abs() < 1e-10);
        }
    }

    // Route selection is by input shape, not by a caller knob. Each case is
    // pinned bit-for-bit against the route it must land on, so a future edit
    // to the predicate fails here rather than silently switching routes.
    // The routes are not caller-selectable, so the comparator side calls
    // `run_split_perm_nr` / `run_split_perm` directly. `run_route`
    // reproduces exactly the preprocessing `pls1_confirmatory_test` performs
    // ahead of dispatch — weight normalization and `resolve_seed` — so the comparator sees the same
    // `(w_norm, rng)` state dispatch would hand the route, and the centred
    // `y` of `centre_targets`. `SplitExact` args
    // never trip the `split_nb` auto-gate (the only other RNG consumer ahead
    // of dispatch), so no extra RNG draws sit between `resolve_seed` and the
    // route call on either side.
    #[test]
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
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
                    k: Some(&[k]),
                    weights,
                    covariates: None,
                    null_y: None,
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
            let (y, _) = centre_targets(y.as_ref(), w_norm.as_ref().map(Col::as_ref), None);
            if refit {
                run_split_perm(
                    ReplicateRoute::Primal,
                    x.as_ref(),
                    y.as_ref(),
                    &[k],
                    n_perm,
                    n_splits,
                    w_norm.as_ref().map(Col::as_ref),
                    crate::linalg::held_out_scale(
                        x.as_ref(),
                        w_norm.as_ref().map(Col::as_ref),
                        false,
                    )
                    .as_ref(),
                    &opts,
                    None,
                    None,
                    &mut rng,
                )
                .unwrap()
                .per_k
                .remove(0)
            } else {
                run_split_perm_nr(
                    x.as_ref(),
                    y.as_ref(),
                    k,
                    n_perm,
                    n_splits,
                    w_norm.as_ref().map(Col::as_ref),
                    crate::linalg::held_out_scale(
                        x.as_ref(),
                        w_norm.as_ref().map(Col::as_ref),
                        false,
                    )
                    .as_ref(),
                    &opts,
                    None,
                    None,
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
            assert_eq!(a.statistic.unwrap().to_bits(), b.statistic.to_bits());
        }
        // The seed reaches the no-refit route: another seed draws other
        // splits, and the statistic (tanh of the observed column's z̄) only
        // matches when the splits do.
        assert_ne!(
            run_exact(1, None, None, 2).statistic.unwrap().to_bits(),
            run_exact(1, None, None, 3).statistic.unwrap().to_bits(),
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
                assert_eq!(
                    e.statistic.unwrap().to_bits(),
                    s.statistic.to_bits(),
                    "{what}"
                );
            }
        }
    }

    // ── split_exact's no-refit route (split_perm_nr) tests ──────────────────

    // Test 1: the equivalence test. Route A is an
    // honest per-split, per-column pls1_fit refit; route B is the shipped
    // batched path (split_perm_nr_zbars). Same seed ⇒ same splits and same
    // permuted Y columns for both routes (each draws draw_splits, then one
    // child seed per null column, from the same RNG stream), so they
    // evaluate the same quantity two ways and must agree — this is the
    // property the method rests on (see "Why no refits" on `split_perm_nr_zbars`).
    // Small B=49, J=5 so the refitting route is affordable in a unit test.
    //
    // Shared by two call sites below (not parameterized #[test]: the crate
    // has no test-matrix macro in use elsewhere) so every route of
    // `split_perm_nr_zbars`' cost model is exercised. `expect` is checked
    // against `k1_route`, the rule `split_perm_nr_zbars` routes by, so the
    // coverage claim is enforced rather than assumed; the callers run the
    // route-B cases inside `with_gram_routes_disabled`.
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
        expect: K1Route,
        weighted: bool,
    ) {
        use crate::linalg::{
            col_row_subset, normalize_weights, row_subset, standardize, standardize_apply,
            standardize_weighted, standardize1, standardize1_weighted,
        };
        use crate::resample::{permutation_from_seed, split_sizes};

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

        // Cost-model check, on the rule `split_perm_nr_zbars` routes by:
        // fails loudly if an edit to the rule moves this configuration to
        // the other branch, so the coverage claim below stays true rather
        // than assumed.
        let (n_train, n_test) = split_sizes(n, k);
        let route = k1_route(n, n_train, n_test, p, n_cols, n_splits, false);
        assert_eq!(
            route, expect,
            "cost-model check: expected {expect:?}, computed {route:?} \
             (n_train={n_train}, n_test={n_test}, p={p}, B+1={n_cols})"
        );

        // Route B (shipped): draws splits + Y from its own RNG stream.
        let z_bar_b = nr_zbars(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed);

        // Route A (reference, test-only): same seed ⇒ resolve_seed +
        // draw_splits/child_seeds draw identically, so re-deriving them
        // here reproduces route B's exact splits and Y columns.
        let (_, mut rng_a) = crate::rng::resolve_seed(Some(seed)).unwrap();
        // The one column scale over all rows, as `pls1_fit` standardizes X.
        let scale = match w_norm {
            Some(w) => standardize_weighted(x.as_ref(), Some(w)).2,
            None => standardize(x.as_ref()).2,
        };
        struct Split {
            tr: Vec<usize>,
            te: Vec<usize>,
            xs_tr: Mat<f64>,
            xs_te: Mat<f64>,
            /// Per-half mean-1-renormalized weights; `None` when unweighted.
            w_tr: Option<Col<f64>>,
            w_te: Option<Col<f64>>,
        }
        let splits: Vec<Split> = draw_splits(n, k, n_splits, &mut rng_a)
            .unwrap()
            .into_iter()
            .map(|SplitIdx { tr, te }| {
                let x_tr = row_subset(x.as_ref(), &tr);
                let x_te = row_subset(x.as_ref(), &te);
                // The √w row-scaling convention, spelled out the long way (this is the
                // reference — it deliberately does not call the production
                // helper): renormalize each half's weights to mean 1,
                // centre both halves with the training half's weighted
                // means, divide by the all-row scale, then √w row-scale.
                let w_tr =
                    w_norm.map(|w| normalize_weights(col_row_subset(w, &tr).as_ref()).unwrap());
                let w_te =
                    w_norm.map(|w| normalize_weights(col_row_subset(w, &te).as_ref()).unwrap());
                let (_, mean, _) = if let Some(w) = w_tr.as_ref() {
                    standardize_weighted(x_tr.as_ref(), Some(w.as_ref()))
                } else {
                    standardize(x_tr.as_ref())
                };
                let xs_tr = standardize_apply(x_tr.as_ref(), mean.as_ref(), scale.as_ref());
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
        let perms: Vec<Vec<usize>> = crate::rng::child_seeds(&mut rng_a, n_perm)
            .into_iter()
            .map(|s| permutation_from_seed(n, s))
            .collect();
        let y_mat = Mat::<f64>::from_fn(n, n_cols, |i, col| {
            if col == 0 { y[i] } else { y[perms[col - 1][i]] }
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

    /// Every route, unweighted and weighted. n = 60, B+1 = 50, J = 5: at
    /// p = 5 route A costs 75,000 against G's 459,000 and B's 495,000; at
    /// p = 40 route G costs 522,000 against A's 600,000 and B's 810,000; at
    /// p = 100, without the Gram routes, route B costs 1,350,000 against
    /// A's 1,500,000. The
    /// weighted rows license the weighted derivation: the batched map folds
    /// `diag(√w_tr)` into a fixed linear map, the reference refits honestly
    /// under the √w row-scaling convention, and every one of the B+1 columns
    /// must agree.
    #[test]
    fn split_perm_nr_route_matches_honest_refit() {
        for (p, expect, weighted) in [
            (5, K1Route::A, false),
            (5, K1Route::A, true),
            (40, K1Route::G, false),
            (40, K1Route::G, true),
            (100, K1Route::B, false),
            (100, K1Route::B, true),
        ] {
            on_route(expect, || {
                assert_split_perm_nr_route_matches_honest_refit(
                    60, p, 4.0, 7, 49, 5, 11, expect, weighted,
                );
            });
        }
    }

    /// Several blocks per split: n = 40 gives 20-row halves and a block of
    /// 13,104 columns, so p = 30,000 is three blocks, the last one short.
    /// At J = 3, B + 1 = 21 takes route G (2.40e7 against B's 3.60e7 and
    /// A's 7.56e7), and route B without the Gram routes; B + 1 = 5 takes
    /// route A (1.80e7 against G's 2.40e7).
    #[test]
    fn split_perm_nr_route_matches_honest_refit_over_several_blocks() {
        assert_eq!(crate::linalg::stream_block_width(20), 13_104);
        for (n_perm, expect, weighted) in [
            (20, K1Route::G, false),
            (20, K1Route::G, true),
            (20, K1Route::B, false),
            (20, K1Route::B, true),
            (4, K1Route::A, false),
            (4, K1Route::A, true),
        ] {
            on_route(expect, || {
                assert_split_perm_nr_route_matches_honest_refit(
                    40, 30_000, 4.0, 7, n_perm, 3, 11, expect, weighted,
                );
            });
        }
    }

    /// `nr_split_r` is one split of the no-refit route: each column agrees
    /// with the refit route's `split_half_r` on the same split. n = 40
    /// gives 20-row halves and a block of 13,104 columns, so p = 30,000 is
    /// three blocks.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nr_split_r_matches_split_half_r_per_column() {
        assert_eq!(crate::linalg::stream_block_width(20), 13_104);
        let (n, p, n_perm) = (40_usize, 30_000_usize, 6_usize);
        let (x, y) = synth(n, p, 3, 4.0, 7);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let (n_train, _) = crate::resample::split_sizes(n, 1);
        let (tr, te) = crate::resample::one_split(n, n_train, &mut rng);
        let sp = SplitIdx { tr, te };
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let y_mat = nr_y_mat(y.as_ref(), &seeds);
        for route_b in [false, true] {
            let r = nr_split_r(
                x.as_ref(),
                y_mat.as_ref(),
                &sp,
                None,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                route_b,
            );
            assert_eq!(r.len(), n_perm + 1);
            for (col, rc) in r.iter().enumerate() {
                let y_col = Col::<f64>::from_fn(n, |i| y_mat[(i, col)]);
                let want = split_half_r(
                    x.as_ref(),
                    y_col.as_ref(),
                    1,
                    &sp,
                    None,
                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                    None,
                );
                assert!(
                    (rc - want).abs() <= 1e-10,
                    "route_b={route_b} col {col}: {rc} vs {want}"
                );
            }
        }
    }

    /// Route A on a tiny-scale X used as given (scale one), 200 × 5 at
    /// `1e-9` and `3e-8`: where `‖X̃_tr'z‖²/‖z‖²` does not clear the
    /// kernel's `t't` exit, route A reads `t't` from its extra pass over the
    /// training half, and every column agrees with the refit on the same
    /// split to `1e-10`. At `1e-9` the refit drops every column (`t't` about
    /// `1e-16`) and some bounds are open; at `3e-8` some column's bound is
    /// open and the refit keeps it (`t't` about `1e-13`), so dropping open
    /// columns would fail as well as keeping them.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nr_split_r_reads_tt_where_its_bound_leaves_it_open() {
        let (n, p, n_perm) = (200_usize, 5_usize, 20_usize);
        let (x0, y) = synth(n, p, 3, 1.0, 9);
        let xs = crate::linalg::standardize(x0.as_ref()).0;
        let ones = Col::<f64>::from_fn(p, |_| 1.0);
        let (_, mut rng) = crate::rng::resolve_seed(Some(4)).unwrap();
        let (n_train, _) = crate::resample::split_sizes(n, 1);
        let (tr, te) = crate::resample::one_split(n, n_train, &mut rng);
        let sp = SplitIdx { tr, te };
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let y_mat = nr_y_mat(y.as_ref(), &seeds);
        for (s, want_kept) in [(1e-9, false), (3e-8, true)] {
            let x = Mat::<f64>::from_fn(n, p, |i, j| xs[(i, j)] * s);
            let prep = prepare_split(x.as_ref(), &sp, None, ones.as_ref());
            let r = nr_split_r(x.as_ref(), y_mat.as_ref(), &sp, None, ones.as_ref(), false);
            let (mut open, mut kept, mut open_kept) = (0_usize, 0_usize, 0_usize);
            for (col, rc) in r.iter().enumerate() {
                let (z, _, _) = nr_train_z(col, y_mat.as_ref(), &sp, None, None);
                let g = crate::linalg::mat_vec(prep.xs_tr.transpose(), z.as_ref(), Par::Seq);
                let is_open =
                    g.squared_norm_l2() / z.squared_norm_l2() < crate::fit::NIPALS_ABS_FLOOR;
                let want = split_column_r(&prep, &sp, &|i| y_mat[(i, col)], &[1], None)[0];
                open += usize::from(is_open);
                kept += usize::from(want != 0.0);
                open_kept += usize::from(is_open && want != 0.0);
                assert!(
                    (rc - want).abs() <= 1e-10,
                    "scale {s:e}, col {col}: {rc} vs {want}"
                );
            }
            assert!(open > 0, "premise: scale {s:e} leaves some bound open");
            assert_eq!(
                kept > 0,
                want_kept,
                "premise: scale {s:e}, refit keeps {kept}"
            );
            assert_eq!(
                open_kept > 0,
                want_kept,
                "premise: scale {s:e}, {open_kept} columns open and kept"
            );
        }
    }

    /// A constant X: route A decides the column from its `X̃_tr'z`, route B
    /// from its `K_A`; `X̃_tr = 0` on both, so the zero model (`r = 0`), the
    /// refit route's answer.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nr_split_r_decides_a_constant_x_as_the_zero_model() {
        let (n, p) = (40_usize, 12_usize);
        let x = Mat::<f64>::from_fn(n, p, |_, j| (j + 1) as f64);
        let (_, y) = synth(n, p, 3, 4.0, 7);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let (n_train, _) = crate::resample::split_sizes(n, 1);
        let (tr, te) = crate::resample::one_split(n, n_train, &mut rng);
        let sp = SplitIdx { tr, te };
        let y_mat = nr_y_mat(y.as_ref(), &[]);
        for route_b in [false, true] {
            let r = nr_split_r(
                x.as_ref(),
                y_mat.as_ref(),
                &sp,
                None,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                route_b,
            );
            assert_eq!(r, vec![0.0], "route_b={route_b}");
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
        let (n_perm, n_splits, seed) = (9_usize, 4_usize, 3_u64);

        let z_bar = nr_zbars(x.as_ref(), y.as_ref(), n_perm, n_splits, None, seed);
        for zb in &z_bar {
            assert_eq!(
                *zb, 0.0,
                "the no-refit route must return exactly 0, not NaN"
            );
        }
    }

    // ── split_exact routes on degenerate training halves and rescaled y ─────

    /// Replays `run_split_perm_nr`'s draws off `seed` (the J splits through
    /// `draw_splits`, then one child seed per null column) and evaluates
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
        let n = x.nrows();
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(n, 1, n_splits, &mut rng).unwrap();
        let perms: Vec<Vec<usize>> = crate::rng::child_seeds(&mut rng, n_perm)
            .into_iter()
            .map(|s| crate::resample::permutation_from_seed(n, s))
            .collect();
        let r_obs = split_half_correlations(
            x,
            y,
            1,
            &splits,
            w_norm,
            crate::linalg::held_out_scale(x, w_norm, false).as_ref(),
            None,
        );
        let mut zbars = vec![mean_fisher_z(&r_obs)];
        for perm in &perms {
            let y_perm = Col::<f64>::from_fn(n, |i| y[perm[i]]);
            let r = split_half_correlations(
                x,
                y_perm.as_ref(),
                1,
                &splits,
                w_norm,
                crate::linalg::held_out_scale(x, w_norm, false).as_ref(),
                None,
            );
            zbars.push(mean_fisher_z(&r));
        }
        (splits, zbars, r_obs)
    }

    /// The splits `split_perm_nr_zbars` draws off `seed`.
    fn nr_splits(n: usize, n_splits: usize, seed: u64) -> Vec<SplitIdx> {
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        draw_splits(n, 1, n_splits, &mut rng).unwrap()
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

    /// The no-refit route's z̄ at k = 1 on `run_split_perm_nr`'s draws off
    /// `seed`: the J splits, then one child seed per null column.
    fn nr_zbars(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        n_perm: usize,
        n_splits: usize,
        w_norm: Option<ColRef<'_, f64>>,
        seed: u64,
    ) -> Vec<f64> {
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(x.nrows(), 1, n_splits, &mut rng).unwrap();
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let cols = crate::resample::Columns {
            y: y.as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        split_perm_nr_zbars(
            x,
            &cols,
            &splits,
            1,
            w_norm,
            crate::linalg::held_out_scale(x, w_norm, false).as_ref(),
            &ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
                seed: Some(seed),
                ..Default::default()
            },
            None,
            None,
        )
        .unwrap()
    }

    /// `nr_outcome_matrix` of `y` and the null columns of `seeds`.
    fn nr_y_mat(y: ColRef<'_, f64>, seeds: &[u64]) -> Mat<f64> {
        nr_outcome_matrix(&crate::resample::Columns {
            y: y.as_mat(),
            seeds,
            null_map: None,
        })
    }

    /// The `(p, weighted, route)` cases the degenerate-half tests cover:
    /// every route of `split_perm_nr_zbars`, with and without weights, at
    /// n = 40, B = 20, J = 4. Run each through [`on_route`].
    const DEGENERATE_HALF_CASES: [(usize, bool, K1Route); 6] = [
        (5, false, K1Route::A),
        (5, true, K1Route::A),
        (60, false, K1Route::G),
        (60, true, K1Route::G),
        (1000, false, K1Route::B),
        (1000, true, K1Route::B),
    ];

    /// `f`, inside `with_gram_routes_disabled` when `route` is B, which
    /// `k1_route` picks only without the Gram routes on these tests' shapes.
    fn on_route<T>(route: K1Route, f: impl FnOnce() -> T) -> T {
        if route == K1Route::B {
            with_gram_routes_disabled(f)
        } else {
            f()
        }
    }

    fn degenerate_half_weights(n: usize) -> Col<f64> {
        crate::linalg::normalize_weights(
            Col::<f64>::from_fn(n, |i| 0.25 + ((i * 7) % 5) as f64 * 0.5).as_ref(),
        )
        .unwrap()
    }

    /// Weights that zero all but four of 40 rows, so some split has a half
    /// of zero total weight, which the streamed routes run unweighted:
    /// route G forms that split's halves from the Gram matrix without the
    /// row scaling and decides every column of every split, and the z̄ of
    /// every column agrees with route B's (the Gram routes taken out).
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn route_g_decides_a_split_with_a_zero_weight_half() {
        let (n, p, n_perm, n_splits, seed) = (40_usize, 120_usize, 99_usize, 8_usize, 5_u64);
        let (x, y) = synth(n, p, 3, 1.0, 61);
        let w = Col::<f64>::from_fn(n, |i| if i % 13 == 0 { 1.0 + i as f64 } else { 0.0 });
        let wn = normalize_weights(w.as_ref()).unwrap();
        let splits = nr_splits(n, n_splits, seed);
        let zero_half = |sp: &SplitIdx| {
            half_weight_sum(wn.as_ref(), &sp.tr) == 0.0
                || half_weight_sum(wn.as_ref(), &sp.te) == 0.0
        };
        assert!(splits.iter().any(zero_half), "premise: a zero-weight half");
        let (n_train, n_test) = crate::resample::split_sizes(n, 1);
        assert_eq!(
            k1_route(n, n_train, n_test, p, n_perm + 1, n_splits, false),
            K1Route::G
        );
        assert_eq!(
            with_gram_routes_disabled(|| k1_route(
                n,
                n_train,
                n_test,
                p,
                n_perm + 1,
                n_splits,
                false
            )),
            K1Route::B
        );
        let scale = crate::linalg::held_out_scale(x.as_ref(), Some(wn.as_ref()), false);
        let (g, plain) =
            k1_route_gram(x.as_ref(), Some(wn.as_ref()), scale.as_ref(), None, &splits);
        assert!(plain.is_some(), "the plain Gram matrix is kept");
        let y_mat = nr_y_mat(y.as_ref(), &[]);
        for (j, sp) in splits.iter().enumerate() {
            let r = nr_split_r_gram(&g, plain.as_deref(), y_mat.as_ref(), sp, Some(wn.as_ref()));
            assert!(r.iter().all(|v| v.is_finite()), "split {j}: {r:?}");
        }
        let run = || {
            nr_zbars(
                x.as_ref(),
                y.as_ref(),
                n_perm,
                n_splits,
                Some(wn.as_ref()),
                seed,
            )
        };
        assert_zbars_close(&run(), &with_gram_routes_disabled(run), "G against B");
    }

    /// `split_nb` at k = 1 on dense input takes each split's correlation
    /// from the no-refit body; it agrees with the refit correlations on a
    /// one-block shape and on a three-block shape (20-row halves,
    /// 13,104-column blocks), weighted or not.
    #[test]
    fn split_nb_k1_dense_matches_the_refit_correlations() {
        assert_eq!(crate::linalg::stream_block_width(20), 13_104);
        for (n, p) in [(60_usize, 5_usize), (40, 30_000)] {
            let (x, y) = synth(n, p, 3, 4.0, 7);
            let w = degenerate_half_weights(n);
            let (_, mut rng) = crate::rng::resolve_seed(Some(11)).unwrap();
            let splits = draw_splits(n, 1, 6, &mut rng).unwrap();
            for wopt in [None, Some(w.as_ref())] {
                let got = split_nb_correlations_nr(
                    x.as_ref(),
                    y.as_ref(),
                    &splits,
                    wopt,
                    crate::linalg::held_out_scale(x.as_ref(), wopt, false).as_ref(),
                    None,
                );
                let want = split_half_correlations(
                    x.as_ref(),
                    y.as_ref(),
                    1,
                    &splits,
                    wopt,
                    crate::linalg::held_out_scale(x.as_ref(), wopt, false).as_ref(),
                    None,
                );
                assert_eq!(got.nrows(), want.nrows());
                for j in 0..got.nrows() {
                    assert!(
                        (got[j] - want[j]).abs() <= 1e-10,
                        "n={n} p={p} weighted={} split {j}: {} vs {}",
                        wopt.is_some(),
                        got[j],
                        want[j]
                    );
                }
            }
        }
    }

    /// `k1_route` at J = 32 on the shapes its cost model is written for:
    /// route G on wide X, route A on tall X, and a Gram matrix the caller
    /// already holds counts as free. Without the Gram routes the wide
    /// shape takes route B.
    #[test]
    fn k1_route_picks_g_on_wide_and_a_on_tall() {
        let route = |n: usize, p: usize, n_cols: usize, gram_built: bool| {
            let (n_train, n_test) = crate::resample::split_sizes(n, 1);
            k1_route(n, n_train, n_test, p, n_cols, 32, gram_built)
        };
        assert_eq!(route(254, 1_447_336, 1001, false), K1Route::G);
        assert_eq!(route(928, 220_000, 1001, false), K1Route::G);
        assert_eq!(route(1000, 20, 1001, false), K1Route::A);
        // G 8.4e7, B 1.44e8, A 2.56e9.
        assert_eq!(route(100, 800, 1001, false), K1Route::G);
        assert_eq!(route(254, 20_000, 1, true), K1Route::G);
        assert_eq!(route(254, 20_000, 1, false), K1Route::A);
        with_gram_routes_disabled(|| {
            assert_eq!(route(100, 800, 1001, false), K1Route::B);
        });
    }

    /// Route G gives `split_exact` at k = 1 and `split_nb` the statistic and
    /// p-value of the streamed routes within `1e-12` relative, on wide
    /// 40 × 120 X, unweighted and with one weight at `1e-6`, on an ordinary
    /// `y` and a constant `y`; on a `y` orthogonal to a training half, a
    /// finite statistic on both. On the
    /// rank-3 design the gate reroutes `split_nb` to `split_exact`, which
    /// runs from the gate's Gram matrix.
    #[test]
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    fn g_route_matches_streamed_route() {
        let (n, p, n_splits, seed) = (40_usize, 120_usize, 8_usize, 5_u64);
        let (n_train, n_test) = crate::resample::split_sizes(n, 1);
        assert_eq!(
            k1_route(n, n_train, n_test, p, 100, n_splits, false),
            K1Route::G
        );
        assert_eq!(
            k1_route(n, n_train, n_test, p, 1, n_splits, true),
            K1Route::G
        );
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        let splits = nr_splits(n, n_splits, seed);
        let mut w = Col::<f64>::from_fn(n, |_| 1.0);
        w[3] = 1e-6;
        let wn = normalize_weights(w.as_ref()).unwrap();
        let (x_plain, y_plain) = synth(n, p, 3, 1.0, 51);
        // Route G decides every column of the ordinary design itself, the
        // observed one and 20 null columns, on every split.
        let y_mat = nr_y_mat(y_plain.as_ref(), &(0..20).collect::<Vec<u64>>());
        for wr in [None, Some(wn.as_ref())] {
            let scale = crate::linalg::held_out_scale(x_plain.as_ref(), wr, false);
            let (g, plain) = k1_route_gram(x_plain.as_ref(), wr, scale.as_ref(), None, &splits);
            for (j, sp) in splits.iter().enumerate() {
                let r = nr_split_r_gram(&g, plain.as_deref(), y_mat.as_ref(), sp, wr);
                assert!(
                    r.iter().all(|v| v.is_finite()),
                    "weighted={} split {j}: {r:?}",
                    wr.is_some()
                );
            }
        }
        let y_const = Col::<f64>::from_fn(n, |_| 2.5);
        for weighted in [false, true] {
            let wopt = weighted.then_some(w.as_ref());
            let (x_orth, y_orth) = design_with_orthogonal_train_half(
                n,
                p,
                &splits[0].tr,
                weighted.then_some(wn.as_ref()),
                1.0,
                17,
            );
            for (label, x, y) in [
                ("ordinary", &x_plain, &y_plain),
                ("constant y", &x_plain, &y_const),
                ("orthogonal half", &x_orth, &y_orth),
            ] {
                for args in [
                    ConfirmatoryArgs::SplitExact {
                        n_perm: 99,
                        n_splits,
                    },
                    ConfirmatoryArgs::SplitNb {
                        n_splits,
                        force: true,
                    },
                    ConfirmatoryArgs::SplitNb {
                        n_splits,
                        force: false,
                    },
                ] {
                    let run = || {
                        pls1_confirmatory_test(
                            ConfirmatoryTestInput::Raw {
                                x: x.as_ref(),
                                y: y.as_ref(),
                                k: Some(&[1]),
                                weights: wopt,
                                covariates: None,
                                null_y: None,
                            },
                            ConfirmatoryTestOpts {
                                args,
                                seed: Some(seed),
                                ..Default::default()
                            },
                        )
                        .unwrap()
                    };
                    let (got, want) = (run(), with_gram_routes_disabled(run));
                    let what = format!("{label}, weighted={weighted}, {args:?}");
                    assert_eq!(got.test_method, want.test_method, "{what}");
                    if label == "orthogonal half"
                        && matches!(args, ConfirmatoryArgs::SplitNb { force: false, .. })
                    {
                        assert_eq!(got.test_method, "split_exact", "premise: {what}");
                    }
                    if label == "orthogonal half" {
                        // Decided at the rounding of each route's `K_A`:
                        // finite, not equal.
                        assert!(
                            got.statistic.unwrap().is_finite()
                                && want.statistic.unwrap().is_finite(),
                            "{what}: {} vs {}",
                            got.statistic.unwrap(),
                            want.statistic.unwrap()
                        );
                        continue;
                    }
                    assert!(
                        close(got.statistic.unwrap(), want.statistic.unwrap()),
                        "{what}: statistic {} vs {}",
                        got.statistic.unwrap(),
                        want.statistic.unwrap()
                    );
                    assert!(
                        close(got.pvalue, want.pvalue),
                        "{what}: p {} vs {}",
                        got.pvalue,
                        want.pvalue
                    );
                }
            }
        }
    }

    /// A wide call that runs the `split_nb` gate builds one Gram matrix, the
    /// gate's, which the k = 1 test then runs from: `split_nb` (weighted or
    /// not), `split_nb` rerouted to `split_exact` on a rank-3 X, `auto`
    /// whether the gate passes or fires, and the first step of the
    /// sequence. On every shape route G wins even when it must build its
    /// own Gram matrix, so a call that dropped the gate's would build two.
    #[test]
    fn split_nb_builds_one_gram() {
        use crate::sequential::{
            IncrementalSequenceOpts, SequentialArgs, run_incremental_sequence,
        };
        let one_build = |what: &str, run: &dyn Fn() -> String, method: &str| {
            let before = crate::held_out_gram::grams_built();
            assert_eq!(run(), method, "{what}");
            assert_eq!(crate::held_out_gram::grams_built() - before, 1, "{what}");
        };
        let auto = |x: &Mat<f64>, y: &Col<f64>| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[1]),
                    weights: None,
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::Auto {
                        n_perm: 99,
                        n_splits: 1000,
                    },
                    seed: Some(3),
                    ..Default::default()
                },
            )
            .unwrap()
            .test_method
        };
        for (n, p, n_cols, n_splits) in
            [(40, 120, 1, 32), (40, 120, 100, 32), (1001, 4000, 1, 1000)]
        {
            let (n_train, n_test) = crate::resample::split_sizes(n, 1);
            assert_eq!(
                k1_route(n, n_train, n_test, p, n_cols, n_splits, false),
                K1Route::G,
                "premise at {n} x {p}, {n_cols} columns"
            );
        }
        let (x, y) = synth(40, 120, 3, 1.0, 51);
        let w = crate::test_support::test_weights(40);
        for weights in [None, Some(w.as_ref())] {
            one_build(
                &format!("split_nb, weighted={}", weights.is_some()),
                &|| gate_run(x.as_ref(), y.as_ref(), weights, 32, true).test_method,
                "split_nb",
            );
        }
        let tr: Vec<usize> = (0..20).collect();
        let (x3, y3) = design_with_orthogonal_train_half(40, 120, &tr, None, 1.0, 17);
        one_build(
            "split_nb rerouted",
            &|| gate_run(x3.as_ref(), y3.as_ref(), None, 32, false).test_method,
            "split_exact",
        );
        let (xa, ya) = synth(1001, 4000, 3, 1.0, 52);
        one_build("auto, gate passes", &|| auto(&xa, &ya), "split_nb");
        let (xf, yf) = design_with_orthogonal_train_half(1001, 4000, &tr, None, 1.0, 18);
        one_build("auto, gate fires", &|| auto(&xf, &yf), "split_exact");
        one_build(
            "sequence step 1",
            &|| {
                run_incremental_sequence(
                    x.as_ref(),
                    y.as_ref(),
                    1,
                    None,
                    40.0,
                    IncrementalSequenceOpts {
                        args: SequentialArgs::SplitNb {
                            n_splits: 32,
                            force: true,
                        },
                        alpha: 0.05,
                        stop_early_override: true,
                        pre_standardized: false,
                        seed: Some(3),
                        verbose: false,
                        keep: None,
                    },
                    None,
                )
                .unwrap()
                .method
            },
            "split_nb",
        );
    }

    /// `split_blocks_bytes` and `resample::fold_blocks_bytes` are what the
    /// standardized halves of a split and the two blocks of a fold allocate,
    /// on halves whose row counts are not multiples of 8 though `n` is.
    #[test]
    fn block_counts_are_what_the_halves_allocate() {
        use crate::resample::fold_blocks_bytes;
        let bytes = |m: &Mat<f64>| m.col_stride().unsigned_abs() * m.ncols() * 8;
        let (n, p) = (40_usize, 3_usize);
        let x = Mat::<f64>::from_fn(n, p, |i, j| ((i * 7 + j * 3) % 11) as f64);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let splits = draw_splits(n, 1, 4, &mut rng).unwrap();
        let most = splits
            .iter()
            .map(|sp| {
                let prep = prepare_split(
                    x.as_ref(),
                    sp,
                    None,
                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                );
                bytes(&prep.xs_tr) + bytes(&prep.xs_te)
            })
            .max()
            .unwrap();
        assert_eq!(split_blocks_bytes(&splits, p), most);
        let idx: Vec<usize> = (0..n).collect();
        let folds = crate::linalg::fold_split(&idx, 3);
        let most = (0..folds.len())
            .map(|fi| {
                let f = prepare_cv_fold(
                    x.as_ref(),
                    &folds,
                    fi,
                    None,
                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                )
                .unwrap();
                bytes(&f.xs_tr) + bytes(&f.xs_val)
            })
            .max()
            .unwrap();
        assert_eq!(fold_blocks_bytes(n, &folds, p), most);
    }

    /// A PLS1 unit whose replicate columns run in parallel counts, on the
    /// primal route, its blocks of X and one fit's buffers per column that
    /// can run at once (one per thread, at most the column count), and on
    /// the n-space route its `G` and `M`. A budget of two such units runs
    /// two folds or splits at once and a byte less runs one, on a pool of 1
    /// thread and on a pool of 4, for `pooled_cv_r2_columns` and
    /// `split_zbars_columns`.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls1_column_units_count_one_fit_per_running_column() {
        use crate::fit::pls1_fit_bytes;
        use crate::resample::fold_blocks_bytes;
        let (n, p, k, n_cols) = (16_usize, 12_usize, 2_usize, 8_usize);
        let (x, y) = synth(n, p, 3, 4.0, 7);
        let idx: Vec<usize> = (0..n).collect();
        let folds = crate::linalg::fold_split(&idx, 4);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let splits = draw_splits(n, k, 4, &mut rng).unwrap();
        let column_y = |c: usize| {
            if c == 0 {
                y.clone()
            } else {
                let perm = crate::resample::permutation_from_seed(n, c as u64);
                Col::<f64>::from_fn(n, |i| y[perm[i]])
            }
        };
        let fit = pls1_fit_bytes(p, k);
        let (fold_n_tr, split_n_tr) = (n - folds[0].len(), splits[0].tr.len());
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for route in [ReplicateRoute::Primal, ReplicateRoute::Nspace] {
                // On Nspace a unit holds only its Gram blocks; on Primal the
                // halves and one fit per running column.
                let nspace = route == ReplicateRoute::Nspace;
                let fold_unit = if nspace {
                    fold_blocks_bytes(n, &folds, fold_n_tr)
                } else {
                    fold_blocks_bytes(n, &folds, p) + threads * fit
                };
                let split_unit = if nspace {
                    split_blocks_bytes(&splits, split_n_tr)
                } else {
                    split_blocks_bytes(&splits, p) + threads * fit
                };
                for (extra, batch) in [(1, 1), (0, 2)] {
                    let budget = |unit: usize| u64::try_from(2 * unit - extra).unwrap();
                    let (_, log) = pool.install(|| {
                        crate::resample::with_batch_log(|| {
                            crate::fit::with_budget(budget(fold_unit), || {
                                pooled_cv_r2_columns(
                                    route,
                                    x.as_ref(),
                                    &folds,
                                    None,
                                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                                    &[k],
                                    None,
                                    n_cols,
                                    &column_y,
                                )
                            })
                        })
                    });
                    crate::resample::assert_batches(&log, 4, batch, 1);
                    let (_, log) = pool.install(|| {
                        crate::resample::with_batch_log(|| {
                            crate::fit::with_budget(budget(split_unit), || {
                                split_zbars_columns(
                                    route,
                                    x.as_ref(),
                                    &splits,
                                    None,
                                    crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                                    &[k],
                                    None,
                                    n_cols,
                                    &column_y,
                                    None,
                                    None,
                                )
                            })
                        })
                    });
                    crate::resample::assert_batches(&log, 4, batch, 1);
                }
            }
        }
    }

    /// `map_split_columns` keeps at most one column per thread live when
    /// the per-column fit is past the `ParChoice::Auto` threshold (here
    /// `n_tr·p·k = 1000·1000·1`), even when each column uses the pool
    /// itself: a worker waiting on its own column's nested tasks would
    /// otherwise start another column. Every column's value comes back in
    /// column order.
    #[test]
    fn map_split_columns_bounds_live_columns_past_the_auto_threshold() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        let (live, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let out = pool.install(|| {
            map_split_columns(1000, 1000, 1, 64, |c| {
                peak.fetch_max(live.fetch_add(1, SeqCst) + 1, SeqCst);
                let s: usize = crate::resample::map_indexed(16, |i| {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    i * c
                })
                .into_iter()
                .sum();
                live.fetch_sub(1, SeqCst);
                s
            })
        });
        assert_eq!(out, (0..64).map(|c| 120 * c).collect::<Vec<usize>>());
        assert!(peak.load(SeqCst) <= 4, "peak {}", peak.load(SeqCst));
    }

    /// A PLS1 split loop with one fit per split counts the split's halves of
    /// X and that fit's buffers: a budget of two such units runs two of the
    /// four splits at once, a byte less runs one (`split_half_correlations`
    /// at K = 2).
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls1_single_fit_split_units_count_the_fit() {
        use crate::fit::pls1_fit_bytes;
        let (n, p) = (40_usize, 12_usize);
        let (x, y) = synth(n, p, 3, 4.0, 7);
        let k = 2_usize;
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let splits = draw_splits(n, k, 4, &mut rng).unwrap();
        let unit = split_blocks_bytes(&splits, p) + pls1_fit_bytes(p, k);
        for (budget, batch) in [(2 * unit - 1, 1), (2 * unit, 2)] {
            let ((), log) = crate::resample::with_batch_log(|| {
                crate::fit::with_budget(u64::try_from(budget).unwrap(), || {
                    split_half_correlations(
                        x.as_ref(),
                        y.as_ref(),
                        k,
                        &splits,
                        None,
                        crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                        None,
                    );
                });
            });
            assert_eq!(
                log.last(),
                Some(&(4, batch)),
                "budget={budget}; log {log:?}"
            );
        }
    }

    /// `split_half_correlations` at K = 2 gives the same bits with its
    /// splits one at a time (budget 0) and all in one batch (no limit),
    /// unweighted and weighted.
    #[test]
    fn split_half_correlations_do_not_depend_on_the_budget() {
        let n = 60_usize;
        let (x, y) = synth(n, 5, 3, 1.0, 29);
        let w = Col::<f64>::from_fn(n, |i| if i % 3 == 0 { 2.0 } else { 0.5 });
        let w_norm = normalize_weights(w.as_ref()).unwrap();
        let (_, mut rng) = crate::rng::resolve_seed(Some(8)).unwrap();
        let splits = draw_splits(n, 2, 20, &mut rng).unwrap();
        for wopt in [None, Some(w_norm.as_ref())] {
            let run = |b| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(b, || {
                        split_half_correlations(
                            x.as_ref(),
                            y.as_ref(),
                            2,
                            &splits,
                            wopt,
                            crate::linalg::held_out_scale(x.as_ref(), wopt, false).as_ref(),
                            None,
                        )
                    })
                })
            };
            let ((one, log_one), (all, log_all)) = (run(0), run(u64::MAX));
            crate::resample::assert_batches(&log_one, 20, 1, 1);
            crate::resample::assert_batches(&log_all, 20, 20, 1);
            assert!(
                (0..one.nrows()).any(|j| one[j] != 0.0),
                "premise: weighted={} gives a nonzero r",
                wopt.is_some()
            );
            for j in 0..splits.len() {
                assert_eq!(
                    one[j].to_bits(),
                    all[j].to_bits(),
                    "weighted={} split {j}",
                    wopt.is_some()
                );
            }
        }
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
        for (p, weighted, expect) in DEGENERATE_HALF_CASES {
            let w_norm = weighted.then_some(w.as_ref());
            let (x, y) = design_with_orthogonal_train_half(n, p, &splits[0].tr, w_norm, 1e8, 17);

            let (n_train, n_test) = crate::resample::split_sizes(n, 1);
            let route = on_route(expect, || {
                k1_route(n, n_train, n_test, p, n_perm + 1, n_splits, false)
            });
            assert_eq!(route, expect, "cost-model premise at p={p}");

            let (_, z_re, r_obs) =
                refit_zbars_on_nr_draws(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed);
            assert!(
                r_obs[0] == 0.0,
                "test premise: the refit route's fit on split 0 truncates (r = {})",
                r_obs[0]
            );
            let z_nr = on_route(expect, || {
                nr_zbars(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed)
            });
            let what = format!("p={p} weighted={weighted} {expect:?}: no-refit vs refit");
            assert_zbars_close_but_orthogonal(&z_nr, &z_re, expect, &what);
        }
    }

    /// [`assert_zbars_close`], except that routes B and G decide the
    /// observed column, orthogonal to a training half, at the rounding of
    /// their `K_A`: there column 0 is only finite.
    fn assert_zbars_close_but_orthogonal(a: &[f64], b: &[f64], route: K1Route, what: &str) {
        if route == K1Route::A {
            assert_zbars_close(a, b, what);
        } else {
            assert!(a[0].is_finite() && b[0].is_finite(), "{what}: column 0");
            assert_zbars_close(&a[1..], &b[1..], what);
        }
    }

    /// Both `split_exact` routes draw their splits through `draw_splits` and
    /// their null columns through `resample::Columns`, so at one seed the
    /// no-refit route (k = 1, no `keep`) and the refit route (`keep = Some(p)`
    /// keeps every column, so the fit is the dense one) test the same splits
    /// on the same null columns and report the same statistic and p-value.
    #[test]
    fn split_exact_routes_share_their_null_columns() {
        let (n, p) = (60_usize, 8_usize);
        let (x, y) = synth(n, p, 3, 0.5, 31);
        let run = |keep: Option<usize>| {
            assert_eq!(split_exact_no_refit_route(1, keep), keep.is_none());
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[1]),
                    weights: None,
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 20,
                        n_splits: 6,
                    },
                    seed: Some(13),
                    keep,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (no_refit, refit) = (run(None), run(Some(p)));
        assert!(
            (no_refit.statistic.unwrap() - refit.statistic.unwrap()).abs() <= 1e-10,
            "no-refit {} vs refit {}",
            no_refit.statistic.unwrap(),
            refit.statistic.unwrap()
        );
        assert_eq!(
            no_refit.pvalue.to_bits(),
            refit.pvalue.to_bits(),
            "no-refit p {} vs refit p {}",
            no_refit.pvalue,
            refit.pvalue
        );
    }

    /// One run of a route that takes its null columns from
    /// `resample::Columns`: its p-value under an optional `null_map`.
    type RouteRun<'a> = Box<dyn for<'m> Fn(Option<crate::resample::NullMap<'m>>) -> f64 + 'a>;

    /// Hands `check` one case per route that takes its null columns from
    /// `resample::Columns`: its label, its outcome and its run at one seed,
    /// after checking that each case takes the route its label names.
    #[allow(clippy::too_many_lines)]
    fn with_null_column_routes(check: impl FnOnce(&[(&str, MatRef<'_, f64>, RouteRun<'_>)])) {
        use crate::resample::Columns;
        type Run<'a> = RouteRun<'a>;
        const N_PERM: usize = 99;
        const SEED: u64 = 7;
        // `split_exact`: the no-refit route when `route` is `None`, else
        // the refit route on `route`.
        fn split<'a>(
            route: Option<ReplicateRoute>,
            x: &'a Mat<f64>,
            y: &'a Col<f64>,
            k: usize,
        ) -> Run<'a> {
            Box::new(move |map| {
                let (_, mut rng) = crate::rng::resolve_seed(Some(SEED)).unwrap();
                let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
                let (x, y, sc) = (x.as_ref(), y.as_ref(), scale.as_ref());
                let opts = ConfirmatoryTestOpts::default();
                match route {
                    None => {
                        run_split_perm_nr(x, y, k, N_PERM, 4, None, sc, &opts, map, None, &mut rng)
                            .map(SetResult::from)
                    }
                    Some(r) => run_split_perm(
                        r,
                        x,
                        y,
                        &[k],
                        N_PERM,
                        4,
                        None,
                        sc,
                        &opts,
                        map,
                        None,
                        &mut rng,
                    ),
                }
                .unwrap()
                .pvalue
            })
        }
        // `split_exact`'s n-space refit route at k = 2 on a Gram matrix the
        // call holds (the `split_nb` gate's).
        fn split_held<'a>(x: &'a Mat<f64>, y: &'a Col<f64>) -> Run<'a> {
            Box::new(move |map| {
                let (_, mut rng) = crate::rng::resolve_seed(Some(SEED)).unwrap();
                let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
                let (x, y, sc) = (x.as_ref(), y.as_ref(), scale.as_ref());
                let g = HeldOutGram::from_x(x, None, sc);
                let opts = ConfirmatoryTestOpts::default();
                let r = ReplicateRoute::Nspace;
                run_split_perm(
                    r,
                    x,
                    y,
                    &[2],
                    N_PERM,
                    4,
                    None,
                    sc,
                    &opts,
                    map,
                    Some(&g),
                    &mut rng,
                )
                .unwrap()
                .pvalue
            })
        }
        // `raw_perm`, on the route `raw_perm_route` picks; `keep = Some(p)`
        // keeps every column, so the fit is the dense one.
        fn raw<'a>(x: &'a Mat<f64>, y: &'a Col<f64>, k: usize, keep: Option<usize>) -> Run<'a> {
            Box::new(move |map| {
                let (_, mut rng) = crate::rng::resolve_seed(Some(SEED)).unwrap();
                let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
                let opts = ConfirmatoryTestOpts {
                    keep,
                    ..Default::default()
                };
                run_raw_perm(
                    x.as_ref(),
                    y.as_ref(),
                    &[k],
                    N_PERM,
                    5,
                    None,
                    scale.as_ref(),
                    &opts,
                    map,
                    &mut rng,
                )
                .unwrap()
                .pvalue
            })
        }
        // PLS3 `split_exact` on the route `route` names: dual, primal or cells.
        fn pls3<'a>(route: &'static str, x: &'a Mat<f64>, y: &'a Mat<f64>) -> Run<'a> {
            Box::new(move |map| {
                let (_, mut rng) = crate::rng::resolve_seed(Some(SEED)).unwrap();
                let design = draw_design(x.nrows(), 1, 4, &mut rng).unwrap();
                let splits = &design.splits;
                let seeds = crate::rng::child_seeds(&mut rng, N_PERM);
                let cols = Columns {
                    y: y.as_ref(),
                    seeds: &seeds,
                    null_map: map,
                };
                let xsc = crate::linalg::held_out_scale(x.as_ref(), None, false);
                let ysc = crate::linalg::held_out_scale(y.as_ref(), None, false);
                let o = crate::pls3_signal_test::Pls3ConfirmatoryTestOpts::default();
                let z = if route == "dual" {
                    crate::dual_route::pls3_split_zbars_columns(
                        &crate::held_out_gram::HeldOutGram::from_x(x.as_ref(), None, xsc.as_ref()),
                        &cols,
                        splits,
                        ysc.as_ref(),
                        o.keep_y,
                        o.max_iter,
                        o.tol,
                    )
                } else if route == "cells" {
                    crate::pls3_signal_test::pls3_split_zbars_columns_cells(
                        x.as_ref(),
                        &cols,
                        &design,
                        xsc.as_ref(),
                        ysc.as_ref(),
                        &o,
                    )
                } else {
                    crate::pls3_signal_test::pls3_split_zbars_columns_primal(
                        x.as_ref(),
                        &cols,
                        splits,
                        xsc.as_ref(),
                        ysc.as_ref(),
                        &o,
                    )
                };
                let exceedances = z[1..].iter().filter(|v| **v >= z[0]).count();
                (exceedances as f64 + 1.0) / z.len() as f64
            })
        }

        // The same run with the Gram routes taken out of the choice.
        fn streamed(run: Run<'_>) -> Run<'_> {
            Box::new(move |map| with_gram_routes_disabled(|| run(map)))
        }
        // The same run on a 1 KiB memory budget, which takes the moment
        // arms out of the choice.
        fn test_rows(run: Run<'_>) -> Run<'_> {
            Box::new(move |map| crate::fit::with_budget(1024, || run(map)))
        }
        // The same run on an 8 KiB budget, which keeps the moment arms but
        // takes the cell sums out of the choice.
        fn per_split(run: Run<'_>) -> Run<'_> {
            Box::new(move |map| crate::fit::with_budget(8192, || run(map)))
        }

        let (wide_x, wide_y) = synth(40, 200, 200, 1.0, 41);
        let (tall_x, tall_y) = synth(60, 6, 3, 2.0, 43);
        let noise = crate::test_support::unif(60, 2, 44);
        let y3 = Mat::<f64>::from_fn(60, 3, |i, j| match j {
            0 => tall_y[i],
            1 => 0.7 * tall_y[i] + noise[(i, 0)],
            _ => noise[(i, 1)],
        });
        // Premises: the no-refit cases take the routes their labels name,
        // and so does `raw_perm`.
        assert_eq!(k1_route(40, 20, 20, 200, N_PERM + 1, 4, false), K1Route::G);
        assert_eq!(
            with_gram_routes_disabled(|| k1_route(40, 20, 20, 200, N_PERM + 1, 4, false)),
            K1Route::B
        );
        assert_eq!(k1_route(60, 30, 30, 6, N_PERM + 1, 4, false), K1Route::A);
        // The tall cases score the test halves from moments, except on the
        // 1 KiB budget.
        for n_test in [
            crate::resample::split_sizes(60, 1).1,
            crate::resample::split_sizes(60, 2).1,
        ] {
            assert!(test_moments_route(60, n_test, 6, N_PERM + 1, 4, false));
            assert!(crate::fit::with_budget(1024, || !test_moments_route(
                60,
                n_test,
                6,
                N_PERM + 1,
                4,
                false
            )));
            assert!(crate::fit::with_budget(8192, || test_moments_route(
                60,
                n_test,
                6,
                N_PERM + 1,
                4,
                false
            )));
        }
        // ... and sum their training rows from the cells, except on the
        // 8 KiB budget.
        for k in [1, 2] {
            let (_, mut rng) = crate::rng::resolve_seed(Some(SEED)).unwrap();
            let d = draw_design(60, k, 4, &mut rng).unwrap();
            let b = crate::cell_sums::cell_batch(N_PERM + 1);
            assert!(crate::cell_sums::cells_eligible(&d, 6, b, false));
            assert!(crate::fit::with_budget(8192, || {
                !crate::cell_sums::cells_eligible(&d, 6, b, false)
            }));
        }
        // PLS3's cell route serves the tall shape (q = 3).
        {
            let (_, mut rng) = crate::rng::resolve_seed(Some(SEED)).unwrap();
            let d = draw_design(60, 1, 4, &mut rng).unwrap();
            assert!(crate::pls3_signal_test::cells_route_eligible(
                &d, 6, 3, N_PERM
            ));
        }
        assert_eq!(
            raw_perm_route(40, 5, 200, N_PERM, 1, None, false),
            ReplicateRoute::Special
        );
        assert_eq!(
            raw_perm_route(40, 5, 200, N_PERM, 2, None, false),
            ReplicateRoute::Nspace
        );
        assert_eq!(
            raw_perm_route(40, 5, 200, N_PERM, 1, Some(200), false),
            ReplicateRoute::Primal
        );
        assert_eq!(
            with_gram_routes_disabled(|| raw_perm_route(40, 5, 200, N_PERM, 2, None, false)),
            ReplicateRoute::Primal
        );

        let (wide, tall) = (wide_y.as_ref().as_mat(), tall_y.as_ref().as_mat());
        let cases: Vec<(&str, MatRef<'_, f64>, Run<'_>)> = vec![
            ("no-refit G, wide", wide, split(None, &wide_x, &wide_y, 1)),
            (
                "no-refit B, wide",
                wide,
                streamed(split(None, &wide_x, &wide_y, 1)),
            ),
            (
                "no-refit A from cell sums, tall",
                tall,
                split(None, &tall_x, &tall_y, 1),
            ),
            (
                "no-refit A from moments, tall",
                tall,
                per_split(split(None, &tall_x, &tall_y, 1)),
            ),
            (
                "no-refit A on the test rows, tall",
                tall,
                test_rows(split(None, &tall_x, &tall_y, 1)),
            ),
            (
                "refit Primal",
                tall,
                split(Some(ReplicateRoute::Primal), &tall_x, &tall_y, 1),
            ),
            (
                "refit Nspace, k = 2, wide",
                wide,
                split(Some(ReplicateRoute::Nspace), &wide_x, &wide_y, 2),
            ),
            (
                "refit Nspace, k = 2, wide, held G",
                wide,
                split_held(&wide_x, &wide_y),
            ),
            (
                "refit GramP from cell sums, k = 2, tall",
                tall,
                split(Some(ReplicateRoute::GramP), &tall_x, &tall_y, 2),
            ),
            (
                "refit GramP from moments, k = 2, tall",
                tall,
                per_split(split(Some(ReplicateRoute::GramP), &tall_x, &tall_y, 2)),
            ),
            (
                "refit GramP on the test rows, k = 2, tall",
                tall,
                test_rows(split(Some(ReplicateRoute::GramP), &tall_x, &tall_y, 2)),
            ),
            (
                "raw_perm Special, k = 1",
                wide,
                raw(&wide_x, &wide_y, 1, None),
            ),
            (
                "raw_perm Nspace, k = 2",
                wide,
                raw(&wide_x, &wide_y, 2, None),
            ),
            (
                "raw_perm Primal, k = 1",
                wide,
                raw(&wide_x, &wide_y, 1, Some(200)),
            ),
            (
                "raw_perm Primal, k = 2",
                wide,
                streamed(raw(&wide_x, &wide_y, 2, None)),
            ),
            ("PLS3 dual", y3.as_ref(), pls3("dual", &tall_x, &y3)),
            ("PLS3 primal", y3.as_ref(), pls3("primal", &tall_x, &y3)),
            (
                "PLS3 from cell sums",
                y3.as_ref(),
                pls3("cells", &tall_x, &y3),
            ),
        ];
        check(&cases);
    }

    /// Every route builds its null columns through `resample::Columns`: a
    /// `null_map` that writes `y` into every null column makes every null
    /// statistic the observed one, so p = 1 (a tie counts as an exceedance),
    /// where the same call without the map has p < 1. One case per route.
    #[test]
    fn every_route_reads_null_columns_through_columns() {
        with_null_column_routes(|cases| {
            for (label, y, run) in cases {
                let fill = |_: usize, mut m: faer::MatMut<'_, f64>| m.copy_from(*y);
                let p_free = run(None);
                assert!(
                    p_free < 1.0,
                    "{label}: premise, p = {p_free} without the map"
                );
                let p = run(Some(&fill));
                assert_eq!(
                    p.to_bits(),
                    1.0_f64.to_bits(),
                    "{label}: p = {p} with y in every null column"
                );
            }
        });
    }

    /// Under the Freedman–Lane map (null columns residualized on `[1, Z]`)
    /// the routes that test one design at one k give the same p-value, as
    /// they do without a map: the map reaches every route the same way.
    #[test]
    fn every_route_agrees_under_a_freedman_lane_null_map() {
        // Cases that test the same design, method and k.
        fn group(label: &str) -> &'static str {
            let k2 = label.contains("k = 2");
            let wide = label.contains("wide");
            if label.starts_with("PLS3") {
                "pls3"
            } else if label.starts_with("raw_perm") {
                if k2 {
                    "raw_perm k = 2"
                } else {
                    "raw_perm k = 1"
                }
            } else if k2 {
                if wide {
                    "split k = 2, wide"
                } else {
                    "split k = 2, tall"
                }
            } else if wide {
                "split k = 1, wide"
            } else {
                "split k = 1, tall"
            }
        }
        with_null_column_routes(|cases| {
            // (group, first label, its p-value, cases in the group)
            let mut groups: Vec<(&str, &str, f64, usize)> = Vec::new();
            for (label, y, run) in cases {
                let z = crate::test_support::unif(y.nrows(), 2, 47);
                let basis = crate::covariates::CovariateBasis::new(z.as_ref(), None).unwrap();
                let fl = |_: usize, m: faer::MatMut<'_, f64>| basis.residualize(m);
                let p = run(Some(&fl));
                let g = group(label);
                match groups.iter_mut().find(|e| e.0 == g) {
                    None => groups.push((g, label, p, 1)),
                    Some((_, l0, p0, count)) => {
                        assert_eq!(
                            p.to_bits(),
                            p0.to_bits(),
                            "{label}: p = {p}, {l0}: p = {p0}"
                        );
                        *count += 1;
                    }
                }
            }
            for (g, _, _, count) in &groups {
                assert!(*count >= 2, "{g}: one route only");
            }
        });
    }

    /// `r_from_moments` is `guarded_pearson` on the test half's scores
    /// `X̃_B·coef` and `y_B` within 1e-12, and gives its exact zero for a
    /// constant `y_B` and for zero scores.
    #[test]
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn r_from_moments_equals_direct_r() {
        use crate::linalg::{mat_vec, standardize_apply_rows};
        use crate::test_support::{unif, unif_col};
        let (n, p) = (90_usize, 7_usize);
        let x = unif(n, p, 3);
        let mean = Col::<f64>::from_fn(p, |j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64);
        let scale = Col::<f64>::from_fn(p, |j| 0.5 + j as f64);
        let te: Vec<usize> = (0..n).filter(|i| i % 3 != 1).collect();
        let n_b = te.len();
        let tm = TestMoments::new(x.as_ref(), &te, mean.as_ref(), scale.as_ref());
        let xb = standardize_apply_rows(x.as_ref(), &te, mean.as_ref(), scale.as_ref(), None);
        let both = |coef: &Col<f64>, y_b: &Col<f64>| {
            let xb_y = mat_vec(xb.transpose(), y_b.as_ref(), Par::Seq);
            let (sum, ss) = y_b_sums(n_b, |i| y_b[i]);
            let s = mat_vec(xb.as_ref(), coef.as_ref(), Par::Seq);
            (
                r_from_moments(coef.as_ref(), xb_y.as_ref(), sum, ss, &tm),
                guarded_pearson(n_b, |i| s[i], |i| y_b[i]),
            )
        };
        for seed in 0..20_u64 {
            let coef = unif_col(p, 100 + seed);
            let noise = unif_col(n_b, 200 + seed);
            // A mean of 3 on y_B: the centring term X̃_B'1·ȳ_B is in play.
            let y_b = Col::<f64>::from_fn(n_b, |i| 3.0 + noise[i]);
            let (m, d) = both(&coef, &y_b);
            assert!(
                (m - d).abs() <= 1e-12,
                "seed {seed}: moments {m} vs direct {d}"
            );
        }
        let coef = unif_col(p, 7);
        let constant = Col::<f64>::from_fn(n_b, |_| 0.1);
        assert_eq!(both(&coef, &constant), (0.0, 0.0), "constant y_B");
        let zero = Col::<f64>::zeros(p);
        let y_b = unif_col(n_b, 8);
        assert_eq!(both(&zero, &y_b), (0.0, 0.0), "zero scores");
    }

    /// `split_exact`'s z̄ for every column on tall data, on the draws of
    /// seed 9 with `J = TALL_SPLITS` and `TALL_PERM` null columns: on the
    /// no-refit route at k = 1 (`route` `None`), else on the refit route
    /// `route`.
    fn tall_zbars(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k: usize,
        route: Option<ReplicateRoute>,
    ) -> Vec<f64> {
        tall_zbars_on(x, y, k, route, false)
    }

    /// [`tall_zbars`], with the split design handed to the route when
    /// `cells` (the production calls hand it over), so the moment arms sum
    /// their training rows from its cells.
    #[allow(clippy::many_single_char_names)]
    fn tall_zbars_on(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k: usize,
        route: Option<ReplicateRoute>,
        cells: bool,
    ) -> Vec<f64> {
        let (_, mut rng) = crate::rng::resolve_seed(Some(9)).unwrap();
        let scale = crate::linalg::held_out_scale(x, None, false);
        let design = draw_design(x.nrows(), k, TALL_SPLITS, &mut rng).unwrap();
        let seeds = crate::rng::child_seeds(&mut rng, TALL_PERM);
        let cols = crate::resample::Columns {
            y: y.as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        let opts = ConfirmatoryTestOpts::default();
        let d = cells.then_some(&design);
        let splits = &design.splits;
        match route {
            None => split_perm_nr_zbars(x, &cols, splits, 1, None, scale.as_ref(), &opts, None, d)
                .unwrap(),
            Some(r) => {
                let z = split_zbars_columns(
                    r,
                    x,
                    splits,
                    None,
                    scale.as_ref(),
                    &[k],
                    None,
                    cols.len(),
                    &|c| cols.column(c),
                    None,
                    d,
                );
                first_row(&z)
            }
        }
    }

    /// Per split, per column: `(r, component count)`.
    type TallRows = Vec<Vec<(f64, usize)>>;

    /// Both routes of [`tall_zbars_on`]'s draws, per (split, column), as
    /// `(r, component count)`: the cell route (the cell arm, the values it
    /// leaves on the test-row arm, as the route runs them) and the per-split
    /// route (the per-split moment arm, the values it leaves on the test-row
    /// arm). The count is each route's own: at k = 1 whether its final `r`
    /// is nonzero (route A gives exactly 0 only when it drops the component
    /// or on degenerate halves, which both routes treat alike), at k ≥ 2
    /// `k_used` of the route's Gram fit (`S_A` from the cells, or `C` of the
    /// prepared half; a value the cell arm leaves takes the per-split count,
    /// since its test-row arm fits from the prepared half). Also how many
    /// values the cell arm decides itself.
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    fn tall_pairs(x: MatRef<'_, f64>, y: ColRef<'_, f64>, k: usize) -> (TallRows, TallRows, usize) {
        let (_, mut rng) = crate::rng::resolve_seed(Some(9)).unwrap();
        let scale = crate::linalg::held_out_scale(x, None, false);
        let scale = scale.as_ref();
        let d = draw_design(x.nrows(), k, TALL_SPLITS, &mut rng).unwrap();
        let seeds = crate::rng::child_seeds(&mut rng, TALL_PERM);
        let cols = crate::resample::Columns {
            y: y.as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        let n_cols = cols.len();
        let sizes = (d.splits[0].tr.len(), d.splits[0].te.len());
        let y_mat = nr_outcome_matrix(&cols);
        let column = |c: usize| {
            if k == 1 {
                y_mat.col(c).to_owned()
            } else {
                cols.column(c)
            }
        };
        let mut frame = CellFrame::new(x, scale, &d, k != 1);
        let levels = d.cells().trailing_zeros() as usize;
        let grams = if k == 1 {
            Vec::new()
        } else {
            frame.take_grams(sizes.0)
        };
        let fill = |c: usize, out: faer::MatMut<'_, f64>| out.col_mut(0).copy_from(column(c));
        let cell = cell_rows(&frame, &d, n_cols, 1, &fill, &|cs, j, c| {
            if k == 1 {
                (nr_cell_r(&frame, cs, j, c, sizes.0, sizes.1, levels), None)
            } else {
                let r = gram_p_cell_r(&frame, &grams[j], cs, j, c, sizes, &[k], None)[0];
                let band = crate::dual_route::RESOLVE_BAND;
                let used = cell_column(&frame, cs, j, c, sizes, band)
                    .map(|cc| gram_p_cell_fit(&grams[j], &cc, k, None).k_used);
                (r, used)
            }
        });
        let decided = cell.iter().flatten().filter(|(r, _)| r.is_some()).count();
        let am = AllRowMoments::new(x, scale, n_cols, &column);
        let (mut cells, mut per_split) = (Vec::new(), Vec::new());
        for (j, sp) in d.splits.iter().enumerate() {
            let prep = prepare_split(x, sp, None, scale);
            // Per column: the per-split route's r, the test-row arm's r, and
            // the per-split route's count.
            let (route, rows, own): (Vec<f64>, Vec<f64>, Vec<usize>) = if k == 1 {
                let rows = nr_split_r(x, y_mat.as_ref(), sp, None, scale, false);
                let moments = nr_split_r_moments(x, y_mat.as_ref(), sp, scale, &am);
                let route: Vec<f64> = (0..n_cols).map(|c| moments[c].unwrap_or(rows[c])).collect();
                let own = route.iter().map(|&r| usize::from(r != 0.0)).collect();
                (route, rows, own)
            } else {
                let gram = crate::gram_p::GramPBlock::new(prep.xs_tr.as_ref(), Par::Seq);
                let tm = TestMoments::new(x, &sp.te, am.mean.as_ref(), scale);
                let (mut route, mut rows, mut own) = (Vec::new(), Vec::new(), Vec::new());
                for c in 0..n_cols {
                    let yc = column(c);
                    let y_of = |i: usize| yc[i];
                    let r_rows = split_column_r_gram_p(&gram, &prep, sp, &y_of, &[k], None)[0];
                    let m = split_column_r_gram_p_moments(
                        &gram,
                        &tm,
                        &am,
                        c,
                        &prep,
                        sp,
                        &y_of,
                        &[k],
                        None,
                    )[0];
                    route.push(m.unwrap_or(r_rows));
                    rows.push(r_rows);
                    let ys = split_train_target(&prep, sp, &y_of);
                    let s1 = crate::gram_p::first_product(
                        prep.xs_tr.as_ref(),
                        ys.as_ref(),
                        SPLIT_FIT_PAR,
                        k,
                    );
                    own.push(gram.fit(s1.as_ref(), ys.norm_l2(), k, None).k_used);
                }
                (route, rows, own)
            };
            per_split.push(
                route
                    .into_iter()
                    .zip(own.iter().copied())
                    .collect::<Vec<_>>(),
            );
            cells.push(
                (0..n_cols)
                    .map(|c| {
                        let r = cell[j][c].0.unwrap_or(rows[c]);
                        let count = if k == 1 {
                            usize::from(r != 0.0)
                        } else {
                            cell[j][c].1.unwrap_or(own[c])
                        };
                        (r, count)
                    })
                    .collect::<Vec<_>>(),
            );
        }
        (cells, per_split, decided)
    }

    /// With the split design's cells, the tall moment arms sum every split's
    /// training rows from them (`crate::cell_sums`), at k = 1 on route A and
    /// at k = 2, 3 on the Gram-p arm. Per (split, column), the cell route
    /// and the per-split route ([`tall_pairs`]) agree within 1e-12 in `r`,
    /// except where their own component counts differ: the cell sums round
    /// differently from a split's own pass, so a training product within
    /// rounding of the truncation floor can keep the component on one route
    /// and drop it on the other. Those flips are counted and listed, and at
    /// most 1% of the values may flip: a flip needs a product at the floor,
    /// which ordinary data reaches rarely, while a wrong product or count from
    /// the cells would flip far more. Every column's z̄ from the production
    /// calls is within 1e-12 of the per-split call's where no split of the
    /// column flips. Premises: the production cell call runs the cells once,
    /// the per-split call never, and the cell arm decides most values itself.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn tall_cells_match_the_per_split_route() {
        let (n, p) = (1000_usize, 20_usize);
        let (x, y) = synth(n, p, 3, 0.5, 73);
        let (x, y) = (x.as_ref(), y.as_ref());
        for k in 1..=3 {
            let route = (k > 1).then_some(ReplicateRoute::GramP);
            let before = crate::cell_sums::cell_grams_built();
            let cells_z = tall_zbars_on(x, y, k, route, true);
            assert_eq!(
                crate::cell_sums::cell_grams_built(),
                before + 1,
                "k={k}: premise, the cells ran"
            );
            let per_split_z = tall_zbars(x, y, k, route);
            assert_eq!(crate::cell_sums::cell_grams_built(), before + 1);
            let (cell, per_split, decided) = tall_pairs(x, y, k);
            let all = TALL_SPLITS * (TALL_PERM + 1);
            assert!(
                decided * 10 >= all * 9,
                "k={k}: premise, the cell arm decides {decided} of {all}"
            );
            let (mut flips, mut worst) = (Vec::new(), 0.0_f64);
            for (j, (a, b)) in cell.iter().zip(&per_split).enumerate() {
                for (c, (&(ra, ka), &(rb, kb))) in a.iter().zip(b).enumerate() {
                    if ka != kb {
                        flips.push((j, c, ka, kb, ra, rb));
                        continue;
                    }
                    worst = worst.max((ra - rb).abs());
                    assert!(
                        (ra - rb).abs() <= 1e-12,
                        "k={k}: split {j} col {c}: cells r {ra} vs per split {rb} ({ka} components)"
                    );
                }
            }
            assert!(
                flips.len() * 100 <= all,
                "k={k}: {} of {all} values flip at the floor: {flips:?}",
                flips.len()
            );
            for (col, (u, v)) in cells_z.iter().zip(&per_split_z).enumerate() {
                if flips.iter().any(|f| f.1 == col) {
                    continue;
                }
                let tol = 1e-12 * u.abs().max(v.abs()).max(1.0);
                assert!((u - v).abs() <= tol, "k={k}: col {col}: z̄ {u} vs {v}");
            }
            eprintln!(
                "tall_cells_match_the_per_split_route k={k}: cell arm decides {decided} of {all}; \
                 {} values flip at the floor (split, col, counts, r): {flips:?}; \
                 worst |Δr| elsewhere {worst:.1e}",
                flips.len()
            );
        }
    }
    const TALL_SPLITS: usize = 32;
    const TALL_PERM: usize = 99;

    /// [`split_zbars_columns`] over `ks` on `TALL_SPLITS` splits drawn for
    /// `k_max` and `TALL_PERM` null columns, with the split design handed
    /// over when `cells`.
    fn per_k_zbars(
        route: ReplicateRoute,
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k_max: usize,
        ks: &[usize],
        cells: bool,
    ) -> Mat<f64> {
        let (_, mut rng) = crate::rng::resolve_seed(Some(9)).unwrap();
        let scale = crate::linalg::held_out_scale(x, None, false);
        let design = draw_design(x.nrows(), k_max, TALL_SPLITS, &mut rng).unwrap();
        let seeds = crate::rng::child_seeds(&mut rng, TALL_PERM);
        let cols = crate::resample::Columns {
            y: y.as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        split_zbars_columns(
            route,
            x,
            &design.splits,
            None,
            scale.as_ref(),
            ks,
            None,
            cols.len(),
            &|c| cols.column(c),
            None,
            cells.then_some(&design),
        )
    }

    /// One fit at `ks.last()` gives every k's z̄: row i of the per-k table
    /// equals the one-element call at `ks[i]` on the same splits, on each
    /// route and on the cell path.
    #[test]
    fn split_zbars_per_k_rows_match_single_k() {
        let ks = [1, 2, 3, 5];
        let check = |label: &str, route, x: MatRef<'_, f64>, y: ColRef<'_, f64>, cells: bool| {
            let all = per_k_zbars(route, x, y, 5, &ks, cells);
            assert_eq!((all.nrows(), all.ncols()), (ks.len(), TALL_PERM + 1));
            for (i, &k) in ks.iter().enumerate() {
                let one = per_k_zbars(route, x, y, 5, &[k], cells);
                for c in 0..all.ncols() {
                    let (a, b) = (all[(i, c)], one[(0, c)]);
                    assert!(
                        (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0),
                        "{label}: k={k} col {c}: {a} vs {b}"
                    );
                }
            }
        };
        for (n, p, seed) in [(60, 20, 31), (40, 200, 32)] {
            let (x, y) = synth(n, p, 3, 0.5, seed);
            for route in [
                ReplicateRoute::Primal,
                ReplicateRoute::Nspace,
                ReplicateRoute::GramP,
            ] {
                let label = format!("n={n} p={p} {route:?}");
                check(&label, route, x.as_ref(), y.as_ref(), false);
            }
        }
        let (x, y) = synth(1000, 20, 3, 0.5, 73);
        let before = crate::cell_sums::cell_grams_built();
        check("cells", ReplicateRoute::GramP, x.as_ref(), y.as_ref(), true);
        assert_eq!(
            crate::cell_sums::cell_grams_built(),
            before + 1 + ks.len(),
            "premise: every call ran the cells"
        );
    }

    /// Past the rank of X the fit at `ks.last()` stops at `k_used = 2`, and
    /// every k past it reads the coefficient at `k_used`: rows 2 and 3 equal
    /// row 1, and nothing is NaN. The n-space route is left out: it resolves
    /// `‖X_a'y_a‖` only to about `√(p·ε)·‖X̃‖_F·‖z‖`, so it can keep a
    /// component past the rank (`dual_route::pls1_nspace_kernel`).
    #[test]
    fn split_zbars_k_past_rank_repeats_k_used() {
        let (base, y) = synth(30, 2, 2, 0.5, 41);
        let x = Mat::<f64>::from_fn(30, 6, |i, j| base[(i, j % 2)]);
        for route in [ReplicateRoute::Primal, ReplicateRoute::GramP] {
            let z = per_k_zbars(route, x.as_ref(), y.as_ref(), 4, &[1, 2, 3, 4], false);
            for c in 0..z.ncols() {
                assert!((0..4).all(|i| z[(i, c)].is_finite()), "{route:?} col {c}");
                for i in 2..4 {
                    assert!(
                        (z[(i, c)] - z[(1, c)]).abs() <= 1e-12,
                        "{route:?} col {c}: row {i} {} vs row 1 {}",
                        z[(i, c)],
                        z[(1, c)]
                    );
                }
            }
        }
    }

    /// One fit at `ks.last()` per (fold, column) gives every k's pooled CV
    /// R²: row i of the per-k table equals the one-element call at `ks[i]`
    /// on the same folds, on each fold route, dense and weighted (the
    /// n-space route is dense only).
    #[test]
    fn pooled_cv_r2_per_k_rows_match_single_k() {
        let ks = [1, 2, 4];
        for (n, p, seed) in [(60, 20, 51), (40, 200, 52)] {
            let (x, y) = synth(n, p, 3, 0.5, seed);
            let wn = normalize_weights(crate::test_support::test_weights(n).as_ref()).unwrap();
            let (_, mut rng) = crate::rng::resolve_seed(Some(9)).unwrap();
            let (folds, seeds) = raw_perm_draws(n, 5, TALL_PERM, &mut rng);
            let cols = crate::resample::Columns {
                y: y.as_mat(),
                seeds: &seeds,
                null_map: None,
            };
            for (route, w) in [
                (ReplicateRoute::Primal, None),
                (ReplicateRoute::Primal, Some(wn.as_ref())),
                (ReplicateRoute::GramP, None),
                (ReplicateRoute::GramP, Some(wn.as_ref())),
                (ReplicateRoute::Nspace, None),
            ] {
                let scale = crate::linalg::held_out_scale(x.as_ref(), w, false);
                let run = |ks: &[usize]| {
                    pooled_cv_r2_columns(
                        route,
                        x.as_ref(),
                        &folds,
                        w,
                        scale.as_ref(),
                        ks,
                        None,
                        cols.len(),
                        &|c| cols.column(c),
                    )
                    .unwrap()
                };
                let all = run(&ks);
                assert_eq!((all.nrows(), all.ncols()), (ks.len(), TALL_PERM + 1));
                for (i, &k) in ks.iter().enumerate() {
                    let one = run(&[k]);
                    for c in 0..all.ncols() {
                        let (a, b) = (all[(i, c)], one[(0, c)]);
                        assert!(
                            (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0),
                            "n={n} {route:?} w={}: k={k} col {c}: {a} vs {b}",
                            w.is_some()
                        );
                    }
                }
            }
        }
    }

    /// `pls1_confirmatory_test` on `(x, y)` with the set `k` and `args`,
    /// seed 7, Raw form unless `gram` is set.
    #[allow(clippy::too_many_arguments)]
    fn k_set_call(
        x: &Mat<f64>,
        y: &Col<f64>,
        k: Option<&[usize]>,
        args: ConfirmatoryArgs,
        weights: Option<ColRef<'_, f64>>,
        covariates: Option<MatRef<'_, f64>>,
        gram: bool,
        opts: ConfirmatoryTestOpts,
    ) -> PlsKitResult<ConfirmatoryTestOutput> {
        let opts = ConfirmatoryTestOpts {
            args,
            seed: Some(7),
            ..opts
        };
        if gram {
            let g = crate::gram_matrix(x.as_ref(), false).unwrap();
            return pls1_confirmatory_test(
                ConfirmatoryTestInput::Gram {
                    gram: &g,
                    y: y.as_ref(),
                    k,
                    covariates,
                    null_y: None,
                },
                opts,
            );
        }
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k,
                weights,
                covariates,
                null_y: None,
            },
            opts,
        )
    }

    const K_SET_EXACT: ConfirmatoryArgs = ConfirmatoryArgs::SplitExact {
        n_perm: 49,
        n_splits: 8,
    };
    const K_SET_RAW: ConfirmatoryArgs = ConfirmatoryArgs::RawPerm {
        n_perm: 49,
        n_folds: 5,
    };

    /// Each k of a set gets the statistic and p of a single call at that k:
    /// the design and the null columns are the same (n = 60 gives the same
    /// halves for every k ≤ 28), only the route can differ.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn k_set_per_k_matches_single_calls() {
        let (x, y) = synth(60, 20, 3, 0.3, 7);
        let u = crate::test_support::unif_col(60, 8);
        let w = Col::<f64>::from_fn(60, |i| 1.5 + u[i]);
        let z = crate::test_support::unif(60, 2, 9);
        for args in [K_SET_EXACT, K_SET_RAW] {
            for (what, wt, cov, gram) in [
                ("dense", None, None, false),
                ("weighted", Some(w.as_ref()), None, false),
                ("covariates", None, Some(z.as_ref()), false),
                ("gram", None, None, true),
            ] {
                let run = |k: &[usize]| {
                    k_set_call(
                        &x,
                        &y,
                        Some(k),
                        args,
                        wt,
                        cov,
                        gram,
                        ConfirmatoryTestOpts::default(),
                    )
                    .unwrap()
                };
                let all = run(&[1, 2, 3]);
                assert_eq!(all.k, vec![1, 2, 3]);
                assert!(all.statistic.is_none());
                for k in [1, 2, 3] {
                    let one = run(&[k]);
                    assert_eq!(one.k, vec![k]);
                    let (a, b) = (all.statistics_by_k[&k], one.statistic.unwrap());
                    assert!(
                        (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0),
                        "{:?} {what} k={k}: {a} vs {b}",
                        args.method()
                    );
                    assert_eq!(
                        all.pvalues_by_k[&k],
                        one.pvalue,
                        "{:?} {what} k={k}",
                        args.method()
                    );
                    assert_eq!(one.pvalues_by_k[&k], one.pvalue);
                    assert_eq!(one.statistics_by_k[&k], b);
                }
            }
        }
    }

    /// Steps 4 and 5 of min-p on a hand table with a tie in row 0 and row 1
    /// dominant: `p_0 = [2, 2, 5, 4, 3]/5`, `p_1 = [1, 5, 4, 3, 2]/5`, so
    /// `m = [1, 2, 4, 3, 2]/5` and only c = 0 reaches `m(0) = 1/5`.
    #[test]
    fn min_p_combine_reference() {
        let rows = [[0.5, 0.5, 0.1, 0.2, 0.3], [0.9, 0.1, 0.2, 0.3, 0.4]];
        let t = Mat::<f64>::from_fn(2, 5, |i, c| rows[i][c]);
        assert_eq!(min_p_combine(t.as_ref(), &[0.4, 0.2]), 0.2);
        // A failed null (+∞, c = 2 in row 0) ranks first in its row, so
        // m(2) = 1/5 reaches m(0) and the p rises to 2/5.
        let t = Mat::<f64>::from_fn(2, 5, |i, c| {
            if (i, c) == (0, 2) {
                f64::INFINITY
            } else {
                rows[i][c]
            }
        });
        assert_eq!(min_p_combine(t.as_ref(), &[0.6, 0.2]), 0.4);
    }

    /// The default set is `1..=8` cut by the design; it is `{1}` for the
    /// methods and options that refuse a set, and where the cut is 1 the
    /// default call is today's k = 1 call, `auto` included.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn default_k_set_cut() {
        let ks = |n: usize, p: usize, args, w: Option<&Col<f64>>, z: Option<&Mat<f64>>| {
            let (x, y) = synth(n, p, 2, 0.5, 11);
            let opts = ConfirmatoryTestOpts::default();
            k_set_call(
                &x,
                &y,
                None,
                args,
                w.map(Col::as_ref),
                z.map(Mat::as_ref),
                false,
                opts,
            )
            .unwrap()
        };
        assert_eq!(ks(12, 50, K_SET_EXACT, None, None).k, vec![1, 2, 3, 4]);
        assert_eq!(ks(60, 3, K_SET_EXACT, None, None).k, vec![1, 2, 3]);
        assert_eq!(
            ks(60, 50, K_SET_EXACT, None, None).k,
            (1..=8).collect::<Vec<_>>()
        );
        assert_eq!(
            ks(60, 50, K_SET_RAW, None, None).k,
            (1..=8).collect::<Vec<_>>()
        );
        // n = 7: ⌊n/2⌋ − 2 = 1.
        let (x, y) = synth(7, 5, 2, 0.5, 11);
        let auto = ConfirmatoryArgs::Auto {
            n_perm: 49,
            n_splits: 8,
        };
        for args in [K_SET_EXACT, auto] {
            let d = ConfirmatoryTestOpts::default();
            let def = k_set_call(&x, &y, None, args, None, None, false, d).unwrap();
            let one = k_set_call(&x, &y, Some(&[1]), args, None, None, false, d).unwrap();
            assert_eq!(def.k, vec![1]);
            assert_eq!(def.pvalue.to_bits(), one.pvalue.to_bits());
            assert_eq!(
                def.statistic.unwrap().to_bits(),
                one.statistic.unwrap().to_bits()
            );
            assert_eq!(def.test_method, one.test_method);
            assert_eq!(
                def.stable_rank.map(f64::to_bits),
                one.stable_rank.map(f64::to_bits)
            );
        }
        // Weights of Kish n_eff in [4, 5): k ≤ ⌊n_eff⌋ − 1 = 3.
        let w = Col::<f64>::from_fn(60, |i| if i < 4 { 1.0 } else { 0.005 });
        let r = ks(60, 20, K_SET_EXACT, Some(&w), None);
        assert!((4.0..5.0).contains(&r.n_eff), "n_eff = {}", r.n_eff);
        assert_eq!(r.k, vec![1, 2, 3]);
        // rank([1, Z]) = 55 at n = 60: 5 residual df, k ≤ 4.
        let z = crate::test_support::unif(60, 54, 12);
        assert_eq!(ks(60, 20, K_SET_EXACT, None, Some(&z)).k, vec![1, 2, 3, 4]);
        // rank([1, Z]) = 45 over the 50 rows of positive weight.
        let z = crate::test_support::unif(60, 44, 13);
        let w = Col::<f64>::from_fn(60, |i| if i < 10 { 0.0 } else { 1.0 });
        assert_eq!(
            ks(60, 20, K_SET_EXACT, Some(&w), Some(&z)).k,
            vec![1, 2, 3, 4]
        );
        // The methods and options that refuse a set default to {1}.
        let (x, y) = synth(60, 20, 2, 0.5, 11);
        let one = |args, opts| {
            k_set_call(&x, &y, None, args, None, None, false, opts)
                .unwrap()
                .k
        };
        let d = ConfirmatoryTestOpts::default();
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: 8,
            force: true,
        };
        for args in [nb, ConfirmatoryArgs::Score, ConfirmatoryArgs::E] {
            assert_eq!(one(args, d), vec![1], "{:?}", args.method());
        }
        let ci = Some(CIOpts {
            n_boot: 100,
            ..CIOpts::default()
        });
        assert_eq!(one(K_SET_EXACT, ConfirmatoryTestOpts { ci, ..d }), vec![1]);
        let keep = Some(5);
        assert_eq!(
            one(K_SET_EXACT, ConfirmatoryTestOpts { keep, ..d }),
            vec![1]
        );
    }

    /// An explicit set is sorted, never fixed: a duplicate, an empty set, a
    /// set with a method or option that refuses one, or a `k_max` past a
    /// single call's limit is an error.
    #[test]
    fn k_set_validation() {
        let (x, y) = synth(60, 20, 2, 0.5, 11);
        let d = ConfirmatoryTestOpts::default();
        let call = |k: &[usize], args, opts, gram| {
            k_set_call(&x, &y, Some(k), args, None, None, gram, opts)
        };
        let invalid = |r: PlsKitResult<ConfirmatoryTestOutput>, what: &str| {
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "{what}: {:?}",
                r.map(|o| o.k)
            );
        };
        assert_eq!(
            call(&[3, 1, 2], K_SET_EXACT, d, false).unwrap().k,
            vec![1, 2, 3]
        );
        invalid(call(&[1, 1, 2], K_SET_EXACT, d, false), "duplicate");
        invalid(call(&[], K_SET_EXACT, d, false), "empty");
        invalid(call(&[0, 1], K_SET_EXACT, d, false), "k = 0");
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: 8,
            force: true,
        };
        for args in [nb, ConfirmatoryArgs::Score, ConfirmatoryArgs::E] {
            invalid(call(&[1, 2], args, d, false), args.method().as_str());
        }
        let ci = Some(CIOpts {
            n_boot: 100,
            ..CIOpts::default()
        });
        invalid(
            call(
                &[1, 2],
                K_SET_EXACT,
                ConfirmatoryTestOpts { ci, ..d },
                false,
            ),
            "ci",
        );
        let keep = Some(5);
        invalid(
            call(
                &[1, 2],
                K_SET_EXACT,
                ConfirmatoryTestOpts { keep, ..d },
                false,
            ),
            "keep",
        );
        // Past the Gram form's K_DUAL_MAX: a single call's error at k = 11.
        let set = call(&[1, 11], K_SET_EXACT, d, true).unwrap_err();
        let single = call(&[11], K_SET_EXACT, d, true).unwrap_err();
        assert_eq!(format!("{set:?}"), format!("{single:?}"));
        assert!(matches!(set, PlsKitError::InvalidArgument(_)), "{set:?}");
        assert!(matches!(
            call(&[1, 30], K_SET_EXACT, d, false),
            Err(PlsKitError::KExceedsMax { k: 30, k_max: 20 })
        ));
    }

    /// An explicit set past the rank of X (rank 2, columns repeated): on
    /// the `Primal` and `GramP` routes every k past the rank repeats the
    /// statistic at the rank; on the n-space route, whose kernel does not
    /// stop at the rank, every k is finite.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn k_set_past_rank_repeats_the_rank() {
        let d = ConfirmatoryTestOpts::default();
        let ks: &[usize] = &[1, 2, 3, 4];
        let (base, y) = synth(30, 2, 2, 0.5, 41);
        let x = Mat::<f64>::from_fn(30, 6, |i, j| base[(i, j % 2)]);
        assert_ne!(
            split_exact_refit_route(30, 6, 49, 8, 4, None, false, false),
            ReplicateRoute::Nspace
        );
        assert_ne!(
            raw_perm_route(30, 5, 6, 49, 4, None, false),
            ReplicateRoute::Nspace
        );
        for args in [K_SET_EXACT, K_SET_RAW] {
            let r = k_set_call(&x, &y, Some(ks), args, None, None, false, d).unwrap();
            let at_rank = r.statistics_by_k[&2];
            for k in [3, 4] {
                let v = r.statistics_by_k[&k];
                assert!(
                    (v - at_rank).abs() <= 1e-12 * at_rank.abs().max(1.0),
                    "{:?} k={k}: {v} vs {at_rank}",
                    args.method()
                );
            }
            assert!(r.pvalue.is_finite());
        }
        let (base, y) = synth(40, 2, 2, 0.5, 42);
        let x = Mat::<f64>::from_fn(40, 200, |i, j| base[(i, j % 2)]);
        assert_eq!(
            split_exact_refit_route(40, 200, 49, 8, 4, None, false, false),
            ReplicateRoute::Nspace
        );
        assert_eq!(
            raw_perm_route(40, 5, 200, 49, 4, None, false),
            ReplicateRoute::Nspace
        );
        for args in [K_SET_EXACT, K_SET_RAW] {
            let r = k_set_call(&x, &y, Some(ks), args, None, None, false, d).unwrap();
            assert!(
                r.statistics_by_k.values().all(|v| v.is_finite()) && r.pvalue.is_finite(),
                "{:?}: {:?}",
                args.method(),
                r.statistics_by_k
            );
        }
    }

    /// `auto` with a set of more than one k runs `split_exact`, on a design
    /// where `auto` at k = 1 picks `split_nb`.
    #[test]
    fn auto_k_set_takes_split_exact() {
        let (x, y) = synth(1001, 20, 3, 1.0, 52);
        let auto = ConfirmatoryArgs::Auto {
            n_perm: 9,
            n_splits: 4,
        };
        let d = ConfirmatoryTestOpts::default();
        for gram in [false, true] {
            let def = k_set_call(&x, &y, None, auto, None, None, gram, d).unwrap();
            assert_eq!(def.test_method, "split_exact", "gram={gram}");
            assert_eq!(def.k, (1..=8).collect::<Vec<_>>());
            assert!(def.stable_rank.is_none());
            let one = k_set_call(&x, &y, Some(&[1]), auto, None, None, gram, d).unwrap();
            assert_eq!(one.test_method, "split_nb", "gram={gram}");
        }
    }

    /// The columns of split 0 of [`tall_zbars`]'s draws that the moment arm
    /// decides itself, out of `TALL_PERM + 1`: `nr_split_r_moments` at
    /// k = 1, `split_column_r_gram_p_moments` at k ≥ 2.
    fn tall_moment_decided(x: MatRef<'_, f64>, y: ColRef<'_, f64>, k: usize) -> usize {
        let (_, mut rng) = crate::rng::resolve_seed(Some(9)).unwrap();
        let scale = crate::linalg::held_out_scale(x, None, false);
        let splits = draw_splits(x.nrows(), k, TALL_SPLITS, &mut rng).unwrap();
        let seeds = crate::rng::child_seeds(&mut rng, TALL_PERM);
        let cols = crate::resample::Columns {
            y: y.as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        let sp = &splits[0];
        if k == 1 {
            let y_mat = nr_outcome_matrix(&cols);
            let am =
                AllRowMoments::new(x, scale.as_ref(), cols.len(), &|c| y_mat.col(c).to_owned());
            return nr_split_r_moments(x, y_mat.as_ref(), sp, scale.as_ref(), &am)
                .iter()
                .filter(|r| r.is_some())
                .count();
        }
        let am = AllRowMoments::new(x, scale.as_ref(), cols.len(), &|c| cols.column(c));
        let prep = prepare_split(x, sp, None, scale.as_ref());
        let gram = crate::gram_p::GramPBlock::new(prep.xs_tr.as_ref(), Par::Seq);
        let tm = TestMoments::new(x, &sp.te, am.mean.as_ref(), scale.as_ref());
        (0..cols.len())
            .filter(|&c| {
                let yc = cols.column(c);
                split_column_r_gram_p_moments(&gram, &tm, &am, c, &prep, sp, &|i| yc[i], &[k], None)
                    [0]
                .is_some()
            })
            .count()
    }

    /// `|u − v| ≤ 1e-12·max(|u|, |v|, 1)` for every column.
    fn assert_zbars_within_1e12(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len());
        for (col, (u, v)) in a.iter().zip(b).enumerate() {
            let tol = 1e-12 * u.abs().max(v.abs()).max(1.0);
            assert!((u - v).abs() <= tol, "{what}: col {col}: {u} vs {v}");
        }
    }

    /// On tall data `split_exact` scores the held-out halves from moments,
    /// at k = 1 on route A and at k = 2, 3 on the Gram-p arm, and every
    /// column's z̄ is within 1e-12 of the call with the Gram routes taken
    /// out of the choice (route A on the test rows at k = 1, the Primal refit
    /// at k ≥ 2). The premises: the moment arm is in the choice, and it
    /// decides most columns of a split itself.
    #[test]
    fn tall_split_exact_from_moments_matches_step2c() {
        let n_cols = TALL_PERM + 1;
        for (n, p) in [(1000_usize, 20_usize), (300, 8)] {
            let (x, y) = synth(n, p, 3, 0.5, 71);
            let (x, y) = (x.as_ref(), y.as_ref());
            for k in 1..=3 {
                let (n_train, n_test) = crate::resample::split_sizes(n, k);
                assert!(
                    test_moments_route(n, n_test, p, n_cols, TALL_SPLITS, false),
                    "{n}x{p} k={k}: premise, moments in the choice"
                );
                let decided = tall_moment_decided(x, y, k);
                assert!(
                    decided * 10 >= n_cols * 9,
                    "{n}x{p} k={k}: premise, the moment arm decides {decided} of {n_cols}"
                );
                let (moments, reference) = if k == 1 {
                    assert_eq!(
                        k1_route(n, n_train, n_test, p, n_cols, TALL_SPLITS, false),
                        K1Route::A
                    );
                    (
                        tall_zbars(x, y, 1, None),
                        with_gram_routes_disabled(|| tall_zbars(x, y, 1, None)),
                    )
                } else {
                    let reference = with_gram_routes_disabled(|| {
                        assert_eq!(
                            split_exact_refit_route(
                                n,
                                p,
                                TALL_PERM,
                                TALL_SPLITS,
                                k,
                                None,
                                false,
                                false
                            ),
                            ReplicateRoute::Primal
                        );
                        tall_zbars(x, y, k, Some(ReplicateRoute::Primal))
                    });
                    (tall_zbars(x, y, k, Some(ReplicateRoute::GramP)), reference)
                };
                assert_zbars_within_1e12(&moments, &reference, &format!("{n}x{p} k={k}"));
            }
        }
    }

    /// A memory budget too small for the moment buffers (1 KiB, below
    /// `p·n_cols·8`) takes the moment arm out of the choice: the call then
    /// runs today's arm, to the bit (route A on the test rows at k = 1, the
    /// Gram-p arm on the test rows at k = 2), and the moment call agrees with
    /// it within 1e-12.
    #[test]
    fn moment_route_respects_memory_budget() {
        let (n, p) = (1000_usize, 20_usize);
        let (x, y) = synth(n, p, 3, 0.5, 72);
        let (x, y) = (x.as_ref(), y.as_ref());
        let n_cols = TALL_PERM + 1;
        let tight = |f: &dyn Fn() -> Vec<f64>| crate::fit::with_budget(1024, f);
        for (k, route) in [(1, None), (2, Some(ReplicateRoute::GramP))] {
            let n_test = crate::resample::split_sizes(n, k).1;
            assert!(test_moments_route(n, n_test, p, n_cols, TALL_SPLITS, false));
            assert!(crate::fit::with_budget(1024, || !test_moments_route(
                n,
                n_test,
                p,
                n_cols,
                TALL_SPLITS,
                false
            )));
            let on_budget = tight(&|| tall_zbars(x, y, k, route));
            let today = with_gram_routes_disabled(|| tall_zbars(x, y, k, route));
            let bits = |z: &[f64]| z.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&on_budget),
                bits(&today),
                "k={k}: 1 KiB budget vs today's arm"
            );
            assert_zbars_within_1e12(&tall_zbars(x, y, k, route), &today, &format!("k={k}"));
        }
    }

    /// The moment arms subtract each outcome column's all-row mean before
    /// any sum, so on an outcome whose mean is far above its spread (mean/SD
    /// `1e4` and `1e6`) every column's z̄ stays within 1e-12 of the refit
    /// route on prepared halves (the Primal arm, which centres `y` before
    /// any product), at k = 1 (route A) and k = 2 (Gram-p), with the moment
    /// arm still deciding most columns itself, and within 1e-12 of the call
    /// with the Gram routes taken out of the choice (route A on the test rows,
    /// which maps the standardized training targets, and the Primal refit).
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn tall_moment_arms_hold_on_an_outcome_with_a_large_mean() {
        let (n, p) = (1000_usize, 20_usize);
        let (x, y) = synth(n, p, 3, 0.5, 73);
        let mean = plain_mean(n, |i| y[i]);
        let sd = ((0..n).map(|i| (y[i] - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        for ratio in [1e4, 1e6] {
            let ys = Col::<f64>::from_fn(n, |i| y[i] + ratio * sd);
            let (x, ys) = (x.as_ref(), ys.as_ref());
            for (k, route, reference) in [
                (1, None, None),
                (2, Some(ReplicateRoute::GramP), Some(ReplicateRoute::Primal)),
            ] {
                let what = format!("mean/SD {ratio:e} k={k}");
                let decided = tall_moment_decided(x, ys, k);
                assert!(
                    decided * 10 >= (TALL_PERM + 1) * 9,
                    "{what}: premise, the moment arm decides {decided}"
                );
                let moments = tall_zbars(x, ys, k, route);
                let refit = tall_zbars(x, ys, k, Some(ReplicateRoute::Primal));
                assert_zbars_within_1e12(&moments, &refit, &format!("{what}, refit"));
                let rel = (moments[0] - refit[0]).abs() / refit[0].abs();
                assert!(rel <= 1e-12, "{what}: observed z̄ rel {rel:e}");
                let today = with_gram_routes_disabled(|| tall_zbars(x, ys, k, reference));
                assert_zbars_within_1e12(&moments, &today, &what);
            }
        }
    }

    /// The conditioning clause of `moments_resolved`: a test half whose
    /// coefficient points into a nearly null direction of `C_B X̃_B` (two
    /// columns `δ` apart, `coef` along their difference over `δ`) has
    /// `κ = F_te·‖coef‖/t_c` near `6/δ`. Up to `κ = 1/SCORE_BAND` the gate
    /// keeps it and `r_from_moments` is `guarded_pearson` on the test rows
    /// within 1e-12 relative; at `κ` near `6e3` and `6.2e4` the gate sends it
    /// to the test rows. At every `κ` from `6e1` to `6e5` the discrepancy
    /// per unit of `κ` stays below `1e-16`, the figure the clause's doc
    /// rests on.
    #[test]
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn moment_gate_sends_ill_conditioned_scores_to_the_test_rows() {
        use crate::linalg::{centered_xt_vec, mat_vec, standardize_apply_rows};
        use crate::test_support::{unif, unif_col};
        let (n, p) = (1000_usize, 20_usize);
        let te: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let n_b = te.len();
        let scale = Col::<f64>::from_fn(p, |_| 1.0);
        for (seed, delta) in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6].into_iter().enumerate() {
            let seed = seed as u64;
            let w = unif_col(n, 300 + seed);
            let mut x = unif(n, p, 400 + seed);
            for i in 0..n {
                x[(i, 1)] = x[(i, 0)] + delta * w[i];
            }
            let mean = Col::<f64>::from_fn(p, |j| plain_mean(n, |i| x[(i, j)]));
            let x_sum = centered_xt_vec(x.as_ref(), mean.as_ref(), scale.as_ref(), |_| 1.0);
            let tm = TestMoments::new(x.as_ref(), &te, mean.as_ref(), scale.as_ref());
            let xb = standardize_apply_rows(x.as_ref(), &te, mean.as_ref(), scale.as_ref(), None);
            let noise = unif_col(p, 500 + seed);
            let coef = Col::<f64>::from_fn(p, |j| {
                let along = match j {
                    0 => -1.0 / delta,
                    1 => 1.0 / delta,
                    _ => 0.0,
                };
                along + 0.01 * noise[j]
            });
            let e = unif_col(n_b, 600 + seed);
            let y_b = Col::<f64>::from_fn(n_b, |i| 0.3 * w[te[i]] + e[i] + 3.0);
            let c = plain_mean(n_b, |i| y_b[i]);
            let xb_y = mat_vec(
                xb.transpose(),
                Col::<f64>::from_fn(n_b, |i| y_b[i] - c).as_ref(),
                Par::Seq,
            );
            let (sum, ss) = y_b_sums(n_b, |i| y_b[i] - c);
            let r_m = r_from_moments(coef.as_ref(), xb_y.as_ref(), sum, ss, &tm);
            let s = mat_vec(xb.as_ref(), coef.as_ref(), Par::Seq);
            let r_d = guarded_pearson(n_b, |i| s[i], |i| y_b[i]);
            let sn = tm.score_norms(coef.as_ref(), x_sum.as_ref(), n - n_b);
            let kappa = sn.f_coef / sn.t_c;
            let diff = (r_m - r_d).abs();
            let resolved = moments_resolved(&sn, sum, ss, n_b, NR_RESOLVE_BAND);
            assert!(
                diff <= 1e-16 * kappa,
                "delta {delta:e}: kappa {kappa:e}, |r_m - r_d| {diff:e}"
            );
            assert_eq!(
                resolved,
                kappa <= 1.0 / crate::dual_route::SCORE_BAND,
                "delta {delta:e}: kappa {kappa:e}"
            );
            if resolved {
                assert!(
                    diff <= 1e-12 * r_d.abs(),
                    "delta {delta:e}: kept at kappa {kappa:e} with rel {:e}",
                    diff / r_d.abs()
                );
            }
            if delta <= 1e-3 {
                assert!(!resolved, "delta {delta:e}: kappa {kappa:e} kept");
            }
        }
    }

    /// The moment arm's choice reads shapes and the budget, never the thread
    /// count: under a 4 MiB budget a tall Gram-p call picks the same arm and
    /// gives the same bits on pools of 1 and 4 threads.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn moment_route_choice_ignores_the_thread_count() {
        let (n, p, k) = (300_usize, 8_usize, 2_usize);
        let (x, y) = synth(n, p, 3, 0.5, 74);
        let n_test = crate::resample::split_sizes(n, k).1;
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                crate::fit::with_budget(4 << 20, || {
                    let arm = test_moments_route(n, n_test, p, TALL_PERM + 1, TALL_SPLITS, false);
                    let z = tall_zbars(x.as_ref(), y.as_ref(), k, Some(ReplicateRoute::GramP));
                    (arm, z.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                })
            })
        };
        let (one, four) = (run(1), run(4));
        assert!(one.0, "premise: the moment arm is in the choice at 4 MiB");
        assert_eq!(one, four, "1 thread vs 4 threads");
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
        for (p, weighted, route) in DEGENERATE_HALF_CASES {
            let w_norm = weighted.then_some(w.as_ref());
            let (x, y_orth) =
                design_with_orthogonal_train_half(n, p, &splits[0].tr, w_norm, 1.0, 17);
            let (_, y_plain) = synth(n, p, 3, 0.3, 23);
            let nr = |y: &Col<f64>| {
                on_route(route, || {
                    nr_zbars(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed)
                })
            };
            for (label, y) in [("orthogonal half", &y_orth), ("ordinary", &y_plain)] {
                let z_nr = nr(y);
                let (_, z_re, _) =
                    refit_zbars_on_nr_draws(x.as_ref(), y.as_ref(), n_perm, n_splits, w_norm, seed);
                let refit_run = |y: ColRef<'_, f64>| {
                    let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
                    run_split_perm(
                        ReplicateRoute::Primal,
                        x.as_ref(),
                        y,
                        &[1],
                        n_perm,
                        n_splits,
                        w_norm,
                        crate::linalg::held_out_scale(x.as_ref(), w_norm, false).as_ref(),
                        &ConfirmatoryTestOpts::default(),
                        None,
                        None,
                        &mut rng,
                    )
                    .unwrap()
                    .per_k
                    .remove(0)
                };
                let base_run = refit_run(y.as_ref());
                for factor in [1e8, 1e-8] {
                    let what = format!("p={p} weighted={weighted} {route:?} {label} y×{factor:e}");
                    let ys = Col::<f64>::from_fn(n, |i| y[i] * factor);
                    let z_nr_s = nr(&ys);
                    let no_refit = format!("{what}, no-refit");
                    if label == "orthogonal half" {
                        assert_zbars_close_but_orthogonal(&z_nr, &z_nr_s, route, &no_refit);
                    } else {
                        assert_zbars_close(&z_nr, &z_nr_s, &no_refit);
                    }
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
                ConfirmatoryTestInput::Raw {
                    x,
                    y,
                    k: Some(&[k]),
                    weights,
                    covariates: None,
                    null_y: None,
                },
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
                    base.statistic.unwrap().abs() > 0.05,
                    "{what}: test premise, statistic = {}",
                    base.statistic.unwrap()
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
                            got.statistic.unwrap().to_bits(),
                            base.statistic.unwrap().to_bits(),
                            "{what}: statistic {} vs {}",
                            got.statistic.unwrap(),
                            base.statistic.unwrap()
                        );
                        assert_eq!(got.pvalue.to_bits(), base.pvalue.to_bits(), "{what}: p");
                    } else {
                        let tol = 1e-10 * base.statistic.unwrap().abs();
                        assert!(
                            (got.statistic.unwrap() - base.statistic.unwrap()).abs() <= tol,
                            "{what}: statistic {} vs {}",
                            got.statistic.unwrap(),
                            base.statistic.unwrap()
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
                k: Some(&[1]),
                weights,
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits, force },
                seed: Some(4242),
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// On wide X the gate sums the Gram matrix over column blocks; its
    /// stable rank agrees with the one of the standardized copy, rows scaled
    /// by √w under weights.
    #[test]
    fn gate_stable_rank_matches_the_standardized_copy_on_wide_x() {
        // 30 rows give a block of 8,736 columns: p=20,000 is three blocks.
        assert_eq!(crate::linalg::stream_block_width(30), 8736);
        let x = crate::test_support::unif(30, 20_000, 17);
        let w = crate::test_support::test_weights(30);
        let wn = crate::linalg::normalize_weights(w.as_ref()).unwrap();
        for wopt in [None, Some(wn.as_ref())] {
            let got = resolve_split_nb(
                x.as_ref(),
                wopt,
                30.0,
                false,
                crate::linalg::held_out_scale(x.as_ref(), wopt, false).as_ref(),
            )
            .stable_rank;
            let (xs, _, _) = crate::linalg::standardize_weighted(x.as_ref(), wopt);
            let xs = Mat::<f64>::from_fn(xs.nrows(), xs.ncols(), |i, j| {
                xs[(i, j)] * wopt.map_or(1.0, |w| w[i].sqrt())
            });
            let want = crate::linalg::stable_rank(xs.as_ref());
            assert!(
                ((got - want) / want).abs() <= 1e-12,
                "weighted={}: {got} vs {want}",
                wopt.is_some()
            );
        }
    }

    /// On tall X the gate runs the standardized copy, to the bit.
    #[test]
    fn gate_stable_rank_on_tall_x_is_the_standardized_copy() {
        let x = crate::test_support::unif(200, 12, 19);
        let got = resolve_split_nb(
            x.as_ref(),
            None,
            200.0,
            false,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        )
        .stable_rank;
        let (xs, _, _) = crate::linalg::standardize_weighted(x.as_ref(), None);
        assert_eq!(
            got.to_bits(),
            crate::linalg::stable_rank(xs.as_ref()).to_bits()
        );
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
        let (x, y) = synth(40, 10, 0, 0.0, 8);
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
        let absent = split_nb_gate(x25.as_ref(), None, false).unwrap();
        assert!(!absent.fires, "n = 25 must clear the gate unweighted");
        let got = split_nb_gate(x25.as_ref(), Some(equal.as_ref()), false).unwrap();
        assert_eq!(
            (got.fires, got.n_eff.to_bits()),
            (false, 25.0_f64.to_bits())
        );
    }

    // ── the public gate query ────────────────────────────────────────────────
    //
    // These are the whole contract of `split_nb_gate`: it must answer exactly
    // what the embedded gate decides, and it must reject bad input with the
    // same errors the test entry points do rather than reaching the
    // eigendecomposition.

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
            let q = split_nb_gate(x.as_ref(), w, false).unwrap();
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
    /// NaN in X gives a NaN stable rank, which fires the gate, instead of
    /// this error.
    #[test]
    fn public_gate_rejects_non_finite_x() {
        let mut x = synth(60, 10, 0, 0.0, 7).0;
        x[(3, 2)] = f64::NAN;
        assert!(matches!(
            split_nb_gate(x.as_ref(), None, false),
            Err(PlsKitError::NonFiniteInput)
        ));
    }

    #[test]
    fn public_gate_rejects_bad_weights() {
        let x = synth(60, 10, 0, 0.0, 7).0;
        let neg = Col::<f64>::from_fn(60, |i| if i == 0 { -1.0 } else { 1.0 });
        assert!(matches!(
            split_nb_gate(x.as_ref(), Some(neg.as_ref()), false),
            Err(PlsKitError::InvalidWeights { reason: "negative" })
        ));
        let short = Col::<f64>::from_fn(59, |_| 1.0);
        assert!(matches!(
            split_nb_gate(x.as_ref(), Some(short.as_ref()), false),
            Err(PlsKitError::InvalidWeights {
                reason: "length_mismatch"
            })
        ));
    }

    /// The gate's stable rank on the gate tests' designs, wide and tall,
    /// stays within `1e-13` relative of the pinned values, and every
    /// decision is the pinned one.
    #[test]
    fn gate_stable_rank_moves_only_in_the_last_bits() {
        type Design<'a> = (&'a str, Mat<f64>, Option<&'a Col<f64>>, bool, f64);
        let wide = crate::test_support::unif(30, 20_000, 17);
        let w30 = crate::test_support::test_weights(30);
        let half = Col::<f64>::from_fn(40, |i| if i % 2 == 0 { 1.0 } else { 0.1 });
        let x40 = synth(40, 10, 0, 0.0, 8).0;
        let designs: [Design<'_>; 8] = [
            ("wide", wide.clone(), None, false, 27.252_282_280_372_626),
            (
                "wide weighted",
                wide,
                Some(&w30),
                true,
                14.052_325_983_749_48,
            ),
            (
                "tall",
                crate::test_support::unif(200, 12, 19),
                None,
                false,
                7.746_608_271_881_038,
            ),
            (
                "n below floor",
                synth(20, 10, 0, 0.0, 5).0,
                None,
                true,
                5.402_203_692_584_337_5,
            ),
            (
                "narrow",
                synth(60, 4, 0, 0.0, 11).0,
                None,
                true,
                3.232_709_008_884_746,
            ),
            (
                "one factor",
                synth_one_factor(40, 5, 6).0,
                None,
                true,
                1.000_095_148_846_522_2,
            ),
            (
                "flat",
                synth(60, 10, 0, 0.0, 7).0,
                None,
                false,
                5.600_602_259_082_254,
            ),
            (
                "n_eff weighted",
                x40,
                Some(&half),
                true,
                4.259_744_666_959_581,
            ),
        ];
        for (name, x, w, fires, sr) in &designs {
            let got = split_nb_gate(x.as_ref(), w.map(Col::as_ref), false).unwrap();
            assert_eq!(got.fires, *fires, "{name}");
            assert!(
                ((got.stable_rank - sr) / sr).abs() <= 1e-13,
                "{name}: {} vs {sr}",
                got.stable_rank
            );
        }
    }

    // ── `test_method = "auto"` ───────────────────────────────────────────────

    /// The sample-size clause at each route's floor, on unweighted X where
    /// `n_eff` is the row count. Ten iid columns clear the gate and the width
    /// clause, so the floor is the only reason for `split_exact`, and that
    /// decision is made without the stable-rank check. The no-refit floor
    /// keeps `n_eff` = 1000 on `split_exact`; the refit floor does not keep 250.
    #[test]
    fn auto_n_eff_floor_by_route() {
        // (n, no_refit, expected method, stable rank read)
        for (n, no_refit, expect, checked) in [
            (1000, true, ConfirmatoryMethod::SplitExact, false),
            (1001, true, ConfirmatoryMethod::SplitNb, true),
            (249, false, ConfirmatoryMethod::SplitExact, false),
            (250, false, ConfirmatoryMethod::SplitNb, true),
        ] {
            let x = synth(n, 10, 0, 0.0, 21).0;
            let (method, sr, _) = resolve_auto_on_x(
                x.as_ref(),
                None,
                n as f64,
                no_refit,
                false,
                false,
                false,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
            );
            assert_eq!(method, expect, "n={n} no_refit={no_refit}");
            assert_eq!(sr.is_some(), checked, "n={n} no_refit={no_refit}");
        }
    }

    /// The width clause at its ceiling. `n_eff` is passed apart from the
    /// row count, so a 10-row X with `n_eff` above the floor tests the
    /// clause without the gate's pass over a 1001-row, 100 100-column X:
    /// `p = AUTO_MAX_P_PER_N · n` goes on to the stable-rank check (iid
    /// columns clear it), one column more decides `split_exact` without it.
    /// Off the no-refit route (k ≥ 2, a set `keep`, PLS3) the ceiling does
    /// not apply.
    #[test]
    fn auto_p_per_n_ceiling() {
        let n = 10;
        let n_eff = AUTO_MIN_N_EFF_NO_REFIT + 1.0;
        let ceiling = AUTO_MAX_P_PER_N * n;
        for (p, no_refit, expect, checked) in [
            (ceiling, true, ConfirmatoryMethod::SplitNb, true),
            (ceiling + 1, true, ConfirmatoryMethod::SplitExact, false),
            (ceiling + 1, false, ConfirmatoryMethod::SplitNb, true),
        ] {
            let x = synth(n, p, 0, 0.0, 22).0;
            let (method, sr, _) = resolve_auto_on_x(
                x.as_ref(),
                None,
                n_eff,
                no_refit,
                false,
                false,
                false,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
            );
            assert_eq!(method, expect, "p={p} no_refit={no_refit}");
            assert_eq!(sr.is_some(), checked, "p={p} no_refit={no_refit}");
        }
    }

    /// The sample-size clause reads Kish `n_eff`, the width clause reads
    /// rows. 300 rows weighted 1 / 0.1 alternately give
    /// `n_eff` = 165² / 151.5 ≈ 180, so `split_exact`; the same X unweighted
    /// runs `split_nb`. Both calls are on the refit route, floor 250.
    #[test]
    fn auto_n_eff_clause_reads_weights() {
        let x = synth(300, 10, 0, 0.0, 23).0;
        let w = Col::<f64>::from_fn(300, |i| if i % 2 == 0 { 1.0 } else { 0.1 });
        let (w_norm, n_eff) =
            crate::fit::validate_and_normalize_weights(Some(w.as_ref()), 300, 0).unwrap();
        assert!(n_eff < AUTO_MIN_N_EFF_REFIT, "n_eff={n_eff}");
        let (method, sr, _) = resolve_auto_on_x(
            x.as_ref(),
            w_norm.as_ref().map(Col::as_ref),
            n_eff,
            false,
            false,
            false,
            false,
            crate::linalg::held_out_scale(x.as_ref(), w_norm.as_ref().map(Col::as_ref), false)
                .as_ref(),
        );
        assert_eq!((method, sr), (ConfirmatoryMethod::SplitExact, None));
        let (method, sr, _) = resolve_auto_on_x(
            x.as_ref(),
            None,
            300.0,
            false,
            false,
            false,
            false,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        );
        assert_eq!(method, ConfirmatoryMethod::SplitNb);
        assert!(sr.is_some());
    }

    /// On the no-refit route the floor reads Kish `n_eff`, not rows: 1200
    /// rows weighted 1 / 0.1 alternately give `n_eff` = 660² / 606 ≈ 719,
    /// below 1000, so `split_exact` without the stable-rank check.
    #[test]
    fn auto_no_refit_floor_reads_weights() {
        let x = synth(1200, 10, 0, 0.0, 23).0;
        let w = Col::<f64>::from_fn(1200, |i| if i % 2 == 0 { 1.0 } else { 0.1 });
        let (w_norm, n_eff) =
            crate::fit::validate_and_normalize_weights(Some(w.as_ref()), 1200, 0).unwrap();
        assert!(n_eff < AUTO_MIN_N_EFF_NO_REFIT, "n_eff={n_eff}");
        let (method, sr, _) = resolve_auto_on_x(
            x.as_ref(),
            w_norm.as_ref().map(Col::as_ref),
            n_eff,
            true,
            false,
            false,
            false,
            crate::linalg::held_out_scale(x.as_ref(), w_norm.as_ref().map(Col::as_ref), false)
                .as_ref(),
        );
        assert_eq!((method, sr), (ConfirmatoryMethod::SplitExact, None));
    }

    /// Above both size clauses the `split_nb` gate still decides: a
    /// one-factor X (stable rank near 1) runs `split_exact` and reports the
    /// rank, and a 4-column X runs `split_exact` on the column precheck
    /// without computing it.
    #[test]
    fn auto_defers_to_the_split_nb_gate() {
        let x = synth_one_factor(1200, 10, 6).0;
        let (method, sr, _) = resolve_auto_on_x(
            x.as_ref(),
            None,
            1200.0,
            true,
            false,
            false,
            false,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        );
        assert_eq!(method, ConfirmatoryMethod::SplitExact);
        let sr = sr.expect("the stable-rank check ran");
        assert!(sr < SPLIT_NB_GATE_MIN_STABLE_RANK, "stable_rank={sr}");

        let x = synth(300, 4, 0, 0.0, 11).0;
        let (method, sr, _) = resolve_auto_on_x(
            x.as_ref(),
            None,
            300.0,
            true,
            false,
            false,
            false,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        );
        assert_eq!((method, sr), (ConfirmatoryMethod::SplitExact, None));
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
                    k: Some(&[1]),
                    weights: None,
                    covariates: None,
                    null_y: None,
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
        let large = synth(1200, 10, 3, 0.3, 13);
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
        let (x, y) = synth(1200, 10, 3, 0.3, 13);
        assert_eq!(
            resolve_auto_on_x(
                x.as_ref(),
                None,
                1200.0,
                true,
                false,
                false,
                false,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref()
            )
            .0,
            ConfirmatoryMethod::SplitNb
        );
        for (n_perm, n_splits) in [(0, 6), (49, 1)] {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[1]),
                    weights: None,
                    covariates: None,
                    null_y: None,
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

    /// Each rule of `AUTO_RULES` on a design that meets it and no earlier
    /// rule, then designs that meet two rules, which the earlier one
    /// decides. Only the last rule reads the stable rank, so every design
    /// decided earlier passes one that panics when read. The PLS3 design is
    /// one the gate would send to `split_nb` (its PLS1 twin below does).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn auto_rules_fire_in_order() {
        use ConfirmatoryMethod::{SplitExact, SplitNb};
        let unread = || -> f64 { panic!("the stable rank was read") };
        let (flat, low) = (|| 10.0, || 1.0);
        let base = AutoInput {
            n: 2000,
            p: 10,
            n_eff: 2000.0,
            no_refit: true,
            pls3: false,
            covariates: false,
            multi_k: false,
            stable_rank: &unread,
        };
        for (input, want) in [
            (
                AutoInput {
                    covariates: true,
                    ..base
                },
                (SplitExact, None, "covariates"),
            ),
            (
                AutoInput {
                    multi_k: true,
                    p: 4,
                    ..base
                },
                (SplitExact, None, "k_set"),
            ),
            (AutoInput { p: 4, ..base }, (SplitExact, None, "narrow_x")),
            (
                AutoInput {
                    n_eff: AUTO_MIN_N_EFF_NO_REFIT,
                    ..base
                },
                (SplitExact, None, "small_n_eff"),
            ),
            (
                AutoInput {
                    n_eff: AUTO_MIN_N_EFF_REFIT - 1.0,
                    no_refit: false,
                    ..base
                },
                (SplitExact, None, "small_n_eff"),
            ),
            (
                AutoInput {
                    n: 10,
                    p: AUTO_MAX_P_PER_N * 10 + 1,
                    ..base
                },
                (SplitExact, None, "too_wide_for_gate"),
            ),
            (
                AutoInput {
                    no_refit: false,
                    pls3: true,
                    ..base
                },
                (SplitExact, None, "pls3_exact"),
            ),
            (
                AutoInput {
                    no_refit: false,
                    stable_rank: &flat,
                    ..base
                },
                (SplitNb, Some(10.0), "split_nb_gate"),
            ),
            (
                AutoInput {
                    stable_rank: &low,
                    ..base
                },
                (SplitExact, Some(1.0), "split_nb_gate"),
            ),
            (
                AutoInput {
                    p: 4,
                    n_eff: 10.0,
                    ..base
                },
                (SplitExact, None, "narrow_x"),
            ),
            (
                AutoInput {
                    n: 10,
                    p: AUTO_MAX_P_PER_N * 10 + 1,
                    n_eff: 10.0,
                    ..base
                },
                (SplitExact, None, "small_n_eff"),
            ),
            (
                AutoInput {
                    n_eff: 10.0,
                    no_refit: false,
                    pls3: true,
                    ..base
                },
                (SplitExact, None, "small_n_eff"),
            ),
            (
                AutoInput {
                    p: 4,
                    n_eff: 10.0,
                    pls3: true,
                    covariates: true,
                    ..base
                },
                (SplitExact, None, "covariates"),
            ),
        ] {
            let (n, p, n_eff, pls3) = (input.n, input.p, input.n_eff, input.pls3);
            assert_eq!(
                resolve_auto(&input),
                want,
                "n={n} p={p} n_eff={n_eff} pls3={pls3}"
            );
        }
    }

    /// Under `pre_standardized` the gate reads X as given, only centred: on
    /// X divided by hand by the unflagged call's scale it agrees with the
    /// unflagged call, and a column rescale `X·D` moves its stable rank
    /// (here past the floor), which the unflagged call does not see.
    #[test]
    fn split_nb_gate_reads_the_flag() {
        let (n, p) = (300, 10);
        let x = synth(n, p, 0, 0.0, 24).0;
        let scale = crate::linalg::held_out_scale(x.as_ref(), None, false);
        let xs = Mat::<f64>::from_fn(n, p, |i, j| x[(i, j)] / scale[j]);
        let xd = Mat::<f64>::from_fn(n, p, |i, j| xs[(i, j)] * if j == 0 { 30.0 } else { 1.0 });
        let gate = |x: &Mat<f64>, flag| split_nb_gate(x.as_ref(), None, flag).unwrap();
        let plain = gate(&x, false);
        assert!(!plain.fires, "{plain:?}");
        for got in [gate(&xs, true), gate(&xd, false)] {
            assert!(
                (got.stable_rank - plain.stable_rank).abs() <= 1e-12 * plain.stable_rank,
                "{got:?} against {plain:?}"
            );
            assert!(!got.fires);
        }
        let moved = gate(&xd, true);
        assert!(moved.fires && moved.stable_rank < 1.5, "{moved:?}");
    }

    // ── input validation ─────────────────────────────────────────────────────

    #[test]
    fn confirmatory_rejects_keep_with_ci() {
        let (x, y) = synth(80, 5, 3, 4.0, 99);
        let e = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[2]),
                weights: None,
                covariates: None,
                null_y: None,
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
                k: Some(&[1]),
                weights: None,
                covariates: None,
                null_y: None,
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
                k: Some(&[2]),
                weights: None,
                covariates: None,
                null_y: None,
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

    #[test]
    fn ci_branch_rejects_m_below_k_plus_2() {
        // n = 20, m_rate = 0.51: m = ceil(20^0.51) = 5 < k + 2 = 6.
        let (x, y) = synth(20, 6, 0, 0.0, 3);
        let err = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[4]),
                weights: None,
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::Score,
                seed: Some(7),
                ci: Some(CIOpts {
                    n_boot: 200,
                    m_rate: 0.51,
                    level: 0.95,
                    max_failure_rate: 0.01,
                }),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(err.to_string().contains("need m ≥ k+2"), "{err}");
    }

    /// The subsample size `m_rate` gives on this data is checked before the
    /// test runs. The input is `raw_perm_errors_when_most_null_fits_fail`'s,
    /// whose test run ends in `PermNullDegenerate`; n = 20 and `m_rate = 0.51`
    /// give m = 5 < k + 2 = 6, and that error must win.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn ci_branch_rejects_m_below_k_plus_2_before_the_test_runs() {
        let n = 20;
        let (x, _) = signal_data(n, 4, 61);
        let mut y = Col::<f64>::zeros(n);
        y[0] = f64::MAX;
        y[1] = -f64::MAX;
        let w = Col::<f64>::from_fn(n, |i| if i < 2 { 1.0 } else { i as f64 + 1.0 });
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[4]),
                weights: Some(w.as_ref()),
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 19,
                    n_folds: 5,
                },
                seed: Some(3),
                ci: Some(CIOpts {
                    m_rate: 0.51,
                    ..CIOpts::default()
                }),
                ..Default::default()
            },
        );
        let Err(err) = r else {
            panic!("expected an error, got {r:?}");
        };
        assert_eq!(err.code(), "invalid_argument", "{err:?}");
        assert!(err.to_string().contains("need m ≥ k+2"), "{err}");
    }

    /// Weight concentrated on five rows: most size-m subsamples have too
    /// few effective rows for k = 3, so the confirmatory CI stops at the
    /// skip-rate guard.
    #[test]
    fn ci_branch_skip_rate_guard_fires_under_concentrated_weights() {
        let (x, y) = synth(60, 5, 1, 4.0, 13);
        let w = Col::<f64>::from_fn(60, |i| if i < 5 { 1.0 } else { 1e-8 });
        let err = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[3]),
                weights: Some(w.as_ref()),
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::Score,
                seed: Some(0),
                ci: Some(CIOpts {
                    n_boot: 500,
                    ..CIOpts::default()
                }),
                ..Default::default()
            },
        )
        .unwrap_err();
        match err {
            PlsKitError::ResamplingDegenerate {
                skipped,
                total,
                skip_rate,
                threshold,
            } => {
                assert_eq!(total, 500);
                assert!((threshold - 0.01).abs() < 1e-15, "{threshold}");
                assert!(
                    skipped > 5 && skip_rate > threshold,
                    "{skipped} {skip_rate}"
                );
            }
            other => panic!("{other:?}"),
        }
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
        let clean = pooled_cv_r2_columns(
            primal,
            x.as_ref(),
            &folds,
            None,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
            &[2],
            None,
            4,
            &|c| column(c, None),
        )
        .map(|m| first_row(&m))
        .unwrap();
        let hit = pooled_cv_r2_columns(
            primal,
            x.as_ref(),
            &folds,
            None,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
            &[2],
            None,
            4,
            &|c| column(c, Some(2)),
        )
        .map(|m| first_row(&m))
        .unwrap();
        assert!(hit[2].is_nan(), "a failed null column is NaN");
        for c in [0, 1, 3] {
            assert_eq!(hit[c].to_bits(), clean[c].to_bits(), "column {c} untouched");
        }
        let err = pooled_cv_r2_columns(
            primal,
            x.as_ref(),
            &folds,
            None,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
            &[2],
            None,
            4,
            &|c| column(c, Some(0)),
        )
        .unwrap_err();
        assert_eq!(
            err.code(),
            "non_finite_input",
            "column 0 propagates its error"
        );
        // A non-finite X fails once, in the preparation of fold 0: the
        // all-row scale of its column is NaN, so every fold's block is.
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
            let fold = prepare_cv_fold(
                x.as_ref(),
                &folds,
                0,
                Some(wn.as_ref()),
                crate::linalg::held_out_scale(x.as_ref(), Some(wn.as_ref()), false).as_ref(),
            )
            .unwrap();
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
        let prep = prepare_split(
            x.as_ref(),
            &split,
            Some(wn.as_ref()),
            crate::linalg::held_out_scale(x.as_ref(), Some(wn.as_ref()), false).as_ref(),
        );
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
        assert!(
            !prepare_split(
                x.as_ref(),
                &sp,
                None,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref()
            )
            .x_finite
        );
        for k in [1_usize, 2] {
            let r = split_half_r(
                x.as_ref(),
                y.as_ref(),
                k,
                &sp,
                None,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                None,
            );
            assert_eq!(r.to_bits(), 0.0_f64.to_bits(), "k={k}");
        }
    }

    /// Both halves are centred with the training half's means and divided
    /// by the one column scale over all rows, then √w-scaled with each
    /// half's own weights.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn prepare_split_centres_with_part_means_and_all_row_scale() {
        let (x, _) = signal_data(40, 6, 87);
        // Unequal column scales, so no half's own scale is the all-row one.
        let x = Mat::<f64>::from_fn(40, 6, |i, j| x[(i, j)] * (1.0 + j as f64) + j as f64);
        let sp = SplitIdx {
            tr: (0..40).step_by(2).collect(),
            te: (1..40).step_by(2).collect(),
        };
        let wn = normalize_weights(crate::test_support::test_weights(40).as_ref()).unwrap();
        for w in [None, Some(wn.as_ref())] {
            let what = format!("weighted={}", w.is_some());
            let scale = crate::linalg::held_out_scale(x.as_ref(), w, false);
            let prep = prepare_split(x.as_ref(), &sp, w, scale.as_ref());
            let w_tr = renormalized_slice_weights(w, &sp.tr);
            let (_, mean, _) = crate::linalg::standardize_weighted(
                crate::linalg::row_subset(x.as_ref(), &sp.tr).as_ref(),
                w_tr.as_ref().map(Col::as_ref),
            );
            for (rows, xs, wp) in [
                (&sp.tr, &prep.xs_tr, &prep.w_tr),
                (&sp.te, &prep.xs_te, &prep.w_te),
            ] {
                for j in 0..6 {
                    for (i, &r) in rows.iter().enumerate() {
                        let f = wp.as_ref().map_or(1.0, |w| w[i].sqrt());
                        let expect = (x[(r, j)] - mean[j]) / scale[j] * f;
                        approx::assert_relative_eq!(xs[(i, j)], expect, max_relative = 1e-14);
                    }
                }
            }
            assert!(prep.w_tr.is_some() == w.is_some(), "{what}");
        }
    }

    /// Both blocks of a `raw_perm` fold are centred with the training rows'
    /// means (weighted with the fold's weights when weighted) and divided by
    /// the one column scale over all rows.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn prepare_cv_fold_centres_with_part_means_and_all_row_scale() {
        let (x, _) = signal_data(40, 6, 89);
        // Unequal column scales, so no fold's own scale is the all-row one.
        let x = Mat::<f64>::from_fn(40, 6, |i, j| x[(i, j)] * (1.0 + j as f64) + j as f64);
        let folds = crate::linalg::fold_split(&(0..40).collect::<Vec<_>>(), 5);
        let wn = normalize_weights(crate::test_support::test_weights(40).as_ref()).unwrap();
        for w in [None, Some(wn.as_ref())] {
            let scale = crate::linalg::held_out_scale(x.as_ref(), w, false);
            for fi in 0..folds.len() {
                let what = format!("weighted={}, fold {fi}", w.is_some());
                let fold = prepare_cv_fold(x.as_ref(), &folds, fi, w, scale.as_ref()).unwrap();
                assert!(fold.w_tr.is_some() == w.is_some(), "{what}");
                let (_, mean, _) = crate::linalg::standardize_weighted(
                    crate::linalg::row_subset(x.as_ref(), &fold.train_idx).as_ref(),
                    fold.w_tr.as_ref().map(Col::as_ref),
                );
                for (rows, xs) in [
                    (&fold.train_idx, &fold.xs_tr),
                    (&fold.val_idx, &fold.xs_val),
                ] {
                    for j in 0..6 {
                        for (i, &r) in rows.iter().enumerate() {
                            let expect = (x[(r, j)] - mean[j]) / scale[j];
                            approx::assert_relative_eq!(xs[(i, j)], expect, max_relative = 1e-14);
                        }
                    }
                }
            }
        }
    }

    /// Every confirmatory runner reads X through owned row gathers
    /// (`prepare_split`, `prepare_cv_fold`), a
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
                                k: Some(&[k]),
                                weights: w,
                                covariates: None,
                                null_y: None,
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
                                k: Some(&[1]),
                                weights: w,
                                covariates: None,
                                null_y: None,
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
    /// column 0 is `y`, column c the permutation of `seeds[c - 1]`. The
    /// driver runs twice, its folds one at a time (budget 0) and all in one
    /// batch (no limit), and the two must agree in every bit.
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
        let cols = Columns {
            y: y.as_mat(),
            seeds,
            null_map: None,
        };
        let run = |budget| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(budget, || {
                    pooled_cv_r2_columns(
                        route,
                        x,
                        folds,
                        w,
                        crate::linalg::held_out_scale(x, w, false).as_ref(),
                        &[k],
                        keep,
                        cols.len(),
                        &|c| cols.column(c),
                    )
                    .map(|m| first_row(&m))
                    .expect("columns")
                })
            })
        };
        let ((one, log_one), (all, log_all)) = (run(0), run(u64::MAX));
        let n_folds = folds.len();
        crate::resample::assert_batches(&log_one, n_folds, 1, 1);
        crate::resample::assert_batches(&log_all, n_folds, n_folds, 1);
        assert_eq!(
            one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{route:?}: the pooled columns depend on the budget"
        );
        one
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
            y: y.as_ref().as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        for (label, k, keep, weighted) in CASES {
            let wn = normalized(weighted.then_some(&w));
            let wref = wn.as_ref().map(Col::as_ref);
            // Closeness alone would pass if the GramP arm ran the Primal
            // arm's body, so units whose `ss_res` bits differ from the
            // Primal arm's are counted too: the Gram kernel must run.
            let mut differing = 0usize;
            for fi in 0..folds.len() {
                let fold = prepare_cv_fold(
                    x.as_ref(),
                    &folds,
                    fi,
                    wref,
                    crate::linalg::held_out_scale(x.as_ref(), wref, false).as_ref(),
                )
                .expect("fold");
                let g_block = fold_block(ReplicateRoute::GramP, &fold, Par::Seq);
                let p_block = fold_block(ReplicateRoute::Primal, &fold, Par::Seq);
                assert!(
                    matches!(g_block, FoldBlock::GramP(_)),
                    "{label}: fold_block(GramP) built another block"
                );
                for c in 0..cols.len() {
                    let yc = cols.column(c);
                    let y_of = |i: usize| yc[i];
                    let (res_g, tot_g) =
                        fold_unit(&g_block, &fold, &y_of, &[k], keep).expect("unit")[0];
                    let (res_p, tot_p) =
                        fold_unit(&p_block, &fold, &y_of, &[k], keep).expect("unit")[0];
                    assert_eq!(
                        tot_g.to_bits(),
                        tot_p.to_bits(),
                        "{label}, fold {fi}, column {c}: ss_tot"
                    );
                    assert!(
                        (res_g - res_p).abs() <= 1e-10 * tot_p.max(1.0),
                        "{label}, fold {fi}, column {c}: ss_res {res_g} vs {res_p}"
                    );
                    if res_g.to_bits() != res_p.to_bits() {
                        differing += 1;
                    }
                }
            }
            assert!(
                differing > 0,
                "{label}: no unit's ss_res bits differed from the Primal arm: the GramP arm \
                 may not run the Gram kernel"
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
        assert!(
            dense
                .iter()
                .zip(&keep_p)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn raw_perm_gram_p_arm_keeps_the_primal_errors_and_nan_columns() {
        // Row 5 is a validation row of fold 0 and a training row of every
        // other fold. A NaN there fails the Primal arm's per-column check in
        // folds 1..5, which the GramP arm runs first: a null column goes
        // NaN, column 0 returns the Primal route's error.
        let x = unif(2000, 50, 93);
        let y = linear_y(&x, 1.0, 94);
        let folds = folds5(2000);
        let seeds = seeds(6, 95);
        let cols = Columns {
            y: y.as_ref().as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        let column = |c: usize, bad: Option<usize>| -> Col<f64> {
            let mut yc = cols.column(c);
            if bad == Some(c) {
                yc[5] = f64::NAN;
            }
            yc
        };
        let run = |route: ReplicateRoute, bad: Option<usize>| {
            pooled_cv_r2_columns(
                route,
                x.as_ref(),
                &folds,
                None,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
                &[2],
                None,
                cols.len(),
                &|c| column(c, bad),
            )
            .map(|m| first_row(&m))
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
                        k: Some(&[2]),
                        weights: wopt.map(Col::as_ref),
                        covariates: None,
                        null_y: None,
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
                (g.statistic.unwrap() - p.statistic.unwrap()).abs() <= 1e-10,
                "{label}: statistic {} vs {}",
                g.statistic.unwrap(),
                p.statistic.unwrap()
            );
        }
    }

    /// Per-column z̄ through the `split_exact` driver on `route`. The driver
    /// runs twice, its splits one at a time (budget 0) and all in one batch
    /// (no limit), and the two must agree in every bit.
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
        let cols = Columns {
            y: y.as_mat(),
            seeds,
            null_map: None,
        };
        let run = |budget| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(budget, || {
                    let table = split_zbars_columns(
                        route,
                        x,
                        splits,
                        w,
                        crate::linalg::held_out_scale(x, w, false).as_ref(),
                        &[k],
                        keep,
                        cols.len(),
                        &|c| cols.column(c),
                        None,
                        None,
                    );
                    (0..table.ncols())
                        .map(|col| table[(0, col)])
                        .collect::<Vec<f64>>()
                })
            })
        };
        let ((one, log_one), (all, log_all)) = (run(0), run(u64::MAX));
        let n_splits = splits.len();
        crate::resample::assert_batches(&log_one, n_splits, 1, 1);
        crate::resample::assert_batches(&log_all, n_splits, n_splits, 1);
        assert_eq!(
            one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{route:?}: the split z̄ depend on the budget"
        );
        one
    }

    const SPLIT_CASES: [(&str, usize, Option<usize>, bool); 5] = [
        ("dense_k2", 2, None, false),
        ("weighted_k2", 2, None, true),
        ("keep10_k1", 1, Some(10), false),
        ("keep10_k2", 2, Some(10), false),
        ("weighted_keep10_k2", 2, Some(10), true),
    ];

    /// [`SPLIT_CASES`] on an `n × p` design: per split r of the `GramP` arm
    /// within 1e-10 absolute of the Primal arm's, then the z̄ statistics,
    /// each at two budgets over four splits (`split_exact_zbars`), with
    /// `n_perm` permuted columns.
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    fn assert_split_exact_gram_units_match_primal_units(n: usize, p: usize, n_perm: usize) {
        let x = unif(n, p, 99);
        let y = linear_y(&x, 2.0, 100);
        let w = gram_p_weights(n);
        let (_, mut rng) = crate::rng::resolve_seed(Some(101)).expect("seed");
        let splits = draw_splits(n, 2, 4, &mut rng).expect("splits");
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let cols = Columns {
            y: y.as_ref().as_mat(),
            seeds: &seeds,
            null_map: None,
        };
        for (label, k, keep, weighted) in SPLIT_CASES {
            let wn = normalized(weighted.then_some(&w));
            let wref = wn.as_ref().map(Col::as_ref);
            // Closeness alone would pass if the GramP arm ran the Primal
            // arm's body, so units whose `r` bits differ from the Primal
            // arm's are counted too: the Gram kernel and its scores must be
            // used.
            let mut differing = 0usize;
            for (si, sp) in splits.iter().enumerate() {
                let prep = prepare_split(
                    x.as_ref(),
                    sp,
                    wref,
                    crate::linalg::held_out_scale(x.as_ref(), wref, false).as_ref(),
                );
                let g_block = split_block(ReplicateRoute::GramP, &prep, Par::Seq);
                let p_block = split_block(ReplicateRoute::Primal, &prep, Par::Seq);
                assert!(
                    matches!(g_block, SplitBlock::GramP(_)),
                    "{label}: split_block(GramP) built another block"
                );
                for c in 0..cols.len() {
                    let yc = cols.column(c);
                    let y_of = |i: usize| yc[i];
                    let r_g = split_unit(&g_block, &prep, sp, &y_of, &[k], keep)[0];
                    let r_p = split_unit(&p_block, &prep, sp, &y_of, &[k], keep)[0];
                    assert!(
                        (r_g - r_p).abs() <= 1e-10,
                        "{label}, split {si}, column {c}: {r_g} vs {r_p}"
                    );
                    if r_g.to_bits() != r_p.to_bits() {
                        differing += 1;
                    }
                }
            }
            assert!(
                differing > 0,
                "{label}: no unit's r bits differed from the Primal arm: the GramP arm may not \
                 run the Gram kernel"
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
    fn split_exact_gram_units_match_primal_units() {
        // Dense, weighted, sparse and weighted sparse: the dense and
        // weighted refit cells have no corpus fixture, so this test and the
        // route-invisibility tests below carry them. n_tr·p·k <= 1e5: both
        // arms' fits are sequential.
        assert_split_exact_gram_units_match_primal_units(2000, 50, 40);
    }

    #[test]
    fn split_exact_gram_units_match_primal_units_past_the_auto_threshold() {
        // n_tr·p·k >= 1_200·900 = 1.08e6 for every case: the Primal arm's
        // fit (`SPLIT_FIT_PAR`) takes the parallel arm, and the GramP arm
        // forms its first step `ŝ₁` under the same `Par`. The test asserts
        // the shape is past the threshold and that the two arms agree.
        let (n, p) = (2400, 900);
        let (n_tr, _) = crate::resample::split_sizes(n, 2);
        for (label, k, _, _) in SPLIT_CASES {
            assert!(
                matches!(
                    crate::fit::resolve_par(SPLIT_FIT_PAR, n_tr, p, k),
                    Par::Rayon(_)
                ),
                "{label}: below the threshold"
            );
        }
        assert_split_exact_gram_units_match_primal_units(n, p, 20);
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
            let prep = prepare_split(
                x.as_ref(),
                sp,
                None,
                crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
            );
            let g_block = split_block(ReplicateRoute::GramP, &prep, Par::Seq);
            let p_block = split_block(ReplicateRoute::Primal, &prep, Par::Seq);
            let y_of = |i: usize| y[i];
            let r_g = split_unit(&g_block, &prep, sp, &y_of, &[2], None)[0];
            let r_p = split_unit(&p_block, &prep, sp, &y_of, &[2], None)[0];
            assert_eq!(r_g.to_bits(), r_p.to_bits());
            assert_eq!(r_p.to_bits(), 0.0_f64.to_bits());
        }
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
                split_exact_refit_route(2000, 50, 600, 5, 2, None, wopt.is_some(), false),
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
                        k: Some(&[2]),
                        weights: wopt.map(Col::as_ref),
                        covariates: None,
                        null_y: None,
                    },
                    opts,
                )
                .expect("split_exact")
            };
            let g = call();
            let p = with_gram_routes_disabled(call);
            assert_eq!(g.pvalue.to_bits(), p.pvalue.to_bits(), "{label}: pvalue");
            assert!(
                (g.statistic.unwrap() - p.statistic.unwrap()).abs() <= 1e-10,
                "{label}: statistic {} vs {}",
                g.statistic.unwrap(),
                p.statistic.unwrap()
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
            split_exact_refit_route(2000, 200, 400, 5, 1, Some(10), false, false),
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
            crate::find_k::spls1_find_k_sequence(x.as_ref(), y.as_ref(), 2, 10, None, None, opts)
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
    fn split_exact_gram_constant_held_out_y_gives_r_zero() {
        // A Gram fit of k components with ordinary test-half scores, but a
        // held-out outcome constant to rounding: r = 0 on both arms.
        let (x, y, sp) = near_constant_test_half(1.0, |_| 7.0);
        let prep = prepare_split(
            x.as_ref(),
            &sp,
            None,
            crate::linalg::held_out_scale(x.as_ref(), None, false).as_ref(),
        );
        let y_of = |i: usize| y[i];
        for k in 1..=2 {
            let SplitBlock::GramP(gram) = split_block(ReplicateRoute::GramP, &prep, Par::Seq)
            else {
                panic!("split_block(GramP) built another block")
            };
            let ys_tr = split_train_target(&prep, &sp, &y_of);
            let s1 =
                crate::gram_p::first_product(prep.xs_tr.as_ref(), ys_tr.as_ref(), SPLIT_FIT_PAR, k);
            let fit = gram.fit(s1.as_ref(), ys_tr.norm_l2(), k, None);
            assert!(fit.k_used == k, "premise: k_used = {}", fit.k_used);
            let r_g = split_column_r_gram_p(&gram, &prep, &sp, &y_of, &[k], None)[0];
            let r_p = split_column_r(&prep, &sp, &y_of, &[k], None)[0];
            assert_eq!(r_g.to_bits(), 0.0_f64.to_bits(), "k = {k}");
            assert_eq!(r_p.to_bits(), 0.0_f64.to_bits(), "k = {k}");
        }
    }
}

/// Covariates: Freedman–Lane null columns, the adjusted statistic, the
/// `covariates` rule of `auto`, the refusals and the `ci` branch.
#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests_covariates {
    use super::*;
    use crate::fit::{FitOpts, KSpec, pls1_fit};
    use crate::linalg::normalize_weights;
    use crate::test_support::synth;

    /// `[1, z0, 5, z0, z1]`: an intercept, a constant and a duplicate on top
    /// of two covariates, so `n_covariates = 2`.
    fn aliased_z(n: usize, seed: u64) -> Mat<f64> {
        let z = crate::test_support::unif(n, 2, seed);
        Mat::<f64>::from_fn(n, 5, |i, j| match j {
            0 => 1.0,
            1 | 3 => z[(i, 0)],
            2 => 5.0,
            _ => z[(i, 1)],
        })
    }

    /// `max_l |Σ_i a_il·w_i·c_i|`: how far `c` is from W-orthogonal to the
    /// columns of `a` (`w = 1` unweighted).
    fn max_weighted_inner(a: &Mat<f64>, w: Option<ColRef<'_, f64>>, c: &Col<f64>) -> f64 {
        (0..a.ncols())
            .map(|l| {
                (0..a.nrows())
                    .map(|i| a[(i, l)] * w.map_or(1.0, |w| w[i]) * c[i])
                    .sum::<f64>()
                    .abs()
            })
            .fold(0.0, f64::max)
    }

    /// Every Freedman–Lane null column is W-orthogonal to `[1, Z]` to
    /// rounding; the permuted residuals without the map are not.
    #[test]
    fn freedman_lane_null_columns_are_orthogonal_to_the_basis() {
        use crate::test_support::{test_weights, unif, with_intercept};
        let n = 50;
        let (x, y) = synth(n, 6, 2, 1.0, 51);
        let z = unif(n, 2, 52);
        let a = with_intercept(z.as_ref());
        let seeds = [3_u64, 99, 7, 21];
        for w in [None, normalize_weights(test_weights(n).as_ref())] {
            let wr = w.as_ref().map(Col::as_ref);
            let adj =
                crate::covariates::adjust(x.as_ref(), y.as_ref(), z.as_ref(), wr, false).unwrap();
            let fl = |_: usize, m: faer::MatMut<'_, f64>| adj.basis.residualize(m);
            let cols = |map: Option<crate::resample::NullMap<'_>>| {
                (1..=seeds.len())
                    .map(|c| {
                        crate::resample::Columns {
                            y: adj.y.as_ref().as_mat(),
                            seeds: &seeds,
                            null_map: map,
                        }
                        .column(c)
                    })
                    .collect::<Vec<_>>()
            };
            for (c, col) in cols(Some(&fl)).iter().enumerate() {
                let worst = max_weighted_inner(&a, wr, col);
                assert!(
                    worst <= 1e-12 * col.norm_l2(),
                    "column {}: max |A'W c| = {worst:e}",
                    c + 1
                );
            }
            let kennedy = cols(None);
            assert!(
                kennedy
                    .iter()
                    .any(|col| max_weighted_inner(&a, wr, col) > 1e-6 * col.norm_l2()),
                "premise: a permuted residual is not orthogonal to [1, Z]"
            );
        }
    }

    /// Under weights, null column c is the WLS residual of `ỹ[perm_c]`,
    /// where `ỹ` is the raw-scale WLS residual of the standardized y (the
    /// √w-scaled residual is never permuted): bit-identical to the column
    /// built by hand.
    #[test]
    fn weighted_freedman_lane_permutes_raw_scale_residuals() {
        use crate::resample::{Columns, permutation_from_seed};
        use crate::test_support::{
            assert_close_rel, test_weights, unif, with_intercept, wls_normal_equations,
        };
        let n = 50;
        let (x, y) = synth(n, 6, 2, 1.0, 53);
        let z = unif(n, 2, 54);
        let a = with_intercept(z.as_ref());
        let wn = normalize_weights(test_weights(n).as_ref()).unwrap();
        let adj =
            crate::covariates::adjust(x.as_ref(), y.as_ref(), z.as_ref(), Some(wn.as_ref()), false)
                .unwrap();
        let (ys, _, _) = crate::linalg::standardize1_weighted(y.as_ref(), Some(wn.as_ref()));
        let b = wls_normal_equations(a.as_ref(), Some(wn.as_ref()), ys.as_ref().as_mat());
        let want_y = Col::<f64>::from_fn(n, |i| {
            ys[i] - (0..a.ncols()).map(|l| a[(i, l)] * b[(l, 0)]).sum::<f64>()
        });
        assert_close_rel(
            adj.y.as_ref().as_mat(),
            want_y.as_ref().as_mat(),
            1e-10,
            "ỹ",
        );
        let seeds = [5_u64, 17, 2024];
        let fl = |_: usize, m: faer::MatMut<'_, f64>| adj.basis.residualize(m);
        let cols = Columns {
            y: adj.y.as_ref().as_mat(),
            seeds: &seeds,
            null_map: Some(&fl),
        };
        for (c, &s) in seeds.iter().enumerate() {
            let perm = permutation_from_seed(n, s);
            let mut v = Col::<f64>::from_fn(n, |i| adj.y[perm[i]]);
            adj.basis.residualize(v.as_mut().as_mat_mut());
            let got = cols.column(c + 1);
            assert!(
                (0..n).all(|i| got[i].to_bits() == v[i].to_bits()),
                "null column {}",
                c + 1
            );
        }
    }

    /// With covariates, `split_exact` and `raw_perm` report the statistic
    /// of the call without covariates on `adjust`'s `(X̃, ỹ)` under
    /// `pre_standardized` (Kennedy's observed statistic), dense and with
    /// `keep` (selection acts on X̃), unweighted and weighted, and record
    /// the kept covariates.
    #[test]
    fn covariates_observed_statistic_equals_adjusted_call() {
        use crate::test_support::test_weights;
        let n = 80;
        let (x, y) = synth(n, 10, 3, 1.0, 61);
        let z = aliased_z(n, 62);
        let run = |x: MatRef<'_, f64>,
                   y: ColRef<'_, f64>,
                   w: Option<ColRef<'_, f64>>,
                   z: Option<MatRef<'_, f64>>,
                   args,
                   keep,
                   pre| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x,
                    y,
                    k: Some(&[1]),
                    weights: w,
                    covariates: z,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(63),
                    keep,
                    pre_standardized: pre,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        for w in [None, Some(test_weights(n))] {
            let wr = w.as_ref().map(Col::as_ref);
            let (wn, _) = crate::fit::validate_and_normalize_weights(wr, n, 1).unwrap();
            let adj = crate::covariates::adjust(
                x.as_ref(),
                y.as_ref(),
                z.as_ref(),
                wn.as_ref().map(Col::as_ref),
                false,
            )
            .unwrap();
            for args in [
                ConfirmatoryArgs::SplitExact {
                    n_perm: 19,
                    n_splits: 4,
                },
                ConfirmatoryArgs::RawPerm {
                    n_perm: 19,
                    n_folds: 5,
                },
            ] {
                for keep in [None, Some(3)] {
                    let tag = format!("{args:?} keep={keep:?} weighted={}", w.is_some());
                    let got = run(
                        x.as_ref(),
                        y.as_ref(),
                        wr,
                        Some(z.as_ref()),
                        args,
                        keep,
                        false,
                    );
                    let want = run(adj.x.as_ref(), adj.y.as_ref(), wr, None, args, keep, true);
                    assert!(
                        (got.statistic.unwrap() - want.statistic.unwrap()).abs()
                            <= 1e-12 * want.statistic.unwrap().abs(),
                        "{tag}: {} vs {}",
                        got.statistic.unwrap(),
                        want.statistic.unwrap()
                    );
                    assert_eq!(got.n_covariates, Some(2), "{tag}");
                    assert_eq!(want.n_covariates, None, "{tag}");
                    assert_eq!(got.test_method, args.method().as_str(), "{tag}");
                }
            }
        }
    }

    /// With covariates, the p-value is the runner's on `adjust`'s `(X̃, ỹ)`
    /// with the Freedman–Lane map at the same seed: every runner the call
    /// dispatches to (`split_exact`'s no-refit and refit routes, `raw_perm`)
    /// receives the map.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn covariates_pvalue_comes_from_freedman_lane_columns() {
        use crate::test_support::test_weights;
        let n = 80;
        // y independent of X, so p-values sit inside (0, 1) where the null
        // columns decide them.
        let (x, y) = synth(n, 10, 0, 0.0, 61);
        let z = aliased_z(n, 62);
        let mut kennedy_differs = false;
        for w in [None, Some(test_weights(n))] {
            let wr = w.as_ref().map(Col::as_ref);
            let (wn, _) = crate::fit::validate_and_normalize_weights(wr, n, 1).unwrap();
            let wn = wn.as_ref().map(Col::as_ref);
            let adj =
                crate::covariates::adjust(x.as_ref(), y.as_ref(), z.as_ref(), wn, false).unwrap();
            let fl = |_: usize, m: faer::MatMut<'_, f64>| adj.basis.residualize(m);
            let ones = Col::<f64>::from_fn(10, |_| 1.0);
            let (xa, ya, sc) = (adj.x.as_ref(), adj.y.as_ref(), ones.as_ref());
            for (args, keep) in [
                (
                    ConfirmatoryArgs::SplitExact {
                        n_perm: 99,
                        n_splits: 4,
                    },
                    None,
                ),
                (
                    ConfirmatoryArgs::SplitExact {
                        n_perm: 99,
                        n_splits: 4,
                    },
                    Some(3),
                ),
                (
                    ConfirmatoryArgs::RawPerm {
                        n_perm: 99,
                        n_folds: 5,
                    },
                    None,
                ),
            ] {
                let opts = ConfirmatoryTestOpts {
                    args,
                    seed: Some(63),
                    keep,
                    ..Default::default()
                };
                let got = pls1_confirmatory_test(
                    ConfirmatoryTestInput::Raw {
                        x: x.as_ref(),
                        y: y.as_ref(),
                        k: Some(&[1]),
                        weights: wr,
                        covariates: Some(z.as_ref()),
                        null_y: None,
                    },
                    opts,
                )
                .unwrap();
                let opts = ConfirmatoryTestOpts {
                    pre_standardized: true,
                    ..opts
                };
                let runner = |map: Option<crate::resample::NullMap<'_>>| {
                    let (_, mut rng) = crate::rng::resolve_seed(Some(63)).unwrap();
                    match args {
                        ConfirmatoryArgs::SplitExact { n_perm, n_splits } => {
                            match split_exact_refit_route(
                                n,
                                10,
                                n_perm,
                                n_splits,
                                1,
                                keep,
                                wn.is_some(),
                                false,
                            ) {
                                ReplicateRoute::Special => run_split_perm_nr(
                                    xa, ya, 1, n_perm, n_splits, wn, sc, &opts, map, None, &mut rng,
                                )
                                .map(SetResult::from),
                                route => run_split_perm(
                                    route,
                                    xa,
                                    ya,
                                    &[1],
                                    n_perm,
                                    n_splits,
                                    wn,
                                    sc,
                                    &opts,
                                    map,
                                    None,
                                    &mut rng,
                                ),
                            }
                        }
                        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => run_raw_perm(
                            xa,
                            ya,
                            &[1],
                            n_perm,
                            n_folds,
                            wn,
                            sc,
                            &opts,
                            map,
                            &mut rng,
                        ),
                        _ => unreachable!(),
                    }
                    .unwrap()
                    .pvalue
                };
                let want = runner(Some(&fl));
                kennedy_differs |= runner(None).to_bits() != want.to_bits();
                assert_eq!(
                    got.pvalue.to_bits(),
                    want.to_bits(),
                    "{args:?} keep={keep:?} weighted={}: {} vs {want}",
                    w.is_some(),
                    got.pvalue,
                );
            }
        }
        assert!(
            kennedy_differs,
            "premise: Kennedy's columns give another p-value"
        );
    }

    /// The `covariates` rule decides first: on a request every later rule
    /// would send to `split_nb` it gives `split_exact` without reading the
    /// stable rank, and a public `auto` call with covariates runs
    /// `split_exact` on a design that otherwise runs `split_nb`.
    #[test]
    fn auto_with_covariates_takes_the_covariates_rule() {
        use ConfirmatoryMethod::{SplitExact, SplitNb};
        let unread = || -> f64 { panic!("the stable rank was read") };
        let flat = || 10.0;
        let to_nb = AutoInput {
            n: 2000,
            p: 10,
            n_eff: 2000.0,
            no_refit: false,
            pls3: false,
            covariates: false,
            multi_k: false,
            stable_rank: &flat,
        };
        assert_eq!(resolve_auto(&to_nb), (SplitNb, Some(10.0), "split_nb_gate"));
        assert_eq!(
            resolve_auto(&AutoInput {
                covariates: true,
                stable_rank: &unread,
                ..to_nb
            }),
            (SplitExact, None, "covariates")
        );

        let (x, y) = synth(1200, 10, 3, 0.3, 13);
        let z = aliased_z(1200, 14);
        let run = |z: Option<MatRef<'_, f64>>| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[1]),
                    weights: None,
                    covariates: z,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::Auto {
                        n_perm: 19,
                        n_splits: 4,
                    },
                    seed: Some(15),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        assert_eq!(run(None).test_method, "split_nb", "premise");
        let got = run(Some(z.as_ref()));
        assert_eq!(got.test_method, "split_exact");
        assert_eq!(got.stable_rank, None);
        assert_eq!(got.n_covariates, Some(2));
    }

    /// `split_nb`, `score` and `e` refuse covariates, naming the methods
    /// that take them.
    #[test]
    fn covariates_refuse_split_nb_score_e() {
        let n = 60;
        let (x, y) = synth(n, 8, 2, 1.0, 64);
        let z = aliased_z(n, 65);
        for args in [
            ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::SplitNb),
            ConfirmatoryArgs::Score,
            ConfirmatoryArgs::E,
        ] {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[1]),
                    weights: None,
                    covariates: Some(z.as_ref()),
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed: Some(1),
                    ..Default::default()
                },
            );
            match r {
                Err(PlsKitError::InvalidArgument(m)) => {
                    for name in ["split_exact", "raw_perm", "auto", args.method().as_str()] {
                        assert!(m.contains(name), "{args:?}: {m}");
                    }
                }
                other => panic!("{args:?}: {other:?}"),
            }
        }
    }

    /// The CI's held-out correlation with covariates standardizes the held-out
    /// y with the training half's moments before its residual, so it does not
    /// move with an affine map of y.
    #[test]
    fn ci_holdout_corr_with_covariates_ignores_an_affine_map_of_y() {
        let n = 80;
        let (x, y) = synth(n, 6, 2, 1.0, 66);
        let z = aliased_z(n, 67);
        let y2 = Col::<f64>::from_fn(n, |i| 100.0 + 3.0 * y[i]);
        let run = |y: ColRef<'_, f64>| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y,
                    k: Some(&[1]),
                    weights: None,
                    covariates: Some(z.as_ref()),
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 19,
                        n_splits: 4,
                    },
                    seed: Some(68),
                    ci: Some(CIOpts {
                        n_boot: 100,
                        ..CIOpts::default()
                    }),
                    ..Default::default()
                },
            )
            .unwrap()
            .ci
            .expect("ci")
            .holdout_corr
        };
        let (a, b) = (run(y.as_ref()), run(y2.as_ref()));
        assert!(a.point.is_finite() && a.sd > 0.0);
        assert!(
            (a.point - b.point).abs() < 1e-10,
            "{} vs {}",
            a.point,
            b.point
        );
        assert!((a.sd - b.sd).abs() < 1e-10);
    }

    /// `ci = True` with covariates refits on the raw data with covariates:
    /// the reference fit is the covariate fit, so the signed β z carries the
    /// sign of its β.
    #[test]
    fn ci_with_covariates_runs_on_the_covariate_fit() {
        let n = 80;
        let (x, y) = synth(n, 6, 2, 1.0, 66);
        let z = aliased_z(n, 67);
        let out = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[1]),
                weights: None,
                covariates: Some(z.as_ref()),
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitExact {
                    n_perm: 19,
                    n_splits: 4,
                },
                seed: Some(68),
                ci: Some(CIOpts {
                    n_boot: 100,
                    ..CIOpts::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let fit = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(1),
            None,
            Some(z.as_ref()),
            FitOpts::default(),
        )
        .unwrap();
        let ci = out.ci.expect("ci");
        for (j, zs) in ci.beta_sign_z_signed.iter().enumerate() {
            if zs.is_finite() && *zs != 0.0 {
                assert_eq!(zs.signum(), fit.beta[j].signum(), "j = {j}");
            }
        }
        assert!(ci.leverage_se.iter().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod tests_null_y {
    use super::*;
    use crate::test_support::{synth, unif};

    fn opts(args: ConfirmatoryArgs) -> ConfirmatoryTestOpts {
        ConfirmatoryTestOpts {
            args,
            seed: Some(7),
            ..Default::default()
        }
    }

    fn exact() -> ConfirmatoryArgs {
        ConfirmatoryArgs::SplitExact {
            n_perm: 1000,
            n_splits: 5,
        }
    }

    fn raw<'a>(
        x: &'a Mat<f64>,
        y: &'a Col<f64>,
        cov: Option<&'a Mat<f64>>,
        ny: Option<&'a Mat<f64>>,
    ) -> ConfirmatoryTestInput<'a> {
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: Some(&[1]),
            weights: None,
            covariates: cov.map(Mat::as_ref),
            null_y: ny.map(Mat::as_ref),
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn null_y_equal_to_y_gives_p_one() {
        let (x, y) = synth(60, 6, 2, 1.0, 11);
        let ny = Mat::<f64>::from_fn(60, 99, |i, _| y[i]);
        let r = pls1_confirmatory_test(raw(&x, &y, None, Some(&ny)), opts(exact())).unwrap();
        assert_eq!(r.n_perm, Some(99));
        assert_eq!(r.pvalue, 1.0);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn null_y_p_is_a_rank() {
        let x = unif(60, 6, 12);
        let y = unif(60, 1, 13).col(0).to_owned();
        let ny = unif(60, 99, 14);
        let r = pls1_confirmatory_test(raw(&x, &y, None, Some(&ny)), opts(exact())).unwrap();
        let rank = r.pvalue * 100.0;
        assert!((rank - rank.round()).abs() < 1e-12, "p = {}", r.pvalue);
        assert!(r.pvalue > 0.01 && r.pvalue < 1.0, "p = {}", r.pvalue);
    }

    /// Weighted, each `null_y` column is centred by its own weighted mean,
    /// so shifting the null columns moves neither p nor the statistic.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn weighted_null_y_ignores_the_null_columns_location() {
        let (x, y) = synth(200, 40, 0, 0.0, 21);
        let ny = unif(200, 199, 16);
        let shifted = Mat::<f64>::from_fn(200, 199, |i, j| ny[(i, j)] + 5.0);
        let w = Col::<f64>::from_fn(200, |i| 0.2 + 2.8 * ((i * 7) % 11) as f64 / 10.0);
        let run = |ny: &Mat<f64>| {
            let mut inp = raw(&x, &y, None, Some(ny));
            if let ConfirmatoryTestInput::Raw { weights, .. } = &mut inp {
                *weights = Some(w.as_ref());
            }
            pls1_confirmatory_test(inp, opts(exact())).unwrap()
        };
        let (a, b) = (run(&ny), run(&shifted));
        assert_eq!(a.pvalue, b.pvalue);
        let (sa, sb) = (a.statistic.unwrap(), b.statistic.unwrap());
        assert!((sa - sb).abs() <= 1e-12, "{sa} vs {sb}");
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn null_y_with_gram_equals_x_call() {
        let (x, y) = synth(50, 6, 2, 1.0, 15);
        let ny = unif(50, 40, 16);
        let a = pls1_confirmatory_test(raw(&x, &y, None, Some(&ny)), opts(exact())).unwrap();
        let g = crate::gram_matrix(x.as_ref(), false).unwrap();
        let b = pls1_confirmatory_test(
            ConfirmatoryTestInput::Gram {
                gram: &g,
                y: y.as_ref(),
                k: Some(&[1]),
                covariates: None,
                null_y: Some(ny.as_ref()),
            },
            opts(exact()),
        )
        .unwrap();
        assert_eq!(b.n_perm, Some(40));
        assert_eq!(a.pvalue, b.pvalue);
        assert!(
            (a.statistic.unwrap() - b.statistic.unwrap()).abs()
                <= 1e-12 * a.statistic.unwrap().abs().max(1e-300),
            "{} {}",
            a.statistic.unwrap(),
            b.statistic.unwrap()
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn null_y_refusals() {
        let (x, y) = synth(40, 5, 2, 1.0, 17);
        let ny = unif(40, 9, 18);
        let g = crate::gram_matrix(x.as_ref(), false).unwrap();
        for args in [
            ConfirmatoryArgs::RawPerm {
                n_perm: 50,
                n_folds: 5,
            },
            ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::SplitNb),
            ConfirmatoryArgs::Score,
            ConfirmatoryArgs::E,
            ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::Auto),
        ] {
            let r = pls1_confirmatory_test(raw(&x, &y, None, Some(&ny)), opts(args));
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "{args:?}: {r:?}"
            );
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Gram {
                    gram: &g,
                    y: y.as_ref(),
                    k: Some(&[1]),
                    covariates: None,
                    null_y: Some(ny.as_ref()),
                },
                opts(args),
            );
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "gram {args:?}: {r:?}"
            );
        }
        let z = unif(40, 2, 19);
        let r = pls1_confirmatory_test(raw(&x, &y, Some(&z), Some(&ny)), opts(exact()));
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))), "{r:?}");
        let short = unif(39, 9, 20);
        let r = pls1_confirmatory_test(raw(&x, &y, None, Some(&short)), opts(exact()));
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))), "{r:?}");
        let mut bad = ny.clone();
        bad[(3, 2)] = f64::NAN;
        let r = pls1_confirmatory_test(raw(&x, &y, None, Some(&bad)), opts(exact()));
        assert!(matches!(r, Err(PlsKitError::NonFiniteInput)), "{r:?}");
        let none = Mat::<f64>::zeros(40, 0);
        let r = pls1_confirmatory_test(raw(&x, &y, None, Some(&none)), opts(exact()));
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))), "{r:?}");
    }

    fn gram_in<'a>(
        g: &'a crate::held_out_gram::GramMatrixOutput,
        y: &'a Col<f64>,
        k: &'a [usize],
        cov: Option<&'a Mat<f64>>,
        ny: &'a Mat<f64>,
    ) -> ConfirmatoryTestInput<'a> {
        ConfirmatoryTestInput::Gram {
            gram: g,
            y: y.as_ref(),
            k: Some(k),
            covariates: cov.map(Mat::as_ref),
            null_y: Some(ny.as_ref()),
        }
    }

    /// k = 2 takes the refit routes; weights take the weighted path.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn null_y_equal_to_y_gives_p_one_on_refit_and_weighted_routes() {
        let (x, y) = synth(60, 6, 2, 1.0, 21);
        let ny = Mat::<f64>::from_fn(60, 30, |i, _| y[i]);
        let mut inp = raw(&x, &y, None, Some(&ny));
        if let ConfirmatoryTestInput::Raw { k, .. } = &mut inp {
            *k = Some(&[2]);
        }
        let r = pls1_confirmatory_test(inp, opts(exact())).unwrap();
        assert_eq!((r.n_perm, r.pvalue), (Some(30), 1.0));
        let g = crate::gram_matrix(x.as_ref(), false).unwrap();
        let r = pls1_confirmatory_test(gram_in(&g, &y, &[2], None, &ny), opts(exact())).unwrap();
        assert_eq!((r.n_perm, r.pvalue), (Some(30), 1.0));
        let w = crate::test_support::test_weights(60);
        let mut inp = raw(&x, &y, None, Some(&ny));
        if let ConfirmatoryTestInput::Raw { weights, .. } = &mut inp {
            *weights = Some(w.as_ref());
        }
        let r = pls1_confirmatory_test(inp, opts(exact())).unwrap();
        assert_eq!((r.n_perm, r.pvalue), (Some(30), 1.0));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn null_y_gram_refusals() {
        let (x, y) = synth(40, 5, 2, 1.0, 22);
        let g = crate::gram_matrix(x.as_ref(), false).unwrap();
        let ny = unif(40, 9, 23);
        let z = unif(40, 2, 24);
        let r = pls1_confirmatory_test(gram_in(&g, &y, &[1], Some(&z), &ny), opts(exact()));
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))), "{r:?}");
        let short = unif(39, 9, 25);
        let r = pls1_confirmatory_test(gram_in(&g, &y, &[1], None, &short), opts(exact()));
        assert!(matches!(r, Err(PlsKitError::ShapeMismatch(_))), "{r:?}");
        let mut bad = ny.clone();
        bad[(3, 2)] = f64::NAN;
        let r = pls1_confirmatory_test(gram_in(&g, &y, &[1], None, &bad), opts(exact()));
        assert!(matches!(r, Err(PlsKitError::NonFiniteInput)), "{r:?}");
    }
}
