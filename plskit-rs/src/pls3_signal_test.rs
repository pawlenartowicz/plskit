//! Confirmatory PLS3 / PLSSVD omnibus test at LV1. Three methods: two split
//! statistics (`split_exact`, `split_nb`) and `sv_perm`. `split_exact`
//! calibrates the held-out LV correlation by permutation, `split_nb`
//! compares it against the NB t-reference instead, and `sv_perm` calibrates
//! `d₁`, the largest in-sample singular value, by permutation (see below).
//!
//! # The statistic
//! Fit PLS3 on the training half for `(u₁, v₁)`, then take the Pearson
//! correlation of the two held-out LV score vectors,
//! `r = cor(X̃_te u₁, Ỹ_te v₁)`. Fisher-z average across the J splits and
//! report `tanh(z̄)`, exactly as `split_nb` and `split_exact` do for PLS1 —
//! the same z̄ scale is also what the permutation comparison happens on
//! (see `signal_test::mean_fisher_z`).
//!
//! # Why no alignment step is needed
//! An SVD fixes `(u₁, v₁)` only up to a simultaneous sign flip, and a flip
//! negates both held-out score vectors at once, leaving `r` untouched. The
//! training fit hands the pair over as a unit, so no procrustes machinery
//! enters here. Per-coordinate CIs would differ: a multi-column coefficient
//! matrix inherits the rotation/sign indeterminacy and would need procrustes
//! alignment per resample, which is why they are PLS1-only (see
//! `docs/python/results.md`, "Per-coordinate β CIs").
//!
//! # `split_nb` here versus for PLS1
//! Both sides of the correlation are directions estimated on the training
//! half, where PLS1 has an observed outcome on one side. What carries over
//! is the per-split marginal null: conditional on X, the training half and
//! the all-row scales, `X̃_te u₁` is fixed and, under block independence,
//! the rows of `Ỹ_te v₁` are exchangeable. (The Y scale reads the test rows
//! too, so `Ỹ_te v₁` is not a fixed combination of independent rows.) So
//! the conditional null of `r` on one split is the ordinary null
//! correlation law to the same approximation as in PLS1; with both
//! `pre_standardized` flags set the statement is exact. PLS1 is the same
//! construction with `v₁ = 1`. That supports the `rho_hat` ruler
//! and nothing more: the p-value comes from `nb_test`, whose between-split
//! inflation `1/J + n_test/n_train` is PLS1's empirical heuristic, not
//! derived for two blocks. Measured on iid Gaussian, heavy-tailed,
//! low-stable-rank and real two-block designs, `atanh(r)·√(n_test−3)` has
//! unit spread and no excess kurtosis and the resulting test is
//! conservative except on X of low effective rank (up to 6.2% at
//! α = 0.05). `split_exact` stays the
//! recommendation because it is exact whenever rows are exchangeable under
//! the null (clustered rows break both methods); `split_nb` is the cheap
//! opt-in.
//!
//! # Why only at k = 1
//! Above k = 1 the training-half component *ordering* need not survive to
//! the test half when singular values are close, and whether the statistic
//! should then be per-component or subspace-level is undecided — so k > 1
//! errors rather than guessing.
//!
//! # `sv_perm`: the in-sample singular-value permutation test
//! The PLSC test of McIntosh & Lobaugh (2004), "Partial least squares
//! analysis of neuroimaging data: applications and advances", NeuroImage
//! 23:S250–S263, at LV1, run for comparison with pyls; `split_exact` stays
//! the recommendation. No split: the statistic is `d₁`, the largest
//! singular value of `X̃ᵀỸ` on all rows (X̃, Ỹ centred and scaled once, as
//! above), and each null replicate permutes the rows of Ỹ against X̃. With
//! `rotate`, each permuted Y-side basis is Procrustes-rotated onto the
//! observed one before LV1 is read, as pyls does by default. For LV1 that
//! can only lower p, so the rotated test is not exact (see
//! `run_sv_perm_pls3`).

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::held_out_gram::{GramMatrixOutput, HeldOutGram};
use crate::resample::SplitDesign;
use crate::route_select::{Candidate, Choice, select};
use crate::signal_test::{
    AutoInput, ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestOutput, PrimalSource,
    ReplicateRoute, SPLIT_NB_REROUTE_N_PERM, SplitIdx, SplitRoute, TestArm,
    check_split_exact_counts, check_split_rows, draw_design, nb_rho_hat, nb_test, perm_pvalue,
    resolve_auto, resolve_auto_on_x, resolve_split_nb, split_nb_gate_fires,
};

/// Cross-cutting tuning knobs for [`pls3_confirmatory_test`].
///
/// Deliberately not `signal_test::ConfirmatoryTestOpts`: PLS3 needs two
/// pre-standardization flags where PLS1 needs one, and cannot honor that
/// struct's `ci`, `keep` or `max_skip_rate`. The `args` enum and the output
/// struct are shared, so method dispatch and result field names stay
/// identical across the two families.
///
/// Four bools, so the `struct_excessive_bools` allow that
/// `signal_test::ConfirmatoryTestOpts` carries is needed here too — the
/// workspace turns clippy's `pedantic` group on as a warning and CI gates
/// on `-D warnings`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct Pls3ConfirmatoryTestOpts {
    /// Method dispatch + per-method args. [`ConfirmatoryArgs::SplitExact`],
    /// [`ConfirmatoryArgs::SplitNb`], [`ConfirmatoryArgs::SvPerm`] and
    /// [`ConfirmatoryArgs::Auto`] (the default, which runs `split_exact` on
    /// every design) are accepted; the other three variants error.
    pub args: ConfirmatoryArgs,
    /// Caller asserts X is already standardized: skip the one
    /// standardization of X over all rows.
    ///
    /// Without the flag, X's columns are divided by their standard
    /// deviations over all rows, the scale `pls3_fit` uses, once per call.
    /// Each split then centres its training rows with their own means and
    /// its held-out rows with those same means; nothing is scaled per
    /// split. With the flag, X is used as given and each split only
    /// centres. The `split_nb` gate reads X under the same rule.
    pub pre_standardized_x: bool,
    /// Caller asserts Y is already standardized: skip the one
    /// standardization of Y over all rows. Y follows the rule of
    /// `pre_standardized_x`. Its scale is taken from Y's rows before any
    /// permutation; permuting rows does not change it.
    pub pre_standardized_y: bool,
    /// RNG seed; `None` draws from OS entropy and the drawn value is
    /// recorded on the result.
    pub seed: Option<u64>,
}

impl Default for Pls3ConfirmatoryTestOpts {
    fn default() -> Self {
        Self {
            args: ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::Auto),
            pre_standardized_x: false,
            pre_standardized_y: false,
            seed: None,
        }
    }
}

/// The shape the PLS3 split routes read: X of `n × p`, `n_splits` splits
/// of `resample::split_sizes(n, 1)` (k = 1), `n_cols` outcome columns,
/// unweighted, and whether the call holds the `split_nb` gate's all-row
/// Gram matrix (`gram_held`).
fn pls3_shape(
    n: usize,
    p: usize,
    n_cols: usize,
    n_splits: usize,
    gram_held: bool,
) -> crate::signal_test::SplitShape<'static> {
    let (n_train, n_test) = crate::resample::split_sizes(n, 1);
    crate::signal_test::SplitShape {
        n,
        p,
        n_splits,
        n_train,
        n_test,
        n_cols,
        ks: &[1],
        weighted: false,
        gram_held,
        gram_kept: false,
    }
}

/// The split design's index vectors on shape `s` (`resample::design_bytes`),
/// held by every split route.
fn design_bytes(s: &crate::signal_test::SplitShape<'_>) -> usize {
    crate::resample::design_bytes(s.n, s.n_train, s.n_test, s.n_splits)
}

/// One replicate column's copies of Y on a split of `n_tr` and `n_te` rows
/// with `q` columns: the permuted block (`resample::Columns::block`) and its
/// two [`centered_halves`].
#[allow(clippy::similar_names)] // n_tr, n_te
fn y_column_bytes(n_tr: usize, n_te: usize, q: usize) -> usize {
    use crate::resample::copy_bytes;
    copy_bytes(n_tr + n_te, q)
        .saturating_add(copy_bytes(n_tr, q))
        .saturating_add(copy_bytes(n_te, q))
}

/// One split of the Gram route (`dual_route::pls3_split_columns_r`) on
/// halves of `n_tr` and `n_te` rows, `q` Y columns, `n_cols` replicate
/// columns at `threads` threads: the split's `G` (`n_tr × n_tr`) and `M`
/// (`n_te × n_tr`), and per column that runs at once its copies of Y
/// ([`y_column_bytes`]) and `G·Ỹ_tr` (`n_tr × q`).
#[allow(clippy::similar_names)] // n_tr, n_te
pub(crate) fn gram_unit_bytes(
    n_tr: usize,
    n_te: usize,
    q: usize,
    n_cols: usize,
    threads: usize,
) -> usize {
    let column = y_column_bytes(n_tr, n_te, q).saturating_add(crate::resample::copy_bytes(n_tr, q));
    crate::resample::pair_blocks_bytes(std::iter::once((n_tr, n_te)), n_tr)
        .saturating_add(crate::signal_test::columns_bytes(column, n_cols, threads))
}

/// The Gram route's bytes on shape `s` with `q` Y columns at `threads`
/// threads: the all-row Gram matrix it reads, held for the call
/// (`signal_test::block_gram_bytes`, the gate's or one built), beside the
/// design's index vectors, and one split's [`gram_unit_bytes`].
fn gram_route_bytes(
    s: &crate::signal_test::SplitShape<'_>,
    q: usize,
    threads: usize,
) -> (usize, usize) {
    let call = crate::signal_test::block_gram_bytes(s.n).saturating_add(design_bytes(s));
    (
        call,
        gram_unit_bytes(s.n_train, s.n_test, q, s.n_cols, threads),
    )
}

/// `split_exact`'s route on X, called by [`pls3_confirmatory_test`] before
/// the loop builds a Gram matrix or a cell frame: the least work per
/// `√lanes` among the routes its rules allow (`route_select::select`), in
/// this order, with `(n_tr, n_te) = resample::split_sizes(n, 1)`,
/// `J = n_splits`, `B' = n_perm + 1`, `q = n_targets` and `p = n_features`,
/// in multiply-adds:
///
/// - `Nspace`, the Gram route ([`gram_route_allowed`]: `q ≥ 1`,
///   `0 < n_tr < p`): `G + J·B'·(q·n_tr² + n_te·n_tr + q²·n_tr + q³)`
///   with `G = n²·p/2`, or `0` when `gram_held` (the gate's matrix): per
///   column and split `G·Ỹ_tr`, `M·(Ỹ_tr·v)`, `Ỹ_tr'(GỸ_tr)` and the
///   `q × q` eigendecomposition.
/// - The cell route (`TestArm::Cells` on `Primal(Copy)`,
///   [`pls3_split_zbars_columns_cells`]), on a design with cells, unless
///   `signal_test::gram_routes_disabled`: [`cells_work`].
/// - `Primal(Copy)` ([`pls3_split_zbars_columns_primal`]):
///   `J·B'·((q·n_tr + n_te)·p + p·q·min(p, q))`, `X̃_tr'Ỹ_tr`, `X̃_te·u` and
///   the `p × q` SVD.
///
/// Each decomposition is counted as one multiply-add per entry of its cube
/// (both cost a few times that, about equally). The bytes are
/// [`gram_route_bytes`], [`cells_route_bytes`] and
/// [`primal_split_unit_bytes`], each beside the split design's index
/// vectors (`resample::design_bytes`). Every input is a shape or the budget, never
/// a data value. `n_perm` is the permutation count; the `+ 1` for the
/// observed column is part of the rule, not of the caller.
///
/// # Errors
/// `MemoryBudget` when no route fits one split.
#[allow(clippy::similar_names)]
pub(crate) fn pls3_split_exact_route(
    n: usize,
    n_features: usize,
    n_targets: usize,
    n_perm: usize,
    n_splits: usize,
    gram_held: bool,
) -> PlsKitResult<Choice<SplitRoute>> {
    use crate::signal_test::ReplicateRoute::{Nspace, Primal};
    let (p, q) = (n_features, n_targets);
    let s = pls3_shape(n, p, n_perm + 1, n_splits, gram_held);
    let gram = |threads| gram_route_bytes(&s, q, threads);
    let cells = |threads| cells_route_bytes(&s, q, threads);
    let primal = |threads| {
        let unit = primal_split_unit_bytes(s.n_train, s.n_test, p, q, s.n_cols, threads);
        (design_bytes(&s), unit)
    };
    let mut candidates = Vec::with_capacity(3);
    if gram_route_allowed(n, p, q) {
        candidates.push(Candidate {
            route: SplitRoute::rows(Nspace),
            work: gram_route_work(n, p, q, n_perm, n_splits, gram_held),
            bytes: &gram,
            group: None,
        });
    }
    if !crate::signal_test::gram_routes_disabled() && s.design().1 > 0 {
        candidates.push(Candidate {
            route: CELL_ROUTE,
            work: cells_work(&s, q),
            bytes: &cells,
            group: None,
        });
    }
    candidates.push(Candidate {
        route: SplitRoute::rows(Primal(PrimalSource::Copy)),
        work: primal_work(n, p, q, n_perm, n_splits),
        bytes: &primal,
        group: None,
    });
    select(n_splits, &candidates)
}

/// The cell route's [`SplitRoute`]: the primal refit's component stage on
/// training sums from the cells and test halves from their moments.
const CELL_ROUTE: SplitRoute = SplitRoute {
    arm: TestArm::Cells,
    ..SplitRoute::rows(crate::signal_test::ReplicateRoute::Primal(
        PrimalSource::Copy,
    ))
};

/// `split_nb`'s route on X, as [`pls3_split_exact_route`] weighs it with no
/// permutations: `Nspace` only when the call holds the gate's all-row Gram
/// matrix (`gram_held`) and [`gram_route_allowed`], at
/// [`gram_route_work`] with the matrix free; then the streamed body
/// (`Primal(Gather)`, [`pls3_split_lv_correlations_streamed`], which writes
/// no half of X), at [`primal_work`]. The bytes are [`gram_route_bytes`]
/// and [`streamed_unit_bytes`], each beside the design's index vectors; the
/// call frees the gate's matrix off the Gram route.
///
/// # Errors
/// `MemoryBudget` when no route fits one split.
#[allow(clippy::similar_names)]
pub(crate) fn pls3_split_nb_route(
    n: usize,
    p: usize,
    q: usize,
    n_splits: usize,
    gram_held: bool,
) -> PlsKitResult<Choice<ReplicateRoute>> {
    let s = pls3_shape(n, p, 1, n_splits, gram_held);
    let design = design_bytes(&s);
    let gram = |threads| gram_route_bytes(&s, q, threads);
    let streamed = |_: usize| (design, streamed_unit_bytes(p, q, s.n_train, s.n_test));
    let mut candidates = Vec::with_capacity(2);
    if gram_held && gram_route_allowed(n, p, q) {
        candidates.push(Candidate {
            route: ReplicateRoute::Nspace,
            work: gram_route_work(n, p, q, 0, n_splits, true),
            bytes: &gram,
            group: None,
        });
    }
    candidates.push(Candidate {
        route: ReplicateRoute::Primal(PrimalSource::Gather),
        work: primal_work(n, p, q, 0, n_splits),
        bytes: &streamed,
        group: None,
    });
    select(n_splits, &candidates)
}

/// `sv_perm`'s route on X, one unit for the call, by the rule of
/// [`pls3_split_exact_route`] with one split: `Nspace`
/// (`HeldOutGram::from_x`, which streams X in column blocks) when
/// [`gram_route_allowed`], at [`gram_route_work`], else `Primal(Copy)`, an
/// `n × p` X̃, at [`primal_work`]. Both hold Ỹ (`n × q`) and one column's
/// permuted block of it (`n × q`); the Gram route also `G` (`n × n`) and
/// the column's `G·Ỹ_π` (`n × q`), the X̃ route X̃ and the column's
/// `C = X̃'Ỹ_π` (`p × q`).
///
/// # Errors
/// `MemoryBudget` when neither fits.
pub(crate) fn pls3_sv_perm_route(
    n: usize,
    p: usize,
    q: usize,
    n_perm: usize,
) -> PlsKitResult<Choice<ReplicateRoute>> {
    use crate::resample::copy_bytes;
    let y = copy_bytes(n, q);
    let gram = |_: usize| {
        let call = crate::signal_test::block_gram_bytes(n).saturating_add(y);
        (call, y.saturating_add(copy_bytes(n, q)))
    };
    let x_tilde = |_: usize| {
        let call = copy_bytes(n, p).saturating_add(y);
        (call, y.saturating_add(copy_bytes(p, q)))
    };
    let mut candidates = Vec::with_capacity(2);
    if gram_route_allowed(n, p, q) {
        candidates.push(Candidate {
            route: ReplicateRoute::Nspace,
            work: gram_route_work(n, p, q, n_perm, 1, false),
            bytes: &gram,
            group: None,
        });
    }
    candidates.push(Candidate {
        route: ReplicateRoute::Primal(PrimalSource::Copy),
        work: primal_work(n, p, q, n_perm, 1),
        bytes: &x_tilde,
        group: None,
    });
    select(1, &candidates)
}

/// Whether [`pls3_split_exact_route`] takes the Gram route on this shape,
/// at the budget in force (test builds).
#[cfg(test)]
pub(crate) fn split_exact_takes_gram(
    n: usize,
    p: usize,
    q: usize,
    n_perm: usize,
    n_splits: usize,
    gram_held: bool,
) -> bool {
    pls3_split_exact_route(n, p, q, n_perm, n_splits, gram_held)
        .unwrap()
        .route
        .route
        == ReplicateRoute::Nspace
}

/// The shapes at which the Gram route is on the menu: `q ≥ 1` and
/// `0 < n_tr < p`.
fn gram_route_allowed(n: usize, n_features: usize, n_targets: usize) -> bool {
    let (n_tr, _) = crate::resample::split_sizes(n, 1);
    n_targets > 0 && n_tr > 0 && n_tr < n_features
}

/// The Gram route's work in [`pls3_split_exact_route`], in
/// multiply-adds: `G + J·B'·(q·n_tr² + n_te·n_tr + q²·n_tr + q³)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]
fn gram_route_work(
    n: usize,
    n_features: usize,
    n_targets: usize,
    n_perm: usize,
    n_splits: usize,
    gram_held: bool,
) -> f64 {
    let (n_tr, n_te) = crate::resample::split_sizes(n, 1);
    let (j, b, q) = (n_splits as f64, (n_perm + 1) as f64, n_targets as f64);
    let (t, s) = (n_tr as f64, n_te as f64);
    crate::dual_route::one_gram_cost(n, n_features, gram_held)
        + j * b * (q * t * t + s * t + q * q * t + q * q * q)
}

/// The primal route's work in [`pls3_split_exact_route`], in
/// multiply-adds: `J·B'·((q·n_tr + n_te)·p + p·q·min(p, q))`.
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]
fn primal_work(
    n: usize,
    n_features: usize,
    n_targets: usize,
    n_perm: usize,
    n_splits: usize,
) -> f64 {
    let (n_tr, n_te) = crate::resample::split_sizes(n, 1);
    let (j, b, q) = (n_splits as f64, (n_perm + 1) as f64, n_targets as f64);
    let (t, s, p) = (n_tr as f64, n_te as f64, n_features as f64);
    j * b * ((q * t + s) * p + p * q * p.min(q))
}

/// Confirmatory PLS3 omnibus test at LV1.
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - `y`: `(n_samples, n_targets)`
/// - `k`: must be `1`
///
/// # Errors
/// - `PlsKitError::DimensionMismatch` when row counts disagree
/// - `PlsKitError::InvalidArgument` for `k != 1` (`k = 0` with the
///   message every entry gives it, "k must be >= 1"), for any method other than
///   `split_exact`, `split_nb`, `sv_perm` or `auto`, for `n_splits < 2`, for
///   `n_perm < 1`, or for `n < k + 5` on the split methods (the split floor
///   `draw_splits` enforces; `sv_perm` draws no splits)
/// - `PlsKitError::KExceedsMax` when `x` has 0 columns or `y` has 0 columns
///   (mirrors the `k_max = n_features.min(n_targets)` check `pls3_fit` itself
///   makes — this call always requests `k = 1`, so `k_max = 0` on either
///   empty block trips it)
/// - `PlsKitError::NonFiniteInput` when X or Y contains NaN/inf
///
/// A failed inner fit on a split (SVD non-convergence, or any other
/// `pls3_fit` error) is not propagated: that split's `r` is `0.0`, counted
/// in `n_failed_splits` (`signal_test::zbars_from_rows`).
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn pls3_confirmatory_test(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    opts: Pls3ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    crate::fit::with_thread_limit(|| pls3_confirmatory_test_impl(x, y, k, opts))
}

/// Confirmatory PLS3 omnibus test at LV1 on the Gram matrix of X in place
/// of X: [`pls3_confirmatory_test`] with every split's kernels taken from
/// `gram.gram`, which is centred over its rows (so a row subset from
/// `GramMatrixOutput::subset` is accepted) and checked on its entries.
/// `pre_standardized_x` is read from `gram`; Y is passed as data. Both
/// methods run on the Gram route, and the `split_nb` gate reads the stable
/// rank of `G`. On `gram_matrix(x, flag)` the result equals the X call on
/// `x` with `pre_standardized_x = flag` and the same seed to the crate's
/// route tolerance: the X call can take the primal route, which at the
/// truncation floor can differ by O(1).
/// Except the `sv_perm` statistic for x beyond `2^±256`, which is that of
/// `x/c` (`c` the power of two `gram_matrix` does not store).
///
/// # Errors
/// What [`pls3_confirmatory_test`] returns, and `PlsKitError::InvalidArgument`
/// when `gram.gram` has another row count than `y`, or
/// `opts.pre_standardized_x` is set (it is read from `gram`); `ShapeMismatch` when `gram.gram` is not square;
/// `NonFiniteInput` when it is not finite; `InvalidInput` when the centred
/// matrix has a negative diagonal entry, is not symmetric or breaks
/// Cauchy-Schwarz, beyond rounding.
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
pub fn pls3_confirmatory_test_gram(
    gram: &GramMatrixOutput,
    y: MatRef<'_, f64>,
    k: usize,
    opts: Pls3ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let n = gram.gram.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::InvalidArgument(format!(
            "gram is {n} x {}, Y has {} rows: gram needs one row and column per row of Y",
            gram.gram.ncols(),
            y.nrows()
        )));
    }
    crate::signal_test::check_gram_flag(opts.pre_standardized_x)?;
    crate::fit::with_thread_limit(|| pls3_test_body(Pls3Input::Gram(gram), y, k, opts))
}

/// Where a PLS3 test reads X: X itself, or the centred all-row Gram matrix
/// of `X̃` from a caller's `GramMatrixOutput`.
#[derive(Clone, Copy)]
enum Pls3X<'a> {
    X(MatRef<'a, f64>),
    Gram(&'a HeldOutGram),
}

/// A PLS3 test's input before its argument checks: X, or a caller's
/// `GramMatrixOutput` not yet centred into a copy.
#[derive(Clone, Copy)]
enum Pls3Input<'a> {
    X(MatRef<'a, f64>),
    Gram(&'a GramMatrixOutput),
}

impl<'a> Pls3Input<'a> {
    /// `(n, n_features)` of the input.
    fn shape(self) -> (usize, usize) {
        match self {
            Self::X(x) => (x.nrows(), x.ncols()),
            Self::Gram(g) => (g.gram.nrows(), g.n_features),
        }
    }

    /// The source the test reads, after its argument checks: X as given,
    /// or the caller's Gram matrix centred into a copy (`n × n`, divided by
    /// a power of four under `pre_standardized` when far from unit scale,
    /// `HeldOutGram::from_gram`), checked against the budget before it is
    /// written and held for the call (`fit::hold_bytes`) while `held` lives.
    fn open<'b>(
        self,
        held: &'b mut Option<(HeldOutGram, crate::fit::HeldBytes)>,
    ) -> PlsKitResult<Pls3X<'b>>
    where
        'a: 'b,
    {
        match self {
            Self::X(x) => Ok(Pls3X::X(x)),
            Self::Gram(gram) => {
                let n = gram.gram.nrows();
                if gram.gram.ncols() != n {
                    return Err(PlsKitError::ShapeMismatch(format!(
                        "gram must be square, got {n} x {}",
                        gram.gram.ncols()
                    )));
                }
                crate::fit::check_finite_mat(gram.gram.as_ref())?;
                let bytes = crate::resample::copy_bytes(n, n);
                crate::resample::units_at_once(1, bytes, 0)?;
                let guard = crate::fit::hold_bytes(bytes);
                let g = HeldOutGram::from_gram(
                    gram.gram.to_owned(),
                    gram.n_features,
                    gram.pre_standardized,
                )?;
                Ok(Pls3X::Gram(&held.insert((g, guard)).0))
            }
        }
    }
}

/// Body of [`pls3_confirmatory_test`], on the caller's pool.
pub(crate) fn pls3_confirmatory_test_impl(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    opts: Pls3ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    pls3_test_body(Pls3Input::X(x), y, k, opts)
}

/// Body of [`pls3_confirmatory_test`] and [`pls3_confirmatory_test_gram`]:
/// one dispatch for both, X read as `src` gives it.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
#[allow(clippy::too_many_lines)] // two method bodies plus the gate, in one dispatch
fn pls3_test_body(
    src: Pls3Input<'_>,
    y: MatRef<'_, f64>,
    k: usize,
    opts: Pls3ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let (n, p) = src.shape();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, p),
            y: y.nrows(),
        });
    }
    if let Pls3Input::X(x) = src {
        crate::fit::check_finite_mat(x)?;
    }
    crate::fit::check_finite_mat(y)?;

    // k = 0 gets the message every other entry gives it, not the k = 1
    // restriction below.
    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    if k != 1 {
        return Err(PlsKitError::InvalidArgument(format!(
            "pls3_confirmatory_test supports k = 1 only (got k={k}): above LV1 neither the \
             component ordering nor the individual directions need survive to the test half \
             when singular values are close, and whether the statistic is per-component or \
             subspace-level is undecided"
        )));
    }

    // Same shape-vs-k check `pls3_fit` makes internally (`k_max =
    // n_features.min(n_targets)`). Catching it here, before the per-split
    // loop, is what keeps a 0-column X or Y out of the per-split fits at
    // all: `pls3::pls3_lv1_prestd` does not repeat it. The per-split
    // fail-soft convention below is for genuine per-split degeneracy, not
    // malformed input reaching every split alike.
    let k_max = p.min(y.ncols());
    if k > k_max {
        return Err(PlsKitError::KExceedsMax { k, k_max });
    }

    // `n_perm` doubles as the mode flag from here down: `Some` means a
    // permutation reference runs — either because split_exact was asked for,
    // or because the gate below rerouted split_nb into it — and `None` means
    // the NB t-reference runs. `Auto` carries its `n_perm` into the floor
    // checks below and is resolved after them.
    let (mut n_perm, n_splits, force) = match opts.args {
        ConfirmatoryArgs::SplitExact { n_perm, n_splits }
        | ConfirmatoryArgs::Auto { n_perm, n_splits } => (Some(n_perm), n_splits, false),
        ConfirmatoryArgs::SplitNb { n_splits, force } => (None, n_splits, force),
        ConfirmatoryArgs::SvPerm { n_perm, rotate } => {
            return pls3_sv_perm(src, y, n_perm, rotate, &opts);
        }
        other => {
            return Err(PlsKitError::InvalidArgument(format!(
                "pls3_confirmatory_test supports test_method='split_exact', 'split_nb', \
                 'sv_perm' or 'auto' (got '{}'): 'raw_perm' needs a CV statistic PLS3 does not have (there is no \
                 pls3_predict); 'score' is not implemented (its symmetric analog, an \
                 RV-type test on ‖X'Y‖_F², tests a different estimand); 'e' needs a \
                 generative model that symmetric cross-decomposition does not supply",
                other.method().as_str()
            )));
        }
    };
    check_split_exact_counts(n_perm, n_splits)?;
    // The split floor before any budget check (the gate's, the routes'), so
    // a bad argument is never a `MemoryBudget`.
    check_split_rows(n, k)?;
    let mut held_gram = None;
    let src = src.open(&mut held_gram)?;

    // The column scales every split divides by, once for the call. Y's comes
    // from its rows in their original order, before any permutation. X's is
    // already in a caller's Gram matrix.
    let x_scale = match src {
        Pls3X::X(x) => crate::linalg::held_out_scale(x, None, opts.pre_standardized_x),
        Pls3X::Gram(_) => Col::<f64>::zeros(0),
    };
    let y_scale = crate::linalg::held_out_scale(y, None, opts.pre_standardized_y);
    let (xsc, ysc) = (x_scale.as_ref(), y_scale.as_ref());

    // The `split_nb` auto-gate, on X only. Y is deliberately never gated: q
    // is 3-10 in ordinary PLSC use, so a stable-rank floor of 3 on Y would
    // fire on nearly every design. The thresholds are PLS1's, calibrated on
    // single-block designs and not re-derived for a two-block one — see
    // `signal_test::SPLIT_NB_GATE_MIN_N_EFF` for their provenance.
    let mut stable_rank_out: Option<f64> = None;
    // The gate's all-row Gram matrix on wide X, which the Gram route reads.
    let mut gate_gram = None;
    // No weights on this family, so Kish n_eff is exactly the row count.
    let n_eff = n as f64;
    if let ConfirmatoryArgs::Auto { .. } = opts.args {
        // The X-only resolution, with the unweighted n_eff and no width
        // ceiling (PLS3's `split_exact` always refits); it takes
        // `split_exact` on every PLS3 design. Were it to resolve to
        // `split_nb`, that would run without a second gate.
        let (method, sr) = match src {
            Pls3X::X(x) => {
                let (method, sr, gram) =
                    resolve_auto_on_x(x, None, n_eff, false, true, false, false, xsc)?;
                gate_gram = gram;
                (method, sr)
            }
            Pls3X::Gram(g) => {
                let (method, sr, _) = resolve_auto(&AutoInput {
                    n,
                    p,
                    n_eff,
                    no_refit: false,
                    pls3: true,
                    covariates: false,
                    multi_k: false,
                    // Unchecked: `pls3_exact` answers first on PLS3, so the rank is never read.
                    stable_rank: &|| g.stable_rank(),
                });
                (method, sr)
            }
        };
        stable_rank_out = sr;
        if method == ConfirmatoryMethod::SplitNb {
            n_perm = None;
        }
    } else if n_perm.is_none() {
        let (reroute, sr) = match src {
            Pls3X::X(x) => {
                let gate = resolve_split_nb(x, None, n_eff, force, xsc)?;
                gate_gram = gate.gram;
                (gate.reroute, gate.stable_rank)
            }
            Pls3X::Gram(_) if force => (false, None),
            Pls3X::Gram(g) => {
                let sr = g.checked_stable_rank()?;
                (split_nb_gate_fires(p, n_eff, sr), Some(sr))
            }
        };
        stable_rank_out = sr;
        if reroute {
            n_perm = Some(SPLIT_NB_REROUTE_N_PERM);
        }
    }

    let seed_used = crate::rng::resolve_seed(opts.seed, n)?;

    // Route choice, before the splits are drawn, since every route holds
    // their index vectors: `pls3_split_nb_route` without permutations,
    // `pls3_split_exact_route` with them; a caller's Gram matrix runs the
    // Gram route always.
    let held = gate_gram.is_some();
    let (nb_choice, exact_choice) = match (src, n_perm) {
        (Pls3X::Gram(_), _) => (None, None),
        (Pls3X::X(_), None) => {
            let c = pls3_split_nb_route(n, p, y.ncols(), n_splits, held)?;
            (Some(c), None)
        }
        (Pls3X::X(_), Some(b)) => {
            let q = y.ncols();
            let c = pls3_split_exact_route(n, p, q, b, n_splits, held)?;
            (None, Some(c))
        }
    };
    // Only the Gram route reads the gate's G; the others free it here,
    // before the splits are drawn.
    let gram_route = nb_choice
        .as_ref()
        .map(|c| c.route == ReplicateRoute::Nspace)
        .or_else(|| {
            exact_choice
                .as_ref()
                .map(|c| c.route.route == ReplicateRoute::Nspace)
        })
        .unwrap_or(false);
    let gate_gram = gate_gram.filter(|_| gram_route);

    // The J splits are drawn once and held fixed across all B permutation
    // replicates: redrawing per replicate would fold split-to-split scatter
    // into the null. The cell route also reads the design's cells.
    // A caller's Gram matrix runs the Gram route with no route choice: its
    // design and one split are checked here, before the splits are drawn.
    let gram_src_call = match src {
        Pls3X::Gram(_) => {
            let s = pls3_shape(n, p, n_perm.map_or(1, |b| b + 1), n_splits, false);
            let threads = rayon::current_num_threads();
            let unit = gram_unit_bytes(s.n_train, s.n_test, y.ncols(), s.n_cols, threads);
            let call = design_bytes(&s);
            crate::resample::units_at_once(n_splits, call, unit)?;
            call
        }
        Pls3X::X(_) => 0,
    };
    let design = draw_design(n, k, n_splits, seed_used)?;
    let splits = design.splits.as_slice();

    let (n_tr, n_te) = crate::resample::split_sizes(n, k);

    let Some(n_perm) = n_perm else {
        // No permutation loop: the J held-out correlations go straight to the
        // NB t-reference. The statistic is the one split_exact reports —
        // `tanh(z̄)` over the same fixed splits — so only the reference
        // changes. A caller's Gram matrix gives each split on the Gram
        // route, and so does the gate's where `pls3_split_nb_route` takes
        // that route; else the blocks are read from X block by block.
        // At a shared seed the two methods
        // agree on the statistic to the route tolerance (~1e-10 relative).
        let call = nb_choice.map_or(gram_src_call, |c| c.call_bytes);
        let held = match src {
            Pls3X::Gram(g) => Some(g),
            Pls3X::X(_) => gate_gram.as_ref(),
        };
        let r_obs = match (src, held) {
            (Pls3X::X(x), None) => {
                pls3_split_lv_correlations_streamed(x, y, splits, xsc, ysc, call)?
            }
            (Pls3X::Gram(_), None) => unreachable!("a caller's Gram matrix is always held"),
            // Each split's correlation on the Gram route.
            (_, Some(g)) => {
                let obs = crate::resample::Columns {
                    y,
                    seed: seed_used,
                    n_perm: 0,
                    null_map: None,
                    w: None,
                };
                let unit = gram_unit_bytes(n_tr, n_te, y.ncols(), 1, rayon::current_num_threads());
                let r = crate::resample::map_units(splits.len(), call, unit, |j| {
                    crate::dual_route::pls3_split_columns_r(g, &obs, &splits[j], ysc)[0]
                })?;
                Col::<f64>::from_fn(r.len(), |j| r[j])
            }
        };
        let (p, statistic, _t, _df) = nb_test(&r_obs, n_tr, n_te);
        let n_failed_splits = Some(crate::signal_test::exact_zeros(r_obs.as_ref()));
        return Ok(ConfirmatoryTestOutput {
            pvalue: p,
            statistic: Some(statistic),
            test_method: ConfirmatoryMethod::SplitNb.as_str().to_owned(),
            k: vec![k],
            pvalues_by_k: std::collections::BTreeMap::from([(k, p)]),
            statistics_by_k: std::collections::BTreeMap::from([(k, statistic)]),
            n_perm: None,
            n_splits: Some(n_splits),
            n_folds: None,
            force: Some(force),
            rotate: None,
            seed: seed_used,
            ci: None,
            n_eff,
            rho_hat: nb_rho_hat(&r_obs, n_te),
            stable_rank: stable_rank_out,
            n_covariates: None,
            n_failed_splits,
            n_failed_perms: None,
            n_failed_folds: None,
        });
    };

    // `pls3_split_exact_route` owns the rule (chosen above). Every route
    // gets the same splits and the same permutations, so the choice is
    // invisible in the output up to the crate's bit-near tolerance, except
    // on a column at the truncation floor (see "Truncation" on
    // `dual_route::pls3_split_zbars_columns`).
    let choice = exact_choice;
    let route = choice.as_ref().map(|c| c.route);
    let call = choice.as_ref().map_or(gram_src_call, |c| c.call_bytes);

    // The B null columns, shared by both routes through `resample::Columns`,
    // the site every route takes its null columns from. They permute the
    // ROWS of Y as units against X: that breaks the cross-block association
    // while leaving Y's within-row structure intact. Y's all-row scale was
    // taken above from its rows in their original order, so every
    // replicate divides by the same scale; only the training means move with
    // the permutation.
    //
    // Column `b` is the rows of Y permuted by
    // `rng::null_permutation(n, b, seed_used)`, a function of `(seed_used, b)`
    // alone, whichever route runs and however Rayon schedules it.
    let cols = crate::resample::Columns {
        y,
        seed: seed_used,
        n_perm,
        null_map: None,
        w: None,
    };

    // Every route returns `n_perm + 1` per-column z̄ values (column 0 is the
    // observed Y, columns `1..=n_perm` are the permutation nulls) and the
    // number of splits whose observed `r` is exactly 0.
    let (z, n_failed) = match (src, route) {
        (Pls3X::X(x), Some(r)) if r.route == ReplicateRoute::Nspace => {
            // One all-row Gram matrix for the call, the gate's when it holds one.
            let all = gate_gram.unwrap_or_else(|| HeldOutGram::from_x(x, None, xsc));
            crate::dual_route::pls3_split_zbars_columns(&all, &cols, splits, ysc, call)?
        }
        (Pls3X::X(x), Some(CELL_ROUTE)) => {
            pls3_split_zbars_columns_cells(x, &cols, &design, xsc, ysc, call)?
        }
        (Pls3X::X(x), _) => pls3_split_zbars_columns_primal(x, &cols, splits, xsc, ysc, call)?,
        (Pls3X::Gram(g), _) => {
            crate::dual_route::pls3_split_zbars_columns(g, &cols, splits, ysc, call)?
        }
    };
    let (z_bar_obs, null_zbars) = (z[0], &z[1..]);
    let (p, n_failed_perms) = perm_pvalue("split-half statistic", z_bar_obs, null_zbars)?;
    Ok(ConfirmatoryTestOutput {
        pvalue: p,
        statistic: Some(z_bar_obs.tanh()),
        test_method: ConfirmatoryMethod::SplitExact.as_str().to_owned(),
        k: vec![k],
        pvalues_by_k: std::collections::BTreeMap::from([(k, p)]),
        statistics_by_k: std::collections::BTreeMap::from([(k, z_bar_obs.tanh())]),
        n_perm: Some(n_perm),
        n_splits: Some(n_splits),
        n_folds: None,
        force: None,
        rotate: None,
        seed: seed_used,
        ci: None,
        n_eff,
        rho_hat: None,
        // `Some` only when split_nb was requested and rerouted here: it is
        // what the gate saw.
        stable_rank: stable_rank_out,
        n_covariates: None,
        n_failed_splits: Some(n_failed),
        n_failed_perms: Some(n_failed_perms),
        n_failed_folds: None,
    })
}

/// PLS3 `sv_perm`: its refusals, scales and seed, then the runner, with the
/// output fields `split_exact` fills (no splits, no gate), and, under
/// `pre_standardized_x` / `_y`, `d₁` on the scale of the blocks as passed.
fn pls3_sv_perm(
    src: Pls3Input<'_>,
    y: MatRef<'_, f64>,
    n_perm: usize,
    rotate: bool,
    opts: &Pls3ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let (x, n_features) = match src {
        Pls3Input::X(x) => (Some(x), x.ncols()),
        Pls3Input::Gram(g) => (None, g.n_features),
    };
    // `n_splits` is not read; 2 is the floor's own minimum.
    check_split_exact_counts(Some(n_perm), 2)?;
    let mut held_gram = None;
    let src = src.open(&mut held_gram)?;
    let x_scale = x.map_or_else(
        || Col::<f64>::zeros(0),
        |x| crate::linalg::held_out_scale(x, None, opts.pre_standardized_x),
    );
    let y_scale = crate::linalg::held_out_scale(y, None, opts.pre_standardized_y);
    // The powers of two the blocks are divided by: `d₁` is multiplied back
    // by each, one at a time, since their product can leave the `f64` range.
    let c_x = match src {
        Pls3X::Gram(g) => g.div(),
        Pls3X::X(_) if opts.pre_standardized_x => x_scale[0],
        Pls3X::X(_) => 1.0,
    };
    let c_y = if opts.pre_standardized_y {
        y_scale[0]
    } else {
        1.0
    };
    let seed_used = crate::rng::resolve_seed(opts.seed, y.nrows())?;
    // `pls3_sv_perm_route` decides between the Gram route, where no `n × p`
    // copy of X̃ is held (`HeldOutGram::from_x` streams X in column blocks),
    // and X̃, before either is built.
    let held = match src {
        Pls3X::X(x) => {
            let route = pls3_sv_perm_route(y.nrows(), n_features, y.ncols(), n_perm)?.route;
            (route == ReplicateRoute::Nspace)
                .then(|| HeldOutGram::from_x(x, None, x_scale.as_ref()))
        }
        Pls3X::Gram(_) => None,
    };
    let src = held.as_ref().map_or(src, Pls3X::Gram);
    let (p, statistic, n_failed_perms) = run_sv_perm_pls3(
        src,
        y,
        x_scale.as_ref(),
        y_scale.as_ref(),
        n_features,
        n_perm,
        rotate,
        seed_used,
    )?;
    let statistic = statistic * c_x * c_y;
    Ok(ConfirmatoryTestOutput {
        pvalue: p,
        statistic: Some(statistic),
        test_method: ConfirmatoryMethod::SvPerm.as_str().to_owned(),
        k: vec![1],
        pvalues_by_k: std::collections::BTreeMap::from([(1, p)]),
        statistics_by_k: std::collections::BTreeMap::from([(1, statistic)]),
        n_perm: Some(n_perm),
        n_splits: None,
        n_folds: None,
        force: None,
        rotate: Some(rotate),
        seed: seed_used,
        ci: None,
        // No weights on this family, so Kish n_eff is exactly the row count.
        n_eff: y.nrows() as f64,
        rho_hat: None,
        stable_rank: None,
        n_covariates: None,
        n_failed_splits: None,
        n_failed_perms: Some(n_failed_perms),
        n_failed_folds: None,
    })
}

/// `(p-value, d₁, n)` of PLS3 `sv_perm` (McIntosh & Lobaugh 2004), `n`
/// the number of nulls that are not finite.
///
/// Every value, observed or permuted, comes from the `q × q` matrix
/// `M = ỸᵀX̃X̃ᵀỸ`: its eigenvalues are the squared singular values of
/// `X̃ᵀỸ` and its eigenvectors the Y-side singular vectors V, so
/// `d₁² = λ_max(ỸᵀX̃X̃ᵀỸ)`. The raw form forms `C = X̃ᵀỸ_π` (`p × q`) and
/// `M = CᵀC`; the gram form forms `M = Ỹ_πᵀ G Ỹ_π` without X. Column `c`
/// of `resample::Columns` permutes the rows of Ỹ; X̃ and both scales stay
/// on their rows. Without `rotate` the null value is `sqrt(λ_max(M_π))`;
/// with it, see [`sv_null`].
///
/// p is [`perm_pvalue`]'s. An eigen or SVD failure is `PlsKitError::Internal`.
///
/// `d₁` here is that of X̃ and Ỹ, the blocks divided by `x_scale` and
/// `y_scale` (on the gram form, X̃ is the one behind `G`). Under
/// `pre_standardized_x` / `_y` a block far from unit scale is divided by a
/// power of two, which [`pls3_sv_perm`] multiplies back, so under the
/// flags the reported `d₁` is on the scale of the blocks as passed.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
#[allow(clippy::doc_markdown)] // "McIntosh"
fn run_sv_perm_pls3(
    src: Pls3X<'_>,
    y: MatRef<'_, f64>,
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
    n_features: usize,
    n_perm: usize,
    rotate: bool,
    seed: u64,
) -> PlsKitResult<(f64, f64, usize)> {
    let seq = faer::Par::Seq;
    let all: Vec<usize> = (0..y.nrows()).collect();
    let ys = crate::linalg::center_rows_fro(y, &all, None, None, y_scale, seq).0;
    // X̃ once for the call. The raw form reaches here only where
    // `pls3_sv_perm_route` declines the Gram route, which counts X̃.
    let xs = match src {
        Pls3X::X(x) => Some(crate::linalg::center_rows_fro(x, &all, None, None, x_scale, seq).0),
        Pls3X::Gram(_) => None,
    };
    let m_of = |yc: MatRef<'_, f64>| match (src, &xs) {
        (Pls3X::X(_), Some(xs)) => {
            let c = crate::linalg::mat_mul(xs.transpose(), yc, seq);
            crate::linalg::mat_mul(c.transpose(), c.as_ref(), seq)
        }
        (Pls3X::Gram(g), _) => {
            let gy = crate::linalg::mat_mul(g.g(), yc, seq);
            crate::linalg::mat_mul(yc.transpose(), gy.as_ref(), seq)
        }
        (Pls3X::X(_), None) => unreachable!("X̃ is formed for the raw form"),
    };
    let q = ys.ncols();
    let (lambda, v) = crate::linalg::self_adjoint_eigen(m_of(ys.as_ref()).as_ref(), seq)
        .map_err(|e| PlsKitError::Internal(format!("sv_perm eigendecomposition: {e:?}")))?;
    let d1 = sqrt_psd(lambda[q - 1]);
    // pyls keeps `min(p, q)` singular vectors on each side; at `p < q` the
    // eigenvectors of M beyond rank p are arbitrary and are dropped.
    let r = n_features.min(q);
    let v_obs = Mat::<f64>::from_fn(q, r, |i, j| v[(i, q - 1 - j)]);

    let cols = crate::resample::Columns {
        y: ys.as_ref(),
        seed,
        n_perm,
        null_map: None,
        w: None,
    };
    let nulls = (1..cols.len())
        .map(|c| {
            let m = m_of(cols.block(c).as_ref());
            sv_null(m.as_ref(), rotate.then_some(v_obs.as_ref()))
        })
        .collect::<PlsKitResult<Vec<f64>>>()?;
    let (p, n_failed) = perm_pvalue("leading singular value", d1, &nulls)?;
    Ok((p, d1, n_failed))
}

/// The LV1 null value of one permuted `M_π` (`q × q`): `sqrt(λ_max(M_π))`
/// when `v_obs` is `None`, else the rotated value against the observed
/// top-`r` Y-side vectors `v_obs` (`q × r`, descending).
///
/// Rotation (pyls `rotate=True`): with `V_π` the top-`r` eigenvectors of
/// `M_π` and `svd(V_obsᵀV_π) = N S Pᵀ`, `R = P Nᵀ` is the `r × r`
/// orthogonal Procrustes rotation with `V_π R ≈ V_obs`, and the null value
/// is `‖D_π R[:, 0]‖ = sqrt(Σ_j λ_j^π R_{j0}²)`. Here `R` comes as `U Wᵀ`
/// from `thin_svd(V_πᵀV_obs) = U Σ Wᵀ`, the same matrix. Since
/// `‖D_π r₁‖ ≤ d₁^π` for a unit `r₁` and the observed data take `R = I`,
/// the rotated p is never above the unrotated one at the same seed.
///
/// At `r = q` both bases are square orthogonal, so `V_πᵀV_obs` is
/// orthogonal, its Procrustes rotation is itself, and
/// `Σ_j λ_j (v_jᵀv₁)² = v₁ᵀ M_π v₁`: no eigendecomposition or SVD runs.
/// The value is then sign-free in `v₁`, and at `r < q` sign-free in every
/// column of either basis, since `R`'s rows and columns flip with them.
#[allow(clippy::many_single_char_names)]
fn sv_null(m: MatRef<'_, f64>, v_obs: Option<MatRef<'_, f64>>) -> PlsKitResult<f64> {
    let seq = faer::Par::Seq;
    let q = m.nrows();
    let evd = |e: faer::linalg::evd::EvdError| {
        PlsKitError::Internal(format!("sv_perm eigendecomposition: {e:?}"))
    };
    match v_obs {
        None => {
            let lambda = crate::linalg::self_adjoint_eigenvalues(m, seq).map_err(evd)?;
            Ok(sqrt_psd(lambda[q - 1]))
        }
        Some(v) if v.ncols() == q => {
            let v1 = v.col(0);
            let mv = crate::linalg::mat_vec(m, v1, seq);
            Ok(sqrt_psd((0..q).map(|i| v1[i] * mv[i]).sum()))
        }
        Some(v) => {
            let r = v.ncols();
            let (lambda, u) = crate::linalg::self_adjoint_eigen(m, seq).map_err(evd)?;
            let v_pi = Mat::<f64>::from_fn(q, r, |i, j| u[(i, q - 1 - j)]);
            let a = crate::linalg::mat_mul(v_pi.transpose(), v, seq);
            let svd = crate::linalg::thin_svd(a.as_ref(), seq)
                .map_err(|e| PlsKitError::Internal(format!("sv_perm rotation SVD: {e:?}")))?;
            // R[j, 0] = Σ_k U[j, k]·W[0, k].
            let d2 = (0..r)
                .map(|j| {
                    let r_j0: f64 = (0..r).map(|k| svd.u[(j, k)] * svd.v[(0, k)]).sum();
                    lambda[q - 1 - j] * r_j0 * r_j0
                })
                .sum();
            Ok(sqrt_psd(d2))
        }
    }
}

/// `sqrt` of a quantity that is non-negative in exact arithmetic: a zero
/// eigenvalue of a rank-deficient `M` (a constant or all-zero Ỹ, or wide X
/// past its rank) can round to a tiny negative, which reads as 0. NaN
/// passes through, so a non-finite null still counts as an exceedance.
fn sqrt_psd(v: f64) -> f64 {
    if v < 0.0 { 0.0 } else { v.sqrt() }
}

/// The large buffers of one split on the primal route when `n_cols`
/// replicate columns of it are fitted: the materialized halves of X
/// (`n_tr` and `n_te` rows, `resample::pair_blocks_bytes`) and, per column
/// that runs at once (`signal_test::columns_bytes`, at `threads` threads),
/// one LV1 fit's buffers (`pls3::lv1_bytes`) and its copies of Y
/// ([`y_column_bytes`]).
#[allow(clippy::similar_names)] // n_tr, n_te
fn primal_split_unit_bytes(
    n_tr: usize,
    n_te: usize,
    p: usize,
    q: usize,
    n_cols: usize,
    threads: usize,
) -> usize {
    crate::resample::pair_blocks_bytes(std::iter::once((n_tr, n_te)), p).saturating_add(
        crate::signal_test::columns_bytes(
            crate::pls3::lv1_bytes(p, q).saturating_add(y_column_bytes(n_tr, n_te, q)),
            n_cols,
            threads,
        ),
    )
}

/// The sizes of the first of `splits` (every split has the same).
fn split_lens(splits: &[SplitIdx]) -> (usize, usize) {
    splits
        .first()
        .map_or((0, 0), |sp| (sp.tr.len(), sp.te.len()))
}

/// One split of the streamed body ([`pls3_split_r_streamed`]) on halves of
/// `n_tr` and `n_te` rows: two stream blocks, one LV1 fit's buffers (`A`,
/// its SVD's `U`, `V` and scratch), the centred halves of Y and the
/// summed held-out scores `T` (`n_te × q`).
#[allow(clippy::similar_names)] // n_tr, n_te
fn streamed_unit_bytes(p: usize, q: usize, n_tr: usize, n_te: usize) -> usize {
    use crate::resample::copy_bytes;
    crate::pls3::lv1_bytes(p, q)
        .saturating_add(crate::linalg::stream_blocks_bytes(n_tr, n_te))
        .saturating_add(copy_bytes(n_tr, q))
        .saturating_add(copy_bytes(n_te, q).saturating_mul(2))
}

/// Held-out LV correlation on each supplied split, computed from an honest
/// per-split PLS3 refit (the primal route).
///
/// Takes the splits rather than drawing them so the permutation loop can
/// hold one set fixed across all B replicates. Nothing here consumes
/// randomness: given `(x, y, split)` the fit and the r are deterministic.
/// The J splits run in batches of `resample::map_units` by the memory
/// budget, one unit being the split's materialized halves of X and one LV1
/// fit's buffers ([`primal_split_unit_bytes`] with one column; the halves
/// of Y are counted as small); the result does not depend on the batch
/// size.
///
/// Both halves of both blocks are [`centered_halves`]: centred with the
/// **training half's** means and divided by the block's all-row scale
/// (`x_scale`, `y_scale`). `Ỹ_te v₁` mixes Y columns, so the test half
/// must sit on the scale `v₁` was estimated on. The fit and the score on
/// those blocks are [`pls3_split_column_r_primal`], the same body every
/// `split_exact` route evaluates per `(split, replicate column)`.
#[cfg(test)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn pls3_split_lv_correlations(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[SplitIdx],
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
    call_bytes: usize,
) -> PlsKitResult<Col<f64>> {
    let per_split = |sp: &SplitIdx| -> f64 {
        let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
        let (xs_tr, xs_te) = centered_halves(x, tr, te, x_scale);
        let (ys_tr, ys_te) = centered_halves(y, tr, te, y_scale);

        pls3_split_column_r_primal(
            xs_tr.as_ref(),
            xs_tr.norm_l2(),
            xs_te.as_ref(),
            ys_tr.as_ref(),
            ys_te.as_ref(),
        )
    };

    // The splits run in batches by the memory budget, collected in split order.
    let (n_tr, n_te) = split_lens(splits);
    let unit = primal_split_unit_bytes(
        n_tr,
        n_te,
        x.ncols(),
        y.ncols(),
        1,
        rayon::current_num_threads(),
    );
    let r_vec =
        crate::resample::map_units(splits.len(), call_bytes, unit, |j| per_split(&splits[j]))?;
    Ok(Col::<f64>::from_fn(r_vec.len(), |i| r_vec[i]))
}

/// Rows `tr` and `te` of one block under the held-out rule: both centred
/// with the means of rows `tr` and divided by the block's all-row column
/// `scale`. The one expression every PLS3 split route builds its
/// materialized halves with, so they agree to the bit.
pub(crate) fn centered_halves(
    a: MatRef<'_, f64>,
    tr: &[usize],
    te: &[usize],
    scale: ColRef<'_, f64>,
) -> (Mat<f64>, Mat<f64>) {
    let (train, mean, _) = crate::linalg::center_rows_fro(a, tr, None, None, scale, faer::Par::Seq);
    let held = crate::linalg::standardize_apply_rows(a, te, mean.as_ref(), scale, None);
    (train, held)
}

/// Held-out LV1 correlation of one split on dense blocks, read from `X`
/// one block of columns at a time.
///
/// # What it computes
/// With `a` and `c` the training and held-out blocks of a column range,
/// centred with the training rows' means and divided by `x_scale`
/// (`linalg::centered_blocks`; Y likewise by `y_scale`), one pass forms
/// `A = X̃_tr'Ỹ_tr` (`p × q`, a block of rows at a time) and sums
/// `T = X̃_te A` (`n_te × q`) over the blocks. The
/// component stage of `pls3::pls3_lv1_prestd` gives `v₁` and `σ₁` from `A`,
/// and since `A v₁ = σ₁ u₁` with `σ₁ > 0`, `T v₁` is `X̃_te u₁` times `σ₁`:
/// the correlation with `Ỹ_te v₁` does not see the factor, and `u₁` is
/// never applied to `X̃_te`.
///
/// # Against the materialized fit
/// Each row block of `A` is one product of the columns the materialized
/// fit multiplies, so it can differ from that fit's one-call `X̃_tr'Ỹ_tr`
/// only where faer evaluates a product of another shape differently. `T`
/// is summed over blocks, where the materialized fit applies `u₁` to
/// `X̃_te` in one product, and `‖X̃_tr‖_F` comes from the column moments,
/// not from `norm_l2`. So the floor and, apart from its factor `σ₁`, the
/// X score can differ from the materialized fit's in the last bits, and
/// where `A` differs so can `σ₁`, `v₁` and the Y score. The truncation rule
/// is the materialized fit's (`σ₁` against `pls3::SIGMA_FLOOR` and the
/// relative floor), applied to this route's `σ₁`, so a split whose `σ₁`
/// sits at the floor can keep LV1 here and not there, or the reverse; a
/// dropped LV1 gives `r = 0`. A failed SVD gives `r = 0`, the value the
/// materialized fit gives a failed fit.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn pls3_split_r_streamed(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    sp: &SplitIdx,
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
) -> f64 {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
    let (p, q) = (x.ncols(), y.ncols());
    let (n_tr, n_te) = (tr.len(), te.len());
    let (ys_tr, ys_te) = centered_halves(y, tr, te, y_scale);

    // Seq inside the per-split worker: outer Rayon owns the pool.
    let seq = faer::Par::Seq;
    let mut a_mat = Mat::<f64>::zeros(p, q);
    let mut t_mat = Mat::<f64>::zeros(n_te, q);
    let x_fro = crate::linalg::centered_blocks(
        x,
        tr,
        te,
        None,
        None,
        None,
        x_scale,
        // The width linalg::stream_blocks_bytes counts (change together).
        crate::linalg::stream_block_width(n_tr.max(n_te)),
        |j0, a, c, _| {
            let mut a_blk = a_mat.as_mut().subrows_mut(j0, a.ncols());
            matmul(
                a_blk.as_mut(),
                Accum::Replace,
                a.transpose(),
                ys_tr.as_ref(),
                1.0,
                seq,
            );
            matmul(t_mat.as_mut(), Accum::Add, c, a_blk.as_ref(), 1.0, seq);
        },
    );
    let rel_floor = crate::pls3::sigma_rel_floor(n_tr, p, q, x_fro, ys_tr.norm_l2());
    let Ok(Some(crate::pls3::Lv1 { v, .. })) =
        crate::pls3::pls3_lv1_from_cross_cov(a_mat.as_ref(), || rel_floor)
    else {
        return 0.0;
    };
    let s = crate::linalg::mat_vec(t_mat.as_ref(), v.as_ref(), seq);
    let t = crate::linalg::mat_vec(ys_te.as_ref(), v.as_ref(), seq);
    pearson_r_guarded(&s, &t)
}

/// `split_nb`'s held-out LV1 correlation on each split of X: each split is
/// [`pls3_split_r_streamed`], the splits in batches of
/// `resample::map_units`.
#[allow(clippy::similar_names)] // n_tr, n_te
fn pls3_split_lv_correlations_streamed(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[SplitIdx],
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
    call_bytes: usize,
) -> PlsKitResult<Col<f64>> {
    let (n_tr, n_te) = split_lens(splits);
    let unit = streamed_unit_bytes(x.ncols(), y.ncols(), n_tr, n_te);
    let r = crate::resample::map_units(splits.len(), call_bytes, unit, |j| {
        pls3_split_r_streamed(x, y, &splits[j], x_scale, y_scale)
    })?;
    Ok(Col::<f64>::from_fn(r.len(), |j| r[j]))
}

/// The held-out LV1 correlation of the columns `which` of `cols` on one
/// split of the primal route, in that order: [`pls3_split_column_r_primal`]
/// per column on [`centered_halves`] of X built once for the split and of
/// `cols.block(c)` built per column, each divided by its block's all-row
/// scale. The columns are independent refits sharing only the read-only X
/// side, so they map in parallel; `map_indexed`'s `collect` keeps their
/// order, and each value does not depend on which worker computes it.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn pls3_split_columns_r_primal(
    x: MatRef<'_, f64>,
    cols: &crate::resample::Columns<'_>,
    sp: &SplitIdx,
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
    which: &[usize],
) -> Vec<f64> {
    let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());

    // Built once per split: invariant across replicates, because only Y
    // is permuted.
    let (xs_tr, xs_te) = centered_halves(x, tr, te, x_scale);
    // The X-side input of the relative floor, likewise per split.
    let xs_tr_fro = xs_tr.norm_l2();

    crate::resample::map_indexed(which.len(), |i| {
        let y_c = cols.block(which[i]);
        let (ys_tr, ys_te) = centered_halves(y_c.as_ref(), tr, te, y_scale);

        pls3_split_column_r_primal(
            xs_tr.as_ref(),
            xs_tr_fro,
            xs_te.as_ref(),
            ys_tr.as_ref(),
            ys_te.as_ref(),
        )
    })
}

/// The clamped Fisher z of each `r`: applies [`crate::signal_test::fisher_z`]
/// per (split, column), where `mean_fisher_z` applies it before summing.
fn fisher_z_clamped(r: Vec<f64>) -> Vec<f64> {
    r.into_iter().map(crate::signal_test::fisher_z).collect()
}

/// Per-column z̄ over the J splits on the primal route: `cols.len()`
/// values, column 0 the observed Y and column `c > 0` null column `c` of
/// `cols`, and the number of failed splits (column 0's `r` exactly 0,
/// `signal_test::zbars_from_rows`). The primal sibling of
/// `dual_route::pls3_split_zbars_columns`, and shaped like it on purpose.
///
/// # What it is
/// Exactly `mean_fisher_z(pls3_split_lv_correlations(..))` evaluated once per
/// replicate, with the two loops interchanged: splits outer, replicates
/// inner. Only Y is permuted, so a split's X side ([`centered_halves`]) is
/// a pure
/// function of `(x, split)` and is built once per split rather than
/// `n_perm + 1` times. That is the same hoist the Gram
/// route already makes ("Built once per split; neither carries p into the
/// replicate loop").
///
/// # Why it is byte-identical, not merely close
/// - Replicate `b`'s permutation is `rng::null_permutation(n, b, seed)`,
///   a pure function of `(seed, b)`, the same on every route.
/// - The per-`(split, column)` body is [`pls3_split_column_r_primal`],
///   the one `pls3_split_lv_correlations` calls, here reached through
///   [`pls3_split_columns_r_primal`], on bit-identical inputs: the
///   hoisted X quantities were already recomputed to the same bits every
///   replicate, and `cols.block(c)` holds `y[(perm[i], j)]`, the rows
///   `centered_halves(y_perm, tr, ..)` reads, divided by the same
///   `y_scale`.
/// - The splits run in batches of `resample::map_units` and the replicate
///   columns of a split map in parallel, so at most
///   `resample::units_at_once` prepared splits are alive at a time. Each
///   column's value is a pure function of its inputs (the inner fit runs
///   with `ParChoice::Seq`), and `collect` keeps split and column order, so
///   neither the batch size nor which worker computes a value can move a
///   bit.
/// - The accumulation is `signal_test::zbars_over_splits`: splits
///   **ascending**, one divide by J at the end, `mean_fisher_z`'s own
///   left-to-right sum-then-divide.
/// - Every product and decomposition of a column runs `Par::Seq`, as in the
///   un-interchanged form; see the note in [`pls3_split_column_r_primal`].
///
/// `pls3_split_lv_correlations` stays as the reference for the interchange:
/// the primal-reference reconstructions in this module's tests rebuild the
/// replicate-outer loop out of it (row-subsetting a permuted Y, X halves
/// rebuilt per replicate, both blocks on the unpermuted call's scales) and
/// assert p-value equality and
/// `statistic.to_bits()` against what this function produces. They check
/// the hoist, the column parallelism and the accumulation, not the shared
/// per-column body.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub(crate) fn pls3_split_zbars_columns_primal(
    x: MatRef<'_, f64>,
    cols: &crate::resample::Columns<'_>,
    splits: &[SplitIdx],
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
    call_bytes: usize,
) -> PlsKitResult<(Vec<f64>, usize)> {
    let n_cols = cols.len();
    let all: Vec<usize> = (0..n_cols).collect();
    let per_split = |sp: &SplitIdx| -> Vec<f64> {
        fisher_z_clamped(pls3_split_columns_r_primal(
            x, cols, sp, x_scale, y_scale, &all,
        ))
    };

    // Splits in batches of `resample::map_units`; the columns of a split
    // map in parallel, so a split counts its halves of X and one LV1 fit
    // per column that runs at once.
    let (n_tr, n_te) = split_lens(splits);
    let unit_bytes = primal_split_unit_bytes(
        n_tr,
        n_te,
        x.ncols(),
        cols.y.ncols(),
        n_cols,
        rayon::current_num_threads(),
    );
    crate::signal_test::zbars_over_splits(splits, n_cols, 0, call_bytes, unit_bytes, |sp| {
        Ok(per_split(sp))
    })
}

/// The cell route's bytes on shape `s` with `q` Y columns, at `threads`
/// threads: the call holds its `signal_test::CellFrame`
/// (`cell_sums::cell_frame_bytes`) and the design's index vectors; a unit is the largest of the frame's
/// split pass (one test half's `signal_test::TestMoments`), one batch of
/// cell sums (`cell_sums::cell_batch_bytes`, `q` columns per outcome) and
/// the primal split the guard sends columns to
/// ([`primal_split_unit_bytes`]).
fn cells_route_bytes(
    s: &crate::signal_test::SplitShape<'_>,
    q: usize,
    threads: usize,
) -> (usize, usize) {
    let (h, _, _) = s.design();
    let call =
        crate::cell_sums::cell_frame_bytes(s.n, h, s.n_splits, s.p).saturating_add(design_bytes(s));
    // shortcut: J units with the largest per-pass unit, though `cell_rows` runs ⌈B'/b⌉ column batches; safe for the fit check, biases the ranking against cells at large shapes.
    let unit = crate::signal_test::test_moments_bytes(s.n_test, s.p)
        .max(crate::cell_sums::cell_batch_bytes(
            s.n, h, s.p, s.n_splits, s.n_cols, q,
        ))
        .max(primal_split_unit_bytes(
            s.n_train, s.n_test, s.p, q, s.n_cols, threads,
        ));
    (call, unit)
}

/// The cell route's work in [`pls3_split_exact_route`] on shape `s` with
/// `q` Y columns, in multiply-adds, with `B'` outcome columns and
/// `(n_A, n_B)` a split's training and test rows:
///
/// ```text
/// n·p² + J·n_B·p² + B'·n·q·(p + q) + J·B'·p·(3p + 2q) + J·B'·p·q·min(p, q)
/// ```
///
/// the cell and test-half cross-products, the cell sums of every column,
/// per split and column the quadratic forms in the test half's factor and
/// the `p × q` products, and the `p × q` SVD the primal route counts too.
/// PLS3 takes no weights.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn cells_work(s: &crate::signal_test::SplitShape<'_>, q: usize) -> f64 {
    let (nf, pf, qf) = (s.n as f64, s.p as f64, q as f64);
    let (j, c) = (s.n_splits as f64, s.n_cols as f64);
    let b = s.n_test as f64;
    nf * pf * pf
        + j * b * pf * pf
        + c * nf * qf * (pf + qf)
        + j * c * pf * 3.0f64.mul_add(pf, 2.0 * qf)
        + j * c * pf * qf * pf.min(qf)
}

/// Per-column z̄ over the splits of `d` on the cell route: what
/// [`pls3_split_zbars_columns_primal`] returns on `d.splits`, to rounding,
/// with every split's training sums from the cells of the balanced design
/// (`crate::cell_sums`), so each column is read once per call, not once per
/// split. The count of failed splits comes with the values, as there.
///
/// # What it computes
/// In the frame of `signal_test::CellFrame`, `X̃` (X centred by its all-row
/// means, divided by `x_scale`) and `Ỹ` (each column of `cols.block(c)`
/// centred by its all-row mean, divided by `y_scale`, [`write_y_block`]):
/// per batch of
/// columns one pass over the rows and the transform give every split's
/// `X̃_A'Ỹ_A`, `1'Ỹ_A` and `Ỹ_A'Ỹ_A` (`cell_sums::cell_sums` in groups of
/// `q`), the test half's as the totals minus these. Per split and column:
///
/// - `A = X̃_A'Ỹ_A − X̃_A'1·(1'Ỹ_A)'/n_A`, the primal's `X̃_tr'Ỹ_tr` (both
///   halves are centred by the training means, which a common shift of the
///   frame does not change), and `‖Ỹ_tr‖_F² = Σ_k (Ỹ_A'Ỹ_A − 1'Ỹ_A·(1'Ỹ_A)'/n_A)_kk`.
/// - The component stage of `pls3::pls3_lv1_prestd` on `A`
///   (`pls3::pls3_lv1_from_cross_cov`), and its floors with
///   `‖X̃_tr‖_F` from the frame.
/// - The Pearson r of `X̃_B u` and `Ỹ_B v` from the test half's moments:
///   `u'S_xy v / √(u'S_xx u · v'S_yy v)` with `S` the cross-products of the
///   test half centred by its own means, `u'S_xx u` from its QR factor
///   (`signal_test::TestMoments`), as `signal_test::r_from_moments` forms it
///   for the score `Ỹ_B v`.
///
/// # Rounding guard
/// A value is kept only where its rounding is bounded well inside the
/// result, by `signal_test::NR_RESOLVE_BAND`; every other `(split, column)`
/// runs the primal route's own body ([`pls3_split_columns_r_primal`], the
/// splits that have one in batches of `resample::map_units`). With `ε` the
/// machine epsilon, `L = log₂ H`, `‖ỹ_k‖` column `k`'s all-row norm and the
/// sums formed as totals minus training sums (each product bounded by
/// Cauchy-Schwarz):
///
/// - The training half: `‖Ỹ_tr‖_F²` must exceed `8n·ε·‖Ỹ‖_F²`, the bound on
///   its cancellation (`cell_y_sums`'s test, summed over the columns).
/// - The truncation: `‖ΔA‖ ≤ E = (2n + p + L + 8)·ε·‖X̃‖_F·(‖Ỹ‖_F + 2√n·‖η_A‖)`,
///   `η_A = 1'Ỹ_A/n_A`, covers this route's `A` and the primal's (whose
///   bound is the relative floor's `n_A·ε·‖X̃_tr‖_F·‖Ỹ_tr‖_F`). LV1 is kept
///   only when `σ₁` clears the floor by the band times `E` plus the floor's
///   own rounding; a dropped or undecided LV1 runs the primal route.
/// - The Y score: `v'S_yy v ≥ 8n·ε·s_v²`, `s_v = Σ_k |v_k|·‖ỹ_k‖`; the cross
///   term `u'S_xy v` moves by at most `2(n + L + 4)·ε·‖X̃‖_F·s_v`, which must
///   sit under `‖R_B u‖·√(v'S_yy v)`; and neither score is constant to
///   rounding in either frame (the X score through
///   `signal_test::moments_resolved`, which also caps its conditioning).
/// - `(u, v)` moves with `A` by at most about `δ = E/gap` ([`lv1_reach`]),
///   so `‖R_B u‖` must clear `F_te·δ` as well (`TestMoments::score_norms`)
///   and `√(v'S_yy v)` must clear `‖Ỹ_B‖_F·δ` (`‖Ỹ_B‖_F² = tr S_yy`); a
///   near-tie of the first two singular values runs the primal route.
///
/// A Y whose raw sums can overflow (`y_halves_finite`) runs the primal
/// route throughout. Every pass holds `call_bytes` for the call, the frame
/// [`cells_route_bytes`] counts.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]
pub(crate) fn pls3_split_zbars_columns_cells(
    x: MatRef<'_, f64>,
    cols: &crate::resample::Columns<'_>,
    d: &SplitDesign,
    x_scale: ColRef<'_, f64>,
    y_scale: ColRef<'_, f64>,
    call_bytes: usize,
) -> PlsKitResult<(Vec<f64>, usize)> {
    let (p, q, n_cols) = (x.ncols(), cols.y.ncols(), cols.len());
    let sizes = (d.splits[0].tr.len(), d.splits[0].te.len());
    let rows = if y_halves_finite(cols.y, y_scale) {
        let frame = crate::signal_test::CellFrame::new(x, x_scale, d, false, call_bytes)?;
        let levels = d.cells().trailing_zeros() as usize;
        crate::signal_test::cell_rows(
            &frame,
            d,
            n_cols,
            q,
            call_bytes,
            &|c, out| write_y_block(cols, c, y_scale, out),
            &|cs, j, c| pls3_cell_r(&frame, cs, j, c, sizes, levels),
        )?
    } else {
        vec![vec![None; n_cols]; d.splits.len()]
    };
    let rows = crate::resample::fill_pending(
        rows,
        call_bytes,
        primal_split_unit_bytes(sizes.0, sizes.1, p, q, n_cols, rayon::current_num_threads()),
        |j, pending| pls3_split_columns_r_primal(x, cols, &d.splits[j], x_scale, y_scale, pending),
    )?;
    let z_rows: Vec<Vec<f64>> = rows.into_iter().map(fisher_z_clamped).collect();
    Ok(crate::signal_test::zbars_from_rows(n_cols, 0, &z_rows))
}

/// Column `c` of `cols` as the cell route sums it, into `out` (`n × q`):
/// each column centred by its row-order mean of the raw values, then
/// divided by `y_scale`, the order `centered_halves` uses, so a mean far
/// above the spread costs only the rounding of the centred values.
#[allow(clippy::cast_precision_loss)]
fn write_y_block(
    cols: &crate::resample::Columns<'_>,
    c: usize,
    y_scale: ColRef<'_, f64>,
    mut out: faer::MatMut<'_, f64>,
) {
    let blk = cols.block(c);
    let n = blk.nrows();
    for k in 0..blk.ncols() {
        let m = (0..n).map(|i| blk[(i, k)]).sum::<f64>() / n as f64;
        for i in 0..n {
            out[(i, k)] = (blk[(i, k)] - m) / y_scale[k];
        }
    }
}

/// Whether every half of `y` the primal route can write, centred by any
/// training rows' means and divided by `scale`, is finite: per column the
/// sum of `n` entries cannot overflow (`n·max|y|` is finite) and
/// `4·(max − min)/scale` is finite. The cell route centres Y from its raw
/// sums ([`write_y_block`]), which can overflow where the primal route's
/// centring does not.
#[allow(clippy::cast_precision_loss)]
fn y_halves_finite(y: MatRef<'_, f64>, scale: ColRef<'_, f64>) -> bool {
    let n = y.nrows() as f64;
    crate::signal_test::column_extremes(y)
        .iter()
        .enumerate()
        .all(|(k, &(hi, lo))| {
            (n * hi.abs().max(lo.abs())).is_finite() && (4.0 * (hi - lo) / scale[k]).is_finite()
        })
}

/// How far `(u, v)` of `lv1` can move when `A` moves by `e_a` (in norm),
/// or `None` when that reach can change what the component stage returns.
/// By Wedin's bound the pair moves by at most about `δ = e_a/gap`, with
/// `gap = σ₁ − σ₂` from the stage's own SVD. `None` when
/// `gap < BAND·e_a` (a near-tie).
fn lv1_reach(lv1: &crate::pls3::Lv1, e_a: f64) -> Option<f64> {
    use crate::signal_test::NR_RESOLVE_BAND as BAND;
    let gap = lv1.sigma - lv1.sigma2;
    (gap >= BAND * e_a).then_some(())?;
    Some(e_a / gap)
}

/// The held-out LV1 correlation of a batch's column `c` (an `n × q` block)
/// on split `j` from the cells, or `None` where the rounding guard of
/// [`pls3_split_zbars_columns_cells`] leaves it to the primal route.
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn pls3_cell_r(
    f: &crate::signal_test::CellFrame,
    cs: &crate::cell_sums::CellSums,
    j: usize,
    c: usize,
    (n_tr, n_te): (usize, usize),
    levels: usize,
) -> Option<f64> {
    use crate::signal_test::{NR_RESOLVE_BAND as BAND, moments_resolved, r_from_moments};
    let (n, p) = (f.xs.nrows(), f.xs.ncols());
    let (nf, na, nb, eps) = (n as f64, n_tr as f64, n_te as f64, f64::EPSILON);
    let w = cs.y_sum.nrows();
    let q = cs.y_ss.nrows() / w;
    // Column `k` of the block, and product `(k, l)` of its group.
    let (o, oo) = (c * q, c * q * q);
    let xa_y = |r: usize, k: usize| cs.xa_y[(r, j * w + o + k)];
    let yy_a = |k: usize, l: usize| cs.ya_ss[(j, oo + k * q + l)];
    let yy = |k: usize, l: usize| cs.y_ss[oo + k * q + l];
    let sum_a: Vec<f64> = (0..q).map(|k| cs.ya_sum[(j, o + k)]).collect();
    let sum_b: Vec<f64> = (0..q).map(|k| cs.y_sum[o + k] - sum_a[k]).collect();
    let eta_a: Vec<f64> = sum_a.iter().map(|s| s / na).collect();

    // The training half.
    let a = Mat::<f64>::from_fn(p, q, |r, k| eta_a[k].mul_add(-f.xa_1[j][r], xa_y(r, k)));
    let css_a: f64 = (0..q)
        .map(|k| sum_a[k].mul_add(-eta_a[k], yy_a(k, k)))
        .sum();
    let y_ss: f64 = (0..q).map(|k| yy(k, k)).sum();
    let e_css = 8.0 * nf * eps * y_ss;
    (css_a >= BAND * e_css).then_some(())?;
    let floor = crate::pls3::sigma_rel_floor(n_tr, p, q, f.tr_fro[j], css_a.sqrt())
        .max(crate::pls3::SIGMA_FLOOR);
    let eta_norm = eta_a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let e_a = (2 * n + p + levels + 8) as f64
        * eps
        * f.x_fro
        * (2.0 * nf.sqrt()).mul_add(eta_norm, y_ss.sqrt());
    let floor_err = floor * (e_css / (2.0 * css_a) + (n + 4) as f64 * eps);
    // The floor is applied below with its margin, so the stage gets none.
    let lv1 = crate::pls3::pls3_lv1_from_cross_cov(a.as_ref(), || 0.0).ok()??;
    (lv1.sigma - BAND * (e_a + floor_err) >= floor).then_some(())?;
    let delta = lv1_reach(&lv1, e_a)?;
    let (u, v) = (lv1.u, lv1.v);

    // The test half, as the score `Ỹ_B v` and its products.
    let t_sum: f64 = (0..q).map(|k| v[k] * sum_b[k]).sum();
    let (mut t_css, mut yb_ss) = (0.0, 0.0);
    for k in 0..q {
        for l in 0..q {
            let s_kl = sum_b[k].mul_add(-sum_b[l] / nb, yy(k, l) - yy_a(k, l));
            t_css = (v[k] * v[l]).mul_add(s_kl, t_css);
            if k == l {
                yb_ss += s_kl;
            }
        }
    }
    let xb_t = Col::<f64>::from_fn(p, |r| {
        (0..q)
            .map(|k| v[k] * (cs.x_y[(r, o + k)] - xa_y(r, k)))
            .sum::<f64>()
    });
    let tm = &f.tms[j];
    let t_c2 = tm.quad(u.as_ref());
    let r = r_from_moments(u.as_ref(), t_c2, xb_t.as_ref(), t_sum, t_css, tm);
    let sn = tm.score_norms(u.as_ref(), t_c2, f.x_sum.as_ref(), n_tr);
    let s_v: f64 = (0..q).map(|k| v[k].abs() * yy(k, k).sqrt()).sum();
    let cross_err = 2.0 * (n + levels + 4) as f64 * eps * f.x_fro * s_v;
    // The mean of the primal route's `Ỹ_te v`, centred by the training means.
    let t_ref: f64 = (0..q).map(|k| v[k] * (sum_b[k] / nb - eta_a[k])).sum();
    let tol = BAND * 2.0 * nb * eps;
    let err = sn
        .f_te
        .mul_add(delta, ((n_te + 2) as f64 * eps).mul_add(sn.s_ref, sn.err));
    // `a >= b` conjunctions, so a NaN leaves the column to the primal route.
    let resolved = t_css >= BAND * 8.0 * nf * eps * s_v * s_v
        && sn.t_c * t_css.sqrt() >= BAND * cross_err
        && t_css.sqrt() >= tol * nb.mul_add(t_ref * t_ref, t_css).sqrt()
        && sn.t_c >= BAND * err
        && t_css.sqrt() >= BAND * yb_ss.sqrt() * delta
        && moments_resolved(&sn, t_sum, t_css, n_te, BAND)
        && r.is_finite();
    resolved.then_some(r)
}

/// Held-out LV1 correlation for one `(split, replicate column)` on the
/// primal route: an honest `pls3_fit` refit on the training half, scored
/// on the test half, both [`centered_halves`].
///
/// The one per-column body of every PLS3 split statistic computed on
/// materialized halves: the `split_exact` primal route
/// (`pls3_split_zbars_columns_primal`) and its test reference
/// `pls3_split_lv_correlations`. Keeping one body is what keeps
/// those paths bit-equal. Returns `r` before the `atanh` clamp.
#[allow(clippy::similar_names)]
pub(crate) fn pls3_split_column_r_primal(
    xs_tr: MatRef<'_, f64>,
    xs_tr_fro: f64,
    xs_te: MatRef<'_, f64>,
    ys_tr: MatRef<'_, f64>,
    ys_te: MatRef<'_, f64>,
) -> f64 {
    // `pls3_lv1_prestd` is the LV1 of `pls3_fit`, pre-standardized and with
    // `ParChoice::Seq` inside the per-split or per-column worker (outer
    // Rayon owns the pool), without the score matmuls those fits would build
    // and this loop would discard. `Seq` governs the SVD too
    // (`linalg::thin_svd`), and the two score products below are `Seq`:
    // nothing here reads faer's global parallelism, whose degree is the
    // pool size.
    match crate::pls3::pls3_lv1_prestd(xs_tr, xs_tr_fro, ys_tr) {
        // `None` is a half with no first component: a zero X̃ or Ỹ block,
        // or a Ỹ orthogonal to X̃ up to rounding (`σ₁` under the relative
        // floor). No direction exists, so the honest answer is no
        // association, not NaN. An
        // `Err` is an inner-fit failure (e.g. SVD non-convergence) on this
        // split, also `0.0`; both are counted in `n_failed_splits`.
        Ok(Some((u, v))) => {
            // Seq inside the per-split or per-column worker.
            let s = crate::linalg::mat_vec(xs_te, u.as_ref(), faer::Par::Seq);
            let t = crate::linalg::mat_vec(ys_te, v.as_ref(), faer::Par::Seq);
            pearson_r_guarded(&s, &t)
        }
        _ => 0.0,
    }
}

/// Pearson r with the crate's degenerate-input convention: an input vector
/// that is constant to rounding yields exactly `0.0`, never NaN, and the
/// result is clamped into `[-1, 1]` against fp overshoot.
///
/// It is PLS1's split-half correlation (`signal_test::guarded_pearson`,
/// which `signal_test::split_half_r` reports and
/// `signal_test::nr_split_r` rebuilds from the same helpers), so
/// PLS1 and PLS3 share one degeneracy guard; `linalg::constant_to_rounding`
/// owns the explanation. The guard is relative to each vector's own
/// magnitude, and the sums behind it are formed on each vector divided by a
/// power of two near its largest entry (`linalg::scaled_moments`), so
/// rescaling either input cannot switch it at any magnitude.
/// `dual_route::pls3_split_zbars_columns` calls this one rather than
/// carrying a third copy.
pub(crate) fn pearson_r_guarded(a: &Col<f64>, b: &Col<f64>) -> f64 {
    let n = a.nrows();
    if n == 0 {
        return 0.0;
    }
    crate::signal_test::guarded_pearson(n, |i| a[i], |i| b[i])
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests {
    use super::*;
    use crate::signal_test::ConfirmatoryArgs;

    /// Whether [`pls3_split_nb_route`] takes the Gram route on dense X with
    /// the gate's matrix held.
    fn split_nb_takes_gram(n: usize, p: usize, q: usize, n_splits: usize) -> bool {
        pls3_split_nb_route(n, p, q, n_splits, true).unwrap().route == ReplicateRoute::Nspace
    }

    /// Whether [`pls3_sv_perm_route`] takes the Gram route.
    fn sv_perm_takes_gram(n: usize, p: usize, q: usize, n_perm: usize) -> bool {
        pls3_sv_perm_route(n, p, q, n_perm).unwrap().route == ReplicateRoute::Nspace
    }

    /// (X, Y) sharing one latent factor at strength `snr`; `snr = 0.0`
    /// gives independent blocks (the null).
    #[allow(clippy::many_single_char_names)]
    fn linked_blocks(n: usize, p: usize, q: usize, snr: f64, seed: u64) -> (Mat<f64>, Mat<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let f: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x = Mat::<f64>::from_fn(n, p, |i, j| {
            let e = rng.random_range(-1.0..1.0);
            if j < 2 { snr * f[i] + e } else { e }
        });
        let y = Mat::<f64>::from_fn(n, q, |i, j| {
            let e = rng.random_range(-1.0..1.0);
            if j == 0 { snr * f[i] + e } else { e }
        });
        (x, y)
    }

    /// The outcome columns of `y` with no null column.
    fn observed(y: MatRef<'_, f64>) -> crate::resample::Columns<'_> {
        crate::resample::Columns {
            y,
            seed: 0,
            n_perm: 0,
            null_map: None,
            w: None,
        }
    }

    /// The all-row column scales `pls3_confirmatory_test` divides X and Y
    /// by without the flags.
    fn scales(x: &Mat<f64>, y: &Mat<f64>) -> (Col<f64>, Col<f64>) {
        (
            crate::linalg::held_out_scale(x.as_ref(), None, false),
            crate::linalg::held_out_scale(y.as_ref(), None, false),
        )
    }

    /// The all-row Gram matrix of `x` on the scale `xsc`, as the runner
    /// builds it for the Gram route.
    fn gram(x: &Mat<f64>, xsc: &Col<f64>) -> crate::held_out_gram::HeldOutGram {
        crate::held_out_gram::HeldOutGram::from_x(x.as_ref(), None, xsc.as_ref())
    }

    fn opts(n_perm: usize, n_splits: usize, seed: u64) -> Pls3ConfirmatoryTestOpts {
        Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            ..Pls3ConfirmatoryTestOpts::default()
        }
    }

    fn nb_opts(n_splits: usize, force: bool, seed: u64) -> Pls3ConfirmatoryTestOpts {
        Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitNb { n_splits, force },
            seed: Some(seed),
            ..Pls3ConfirmatoryTestOpts::default()
        }
    }

    /// Both blocks standardized and then times `2^-27`, both flags set: the
    /// blocks are divided by a power of two first, so `SIGMA_FLOOR` does not
    /// cut a split's LV1, and `split_exact` gives the p-value and the
    /// statistic of the unscaled blocks, bit for bit at the same seed.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_small_pre_standardized_blocks_keep_their_p_value() {
        let (x, y) = linked_blocks(40, 10, 3, 1.0, 5);
        let xs = crate::linalg::standardize(x.as_ref()).0;
        let ys = crate::linalg::standardize(y.as_ref()).0;
        let s = 2f64.powi(-27);
        let times = |m: &Mat<f64>| Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * s);
        let fit = crate::pls3::pls3_fit(
            xs.as_ref(),
            ys.as_ref(),
            1,
            None,
            crate::pls3::Pls3FitOpts {
                pre_standardized_x: true,
                pre_standardized_y: true,
                ..crate::pls3::Pls3FitOpts::default()
            },
        )
        .unwrap();
        assert!(
            fit.singular_values[0] * s * s < 1e-14,
            "premise: σ₁ of the scaled blocks is under SIGMA_FLOOR"
        );
        let o = Pls3ConfirmatoryTestOpts {
            pre_standardized_x: true,
            pre_standardized_y: true,
            ..opts(19, 6, 7)
        };
        let a = pls3_confirmatory_test(xs.as_ref(), ys.as_ref(), 1, o).unwrap();
        assert!(a.pvalue < 1.0, "premise: the unscaled blocks have p < 1");
        let b = pls3_confirmatory_test(times(&xs).as_ref(), times(&ys).as_ref(), 1, o).unwrap();
        assert_eq!(
            b.pvalue.to_bits(),
            a.pvalue.to_bits(),
            "pvalue {} against {}",
            b.pvalue,
            a.pvalue
        );
        assert_eq!(
            b.statistic.map(f64::to_bits),
            a.statistic.map(f64::to_bits),
            "statistic {:?} against {:?}",
            b.statistic,
            a.statistic
        );
    }

    /// `sv_perm` reports `d₁` on the scale of the blocks as passed: on both
    /// standardized blocks times `2^-30`, both flags set, the p-value is the
    /// unscaled blocks' and the statistic theirs times `2^-60`, bit for bit
    /// at the same seed, on the X form and on the Gram form.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_sv_perm_statistic_is_on_the_blocks_scale() {
        let (x, y) = linked_blocks(40, 10, 3, 1.0, 9);
        let xs = crate::linalg::standardize(x.as_ref()).0;
        let ys = crate::linalg::standardize(y.as_ref()).0;
        let s = 2f64.powi(-30);
        let times = |m: &Mat<f64>| Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * s);
        let (xt, yt) = (times(&xs), times(&ys));
        let o = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SvPerm {
                n_perm: 99,
                rotate: false,
            },
            seed: Some(7),
            pre_standardized_y: true,
            ..Pls3ConfirmatoryTestOpts::default()
        };
        let xo = Pls3ConfirmatoryTestOpts {
            pre_standardized_x: true,
            ..o
        };
        let g_of = |x: &Mat<f64>| crate::gram_matrix(x.as_ref(), true).unwrap();
        let forms = [
            (
                "X",
                pls3_confirmatory_test(xs.as_ref(), ys.as_ref(), 1, xo).unwrap(),
                pls3_confirmatory_test(xt.as_ref(), yt.as_ref(), 1, xo).unwrap(),
            ),
            (
                "Gram",
                pls3_confirmatory_test_gram(&g_of(&xs), ys.as_ref(), 1, o).unwrap(),
                pls3_confirmatory_test_gram(&g_of(&xt), yt.as_ref(), 1, o).unwrap(),
            ),
        ];
        let mut failed = Vec::new();
        for (form, a, b) in forms {
            let want = a.statistic.map(|v| v * s * s);
            if b.pvalue.to_bits() != a.pvalue.to_bits()
                || b.statistic.map(f64::to_bits) != want.map(f64::to_bits)
            {
                failed.push(format!(
                    "{form}: p {} against {}, statistic {:?} against {want:?}",
                    b.pvalue, a.pvalue, b.statistic
                ));
            }
        }
        assert!(failed.is_empty(), "{failed:#?}");
    }

    #[test]
    fn strong_shared_factor_rejects() {
        let (x, y) = linked_blocks(80, 8, 4, 4.0, 3);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(199, 10, 42)).unwrap();
        assert_eq!(r.test_method, "split_exact");
        assert_eq!(r.k, vec![1]);
        assert_eq!(r.n_perm, Some(199));
        assert_eq!(r.n_splits, Some(10));
        assert_eq!(r.seed, 42);
        assert!(r.rho_hat.is_none());
        assert!(r.stable_rank.is_none());
        assert!(r.ci.is_none());
        assert!(r.pvalue <= 0.01, "p = {}", r.pvalue);
        assert!(
            r.statistic.unwrap() > 0.3,
            "statistic = {}",
            r.statistic.unwrap()
        );
    }

    #[test]
    fn independent_blocks_do_not_reject() {
        let (x, y) = linked_blocks(80, 8, 4, 0.0, 17);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(199, 10, 42)).unwrap();
        assert!(r.pvalue > 0.05, "p = {}", r.pvalue);
    }

    #[test]
    fn pvalue_is_in_the_exact_permutation_grid() {
        let (x, y) = linked_blocks(60, 6, 3, 1.0, 5);
        let n_perm = 99;
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(n_perm, 8, 42)).unwrap();
        let grid_step = 1.0 / (n_perm as f64 + 1.0);
        let steps = r.pvalue / grid_step;
        assert!(
            (steps - steps.round()).abs() < 1e-9,
            "p = {} is not on the 1/(B+1) grid",
            r.pvalue
        );
        assert!(r.pvalue >= grid_step - 1e-12);
    }

    /// Every split route reads X and Y only through row-subset centring
    /// and scaling into owned matrices or column blocks, and through the
    /// all-row scales, so a padded
    /// submatrix, a row-major view and a negative-stride view of X, or of
    /// Y, give the owned matrix's result to the bit: on `split_exact`'s
    /// primal route (dense 48x7) and Gram route (wide 30x400), and on
    /// `split_nb`'s streamed route.
    #[test]
    fn confirmatory_output_is_bit_identical_across_x_and_y_layouts() {
        use crate::test_support::{assert_layout_invariant, copy_free_families};
        assert!(
            split_exact_takes_gram(30, 400, 4, 9, 4, false),
            "test premise: the wide family must take the Gram route"
        );
        for f in copy_free_families().into_iter().filter(|f| f.w.is_none()) {
            let n = f.x.nrows();
            let y = Mat::<f64>::from_fn(n, 4, |i, j| f.x[(i, j)] + 0.5 * f.y[(i + 3 * j) % n]);
            for args in [
                ConfirmatoryArgs::SplitExact {
                    n_perm: 9,
                    n_splits: 4,
                },
                ConfirmatoryArgs::SplitNb {
                    n_splits: 4,
                    force: true,
                },
            ] {
                let o = Pls3ConfirmatoryTestOpts {
                    args,
                    seed: Some(6),
                    ..Pls3ConfirmatoryTestOpts::default()
                };
                let what = format!("{} {args:?}", f.name);
                assert_layout_invariant(&f.x, &format!("X {what}"), |xv| {
                    pls3_confirmatory_test(xv, y.as_ref(), 1, o).unwrap()
                });
                assert_layout_invariant(&y, &format!("Y {what}"), |yv| {
                    pls3_confirmatory_test(f.x.as_ref(), yv, 1, o).unwrap()
                });
            }
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn seed_none_is_recorded_and_replayable() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let mut o = opts(49, 6, 0);
        o.seed = None;
        let a = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o).unwrap();
        let b = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(49, 6, a.seed)).unwrap();
        assert_eq!(a.pvalue.to_bits(), b.pvalue.to_bits());
    }

    /// `|a − b| ≤ 1e-12·max(|b|, 1e-300)`.
    fn assert_close(a: f64, b: f64, what: &str) {
        assert!(
            (a - b).abs() <= 1e-12 * b.abs().max(1e-300),
            "{what}: {a} vs {b}"
        );
    }

    /// A held-out half whose plain centring overflows while its training
    /// half does not gives the `r` of X/8. Column 0 holds `-f64::MAX` on
    /// training row 0 and `f64::MAX` on held-out row 6, which overflows when
    /// centred with the training mean.
    #[test]
    fn a_held_out_half_near_max_reads_as_x_over_8() {
        let (mut x, y) = linked_blocks(12, 3, 2, 1.0, 5);
        x[(0, 0)] = -f64::MAX;
        x[(6, 0)] = f64::MAX;
        let splits = vec![SplitIdx {
            tr: (0..6).collect(),
            te: (6..12).collect(),
        }];
        let rz = |x: &Mat<f64>| {
            let (xsc, ysc) = scales(x, &y);
            let r = pls3_split_lv_correlations(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap();
            let z = pls3_split_columns_r_primal(
                x.as_ref(),
                &observed(y.as_ref()),
                &splits[0],
                xsc.as_ref(),
                ysc.as_ref(),
                &[0],
            );
            (r[0], z[0])
        };
        let x8 = Mat::<f64>::from_fn(12, 3, |i, j| x[(i, j)] / 8.0);
        let ((r, z), (r8, z8)) = (rz(&x), rz(&x8));
        assert_close(r, r8, "r");
        assert_close(z, z8, "z");
    }

    /// The same on the Y side gives the value of Y/8 on every route: Y
    /// column 0 at `-f64::MAX` on training row 0 and `f64::MAX` on held-out
    /// row 6. `split_nb` on the materialized halves and on the streamed
    /// body, and `split_exact` on the primal and the Gram route.
    #[test]
    fn a_held_out_y_half_near_max_reads_as_y_over_8_on_every_route() {
        let (x, mut y) = linked_blocks(12, 3, 2, 1.0, 5);
        y[(0, 0)] = -f64::MAX;
        y[(6, 0)] = f64::MAX;
        let splits = vec![SplitIdx {
            tr: (0..6).collect(),
            te: (6..12).collect(),
        }];
        let routes = |y: &Mat<f64>| {
            let (xsc, ysc) = scales(&x, y);
            let r = pls3_split_lv_correlations(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap();
            let rs = pls3_split_lv_correlations_streamed(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap();
            let zp = pls3_split_columns_r_primal(
                x.as_ref(),
                &observed(y.as_ref()),
                &splits[0],
                xsc.as_ref(),
                ysc.as_ref(),
                &[0],
            );
            let zg = crate::dual_route::pls3_split_zbars_columns(
                &gram(&x, &xsc),
                &observed(y.as_ref()),
                &splits,
                ysc.as_ref(),
                0,
            )
            .unwrap()
            .0;
            [r[0], rs[0], zp[0], zg[0]]
        };
        let y8 = Mat::<f64>::from_fn(12, 2, |i, j| y[(i, j)] / 8.0);
        for (i, (a, b)) in routes(&y).iter().zip(routes(&y8)).enumerate() {
            assert_close(*a, b, &format!("route {i}"));
        }
    }

    /// Every PLS3 split statistic off the Gram route stays finite on an X
    /// whose plain centring overflows only in held-out halves (column 0 at `f64::MAX` on row
    /// 0 and `-f64::MAX` on row 1: a split holding one of them in each half
    /// overflows when the held-out one is centred): `split_nb` (the
    /// streamed body) and `split_exact` on the primal route.
    #[test]
    fn split_statistics_stay_finite_when_a_held_out_half_overflows() {
        let (mut x, y) = linked_blocks(40, 6, 3, 1.0, 11);
        x[(0, 0)] = f64::MAX;
        x[(1, 0)] = -f64::MAX;
        assert!(
            !split_exact_takes_gram(40, 6, 3, 19, 6, false),
            "premise: primal route"
        );
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: 20,
            force: true,
        };
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 19,
            n_splits: 6,
        };
        for args in [nb, exact] {
            let r = pls3_confirmatory_test(
                x.as_ref(),
                y.as_ref(),
                1,
                Pls3ConfirmatoryTestOpts {
                    args,
                    seed: Some(7),
                    ..Default::default()
                },
            )
            .expect("a result");
            assert!(
                r.statistic.unwrap().is_finite(),
                "{args:?}: {}",
                r.statistic.unwrap()
            );
        }
    }

    #[test]
    fn zero_column_blocks_error_instead_of_reporting_no_association() {
        let (x, y) = linked_blocks(60, 5, 3, 2.0, 5);
        let empty = Mat::<f64>::from_fn(60, 0, |_, _| 0.0);
        for (block, x, y) in [
            ("x", empty.as_ref(), y.as_ref()),
            ("y", x.as_ref(), empty.as_ref()),
        ] {
            let r = pls3_confirmatory_test(x, y, 1, opts(49, 6, 7));
            assert!(
                matches!(r, Err(PlsKitError::KExceedsMax { k: 1, k_max: 0 })),
                "empty {block}: {r:?}"
            );
        }
    }

    #[test]
    fn k_above_one_errors() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 2, opts(49, 6, 7));
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn methods_outside_the_split_family_error() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        for args in [
            ConfirmatoryArgs::RawPerm {
                n_perm: 10,
                n_folds: 5,
            },
            ConfirmatoryArgs::Score { n_perm: None },
            ConfirmatoryArgs::Score { n_perm: Some(10) },
            ConfirmatoryArgs::E,
        ] {
            let o = Pls3ConfirmatoryTestOpts {
                args,
                seed: Some(1),
                ..Pls3ConfirmatoryTestOpts::default()
            };
            let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o);
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "method {} should be rejected",
                args.method().as_str()
            );
        }
    }

    #[test]
    fn count_floors_are_enforced() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let bad_splits = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 10,
                n_splits: 1,
            },
            seed: Some(1),
            ..Pls3ConfirmatoryTestOpts::default()
        };
        assert!(matches!(
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, bad_splits),
            Err(PlsKitError::InvalidArgument(_))
        ));
        let bad_perm = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 0,
                n_splits: 10,
            },
            seed: Some(1),
            ..Pls3ConfirmatoryTestOpts::default()
        };
        assert!(matches!(
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, bad_perm),
            Err(PlsKitError::InvalidArgument(_))
        ));
        assert!(matches!(
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(1, false, 1)),
            Err(PlsKitError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tiny_n_is_rejected_by_the_split_floor() {
        let (x, y) = linked_blocks(5, 4, 2, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(10, 4, 7));
        assert!(matches!(r, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn constant_x_gives_zero_statistic_not_nan() {
        // Degenerate X ⇒ X̃ is exactly zero ⇒ no component survives the σ
        // floor ⇒ every split must report r = 0.0, never NaN.
        let x = Mat::<f64>::from_fn(60, 5, |_, _| 1.25);
        let (_, y) = linked_blocks(60, 5, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(49, 6, 7)).unwrap();
        assert!(r.statistic.unwrap().is_finite());
        assert!(
            r.statistic.unwrap().abs() < 1e-12,
            "statistic = {}",
            r.statistic.unwrap()
        );
    }

    // ── split_nb tests ──────────────────────────────────────────────────

    #[test]
    fn split_nb_runs_and_reports_its_own_method() {
        // 6 columns and n = 60 clear the gate's column, n_eff and stable-rank
        // floors, so the request is not rerouted. q = 3 is inside the gate's
        // 4-column precheck, so this also fails if Y were ever gated.
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(10, false, 7)).unwrap();
        assert_eq!(r.test_method, "split_nb");
        assert!(r.n_perm.is_none(), "split_nb runs no permutations");
        assert_eq!(r.n_splits, Some(10));
        assert!(r.pvalue > 0.0 && r.pvalue <= 1.0, "p = {}", r.pvalue);
        assert!(r.statistic.unwrap().is_finite());
        // stable_rank is what the gate saw; rho_hat needs n_test ≥ 4 (30 here).
        assert!(r.stable_rank.is_some());
        assert!(r.rho_hat.is_some());
    }

    #[test]
    fn split_nb_statistic_matches_split_exact_at_the_same_seed() {
        // Both methods report tanh(z̄) over the splits `draw_splits` produces
        // from the seed, so only the reference differs. Dense `split_nb`
        // reads X block by block and `split_exact` fits materialized halves
        // here, so the two agree to rounding.
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let nb = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(8, false, 42)).unwrap();
        let ex = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(99, 8, 42)).unwrap();
        assert_eq!(nb.test_method, "split_nb");
        assert_eq!(ex.test_method, "split_exact");
        assert!(
            (nb.statistic.unwrap() - ex.statistic.unwrap()).abs() <= 1e-10,
            "{} vs {}",
            nb.statistic.unwrap(),
            ex.statistic.unwrap()
        );
    }

    #[test]
    fn split_nb_gate_reroutes_a_flagged_design() {
        // 3 columns trips the gate's column precheck, so the run is rerouted
        // to split_exact at that method's own default n_perm.
        let (x, y) = linked_blocks(60, 3, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(4, false, 7)).unwrap();
        assert_eq!(r.test_method, "split_exact");
        assert_eq!(r.n_perm, Some(1000));
        assert_eq!(r.n_splits, Some(4), "the requested n_splits carries over");
        assert!(r.stable_rank.is_some(), "the gate reports what it saw");
    }

    #[test]
    fn split_nb_force_overrides_the_gate() {
        let (x, y) = linked_blocks(60, 3, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(4, true, 7)).unwrap();
        assert_eq!(r.test_method, "split_nb");
        assert!(r.n_perm.is_none());
        assert_eq!((r.n_folds, r.force, r.rotate), (None, Some(true), None));
    }

    /// Dense `split_nb` reads X block by block; each split's `r` agrees
    /// with the materialized fit on a one-block shape, a three-block wide
    /// shape (20-row halves, 13,104-column blocks) and a tall shape.
    #[test]
    fn pls3_split_nb_streamed_matches_the_materialized_correlations() {
        use crate::signal_test::draw_splits;
        assert_eq!(crate::linalg::stream_block_width(20), 13_104);
        for (n, p, q) in [(60_usize, 6_usize, 3_usize), (40, 30_000, 3), (200, 12, 4)] {
            let (x, y) = linked_blocks(n, p, q, 2.0, 5);
            let (xsc, ysc) = scales(&x, &y);
            let splits = draw_splits(n, 1, 6, 7).unwrap();
            let got = pls3_split_lv_correlations_streamed(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap();
            let want = pls3_split_lv_correlations(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap();
            for j in 0..splits.len() {
                assert!(
                    (got[j] - want[j]).abs() <= 1e-10,
                    "n={n} p={p} split {j}: {} vs {}",
                    got[j],
                    want[j]
                );
            }
        }
    }

    /// A constant X has no first component: the streamed body gives
    /// `r = 0`, the materialized fit's value, on every split.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_split_nb_streamed_gives_zero_on_a_constant_x() {
        use crate::signal_test::draw_splits;
        let (n, p, q) = (40_usize, 9_usize, 3_usize);
        let x = Mat::<f64>::from_fn(n, p, |_, j| (j + 1) as f64);
        let (_, y) = linked_blocks(n, p, q, 2.0, 5);
        let (xsc, ysc) = scales(&x, &y);
        let splits = draw_splits(n, 1, 4, 3).unwrap();
        let got = pls3_split_lv_correlations_streamed(
            x.as_ref(),
            y.as_ref(),
            &splits,
            xsc.as_ref(),
            ysc.as_ref(),
            0,
        )
        .unwrap();
        let want = pls3_split_lv_correlations(
            x.as_ref(),
            y.as_ref(),
            &splits,
            xsc.as_ref(),
            ysc.as_ref(),
            0,
        )
        .unwrap();
        for j in 0..splits.len() {
            assert_eq!(got[j].to_bits(), 0.0_f64.to_bits(), "split {j}");
            assert_eq!(got[j].to_bits(), want[j].to_bits(), "split {j}");
        }
    }

    /// A held-out X score that is nearly constant but not constant to
    /// rounding is correlated by the streamed body as by the materialized
    /// fit. Column 0 of X carries the shared factor and is shifted by 1e6 on
    /// the held-out rows only, so the training half is ordinary and `σ₁`
    /// clears its floors, while the held-out X score is about 5e5 on every
    /// row plus variation of order one. Both blocks are divided by the
    /// all-row scales of the unshifted design, so the shift does not shrink
    /// column 0. The score's centered norm is about 2e-6 of its norm, so the
    /// two routes' rounding of the score, a few `ε` of its norm, moves `r` by
    /// about `ε` over that ratio: the two agree to `8·ε` over it (measured:
    /// 3e-12 against a bound of 6e-10).
    #[test]
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    fn pls3_split_nb_streamed_correlates_a_nearly_constant_held_out_score() {
        use crate::linalg::scaled_moments;
        let (n, p, q) = (60_usize, 6_usize, 3_usize);
        let (base, y) = linked_blocks(n, p, q, 2.0, 5);
        let (xsc, ysc) = scales(&base, &y);
        let x = Mat::<f64>::from_fn(n, p, |i, j| {
            if i >= 30 && j == 0 {
                base[(i, j)] + 1e6
            } else {
                base[(i, j)]
            }
        });
        let sp = SplitIdx {
            tr: (0..30).collect(),
            te: (30..60).collect(),
        };

        let (xs_tr, xs_te) = centered_halves(x.as_ref(), &sp.tr, &sp.te, xsc.as_ref());
        let (ys_tr, _) = centered_halves(y.as_ref(), &sp.tr, &sp.te, ysc.as_ref());
        let (u, _) = crate::pls3::pls3_lv1_prestd(xs_tr.as_ref(), xs_tr.norm_l2(), ys_tr.as_ref())
            .unwrap()
            .expect("premise: the training half keeps LV1");
        let s = crate::linalg::mat_vec(xs_te.as_ref(), u.as_ref(), faer::Par::Seq);
        let ms = scaled_moments(30, |i| s[i], None);
        let ratio = (ms.ss / ms.sq).sqrt();
        assert!(
            !ms.is_constant(30) && ratio < 1e-5,
            "premise: the held-out X score is nearly constant but not constant (ratio {ratio:e})"
        );

        let one = std::slice::from_ref(&sp);
        let want =
            pls3_split_lv_correlations(x.as_ref(), y.as_ref(), one, xsc.as_ref(), ysc.as_ref(), 0)
                .unwrap();
        assert!(
            want[0].is_finite() && want[0] != 0.0,
            "premise: the materialized fit reports a nonzero r, got {}",
            want[0]
        );
        let got = pls3_split_lv_correlations_streamed(
            x.as_ref(),
            y.as_ref(),
            one,
            xsc.as_ref(),
            ysc.as_ref(),
            0,
        )
        .unwrap();
        let bound = 8.0 * f64::EPSILON / ratio;
        assert!(
            (got[0] - want[0]).abs() <= bound,
            "streamed {} against materialized {}, bound {bound:e}",
            got[0],
            want[0]
        );
    }

    // ── `test_method = "auto"` ──────────────────────────────────────────

    /// At one seed `Auto` returns what an explicit `split_exact` call
    /// returns, field by field: on n = 60 (the sample-size rule) and on
    /// n = 300 with eight mostly-iid X columns, which the `split_nb` gate
    /// clears and PLS1 would run as `split_nb`. `stable_rank` stays unset.
    #[test]
    fn auto_matches_the_resolved_method_at_the_same_seed() {
        let run = |(x, y): &(Mat<f64>, Mat<f64>), args| {
            let o = Pls3ConfirmatoryTestOpts {
                args,
                seed: Some(13),
                ..Pls3ConfirmatoryTestOpts::default()
            };
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o).unwrap()
        };
        let auto = ConfirmatoryArgs::Auto {
            n_perm: 19,
            n_splits: 4,
        };
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 19,
            n_splits: 4,
        };
        let large = linked_blocks(300, 8, 3, 0.5, 9);
        assert!(
            !crate::signal_test::split_nb_gate(large.0.as_ref(), None, false)
                .unwrap()
                .fires
        );
        for data in [linked_blocks(60, 6, 3, 1.0, 5), large] {
            let got = run(&data, auto);
            assert_eq!(got.test_method, "split_exact");
            assert_eq!(got.stable_rank, None);
            assert_eq!(format!("{got:?}"), format!("{:?}", run(&data, exact)));
        }
    }

    /// `Auto` checks `split_exact`'s count floors.
    #[test]
    fn auto_count_floors_hold() {
        let (x, y) = linked_blocks(300, 8, 3, 0.5, 9);
        for (n_perm, n_splits) in [(0, 4), (19, 1)] {
            let o = Pls3ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::Auto { n_perm, n_splits },
                seed: Some(13),
                ..Pls3ConfirmatoryTestOpts::default()
            };
            let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o);
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "n_perm={n_perm} n_splits={n_splits}: {r:?}"
            );
        }
    }

    // ── split_exact dual (Gram) route tests ─────────────────────────────

    /// The equivalence test the PLS3 dual route rests on. Route A refits
    /// PLS3 honestly on each training half for every replicate column
    /// (`pls3_split_lv_correlations`, the shipped primal path); route B is
    /// `dual_route::pls3_split_zbars_columns`. Both get the same splits and
    /// the same permutations, so every one of the B+1 z̄ columns must agree.
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    fn assert_pls3_split_exact_dual_route_matches_honest_refit(p: usize) {
        use crate::signal_test::{draw_splits, mean_fisher_z};

        // n=60 ⇒ n_tr=30; q=3, B+1=50 ⇒ Bq=150. Dual is live when
        // 30·(150+p) < p·150, which the `split_exact_takes_gram` assert
        // below checks.
        let (n, q) = (60_usize, 3_usize);
        let (n_perm, n_splits, seed) = (49_usize, 6_usize, 11_u64);
        let (x, y) = linked_blocks(n, p, q, 3.0, 7);
        // Y's scale is the unpermuted Y's for every replicate, as in the runner.
        let (xsc, ysc) = scales(&x, &y);
        let n_cols = n_perm + 1;

        // Rebuild the runner's split draw and null permutations off the same seed.
        let splits = draw_splits(n, 1, n_splits, seed).unwrap();
        let perms: Vec<Vec<u32>> = (1..=n_perm)
            .map(|c| crate::rng::null_permutation(n, c, seed))
            .collect();
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed,
            n_perm,
            null_map: None,
            w: None,
        };

        let (n_tr, _) = crate::resample::split_sizes(n, 1);
        assert!(
            split_exact_takes_gram(n, p, q, n_perm, n_splits, false),
            "test premise: the runner must send this shape to the dual route \
             (n_tr={n_tr}, p={p}, q={q}, n_perm={n_perm})"
        );

        // Route A: honest per-split, per-column PLS3 refits.
        let mut a = Vec::with_capacity(n_cols);
        a.push(mean_fisher_z(
            &pls3_split_lv_correlations(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap(),
        ));
        for perm in &perms {
            let y_perm = Mat::<f64>::from_fn(n, q, |i, j| y[(perm[i] as usize, j)]);
            a.push(mean_fisher_z(
                &pls3_split_lv_correlations(
                    x.as_ref(),
                    y_perm.as_ref(),
                    &splits,
                    xsc.as_ref(),
                    ysc.as_ref(),
                    0,
                )
                .unwrap(),
            ));
        }

        // Route B: the Gram path, its splits one at a time (budget 0) and
        // all in one batch (no limit): the same bits.
        let run = |budget| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(budget, || {
                    crate::dual_route::pls3_split_zbars_columns(
                        &gram(&x, &xsc),
                        &cols,
                        &splits,
                        ysc.as_ref(),
                        0,
                    )
                    .unwrap()
                    .0
                })
            })
        };
        let ((b, log_one), (b_all, log_all)) = (run(0), run(u64::MAX));
        crate::resample::assert_batches(&log_one, n_splits, 1, 1);
        crate::resample::assert_batches(&log_all, n_splits, n_splits, 1);
        assert_eq!(
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b_all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the Gram route's bits depend on the budget"
        );

        assert_eq!(a.len(), b.len());
        for (col, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            let scale = av.abs().max(bv.abs());
            let rel = if scale == 0.0 {
                0.0
            } else {
                (av - bv).abs() / scale
            };
            assert!(rel < 1e-10, "col {col}: primal={av} dual={bv} rel={rel}");
        }

        let count = |v: &[f64]| {
            v[1..]
                .iter()
                .filter(|z| !z.is_finite() || **z >= v[0])
                .count()
        };
        assert_eq!(count(&a), count(&b), "exceedance counts split on a tie");
    }

    #[test]
    fn pls3_gram_rule_crosses_over_at_the_one_gram_cost() {
        // 60 x 40, q = 3, J = 6 (halves of 30): without the gate's G the
        // route pays from B' = 10; holding it, wherever n_tr < p.
        assert!(!split_exact_takes_gram(60, 40, 3, 8, 6, false));
        assert!(split_exact_takes_gram(60, 40, 3, 9, 6, false));
        // Many targets: the per-column `Ỹ'GỸ` and `q × q` eigendecomposition
        // keep 60 x 40, q = 200 primal at any B', held G or not.
        assert!(!split_exact_takes_gram(60, 40, 200, 999, 6, true));
        assert!(!split_exact_takes_gram(60, 40, 3, 5, 6, false));
        assert!(split_exact_takes_gram(60, 40, 3, 5, 6, true));
    }

    /// 60 x 36, q = 3, `n_perm` 49, J = 6: the one-G rule takes the Gram
    /// route, and every column's z̄ agrees with the primal route's to a
    /// relative 1e-12.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_gram_route_matches_the_primal_on_a_shape_the_one_gram_rule_adds() {
        let (n, p, q, n_perm, n_splits) = (60, 36, 3, 49, 6);
        assert!(split_exact_takes_gram(n, p, q, n_perm, n_splits, false));
        let (x, y) = linked_blocks(n, p, q, 3.0, 7);
        let (xsc, ysc) = scales(&x, &y);
        let splits = crate::signal_test::draw_splits(n, 1, n_splits, 11).unwrap();
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 11,
            n_perm,
            null_map: None,
            w: None,
        };
        let dual = crate::dual_route::pls3_split_zbars_columns(
            &gram(&x, &xsc),
            &cols,
            &splits,
            ysc.as_ref(),
            0,
        )
        .unwrap()
        .0;
        let primal = pls3_split_zbars_columns_primal(
            x.as_ref(),
            &cols,
            &splits,
            xsc.as_ref(),
            ysc.as_ref(),
            0,
        )
        .unwrap()
        .0;
        for (c, (a, b)) in dual.iter().zip(&primal).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
                "column {c}: {a:e} vs {b:e}"
            );
        }
    }

    #[test]
    fn pls3_split_exact_dual_route_matches_honest_refit_when_selected() {
        assert_pls3_split_exact_dual_route_matches_honest_refit(100);
    }

    #[test]
    fn pls3_split_exact_dual_route_matches_honest_refit_over_several_blocks() {
        // 30-row halves give a block of 8,736 columns: p=20,000 is three.
        assert_eq!(crate::linalg::stream_block_width(30), 8736);
        assert_pls3_split_exact_dual_route_matches_honest_refit(20_000);
    }

    /// The primal route's split driver on six splits of a 60 × 100 design,
    /// its splits one at a time (budget 0) and all in one
    /// batch (no limit): the same bits, column 0 being the observed z̄ of the
    /// split-by-split reference.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn primal_split_driver_gives_the_same_bits_at_every_budget() {
        use crate::signal_test::{draw_splits, mean_fisher_z};
        let (n, p, q) = (60_usize, 100_usize, 3_usize);
        let (n_perm, n_splits, seed) = (49_usize, 6_usize, 11_u64);
        let (x, y) = linked_blocks(n, p, q, 3.0, 7);
        let (xsc, ysc) = scales(&x, &y);
        let splits = draw_splits(n, 1, n_splits, seed).unwrap();
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed,
            n_perm,
            null_map: None,
            w: None,
        };
        let run = |b| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(b, || {
                    pls3_split_zbars_columns_primal(
                        x.as_ref(),
                        &cols,
                        &splits,
                        xsc.as_ref(),
                        ysc.as_ref(),
                        0,
                    )
                    .unwrap()
                    .0
                })
            })
        };
        let ((one, log_one), (all, log_all)) = (run(0), run(u64::MAX));
        crate::resample::assert_batches(&log_one, n_splits, 1, 1);
        crate::resample::assert_batches(&log_all, n_splits, n_splits, 1);
        assert_eq!(
            one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            all.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let z_obs = mean_fisher_z(
            &pls3_split_lv_correlations(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap(),
        );
        assert_eq!(one[0].to_bits(), z_obs.to_bits());
    }

    /// Degenerate training halves on the Gram route, decided from `G` alone.
    /// X is rank 3 with `p ≫ n_tr` (the only way a training `Ỹ` can be
    /// orthogonal to it there). With Y's training rows residualized on
    /// `[1, B_tr]`, `Ỹ_tr` is orthogonal to `X̃_tr` up to rounding: `σ₁` of
    /// `X̃_tr'Ỹ_tr` is rounding noise under the relative floor and the
    /// primal fit keeps nothing (`r = 0`), while the route's `λ_max` carries
    /// `G`'s rounding, many orders above the floor, so its value is only
    /// required to be finite. With Y's training rows constant, `Ỹ_tr` is
    /// exactly zero and both routes give exactly 0. The permutation columns
    /// are ordinary and agree with the primal to 1e-10.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_dual_route_decides_degenerate_training_halves_from_gram() {
        use rand::RngExt;
        use rand::SeedableRng;
        let (n, p, q, r) = (40, 200, 3, 3);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(61);
        let b = Mat::<f64>::from_fn(n, r, |_, _| rng.random_range(-1.0..1.0));
        let load = Mat::<f64>::from_fn(r, p, |_, _| rng.random_range(-1.0..1.0));
        let x: Mat<f64> = &b * &load;
        let tr: Vec<usize> = (0..n / 2).collect();
        let te: Vec<usize> = (n / 2..n).collect();
        let b_tr = Mat::<f64>::from_fn(tr.len(), r, |i, j| b[(tr[i], j)]);
        let basis = crate::test_support::orthonormal_basis(
            Col::<f64>::from_fn(tr.len(), |_| 1.0).as_ref(),
            b_tr.as_ref(),
            0.0,
        );
        let mut y = Mat::<f64>::from_fn(n, q, |_, _| rng.random_range(-1.0..1.0));
        for j in 0..q {
            let mut c = Col::<f64>::from_fn(tr.len(), |i| y[(tr[i], j)]);
            crate::test_support::project_off(&basis, &mut c);
            for (i, &row) in tr.iter().enumerate() {
                y[(row, j)] = c[i];
            }
        }
        let constant = Mat::<f64>::from_fn(n, q, |i, j| if i < n / 2 { 1.25 } else { y[(i, j)] });
        let splits = vec![SplitIdx { tr, te }];
        let xsc = crate::linalg::held_out_scale(x.as_ref(), None, false);
        let g = gram(&x, &xsc);
        assert!(split_exact_takes_gram(n, p, q, 9, 2, false));

        for (label, y) in [("orthogonal", &y), ("constant", &constant)] {
            let ysc = crate::linalg::held_out_scale(y.as_ref(), None, false);
            let cols = crate::resample::Columns {
                y: y.as_ref(),
                seed: 61,
                n_perm: 9,
                null_map: None,
                w: None,
            };
            let primal = pls3_split_zbars_columns_primal(
                x.as_ref(),
                &cols,
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap()
            .0;
            let dual =
                crate::dual_route::pls3_split_zbars_columns(&g, &cols, &splits, ysc.as_ref(), 0)
                    .unwrap()
                    .0;
            // Premise: the primal kept no component on column 0, and did
            // on the others.
            assert_eq!(primal[0].to_bits(), 0.0_f64.to_bits(), "{label}");
            assert!(primal[1..].iter().all(|z| *z != 0.0), "{label}");
            if label == "constant" {
                assert_eq!(dual[0].to_bits(), 0.0_f64.to_bits(), "{}", dual[0]);
            } else {
                assert!(dual[0].is_finite(), "{}", dual[0]);
            }
            for col in 1..dual.len() {
                let (a, d) = (primal[col], dual[col]);
                let rel = (a - d).abs() / a.abs().max(d.abs());
                assert!(rel < 1e-10, "{label} col {col}: primal={a} dual={d}");
            }
        }
    }

    /// Constant X drives every split's `X̃_tr` to the exact zero matrix, so
    /// `G` and `M` are the exact zero matrix too. On a fully degenerate X,
    /// both routes must agree and report exactly zero rather than NaN: the
    /// dual side reaches it because `M` is zero (independent of whatever
    /// eigenvector the degenerate eigendecomposition returns), the primal
    /// side through its own `k_used == 0` branch. The comparison against
    /// the honest per-split refit below checks that route agreement. On the
    /// dual side `λ_max = 0` is under the truncation floor, so it reports
    /// the zero model.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_dual_route_degenerate_x_gives_zero_not_nan() {
        use crate::signal_test::{draw_splits, mean_fisher_z};
        // 6 splits of 30 training rows at p = 5 and p = 400, one at a time
        // (budget 0) and all in one batch (no limit), with the same bits.
        for p in [5_usize, 400] {
            let x = Mat::<f64>::from_fn(60, p, |_, _| 1.25);
            let (_, y) = linked_blocks(60, p, 3, 3.0, 7);
            let (xsc, ysc) = scales(&x, &y);
            let splits = draw_splits(60, 1, 6, 4).unwrap();
            let perms = vec![crate::rng::null_permutation(60, 1, 4)];
            let cols = crate::resample::Columns {
                y: y.as_ref(),
                seed: 4,
                n_perm: 1,
                null_map: None,
                w: None,
            };
            let run = |b| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(b, || {
                        crate::dual_route::pls3_split_zbars_columns(
                            &gram(&x, &xsc),
                            &cols,
                            &splits,
                            ysc.as_ref(),
                            0,
                        )
                        .unwrap()
                        .0
                    })
                })
            };
            let ((z, log_one), (all, log_all)) = (run(0), run(u64::MAX));
            crate::resample::assert_batches(&log_one, 6, 1, 1);
            crate::resample::assert_batches(&log_all, 6, 6, 1);
            assert_eq!(
                z.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "p={p}"
            );
            for (col, v) in z.iter().enumerate() {
                assert!(v.is_finite(), "p={p}: col {col} not finite: {v}");
                assert!(v.abs() < 1e-12, "p={p}: col {col}: {v}");
            }

            // Route agreement, not just finiteness: the same honest per-split
            // refit the equivalence test uses, on the same splits and the same
            // permutation.
            let mut a = Vec::with_capacity(z.len());
            a.push(mean_fisher_z(
                &pls3_split_lv_correlations(
                    x.as_ref(),
                    y.as_ref(),
                    &splits,
                    xsc.as_ref(),
                    ysc.as_ref(),
                    0,
                )
                .unwrap(),
            ));
            for perm in &perms {
                let y_perm = Mat::<f64>::from_fn(60, 3, |i, j| y[(perm[i] as usize, j)]);
                a.push(mean_fisher_z(
                    &pls3_split_lv_correlations(
                        x.as_ref(),
                        y_perm.as_ref(),
                        &splits,
                        xsc.as_ref(),
                        ysc.as_ref(),
                        0,
                    )
                    .unwrap(),
                ));
            }
            assert_eq!(a.len(), z.len());
            for (col, (av, bv)) in a.iter().zip(z.iter()).enumerate() {
                let scale = av.abs().max(bv.abs());
                let rel = if scale == 0.0 {
                    0.0
                } else {
                    (av - bv).abs() / scale
                };
                assert!(
                    rel < 1e-10,
                    "p={p}: col {col}: primal={av} dual={bv} rel={rel}"
                );
            }
        }
    }

    /// Wiring test: the public runner, on an input the rule sends *dual*,
    /// must return exactly the p-value and statistic the primal branch
    /// would have returned at the same seed.
    ///
    /// The kernel-level equivalence test hands both routes the same
    /// locally-built permutations, so it cannot catch a column-0 convention
    /// slip or a permutation that drifts from what `rng::null_permutation`
    /// draws at the run seed. Here the reference comes
    /// from the runner's own primal machinery, so both of those separate the
    /// two p-values. A wrong routing argument (`n_tr`, `n_perm + 1`, `q`)
    /// flips the runner to primal instead, which the precondition assert
    /// below covers. The same input also runs dense `split_nb` (the
    /// streamed route), whose statistic must equal the dual `split_exact`
    /// one to the route tolerance.
    // The p-value is a count over a fixed denominator and must match exactly.
    #[allow(clippy::float_cmp)]
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_split_exact_dual_route_runner_matches_the_primal_reference() {
        use crate::signal_test::{draw_splits, mean_fisher_z};

        let (n, p, q) = (60_usize, 100_usize, 3_usize);
        let (n_perm, n_splits, seed) = (49_usize, 6_usize, 11_u64);
        let (x, y) = linked_blocks(n, p, q, 3.0, 7);
        let (xsc, ysc) = scales(&x, &y);

        assert!(
            split_exact_takes_gram(n, p, q, n_perm, n_splits, false),
            "test premise: this shape must take the dual route"
        );

        let o = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            ..Pls3ConfirmatoryTestOpts::default()
        };

        // Reference: the runner's pre-branch split draw, then its primal
        // branch, statement for statement.
        let splits = draw_splits(n, 1, n_splits, seed).unwrap();
        let z_obs = mean_fisher_z(
            &pls3_split_lv_correlations(
                x.as_ref(),
                y.as_ref(),
                &splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap(),
        );
        let nulls: Vec<f64> = (1..=n_perm)
            .map(|c| {
                let perm = crate::rng::null_permutation(n, c, seed);
                let y_perm = Mat::<f64>::from_fn(n, q, |i, j| y[(perm[i] as usize, j)]);
                mean_fisher_z(
                    &pls3_split_lv_correlations(
                        x.as_ref(),
                        y_perm.as_ref(),
                        &splits,
                        xsc.as_ref(),
                        ysc.as_ref(),
                        0,
                    )
                    .unwrap(),
                )
            })
            .collect();
        let exceedances = nulls
            .iter()
            .filter(|z| !z.is_finite() || **z >= z_obs)
            .count();
        let p_expected = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);

        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o).unwrap();

        assert_eq!(
            r.pvalue, p_expected,
            "dual runner p={} vs primal reference p={p_expected}",
            r.pvalue
        );
        // The runner reports tanh(z̄); compare on that scale.
        let expected_stat = z_obs.tanh();
        let scale = r.statistic.unwrap().abs().max(expected_stat.abs());
        let rel = if scale == 0.0 {
            0.0
        } else {
            (r.statistic.unwrap() - expected_stat).abs() / scale
        };
        assert!(
            rel < 1e-10,
            "statistic: dual={} primal={expected_stat} rel={rel}",
            r.statistic.unwrap()
        );

        // Dense split_nb reads X block by block; `force` keeps the gate
        // from deciding what runs.
        let nb = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(n_splits, true, seed))
            .unwrap();
        assert_eq!(nb.test_method, "split_nb");
        let rel = (nb.statistic.unwrap() - r.statistic.unwrap()).abs()
            / r.statistic.unwrap().abs().max(1e-300);
        assert!(
            rel < 1e-10,
            "split_nb={} split_exact(dual)={} rel={rel}",
            nb.statistic.unwrap(),
            r.statistic.unwrap()
        );
    }

    /// A split on the primal route counts its materialized halves and one
    /// LV1 fit's buffers and copies of Y per replicate column that can run
    /// at once: one per
    /// thread, at most the column count. A budget of two such units runs
    /// two splits at once, on a pool of 1 thread and on a pool of 4, for
    /// `split_exact`'s splits (8 columns) and for `split_nb`'s (1 column).
    #[test]
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    fn primal_splits_count_one_lv1_fit_per_running_column() {
        use crate::signal_test::{draw_splits, split_blocks_bytes};
        let (n, p, q) = (40_usize, 9_usize, 3_usize);
        let (x, y) = linked_blocks(n, p, q, 2.0, 5);
        let (xsc, ysc) = scales(&x, &y);
        let splits = draw_splits(n, 1, 4, 3).unwrap();
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 0,
            n_perm: 7,
            null_map: None,
            w: None,
        };
        let (n_tr, n_te) = (splits[0].tr.len(), splits[0].te.len());
        let lv1 = crate::pls3::lv1_bytes(p, q) + y_column_bytes(n_tr, n_te, q);
        let budget = |live: usize| {
            u64::try_from(2 * (split_blocks_bytes(n_tr, n_te, p) + live * lv1)).unwrap()
        };
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (_, log) = pool.install(|| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(budget(threads), || {
                        pls3_split_zbars_columns_primal(
                            x.as_ref(),
                            &cols,
                            &splits,
                            xsc.as_ref(),
                            ysc.as_ref(),
                            0,
                        )
                        .unwrap()
                        .0
                    })
                })
            });
            crate::resample::assert_batches(&log, 4, 2, 1);
            let (_, log) = pool.install(|| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(budget(1), || {
                        pls3_split_lv_correlations(
                            x.as_ref(),
                            y.as_ref(),
                            &splits,
                            xsc.as_ref(),
                            ysc.as_ref(),
                            0,
                        )
                        .unwrap()
                    })
                })
            });
            crate::resample::assert_batches(&log, 4, 2, 1);
        }
    }

    /// A tall `split_exact` call's design at one seed, as `pls3_test_body`
    /// draws it.
    fn tall_call(x: &Mat<f64>, n_splits: usize, seed: u64) -> SplitDesign {
        draw_design(x.nrows(), 1, n_splits, seed).unwrap()
    }

    /// Per (split, column), whether the cell route decides the value itself
    /// (`Some`) or leaves it to the primal route.
    #[allow(clippy::many_single_char_names)]
    fn cell_decisions(
        x: &Mat<f64>,
        cols: &crate::resample::Columns<'_>,
        d: &SplitDesign,
        (xsc, ysc): (&Col<f64>, &Col<f64>),
    ) -> Vec<Vec<Option<f64>>> {
        let q = cols.y.ncols();
        let frame =
            crate::signal_test::CellFrame::new(x.as_ref(), xsc.as_ref(), d, false, 0).unwrap();
        let sizes = (d.splits[0].tr.len(), d.splits[0].te.len());
        let levels = d.cells().trailing_zeros() as usize;
        crate::signal_test::cell_rows(
            &frame,
            d,
            cols.len(),
            q,
            0,
            &|c, out| write_y_block(cols, c, ysc.as_ref(), out),
            &|cs, j, c| pls3_cell_r(&frame, cs, j, c, sizes, levels),
        )
        .unwrap()
    }

    /// The cell route's z̄ against the primal route's on the same draws:
    /// every column within `1e-12·max(1, |z̄|)`, the same p-value, and the
    /// share of values the cell arm decides itself.
    #[allow(clippy::cast_precision_loss)]
    fn cells_against_primal(
        x: &Mat<f64>,
        y: &Mat<f64>,
        (n_splits, n_perm): (usize, usize),
        flags: (bool, bool),
        at: &str,
    ) -> f64 {
        let d = tall_call(x, n_splits, 17);
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 17,
            n_perm,
            null_map: None,
            w: None,
        };
        let xsc = crate::linalg::held_out_scale(x.as_ref(), None, flags.0);
        let ysc = crate::linalg::held_out_scale(y.as_ref(), None, flags.1);
        let (xs, ys) = (xsc.as_ref(), ysc.as_ref());
        let cells = pls3_split_zbars_columns_cells(x.as_ref(), &cols, &d, xs, ys, 0)
            .unwrap()
            .0;
        let primal = pls3_split_zbars_columns_primal(x.as_ref(), &cols, &d.splits, xs, ys, 0)
            .unwrap()
            .0;
        for (c, (a, b)) in cells.iter().zip(&primal).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "{at}: column {c}: cells {a} vs primal {b}"
            );
        }
        let pval = |z: &[f64]| z[1..].iter().filter(|v| **v >= z[0]).count();
        assert_eq!(pval(&cells), pval(&primal), "{at}: exceedances");
        let rows = cell_decisions(x, &cols, &d, (&xsc, &ysc));
        let decided = rows.iter().flatten().filter(|r| r.is_some()).count();
        decided as f64 / (rows.len() * cols.len()) as f64
    }

    /// On tall designs with cells the cell route gives the primal route's
    /// z̄ and p, with the cell arm deciding at least 90% of the values: n
    /// from 200 to 3001 (odd n and leftover rows included), p up to 50,
    /// q = 1, 2, 3, 5, J = 32, with and without the `pre_standardized`
    /// flags, and with Y or X offset by `1e7` (a mean far above the
    /// spread).
    #[test]
    fn cell_route_matches_the_primal_route_on_tall_designs() {
        for (n, p, q) in [
            (200, 8, 1),
            (200, 8, 3),
            (640, 2, 2),
            (1000, 20, 2),
            (1000, 50, 5),
            (3001, 5, 3),
        ] {
            let (x, y) = linked_blocks(n, p, q, 0.5, (n * p + q) as u64);
            for flags in [(false, false), (true, false), (false, true)] {
                let at = format!("{n} x {p}, q = {q}, flags {flags:?}");
                let share = cells_against_primal(&x, &y, (32, 39), flags, &at);
                assert!(share >= 0.9, "{at}: cell arm decided {share}");
            }
        }
        // Offsets: both routes centre the raw values before they scale.
        let (x, y) = linked_blocks(1000, 6, 3, 0.5, 9);
        let shift =
            |m: &Mat<f64>| Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] + 1e7);
        for (at, xo, yo) in [
            ("Y + 1e7", x.clone(), shift(&y)),
            ("X + 1e7", shift(&x), y.clone()),
        ] {
            let share = cells_against_primal(&xo, &yo, (32, 39), (false, false), at);
            assert!(share >= 0.9, "{at}: cell arm decided {share}");
        }
    }

    /// A tie of the first two singular values: with both flags (scale 1),
    /// split 0's centred training rows of X have orthogonal columns of
    /// equal norm and Y equals X there, so its `A` is `c·I` and `u` is set by
    /// rounding on every route. The guard leaves that value to the primal
    /// route, and z̄ stays the primal's.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn cell_route_leaves_a_singular_value_tie_to_the_primal() {
        let (n, p) = (300, 2);
        let (mut x, _) = linked_blocks(n, p, 2, 0.5, 13);
        let d = tall_call(&x, 32, 6);
        let tr = d.splits[0].tr.clone();
        // Centre split 0's training rows of X, then make the two columns
        // orthogonal and of norm 10 there (Gram-Schmidt, applied twice).
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
        let mut c: Vec<Vec<f64>> = (0..p)
            .map(|f| {
                let v: Vec<f64> = tr.iter().map(|&i| x[(i, f)]).collect();
                let m = v.iter().sum::<f64>() / v.len() as f64;
                v.into_iter().map(|e| e - m).collect()
            })
            .collect();
        for _ in 0..2 {
            let k = dot(&c[1], &c[0]) / dot(&c[0], &c[0]);
            let c0 = c[0].clone();
            c[1].iter_mut().zip(&c0).for_each(|(e, b)| *e -= k * b);
        }
        for col in &mut c {
            let norm = dot(col, col).sqrt();
            for e in col.iter_mut() {
                *e *= 10.0 / norm;
            }
        }
        for (f, col) in c.iter().enumerate() {
            for (i, &row) in tr.iter().enumerate() {
                x[(row, f)] = col[i];
            }
        }
        let y = Mat::<f64>::from_fn(n, p, |i, f| {
            if tr.contains(&i) {
                x[(i, f)]
            } else {
                ((i * 7 + f * 3) % 11) as f64 - 5.0
            }
        });
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 6,
            n_perm: 19,
            null_map: None,
            w: None,
        };
        let ones = Col::<f64>::from_fn(p, |_| 1.0);
        let rows = cell_decisions(&x, &cols, &d, (&ones, &ones));
        assert!(rows[0][0].is_none(), "split 0, column 0: {:?}", rows[0][0]);
        let (xs, ys) = (ones.as_ref(), ones.as_ref());
        let cells = pls3_split_zbars_columns_cells(x.as_ref(), &cols, &d, xs, ys, 0)
            .unwrap()
            .0;
        let primal = pls3_split_zbars_columns_primal(x.as_ref(), &cols, &d.splits, xs, ys, 0)
            .unwrap()
            .0;
        for (c, (a, b)) in cells.iter().zip(&primal).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "column {c}: {a} vs {b}"
            );
        }
    }

    /// A Y whose halves can overflow (column 0 at `−f64::MAX` and
    /// `f64::MAX` on two rows, so a training mean of the other sign makes
    /// `y − m_A` overflow and the primal route gives `r = 0` there) runs the
    /// whole call on the primal route: the result is the primal route's to
    /// the bit.
    #[test]
    fn cell_route_runs_an_overflowing_y_on_the_primal() {
        let (x, mut y) = linked_blocks(300, 6, 2, 0.5, 15);
        y[(3, 0)] = -f64::MAX;
        y[(200, 0)] = f64::MAX;
        let d = tall_call(&x, 32, 2);
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 2,
            n_perm: 19,
            null_map: None,
            w: None,
        };
        let (xsc, ysc) = scales(&x, &y);
        assert!(!y_halves_finite(y.as_ref(), ysc.as_ref()), "premise");
        let (xs, ys) = (xsc.as_ref(), ysc.as_ref());
        let cells = pls3_split_zbars_columns_cells(x.as_ref(), &cols, &d, xs, ys, 0)
            .unwrap()
            .0;
        let primal = pls3_split_zbars_columns_primal(x.as_ref(), &cols, &d.splits, xs, ys, 0)
            .unwrap()
            .0;
        assert_eq!(
            cells.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            primal.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// Where the sums cannot settle a value the guard hands it to the
    /// primal route: Y nonzero on one row only, so on every split one half
    /// of Y is constant (its centred sums are pure cancellation). The cell
    /// arm decides nothing and the result is the primal route's to the bit.
    #[test]
    fn cell_route_falls_back_where_the_sums_cannot_settle() {
        let (x, _) = linked_blocks(300, 6, 2, 0.5, 8);
        let y = Mat::<f64>::from_fn(300, 2, |i, k| if i == 17 { 1.0 + k as f64 } else { 0.0 });
        let d = tall_call(&x, 32, 3);
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 3,
            n_perm: 19,
            null_map: None,
            w: None,
        };
        let (xsc, ysc) = scales(&x, &y);
        let rows = cell_decisions(&x, &cols, &d, (&xsc, &ysc));
        assert!(rows.iter().flatten().all(Option::is_none), "{rows:?}");
        let (xs, ys) = (xsc.as_ref(), ysc.as_ref());
        let cells = pls3_split_zbars_columns_cells(x.as_ref(), &cols, &d, xs, ys, 0)
            .unwrap()
            .0;
        let primal = pls3_split_zbars_columns_primal(x.as_ref(), &cols, &d.splits, xs, ys, 0)
            .unwrap()
            .0;
        assert_eq!(
            cells.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            primal.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// A column at the truncation floor: on split 0 the observed Y's
    /// training rows are orthogonal to `[1, X_tr]` up to rounding, so `σ₁`
    /// of its `A` is rounding noise. The guard leaves that value to the
    /// primal route, which applies its own floor, and z̄ stays the primal's.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn cell_route_leaves_a_column_at_the_truncation_floor_to_the_primal() {
        let (n, p) = (400, 5);
        let (x, y0) = linked_blocks(n, p, 2, 0.5, 21);
        let d = tall_call(&x, 32, 4);
        let tr = &d.splits[0].tr;
        // The training rows of y0 projected off `[1, X_tr]` (Gram-Schmidt,
        // each projection applied twice).
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
        let off = |v: &mut Vec<f64>, basis: &[Vec<f64>]| {
            for _ in 0..2 {
                for b in basis {
                    let c = dot(v, b);
                    v.iter_mut().zip(b).for_each(|(e, bi)| *e -= c * bi);
                }
            }
        };
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for f in 0..=p {
            let mut v: Vec<f64> = tr
                .iter()
                .map(|&i| if f == p { 1.0 } else { x[(i, f)] })
                .collect();
            off(&mut v, &basis);
            let norm = dot(&v, &v).sqrt();
            basis.push(v.into_iter().map(|e| e / norm).collect());
        }
        let mut y = y0.clone();
        for k in 0..2 {
            let mut v: Vec<f64> = tr.iter().map(|&i| y0[(i, k)]).collect();
            off(&mut v, &basis);
            for (i, &row) in tr.iter().enumerate() {
                y[(row, k)] = v[i];
            }
        }
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 4,
            n_perm: 19,
            null_map: None,
            w: None,
        };
        let (xsc, ysc) = scales(&x, &y);
        let rows = cell_decisions(&x, &cols, &d, (&xsc, &ysc));
        assert!(rows[0][0].is_none(), "split 0, column 0: {:?}", rows[0][0]);
        let (xs, ys) = (xsc.as_ref(), ysc.as_ref());
        let cells = pls3_split_zbars_columns_cells(x.as_ref(), &cols, &d, xs, ys, 0)
            .unwrap()
            .0;
        let primal = pls3_split_zbars_columns_primal(x.as_ref(), &cols, &d.splits, xs, ys, 0)
            .unwrap()
            .0;
        for (c, (a, b)) in cells.iter().zip(&primal).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "column {c}: {a} vs {b}"
            );
        }
    }

    /// The runner takes the cell route on tall X with cells at the default
    /// budget, and not with fewer rows than cells, on a budget the route
    /// does not fit, or with the Gram routes taken out of the choice; at the
    /// public entry the routes give the same statistic and p.
    #[test]
    fn cell_route_is_taken_where_eligible() {
        let (n_splits, n_perm) = (32, 99);
        let (x, y) = linked_blocks(500, 6, 3, 0.6, 31);
        let d = tall_call(&x, n_splits, 1);
        let small = tall_call(&x.subrows(0, 40).to_owned(), n_splits, 1);
        assert!(
            d.m > 0 && small.m == 0,
            "premise: cells at n = 500, none at 40"
        );
        let cells = |n| {
            pls3_split_exact_route(n, 6, 3, n_perm, n_splits, false)
                .unwrap()
                .route
                == CELL_ROUTE
        };
        assert!(cells(500));
        assert!(!cells(40));
        // The least budget the primal route runs at, from the crate's own
        // byte counts (they differ by platform), does not hold the cells.
        let s = pls3_shape(500, 6, n_perm + 1, n_splits, false);
        let need = |(call, unit): (usize, usize)| call + unit;
        let primal = |threads| {
            let unit = primal_split_unit_bytes(s.n_train, s.n_test, 6, 3, s.n_cols, threads);
            need((design_bytes(&s), unit))
        };
        let budget = primal(1).max(primal(rayon::current_num_threads()));
        assert!(
            need(cells_route_bytes(&s, 3, 1)) > budget,
            "premise: the cells need more than {budget} bytes"
        );
        assert!(!crate::fit::with_budget(
            u64::try_from(budget).unwrap(),
            || cells(500)
        ));
        assert!(!crate::signal_test::with_gram_routes_disabled(|| cells(
            500
        )));
        // A large tall call, 10^6 x 20 at J = 32 and q = 1: its frame and one
        // batch fit the default budget.
        assert_eq!(
            pls3_split_exact_route(1_000_000, 20, 1, 31, 32, false)
                .unwrap()
                .route,
            CELL_ROUTE
        );
        let run = || {
            let before = crate::cell_sums::cell_grams_built();
            let out = pls3_confirmatory_test_impl(
                x.as_ref(),
                y.as_ref(),
                1,
                Pls3ConfirmatoryTestOpts {
                    pre_standardized_x: true,
                    ..opts(n_perm, n_splits, 1)
                },
            )
            .unwrap();
            (out, crate::cell_sums::cell_grams_built() - before)
        };
        let (cells, built) = run();
        assert_eq!(built, 1, "the runner builds one cell frame");
        let (primal, built) = crate::signal_test::with_gram_routes_disabled(run);
        assert_eq!(built, 0, "the primal route builds none");
        assert!(
            (cells.statistic.unwrap() - primal.statistic.unwrap()).abs()
                <= 1e-12 * primal.statistic.unwrap().abs(),
            "cells {} vs primal {}",
            cells.statistic.unwrap(),
            primal.statistic.unwrap()
        );
        assert_eq!(cells.pvalue.to_bits(), primal.pvalue.to_bits());
    }

    // PLS3 `split_nb` on wide X reads its splits from the gate's G where
    // `pls3_split_nb_route` takes the Gram route (q = 3): its statistic is
    // the `gram=` call's, not the streamed one's bits. Where the rule says
    // primal (q = 200) it streams X. Either way the gate's G is the only
    // one built.
    #[test]
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    fn pls3_split_nb_reads_the_gate_gram() {
        let (n, p, n_splits) = (60, 200, 8);
        let opts = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitNb {
                n_splits,
                force: true,
            },
            seed: Some(31),
            ..Default::default()
        };
        for q in [3, 200] {
            let (x, y) = linked_blocks(n, p, q, 1.0, 31);
            let before = crate::held_out_gram::grams_built();
            let a = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts).expect("X call");
            assert_eq!(crate::held_out_gram::grams_built(), before + 1, "q = {q}");
            assert_eq!(a.test_method, "split_nb");
            let stat = a.statistic.unwrap();
            let d = draw_design(n, 1, n_splits, 31).unwrap();
            let xsc = crate::linalg::held_out_scale(x.as_ref(), None, false);
            let ysc = crate::linalg::held_out_scale(y.as_ref(), None, false);
            let r = pls3_split_lv_correlations_streamed(
                x.as_ref(),
                y.as_ref(),
                &d.splits,
                xsc.as_ref(),
                ysc.as_ref(),
                0,
            )
            .unwrap();
            let (n_tr, n_te) = crate::resample::split_sizes(n, 1);
            let streamed = nb_test(&r, n_tr, n_te).1;
            if q == 3 {
                assert!(split_nb_takes_gram(n, p, q, n_splits));
                let g = crate::held_out_gram::gram_matrix(x.as_ref(), false).expect("gram");
                let b = pls3_confirmatory_test_gram(&g, y.as_ref(), 1, opts).expect("gram call");
                assert!((stat - b.statistic.unwrap()).abs() < 1e-12, "{stat}");
                assert_ne!(stat.to_bits(), streamed.to_bits(), "the Gram route");
            } else {
                assert!(!split_nb_takes_gram(n, p, q, n_splits));
                assert_eq!(stat.to_bits(), streamed.to_bits(), "the streamed route");
            }
        }
    }

    fn sv_opts(n_perm: usize, rotate: bool, seed: u64) -> Pls3ConfirmatoryTestOpts {
        Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SvPerm { n_perm, rotate },
            seed: Some(seed),
            ..Pls3ConfirmatoryTestOpts::default()
        }
    }

    /// `sv_perm` on tall X, where the Gram route is not on the menu, holds
    /// X̃ (`n × p`), Ỹ and one column's permuted block of it (`n × q` each)
    /// and the column's `C = X̃'Ỹ_π` (`p × q`): a budget one byte below their
    /// sum is `MemoryBudget` naming it, the sum runs.
    #[test]
    fn sv_perm_counts_x_tilde() {
        use crate::resample::copy_bytes;
        let (x, y) = linked_blocks(40, 12, 4, 1.0, 3);
        let ys = copy_bytes(40, 4);
        let needed = u64::try_from(copy_bytes(40, 12) + copy_bytes(12, 4) + 2 * ys).unwrap();
        let run = |budget| {
            crate::fit::with_budget(budget, || {
                pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(19, false, 3))
            })
        };
        let r = run(needed - 1);
        assert!(
            matches!(r, Err(PlsKitError::MemoryBudget { needed: n, .. }) if n == needed),
            "{r:?}"
        );
        assert!(run(needed).is_ok());
    }

    /// Too few rows is `InvalidArgument` at any budget, never
    /// `MemoryBudget`: the split floor runs before the route check, on
    /// `split_exact` and `split_nb`, here and in PLS1.
    #[test]
    fn too_few_rows_is_invalid_argument_at_a_tiny_budget() {
        let (x, y) = linked_blocks(5, 8, 2, 1.0, 3);
        let y1 = Col::<f64>::from_fn(5, |i| y[(i, 0)]);
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: 9,
            n_splits: 4,
        };
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: 4,
            force: true,
        };
        for args in [exact, nb] {
            let pls3 = crate::fit::with_budget(1, || {
                let o = Pls3ConfirmatoryTestOpts {
                    args,
                    seed: Some(3),
                    ..Pls3ConfirmatoryTestOpts::default()
                };
                pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o)
            });
            let pls1 = crate::fit::with_budget(1, || {
                crate::pls1_confirmatory_test(
                    crate::signal_test::ConfirmatoryTestInput::Raw {
                        x: x.as_ref(),
                        y: y1.as_ref(),
                        k: Some(&[1]),
                        weights: None,
                        covariates: None,
                        null_y: None,
                    },
                    crate::signal_test::ConfirmatoryTestOpts {
                        args,
                        seed: Some(3),
                        ..Default::default()
                    },
                )
            });
            for (label, r) in [("pls3", pls3.map(|_| ())), ("pls1", pls1.map(|_| ()))] {
                assert!(
                    matches!(r, Err(PlsKitError::InvalidArgument(_))),
                    "{label} {args:?}: {r:?}"
                );
            }
        }
    }

    #[test]
    fn pls3_sv_perm_statistic_is_top_singular_value() {
        let (x, y) = linked_blocks(40, 12, 4, 1.0, 3);
        assert!(!sv_perm_takes_gram(40, 12, 4, 99), "premise: primal route");
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(99, false, 3)).unwrap();
        assert_eq!(r.test_method, "sv_perm");
        assert_eq!(r.n_perm, Some(99));
        assert_eq!(r.n_splits, None);
        assert_eq!((r.n_folds, r.force, r.rotate), (None, None, Some(false)));
        assert_eq!(r.stable_rank, None);
        assert_eq!(r.n_covariates, None);
        assert!((r.n_eff - 40.0).abs() < 1e-12);
        let xs = crate::linalg::standardize(x.as_ref()).0;
        let ys = crate::linalg::standardize(y.as_ref()).0;
        let c = xs.transpose() * &ys;
        let s0 = crate::linalg::thin_svd(c.as_ref(), faer::Par::Seq)
            .unwrap()
            .s[0];
        let rel = (r.statistic.unwrap() / s0 - 1.0).abs();
        assert!(rel < 1e-10, "rel {rel}");
    }

    #[test]
    fn pls3_sv_perm_rotation_never_raises_p() {
        let floor = 1.0 / 200.0;
        let mut seen_strict_drop = false;
        for seed in 0..5 {
            let (x, y) = linked_blocks(30, 20, 5, 0.4, 100 + seed);
            let un =
                pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(199, false, 9)).unwrap();
            let rot =
                pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(199, true, 9)).unwrap();
            assert_eq!(rot.statistic, un.statistic);
            assert_eq!((rot.rotate, un.rotate), (Some(true), Some(false)));
            assert!(
                rot.pvalue <= un.pvalue,
                "seed {seed}: {} > {}",
                rot.pvalue,
                un.pvalue
            );
            seen_strict_drop |= un.pvalue > floor && rot.pvalue < un.pvalue;
        }
        assert!(
            seen_strict_drop,
            "no seed had p off the floor with rotation lowering it"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // p-values are counts over the same nulls
    fn pls3_sv_perm_q1_rotation_is_identity() {
        let (x, y) = linked_blocks(30, 6, 1, 0.5, 4);
        let un = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(199, false, 2)).unwrap();
        let rot = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(199, true, 2)).unwrap();
        assert_eq!(rot.pvalue, un.pvalue);
    }

    #[test]
    fn pls3_sv_perm_refusals() {
        let (x, y) = linked_blocks(40, 6, 3, 1.0, 5);
        let bad = |o: Pls3ConfirmatoryTestOpts, k: usize| {
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), k, o)
        };
        assert!(bad(sv_opts(19, false, 1), 2).is_err());
        assert!(matches!(
            bad(sv_opts(0, false, 1), 1),
            Err(PlsKitError::InvalidArgument(_))
        ));
        let score = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::Score { n_perm: Some(19) },
            seed: Some(1),
            ..Pls3ConfirmatoryTestOpts::default()
        };
        assert!(matches!(
            bad(score, 1),
            Err(PlsKitError::InvalidArgument(_))
        ));
    }

    /// `q` above `n_features` keeps `r = n_features` eigenpairs on both sides.
    #[test]
    fn pls3_sv_perm_q_above_p_runs() {
        let (x, y) = linked_blocks(30, 2, 5, 1.0, 6);
        for rotate in [false, true] {
            let r =
                pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(99, rotate, 3)).unwrap();
            assert!(
                r.pvalue > 0.0 && r.pvalue <= 1.0,
                "rotate {rotate}: {}",
                r.pvalue
            );
        }
    }

    /// A constant Y column answers as `split_exact` answers it.
    #[test]
    fn pls3_sv_perm_constant_y_column_answers_as_split_exact() {
        let (x, mut y) = linked_blocks(40, 6, 3, 1.0, 7);
        for i in 0..40 {
            y[(i, 1)] = 2.5;
        }
        let se = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(99, 8, 1));
        for rotate in [false, true] {
            let sv = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(99, rotate, 1));
            match (&se, sv) {
                (Err(a), Err(b)) => assert_eq!(
                    std::mem::discriminant(a),
                    std::mem::discriminant(&b),
                    "{a:?} vs {b:?}"
                ),
                (Ok(_), Ok(b)) => {
                    assert!(b.statistic.unwrap().is_finite());
                    assert!(b.pvalue > 0.0 && b.pvalue <= 1.0);
                }
                (a, b) => panic!("split_exact {a:?} vs sv_perm {b:?}"),
            }
        }
    }

    /// Wide X with `p` far above `n` (the Gram route) and rotation at
    /// `r = q = 4`, the closed form: the p-value and statistic are finite.
    #[test]
    fn pls3_sv_perm_wide_rank_deficient_rotation_is_finite() {
        let (x, y) = linked_blocks(20, 200, 4, 1.0, 8);
        assert!(sv_perm_takes_gram(20, 200, 4, 99), "premise: Gram route");
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(99, true, 3)).unwrap();
        assert!(r.pvalue.is_finite() && r.statistic.unwrap().is_finite());
    }

    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_sv_perm_pre_standardized_matches_default() {
        let (x, y) = linked_blocks(40, 12, 4, 1.0, 9);
        let xs = crate::linalg::standardize(x.as_ref()).0;
        let ys = crate::linalg::standardize(y.as_ref()).0;
        let a = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sv_opts(99, false, 3)).unwrap();
        let o = Pls3ConfirmatoryTestOpts {
            pre_standardized_x: true,
            pre_standardized_y: true,
            ..sv_opts(99, false, 3)
        };
        let b = pls3_confirmatory_test(xs.as_ref(), ys.as_ref(), 1, o).unwrap();
        let rel = (a.statistic.unwrap() / b.statistic.unwrap() - 1.0).abs();
        assert!(rel < 1e-10, "rel {rel}");
    }

    /// The rotated null value against `‖D_π R[:, 0]‖`, with R built here in
    /// pyls's form: `svd(V_obsᵀV_π) = N S Pᵀ`, `R = P Nᵀ`. At r = q the
    /// helper takes its closed form, at r < q the SVD.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_sv_perm_rotated_null_matches_explicit_procrustes() {
        use rand::RngExt;
        use rand::SeedableRng;
        let seq = faer::Par::Seq;
        for (q, r) in [(4_usize, 4_usize), (5, 2)] {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(40 + q as u64);
            let mut psd = |rows: usize| {
                let a = Mat::<f64>::from_fn(rows, q, |_, _| rng.random_range(-1.0..1.0));
                a.transpose() * &a
            };
            let m_obs = psd(r);
            let m_pi = psd(r);
            let top = |m: &Mat<f64>| {
                let (l, v) = crate::linalg::self_adjoint_eigen(m.as_ref(), seq).unwrap();
                let lam: Vec<f64> = (0..r).map(|j| l[q - 1 - j]).collect();
                (lam, Mat::<f64>::from_fn(q, r, |i, j| v[(i, q - 1 - j)]))
            };
            let (_, v_obs) = top(&m_obs);
            let (lam, v_pi) = top(&m_pi);
            let a = v_obs.transpose() * &v_pi;
            let svd = crate::linalg::thin_svd(a.as_ref(), seq).unwrap();
            let rot = &svd.v * svd.u.transpose();
            let want = (0..r)
                .map(|j| lam[j] * rot[(j, 0)].powi(2))
                .sum::<f64>()
                .sqrt();
            let got = sv_null(m_pi.as_ref(), Some(v_obs.as_ref())).unwrap();
            assert!(
                (got / want - 1.0).abs() < 1e-10,
                "q {q} r {r}: {got} vs {want}"
            );
        }
    }

    /// The PLS1 test of `n_failed_splits` with every column of Y the same
    /// two-row indicator: a half with neither positive row has a constant Y
    /// block (no first component on a training half, a constant test half),
    /// so its `r` is exactly 0. The test counts those splits on the crate's
    /// draw at the call's seed, on the primal, cell and Gram routes of
    /// `split_exact`, and through the public calls on X and on its Gram
    /// matrix for `split_exact` and `split_nb`. A continuous Y gives
    /// `Some(0)`; `sv_perm` gives `None`.
    #[test]
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    fn pls3_n_failed_splits_counts_splits_with_a_half_without_positives() {
        const N_SPLITS: usize = 32;
        const N_PERM: usize = 19;
        const SEED: u64 = 23;
        let pos = [3_usize, 17];
        let has = |idx: &[usize]| idx.iter().any(|i| pos.contains(i));
        let want = |n: usize| {
            let want = crate::signal_test::draw_splits(n, 1, N_SPLITS, SEED)
                .unwrap()
                .iter()
                .filter(|sp| !(has(&sp.tr) && has(&sp.te)))
                .count();
            assert!(want > 0 && want < N_SPLITS, "premise: {want}");
            want
        };
        let data = |n: usize| {
            let (x, _) = linked_blocks(n, 6, 3, 0.5, 5);
            let y = Mat::<f64>::from_fn(n, 3, |i, _| if pos.contains(&i) { 1.0 } else { 0.0 });
            (x, y)
        };
        let exact = ConfirmatoryArgs::SplitExact {
            n_perm: N_PERM,
            n_splits: N_SPLITS,
        };
        let nb = ConfirmatoryArgs::SplitNb {
            n_splits: N_SPLITS,
            force: true,
        };
        // The public calls: on X, on X with the Gram routes out of the
        // choice, and on the Gram matrix of X.
        let calls = |x: &Mat<f64>, y: &Mat<f64>, args: ConfirmatoryArgs| {
            let o = || Pls3ConfirmatoryTestOpts {
                args,
                seed: Some(SEED),
                ..Pls3ConfirmatoryTestOpts::default()
            };
            let g = crate::gram_matrix(x.as_ref(), false).unwrap();
            [
                pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o()).unwrap(),
                crate::signal_test::with_gram_routes_disabled(|| {
                    pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o()).unwrap()
                }),
                pls3_confirmatory_test_gram(&g, y.as_ref(), 1, o()).unwrap(),
            ]
            .map(|out| out.n_failed_splits)
        };

        for n in [40, 80] {
            let (x, y) = data(n);
            let (xsc, ysc) = scales(&x, &y);
            let (xs, ys) = (xsc.as_ref(), ysc.as_ref());
            let d = draw_design(n, 1, N_SPLITS, SEED).unwrap();
            let cols = crate::resample::Columns {
                y: y.as_ref(),
                seed: SEED,
                n_perm: N_PERM,
                null_map: None,
                w: None,
            };
            let mut got = vec![
                (
                    "primal",
                    pls3_split_zbars_columns_primal(x.as_ref(), &cols, &d.splits, xs, ys, 0)
                        .unwrap()
                        .1,
                ),
                (
                    "Gram",
                    crate::dual_route::pls3_split_zbars_columns(
                        &gram(&x, &xsc),
                        &cols,
                        &d.splits,
                        ys,
                        0,
                    )
                    .unwrap()
                    .1,
                ),
            ];
            if n == 80 {
                assert!(d.m > 0, "premise: cells at n = 80");
                let z = pls3_split_zbars_columns_cells(x.as_ref(), &cols, &d, xs, ys, 0);
                got.push(("cells", z.unwrap().1));
            }
            let want = want(n);
            for (label, count) in got {
                assert_eq!(count, want, "n = {n}, {label}");
            }
            for args in [exact, nb] {
                for (i, count) in calls(&x, &y, args).into_iter().enumerate() {
                    assert_eq!(count, Some(want), "n = {n}, {args:?}, call {i}");
                }
            }
        }

        // A continuous Y fails no split; sv_perm draws none.
        let (x, y) = linked_blocks(40, 6, 3, 0.5, 5);
        for args in [exact, nb] {
            assert_eq!(calls(&x, &y, args), [Some(0); 3], "{args:?}");
        }
        let sv = ConfirmatoryArgs::SvPerm {
            n_perm: N_PERM,
            rotate: false,
        };
        assert_eq!(calls(&x, &y, sv), [None; 3]);
    }
}
