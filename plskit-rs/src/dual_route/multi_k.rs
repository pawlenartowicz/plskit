//! n-space Gram route for PLS1 fits of `k ≥ 1` components.
//!
//! Three PLS1 loops refit on a fixed standardized block `X̃` while only the
//! outcome changes between replicates: `pls1_perm_null`, the `raw_perm`
//! folds and the `split_exact` refit route. At `p ≫ n` each refit costs
//! `O(k·n·p)`. Every quantity of the PLS1 kernel stays in the row space of
//! `X̃`, and deflation acts in n-space, so the whole kernel runs from
//! `G = X̃X̃'`, built once per block, at `O(k·n²)` per replicate (see
//! [`pls1_nspace_kernel`]).
//!
//! The route has no loop of its own. The replicate drivers in
//! `signal_test.rs` and `perm_null.rs` select it per call
//! (`ReplicateRoute::Nspace`), build one block per fold, split or call from
//! the constructors here, and run the unit bodies here per replicate, every
//! unit decided from `G` alone. The `raw_perm` folds and the `split_exact` and `split_nb` splits take their
//! blocks from one all-row Gram matrix per call
//! ([`NspaceFold::from_gram`], [`NspaceSplit::from_gram`]); `pls1_perm_null`
//! builds its one block from `X̃`.
//! `perm_alpha::pls1_perm_alpha` is the one caller without an `X̃`: it
//! builds its block from the caller's Gram matrix
//! ([`NspaceGram::from_gram`]).
//!
//! The parent module's `pls1_cv_r2_columns` (the `K = 1` closed form at
//! `raw_perm`) and `signal_test::split_perm_nr_zbars` (the `K = 1` no-refit
//! route at `split_exact`) keep their own kernels: this route serves `k ≥ 2` at
//! those two sites and at `split_nb`, and every `k` up to [`K_DUAL_MAX`] at `pls1_perm_null`,
//! which has no other Gram route. Dense, unweighted fits only: hard `keep`
//! selection acts on the full p-length weight vector, and weighted fits
//! stay primal.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColMut, ColRef, Mat, MatRef, Par};

use super::{DUAL_ROUTE_MAX_N_TR, one_gram_cost, use_dual_route};
use crate::fit::{NIPALS_ABS_FLOOR, w_rel_floor};
use crate::held_out_gram::HeldOutGram;
use crate::linalg::standardize1;
#[cfg(test)]
use crate::signal_test::{CvFold, PreparedSplit};
use crate::signal_test::{SplitIdx, guarded_pearson};

/// Largest component count the n-space route serves: the largest `k ≤ 10`
/// up to which, on every design family of the accuracy sweep (`sweep.rs`,
/// `nspace_matches_pspace_up_to_k_dual_max`), the n-space fit from the
/// all-row Gram matrix's block meets `|Δcoef| ≤ 1e-10·max(1, ‖coef‖_∞)` and
/// `|Δr| ≤ 1e-12` against `pls1_fit` on the same scaled X, on every fit where
/// both keep the same number of components. Past it, wide `k` refits in
/// p-space. Worst errors per `k` (family):
///
/// | k | worst `|Δcoef|/max(1, ‖coef‖_∞)` | worst `|Δr|` |
/// |---|---|---|
/// | 1 | 4.3e-16 (`tiny_scale`) | 1.0e-15 (`small_n`) |
/// | 2 | 1.0e-15 (`tiny_scale`) | 9.4e-16 (`ordinary`) |
/// | 3 | 1.0e-15 (`tiny_scale`) | 6.8e-15 (`y_orthogonal`) |
/// | 4 | 7.4e-15 (`y_exhausted`) | 2.0e-14 (`y_orthogonal`) |
/// | 5 | 1.2e-15 (`tiny_scale`) | 4.5e-14 (`y_orthogonal`) |
/// | 6 | 1.5e-15 (`tiny_scale`) | 2.4e-15 (`y_orthogonal`) |
/// | 7 | 1.5e-15 (`tiny_scale`) | 2.4e-15 (`y_orthogonal`) |
/// | 8 | 1.6e-15 (`tiny_scale`) | 2.4e-15 (`y_orthogonal`) |
/// | 9 | 6.0e-15 (`rank_deficient`) | 2.4e-15 (`y_orthogonal`) |
/// | 10 | 1.1e-13 (`rank_deficient`) | 2.9e-15 (`rank_deficient`) |
///
/// Fits where the two keep different component counts, all at the floor,
/// where `z'Gz` resolves `‖X̃'z‖` only to `G`'s rounding (see "Truncation"
/// on [`pls1_nspace_kernel`]):
///
/// - `y_orthogonal` design 0, all rows, `k = 1..=10`: the n-space fit keeps
///   `min(k, 6)` components, the p-space fit 0, `|Δcoef|/max(1, ‖coef‖_∞)`
///   1.7e-4 to 2.1e-4; its split half, `k = 6..=10`: 6 against 5,
///   `|Δr|` = 5.4e-1. Design 1 agrees (both zero models).
/// - `y_exhausted` design 0 (X̃ of rank 4), `k = 5..=10`: 5 against 4 on
///   all rows (`|Δcoef|/max(1, ‖coef‖_∞)` = 2.2e-3) and on its split half
///   (`|Δr|` = 1.7e-1).
pub(crate) const K_DUAL_MAX: usize = 10;

/// Should `pls1_perm_null` take the n-space route?
///
/// `n × p` is the full standardized block, `n_perm` the number of
/// permutations (the observed fit stays primal, so it does not amortize the
/// Gram), `k` the component count. `dense_unweighted` is `true` when no
/// weights reached the engine (`pls1_perm_null` takes no `keep`). Requires
/// `1 ≤ k ≤ K_DUAL_MAX`, `k ≤ n − 1` (the rank of `X̃` after centering) and
/// the flop rule [`use_dual_route`] with `k` as its per-replicate
/// multiplicity. The β output adds one `O(n·p)` GEMV per permutation, at
/// most one primal sweep, so the rule stays conservative here.
pub(crate) fn nspace_eligible_perm_null(
    n: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    dense_unweighted: bool,
) -> bool {
    dense_unweighted && (1..=K_DUAL_MAX).contains(&k) && k < n && use_dual_route(n, p, n_perm, k)
}

/// Should `raw_perm` take the n-space route?
///
/// `k = 1` stays on the closed-form route (`pls1_cv_r2_columns`), so this
/// starts at `k = 2`. `fold_split` makes folds that differ by at most one
/// row: the largest training fold (`n − ⌊n/n_folds⌋`) feeds the cost and
/// memory rule, so one route serves the whole statistic, and the smallest
/// (`n − ⌈n/n_folds⌉`) must hold `k + 1` rows, so `k ≤ rank(X̃_tr)` on
/// every fold: past the rank every component sits at the floor, where the
/// kernel's decision is only as good as `G`'s rounding.
///
/// The route builds one all-row Gram matrix for the call and reads each
/// fold's blocks from it, so it is taken when, with `F = n_folds`,
/// `B' = n_perm + 1`, `n_tr` the largest training fold and
/// `n_val = n − n_tr`,
///
/// ```text
/// n²·p/2 + F·B'·(k·n_tr² + n_val·n_tr)  <  F·B'·(2k·n_tr + n_val)·p
/// ```
///
/// in multiply-adds: per column and fold one `G·r` per component and
/// `M·alpha`, against the
/// primal's `X̃'z`, a score and a loading pass per component (the last
/// loading not counted) and `X̃_val·coef`. Also `n_tr < p` and
/// `n_tr ≤ DUAL_ROUTE_MAX_N_TR`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub(crate) fn nspace_eligible_raw_perm(
    n: usize,
    n_folds: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    dense_unweighted: bool,
) -> bool {
    if !dense_unweighted || !(2..=K_DUAL_MAX).contains(&k) || !(2..=n).contains(&n_folds) {
        return false;
    }
    let n_tr = n - n / n_folds;
    let n_tr_min = n - n.div_ceil(n_folds);
    if k >= n_tr_min || n_tr >= p || n_tr > DUAL_ROUTE_MAX_N_TR {
        return false;
    }
    let (f, b, kf) = (n_folds as f64, n_perm.saturating_add(1) as f64, k as f64);
    let (t, v) = (n_tr as f64, (n - n_tr) as f64);
    let gram = one_gram_cost(n, p, false) + f * b * (kf * t * t + v * t);
    gram < f * b * (2.0 * kf * t + v) * p as f64
}

/// Should the `split_exact` refit route take the n-space route?
///
/// Every split of the `n` rows trains on `n_train` rows and tests on
/// `n_test`, `resample::split_sizes(n, k)`. `k = 1` dense never reaches the
/// refit route (the no-refit route claims it), so this starts at `k = 2`.
/// `split_sizes` already gives `n_train ≥ k + 2` when `n ≥ k + 5`;
/// `k < n_train` is checked anyway.
///
/// The route reads every split's blocks from one all-row Gram matrix, the
/// gate's when the call holds it (`gram_held`), so it is taken when, with
/// `J = n_splits` and `B' = n_perm + 1`,
///
/// ```text
/// G + J·B'·(k·n_train² + n_test·n_train)  <  J·B'·(2k·n_train + n_test)·p
/// ```
///
/// in multiply-adds, with `G = n²·p/2`, or `0` when `gram_held`; the terms
/// are counted as for [`nspace_eligible_raw_perm`], `M·A` and `X̃_te·coef`
/// scoring the test half. Also `n_train < p` and
/// `n_train ≤ DUAL_ROUTE_MAX_N_TR`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub(crate) fn nspace_eligible_split_exact(
    n: usize,
    p: usize,
    n_perm: usize,
    n_splits: usize,
    k: usize,
    dense_unweighted: bool,
    gram_held: bool,
) -> bool {
    let (n_train, n_test) = crate::resample::split_sizes(n, k);
    if !dense_unweighted
        || !(2..=K_DUAL_MAX).contains(&k)
        || k >= n_train
        || n_train >= p
        || n_train > DUAL_ROUTE_MAX_N_TR
    {
        return false;
    }
    let (j, b, kf) = (n_splits as f64, n_perm.saturating_add(1) as f64, k as f64);
    let (t, s) = (n_train as f64, n_test as f64);
    let gram = one_gram_cost(n, p, gram_held) + j * b * (kf * t * t + s * t);
    gram < j * b * (2.0 * kf * t + s) * p as f64
}

/// Should `split_nb` at `k ≥ 2` take the n-space route? Only on a call that
/// holds the gate's all-row Gram matrix (`signal_test::run_split_nb`
/// checks that): each split then costs `O(n²)` for its blocks and
/// `O(k·n_train²)` for the fit, against `O(k·n_train·p)` for a refit on
/// halves, and the gate holds `G` only on wide X. `n_train` as for
/// [`nspace_eligible_split_exact`].
pub(crate) fn nspace_eligible_split_nb(n_train: usize, k: usize, dense_unweighted: bool) -> bool {
    dense_unweighted
        && (2..=K_DUAL_MAX).contains(&k)
        && k < n_train
        && n_train <= super::DUAL_ROUTE_MAX_N_TR
}

/// `G = X̃X̃'` (`n × n`) for the block `xs` (`n × p`). Callers pass
/// `fit::par_fixed()`, not faer's `*` operator,
/// which reads the global parallelism setting;
/// `gram_products_are_thread_count_invariant_at_the_route_shapes` pins that
/// the result does not depend on the pool size.
fn gram_of(xs: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let n = xs.nrows();
    let mut g = Mat::<f64>::zeros(n, n);
    matmul(g.as_mut(), Accum::Replace, xs, xs.transpose(), 1.0, par);
    g
}

/// `M = X̃_te X̃_tr'` (`n_te × n_tr`), the map from dual coefficients on the
/// training rows to scores (or predictions) on the held-out rows. `par` as
/// for [`gram_of`]. The reference the blocks taken from the all-row Gram
/// matrix are checked against (test builds).
#[cfg(test)]
#[allow(clippy::similar_names)]
fn cross_of(xs_te: MatRef<'_, f64>, xs_tr: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(xs_te.nrows(), xs_tr.nrows());
    matmul(
        m.as_mut(),
        Accum::Replace,
        xs_te,
        xs_tr.transpose(),
        1.0,
        par,
    );
    m
}

/// `m·v`, sequential. The per-replicate products of this route.
fn seq_gemv(m: MatRef<'_, f64>, v: &Col<f64>) -> Col<f64> {
    let mut out = Col::<f64>::zeros(m.nrows());
    matmul(
        out.as_mut().as_mat_mut(),
        Accum::Replace,
        m,
        v.as_ref().as_mat(),
        1.0,
        Par::Seq,
    );
    out
}

/// What one n-space fit returns.
pub(crate) enum NspaceOutcome {
    /// The fit keeps `k_used` components, `1 ≤ k_used ≤ k` (fewer than `k`
    /// when a truncation exit fires): `coef = X̃'·alpha`.
    Resolved { alpha: Col<f64>, k_used: usize },
    /// The fit keeps no component (a constant outcome, or a first component
    /// at a truncation exit). The zero model: `coef = 0`, predictions and
    /// scores exactly zero.
    ZeroModel,
}

/// One fixed training block, shared read-only by every replicate: its Gram
/// matrix `g = X̃X̃'` (`n_tr × n_tr`), `x_fro = ‖X̃‖_F` for the relative
/// floor, and the feature count `p`.
pub(crate) struct NspaceBlock<'a> {
    g: MatRef<'a, f64>,
    x_fro: f64,
    p: usize,
}

/// The owned once-per-block products of the route for one standardized
/// block `X̃`: `G`, `‖X̃‖_F` and `p`. Built from `X̃` (`pls1_perm_null`, on
/// the thread that runs its call), from a block of the all-row Gram matrix
/// (a `raw_perm` fold or a `split_exact` or `split_nb` split, on the thread
/// that runs it), or from a caller's Gram matrix by [`NspaceGram::from_gram`]
/// (one per `pls1_perm_alpha` call, on the calling thread).
pub(crate) struct NspaceGram {
    g: Mat<f64>,
    x_fro: f64,
    p: usize,
}

impl NspaceGram {
    /// `G = xs·xs'` with `par` (the drivers pass `fit::par_fixed()`), and
    /// `x_fro = xs.norm_l2()`, the norm the primal kernel takes of the same
    /// block.
    pub(crate) fn new(xs: MatRef<'_, f64>, par: Par) -> Self {
        #[cfg(test)]
        BLOCKS_BUILT.with(|c| c.set(c.get() + 1));
        Self {
            g: gram_of(xs, par),
            x_fro: xs.norm_l2(),
            p: xs.ncols(),
        }
    }

    /// The block of a caller's Gram matrix `g = X̃X̃'` (`n × n`) of a block
    /// with `p` columns that plskit does not hold (`pls1_perm_alpha`).
    /// `g` is copied into column-major storage, so the products read the
    /// same bits whatever the caller's layout. The caller has checked that
    /// `g` is square and finite with a non-negative diagonal.
    pub(crate) fn from_gram(g: MatRef<'_, f64>, p: usize) -> Self {
        let n = g.nrows();
        Self::from_owned(Mat::<f64>::from_fn(n, n, |i, j| g[(i, j)]), p)
    }

    /// The block of a column-major `g` it takes over, with
    /// `x_fro = √trace(g)`, the diagonal summed in index order and clamped
    /// at zero, equal to `‖X̃‖_F` to rounding.
    fn from_owned(g: Mat<f64>, p: usize) -> Self {
        #[cfg(test)]
        BLOCKS_BUILT.with(|c| c.set(c.get() + 1));
        let n = g.nrows();
        let x_fro = (0..n).map(|i| g[(i, i)]).sum::<f64>().max(0.0).sqrt();
        Self { g, x_fro, p }
    }

    /// The kernel's read-only view.
    pub(crate) fn block(&self) -> NspaceBlock<'_> {
        NspaceBlock {
            g: self.g.as_ref(),
            x_fro: self.x_fro,
            p: self.p,
        }
    }
}

#[cfg(test)]
thread_local! {
    static BLOCKS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many [`NspaceGram`]s this thread has built (test builds). The
/// drivers build a block inside each fold or split unit, which runs on the
/// calling thread when the units run one at a time (`fit::with_budget(0,
/// ..)`), so a test can then assert that a public call ran the n-space
/// route; units run in a batch build their blocks on Rayon workers. With `PLSKIT_NUM_THREADS` set to a
/// cap, a public call runs on a plskit pool worker, which this thread-local
/// does not reach, so this panics when the variable sets a cap.
#[cfg(test)]
pub(crate) fn nspace_blocks_built() -> usize {
    assert!(
        std::env::var_os(crate::fit::NUM_THREADS_ENV).is_none_or(|v| v == "0"),
        "unset PLSKIT_NUM_THREADS or set it to 0: the block counter does not reach plskit's pool"
    );
    BLOCKS_BUILT.with(std::cell::Cell::get)
}

/// `v ← P_i v = v − t_i·(v_i'v)` with `v_i = t_i / (t_i't_i)`: one
/// deflation projection, dot product in ascending index order.
fn project_off_score(v: &mut Col<f64>, t_i: &Col<f64>, v_i: &Col<f64>) {
    let n = v.nrows();
    let c: f64 = (0..n).map(|l| v_i[l] * v[l]).sum();
    for l in 0..n {
        v[l] -= c * t_i[l];
    }
}

/// One PLS1 fit of `k` components on a fixed training block, for one
/// outcome `z`, computed in n-space from the block's Gram matrix
/// `G = X̃X̃'` (kernel PLS for wide data: Rännar, Lindgren, Geladi and Wold
/// 1994; Rosipal and Trejo 2001). Returns dual coefficients `alpha` with
/// `coef = X̃'alpha` and the components kept, or `ZeroModel`.
///
/// # The kernel
/// The primal PLS1 kernel's deflation `X_{a+1} = X_a − t_a p_a'`, with
/// `p_a = X_a't_a/(t_a't_a)`, is exactly `P_a X_a` for the projection
/// `P_a = I − t_a t_a'/(t_a't_a)`. So `X_a = D_a X̃` with
/// `D_a = P_{a−1} ⋯ P_1`, `X_a'y_a = X̃'D_a'y_a`, and every quantity stays in
/// the row space of `X̃`. For component `a`, from `y_1 = z`:
///
/// ```text
/// r_a     = D_a' y_a              P_{a−1} applied first, P_1 last
/// h_a     = G r_a
/// wn_a²   = r_a' h_a              = ‖X_a' y_a‖²
/// u_a     = r_a / wn_a            (w_a = X̃' u_a)
/// g_a     = h_a / wn_a            (= G u_a = X̃ w_a)
/// t_a     = D_a g_a               P_1 applied first
/// tt_a    = t_a' t_a
/// v_a     = t_a / tt_a            (p_a = X̃' v_a, since D_a' t_a = t_a)
/// q_a     = y_a' t_a / tt_a
/// y_{a+1} = y_a − q_a t_a
/// ```
///
/// `fit::pls1_coef_at_k`'s `coef = W (P'W)⁻¹ Q` then needs only n-vectors:
/// `P'W = V' [g_1 … g_k]` (`k × k`), `alpha = U (P'W)⁻¹ q` with the same
/// `PartialPivLu` solve, `coef = X̃' alpha`. At `k = 1` this is
/// `alpha = z·(z'Gz)/(z'G²z)`, the closed form of `pls1_cv_r2_columns`. The
/// projections are applied one at a time in the order the primal's
/// sequential deflations imply, not as the one-shot sum
/// `I − Σ t_i t_i'/(t_i't_i)`, and nothing is reorthogonalized (the primal
/// does not reorthogonalize either). Every product here is sequential
/// (`Par::Seq`); the drivers map replicates in parallel. Cost per call: `k`
/// products with `G` (`O(n_tr²)` each), `O(k²·n_tr)` for the projections
/// and `P'W`, and one `k × k` LU. Nothing carries `p`.
///
/// # Truncation
/// The fit stops at the first component `a` where `wn_a` falls under
/// `max(fit::NIPALS_ABS_FLOOR, fit::w_rel_floor(n_tr, p, ‖X̃‖_F, ‖z‖))` or
/// `tt_a` under `NIPALS_ABS_FLOOR`, the exits of the primal kernel
/// (`fit::pls1_kernel`), with `‖X̃‖_F` the block's `x_fro`. It keeps the
/// components before `a`; at `a = 1` it is the zero model, and so is an
/// exactly zero `z` (a constant outcome). A NaN `wn_a` or `tt_a` passes both
/// tests, as in the primal kernel. `wn_a² = r_a'h_a` is formed from a
/// rounded `G` and resolves `‖X_a'y_a‖` only to about
/// `√(p·ε)·‖X̃‖_F·‖z‖`, far above the relative floor, so near the floor
/// (an outcome orthogonal to `X̃`, or the components past the rank of
/// `X̃`) this route and the primal can keep different numbers of
/// components; [`K_DUAL_MAX`]'s sweep measures the agreement elsewhere.
///
/// # Precision follows the largest column
/// `block.g` is a re-centred block of the all-row Gram matrix, whose
/// rounding is relative to the largest entries of `X̃`, so it carries an
/// absolute error of about `ε·F²`, with `F` the training rows' all-row
/// norm, whatever the block's own size. Under `pre_standardized` with
/// columns of very different scale, or under any scaling with a row far
/// from the rest (about 1e9 of the spread or more), the large part buries
/// the structure of the rest: the route can then lose precision on
/// ordinary columns, not only at the floor, by up to O(1).
///
/// # Preconditions
/// `block.g` is `n_tr × n_tr` and `z` has `n_tr` rows (the eligibility
/// functions, or `pls1_perm_alpha`'s argument checks, guarantee both).
pub(crate) fn pls1_nspace_kernel(
    block: &NspaceBlock<'_>,
    z: ColRef<'_, f64>,
    k: usize,
) -> NspaceOutcome {
    let cols = nspace_kernel_cols(block, z.as_mat(), k);
    debug_assert_eq!(cols.len(), 1);
    cols[0].outcome_at(k)
}

/// [`pls1_nspace_kernel`] on every column of `zs` (`n_tr × m`), fitted once
/// at `ks.last()` and read at each k of `ks` (sorted ascending, non-empty),
/// with the `m` products of each component taken as one `G·R` product:
/// element `j` holds column `j`'s outcome at each k, the solve on the fit's
/// first `k.min(k_used)` components ([`KernelColumn::outcome_at`]). A
/// column's bits can differ from the single-column call's in the last
/// places (the product's summation order), within the same rounding bounds;
/// for thread-count invariance a caller fixes the column sets it passes
/// independently of the pool (see [`NSPACE_BATCH`]).
pub(crate) fn pls1_nspace_kernel_cols(
    block: &NspaceBlock<'_>,
    zs: MatRef<'_, f64>,
    ks: &[usize],
) -> Vec<Vec<NspaceOutcome>> {
    let k_max = *ks.last().expect("ks is non-empty");
    nspace_kernel_cols(block, zs, k_max)
        .iter()
        .map(|c| ks.iter().map(|&k| c.outcome_at(k)).collect())
        .collect()
}

/// Per-column state of [`nspace_kernel_cols`] between components.
struct KernelColumn {
    /// `y_a`, deflated after each component, on the scale `z / scale`.
    y: Col<f64>,
    /// The power of two `z` was divided by.
    scale: f64,
    ts: Vec<Col<f64>>,
    vs: Vec<Col<f64>>,
    u_mat: Mat<f64>,
    g_mat: Mat<f64>,
    q: Col<f64>,
    /// `max(NIPALS_ABS_FLOOR, w_rel_floor)` of the column, on the scale of
    /// `y`.
    w_floor: f64,
    /// `None` while the column is live; the components it keeps once it
    /// stops.
    k_used: Option<usize>,
}

impl KernelColumn {
    /// The outcome on the first `k.min(k_used)` components: `fit::pls1_coef_at_k`'s
    /// `W (P'W)⁻¹ Q` on them, as `alpha`. A fit's first `k` components are
    /// the fit at `k`, so this is the outcome a fit at `k` gives.
    fn outcome_at(&self, k: usize) -> NspaceOutcome {
        let k_used = self.k_used.expect("set when the kernel loop ends").min(k);
        if k_used == 0 {
            return NspaceOutcome::ZeroModel;
        }
        let n = self.y.nrows();
        // P'W = V' [g_1 … g_k], dots in ascending index order.
        let pw = Mat::<f64>::from_fn(k_used, k_used, |i, j| {
            (0..n)
                .map(|l| self.vs[i][l] * self.g_mat[(l, j)])
                .sum::<f64>()
        });
        // The same solve as `fit::pls1_coef_at_k`.
        let mut sol: Col<f64> = self.q.subrows(0, k_used).to_owned();
        crate::linalg::lu_solve_in_place(pw.as_ref(), sol.as_mut().as_mat_mut());
        let mut alpha = Col::<f64>::zeros(n);
        matmul(
            alpha.as_mut().as_mat_mut(),
            Accum::Replace,
            self.u_mat.subcols(0, k_used),
            sol.as_ref().as_mat(),
            self.scale,
            Par::Seq,
        );
        NspaceOutcome::Resolved { alpha, k_used }
    }
}

/// [`pls1_nspace_kernel`] on the `m` columns of `zs` at once: the same
/// per-column arithmetic, with the `m` products `h_a = G r_a` of one
/// component taken as a single `G·R` product (`Par::Seq`). Element `j` of
/// the result is column `j`'s fit, `k_used` set, read through
/// [`KernelColumn::outcome_at`]. The per-column steps (the
/// gates, projections, deflation, `P'W` and its solve) are the single
/// column kernel's, call for call; only the `G·R` product changes, and a
/// column of it can differ from the `m = 1` product in the last bits,
/// which the gates' rounding bounds cover (each entry is still one
/// length-`n_tr` dot product). At `m = 1` it is the single-column kernel.
/// A column that leaves early (a gate, `ZeroModel`) keeps a zero column in
/// `R`, which changes no other column. `trace` records column 0's
/// components and needs `m = 1`.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn nspace_kernel_cols(block: &NspaceBlock<'_>, zs: MatRef<'_, f64>, k: usize) -> Vec<KernelColumn> {
    let n = block.g.nrows();
    let m = zs.ncols();
    debug_assert_eq!((zs.nrows(), block.g.ncols()), (n, n));
    let mut cols: Vec<KernelColumn> = (0..m)
        .map(|j| {
            // `z` divided by a power of two near its largest entry, exact:
            // every quantity below is linear in `z` or free of its scale, the
            // absolute exit moved with it, so in range this changes no bit,
            // and out of range it keeps `wn_a²` and `‖z‖²` finite and normal.
            let z = zs.col(j);
            let (s, inv) =
                crate::linalg::pow2_scale((0..n).fold(0.0_f64, |a, i| a.max(z[i].abs())));
            let y = Col::<f64>::from_fn(n, |i| z[i] * inv);
            let z_norm = y.norm_l2();
            KernelColumn {
                y,
                scale: s,
                ts: Vec::with_capacity(k),
                vs: Vec::with_capacity(k),
                u_mat: Mat::<f64>::zeros(n, k),
                g_mat: Mat::<f64>::zeros(n, k),
                q: Col::<f64>::zeros(k),
                // The absolute exit is on the scale of `z`, so on `y` it is
                // `NIPALS_ABS_FLOOR / s`; the relative floor scales with `y`.
                w_floor: (NIPALS_ABS_FLOOR * inv).max(w_rel_floor(n, block.p, block.x_fro, z_norm)),
                k_used: if z_norm == 0.0 { Some(0) } else { None },
            }
        })
        .collect();
    let mut r_mat = Mat::<f64>::zeros(n, m);
    let mut h_mat = Mat::<f64>::zeros(n, m);

    for a in 0..k {
        if cols.iter().all(|c| c.k_used.is_some()) {
            break;
        }
        // r_a = D_a' y_a = P_1 ⋯ P_{a−1} y_a: P_{a−1} first.
        for (j, c) in cols.iter().enumerate() {
            if c.k_used.is_some() {
                r_mat.col_mut(j).fill(0.0);
                continue;
            }
            let mut r = c.y.clone();
            for (t_i, v_i) in c.ts.iter().zip(&c.vs).rev() {
                project_off_score(&mut r, t_i, v_i);
            }
            r_mat.col_mut(j).copy_from(&r);
        }
        // h_a = G r_a, every live column at once.
        matmul(
            h_mat.as_mut(),
            Accum::Replace,
            block.g,
            r_mat.as_ref(),
            1.0,
            Par::Seq,
        );
        for (j, c) in cols.iter_mut().enumerate() {
            if c.k_used.is_some() {
                continue;
            }
            let r = r_mat.col(j);
            let h = h_mat.col(j);
            let wn2: f64 = (0..n).map(|i| r[i] * h[i]).sum();
            let wn = wn2.max(0.0).sqrt();
            // `fit::pls1_kernel`'s exits, on this route's `‖X_a'y_a‖`.
            if wn < c.w_floor {
                c.k_used = Some(a);
                continue;
            }
            let inv_wn = 1.0 / wn;

            // g_a = h_a / wn_a; t_a = D_a g_a = P_{a−1} ⋯ P_1 g_a: P_1 first.
            let g = Col::<f64>::from_fn(n, |i| h[i] * inv_wn);
            let mut t = g.clone();
            for (t_i, v_i) in c.ts.iter().zip(&c.vs) {
                project_off_score(&mut t, t_i, v_i);
            }
            let tt: f64 = (0..n).map(|i| t[i] * t[i]).sum();
            if tt < NIPALS_ABS_FLOOR {
                c.k_used = Some(a);
                continue;
            }
            let inv_tt = 1.0 / tt;
            let v = Col::<f64>::from_fn(n, |i| t[i] * inv_tt);
            let qa: f64 = (0..n).map(|i| c.y[i] * t[i]).sum::<f64>() * inv_tt;
            for i in 0..n {
                c.y[i] -= qa * t[i];
            }
            for i in 0..n {
                c.u_mat[(i, a)] = r[i] * inv_wn;
                c.g_mat[(i, a)] = g[i];
            }
            c.q[a] = qa;
            c.ts.push(t);
            c.vs.push(v);
        }
    }

    for c in &mut cols {
        c.k_used.get_or_insert(k);
    }
    cols
}

/// One `pls1_perm_null` permutation row on the n-space route: the kernel on
/// the null column `z` of the standardized outcome, then `β = X̃'·alpha`
/// straight into `out` by one sequential GEMV; `ZeroModel` writes zeros.
/// Dense and unweighted only: `xs_fit` is the block `gram` was built from,
/// and `β` equals `coef` because the engine fits pre-standardized arrays.
pub(crate) fn nspace_perm_row(
    gram: &NspaceGram,
    xs_fit: MatRef<'_, f64>,
    z: ColRef<'_, f64>,
    k: usize,
    out: &mut [f64],
) {
    debug_assert!(k >= 1);
    match pls1_nspace_kernel(&gram.block(), z, k) {
        NspaceOutcome::Resolved { alpha, .. } => matmul(
            ColMut::from_slice_mut(out).as_mat_mut(),
            Accum::Replace,
            xs_fit.transpose(),
            alpha.as_ref().as_mat(),
            1.0,
            Par::Seq,
        ),
        NspaceOutcome::ZeroModel => out.fill(0.0),
    }
}

/// The n-space block of one `raw_perm` fold: [`NspaceGram`] of `X̃_tr` and
/// `M = X̃_val X̃_tr'`, both halves centred by the training means. Built
/// once per fold by the thread that runs the fold. Unweighted folds only.
pub(crate) struct NspaceFold {
    gram: NspaceGram,
    m: Mat<f64>,
}

impl NspaceFold {
    /// From the products of the prepared fold's halves, with `par`: the
    /// reference for [`NspaceFold::from_gram`] (test builds).
    #[cfg(test)]
    pub(crate) fn new(fold: &CvFold, par: Par) -> Self {
        Self {
            gram: NspaceGram::new(fold.xs_tr.as_ref(), par),
            m: cross_of(fold.xs_val.as_ref(), fold.xs_tr.as_ref(), par),
        }
    }

    /// From the blocks of the unweighted all-row Gram matrix `g`
    /// ([`HeldOutGram::part_blocks`] on `train_idx`, `val_idx`), in
    /// `O(n_tr·n)` with no product over the columns of X; `x_fro` is
    /// `√trace(K_A)` ([`NspaceGram::from_owned`]).
    pub(crate) fn from_gram(g: &HeldOutGram, train_idx: &[usize], val_idx: &[usize]) -> Self {
        let (k_a, m) = g.part_blocks(train_idx, val_idx, None);
        Self {
            gram: NspaceGram::from_owned(k_a, g.n_features()),
            m,
        }
    }
}

/// One (fold, column) unit of the n-space `raw_perm` route on the fold with
/// training rows `train_idx` and validation rows `val_idx`: this fold's
/// `(ss_res, ss_tot)` for the outcome `y_of` (`y_of(i)` is the raw outcome
/// of full-data row `i`). The outcome is standardized with the fold's own
/// training moments per column, exactly as `cv_fold_contribution` does it
/// (so `ss_tot` is the same bits on both routes), and the validation
/// predictions are `M·alpha`, zero for the zero model. Per k of `ks`
/// (sorted ascending, non-empty): the kernel runs once at `ks.last()` and
/// is read at each k ([`pls1_nspace_kernel_cols`]); element i is the pair
/// at `ks[i]`.
///
/// # Errors
/// `NonFiniteInput` when the standardized training outcome is not finite,
/// as `cv_fold_contribution`'s y check answers it.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn fold_column_nspace(
    nf: &NspaceFold,
    train_idx: &[usize],
    val_idx: &[usize],
    y_of: &dyn Fn(usize) -> f64,
    ks: &[usize],
) -> crate::error::PlsKitResult<Vec<(f64, f64)>> {
    debug_assert!(ks.first().is_some_and(|&k| k >= 1));
    let n_tr = train_idx.len();
    let n_val = val_idx.len();
    let y_tr = Col::<f64>::from_fn(n_tr, |i| y_of(train_idx[i]));
    let (z, y_mean, y_scale) = standardize1(y_tr.as_ref());
    crate::fit::check_finite_col(z.as_ref())?;
    // `cv_fold_contribution`'s arithmetic, call for call (change together).
    let ys_val = Col::<f64>::from_fn(n_val, |i| (y_of(val_idx[i]) - y_mean) / y_scale);
    let mean_val: f64 = (0..n_val).map(|i| ys_val[i]).sum::<f64>() / n_val as f64;
    let tot: f64 = (0..n_val).map(|i| (ys_val[i] - mean_val).powi(2)).sum();
    let mut outcomes = pls1_nspace_kernel_cols(&nf.gram.block(), z.as_ref().as_mat(), ks);
    debug_assert_eq!(outcomes.len(), 1);
    Ok(outcomes
        .swap_remove(0)
        .into_iter()
        .map(|out| {
            let y_pred: Col<f64> = match out {
                NspaceOutcome::Resolved { alpha, .. } => seq_gemv(nf.m.as_ref(), &alpha),
                NspaceOutcome::ZeroModel => Col::<f64>::zeros(n_val),
            };
            let res: f64 = (0..n_val).map(|i| (y_pred[i] - ys_val[i]).powi(2)).sum();
            (res, tot)
        })
        .collect())
}

/// The n-space block of one `split_exact` or `split_nb` split:
/// [`NspaceGram`] of `X̃_tr` and `M = X̃_te X̃_tr'`, both halves centred by
/// the training means. Built once per split by the thread that runs the
/// split. Unweighted splits only.
pub(crate) struct NspaceSplit {
    gram: NspaceGram,
    m: Mat<f64>,
}

impl NspaceSplit {
    /// From the products of the prepared halves, with `par`: the reference
    /// for [`NspaceSplit::from_gram`] (test builds).
    #[cfg(test)]
    pub(crate) fn new(prep: &PreparedSplit, par: Par) -> Self {
        Self {
            gram: NspaceGram::new(prep.xs_tr.as_ref(), par),
            m: cross_of(prep.xs_te.as_ref(), prep.xs_tr.as_ref(), par),
        }
    }

    /// From the blocks of the unweighted all-row Gram matrix `g`
    /// ([`HeldOutGram::part_blocks`] on `sp`), in `O(n_tr·n)` with no product
    /// over the columns of X; `x_fro` is `√trace(K_A)`
    /// ([`NspaceGram::from_owned`]).
    pub(crate) fn from_gram(g: &HeldOutGram, sp: &SplitIdx) -> Self {
        let (k_a, m) = g.part_blocks(&sp.tr, &sp.te, None);
        Self {
            gram: NspaceGram::from_owned(k_a, g.n_features()),
            m,
        }
    }

    /// The training-half fit of `k` components on the standardized
    /// training outcome `z`, by [`pls1_nspace_kernel`] on this split's
    /// block, as its predictions on both halves:
    /// `(X̃_tr·coef, X̃_te·coef) = (G·alpha, M·alpha)`, both zero for the
    /// zero model. `signal_test`'s `e` on a caller's Gram matrix reads them.
    pub(crate) fn predictions(&self, z: ColRef<'_, f64>, k: usize) -> (Col<f64>, Col<f64>) {
        match pls1_nspace_kernel(&self.gram.block(), z, k) {
            NspaceOutcome::Resolved { alpha, .. } => (
                seq_gemv(self.gram.g.as_ref(), &alpha),
                seq_gemv(self.m.as_ref(), &alpha),
            ),
            NspaceOutcome::ZeroModel => (
                Col::<f64>::zeros(self.m.ncols()),
                Col::<f64>::zeros(self.m.nrows()),
            ),
        }
    }
}

/// Columns per batched call of the n-space `split_exact` refit route
/// ([`split_columns_r_nspace`]): the driver cuts the `B + 1` outcome
/// columns into consecutive runs of this many, starting at column 0, and
/// hands each run to one call, so which columns share a product never
/// depends on the thread count. Measured on an Apple M4 at `n_tr = 500`:
/// one `G·R` product of 8 to 256 columns runs at the same rate as one of
/// 1001 columns, about 3.3x the rate of 1001 matrix-vector products, so a
/// small run keeps that rate and leaves the runs enough to map in parallel.
pub(crate) const NSPACE_BATCH: usize = 16;

/// A run of `m` (split, column) units of the n-space `split_exact` refit
/// route: element `j` holds, for each k of `ks` (sorted ascending,
/// non-empty), the split-half Pearson `r` for the outcome `y_of(j, ·)`
/// (`y_of(j, i)` is column `j`'s raw outcome at full-data row `i`; `r` is
/// taken before the driver's clamp and Fisher z). Each training outcome is
/// standardized as `split_column_r` does it; the kernel runs once at
/// `ks.last()` on all `m` columns at once ([`pls1_nspace_kernel_cols`]); per
/// k the test-half scores are `S = M·A` for the columns' dual coefficients
/// `A` at that k, one product; each correlation is `guarded_pearson`'s. A training outcome
/// whose standardization is not finite, a constant one and a fit that keeps
/// no component (`ZeroModel`) give `r = 0`, as on the primal route.
#[allow(clippy::similar_names)]
pub(crate) fn split_columns_r_nspace(
    sp: &SplitIdx,
    ns: &NspaceSplit,
    m: usize,
    y_of: &dyn Fn(usize, usize) -> f64,
    ks: &[usize],
) -> Vec<Vec<f64>> {
    debug_assert!(ks.first().is_some_and(|&k| k >= 1));
    let n_tr = sp.tr.len();
    let n_te = sp.te.len();
    let mut zs = Mat::<f64>::zeros(n_tr, m);
    for j in 0..m {
        let y_tr = Col::<f64>::from_fn(n_tr, |i| y_of(j, sp.tr[i]));
        let (z, _, _) = standardize1(y_tr.as_ref());
        // A non-finite target fails the primal fit's y check (`r = 0`); a
        // zero column is the kernel's zero model.
        if crate::fit::check_finite_col(z.as_ref()).is_ok() {
            zs.col_mut(j).copy_from(&z);
        }
    }
    let outcomes = pls1_nspace_kernel_cols(&ns.gram.block(), zs.as_ref(), ks);
    let mut rs = vec![Vec::with_capacity(ks.len()); m];
    for ki in 0..ks.len() {
        let mut alphas = Mat::<f64>::zeros(n_tr, m);
        for (j, out) in outcomes.iter().enumerate() {
            if let NspaceOutcome::Resolved { alpha, .. } = &out[ki] {
                alphas.col_mut(j).copy_from(alpha);
            }
        }
        let mut scores = Mat::<f64>::zeros(n_te, m);
        matmul(
            scores.as_mut(),
            Accum::Replace,
            ns.m.as_ref(),
            alphas.as_ref(),
            1.0,
            Par::Seq,
        );
        for (j, out) in outcomes.iter().enumerate() {
            rs[j].push(match out[ki] {
                NspaceOutcome::ZeroModel => 0.0,
                NspaceOutcome::Resolved { .. } => {
                    let s = scores.col(j);
                    guarded_pearson(n_te, |i| s[i], |i| y_of(j, sp.te[i]))
                }
            });
        }
    }
    rs
}

#[cfg(test)]
mod sweep;

#[cfg(test)]
mod site_tests;

#[cfg(test)]
mod tests;
