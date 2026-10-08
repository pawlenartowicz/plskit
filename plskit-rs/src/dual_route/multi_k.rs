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
//! the constructors here, and run the unit bodies here per replicate; a
//! unit the kernel cannot certify runs the primal unit body instead.
//!
//! The parent module's `pls1_cv_r2_columns` (the `K = 1` closed form at
//! `raw_perm`) and `signal_test::split_perm_nr_zbars` (the `K = 1` no-refit
//! route at `split_exact`) keep their inputs: this route serves `k ≥ 2` at
//! those two sites and every `k` up to [`K_DUAL_MAX`] at `pls1_perm_null`,
//! which has no other Gram route. Dense, unweighted fits only: hard `keep`
//! selection acts on the full p-length weight vector, and weighted fits
//! stay primal.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColMut, ColRef, Mat, MatRef, Par};

use super::{use_dual_route, ABS_BAND, RESOLVE_BAND, SCORE_BAND};
use crate::error::PlsKitResult;
use crate::fit::{w_rel_floor, NIPALS_ABS_FLOOR};
use crate::linalg::{scaled_moments, standardize1};
use crate::signal_test::{cv_fold_contribution, pearson_scaled, CvFold, PreparedSplit, SplitIdx};

/// Largest component count the n-space route serves. Chosen with
/// `scripts/gate_feasibility.py`: the largest `k ≤ 10`
/// whose fallback rate on ordinary data stays at or under 1% at every
/// block shape it runs (40 × 2000 to 1000 × 4000 and the wide corpus
/// fixtures). Past it the first-order history term of
/// [`pls1_nspace_kernel`] grows faster than the Gram gate can absorb and
/// nearly every replicate falls back, paying twice. The validation sweep
/// (`sweep.rs`) may only lower it.
pub(crate) const K_DUAL_MAX: usize = 2;

/// Iteration cap of [`gram_norm_bound`]'s power iteration.
const POWER_MAX_ITERS: usize = 50;

/// [`gram_norm_bound`] stops once its Rayleigh quotient moves by at most
/// this fraction of itself.
const POWER_REL_TOL: f64 = 1e-3;

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
/// row: the largest training fold (`n − ⌊n/n_folds⌋`) feeds the flop and
/// memory rule, so one route serves the whole statistic, and the smallest
/// (`n − ⌈n/n_folds⌉`) must hold `k + 1` rows. Below that `rank(X̃_tr) < k`,
/// every replicate exhausts, falls back, and the route pays twice.
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
    let n_tr_max = n - n / n_folds;
    let n_tr_min = n - n.div_ceil(n_folds);
    k < n_tr_min && use_dual_route(n_tr_max, p, n_perm.saturating_add(1), k)
}

/// Should the `split_exact` refit route take the n-space route?
///
/// `n_train` is `resample::split_sizes(n, k).0`, the training-half size
/// every split shares. `k = 1` dense never reaches the refit route (the
/// no-refit route claims it), so this starts at `k = 2`. `split_sizes`
/// already gives `n_train ≥ k + 2` when `n ≥ k + 5`; `k < n_train` is
/// checked anyway.
pub(crate) fn nspace_eligible_split_exact(
    n_train: usize,
    p: usize,
    n_perm: usize,
    k: usize,
    dense_unweighted: bool,
) -> bool {
    dense_unweighted
        && (2..=K_DUAL_MAX).contains(&k)
        && k < n_train
        && use_dual_route(n_train, p, n_perm.saturating_add(1), k)
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
/// for [`gram_of`].
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

/// Upper estimate of `‖G‖₂` for the kernel's history term, computed once
/// per block: `min(2·λ̂, ‖X̃‖_F²)` with `x_fro = ‖X̃‖_F`.
///
/// `λ̂` is the Rayleigh quotient of a power iteration on `g`, sequential
/// (so the same bits on every thread count), from the fixed ramp start
/// `v ∝ (1, 2, …, n)`, stopped after [`POWER_MAX_ITERS`] products or once
/// `λ̂` moves by at most [`POWER_REL_TOL`] of itself. A Rayleigh quotient
/// never exceeds `‖G‖₂` and a slowly converging iteration can sit below it,
/// hence the safety factor 2; `‖G‖₂ ≤ ‖X̃‖_F²` always holds and caps the
/// estimate. A non-positive or NaN estimate returns `‖X̃‖_F²`. The start is
/// a ramp, not the all-ones vector: `X̃`'s columns are centered, so
/// `G·1 = 0` and the ones vector lies in `G`'s null space. First-order:
/// the factor 2 is checked by the tests and the validation sweep, not
/// proved. `scripts/gate_feasibility.py` transcribes this function (change
/// together).
#[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
fn gram_norm_bound(g: MatRef<'_, f64>, x_fro: f64) -> f64 {
    let n = g.nrows();
    let fro2 = x_fro * x_fro;
    if n == 0 {
        return fro2;
    }
    let ramp_norm = (1..=n).map(|i| (i as f64) * (i as f64)).sum::<f64>().sqrt();
    let mut v = Col::<f64>::from_fn(n, |i| (i + 1) as f64 / ramp_norm);
    let mut lam = 0.0_f64;
    for _ in 0..POWER_MAX_ITERS {
        let w = seq_gemv(g, &v);
        let lam_new: f64 = (0..n).map(|i| v[i] * w[i]).sum();
        let norm_w = w.norm_l2();
        if norm_w.is_nan() || norm_w <= 0.0 {
            lam = lam_new;
            break;
        }
        let done = (lam_new - lam).abs() <= POWER_REL_TOL * lam_new;
        lam = lam_new;
        v = Col::<f64>::from_fn(n, |i| w[i] / norm_w);
        if done {
            break;
        }
    }
    let est = 2.0 * lam;
    if est > 0.0 && est < fro2 {
        est
    } else {
        fro2
    }
}

/// What one n-space fit returns.
pub(crate) enum NspaceOutcome {
    /// Every component `1..=k` cleared every gate, so the primal kernel
    /// keeps all `k` too (`k_used == k` always): `coef = X̃'·alpha`.
    Resolved { alpha: Col<f64>, k_used: usize },
    /// `z` exactly zero (a constant outcome): `X̃'z` is an exact zero on the
    /// primal route as well, which keeps no component. The zero model:
    /// `coef = 0`, predictions and scores exactly zero.
    ZeroModel,
    /// Some component sat inside a gate band, below it, or was NaN: the
    /// caller recomputes this unit with the primal kernel.
    Unresolved,
}

/// One fixed training block, shared read-only by every replicate: its Gram
/// matrix `g = X̃X̃'` (`n_tr × n_tr`), `x_fro = ‖X̃‖_F` computed by
/// `norm_l2` on the same view the primal kernel is handed (so the relative
/// floor is evaluated on bit-identical inputs), `g2`, the block's
/// [`gram_norm_bound`], and the feature count `p`.
pub(crate) struct NspaceBlock<'a> {
    g: MatRef<'a, f64>,
    x_fro: f64,
    g2: f64,
    p: usize,
}

/// The owned once-per-block products of the route for one standardized
/// block `X̃`: `G`, `‖X̃‖_F`, the `‖G‖₂` bound and `p`. Built by the drivers'
/// block constructors (one per `perm_null` call, `raw_perm` fold or
/// `split_exact` split) on the calling thread.
pub(crate) struct NspaceGram {
    g: Mat<f64>,
    x_fro: f64,
    g2: f64,
    p: usize,
}

impl NspaceGram {
    /// `G = xs·xs'` with `par` (the drivers pass
    /// `fit::par_fixed()`), then `‖xs‖_F` and the
    /// sequential `‖G‖₂` bound.
    pub(crate) fn new(xs: MatRef<'_, f64>, par: Par) -> Self {
        #[cfg(test)]
        BLOCKS_BUILT.with(|c| c.set(c.get() + 1));
        let g = gram_of(xs, par);
        let x_fro = xs.norm_l2();
        let g2 = gram_norm_bound(g.as_ref(), x_fro);
        Self {
            g,
            x_fro,
            g2,
            p: xs.ncols(),
        }
    }

    /// `‖xs‖_F`: `xs.norm_l2()`, the norm the primal kernel takes of the
    /// same block, so a fallback to it reads the same bits.
    pub(crate) fn x_fro(&self) -> f64 {
        self.x_fro
    }

    /// The kernel's read-only view.
    pub(crate) fn block(&self) -> NspaceBlock<'_> {
        NspaceBlock {
            g: self.g.as_ref(),
            x_fro: self.x_fro,
            g2: self.g2,
            p: self.p,
        }
    }
}

#[cfg(test)]
thread_local! {
    static BLOCKS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many [`NspaceGram`]s this thread has built (test builds). The
/// drivers build blocks on the calling thread, so a test can assert that a
/// public call ran the n-space route. With `PLSKIT_NUM_THREADS` set to a
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

/// Per-earlier-component multiples of `(n_tr + 2)·ε·‖z‖²·‖X̃‖_F²` in the
/// base bound on `wn_a²` and of `(n_tr + 2)·ε·‖z‖²·‖X̃‖_F⁴` in the base
/// bound on `tt_a·wn_a²`. Mirrored in `scripts/gate_feasibility.py`.
const STEP_WN: usize = 6;
const STEP_TT: usize = 10;

/// Multiple of `Σ_{i<a} ρ_i·‖G‖₂` (and `Σ ρ_i·‖G‖₂²`) that the history term
/// adds to component `a`'s bounds: each earlier direction's error enters
/// both the deflation and the projection, each up to `2·ρ_i·‖z‖`. See
/// "History term" on [`pls1_nspace_kernel`]. Mirrored in
/// `scripts/gate_feasibility.py` (change together).
const HISTORY_COEF: f64 = 8.0;

/// Multiple of the rounding bounds that gates 1 and 2 of
/// [`pls1_nspace_kernel`] require. `RESOLVE_BAND` protects the truncation
/// decision.
const BOUND_BAND: f64 = RESOLVE_BAND;

/// `(E_w^base(a), E_t^base(a))`: the rounding this route commits on
/// component `a`'s own arithmetic (1-based), per unit `‖z‖²`. At `a = 1`
/// they are, bit for bit, `pls1_cv_r2_columns`' bounds on `z'Gz` and
/// `z'G²z`. Derivation on [`pls1_nspace_kernel`].
#[allow(clippy::cast_precision_loss)]
fn nspace_base_bounds(a: usize, n_tr: usize, p: usize, x_fro: f64) -> (f64, f64) {
    let fro2 = x_fro * x_fro;
    let hist = (a - 1) * (n_tr + 2);
    let cw = p + 2 * n_tr + STEP_WN * hist;
    let ct = 2 * p + 3 * n_tr + STEP_TT * hist;
    (
        (cw as f64) * f64::EPSILON * fro2,
        (ct as f64) * f64::EPSILON * fro2 * fro2,
    )
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

/// Per-component record of one kernel call: the computed `wn_a²`, the
/// bounds `E_w(a)` and `E_t(a)` (history included) and
/// `ρ_a = E_w(a)·‖z‖² / wn_a²`. Read by the tests and the sweep only.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ComponentTrace {
    pub(crate) wn2: f64,
    pub(crate) e_w: f64,
    pub(crate) e_t: f64,
    pub(crate) rho: f64,
}

/// One PLS1 fit of `k` components on a fixed training block, for one
/// outcome `z`, computed in n-space from the block's Gram matrix
/// `G = X̃X̃'` (kernel PLS for wide data: Rännar, Lindgren, Geladi and Wold
/// 1994; Rosipal and Trejo 2001). Returns dual coefficients `alpha` with
/// `coef = X̃'alpha`, or `Unresolved` when rounding could put the primal
/// kernel on the other side of a truncation exit.
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
/// # Truncation gates
/// The primal kernel (`fit::pls1_kernel`) stops at the first component
/// where `‖X_a'y_a‖` falls under `fit::NIPALS_ABS_FLOOR` or under
/// `fit::w_rel_floor(n_tr, p, ‖X̃‖_F, ‖z‖)`, or `t_a't_a` under
/// `NIPALS_ABS_FLOOR`. Here `wn_a²` and `tt_a` come from a rounded `G` and
/// from accumulated projections, so near those exits this route cannot
/// always tell which side the primal falls on. As in `pls1_cv_r2_columns`,
/// it only ever proves "keep": it resolves `‖X_a'y_a‖` to about
/// `√(E_w(a))·‖z‖`, far above the relative floor, so it can never prove a
/// stop. A call is `Resolved` only when every component `1..=k` clears all
/// five gates, each an `a >= b` comparison so a NaN fails it:
///
/// 1. `wn_a² ≥ BOUND_BAND · E_w(a) · ‖z‖²`
/// 2. `tt_a · wn_a² ≥ BOUND_BAND · E_t(a) · ‖z‖²`
/// 3. `wn_a ≥ RESOLVE_BAND · w_rel_floor`
/// 4. `wn_a ≥ ABS_BAND · NIPALS_ABS_FLOOR`
/// 5. `tt_a ≥ ABS_BAND · NIPALS_ABS_FLOOR`
///
/// Everything else is `Unresolved`, and the caller recomputes that unit
/// with the primal kernel, whose answer (truncation included) is then the
/// primal route's to the bit. The one truncation decided here is `z`
/// exactly zero (`ZeroModel`). The gates are first-order bounds, not worst
/// cases: they protect the decisions, and the validation sweep checks that
/// no `Resolved` call ever keeps a different number of components than the
/// primal.
///
/// # Base rounding bounds (`nspace_base_bounds`)
/// ```text
/// E_w^base(a) = (p + 2·n_tr + 6·(a − 1)·(n_tr + 2)) · ε · ‖X̃‖_F²
/// E_t^base(a) = (2p + 3·n_tr + 10·(a − 1)·(n_tr + 2)) · ε · ‖X̃‖_F⁴
/// ```
/// At `a = 1` these are `pls1_cv_r2_columns`' bounds on `z'Gz` and `z'G²z`
/// (`tt_1·wn_1² = ‖Gz‖² = z'G²z`): each entry of `G` carries a
/// length-`p` dot product's rounding, `p·ε·|x̃_i|·|x̃_j|`, the product `G r`
/// and the dot `r'h` add `2·n_tr·ε`, and Cauchy-Schwarz turns the entrywise
/// sums into `‖z‖²·‖X̃‖_F²` (`‖X̃‖_F⁴` for the squared product).
///
/// Each earlier component `i < a` adds one projection `P_i` to `r_a` (a dot
/// product and an axpy; `‖t_i‖·‖v_i‖ = 1`, so at most `2·(n_tr + 2)·ε·‖r‖`)
/// and one deflation `y ← y − q_i t_i` (at most `(n_tr + 2)·ε·‖z‖`, since
/// `|q_i|·‖t_i‖ ≤ ‖y_i‖ ≤ ‖z‖`). Projections and deflations do not grow a
/// vector in exact arithmetic, so `‖r_a‖ ≤ ‖z‖` and the perturbation `δr`
/// they leave is at most `3·(n_tr + 2)·ε·‖z‖`. It moves `wn_a² = r'Gr` by
/// at most `2·‖δr‖·‖Gr‖ ≤ 6·(n_tr + 2)·ε·‖z‖²·‖X̃‖_F²` (using
/// `‖G‖₂ ≤ ‖X̃‖_F²`): the per-component term of `E_w^base`. For
/// `tt_a·wn_a² = ‖D_a h_a‖²` the same `δr` contributes `6·(n_tr + 2)`, and
/// the `a − 1` projections applied to `g_a` add `2·(n_tr + 2)·ε·‖h_a‖` each
/// to `D_a h_a`, another `4·(n_tr + 2)`: `10·(n_tr + 2)` per component in
/// `E_t^base`.
///
/// # History term
/// The base bounds cover this route's rounding on component `a`'s own
/// arithmetic. Each earlier component `i < a` also hands on its own error:
/// its `wn_i²` is known only to within `E_w(i)·‖z‖²`, so its direction
/// (`u_i`, hence `t_i` and the projection `P_i`) carries a relative error
/// of about `ρ_i = E_w(i)·‖z‖² / wn_i²`, which gate 1 holds under
/// `1/BOUND_BAND`. That error enters twice, in the deflation
/// `y ← y − q_i t_i` and in the projection of `r`, each moving `r_a` by up
/// to `2·ρ_i·‖z‖`. So `wn_a² = r_a'G r_a` moves by up to
/// `2·(4·ρ_i·‖z‖)·‖G‖₂·‖z‖ = 8·ρ_i·‖G‖₂·‖z‖²`, and `tt_a·wn_a² = ‖D_a G r_a‖²`
/// by up to `8·ρ_i·‖G‖₂²·‖z‖²`:
///
/// ```text
/// E_w(a) = E_w^base(a) + HISTORY_COEF · Σ_{i<a} ρ_i · ‖G‖₂
/// E_t(a) = E_t^base(a) + HISTORY_COEF · Σ_{i<a} ρ_i · ‖G‖₂²
/// ```
///
/// with `HISTORY_COEF = 8` and `‖G‖₂` replaced by the block's `g2`
/// ([`gram_norm_bound`]). This is a first-order bound: products of two
/// small errors are dropped and the `‖G‖₂` estimate carries a safety factor
/// rather than a proof. The history term above counts the error carried in
/// `r`; perturbed projections applied to `g_a` add roughly another
/// `4·ρ_i·‖G‖₂²` to the `tt` bound (a coefficient near 12 rather than 8 by
/// this accounting), which the `BOUND_BAND` margin absorbs: the `tt`
/// relative error stays below `0.375`, so every keep/stop decision is
/// still protected. It compounds: `ρ_a` grows by several orders of
/// magnitude per component on ordinary data, which is why [`K_DUAL_MAX`] is
/// small. `scripts/gate_feasibility.py` transcribes this recursion and set
/// `K_DUAL_MAX`; `history_bounds_match_gate_feasibility_script` pins the
/// two to each other, and the validation sweep checks soundness, including
/// an earlier component placed just past gate 1 (family `gate_edge`).
///
/// # The primal's own error
/// "Clears" has to hold for the primal's computed values, not only the
/// exact ones. The primal's error on `‖X_a'y_a‖` is adopted from
/// `w_rel_floor`'s measured margin: `M = 0.33` is the larger of the
/// floor's two measured margins in `fit::w_rel_floor`'s doc, `0.020×` for
/// the first noise component over all components and `0.33×` for an
/// outcome orthogonal to `X` at the first component, so that error is
/// taken as `0.33 · w_rel_floor`. Past gate 1 the exact `wn_a` is at least
/// `√(1 − 1/BOUND_BAND)` of the computed one, so past gate 3 it is at least
/// `√0.75 · 4 = 3.46×` the floor and the primal's computed value at least
/// `3.13×`: the primal keeps the component. The primal's `tt_a` carries a
/// relative error of the order of `E_t(a) / (tt_a·wn_a²)`, which gate 2
/// holds under `1/BOUND_BAND`, so it stays within a factor 2 of this
/// route's and gate 5 keeps it 50× clear of its exit.
///
/// Measured on the validation sweep (421 calls, k = 1..=10, `K_DUAL_MAX` =
/// 2): 50 resolved, 0 zero-model, 371 sent to the primal kernel; 0
/// fallbacks on ordinary data at k ≤ `K_DUAL_MAX`; worst |Δcoef| / max(1,
/// ‖coef‖_∞) = 2.9e-15. Test-half score discrepancy at `k = 2..=K_DUAL_MAX`
/// (split halves of the same designs): `η_max = 5.38e-15`, which sets
/// `dual_route::SCORE_BAND`.
///
/// # Accuracy
/// The gates protect decisions, not the `1e-10` agreement of a `Resolved`
/// call's coefficients. There is no per-replicate accuracy gate: a
/// worst-case one compounds geometrically with `k` and would reject every
/// replicate. Agreement is validated by the sweep
/// (`|Δcoef| ≤ 1e-10 · max(1, ‖coef‖_∞)`) across collinear, rank-deficient,
/// duplicated-row and near-gate designs; `G` squares the conditioning of
/// `X̃`, which is why those families are in it. At `pls1_perm_null` a
/// coefficient error is amplified into `beta_perm_z` by about `z / sd`,
/// which the corpus tolerance on `beta_perm_z` checks.
///
/// # Preconditions
/// `block.g` is `n_tr × n_tr`, `z` has `n_tr` rows, and `1 ≤ k ≤ n_tr − 1`
/// (the eligibility functions guarantee all three). A shape or `k`
/// violation returns `Unresolved` (an exactly zero `z` returns
/// `ZeroModel` at any `k`), never a panic.
pub(crate) fn pls1_nspace_kernel(
    block: &NspaceBlock<'_>,
    z: ColRef<'_, f64>,
    k: usize,
) -> NspaceOutcome {
    one_outcome(nspace_kernel_cols(block, z.as_mat(), k, None))
}

/// [`pls1_nspace_kernel`] on every column of `zs` (`n_tr × m`), with the
/// `m` products of each component taken as one `G·R` product: element `j`
/// is column `j`'s outcome. A column's bits can differ from the
/// single-column call's in the last places (the product's summation order),
/// within the same rounding bounds; for thread-count invariance a caller
/// fixes the column sets it passes independently of the pool (see
/// [`NSPACE_BATCH`]).
pub(crate) fn pls1_nspace_kernel_cols(
    block: &NspaceBlock<'_>,
    zs: MatRef<'_, f64>,
    k: usize,
) -> Vec<NspaceOutcome> {
    nspace_kernel_cols(block, zs, k, None)
}

fn one_outcome(mut v: Vec<NspaceOutcome>) -> NspaceOutcome {
    debug_assert_eq!(v.len(), 1);
    v.pop().unwrap_or(NspaceOutcome::Unresolved)
}

/// [`pls1_nspace_kernel`] with its per-component bounds recorded.
#[cfg(test)]
pub(crate) fn pls1_nspace_kernel_traced(
    block: &NspaceBlock<'_>,
    z: ColRef<'_, f64>,
    k: usize,
) -> (NspaceOutcome, Vec<ComponentTrace>) {
    let mut trace = Vec::with_capacity(k);
    let out = one_outcome(nspace_kernel_cols(block, z.as_mat(), k, Some(&mut trace)));
    (out, trace)
}

/// Per-column state of [`nspace_kernel_cols`] between components.
struct KernelColumn {
    /// `y_a`, deflated after each component.
    y: Col<f64>,
    ts: Vec<Col<f64>>,
    vs: Vec<Col<f64>>,
    u_mat: Mat<f64>,
    g_mat: Mat<f64>,
    q: Col<f64>,
    zz: f64,
    w_floor: f64,
    /// `Σ_{i<a} ρ_i`, the history term's weight.
    rho_sum: f64,
    /// `None` while the column is live; its outcome once decided.
    done: Option<NspaceOutcome>,
}

/// [`pls1_nspace_kernel`] on the `m` columns of `zs` at once: the same
/// per-column arithmetic, with the `m` products `h_a = G r_a` of one
/// component taken as a single `G·R` product (`Par::Seq`). Element `j` of
/// the result is the outcome for column `j`. The per-column steps (the
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
fn nspace_kernel_cols(
    block: &NspaceBlock<'_>,
    zs: MatRef<'_, f64>,
    k: usize,
    mut trace: Option<&mut Vec<ComponentTrace>>,
) -> Vec<NspaceOutcome> {
    debug_assert!(trace.is_none() || zs.ncols() == 1);
    let n = block.g.nrows();
    let m = zs.ncols();
    if zs.nrows() != n || block.g.ncols() != n {
        return (0..m).map(|_| NspaceOutcome::Unresolved).collect();
    }
    let mut cols: Vec<KernelColumn> = (0..m)
        .map(|j| {
            let z = zs.col(j);
            // `z` exactly zero: `X̃'z` is an exact zero on the primal route too.
            let done = if (0..n).all(|i| z[i] == 0.0) {
                Some(NspaceOutcome::ZeroModel)
            } else if (1..n).contains(&k) {
                None
            } else {
                Some(NspaceOutcome::Unresolved)
            };
            let (zz, w_floor) = if done.is_none() {
                let z_norm = z.norm_l2();
                (
                    z_norm * z_norm,
                    w_rel_floor(n, block.p, block.x_fro, z_norm),
                )
            } else {
                (0.0, 0.0)
            };
            KernelColumn {
                y: z.to_owned(),
                ts: Vec::with_capacity(k),
                vs: Vec::with_capacity(k),
                u_mat: Mat::<f64>::zeros(n, if done.is_none() { k } else { 0 }),
                g_mat: Mat::<f64>::zeros(n, if done.is_none() { k } else { 0 }),
                q: Col::<f64>::zeros(k),
                zz,
                w_floor,
                rho_sum: 0.0,
                done,
            }
        })
        .collect();
    let mut r_mat = Mat::<f64>::zeros(n, m);
    let mut h_mat = Mat::<f64>::zeros(n, m);

    for a in 0..k {
        if cols.iter().all(|c| c.done.is_some()) {
            break;
        }
        let (base_w, base_t) = nspace_base_bounds(a + 1, n, block.p, block.x_fro);
        // r_a = D_a' y_a = P_1 ⋯ P_{a−1} y_a: P_{a−1} first.
        for (j, c) in cols.iter().enumerate() {
            if c.done.is_some() {
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
            if c.done.is_some() {
                continue;
            }
            let r = r_mat.col(j);
            let h = h_mat.col(j);
            let wn_err = base_w + HISTORY_COEF * c.rho_sum * block.g2;
            let tt_err = base_t + HISTORY_COEF * c.rho_sum * block.g2 * block.g2;
            let wn2: f64 = (0..n).map(|i| r[i] * h[i]).sum();
            let wn = wn2.max(0.0).sqrt();
            let rho = wn_err * c.zz / wn2;
            if let Some(tr) = trace.as_deref_mut() {
                tr.push(ComponentTrace {
                    wn2,
                    e_w: wn_err,
                    e_t: tt_err,
                    rho,
                });
            }
            // Gates 1, 3, 4 (see the doc comment). `a >= b`, so NaN fails.
            let keep_w = wn2 >= BOUND_BAND * wn_err * c.zz
                && wn >= RESOLVE_BAND * c.w_floor
                && wn >= ABS_BAND * NIPALS_ABS_FLOOR;
            if !keep_w {
                c.done = Some(NspaceOutcome::Unresolved);
                continue;
            }
            c.rho_sum += rho;
            let inv_wn = 1.0 / wn;

            // g_a = h_a / wn_a; t_a = D_a g_a = P_{a−1} ⋯ P_1 g_a: P_1 first.
            let g = Col::<f64>::from_fn(n, |i| h[i] * inv_wn);
            let mut t = g.clone();
            for (t_i, v_i) in c.ts.iter().zip(&c.vs) {
                project_off_score(&mut t, t_i, v_i);
            }
            let tt: f64 = (0..n).map(|i| t[i] * t[i]).sum();
            // Gates 2 and 5.
            let keep_t =
                tt * wn2 >= BOUND_BAND * tt_err * c.zz && tt >= ABS_BAND * NIPALS_ABS_FLOOR;
            if !keep_t {
                c.done = Some(NspaceOutcome::Unresolved);
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

    cols.into_iter()
        .map(|c| {
            if let Some(done) = c.done {
                return done;
            }
            // P'W = V' [g_1 … g_k], dots in ascending index order.
            let pw = Mat::<f64>::from_fn(k, k, |i, j| {
                (0..n).map(|l| c.vs[i][l] * c.g_mat[(l, j)]).sum::<f64>()
            });
            // The same solve as `fit::pls1_coef_at_k`.
            let mut sol: Col<f64> = c.q.clone();
            crate::linalg::lu_solve_in_place(pw.as_ref(), sol.as_mut().as_mat_mut());
            let mut alpha = Col::<f64>::zeros(n);
            matmul(
                alpha.as_mut().as_mat_mut(),
                Accum::Replace,
                c.u_mat.as_ref(),
                sol.as_ref().as_mat(),
                1.0,
                Par::Seq,
            );
            if (0..n).all(|i| alpha[i].is_finite()) {
                NspaceOutcome::Resolved { alpha, k_used: k }
            } else {
                NspaceOutcome::Unresolved
            }
        })
        .collect()
}

/// One `pls1_perm_null` permutation row on the n-space route: the kernel on
/// the permuted outcome `z = ys_std[perm]`, then `β = X̃'·alpha` straight
/// into `out` by one sequential GEMV. Returns `true` when the row is
/// written and `false` when the kernel is `Unresolved`; the caller then
/// runs the primal row body (`perm_null::perm_row`'s Primal arm), so that
/// row is the primal route's to the bit, failure included. `ZeroModel`
/// writes zeros, the primal's `β` for an exactly zero outcome. Dense and
/// unweighted only: `xs_fit` is the block `gram` was built from, and `β`
/// equals `coef` because the engine fits pre-standardized arrays.
pub(crate) fn nspace_perm_row(
    gram: &NspaceGram,
    xs_fit: MatRef<'_, f64>,
    ys_std: ColRef<'_, f64>,
    perm: &[usize],
    k: usize,
    out: &mut [f64],
) -> bool {
    debug_assert!(k >= 1);
    let n = xs_fit.nrows();
    let z = Col::<f64>::from_fn(n, |i| ys_std[perm[i]]);
    match pls1_nspace_kernel(&gram.block(), z.as_ref(), k) {
        NspaceOutcome::Resolved { alpha, k_used } => {
            debug_assert_eq!(k_used, k);
            matmul(
                ColMut::from_slice_mut(out).as_mat_mut(),
                Accum::Replace,
                xs_fit.transpose(),
                alpha.as_ref().as_mat(),
                1.0,
                Par::Seq,
            );
            true
        }
        NspaceOutcome::ZeroModel => {
            out.fill(0.0);
            true
        }
        NspaceOutcome::Unresolved => false,
    }
}

/// The n-space block of one prepared `raw_perm` fold: [`NspaceGram`] of
/// `X̃_tr` and `M = X̃_val X̃_tr'`, built once per fold on the calling thread
/// with `par` (`fit::par_fixed()`). Unweighted folds only, so `xs_tr` is
/// the matrix the primal fold fit is handed.
pub(crate) struct NspaceFold {
    gram: NspaceGram,
    m: Mat<f64>,
}

impl NspaceFold {
    pub(crate) fn new(fold: &CvFold, par: Par) -> Self {
        Self {
            gram: NspaceGram::new(fold.xs_tr.as_ref(), par),
            m: cross_of(fold.xs_val.as_ref(), fold.xs_tr.as_ref(), par),
        }
    }
}

/// One (fold, column) unit of the n-space `raw_perm` route: this fold's
/// `(ss_res, ss_tot)` for the outcome `y_of` (`y_of(i)` is the raw outcome
/// of full-data row `i`). The outcome is standardized with the fold's own
/// training moments per column, exactly as `cv_fold_contribution` does it
/// (so `ss_tot` is the same bits on both routes), and the validation
/// predictions are `M·alpha`. An `Unresolved` fit is `cv_fold_contribution`
/// itself (the Primal arm), so that unit is the primal route's to the bit,
/// truncation and errors included.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn fold_column_nspace(
    fold: &CvFold,
    nf: &NspaceFold,
    y_of: &dyn Fn(usize) -> f64,
    k: usize,
) -> PlsKitResult<(f64, f64)> {
    debug_assert!(k >= 1);
    let n_tr = fold.train_idx.len();
    let n_val = fold.val_idx.len();
    let y_tr = Col::<f64>::from_fn(n_tr, |i| y_of(fold.train_idx[i]));
    let (z, y_mean, y_scale) = standardize1(y_tr.as_ref());
    let y_pred: Col<f64> = match pls1_nspace_kernel(&nf.gram.block(), z.as_ref(), k) {
        NspaceOutcome::Resolved { alpha, k_used } => {
            debug_assert_eq!(k_used, k);
            seq_gemv(nf.m.as_ref(), &alpha)
        }
        // The primal keeps no component, so its prediction is zero too.
        NspaceOutcome::ZeroModel => Col::<f64>::zeros(n_val),
        NspaceOutcome::Unresolved => return cv_fold_contribution(fold, y_of, k, None),
    };
    // `cv_fold_contribution`'s arithmetic, call for call (change together).
    let ys_val = Col::<f64>::from_fn(n_val, |i| (y_of(fold.val_idx[i]) - y_mean) / y_scale);
    let mean_val: f64 = (0..n_val).map(|i| ys_val[i]).sum::<f64>() / n_val as f64;
    let res: f64 = (0..n_val).map(|i| (y_pred[i] - ys_val[i]).powi(2)).sum();
    let tot: f64 = (0..n_val).map(|i| (ys_val[i] - mean_val).powi(2)).sum();
    Ok((res, tot))
}

/// The n-space block of one prepared `split_exact` split: [`NspaceGram`] of
/// `X̃_tr` and `M = X̃_te X̃_tr'`, built once per split on the calling
/// thread with `par` (`fit::par_fixed()`). Unweighted splits only.
pub(crate) struct NspaceSplit {
    gram: NspaceGram,
    m: Mat<f64>,
}

impl NspaceSplit {
    pub(crate) fn new(prep: &PreparedSplit, par: Par) -> Self {
        Self {
            gram: NspaceGram::new(prep.xs_tr.as_ref(), par),
            m: cross_of(prep.xs_te.as_ref(), prep.xs_tr.as_ref(), par),
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
/// route: element `j` is `Some(r)`, the split-half Pearson `r` for the
/// outcome `y_of(j, ·)`, or `None` when the column needs the Primal arm
/// (`y_of(j, i)` is column `j`'s raw outcome at full-data row `i`; `r` is
/// taken before the driver's clamp and Fisher z). Each training outcome
/// is standardized as `split_column_r` does it; the kernel runs on all `m`
/// columns at once ([`pls1_nspace_kernel_cols`]); the test-half scores are
/// `S = M·A` for the resolved columns' dual coefficients `A`, one product;
/// each correlation is `guarded_pearson`'s, formed from the same
/// `scaled_moments` and `pearson_scaled` calls.
///
/// Two discontinuities decide `r`: truncation (the kernel's gates) and
/// `guarded_pearson`'s `constant_to_rounding` test. A column's scores are
/// kept only when they are not constant to rounding and their centered
/// norm is at least `dual_route::SCORE_BAND` times their norm; otherwise,
/// and whenever the kernel is `Unresolved`, the column is `None`: the
/// caller runs `split_column_r` (the Primal arm) on it, so its `r` is the
/// primal route's to the bit. The fallbacks are left to the caller so that
/// it can spread them over the pool rather than run them inside this run
/// (`signal_test::split_columns_nspace`).
/// Decided directly, because they are exact on both routes: `ZeroModel`
/// (the primal keeps no component, its scores are exactly zero, its guard
/// returns `0.0`) and a test-half outcome constant to rounding (the same
/// bits through the same helper). The caller has already returned `0.0`
/// for a split whose X is not finite.
#[allow(clippy::similar_names)]
pub(crate) fn split_columns_r_nspace(
    sp: &SplitIdx,
    ns: &NspaceSplit,
    m: usize,
    y_of: &dyn Fn(usize, usize) -> f64,
    k: usize,
) -> Vec<Option<f64>> {
    debug_assert!(k >= 1);
    let n_tr = sp.tr.len();
    let n_te = sp.te.len();
    let mut zs = Mat::<f64>::zeros(n_tr, m);
    for j in 0..m {
        let y_tr = Col::<f64>::from_fn(n_tr, |i| y_of(j, sp.tr[i]));
        let (z, _, _) = standardize1(y_tr.as_ref());
        zs.col_mut(j).copy_from(&z);
    }
    let outcomes = pls1_nspace_kernel_cols(&ns.gram.block(), zs.as_ref(), k);
    let mut alphas = Mat::<f64>::zeros(n_tr, m);
    for (j, out) in outcomes.iter().enumerate() {
        if let NspaceOutcome::Resolved { alpha, k_used } = out {
            debug_assert_eq!(*k_used, k);
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
    outcomes
        .iter()
        .enumerate()
        .map(|(j, out)| {
            match out {
                NspaceOutcome::ZeroModel => return Some(0.0),
                NspaceOutcome::Unresolved => return None,
                NspaceOutcome::Resolved { .. } => {}
            }
            let s = scores.col(j);
            let y_te = Col::<f64>::from_fn(n_te, |i| y_of(j, sp.te[i]));
            let ms = scaled_moments(n_te, |i| s[i], None);
            let my = scaled_moments(n_te, |i| y_te[i], None);
            if my.is_constant(n_te) {
                return Some(0.0);
            }
            // Score gate: `a >= b`, so a NaN fails it.
            (!ms.is_constant(n_te) && ms.ss.sqrt() >= SCORE_BAND * ms.sq.sqrt())
                .then(|| pearson_scaled(n_te, |i| s[i], |i| y_te[i], &ms, &my))
        })
        .collect()
}

#[cfg(test)]
mod sweep;

#[cfg(test)]
mod site_tests;

#[cfg(test)]
mod tests;
