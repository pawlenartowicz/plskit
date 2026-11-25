//! Dual (Gram) execution route for resampling loops.
//!
//! Every loop this module serves refits on the same fixed training subset
//! over and over, with only the outcome changing between replicates. The
//! fitted quantities reach `X` through two matrices that do not depend on
//! the outcome at all:
//!
//! ```text
//! G = X̃_tr X̃_tr'     (n_tr × n_tr)   — training Gram
//! M = X̃_te X̃_tr'     (n_te × n_tr)   — train-to-test map
//! ```
//!
//! Build those once per fold or split and every replicate afterwards costs
//! nothing in `p`. `split_exact`'s no-refit route
//! (`signal_test::split_perm_nr_zbars`) uses the same idea: it computes
//! `X̃_te·X̃_tr'·y_tr` as a fixed linear map and batches all `B+1` columns
//! through it. This module covers loops whose argument that batched map does
//! not cover.
//!
//! **The primal route stays primary.** The dual route is a conditional
//! optimization for one corner of the input space (`p` large, `n` small).
//! Where the condition does not hold it is slower — sometimes by an order
//! of magnitude — so an unclassified input takes the primal path.
//!
//! **The route changes no number away from the floor and ties.** Both
//! routes must agree to the crate's bit-near tolerance on every replicate
//! column, not merely on the final p-value, except on a column at the
//! truncation floor or, on PLS3's sparse Y side, near a selection or
//! stopping tie, where the two can differ by up to O(1); the equivalence
//! tests in `signal_test.rs` and `pls3_signal_test.rs` are what makes a
//! second route safe to maintain.

// The n-space PLS1 route for k ≥ 1 lives in `dual_route/multi_k.rs`; its
// items are re-exported below, so their paths are `crate::dual_route::…`.
mod multi_k;
#[cfg(test)]
pub(crate) use multi_k::nspace_blocks_built;
pub(crate) use multi_k::{
    K_DUAL_MAX, NSPACE_BATCH, NspaceFold, NspaceGram, NspaceOutcome, NspaceSplit,
    fold_column_nspace, nspace_allowed_perm_null, nspace_allowed_raw_perm, nspace_allowed_split,
    nspace_perm_row, nspace_run_bytes, pls1_nspace_kernel, split_columns_r_nspace,
};

/// Multiply-adds of the one all-row Gram matrix a route reads its per-unit
/// blocks from (`HeldOutGram::from_x`, a symmetric product): `n²·p/2`, or
/// none when the call already holds it (`gram_held`: the `split_nb` gate's,
/// which `auto` and `split_nb` pass on). The eligibility rules of those
/// routes add their per-unit and per-replicate work and compare the sum
/// with the primal route's work, both in multiply-adds, evaluated in f64,
/// where a rounding can only pick the slower of two routes that agree.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn one_gram_cost(n: usize, p: usize, gram_held: bool) -> f64 {
    if gram_held {
        0.0
    } else {
        n as f64 * n as f64 * p as f64 / 2.0
    }
}

/// Multiply-adds of an n-space route: `gram` for its Gram matrix
/// ([`one_gram_cost`], or `n²·p` for `pls1_perm_null`'s block built from
/// `X̃`), then per unit and outcome column `passes` products with the
/// unit's `n_tr × n_tr` block and one with its `n_val × n_tr` map.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub(crate) fn nspace_work(
    gram: f64,
    units: usize,
    n_cols: usize,
    passes: usize,
    n_tr: usize,
    n_val: usize,
) -> f64 {
    let (u, c, q) = (units as f64, n_cols as f64, passes as f64);
    let (t, v) = (n_tr as f64, n_val as f64);
    gram + u * c * (q * t * t + v * t)
}

/// Multiply-adds of the primal refit over `units` units of `n_cols` outcome
/// columns at `k` components, as the n-space rules count them: per column
/// and unit `X̃'z`, a score and a loading pass per component (the last
/// loading not counted) on the `n_tr` training rows, and `X̃_val·coef` on the
/// `n_val` others, `p` columns each.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn primal_refit_work(
    units: usize,
    n_cols: usize,
    k: usize,
    n_tr: usize,
    n_val: usize,
    p: usize,
) -> f64 {
    let (u, c, kf) = (units as f64, n_cols as f64, k as f64);
    let (t, v) = (n_tr as f64, n_val as f64);
    u * c * (2.0 * kf * t + v) * p as f64
}

/// [`primal_refit_work`] of `raw_perm`'s folds: `n_folds` folds of
/// `n_perm + 1` columns, the largest training fold `n − ⌊n/n_folds⌋`.
pub(crate) fn primal_work_raw_perm(
    n: usize,
    n_folds: usize,
    p: usize,
    n_perm: usize,
    k: usize,
) -> f64 {
    let n_tr = n - n / n_folds;
    primal_refit_work(n_folds, n_perm.saturating_add(1), k, n_tr, n - n_tr, p)
}

/// [`primal_refit_work`] of `split_exact`'s refit route: `n_splits` splits
/// of `n_perm + 1` columns, the halves of `resample::split_sizes(n, k)`.
pub(crate) fn primal_work_split_exact(
    n: usize,
    p: usize,
    n_perm: usize,
    n_splits: usize,
    k: usize,
) -> f64 {
    let (n_train, n_test) = crate::resample::split_sizes(n, k);
    primal_refit_work(n_splits, n_perm.saturating_add(1), k, n_train, n_test, p)
}

/// `split_nb`'s refit on halves: `split_exact`'s with the one observed
/// column per split.
pub(crate) fn primal_work_split_nb(n: usize, p: usize, n_splits: usize, k: usize) -> f64 {
    primal_work_split_exact(n, p, 0, n_splits, k)
}

use faer::Col;

/// Per-column z̄ for the PLS3 `split_exact` statistic through the Gram
/// route. Returns `cols.len()` values; column 0 is the observed Y and
/// columns `1..` are the null columns of `cols`, in order. The route reads
/// X only through `all`, the unweighted all-row Gram matrix of X on its
/// all-row scale (`HeldOutGram::from_x`, or the `split_nb` gate's when the
/// caller holds it), and each split's `G` and `M` are its blocks
/// (`HeldOutGram::part_blocks`). Splits run in batches of
/// `resample::map_units`, each counted as its `G` and `M`; the result does
/// not depend on the batch size.
///
/// `X̃` and `Ỹ` below are the halves of X and Y centred with the training
/// half's means and divided by the block's all-row scale (the scale `all`
/// was built on, and `y_scale`, computed once per call before any
/// permutation), as `pls3_signal_test::centered_halves` builds them on the
/// primal route.
///
/// # The reduction this function exploits
/// PLS3's fit is a singular decomposition, so unlike PLS1 at `K = 1` there
/// is no "no fit" shortcut: permuting Y genuinely moves `u₁`. What there
/// is instead is a reduction that removes `p` from every replicate. With
/// `G = X̃_tr X̃_tr'` and `M = X̃_te X̃_tr'`, the right singular vectors of
/// `A = X̃_tr'Ỹ_tr` are the eigenvectors of
///
/// ```text
/// A'A = Ỹ_tr' G Ỹ_tr          (q × q)
/// ```
///
/// and since `A v₁ = σ₁ u₁` with `σ₁ > 0`,
///
/// ```text
/// X̃_te u₁  =  M (Ỹ_tr v₁) / σ₁
/// ```
///
/// The reported statistic is `cor(X̃_te u₁, Ỹ_te v₁)`, and correlation is
/// invariant to positive scaling of either argument, so the `1/σ₁` drops
/// out and `M (Ỹ_tr v₁)` can stand in for the X-side score directly. The
/// eigenvector's own sign is arbitrary, and flipping it negates both score
/// vectors at once, so `r` is untouched: the same joint-flip argument that
/// makes the primal statistic well defined. `pearson_r_guarded`'s
/// degeneracy guard is relative to each score vector's own magnitude
/// (`linalg::constant_to_rounding`), so the un-rescaled `s` trips it
/// exactly when `X̃_te u₁` would, up to rounding, however small `σ₁` is.
///
/// So each replicate permutes `Ỹ_tr`, forms a `q × q` matrix, eigendecomposes
/// it, and applies `M`. `G` and `M` are built once per split. Unlike PLS1's
/// batched map this is a *loop of cheap work*, not one operation.
///
/// # Truncation
/// The primal fit keeps LV1 only when `σ₁` clears `pls3::SIGMA_FLOOR` and
/// the relative floor `pls3::sigma_rel_floor(n_tr, p, q, ‖X̃_tr‖_F,
/// ‖Ỹ_tr‖_F)` (`max(n_tr, p, q)·ε·‖X̃_tr‖_F·‖Ỹ_tr‖_F`, `fit::w_rel_floor`);
/// otherwise it keeps nothing and `r = 0`. This route applies the same rule
/// to its own `σ₁ = √λ_max`, with `‖X̃_tr‖_F = √trace(G)` and `Ỹ_tr` from
/// the same `centered_halves` call as on the primal route. `λ_max` is an
/// eigenvalue of a `C` formed from a re-centred block of the all-row Gram
/// matrix, so `√λ_max` resolves `σ₁` only to about
/// `√((p + n_tr + q)·ε)·F·‖Ỹ_tr‖_F`, with `F` the training rows' all-row
/// norm, far above the floor. While `F` is of the order of `‖X̃_tr‖_F`,
/// `σ₁` on ordinary data sits orders of magnitude above both and the two
/// routes keep LV1 alike. On a column
/// whose training `Ỹ` is orthogonal to `X̃_tr` up to rounding (a
/// residualized or constructed outcome, possible at `p ≫ n_tr` only when X
/// is rank-deficient) `λ_max` is rounding, and this route can keep a noise
/// LV1 the primal drops: its `r` is then finite but can differ from an X
/// refit by O(1). An exactly zero `X̃_tr` or `Ỹ_tr` gives `λ_max = 0` and
/// `r = 0`, as on the primal route.
///
/// # Precision follows the largest column
/// Every block is formed from `G`, whose rounding is relative to the
/// largest entries of `X̃`, and re-centred, so it carries an absolute error
/// of about `ε·F²` whatever the training block's own size. Under
/// `pre_standardized_x` with columns of very different scale, or under any
/// scaling with a row far from the rest (about 1e9 of the spread or more;
/// 3e-4 relative at 1e9 under the default scaling), the large part buries
/// the structure of the rest: the route can then lose precision on
/// ordinary columns, not only at the floor, by up to O(1).
///
/// A split whose `G` or `M` is not finite (a re-centring of a caller's
/// `gram` that overflows), and a column whose `C` is not finite or whose
/// eigendecomposition fails, give `0`.
///
/// # No `K = 1` restriction in principle
/// PLS3 has no deflation, so the same reduction would serve any
/// `k ≤ min(p, q, n_tr − 1)` by taking the top `k` eigenvectors of the same
/// `q × q` matrix. Only LV1 is computed here because that is the only
/// statistic `pls3_confirmatory_test` accepts.
///
/// # Conditioning
/// `Ỹ'GỸ` squares the condition number of `A`. That is acceptable at the
/// small `q` this family runs at; the equivalence test against the honest
/// SVD refit is what proves it in practice, and the fix if it ever bites is
/// to factor rather than form the Gram.
///
/// A second precondition is that `σ₁` is *simple*. When `σ₁ ≈ σ₂` the top
/// direction is not identified (any rotation inside the leading subspace
/// is an equally valid answer), so the eigendecomposition of `C` here and
/// the SVD of `A` in the primal route can legitimately return different
/// `v₁`, and `r` moves with them by far more than 1e-10. Exact ties are
/// measure-zero on real data, but a near-tie is common at small `q` and is
/// the likeliest cause of an equivalence-test failure that is *not* a bug
/// in either route.
///
/// # What is NOT hoisted
/// Y's training means are recomputed per split *per replicate*: a
/// permutation changes which rows land in the half. Its scale is
/// `y_scale` throughout. `O(n_tr q)` each.
///
/// # Sparse Y side
/// `keep_y = Some(ky)` with `ky < q` makes each training-half fit sparse on
/// the Y side, the way `pls3::spls3_fit` does, and the reduction survives
/// it. With a dense `u` (`keep_x == p`) the primal alternation is
///
/// ```text
/// u <- normalize(A v),   v <- normalize(select(A' u))
/// ```
///
/// so `u` is a strictly positive multiple of `A v = X̃'Ỹv`, and therefore
/// `A'u` is the same positive multiple of `A'A v = Ỹ'GỸ v = C v`.
/// `fit::hard_select_keep` picks coordinates by `|·|` order and a positive
/// rescale cannot change that order, so selecting on `C v` selects exactly
/// the same support as selecting on `A'u`, and normalizing afterwards
/// removes the factor. The two-step primal alternation is thus the one-step
/// map `v <- normalize(select(C v))` run here, and because both routes stop
/// on the same `‖v − v_prev‖_∞ < tol` criterion their sweep counts are
/// expected to agree; see "Near-ties" below.
///
/// The X-side score uses the `v` from *before* the last update, not the
/// returned one: `spls3_component` hands back `u_T = normalize(A v_{T−1})`
/// together with `v_T`, and at the stopping sweep those two differ by up to
/// `tol`, which is far above the 1e-10 the equivalence test asserts. The
/// Y-side score uses `v_T`. The initializer is the same leading direction on
/// both routes, and the map is odd in `v`, so an eigenvector sign that
/// disagrees with the SVD's flips both held-out scores at once and leaves
/// `r` alone.
///
/// `pls3::spls3_fit` drops a component whose sparse `σ = u'Av` falls under
/// either floor, which on this call means `k_used == 0` and `r = 0` on the
/// primal route. Here that `σ` is `v_T'C v_{T−1} / √(v_{T−1}'C v_{T−1})`,
/// under the same rule, and the selected `v`'s norm is guarded against
/// `V_NORM_FLOOR` on every sweep, as `spls3_component` guards it.
///
/// # Near-ties (sparse Y side)
/// "The same map" holds in exact arithmetic only. The two routes evaluate it
/// on different floats (`C v` with `C` formed in full here, `A'u` there,
/// from an eigenvector here and an SVD vector there), and the map has two
/// discontinuities: the support `hard_select_keep` picks flips where the
/// `keep_y`-th and `(keep_y + 1)`-th largest `|·|` cross, and the sweep count
/// jumps where some sweep's `dv` crosses `tol`. Within a few ulps of either
/// the routes can take different branches (a different support, or a stop
/// one sweep apart), and their `r` can separate by up to `O(1)` or `O(tol)`.
/// A third difference grows without crossing a branch: on a nearly
/// degenerate spectrum of `C` on the selected support (relative gap `g`)
/// the alternation damps each sweep's rounding difference only at rate
/// `g`, so `|Δz|` accumulates as about `ε·min(sweeps, 1/g)`, 3e-14 at the
/// default `max_iter = 100`. Near a tie neither answer is more correct;
/// this route's own is finite and does not depend on the thread count.
/// Away from the ties every branch the primal takes is the one taken here
/// (measured `‖v_primal − v_dual‖_∞ ≤ 9e-16` per sweep, `dv` discrepancy
/// `≤ 2e-15`, on the equivalence fixture).
///
/// A sparse X side never reaches here: `keep_x < n_features` zeroes
/// coordinates of `u`, which is not a positive rescale, the proportionality
/// above fails, and `pls3_confirmatory_test` refuses the dual route for
/// that configuration.
///
/// # Preconditions
/// `q >= 1`. Not checked here: `pls3_signal_test::gram_route_allowed`
/// refuses `q == 0` before this function is ever called. `max_iter >= 1`
/// whenever `keep_y` is `Some`, which `pls3_confirmatory_test` validates
/// before either route runs.
///
/// # Shapes
/// - `all`: `HeldOutGram` (`n × n`) of X on its all-row scale, unweighted
/// - `cols`: the `(n, q)` raw outcome `cols.y` and its null columns
/// - `y_scale`: `(q,)`
/// - returns: `(cols.len(),)`, and the number of failed splits (column 0's
///   `r` exactly 0, `signal_test::zbars_from_rows`)
///
/// # Errors
/// `MemoryBudget` when one split does not fit the budget beside the
/// `call_bytes` the caller holds (the all-row Gram matrix it built or holds).
///
/// # Panics
/// Never (indexing is over the caller-supplied splits and the row
/// permutations `cols` draws from its own seeds).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names, clippy::too_many_arguments)]
pub(crate) fn pls3_split_zbars_columns(
    all: &crate::held_out_gram::HeldOutGram,
    cols: &crate::resample::Columns<'_>,
    splits: &[crate::signal_test::SplitIdx],
    y_scale: faer::ColRef<'_, f64>,
    keep_y: Option<usize>,
    max_iter: usize,
    tol: f64,
    call_bytes: usize,
) -> crate::error::PlsKitResult<(Vec<f64>, usize)> {
    // Splits in batches of `resample::map_units`, each counted by
    // `pls3_signal_test::gram_unit_bytes`; the columns of a split map in
    // parallel. Nothing the size of X is written. Every split has the
    // same training size (the split sizes depend on `n` and `k` only), so
    // the first split's stands for all.
    let (n_tr, n_te) = splits
        .first()
        .map_or((0, 0), |sp| (sp.tr.len(), sp.te.len()));
    let gram_unit = crate::pls3_signal_test::gram_unit_bytes(
        n_tr,
        n_te,
        cols.y.ncols(),
        cols.len(),
        rayon::current_num_threads(),
    );
    crate::signal_test::zbars_over_splits(splits, cols.len(), 0, call_bytes, gram_unit, |sp| {
        Ok(
            pls3_split_columns_r(all, cols, sp, y_scale, keep_y, max_iter, tol)
                .into_iter()
                .map(crate::signal_test::fisher_z)
                .collect(),
        )
    })
}

/// The held-out LV correlations `r` of split `sp` for every column of
/// `cols`, before the clamp and Fisher z: the per-split body of
/// [`pls3_split_zbars_columns`], which owns the reduction, its truncation
/// and its degenerate cases (`r = 0`). `pls3_confirmatory_test_gram`'s
/// `split_nb` reads column 0's.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::items_after_statements)]
#[allow(clippy::too_many_lines)]
pub(crate) fn pls3_split_columns_r(
    all: &crate::held_out_gram::HeldOutGram,
    cols: &crate::resample::Columns<'_>,
    sp: &crate::signal_test::SplitIdx,
    y_scale: faer::ColRef<'_, f64>,
    keep_y: Option<usize>,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let (p, q) = (all.n_features(), cols.y.ncols());
    let n_cols = cols.len();

    /// The floor `spls3_component` applies to the selected `v` before
    /// normalizing it. Named directly from `pls3` rather than re-declared, so
    /// the two routes cannot drift apart. The two routes select on vectors
    /// that differ by a positive scalar (`Cv` here, `A'u = Cv/‖Av‖` there),
    /// so this is a defensive degenerate-input guard, not a boundary the two
    /// routes can be expected to cross on the same input; reaching it needs a
    /// selected `v` that is numerical dust, where both routes already report
    /// no association.
    use crate::pls3::SIGMA_FLOOR as V_NORM_FLOOR;

    let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
    let n_tr = tr.len();
    // Built once per split; neither carries p into the replicate loop.
    // Both are blocks of the all-row Gram matrix with the halves centred
    // by the training means, so this route writes no half of X.
    let (g, m) = all.part_blocks(tr, te, None);
    if !(g.is_all_finite() && m.is_all_finite()) {
        return vec![0.0; n_cols];
    }
    // `‖X̃_tr‖_F` for the relative floor.
    let x_fro = (0..n_tr).map(|i| g[(i, i)]).sum::<f64>().max(0.0).sqrt();

    crate::resample::map_indexed(n_cols, |col| {
        // Column `col` of `cols`, the block the primal route reads too.
        let y_c = cols.block(col);
        let (ys_tr, ys_te) =
            crate::pls3_signal_test::centered_halves(y_c.as_ref(), tr, te, y_scale);

        // C = Ỹ_tr' G Ỹ_tr  (q × q), formed as Ỹ'(GỸ) so the
        // n_tr × n_tr Gram is applied, never squared. Seq inside the
        // per-column worker: outer Rayon owns the pool.
        let seq = faer::Par::Seq;
        let gy = crate::linalg::mat_mul(g.as_ref(), ys_tr.as_ref(), seq);
        let c = crate::linalg::mat_mul(ys_tr.transpose(), gy.as_ref(), seq);
        if !c.is_all_finite() {
            return 0.0;
        }

        // Lower triangle pinned for byte-parity stability. faer returns
        // eigenvalues ascending, so the leading pair is the last.
        let Ok((lambda, eig_u)) = crate::linalg::self_adjoint_eigen(c.as_ref(), seq) else {
            return 0.0;
        };
        // The primal fit's truncation rule (see "Truncation" above), on
        // this route's `σ`. A NaN fails it.
        let floor = crate::pls3::sigma_rel_floor(n_tr, p, q, x_fro, ys_tr.norm_l2());
        let keeps = |sigma: f64| sigma >= crate::pls3::SIGMA_FLOOR && sigma >= floor;
        // Dense LV1: `σ₁ = √λ_max`. A sparse `σ` is at most `σ₁`, so a
        // dropped `σ₁` drops the sparse component too.
        if !keeps(lambda[q - 1].sqrt()) {
            return 0.0;
        }
        // Sparse Y side: alternate in q-space. `None`, and the
        // dense endpoint `keep_y == q`, are a LITERAL skip of the
        // alternation (`_ => None` below), so the dense route's
        // float sequence is untouched.
        let sparse_v: Option<(Col<f64>, Col<f64>)> = match keep_y {
            Some(ky) if ky < q => {
                let mut cv: Col<f64> = Col::<f64>::zeros(q);
                // The primal map is
                //   v <- normalize(select(A' normalize(A v)))
                // and with a dense `u` the inner normalization is a
                // positive scalar, leaving
                //   v <- normalize(select(C v)),  C = Ỹ'GỸ.
                // Selection by |v| order is invariant to that
                // positive factor, so this is the same map, and the
                // `v`-only convergence test is the same test.
                // Select, norm, reciprocal, multiply is
                // `fit::select_and_normalize`, the call
                // `pls3::spls3_component` makes, so the two routes
                // share that float sequence by construction: a
                // one-ulp split here can move the stopping sweep.
                let mut v: Col<f64> = eig_u.col(q - 1).to_owned();
                // The X-side score uses the `v` that produced the
                // returned `u`, which is the `v` from *before* the
                // last update: `spls3_component` returns
                // `u_T = normalize(A v_{T-1})` alongside `v_T`, and
                // at the stopping sweep those two differ by up to
                // `tol`, far above the 1e-10 the equivalence test
                // asserts.
                // Overwritten by the first sweep before any read (`max_iter >= 1`
                // is validated upstream); the binding exists only because `v_for_u`
                // is read after the loop, so it must be definitely initialised.
                let mut v_for_u: Col<f64> = Col::<f64>::zeros(q);
                for _ in 0..max_iter {
                    v_for_u.copy_from(&v);
                    // `C v` into the scratch column, then swap it in.
                    // This is `linalg::mat_vec` (zeroed destination,
                    // `Accum::Replace`, `Par::Seq`), minus its
                    // allocation.
                    faer::linalg::matmul::matmul(
                        cv.as_mut(),
                        faer::Accum::Replace,
                        c.as_ref(),
                        v.as_ref(),
                        1.0,
                        seq,
                    );
                    std::mem::swap(&mut v, &mut cv);
                    if crate::fit::select_and_normalize(&mut v, ky, V_NORM_FLOOR).is_none() {
                        return 0.0;
                    }
                    let dv = (0..q)
                        .map(|j| (v[j] - v_for_u[j]).abs())
                        .fold(0.0_f64, f64::max);
                    if dv < tol {
                        break;
                    }
                }
                // The sparse `σ = u'Av` the primal fit truncates
                // on, with `u = Av_{T−1}/‖Av_{T−1}‖`:
                // `σ = v_T'C v_{T−1} / √(v_{T−1}'C v_{T−1})`.
                let cvp = crate::linalg::mat_vec(c.as_ref(), v_for_u.as_ref(), seq);
                let num: f64 = (0..q).map(|j| v[j] * cvp[j]).sum();
                let den2: f64 = (0..q).map(|j| v_for_u[j] * cvp[j]).sum();
                if !keeps(num / den2.sqrt()) {
                    return 0.0;
                }
                Some((v_for_u, v))
            }
            _ => None,
        };
        // Dense arm keeps borrowing the eigenvector column itself,
        // so not one float of the dense path moves.
        let (v_u, v1) = match &sparse_v {
            Some((vp, vt)) => (vp.as_ref(), vt.as_ref()),
            None => (eig_u.col(q - 1), eig_u.col(q - 1)),
        };

        let yv = crate::linalg::mat_vec(ys_tr.as_ref(), v_u, seq);
        let s = crate::linalg::mat_vec(m.as_ref(), yv.as_ref(), seq); // ∝ X̃_te u₁, positive factor
        let t = crate::linalg::mat_vec(ys_te.as_ref(), v1, seq);

        crate::pls3_signal_test::pearson_r_guarded(&s, &t)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// A split whose kernel is not finite gives every column 0, the value a
    /// non-finite block gets on the primal route, in the route's one pass.
    /// X is used as given (unit column scales, as under
    /// `pre_standardized_x`): column 0 at `±1e300` on a training and a
    /// held-out row overflows `K_A` (`G_00`) and `M`; at `1e10` on a
    /// training row and `±1e300` on two held-out rows it overflows only `M`
    /// (`G_60`), and an X refit on that input sees finite halves and gives
    /// non-zero r: 0 is the route's convention there. Dense and with
    /// `keep_y = 1`, on the observed and five null columns.
    #[test]
    fn pls3_dual_gives_zero_on_a_non_finite_kernel() {
        use crate::signal_test::SplitIdx;
        let sp = SplitIdx {
            tr: (0..6).collect(),
            te: (6..12).collect(),
        };
        let fit = crate::pls3::Pls3FitOpts::default();
        let (x0, _) = crate::test_support::signal_data(12, 3, 5);
        let y = Mat::<f64>::from_fn(12, 2, |i, j| x0[(i, 1)] + x0[(i, j)]);
        let mut x_both = x0.clone();
        x_both[(0, 0)] = 1e300;
        x_both[(6, 0)] = -1e300;
        let mut x_cross = x0.clone();
        x_cross[(0, 0)] = 1e10;
        x_cross[(6, 0)] = 1e300;
        x_cross[(7, 0)] = -1e300;
        let ones = Col::<f64>::from_fn(3, |_| 1.0);
        let ys = crate::linalg::held_out_scale(y.as_ref(), None, false);
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 0,
            n_perm: 5,
            null_map: None,
            w: None,
        };
        for (label, x, k_a_finite) in [("K_A and M", &x_both, false), ("M", &x_cross, true)] {
            let g = crate::held_out_gram::HeldOutGram::from_x(x.as_ref(), None, ones.as_ref());
            let (k_a, m) = g.part_blocks(&sp.tr, &sp.te, None);
            assert_eq!(k_a.is_all_finite(), k_a_finite, "{label}: premise, K_A");
            assert!(!m.is_all_finite(), "{label}: premise, M");
            for keep_y in [None, Some(1)] {
                let (z, log) = crate::resample::with_batch_log(|| {
                    pls3_split_zbars_columns(
                        &g,
                        &cols,
                        std::slice::from_ref(&sp),
                        ys.as_ref(),
                        keep_y,
                        fit.max_iter,
                        fit.tol,
                        0,
                    )
                    .unwrap()
                    .0
                });
                assert_eq!(log.len(), 1, "{label} {keep_y:?}: one pass, {log:?}");
                assert!(
                    z.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()),
                    "{label} {keep_y:?}: {z:?}"
                );
            }
        }
    }

    /// Through `pls3_confirmatory_test` on a shape that takes the Gram
    /// route, both overflowing inputs give a finite statistic, on the dense
    /// arm and on the sparse Y arm.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_split_exact_on_the_gram_route_stays_finite_when_x_overflows() {
        let (n, p, q, n_perm) = (40, 200, 3, 19);
        assert!(
            crate::pls3_signal_test::split_exact_takes_gram(None, n, p, q, n_perm, 6, false),
            "premise: Gram route"
        );
        let (x0, _) = crate::test_support::signal_data(n, p, 11);
        let y = Mat::<f64>::from_fn(n, q, |i, j| x0[(i, j + 1)] + x0[(i, j + 2)]);
        // A split holding one of rows 0 and 1 in each half overflows when
        // the held-out one is centred with the training mean.
        let mut held_out = x0.clone();
        held_out[(0, 0)] = f64::MAX;
        held_out[(1, 0)] = -f64::MAX;
        let mut centering = x0.clone();
        centering[(0, 0)] = f64::MAX;
        centering[(1, 0)] = f64::MAX;
        centering[(2, 0)] = -f64::MAX;
        for (label, x) in [("held-out", &held_out), ("centering", &centering)] {
            for keep_y in [None, Some(1)] {
                let r = crate::pls3_confirmatory_test(
                    x.as_ref(),
                    y.as_ref(),
                    1,
                    crate::Pls3ConfirmatoryTestOpts {
                        args: crate::signal_test::ConfirmatoryArgs::SplitExact {
                            n_perm,
                            n_splits: 6,
                        },
                        keep_y,
                        seed: Some(7),
                        ..Default::default()
                    },
                )
                .expect("a result");
                assert!(
                    r.statistic.unwrap().is_finite(),
                    "{label} {keep_y:?}: {}",
                    r.statistic.unwrap()
                );
            }
        }
    }

    /// Wide X and a Y whose column `j` is `x_j + x_{j+q}`, with the all-row
    /// scales of both.
    #[allow(clippy::many_single_char_names)]
    fn pls3_linked(n: usize, p: usize, q: usize, seed: u64) -> [Mat<f64>; 2] {
        let (x, _) = crate::test_support::signal_data(n, p, seed);
        let y = Mat::<f64>::from_fn(n, q, |i, j| x[(i, j)] + x[(i, j + q)]);
        [x, y]
    }

    fn all_row_scales(x: &Mat<f64>, y: &Mat<f64>) -> [Col<f64>; 2] {
        [
            crate::linalg::held_out_scale(x.as_ref(), None, false),
            crate::linalg::held_out_scale(y.as_ref(), None, false),
        ]
    }

    /// The kernels of every split read from one all-row Gram matrix give
    /// each column's z̄ within 1e-12 of the primal route, which builds the
    /// halves of X (40 × 120, q = 3, B = 49, dense and `keep_y = 2`).
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_dual_from_gram_matches_the_streamed_kernels() {
        let (n, p, q, n_perm) = (40, 120, 3, 49);
        let [x, y] = pls3_linked(n, p, q, 23);
        let [xs, ys] = all_row_scales(&x, &y);
        let splits = crate::signal_test::draw_splits(n, 1, 6, 23).expect("splits");
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 23,
            n_perm,
            null_map: None,
            w: None,
        };
        assert!(
            crate::pls3_signal_test::split_exact_takes_gram(None, n, p, q, n_perm, 6, false),
            "premise: Gram route"
        );
        let held = crate::held_out_gram::HeldOutGram::from_x(x.as_ref(), None, xs.as_ref());
        for keep_y in [None, Some(2)] {
            let o = crate::Pls3ConfirmatoryTestOpts {
                keep_y,
                ..Default::default()
            };
            let primal = crate::pls3_signal_test::pls3_split_zbars_columns_primal(
                x.as_ref(),
                &cols,
                &splits,
                xs.as_ref(),
                ys.as_ref(),
                &o,
                0,
            )
            .unwrap()
            .0;
            if let Some(ky) = keep_y {
                crate::pls3_signal_test::assert_away_from_keep_ties(
                    x.as_ref(),
                    &cols,
                    &splits,
                    xs.as_ref(),
                    ys.as_ref(),
                    ky,
                    o.max_iter,
                    o.tol,
                );
            }
            let dual = pls3_split_zbars_columns(
                &held,
                &cols,
                &splits,
                ys.as_ref(),
                keep_y,
                o.max_iter,
                o.tol,
                0,
            )
            .unwrap()
            .0;
            for (c, (g, z)) in dual.iter().zip(&primal).enumerate() {
                assert!(
                    (g - z).abs() <= 1e-12,
                    "keep_y={keep_y:?}, column {c}: gram {g:e}, primal {z:e}"
                );
            }
        }
    }

    /// Every null block of `cols` overwritten with Y by its `null_map` gives
    /// each null column the observed z̄, so p = 1.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_dual_reads_columns_as_given() {
        let (n, p, q, n_perm) = (40, 120, 3, 49);
        let [x, y] = pls3_linked(n, p, q, 31);
        let [xs, ys] = all_row_scales(&x, &y);
        let splits = crate::signal_test::draw_splits(n, 1, 6, 31).expect("splits");
        let observed = |_: usize, mut m: faer::MatMut<'_, f64>| m.copy_from(y.as_ref());
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seed: 31,
            n_perm,
            null_map: Some(&observed),
            w: None,
        };
        let fit = crate::pls3::Pls3FitOpts::default();
        let g = crate::held_out_gram::HeldOutGram::from_x(x.as_ref(), None, xs.as_ref());
        let z = pls3_split_zbars_columns(
            &g,
            &cols,
            &splits,
            ys.as_ref(),
            None,
            fit.max_iter,
            fit.tol,
            0,
        )
        .unwrap()
        .0;
        let exceed = z[1..].iter().filter(|v| **v >= z[0]).count();
        assert_eq!(exceed, n_perm, "p must be 1: z = {z:?}");
    }

    /// One G per call on wide X: `auto` runs `split_exact` (as on every PLS3
    /// design) on the Gram route, which builds its G, and a `split_nb`
    /// request, which the gate reroutes on this near rank-one X, runs it
    /// from the gate's G.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_auto_builds_one_gram() {
        use rand::{RngExt, SeedableRng};
        let (n, p, q) = (250, 2000, 3);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(29);
        let f: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x = Mat::<f64>::from_fn(n, p, |i, _| 3.0 * f[i] + rng.random_range(-1.0..1.0));
        let y = Mat::<f64>::from_fn(n, q, |i, j| {
            let e: f64 = rng.random_range(-1.0..1.0);
            if j == 0 { f[i] + e } else { e }
        });
        assert!(
            crate::pls3_signal_test::split_exact_takes_gram(None, n, p, q, 99, 4, true),
            "premise: Gram route"
        );
        for args in [
            crate::signal_test::ConfirmatoryArgs::Auto {
                n_perm: 99,
                n_splits: 4,
            },
            crate::signal_test::ConfirmatoryArgs::SplitNb {
                n_splits: 4,
                force: false,
            },
        ] {
            let before = crate::held_out_gram::grams_built();
            let r = crate::pls3_confirmatory_test(
                x.as_ref(),
                y.as_ref(),
                1,
                crate::Pls3ConfirmatoryTestOpts {
                    args,
                    seed: Some(29),
                    ..Default::default()
                },
            )
            .expect("a result");
            assert_eq!(r.test_method, "split_exact", "{args:?}");
            assert_eq!(crate::held_out_gram::grams_built(), before + 1, "{args:?}");
        }
    }
}
