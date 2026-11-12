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
    fold_column_nspace, nspace_eligible_perm_null, nspace_eligible_raw_perm,
    nspace_eligible_split_exact, nspace_eligible_split_nb, nspace_perm_row, pls1_nspace_kernel,
    split_columns_r_nspace,
};

/// Hard cap on the training-set size the dual route will accept.
///
/// The routes build the all-row `G` (`n²` f64s, about 512 MB at a split
/// route's `n ≈ 2·4000`, 200 MB with 5 folds), then one training block
/// `K_A` and `M` per running unit: folds (`pls1_cv_r2_columns`) and splits
/// (`pls3_split_zbars_columns`) run in batches of `resample::map_units`.
/// The flop rule below already excludes training sets that large,
/// so this should never bind; it bounds `G` specifically, so getting the
/// flop rule wrong should make `G` degrade to "slower", never to an
/// out-of-memory kill. It does not bound `run_raw_perm`'s `n × (B + 1)`
/// outcome buffer `y_mat` in `signal_test.rs`, which scales with `B`
/// unchecked.
pub(crate) const DUAL_ROUTE_MAX_N_TR: usize = 4000;

/// Should this loop take the dual route?
///
/// The cost rule of a Gram route that builds its Gram matrix per unit:
/// `pls1_perm_null`'s n-space route (`multi_k::nspace_eligible_perm_null`),
/// whose one unit is the call and whose `G = X̃X̃'` is a full product of the
/// standardized block. The routes that read their blocks from one all-row
/// Gram matrix per call decide by that cost instead ([`one_gram_cost`]).
///
/// `n_tr` is the block's row count, `p` the feature count,
/// `n_replicates` the number of outcome columns amortizing one precompute
/// (`B` at `pls1_perm_null`, whose observed fit stays primal), and `q` the
/// per-replicate multiplicity: how many `n_tr × p` passes a primal
/// replicate makes, and how many `n_tr × n_tr` products a Gram replicate
/// makes, the component count `k`, whose `G` precompute does not grow with
/// `k`. The rule is conservative: a primal PLS1 replicate makes about
/// `2k + 1` passes over `X̃`, not `k`.
///
/// # The rule
/// Flop counts for one unit are `B·n_tr·p·q` primal versus
/// `n_tr²·p + B·n_tr²·q` dual, giving
/// `speedup ≈ (1/p + 1/(B·q))⁻¹ / n_tr`, so the route pays exactly when
///
/// ```text
/// n_tr < (1/p + 1/(B·q))⁻¹
/// ```
///
/// Rearranged to `n_tr·(B·q + p) < p·B·q` to avoid dividing. The numerator
/// is a *harmonic* combination of `p` and `B·q`, not their minimum: it falls
/// to half the min when the two are comparable, so rounding the rule up to
/// `min(p, B·q)` would admit a band of inputs that are slower, not faster.
/// Both halves have to hold and they fail for different reasons:
/// `n_tr ≥ p` means the Gram is bigger than the data it came from, and
/// `n_tr ≥ B·q` means there are not enough replicates to amortize the
/// precompute.
///
/// These are flop counts, not measurements. The primal inner operation is
/// memory-bandwidth bound at small `q` while the Gram is a high-intensity
/// BLAS-3 call, so the measured crossover sits somewhere other than this
/// one, which is why the rule is used as a conservative guard and not as a
/// tuned threshold.
#[allow(clippy::many_single_char_names)]
pub(crate) fn use_dual_route(n_tr: usize, p: usize, n_replicates: usize, q: usize) -> bool {
    // `n_tr = 0` would satisfy the rule below for any positive `p` and
    // `B·q`. A zero `p`, `n_replicates` or `q` fails it on its own.
    if n_tr == 0 || n_tr > DUAL_ROUTE_MAX_N_TR {
        return false;
    }
    let bq = n_replicates.saturating_mul(q);
    // f64 rather than integer arithmetic: `n_tr·(bq + p)` and `p·bq` stay
    // inside f64's exact-integer range (2^53) for the `p`, `n_replicates`
    // and `q` this crate's call sites actually pass — 1e11 at the largest
    // test case — so in practice the comparison is exact. Nothing here
    // enforces that bound, but a rounding past it is harmless: it can only
    // pick the slower of two routes the equivalence tests already require to
    // agree numerically.
    #[allow(clippy::cast_precision_loss)]
    let (n_f, p_f, bq_f) = (n_tr as f64, p as f64, bq as f64);
    n_f * (bq_f + p_f) < p_f * bq_f
}

/// Multiply-adds of the one all-row Gram matrix a route reads its per-unit
/// blocks from (`HeldOutGram::from_x`, a symmetric product): `n²·p/2`, or
/// none when the call already holds it (`gram_held`: the `split_nb` gate's,
/// which `auto` and `split_nb` pass on). The eligibility rules of those
/// routes add their per-unit and per-replicate work and compare the sum
/// with the primal route's work, both in multiply-adds. Like
/// [`use_dual_route`] they are evaluated in f64, where a rounding can only
/// pick the slower of two routes that agree.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn one_gram_cost(n: usize, p: usize, gram_held: bool) -> f64 {
    if gram_held {
        0.0
    } else {
        n as f64 * n as f64 * p as f64 / 2.0
    }
}

use faer::{Col, MatRef};

/// Multiple of its rounding bound by which `split_exact`'s tall moment arm
/// at k ≥ 2 (`signal_test::split_column_r_gram_p_moments`) requires its
/// centred test-half scores, and its constant tests, to clear zero before
/// it keeps its own correlation instead of the test-row arm's.
pub(crate) const RESOLVE_BAND: f64 = 4.0;

/// Conditioning cap of the tall moment arms (`signal_test::moments_resolved`):
/// a column keeps the correlation formed from the test half's moments only
/// when its centred test-half scores have norm `t_c ≥ SCORE_BAND·F_te·‖coef‖`,
/// that is `κ = F_te·‖coef‖/t_c ≤ 1/SCORE_BAND`; otherwise the test-row arm
/// scores it. The two arms' `r` differ by about `η·κ`, with `η` below
/// `1e-16` as measured by
/// `signal_test::tests::moment_gate_sends_ill_conditioned_scores_to_the_test_rows`,
/// so past the cap `r` moves by at most about `1e-13`, inside the `1e-12`
/// bit-near tolerance.
pub(crate) const SCORE_BAND: f64 = 1e-3;

/// Pooled K-fold CV R² for PLS1 at K = 1, for every column of `y_mat` at
/// once, through the Gram route.
///
/// # The identity this function exploits
/// At `K = 1` NIPALS gives `w = X̃'z/‖X̃'z‖`, `t = X̃w`, `p = X̃'t/(t't)`, so
/// `p'w = t't/(t't) = 1` *exactly* and `pls1_coef_at_k`'s
/// `coef = W(P'W)⁻¹Q` collapses to `coef = w·q`. Writing `G = X̃_tr X̃_tr'`
/// and `M = X̃_te X̃_tr'` and substituting:
///
/// ```text
/// coef = X̃_tr'z · c   with   c = (z'G z) / (z'G² z)
/// ŷ_te = X̃_te coef   = c · M z
/// ```
///
/// Both quadratic forms come from two `O(n_tr²)` matrix-vector products —
/// `G²z` as `G(Gz)`, never forming `G²`. So CV R², which is *not* invariant
/// to the scale of the coefficient, is reachable without refits: unlike
/// `split_perm_nr_zbars`' correlation argument, the scalar itself is needed,
/// and it too is a dual quantity. `K ≥ 2` breaks this closed form, because
/// deflation makes component 2's weights depend on component 1's
/// y-dependent scores. It does not break the Gram route itself: `raw_perm`
/// sends dense, unweighted `K ≥ 2` input (up to `multi_k::K_DUAL_MAX`) to
/// the n-space kernel (`multi_k::pls1_nspace_kernel`) instead.
///
/// # Truncation
/// `fit::pls1_kernel` drops the component, and the fold then predicts zero,
/// when `‖X̃'z‖` falls under
/// `max(1e-14, fit::w_rel_floor(n_tr, p, ‖X̃_tr‖_F, ‖z‖))` or `t't` under
/// `1e-14`. Here `‖X̃'z‖ = √(z'Gz)` and `t't = (z'G²z)/(z'Gz)`, with `z`
/// from the same `standardize1` call as on the primal route and
/// `‖X̃_tr‖_F = √trace(G)`, equal to the primal route's `norm_l2` to
/// rounding, and the same exits apply. `z'Gz` is formed from a rounded `G`
/// and resolves `‖X̃'z‖` only to about `√(p·ε)·‖X̃_tr‖_F·‖z‖`, far above the
/// floor, so on a training `y` orthogonal to `X̃_tr` up to rounding this
/// route can keep a component (`c` then divides one rounding residue by
/// another) where the primal predicts zero. A constant training `y` gives
/// `z = 0` and a zero prediction on both routes.
///
/// # What is NOT hoisted
/// `y` is standardized with each fold's own training moments and
/// `signal_test::pooled_cv_r2_columns` pools `ss_res` / `ss_tot` across
/// folds, so the per-fold `y` scale does not cancel out of the pooled ratio.
/// That standardization is therefore recomputed per fold *per column*:
/// `O(n)` each, free next to the Gram products, and wrong if hoisted.
///
/// # Scope
/// Dense, unweighted, `K = 1`. Those are not checked: the parameters that
/// would make them otherwise are simply absent from the signature, so an
/// ineligible call cannot be written. `run_raw_perm` owns the routing
/// decision and passes the call's all-row Gram matrix `g`
/// (`signal_test::block_gram`); each fold's `G` and `M` are its blocks
/// (`HeldOutGram::part_blocks`). Folds run in batches of
/// `resample::map_units`, each counted as its `G` and `M`; the result does
/// not depend on the batch size.
///
/// # Shapes
/// - `g`: the all-row Gram matrix of `X̃` (`n × n`), X centred over all
///   rows and divided by the one column scale over all rows
///   (`linalg::held_out_scale`)
/// - `y_mat`: `(n, n_cols)` raw outcomes; column 0 is the observed `y`
/// - returns: `(n_cols,)` pooled CV R², one per column
///
/// # Errors
/// `NonFiniteInput` when a fold's standardized training y of column 0 is
/// not finite (a finite y near `±f64::MAX` can overflow in `y − mean`). A
/// null column that fails the same way is NaN instead, as in
/// `signal_test::pooled_cv_r2_columns` (change together).
///
/// # Panics
/// Never (all indexing is over caller-supplied fold index vectors).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub(crate) fn pls1_cv_r2_columns(
    g: &crate::held_out_gram::HeldOutGram,
    y_mat: MatRef<'_, f64>,
    folds: &[Vec<usize>],
) -> crate::error::PlsKitResult<Vec<f64>> {
    use crate::fit::NIPALS_ABS_FLOOR;
    use crate::linalg::standardize1;

    type Contrib = crate::error::PlsKitResult<(f64, f64)>;

    let n_cols = y_mat.ncols();
    let p = g.n_features();
    let train_rows = |fi: usize| crate::signal_test::fold_train_rows(folds, fi);

    // One fold's contribution for every column. Nothing the size of X is
    // written.
    let fold_columns = |fi: usize| -> Vec<Contrib> {
        let val_idx = folds[fi].as_slice();
        let train_idx = train_rows(fi);
        let n_tr = train_idx.len();
        let n_val = val_idx.len();
        // Built once per fold and reused by every column. G is n_tr × n_tr,
        // M is n_val × n_tr, both blocks of the all-row Gram matrix with the
        // halves centred by the training means; neither carries p.
        let (gk, m) = g.part_blocks(&train_idx, val_idx, None);
        let x_fro = (0..n_tr).map(|i| gk[(i, i)]).sum::<f64>().max(0.0).sqrt();

        crate::resample::map_indexed(n_cols, |col| {
            // Column `col` on the training rows, standardized with its own
            // moments.
            let y_tr = Col::<f64>::from_fn(n_tr, |i| y_mat[(train_idx[i], col)]);
            let (z, y_mean, y_scale) = standardize1(y_tr.as_ref());
            // The primal fold fit's y-side check (`check_fit_y_and_k` in
            // `signal_test::cv_fold_contribution`; change together).
            crate::fit::check_finite_col(z.as_ref())?;
            let z_norm = z.norm_l2();

            // Seq inside the per-column worker: outer Rayon owns the pool.
            let seq = faer::Par::Seq;
            let g1 = crate::linalg::mat_vec(gk.as_ref(), z.as_ref(), seq); // G z
            let g2 = crate::linalg::mat_vec(gk.as_ref(), g1.as_ref(), seq); // G² z, as G(Gz)
            let num: f64 = (0..n_tr).map(|i| z[i] * g1[i]).sum(); // z'Gz
            let den: f64 = (0..n_tr).map(|i| z[i] * g2[i]).sum(); // z'G²z

            // fit::pls1_kernel's exits (change together): a truncated K = 1
            // fit has coef = 0, hence a zero prediction.
            let w_norm = num.max(0.0).sqrt();
            let tt = if num > 0.0 { den / num } else { 0.0 };
            let w_floor = crate::fit::w_rel_floor(n_tr, p, x_fro, z_norm);
            let stops = w_norm < NIPALS_ABS_FLOOR.max(w_floor) || tt < NIPALS_ABS_FLOOR;
            let y_pred: Col<f64> = if stops {
                Col::<f64>::zeros(n_val)
            } else {
                let c = num / den;
                let mz = crate::linalg::mat_vec(m.as_ref(), z.as_ref(), seq);
                Col::<f64>::from_fn(n_val, |i| c * mz[i])
            };
            // `(ss_res, ss_tot)` on the validation rows, y on the training
            // fold's scale (`signal_test::cv_fold_ss`; change together).
            let ys_val =
                Col::<f64>::from_fn(n_val, |i| (y_mat[(val_idx[i], col)] - y_mean) / y_scale);
            #[allow(clippy::cast_precision_loss)]
            let mean_val: f64 = (0..n_val).map(|i| ys_val[i]).sum::<f64>() / n_val as f64;
            let res: f64 = (0..n_val).map(|i| (y_pred[i] - ys_val[i]).powi(2)).sum();
            let tot: f64 = (0..n_val).map(|i| (ys_val[i] - mean_val).powi(2)).sum();
            Ok((res, tot))
        })
    };

    // Folds in batches of `resample::map_units`, each counted as `G`
    // (`n_tr × n_tr`) and `M` (`n_val × n_tr`) at the largest training set;
    // the columns of a fold map in parallel.
    let n = y_mat.nrows();
    let n_tr_max = n - folds.iter().map(Vec::len).min().unwrap_or(0);
    let gram_unit = crate::resample::fold_blocks_bytes(n, folds, n_tr_max);
    let rows: Vec<Vec<Contrib>> = crate::resample::map_units(folds.len(), gram_unit, fold_columns);

    // Fold order, from `0.0`: a failed fit of column 0 is returned (the
    // first in fold order); a failed fit of another column makes it NaN,
    // and its contributions from the other folds are dropped.
    let mut ss_res = vec![0.0_f64; n_cols];
    let mut ss_tot = vec![0.0_f64; n_cols];
    let mut failed = vec![false; n_cols];
    for row in rows {
        for (col, r) in row.into_iter().enumerate() {
            if failed[col] {
                continue;
            }
            match r {
                Ok((res, tot)) => {
                    ss_res[col] += res;
                    ss_tot[col] += tot;
                }
                Err(e) if col == 0 => return Err(e),
                Err(_) => failed[col] = true,
            }
        }
    }

    // Pooled ratio, matching signal_test::pooled_cv_r2_columns's final
    // expression exactly.
    Ok((0..n_cols)
        .map(|col| {
            if failed[col] {
                f64::NAN
            } else if ss_tot[col] > 0.0 {
                1.0 - ss_res[col] / ss_tot[col]
            } else {
                0.0
            }
        })
        .collect())
}

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
/// A split whose `G` or `M` is not finite (a re-centring that overflows),
/// a column whose `C` is not finite or whose eigendecomposition fails, and a
/// column whose held-out Y block is not finite give `0`, the value the
/// primal route gives a non-finite block.
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
/// `q >= 1`. Not checked here: `pls3_signal_test::dual_route_eligible`
/// refuses `q == 0` before this function is ever called. `max_iter >= 1`
/// whenever `keep_y` is `Some`, which `pls3_confirmatory_test` validates
/// before either route runs.
///
/// # Shapes
/// - `all`: `HeldOutGram` (`n × n`) of X on its all-row scale, unweighted
/// - `cols`: the `(n, q)` raw outcome `cols.y` and its null columns
/// - `y_scale`: `(q,)`
/// - returns: `(cols.len(),)`
///
/// # Panics
/// Never (indexing is over the caller-supplied splits and the row
/// permutations `cols` draws from its own seeds).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub(crate) fn pls3_split_zbars_columns(
    all: &crate::held_out_gram::HeldOutGram,
    cols: &crate::resample::Columns<'_>,
    splits: &[crate::signal_test::SplitIdx],
    y_scale: faer::ColRef<'_, f64>,
    keep_y: Option<usize>,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    // Splits in batches of `resample::map_units`, each counted as `G`
    // (`n_tr × n_tr`) and `M` (`n_te × n_tr`); the columns of a split map
    // in parallel. Nothing the size of X is written. Every split has the
    // same training size (the split sizes depend on `n` and `k` only), so
    // the first split's stands for all.
    let n_tr = splits.first().map_or(0, |sp| sp.tr.len());
    let gram_unit = crate::signal_test::split_blocks_bytes(splits, n_tr);
    // ±0.9999 pre-atanh clamp mirrored from `signal_test::nb_test` (change
    // together): keeps the statistic identical to the primal route's and z̄
    // finite at |r| = 1.
    crate::signal_test::zbars_over_splits(splits, cols.len(), gram_unit, |sp| {
        pls3_split_columns_r(all, cols, sp, y_scale, keep_y, max_iter, tol)
            .into_iter()
            .map(|r| r.clamp(-0.9999, 0.9999).atanh())
            .collect()
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
        // A non-finite held-out Y block gives the primal's value, 0
        // (`pls3_signal_test::pls3_split_columns_r_primal`; change
        // together).
        if crate::fit::check_finite_mat(ys_te.as_ref()).is_err() {
            return 0.0;
        }

        // C = Ỹ_tr' G Ỹ_tr  (q × q), formed as Ỹ'(GỸ) so the
        // n_tr × n_tr Gram is applied, never squared. Seq inside the
        // per-column worker: outer Rayon owns the pool.
        let seq = faer::Par::Seq;
        let gy = crate::linalg::mat_mul(g.as_ref(), ys_tr.as_ref(), seq);
        let c = crate::linalg::mat_mul(ys_tr.transpose(), gy.as_ref(), seq);
        if !c.is_all_finite() {
            return 0.0;
        }

        // Lower triangle pinned for byte-parity stability, matching
        // `signal_test::eigenvalues_symmetric`. faer returns
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
        let seeds: Vec<u64> = (0..5).collect();
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seeds: &seeds,
            null_map: None,
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
                    )
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
            crate::pls3_signal_test::dual_route_eligible(None, n, p, q, n_perm, 6, false),
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

    // Worked cases spanning the shapes the rule has to separate.
    // `n_replicates` is the third argument throughout — the runners pass
    // `B + 1`, so 1000 here stands for B = 1000, a difference far inside the
    // rule's resolution. These are the rule's contract;
    // if a future edit moves any of them the rule changed meaning, not just
    // its constants.

    #[test]
    fn wide_shapes_take_the_dual_route() {
        // PLS3 at fMRI scale: n_tr = 50, p = 300_000, q = 10 ⇒ est. ~200× speedup.
        assert!(use_dual_route(50, 300_000, 1000, 10));
        // PLS1, small n and wide p: n_tr = 80, p = 300_000, q = 1 ⇒ est. ~12× speedup.
        assert!(use_dual_route(80, 300_000, 1000, 1));
    }

    #[test]
    fn memory_cap_overrides_a_favourable_flop_count() {
        // n_tr = 4001, p = 1e6, B·q = 1e5:
        //   4001 · (1e5 + 1e6) = 4.40e9  <  1e6 · 1e5 = 1e11
        // so the flop rule alone would take the dual route. The cap is what
        // stops it. The premise is asserted so a future edit to the rule
        // cannot quietly make this test vacuous.
        let n_tr = DUAL_ROUTE_MAX_N_TR + 1;
        let p = 1_000_000_usize;
        let n_rep = 100_000_usize;
        let flop_rule_says_yes = (n_tr as f64) * ((n_rep + p) as f64) < (p as f64) * (n_rep as f64);
        assert!(
            flop_rule_says_yes,
            "test premise: flop rule must favour dual here"
        );
        assert!(!use_dual_route(n_tr, p, n_rep, 1));
        // One row under the cap, the same input routes dual.
        assert!(use_dual_route(DUAL_ROUTE_MAX_N_TR, p, n_rep, 1));
    }

    #[test]
    fn too_few_replicates_stays_primal() {
        // `n_tr ≥ n_replicates·q` means the precompute cannot be amortized,
        // and it sinks the route on its own even when p is enormous:
        //   1000 · (2 + 1e7) = 1.0e10  >  1e7 · 2 = 2e7.
        // n_tr is well under the memory cap, so the cap is not what decides.
        assert!(!use_dual_route(1000, 10_000_000, 2, 1));
    }

    #[test]
    fn gram_bigger_than_the_data_stays_primal() {
        // `n_tr ≥ p` means the Gram is bigger than the matrix it came from,
        // and it sinks the route on its own even with a huge replicate count.
        assert!(!use_dual_route(500, 100, 1_000_000, 1));
        // n_tr = 400, p = 200, q = 1: est. 0.5x (slower).
        assert!(!use_dual_route(400, 200, 1000, 1));
    }

    #[test]
    fn harmonic_form_is_stricter_than_the_min() {
        // At p == B·q the harmonic combination is half the min, so the band
        // between the two must route primal — rounding the rule up to
        // min(p, B·q) would admit inputs that are slower, not faster.
        let (p, bq) = (1000_usize, 1000_usize);
        let half = p / 2;
        assert!(!use_dual_route(half + 10, p, bq, 1));
        assert!(use_dual_route(half - 10, p, bq, 1));
    }

    #[test]
    fn degenerate_inputs_stay_primal() {
        assert!(!use_dual_route(0, 0, 0, 0));
        assert!(!use_dual_route(10, 0, 100, 1));
        assert!(!use_dual_route(10, 100, 0, 1));
        assert!(!use_dual_route(10, 100, 100, 0));
        // No training rows: the flop rule alone would accept it
        // (0 < 100·100), so the guard is what refuses.
        assert!(!use_dual_route(0, 100, 100, 1));
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
        let (_, mut rng) = crate::rng::resolve_seed(Some(23)).expect("seed");
        let splits = crate::signal_test::draw_splits(n, 1, 6, &mut rng).expect("splits");
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seeds: &seeds,
            null_map: None,
        };
        assert!(
            use_dual_route(splits[0].tr.len(), p, cols.len(), q),
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
            );
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
            );
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
        let (_, mut rng) = crate::rng::resolve_seed(Some(31)).expect("seed");
        let splits = crate::signal_test::draw_splits(n, 1, 6, &mut rng).expect("splits");
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let observed = |_: usize, mut m: faer::MatMut<'_, f64>| m.copy_from(y.as_ref());
        let cols = crate::resample::Columns {
            y: y.as_ref(),
            seeds: &seeds,
            null_map: Some(&observed),
        };
        let fit = crate::pls3::Pls3FitOpts::default();
        let g = crate::held_out_gram::HeldOutGram::from_x(x.as_ref(), None, xs.as_ref());
        let z =
            pls3_split_zbars_columns(&g, &cols, &splits, ys.as_ref(), None, fit.max_iter, fit.tol);
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
            crate::pls3_signal_test::dual_route_eligible(None, n, p, q, 99, 4, true),
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
