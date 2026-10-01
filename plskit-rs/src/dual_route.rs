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
//! **The route changes no number.** Both routes must agree to the crate's
//! bit-near tolerance on every replicate column, not merely on the final
//! p-value;
//! the equivalence tests in `signal_test.rs` and `pls3_signal_test.rs` are
//! what makes a second route safe to maintain.

// The n-space PLS1 route for k ≥ 1 lives in `dual_route/multi_k.rs`; its
// items are re-exported below, so their paths are `crate::dual_route::…`.
mod multi_k;
pub(crate) use multi_k::{
    fold_column_nspace, nspace_eligible_perm_null, nspace_eligible_raw_perm,
    nspace_eligible_split_exact, nspace_perm_row, split_columns_r_nspace, NspaceFold, NspaceGram,
    NspaceSplit, NSPACE_BATCH,
};
#[cfg(test)]
pub(crate) use multi_k::{nspace_blocks_built, K_DUAL_MAX};

/// Hard cap on the training-set size the dual route will accept.
///
/// `G` is `n_tr²` f64s, 128 MB at `n_tr = 4000`. Folds (`pls1_cv_r2_columns`)
/// and splits (`pls3_split_zbars_columns`) run one at a time, so one `G` is
/// alive. The flop rule below already excludes that regime,
/// so this should never bind; it bounds `G` specifically, so getting the
/// flop rule wrong should make `G` degrade to "slower", never to an
/// out-of-memory kill. It does not bound the `B × n` permutation and
/// outcome buffers the dual branches also build per split (`perms` in
/// `pls3_signal_test.rs`, plus `run_raw_perm`'s `y_mat` in
/// `signal_test.rs`) — those scale with `B` unchecked.
pub(crate) const DUAL_ROUTE_MAX_N_TR: usize = 4000;

/// Should this loop take the dual route?
///
/// `n_tr` is the training-subset size, `p` the feature count,
/// `n_replicates` the number of outcome columns amortizing one precompute
/// (`B + 1`: the observed column plus `B` nulls; `B` at `pls1_perm_null`,
/// whose observed fit stays primal), and `q` the per-replicate
/// multiplicity: how many `n_tr × p` passes a primal replicate makes, and
/// how many `n_tr × n_tr` products a Gram replicate makes. That is the
/// outcome column count on the PLS3 route, `1` on the PLS1 `K = 1` routes,
/// and the component count `k` on the PLS1 n-space route (`multi_k`), whose
/// `G` precompute does not grow with `k`. On that route the rule is
/// conservative: a primal PLS1 replicate makes about `2k + 1` passes over
/// `X̃`, not `k`.
///
/// # The rule
/// Flop counts for one fold/split are `B·n_tr·p·q` primal versus
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
/// Both halves have to hold and they fail for different reasons —
/// `n_tr ≥ p` means the Gram is bigger than the data it came from, and
/// `n_tr ≥ B·q` means there are not enough replicates to amortize the
/// precompute.
///
/// These are flop counts, not measurements. The primal inner operation is
/// memory-bandwidth bound at small `q` while the Gram is a high-intensity
/// BLAS-3 call, so the measured crossover sits somewhere other than this
/// one — which is why the rule is used as a conservative guard and not as a
/// tuned threshold.
#[allow(clippy::many_single_char_names)]
pub(crate) fn use_dual_route(n_tr: usize, p: usize, n_replicates: usize, q: usize) -> bool {
    if n_tr == 0 || p == 0 || n_replicates == 0 || q == 0 {
        return false;
    }
    if n_tr > DUAL_ROUTE_MAX_N_TR {
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

use faer::{Col, Mat, MatRef};

/// Multiple of a Gram quantity's rounding bound (and of the PLS1 relative
/// floor) that `pls1_cv_r2_columns`, and per component
/// `multi_k::pls1_nspace_kernel`, require before trusting their own
/// truncation decision. Past it, the dual `‖X̃'z‖²` and `t't` are within
/// a factor `1 ± 1/RESOLVE_BAND` of the true values, which puts the primal
/// route's computed norm clear of its floor. The `K = 1` bound it
/// multiplies is a worst case; typical rounding sits orders of magnitude
/// below it. Also the decision band of the p-space Gram backend (`gram_p`)
/// and of `split_exact`'s Gram-p score gate.
pub(crate) const RESOLVE_BAND: f64 = 4.0;

/// Multiple of the PLS1 kernel's `1e-14` absolute exits that
/// `pls1_cv_r2_columns` (and, per component, `multi_k::pls1_nspace_kernel`)
/// requires of its `√(z'Gz)` and `t't` before trusting its own decision.
pub(crate) const ABS_BAND: f64 = 100.0;

/// Relative floor on the centered norm of a Gram route's test-half scores
/// `s = M·alpha` at `split_exact`: a column keeps them only when
/// `‖s_c‖ ≥ SCORE_BAND · ‖s‖` and `s` is not constant to rounding;
/// otherwise the primal refit decides that column. Shared by every PLS1
/// Gram route that forms test-half scores.
///
/// `guarded_pearson`'s `constant_to_rounding` test is a second
/// discontinuity after truncation: it returns `r = 0` for scores whose
/// centered norm is at most `2·n_te·ε` of their norm. At `k ≥ 2` a Gram
/// route's scores and the primal's are not exact positive multiples of
/// each other; they differ by both routes' rounding and by the kernel
/// discrepancy. With `η` the relative discrepancy
/// `‖ŝ − s_primal‖ / ‖s_primal‖`, the correlation moves by at most about
/// `η·‖s‖/‖s_c‖ ≤ η / SCORE_BAND` past the gate, so
/// `SCORE_BAND` must be at least `10·η_max / 1e-10`, rounded up to a power
/// of ten, to keep `r` inside the `1e-10` tolerance with a factor 10 to
/// spare. `η_max` is measured, not derived, by
/// `multi_k::sweep::score_band_covers_the_measured_score_discrepancy`. It
/// sits far above `2·n_te·ε`, so a score vector the primal would call
/// constant never passes. Measured over the validation sweep:
/// `η_max = 5.38e-15` (so `10·η_max/1e-10 = 5.38e-4`, needing at least
/// `1e-3`), hence `1e-3`.
///
/// The band is measured for the `k` the n-space route serves on the
/// `split_exact` score path, `2..=multi_k::K_DUAL_MAX`. The largest
/// discrepancy the sweep saw beyond that range (`k = 3`, the 20-row
/// training half of a design whose outcome is orthogonal to the full
/// `X̃`, `η = 2.29e-13`) means that raising `K_DUAL_MAX` requires
/// re-measuring this band. `SCORE_BAND` is the max of this n-space
/// requirement and any other Gram route's (such as `gram_p`'s) own
/// requirement, up to the `1e-2` ceiling asserted alongside it.
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
/// The NIPALS early exits translate to the same quantities:
/// `‖X̃'z‖ = √(z'Gz)` and `t't = (z'G²z)/(z'Gz)`. Only in exact arithmetic,
/// though, and that is not enough to mirror them. See "Truncation" below.
///
/// # Truncation
/// `fit::pls1_kernel` drops the component, and the fold then predicts zero, when
/// `‖X̃'z‖` falls under `fit::w_rel_floor(n_tr, p, ‖X̃‖_F, ‖z‖)`
/// (`max(n_tr, p)·ε·‖X̃‖_F·‖z‖`) or under `1e-14`, or when `t't < 1e-14`.
/// The floor itself is evaluated here on bit-identical inputs (`X̃_tr` and `z`
/// come out of the same `standardize` / `standardize1` calls as on the primal
/// route). The quantity compared against it is not: `z'Gz` is formed from a
/// rounded `G`, and its absolute error is bounded by
/// `(p + 2·n_tr)·ε·‖z‖²·‖X̃‖_F²`, so `√(z'Gz)` resolves `‖X̃'z‖` only down to
/// about `√((p + 2·n_tr)·ε)·‖X̃‖_F·‖z‖`, far above the floor. A `z` orthogonal
/// to the columns of `X̃_tr` makes `z'Gz` pure rounding: it can come out
/// negative (a zero prediction here) or positive, and then `c` divides one
/// rounding residue by another and the prediction is of order one where the
/// primal route predicts zero. No threshold on the dual quantities can decide
/// such a column the way the primal route does.
///
/// So this route decides only the columns it can resolve and hands the rest
/// to the primal kernel. A column keeps the Gram formula only when `z'Gz` and
/// `z'G²z` each clear `RESOLVE_BAND` times their worst-case rounding bound
/// (`(p + 2·n_tr)·ε·‖z‖²·‖X̃‖_F²` and `(2p + 3·n_tr)·ε·‖z‖²·‖X̃‖_F⁴`), `√(z'Gz)`
/// clears `RESOLVE_BAND` times the relative floor, and `√(z'Gz)` and `t't`
/// clear `ABS_BAND` times the `1e-14` exits. Past those gates the true
/// `‖X̃'z‖` and `t't` lie within a factor 2 of the dual values, so the
/// primal route keeps the component too. Every other column (inside a band, or a
/// NaN) is recomputed for this fold by `fit::pls1_fit_prepared_fro` on the same
/// `X̃_tr`, `z` and `‖X̃_tr‖_F`, exactly as `signal_test::cv_fold_contribution`
/// does, so its contribution is the primal route's to the bit, truncation
/// decision included. The one exception is `z` exactly zero (a constant
/// training `y`): `X̃'z` is then an exact zero on both routes, and the zero
/// prediction is written directly.
///
/// The gates cost nothing on ordinary data: when `p ≥ n_tr`,
/// `z'Gz / (‖z‖²·‖X̃‖_F²)` is of order `1/n_tr` and
/// `z'G²z / (‖z‖²·‖X̃‖_F⁴)` of order `1/n_tr²`, against bands of order
/// `(p + n_tr)·ε` (about `3e-10` at `p = 3e5`). They fire on columns whose
/// training `y` is orthogonal, or nearly so, to `X̃_tr` or to its leading
/// directions (constructed or residualized outcomes, duplicated rows, a
/// nearly singular `X̃_tr`), and cost one primal fold fit, `O(n_tr·p)`,
/// for each such column.
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
/// decision.
///
/// # Shapes
/// - `x`: `(n, p)` raw (unstandardized) predictors
/// - `y_mat`: `(n, n_cols)` raw outcomes; column 0 is the observed `y`
/// - returns: `(n_cols,)` pooled CV R², one per column
///
/// # Panics
/// Never (all indexing is over caller-supplied fold index vectors).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
pub(crate) fn pls1_cv_r2_columns(
    x: MatRef<'_, f64>,
    y_mat: MatRef<'_, f64>,
    folds: &[Vec<usize>],
    disable_parallelism: bool,
) -> Vec<f64> {
    use crate::linalg::{standardize1, standardize_apply_rows, standardize_rows};

    let n_cols = y_mat.ncols();
    let mut ss_res = vec![0.0_f64; n_cols];
    let mut ss_tot = vec![0.0_f64; n_cols];

    for (fi, val_idx) in folds.iter().enumerate() {
        let train_idx: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != fi)
            .flat_map(|(_, f)| f.iter().copied())
            .collect();

        // Training moments on the training fold only, applied to the
        // validation fold: mirrors signal_test::pooled_cv_r2_columns
        // (change together).
        let (xs_tr, x_mean, x_scale) = standardize_rows(x, &train_idx, None, None);
        let xs_val = standardize_apply_rows(x, val_idx, x_mean.as_ref(), x_scale.as_ref(), None);

        // Built once per fold and reused by every column: this is the whole
        // saving. G is n_tr × n_tr, M is n_val × n_tr; neither carries p.
        let bpar = crate::resample::block_par(disable_parallelism);
        let g = crate::linalg::mat_mul(xs_tr.as_ref(), xs_tr.transpose(), bpar);
        let m = crate::linalg::mat_mul(xs_val.as_ref(), xs_tr.transpose(), bpar);

        let n_tr = train_idx.len();
        let n_val = val_idx.len();
        let p = x.ncols();

        // Once per fold: the inputs of the truncation gates (see
        // "Truncation" above). `x_fro` is the same `‖X̃_tr‖_F` that
        // `fit::pls1_kernel` computes for its floor, from the same matrix.
        let x_fro = xs_tr.norm_l2();
        let fro2 = x_fro * x_fro;
        #[allow(clippy::cast_precision_loss)]
        let (num_err, den_err) = (
            ((p + 2 * n_tr) as f64) * f64::EPSILON * fro2,
            ((2 * p + 3 * n_tr) as f64) * f64::EPSILON * fro2 * fro2,
        );

        let per_col = |col: usize| -> (f64, f64) {
            let y_tr = Col::<f64>::from_fn(n_tr, |i| y_mat[(train_idx[i], col)]);
            let (z, y_mean, y_scale) = standardize1(y_tr.as_ref());
            let z_norm = z.norm_l2();

            // Seq inside the per-column worker: outer Rayon owns the pool.
            let seq = faer::Par::Seq;
            let g1 = crate::linalg::mat_vec(g.as_ref(), z.as_ref(), seq); // G z
            let g2 = crate::linalg::mat_vec(g.as_ref(), g1.as_ref(), seq); // G² z, as G(Gz)
            let num: f64 = (0..n_tr).map(|i| z[i] * g1[i]).sum(); // z'Gz
            let den: f64 = (0..n_tr).map(|i| z[i] * g2[i]).sum(); // z'G²z

            // Mirrors fit::pls1_kernel's early exits (change together): a
            // truncated K=1 fit has coef = 0, hence a zero prediction. The
            // gates below keep the Gram formula only where the primal route
            // provably keeps the component; everything else is decided by
            // the primal kernel itself.
            let zz = z_norm * z_norm;
            let w_norm = num.max(0.0).sqrt();
            let tt = if num > 0.0 { den / num } else { 0.0 };
            let w_floor = crate::fit::w_rel_floor(n_tr, p, x_fro, z_norm);
            // Written as `a >= b` conjunctions so a NaN anywhere fails the
            // gate and takes the primal fallback.
            let resolved = num >= RESOLVE_BAND * num_err * zz
                && den >= RESOLVE_BAND * den_err * zz
                && w_norm >= RESOLVE_BAND * w_floor
                && w_norm >= ABS_BAND * crate::fit::NIPALS_ABS_FLOOR
                && tt >= ABS_BAND * crate::fit::NIPALS_ABS_FLOOR;
            let y_pred: Col<f64> = if z_norm == 0.0 {
                // X̃'z is an exact zero on both routes.
                Col::<f64>::zeros(n_val)
            } else if resolved {
                let c = num / den;
                let mz = crate::linalg::mat_vec(m.as_ref(), z.as_ref(), seq);
                Col::<f64>::from_fn(n_val, |i| c * mz[i])
            } else {
                // The primal fold fit, bit for bit as in
                // `signal_test::cv_fold_contribution` (change together).
                match crate::fit::pls1_fit_prepared_fro(
                    xs_tr.as_ref(),
                    z.as_ref(),
                    1,
                    None,
                    crate::fit::ParChoice::Seq,
                    x_fro,
                ) {
                    Ok(fit) => crate::linalg::mat_vec(xs_val.as_ref(), fit.coef.as_ref(), seq),
                    // Unreachable: the prepared kernel has no error path.
                    // NaN rather than a panic, which `run_raw_perm` counts
                    // as an exceedance.
                    Err(_) => Col::<f64>::from_fn(n_val, |_| f64::NAN),
                }
            };

            let ys_val =
                Col::<f64>::from_fn(n_val, |i| (y_mat[(val_idx[i], col)] - y_mean) / y_scale);
            #[allow(clippy::cast_precision_loss)]
            let mean_val: f64 = (0..n_val).map(|i| ys_val[i]).sum::<f64>() / n_val as f64;
            let res: f64 = (0..n_val).map(|i| (y_pred[i] - ys_val[i]).powi(2)).sum();
            let tot: f64 = (0..n_val).map(|i| (ys_val[i] - mean_val).powi(2)).sum();
            (res, tot)
        };

        // `collect` preserves column order in both arms, so serial and
        // parallel results are byte-equal (same shape as
        // split_perm_nr_zbars' per-split dispatch).
        let contrib: Vec<(f64, f64)> = if disable_parallelism {
            (0..n_cols).map(per_col).collect()
        } else {
            use rayon::prelude::*;
            (0..n_cols).into_par_iter().map(per_col).collect()
        };
        for (col, (res, tot)) in contrib.into_iter().enumerate() {
            ss_res[col] += res;
            ss_tot[col] += tot;
        }
    }

    // Pooled ratio, matching signal_test::pooled_cv_r2_columns's final
    // expression exactly.
    (0..n_cols)
        .map(|col| {
            if ss_tot[col] > 0.0 {
                1.0 - ss_res[col] / ss_tot[col]
            } else {
                0.0
            }
        })
        .collect()
}

/// Per-column z̄ for the PLS3 `split_exact` statistic through the Gram
/// route. Returns `perms.len() + 1` values; column 0 is the observed Y and
/// columns `1..` are the permutation nulls, in `perms` order.
///
/// # The reduction this function exploits
/// PLS3's fit is a singular decomposition, so unlike PLS1 at `K = 1` there
/// is no "no fit" shortcut — permuting Y genuinely moves `u₁`. What there
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
/// vectors at once, so `r` is untouched — the same joint-flip argument that
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
/// ‖Ỹ_tr‖_F)` (`max(n_tr, p, q)·ε·‖X̃_tr‖_F·‖Ỹ_tr‖_F`); otherwise it keeps
/// nothing and `r = 0`. The floor itself is evaluated here on bit-identical
/// inputs (`X̃_tr` and `Ỹ_tr` come out of the same `standardize` calls as on
/// the primal route). The quantity compared against it is not: `λ_max` is
/// an eigenvalue of a rounded `C`, and bounding the rounding of each entry
/// `ỹ_j'Gỹ_k` as `pls1_cv_r2_columns` bounds `z'Gz`, plus the symmetric
/// eigensolver's backward error, puts it within
/// `λ_err = (p + 2·n_tr + q)·ε·‖X̃_tr‖_F²·‖Ỹ_tr‖_F²` of `σ₁²`. So `√λ_max`
/// resolves `σ₁` only down to about `√((p + 2·n_tr + q)·ε)·‖X̃_tr‖_F·‖Ỹ_tr‖_F`,
/// far above the floor, and a `Ỹ_tr` orthogonal to `X̃_tr` up to rounding
/// (a residualized or constructed outcome, possible at `p ≫ n_tr` only when
/// X is rank-deficient) makes `λ_max` pure rounding whose eigenvector the
/// route would otherwise correlate.
///
/// A column therefore keeps the Gram formula only when `λ_max` clears
/// `RESOLVE_BAND·λ_err`, `√λ_max` clears `FLOOR_BAND` times the relative
/// floor and `ABS_BAND` times `SIGMA_FLOOR`. Past the first gate the true
/// `σ₁` is at least `√(3/4)·√λ_max`, so past the second it is at least
/// `6.9×` the floor; the primal's computed `σ₁` departs from the true one by
/// at most the rounding of `A` (`n_tr·ε·‖X̃_tr‖_F·‖Ỹ_tr‖_F`) plus its SVD's
/// backward error (a small multiple of `max(p, q)·ε·σ₁`), about twice the
/// floor, so it keeps LV1 too. `FLOOR_BAND` is 8 rather than
/// `RESOLVE_BAND` for that second term, which `pls1_cv_r2_columns` has no
/// counterpart of. Every other column is recomputed by
/// `pls3_signal_test::pls3_split_column_r_primal` on the same blocks, so its
/// value is the primal route's to the bit, truncation included; that also
/// covers a degenerate half (`λ_max = 0`, a constant X or Y block), where
/// the primal keeps nothing either.
///
/// On the sparse Y side the primal truncates on the sparse `σ = u'Av`
/// instead, with `u = Av_{T−1}/‖Av_{T−1}‖` from the last sweep, which is
/// `v_T'C v_{T−1} / √(v_{T−1}'C v_{T−1})`. Both forms are computed here and
/// gated the same way (each against `RESOLVE_BAND·λ_err`, their ratio against
/// the two floors) after the alternation, before the scores.
///
/// The gates cost nothing on ordinary data: `σ₁ / (‖X̃_tr‖_F·‖Ỹ_tr‖_F)` is
/// of order `1/√(n_tr·min(p, q))` or larger, against bands of order
/// `√((p + n_tr)·ε)` and `max(n_tr, p, q)·ε`. They fire on columns whose
/// training `Ỹ` is orthogonal, or nearly so, to `X̃_tr`, and cost one
/// primal fit of that column.
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
/// direction is not identified — any rotation inside the leading subspace
/// is an equally valid answer — so the eigendecomposition of `C` here and
/// the SVD of `A` in the primal route can legitimately return different
/// `v₁`, and `r` moves with them by far more than 1e-10. Exact ties are
/// measure-zero on real data, but a near-tie is common at small `q` and is
/// the likeliest cause of an equivalence-test failure that is *not* a bug
/// in either route.
///
/// # What is NOT hoisted
/// Y's column moments are recomputed per split *per replicate*: the primal
/// route standardizes each training half with that half's own moments, and
/// a permutation changes which rows land in the half. `O(n_tr q)` each.
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
/// expected to agree. Sharing the criterion is necessary but not
/// sufficient; see "Near-ties" below.
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
/// `pls3::spls3_fit` drops a component whose `σ = u'Av` falls under
/// either floor, which on this call means `k_used == 0` and `r = 0` on the
/// primal route. The truncation gate above covers that `σ` (see
/// "Truncation"), and the selected `v`'s norm is guarded against
/// `V_NORM_FLOOR` on every sweep, as `spls3_component` guards it.
///
/// # Near-ties (sparse Y side)
/// "The same map" holds in exact arithmetic only. The two routes evaluate it
/// on different floats (`C v` with `C` formed in full here, `A'u` there,
/// from an eigenvector here and an SVD vector there), and the map has two
/// discontinuities: the support `hard_select_keep` picks flips where the
/// `keep_y`-th and `(keep_y + 1)`-th largest `|·|` cross, and the sweep count
/// jumps where some sweep's `dv` crosses `tol`. Within a few ulps of either,
/// the routes can take different branches (a different support, or a stop
/// one sweep apart) and their `r` separates by up to `O(1)` or `O(tol)`.
/// No tie-break rule evaluated separately on each route's own floats can
/// close that: any rule is a discontinuous function of route-dependent
/// inputs and only moves the boundary.
///
/// So this route guards itself instead. On every sweep it measures how
/// close it came to each discontinuity: the relative gap between the two
/// entries straddling the `keep_y` boundary (`SEL_BAND`), and `|dv − tol|`
/// (`STOP_BAND_REL`, `STOP_BAND_ABS`); and once, up front, the relative
/// eigengap `(λ_q − λ_{q−1}) / λ_q` that sets how far the two initializers
/// can sit apart (`GAP_BAND`). If any is inside its band, the column is
/// recomputed by `pls3_signal_test::pls3_split_column_r_primal`, the exact
/// primal computation, on the same standardized blocks, so that column's
/// value is the primal route's to the bit. Outside the bands every branch
/// the primal takes is the one taken here, because the bands sit orders of
/// magnitude above the measured route discrepancy (`‖v_primal − v_dual‖_∞`
/// ≤ 9e-16 per sweep, dv discrepancy ≤ 2e-15, on the equivalence fixture
/// and on constructed near-ties). The gate's own threshold is harmless: on
/// either side of it the answer agrees with the primal.
///
/// One discrepancy grows without ever crossing a branch, so none of those
/// bands sees it: on a nearly degenerate *restricted* spectrum (the top two
/// eigenvalues of `C` on the selected support within a relative `g`) the
/// alternation damps each sweep's rounding difference only at rate `g`,
/// and `|Δz|` accumulates as about `ε·min(sweeps, 1/g)`: 3e-14 at the
/// default `max_iter = 100`, past `1e-12` beyond about 4000 sweeps
/// (`dual_route_matches_primal_on_a_near_degenerate_restricted_spectrum`
/// in `pls3_signal_test.rs`). A fourth band, `DRIFT_SWEEPS`, bounds it:
/// after a run of more than that many sweeps, the gap of `C` on the final
/// support is measured, and if `min(sweeps, 1/g)` exceeds the band the
/// column falls back to the primal.
///
/// The bands cost nothing when not hit (a partial selection over `q` values per sweep) and
/// the fallback fires on a fraction of columns of order
/// `2·stop_band/(tol·ln(1/rate))`, with `rate` the alternation's per-sweep
/// contraction, about 1e-5 at the default `tol` (the eigengap band adds
/// under 5e-6 on null columns). The
/// dense arm is not gated, so not one of its floats moves.
///
/// A sparse X side never reaches here: `keep_x < n_features` zeroes
/// coordinates of `u`, which is not a positive rescale, the proportionality
/// above fails, and `pls3_confirmatory_test` refuses the dual route for
/// that configuration.
///
/// # Preconditions
/// `q >= 1`. Not checked here: `use_dual_route`'s `q == 0` early return is
/// what guarantees it before this function is ever called. `max_iter >= 1`
/// whenever `keep_y` is `Some`, which `pls3_confirmatory_test` validates
/// before either route runs.
///
/// # Shapes
/// - `x`: `(n, p)` raw predictors
/// - `y`: `(n, q)` raw outcomes
/// - returns: `(perms.len() + 1,)`
///
/// # Panics
/// Never (indexing is over caller-supplied split and permutation vectors).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::items_after_statements)]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(crate) fn pls3_split_zbars_columns(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[crate::signal_test::SplitIdx],
    perms: &[Vec<usize>],
    keep_y: Option<usize>,
    max_iter: usize,
    tol: f64,
    disable_parallelism: bool,
) -> Vec<f64> {
    use crate::linalg::{standardize, standardize_apply, standardize_apply_rows, standardize_rows};

    let (p, q) = (x.ncols(), y.ncols());
    let n_cols = perms.len() + 1;

    /// Multiple of the relative floor (`pls3::sigma_rel_floor`) that the
    /// Gram route's lower bound on `σ₁` (and, on the sparse Y side, on the
    /// sparse `σ`) must clear before it trusts that the primal fit keeps
    /// LV1. See "Truncation" in the doc comment for why 8 and not
    /// `RESOLVE_BAND`.
    const FLOOR_BAND: f64 = 8.0;

    /// The floor `spls3_component` applies to the selected `v` before
    /// normalizing it. Named directly from `pls3` rather than re-declared, so
    /// the two routes cannot drift apart. The two routes select on vectors
    /// that differ by a positive scalar (`Cv` here, `A'u = Cv/‖Av‖` there),
    /// so this is a defensive degenerate-input guard, not a boundary the two
    /// routes can be expected to cross on the same input; reaching it needs a
    /// selected `v` that is numerical dust, where both routes already report
    /// no association.
    use crate::pls3::SIGMA_FLOOR as V_NORM_FLOOR;

    /// Near-tie guard bands (see "Near-ties" in the doc comment). Each sits
    /// far above the measured primal-vs-dual discrepancy of the quantity it
    /// guards; inside a band the column falls back to the primal refit.
    /// Relative eigengap of `C` below which the SVD and eigen initializers
    /// may differ by more than the other bands absorb (`≈ eps / gap`).
    const GAP_BAND: f64 = 1e-4;
    /// Relative gap `(|·|_(ky) − |·|_(ky+1)) / |·|_(1)` at the keep boundary.
    const SEL_BAND: f64 = 1e-10;
    /// `|dv − tol|` band: relative to `tol`, floored in absolute terms so a
    /// tiny user `tol` cannot shrink it under the discrepancy itself.
    const STOP_BAND_REL: f64 = 1e-5;
    const STOP_BAND_ABS: f64 = 1e-13;
    /// Sweep budget for the rounding drift no branch test sees. On the
    /// selected support `S` the alternation is power iteration on `C_SS`,
    /// which damps each sweep's route discrepancy (`C v` here, `A'(Av)`
    /// there) only at the relative gap `g` of `C_SS`'s top two
    /// eigenvalues, so `|Δz|` accumulates as about
    /// `ε·min(sweeps, 1/g)`: measured at 2.5e-16 per effective sweep, flat
    /// in `g` (`dual_route_matches_primal_on_a_near_degenerate_restricted_spectrum`).
    /// Past this many effective sweeps the column falls back to the primal;
    /// 400 keeps the drift near 1e-13, a decade inside the corpus `1e-12`.
    /// The default `max_iter = 100` can never reach it, so the default path
    /// never even evaluates the gate.
    const DRIFT_SWEEPS: usize = 400;

    let per_split = |sp: &crate::signal_test::SplitIdx| -> Vec<f64> {
        let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
        let (xs_tr, x_mean, x_scale) = standardize_rows(x, tr, None, None);
        let xs_te = standardize_apply_rows(x, te, x_mean.as_ref(), x_scale.as_ref(), None);

        // Built once per split; neither carries p into the replicate loop.
        let bpar = crate::resample::block_par(disable_parallelism);
        let g = crate::linalg::mat_mul(xs_tr.as_ref(), xs_tr.transpose(), bpar);
        let m = crate::linalg::mat_mul(xs_te.as_ref(), xs_tr.transpose(), bpar);
        let (n_tr, n_te) = (tr.len(), te.len());
        // Once per split: the X-side inputs of the truncation gate (see
        // "Truncation" above). `x_fro` is the same `‖X̃_tr‖_F` the primal
        // fit computes for its relative floor, from the same matrix.
        let x_fro = xs_tr.norm_l2();
        #[allow(clippy::cast_precision_loss)]
        let lam_err_x = ((p + 2 * n_tr + q) as f64) * f64::EPSILON * x_fro * x_fro;

        crate::resample::map_indexed(n_cols, disable_parallelism, |col| {
            // Column 0 is the identity row map; column c > 0 applies
            // permutation c−1, exactly as the primal route permutes Y's
            // rows as units against X.
            let row_of = |i: usize| if col == 0 { i } else { perms[col - 1][i] };

            let y_tr = Mat::<f64>::from_fn(n_tr, q, |i, j| y[(row_of(tr[i]), j)]);
            let y_te = Mat::<f64>::from_fn(n_te, q, |i, j| y[(row_of(te[i]), j)]);
            let (ys_tr, y_mean, y_scale) = standardize(y_tr.as_ref());
            let ys_te = standardize_apply(y_te.as_ref(), y_mean.as_ref(), y_scale.as_ref());

            // C = Ỹ_tr' G Ỹ_tr  (q × q), formed as Ỹ'(GỸ) so the
            // n_tr × n_tr Gram is applied, never squared. Seq inside the
            // per-column worker: outer Rayon owns the pool.
            let seq = faer::Par::Seq;
            let gy = crate::linalg::mat_mul(g.as_ref(), ys_tr.as_ref(), seq);
            let c = crate::linalg::mat_mul(ys_tr.transpose(), gy.as_ref(), seq);

            // Lower triangle pinned for byte-parity stability, matching
            // `signal_test::eigenvalues_symmetric`. faer returns
            // eigenvalues ascending, so the leading pair is the last.
            let Ok((lambda, eig_u)) = crate::linalg::self_adjoint_eigen(c.as_ref(), seq) else {
                return 0.0;
            };
            let lambda_max = lambda[q - 1];
            // Exact primal recomputation of this column, for the
            // truncation gate and the near-tie guards below.
            // `keep_x = None` is the dense X side this route is
            // restricted to (`Some(p)` would give the same fit).
            let primal_fallback = || {
                crate::pls3_signal_test::pls3_split_column_r_primal(
                    xs_tr.as_ref(),
                    x_fro,
                    xs_te.as_ref(),
                    ys_tr.as_ref(),
                    ys_te.as_ref(),
                    None,
                    keep_y,
                    max_iter,
                    tol,
                )
                .clamp(-0.9999, 0.9999)
                .atanh()
            };
            // Truncation gate (see "Truncation" above). The primal fit
            // keeps LV1 only when `σ₁` clears both floors; this route
            // keeps its own formula only where it can prove that, and
            // hands every other column (a degenerate half, a `Ỹ_tr`
            // orthogonal to `X̃_tr` up to rounding, a NaN) to the primal,
            // whose answer is then the primal's to the bit. `a >= b`
            // conjunctions, so a NaN anywhere fails the gate.
            let y_fro = ys_tr.norm_l2();
            let lam_err = lam_err_x * y_fro * y_fro;
            let floor = crate::pls3::sigma_rel_floor(n_tr, p, q, x_fro, y_fro);
            let resolved = |num: f64, den2: f64| {
                num >= RESOLVE_BAND * lam_err
                    && den2 >= RESOLVE_BAND * lam_err
                    && num >= FLOOR_BAND * floor * den2.sqrt()
                    && num >= ABS_BAND * crate::pls3::SIGMA_FLOOR * den2.sqrt()
            };
            // Dense LV1: `σ₁ = λ_max / √λ_max`.
            if !resolved(lambda_max, lambda_max) {
                return primal_fallback();
            }
            // Sparse Y side: alternate in q-space. `None`, and the
            // dense endpoint `keep_y == q`, are a LITERAL skip of the
            // alternation (`_ => None` below), so the dense route's
            // float sequence is untouched.
            let sparse_v: Option<(Col<f64>, Col<f64>)> = match keep_y {
                Some(ky) if ky < q => {
                    // `ky < q` implies `q >= 2`, so `lambda[q - 2]` exists.
                    if lambda_max - lambda[q - 2] < GAP_BAND * lambda_max {
                        return primal_fallback();
                    }
                    let stop_band = (STOP_BAND_REL * tol).max(STOP_BAND_ABS);
                    // Per-column scratch, reused across sweeps.
                    let mut mags: Vec<f64> = vec![0.0; q];
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
                    let mut sweeps = 0usize;
                    for it in 1..=max_iter {
                        sweeps = it;
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
                        for j in 0..q {
                            mags[j] = v[j].abs();
                        }
                        // The keep boundary needs three order
                        // statistics of `|v|`: the largest, and the
                        // `ky`-th and `(ky+1)`-th largest. A partial
                        // selection puts the `(ky+1)`-th at index `ky`
                        // and the `ky` largest before it, so the other
                        // two are the max and the min of that prefix.
                        // Order statistics are exact values, so this
                        // reads the same numbers a full sort would.
                        mags.select_nth_unstable_by(ky, |a, b| b.total_cmp(a));
                        let kth_next = mags[ky];
                        let (top, kth) = mags[..ky].iter().fold(
                            (f64::NEG_INFINITY, f64::INFINITY),
                            |(hi, lo), &m| {
                                (
                                    if m.total_cmp(&hi).is_gt() { m } else { hi },
                                    if m.total_cmp(&lo).is_lt() { m } else { lo },
                                )
                            },
                        );
                        if kth - kth_next <= SEL_BAND * top {
                            return primal_fallback();
                        }
                        if crate::fit::select_and_normalize(&mut v, ky, V_NORM_FLOOR).is_none() {
                            return 0.0;
                        }
                        let dv = (0..q)
                            .map(|j| (v[j] - v_for_u[j]).abs())
                            .fold(0.0_f64, f64::max);
                        if (dv - tol).abs() <= stop_band {
                            return primal_fallback();
                        }
                        if dv < tol {
                            break;
                        }
                    }
                    // Drift gate (`DRIFT_SWEEPS`). Checked only when the
                    // run was long enough to matter, so short runs pay
                    // nothing and take no new branch.
                    if sweeps > DRIFT_SWEEPS
                        && restricted_sweeps(c.as_ref(), &v) > DRIFT_SWEEPS as f64
                    {
                        return primal_fallback();
                    }
                    // The sparse `σ = u'Av` the primal fit truncates
                    // on, with `u = Av_{T−1}/‖Av_{T−1}‖`:
                    // `σ = v_T'C v_{T−1} / √(v_{T−1}'C v_{T−1})`.
                    let cvp = crate::linalg::mat_vec(c.as_ref(), v_for_u.as_ref(), seq);
                    let num: f64 = (0..q).map(|j| v[j] * cvp[j]).sum();
                    let den2: f64 = (0..q).map(|j| v_for_u[j] * cvp[j]).sum();
                    if !resolved(num, den2) {
                        return primal_fallback();
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

            let r = crate::pls3_signal_test::pearson_r_guarded(&s, &t);
            // ±0.9999 pre-atanh clamp mirrored from
            // `signal_test::nb_test` (change together): keeps the
            // statistic identical to the primal route's and z̄ finite at
            // |r| = 1.
            r.clamp(-0.9999, 0.9999).atanh()
        })
    };

    // Splits run one at a time (`true`): one prepared split (x_tr, x_te,
    // G, M) is alive, and the columns of each split run in parallel.
    crate::signal_test::zbars_over_splits(splits, n_cols, true, per_split)
}

/// `1/g` for the relative gap `g = (λ₁ − λ₂)/λ₁` between the top two
/// eigenvalues of `C` restricted to the support of `v` (its non-zero
/// entries): the number of sweeps it takes the sparse alternation, which on
/// a fixed support is power iteration on that block, to damp a
/// discrepancy. `+∞` when there is no usable gap (a non-positive `λ₁`, a
/// failed decomposition, or `g = 0`), which sends the caller to the primal;
/// `0` for a support of one entry, which has nothing to damp.
// `!(l1 > 0.0)` also catches NaN.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn restricted_sweeps(c: MatRef<'_, f64>, v: &Col<f64>) -> f64 {
    let support: Vec<usize> = (0..v.nrows()).filter(|&j| v[j] != 0.0).collect();
    let k = support.len();
    if k < 2 {
        return 0.0;
    }
    let c_ss = Mat::<f64>::from_fn(k, k, |a, b| c[(support[a], support[b])]);
    // Sequential: this runs inside the caller's per-column worker.
    let Ok((l, _)) = crate::linalg::self_adjoint_eigen(c_ss.as_ref(), faer::Par::Seq) else {
        return f64::INFINITY;
    };
    // Ascending, as in the caller.
    let (l1, l2) = (l[k - 1], l[k - 2]);
    if !(l1 > 0.0) {
        return f64::INFINITY;
    }
    l1 / (l1 - l2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exact comparisons: diagonal blocks, whose eigenvalues and gaps are
    // exact small integers.
    #[allow(clippy::float_cmp)]
    #[test]
    fn restricted_sweeps_reads_the_gap_on_the_support() {
        let c = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { [1.0, 2.0, 5.0][i] } else { 0.0 });
        // Support {0, 1}: eigenvalues 1 and 2, so 1/g = 2 / (2 − 1).
        let v = Col::<f64>::from_fn(3, |j| [0.6, 0.8, 0.0][j]);
        assert_eq!(restricted_sweeps(c.as_ref(), &v), 2.0);
        // One entry has nothing to damp.
        let v = Col::<f64>::from_fn(3, |j| [0.0, 0.0, 1.0][j]);
        assert_eq!(restricted_sweeps(c.as_ref(), &v), 0.0);
        // A tie on the support is no gap at all.
        let c = Mat::<f64>::identity(3, 3);
        let v = Col::<f64>::from_fn(3, |j| [0.6, 0.8, 0.0][j]);
        assert_eq!(restricted_sweeps(c.as_ref(), &v), f64::INFINITY);
    }

    // Worked cases spanning the shapes the rule has to separate.
    // `n_replicates` is the third argument throughout — the runners pass
    // `B + 1`, so 1000 here stands for B = 1000, a difference far inside the
    // rule's resolution. These are the rule's contract;
    // if a future edit moves any of them the rule changed meaning, not just
    // its constants.

    #[test]
    fn pls3_fmri_scale_takes_the_dual_route() {
        // n_tr = 50, p = 300_000, q = 10 ⇒ est. ~200× speedup.
        assert!(use_dual_route(50, 300_000, 1000, 10));
    }

    #[test]
    fn pls1_small_n_wide_p_takes_the_dual_route() {
        // n_tr = 80, p = 300_000, q = 1 ⇒ est. ~12× speedup.
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
    }
}

#[cfg(test)]
mod layout_invariance {
    use super::*;
    use crate::test_support::{assert_bits_eq, copy_free_families, for_each_layout};

    fn perms(n: usize, count: usize, seed: u64) -> Vec<Vec<usize>> {
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        (0..count)
            .map(|_| crate::resample::permute_indices(n, &mut rng))
            .collect()
    }

    /// Both column engines read X only through `standardize_rows` /
    /// `standardize_apply_rows`, which build owned column-major copies, so
    /// every memory layout of X gives the owned matrix's bits.
    #[test]
    fn the_gram_routes_are_bit_identical_across_layouts() {
        let tol = crate::pls3::Pls3FitOpts::default().tol;
        let max_iter = crate::pls3::Pls3FitOpts::default().max_iter;
        for f in copy_free_families().into_iter().filter(|f| f.w.is_none()) {
            let n = f.x.nrows();
            // pls1_cv_r2_columns: column 0 is y, the rest permuted copies.
            let ps = perms(n, 7, 4);
            let y_mat =
                Mat::<f64>::from_fn(n, 8, |i, c| if c == 0 { f.y[i] } else { f.y[ps[c - 1][i]] });
            let folds = crate::linalg::fold_split(&ps[0], 5);
            for dp in [true, false] {
                for_each_layout(
                    &f.x,
                    |_, xv| pls1_cv_r2_columns(xv, y_mat.as_ref(), &folds, dp),
                    |view, got, want| {
                        assert_bits_eq(got, want, &format!("pls1 {} {view} dp={dp}", f.name));
                    },
                );
            }
            // pls3_split_zbars_columns.
            let y = Mat::<f64>::from_fn(n, 4, |i, j| f.x[(i, j)] + 0.5 * f.y[(i + 3 * j) % n]);
            let (_, mut rng) = crate::rng::resolve_seed(Some(6)).unwrap();
            let splits = crate::signal_test::draw_splits(n, 1, 4, true, &mut rng).unwrap();
            let ps = perms(n, 5, 7);
            for keep_y in [None, Some(2)] {
                for dp in [true, false] {
                    for_each_layout(
                        &f.x,
                        |_, xv| {
                            pls3_split_zbars_columns(
                                xv,
                                y.as_ref(),
                                &splits,
                                &ps,
                                keep_y,
                                max_iter,
                                tol,
                                dp,
                            )
                        },
                        |view, got, want| {
                            assert_bits_eq(
                                got,
                                want,
                                &format!("pls3 {} {view} keep_y={keep_y:?} dp={dp}", f.name),
                            );
                        },
                    );
                }
            }
        }
    }
}
