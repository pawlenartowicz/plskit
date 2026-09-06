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
//! nothing in `p`. This is not new in the crate: `split_exact`'s no-refit
//! route (`signal_test::split_perm_nr_zbars`) already computes
//! `X̃_te·X̃_tr'·y_tr` as a fixed linear map and batches all `B+1` columns
//! through it. That route is left exactly as it is; this module generalizes
//! the idea to loops whose argument the batched map does not cover.
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

/// Hard cap on the training-set size the dual route will accept.
///
/// `G` is `n_tr²` f64s — 128 MB at `n_tr = 4000` — and splits run in
/// parallel, so the figure multiplies by the worker count (~2 GB on 16
/// threads at the cap). The flop rule below already excludes that regime,
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
/// (`B + 1`: the observed column plus `B` nulls), and `q` the outcome
/// column count (`1` for the PLS1 family).
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
/// and it too is a dual quantity. `K ≥ 2` breaks all of this — deflation
/// makes component 2's weights depend on component 1's y-dependent scores.
///
/// The two NIPALS early exits translate to the same quantities:
/// `‖X̃'z‖ = √(z'Gz)` and `t't = (z'G²z)/(z'Gz)`, so a truncating fold
/// produces a zero prediction here exactly as it does primally.
///
/// # What is NOT hoisted
/// `y` is standardized with each fold's own training moments and
/// `pls1_cv_r2` pools `ss_res` / `ss_tot` across folds, so the per-fold `y`
/// scale does not cancel out of the pooled ratio. That standardization is
/// therefore recomputed per fold *per column* — `O(n)` each, free next to
/// the Gram products, and wrong if hoisted.
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
    use crate::linalg::{row_subset, standardize, standardize1, standardize_apply};

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

        let x_tr = row_subset(x, &train_idx);
        let x_val = row_subset(x, val_idx);
        // Training moments on the training fold only, applied to the
        // validation fold — mirrors pls1_cv_r2 (change together).
        let (xs_tr, x_mean, x_scale) = standardize(x_tr.as_ref());
        let xs_val = standardize_apply(x_val.as_ref(), x_mean.as_ref(), x_scale.as_ref());

        // Built once per fold and reused by every column: this is the whole
        // saving. G is n_tr × n_tr, M is n_val × n_tr; neither carries p.
        let g: Mat<f64> = xs_tr.as_ref() * xs_tr.transpose();
        let m: Mat<f64> = xs_val.as_ref() * xs_tr.transpose();

        let n_tr = train_idx.len();
        let n_val = val_idx.len();

        let per_col = |col: usize| -> (f64, f64) {
            let y_tr = Col::<f64>::from_fn(n_tr, |i| y_mat[(train_idx[i], col)]);
            let (z, y_mean, y_scale) = standardize1(y_tr.as_ref());

            let g1: Col<f64> = g.as_ref() * z.as_ref(); // G z
            let g2: Col<f64> = g.as_ref() * g1.as_ref(); // G² z, as G(Gz)
            let num: f64 = (0..n_tr).map(|i| z[i] * g1[i]).sum(); // z'Gz
            let den: f64 = (0..n_tr).map(|i| z[i] * g2[i]).sum(); // z'G²z

            // Mirrors nipals_pls1's two early exits (change together): a
            // truncated K=1 fit has coef = 0, hence a zero prediction.
            let w_norm = num.max(0.0).sqrt();
            let tt = if num > 0.0 { den / num } else { 0.0 };
            let y_pred: Col<f64> = if w_norm < 1e-14 || tt < 1e-14 {
                Col::<f64>::zeros(n_val)
            } else {
                let c = num / den;
                let mz: Col<f64> = m.as_ref() * z.as_ref();
                Col::<f64>::from_fn(n_val, |i| c * mz[i])
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

    // Pooled ratio, matching pls1_cv_r2's final expression exactly.
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
/// makes the primal statistic well defined. That invariance covers the
/// correlation only, not `pearson_r_guarded`'s absolute `ss_a < 1e-15`
/// degeneracy guard downstream: for `σ₁` roughly in `[1e-14, 1e-8)`, above
/// `SIGMA_FLOOR`, the un-rescaled `s` can fall under the guard and return
/// `0.0` where the primal route would keep the component; this needs
/// near-exact block orthogonality with both blocks non-degenerate, so it is
/// not thought reachable on real data and the equivalence tests do not
/// cover it.
///
/// So each replicate permutes `Ỹ_tr`, forms a `q × q` matrix, eigendecomposes
/// it, and applies `M`. `G` and `M` are built once per split. Unlike PLS1's
/// batched map this is a *loop of cheap work*, not one operation.
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
/// # Preconditions
/// `q >= 1`. Not checked here: `use_dual_route`'s `q == 0` early return is
/// what guarantees it before this function is ever called.
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
pub(crate) fn pls3_split_zbars_columns(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[crate::signal_test::SplitIdx],
    perms: &[Vec<usize>],
    disable_parallelism: bool,
) -> Vec<f64> {
    use crate::linalg::{row_subset, standardize, standardize_apply};

    let q = y.ncols();
    let n_cols = perms.len() + 1;

    /// Square of `pls3::SIGMA_FLOOR`: the primal route drops a component
    /// when `σ₁ < 1e-14`, and the eigenvalues of `Ỹ'GỸ` are the `σ²`.
    /// Mirrors `pls3.rs` — change together. Squaring rounds, so
    /// `σ² == SIGMA_FLOOR²` and `σ == SIGMA_FLOOR` are not the same
    /// boundary in floating point — the two routes' truncation points can
    /// differ by one ulp exactly at the floor. This guard is defensive and
    /// untested: `G` and `Ỹ` carry standardized scale, so driving
    /// `lambda_max` below the floor in practice requires exact zeros,
    /// where every route already returns zero through its own
    /// degenerate-input convention regardless of this constant.
    const LAMBDA_FLOOR: f64 = 1e-28;

    let per_split = |sp: &crate::signal_test::SplitIdx| -> Vec<f64> {
        let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
        let x_tr = row_subset(x, tr);
        let x_te = row_subset(x, te);
        let (xs_tr, x_mean, x_scale) = standardize(x_tr.as_ref());
        let xs_te = standardize_apply(x_te.as_ref(), x_mean.as_ref(), x_scale.as_ref());

        // Built once per split; neither carries p into the replicate loop.
        let g: Mat<f64> = xs_tr.as_ref() * xs_tr.transpose();
        let m: Mat<f64> = xs_te.as_ref() * xs_tr.transpose();
        let (n_tr, n_te) = (tr.len(), te.len());

        (0..n_cols)
            .map(|col| {
                // Column 0 is the identity row map; column c > 0 applies
                // permutation c−1, exactly as the primal route permutes Y's
                // rows as units against X.
                let row_of = |i: usize| if col == 0 { i } else { perms[col - 1][i] };

                let y_tr = Mat::<f64>::from_fn(n_tr, q, |i, j| y[(row_of(tr[i]), j)]);
                let y_te = Mat::<f64>::from_fn(n_te, q, |i, j| y[(row_of(te[i]), j)]);
                let (ys_tr, y_mean, y_scale) = standardize(y_tr.as_ref());
                let ys_te = standardize_apply(y_te.as_ref(), y_mean.as_ref(), y_scale.as_ref());

                // C = Ỹ_tr' G Ỹ_tr  (q × q), formed as Ỹ'(GỸ) so the
                // n_tr × n_tr Gram is applied, never squared.
                let gy: Mat<f64> = g.as_ref() * ys_tr.as_ref();
                let c: Mat<f64> = ys_tr.transpose() * gy.as_ref();

                // Side::Lower pinned for byte-parity stability, matching
                // `signal_test::eigenvalues_symmetric`. faer returns
                // eigenvalues ascending, so the leading pair is the last.
                let Ok(eig) = c.as_ref().self_adjoint_eigen(faer::Side::Lower) else {
                    return 0.0;
                };
                let lambda = eig.S().column_vector();
                let lambda_max = lambda[q - 1];
                if !(lambda_max >= LAMBDA_FLOOR) {
                    // Degenerate half: no direction exists. The primal route
                    // reports k_used == 0 and r = 0.0 here — match it, and
                    // never emit NaN. (`!(a >= b)` also catches NaN, and the
                    // non-strict comparison matches SIGMA_FLOOR's
                    // keep-at-equality direction in pls3.rs.)
                    return 0.0;
                }
                let v1 = eig.U().col(q - 1);

                let yv: Col<f64> = ys_tr.as_ref() * v1;
                let s: Col<f64> = m.as_ref() * yv.as_ref(); // ∝ X̃_te u₁, positive factor
                let t: Col<f64> = ys_te.as_ref() * v1;

                let r = crate::pls3_signal_test::pearson_r_guarded(&s, &t);
                // ±0.9999 pre-atanh clamp mirrored from
                // `signal_test::nb_test` (change together): keeps the
                // statistic identical to the primal route's and z̄ finite at
                // |r| = 1.
                r.clamp(-0.9999, 0.9999).atanh()
            })
            .collect()
    };

    let per_split_z: Vec<Vec<f64>> = if disable_parallelism {
        splits.iter().map(per_split).collect()
    } else {
        use rayon::prelude::*;
        splits.par_iter().map(per_split).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    // Four worked cases spanning the shapes the rule has to separate.
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
    fn pls1_mid_n_narrow_p_stays_primal() {
        // n_tr = 400, p = 200, q = 1 ⇒ est. 0.5× (slower).
        assert!(!use_dual_route(400, 200, 1000, 1));
    }

    #[test]
    fn embedding_scale_stays_primal() {
        // n_tr = 10_692, p = 400, q = 1 ⇒ est. 0.027× (37× slower), and
        // 914 MB of Gram per worker on top.
        assert!(!use_dual_route(10_692, 400, 1000, 1));
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
