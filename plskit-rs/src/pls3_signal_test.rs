//! Confirmatory PLS3 / PLSSVD omnibus test at LV1. Two methods over one
//! statistic: `split_exact` calibrates the held-out LV correlation by
//! permutation, `split_nb` compares it against the NB t-reference instead.
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
//! `_docs/python/results.md`, "Per-coordinate β CIs").
//!
//! # `split_nb` here versus for PLS1
//! Both sides of the correlation are directions estimated on the training
//! half, where PLS1 has an observed outcome on one side. What carries over
//! is the per-split marginal null: conditional on the training half,
//! `X̃_te u₁` and `Ỹ_te v₁` are fixed linear combinations of test-half rows,
//! the test half is independent of the training half, and under block
//! independence the two projections are independent. So the conditional
//! null of `r` on one split is the ordinary null correlation law. PLS1 is
//! the same construction with `v₁ = 1`. That supports the `rho_hat` ruler
//! and nothing more: the p-value comes from `nb_test`, whose between-split
//! inflation `1/J + n_test/n_train` is PLS1's empirical heuristic, not
//! derived for two blocks. Measured on iid Gaussian, heavy-tailed,
//! low-stable-rank and real two-block designs, `atanh(r)·√(n_test−3)` has
//! unit spread and no excess kurtosis and the resulting test is
//! conservative, never anti-conservative. `split_exact` stays the
//! recommendation because it is exact whenever rows are exchangeable under
//! the null (clustered rows break both methods); `split_nb` is the cheap
//! opt-in.
//!
//! # Why only at k = 1
//! Above k = 1 the training-half component *ordering* need not survive to
//! the test half when singular values are close, and whether the statistic
//! should then be per-component or subspace-level is undecided — so k > 1
//! errors rather than guessing.

use faer::{Col, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::pls3::Pls3FitOpts;
use crate::signal_test::{
    ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestOutput, SPLIT_NB_REROUTE_N_PERM,
    SplitIdx, check_split_exact_counts, draw_splits, nb_rho_hat, nb_test, resolve_auto,
    resolve_split_nb,
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
    /// [`ConfirmatoryArgs::SplitNb`] and [`ConfirmatoryArgs::Auto`] (the
    /// default; resolved on X alone with `n_eff = n`) are accepted; the other
    /// three variants error.
    pub args: ConfirmatoryArgs,
    /// Caller asserts X is already standardized.
    ///
    /// **No effect on either method** — every split standardizes its
    /// training half with that half's own moments by design (into a copy,
    /// or block by block on the routes that write none), so there is
    /// nothing for the flag to skip. The `split_nb` auto-gate likewise
    /// standardizes X itself regardless. It is kept on the struct so the
    /// signature does not change when a method that reads it lands. PLS1's
    /// `split_nb` and `split_exact` do the same: every split standardizes
    /// its training half whatever the caller's `pre_standardized` says.
    pub pre_standardized_x: bool,
    /// Caller asserts Y is already standardized. **No effect on either
    /// method** — see `pre_standardized_x`.
    pub pre_standardized_y: bool,
    /// X-side keep-count applied to every training-half fit. `None`
    /// (default) fits dense `pls3_fit`. Selection happens strictly inside
    /// the training half; the held-out half is only scored.
    ///
    /// **Route note.** A `Some(keep_x)` below `n_features` forces the
    /// primal route: the Gram reduction in `dual_route.rs` needs `u` to be
    /// a positive multiple of `X̃'Ỹv`, and zeroing coordinates of `u` is
    /// not a positive multiple. `keep_x = Some(n_features)` is the dense
    /// endpoint and leaves the route free.
    pub keep_x: Option<usize>,
    /// Y-side keep-count applied to every training-half fit. Compatible
    /// with both routes: the `v` step is the one the Gram reduction
    /// computes anyway, and hard selection by `|v|` order is invariant to
    /// the positive rescale that separates the two formulations.
    pub keep_y: Option<usize>,
    /// Iteration cap of each training-half sparse fit. Must be at least
    /// `1` when either keep-count selects fewer than all columns.
    pub max_iter: usize,
    /// Convergence tolerance of each training-half sparse fit. Must be
    /// finite and non-negative when either keep-count selects fewer than
    /// all columns.
    pub tol: f64,
    /// RNG seed; `None` draws from OS entropy and the drawn value is
    /// recorded on the result.
    pub seed: Option<u64>,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
}

impl Default for Pls3ConfirmatoryTestOpts {
    fn default() -> Self {
        // `max_iter` / `tol` are read from the fit options rather than
        // re-typed: the sparse alternation they drive is `spls3_fit`'s, so a
        // change there must not leave the confirmatory test on a stale cap.
        // `Pls3FitOpts` is a different type, so `..Pls3FitOpts::default()`
        // cannot bridge them and the two fields are named explicitly.
        let fit = Pls3FitOpts::default();
        Self {
            args: ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::Auto),
            pre_standardized_x: false,
            pre_standardized_y: false,
            keep_x: None,
            keep_y: None,
            max_iter: fit.max_iter,
            tol: fit.tol,
            seed: None,
            verbose: false,
        }
    }
}

/// Is the Gram route on the menu for this configuration?
///
/// The route choice is internal and silent and decided from shape alone,
/// except that a sparse X side removes the Gram route outright: the
/// reduction in `dual_route::pls3_split_zbars_columns` needs `u` to be a
/// positive multiple of `X̃'Ỹv`, and zeroing coordinates of `u` is not a
/// positive multiple. `keep_x = Some(n_features)` is the dense endpoint and
/// leaves the route free, as does `None`.
///
/// This lives in one place so the tests can assert the real rule instead of
/// a copy of it: a test that re-derives the condition cannot fail when the
/// condition changes. `n_perm` is the permutation count, and the `+ 1` for
/// the observed column is part of the rule, not of the caller.
fn dual_route_eligible(
    keep_x: Option<usize>,
    n_tr: usize,
    n_features: usize,
    n_perm: usize,
    n_targets: usize,
) -> bool {
    let x_side_is_sparse = keep_x.is_some_and(|kx| kx < n_features);
    !x_side_is_sparse && crate::dual_route::use_dual_route(n_tr, n_features, n_perm + 1, n_targets)
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
///   `split_exact`, `split_nb` or `auto`, for `n_splits < 2`, for `n_perm < 1`, for
///   `n < k + 5` (the split floor `draw_splits` enforces), for a `keep_x` /
///   `keep_y` of `0` or above its dimension, or, when either keep-count
///   selects fewer than all columns, for `max_iter == 0` or a NaN,
///   infinite or negative `tol`
/// - `PlsKitError::KExceedsMax` when `x` has 0 columns or `y` has 0 columns
///   (mirrors the `k_max = n_features.min(n_targets)` check `pls3_fit` itself
///   makes — this call always requests `k = 1`, so `k_max = 0` on either
///   empty block trips it)
/// - `PlsKitError::NonFiniteInput` when X or Y contains NaN/inf
///
/// A failed inner fit on a split (SVD non-convergence, or any other
/// `pls3_fit` error) is not propagated — it degrades that split's `r` to
/// `0.0`, the same convention PLS1's `split_half_correlations` documents
/// and accepts (`signal_test.rs`, "A failed per-half fit already degrades
/// to r = 0").
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
#[allow(clippy::too_many_lines)] // two method bodies plus the gate, in one dispatch
pub fn pls3_confirmatory_test(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    opts: Pls3ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    crate::fit::with_thread_limit(|| pls3_confirmatory_test_impl(x, y, k, opts))
}

/// Body of [`pls3_confirmatory_test`], on the caller's pool.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
#[allow(clippy::too_many_lines)] // two method bodies plus the gate, in one dispatch
pub(crate) fn pls3_confirmatory_test_impl(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    k: usize,
    opts: Pls3ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x.ncols()),
            y: y.nrows(),
        });
    }
    crate::fit::check_finite_mat(x)?;
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
    let k_max = x.ncols().min(y.ncols());
    if k > k_max {
        return Err(PlsKitError::KExceedsMax { k, k_max });
    }

    // The only validation the sparse knobs get: the per-split fits
    // (`pls3::pls3_lv1_prestd`) take `keep_x` / `keep_y` / `max_iter` /
    // `tol` as given. Checking once here is also right on its own terms:
    // malformed input hits every split alike, which is not per-split
    // degeneracy, and the Gram route never runs the primal fit at all, so
    // an error swallowed per split on one route only would make the route
    // choice observable.
    crate::pls3::validate_sparse(
        opts.keep_x,
        opts.keep_y,
        x.ncols(),
        y.ncols(),
        opts.max_iter,
        opts.tol,
    )?;

    // `n_perm` doubles as the mode flag from here down: `Some` means a
    // permutation reference runs — either because split_exact was asked for,
    // or because the gate below rerouted split_nb into it — and `None` means
    // the NB t-reference runs. `Auto` carries its `n_perm` into the floor
    // checks below and is resolved after them.
    let (mut n_perm, n_splits, force) = match opts.args {
        ConfirmatoryArgs::SplitExact { n_perm, n_splits }
        | ConfirmatoryArgs::Auto { n_perm, n_splits } => (Some(n_perm), n_splits, false),
        ConfirmatoryArgs::SplitNb { n_splits, force } => (None, n_splits, force),
        other => {
            return Err(PlsKitError::InvalidArgument(format!(
                "pls3_confirmatory_test supports test_method='split_exact', 'split_nb' or \
                 'auto' (got '{}'): 'raw_perm' needs a CV statistic PLS3 does not have (there is no \
                 pls3_predict); 'score' is not implemented (its symmetric analog, an \
                 RV-type test on ‖X'Y‖_F², tests a different estimand); 'e' needs a \
                 generative model that symmetric cross-decomposition does not supply",
                other.method().as_str()
            )));
        }
    };
    check_split_exact_counts(n_perm, n_splits)?;

    // The `split_nb` auto-gate, on X only. Y is deliberately never gated: q
    // is 3-10 in ordinary PLSC use, so a stable-rank floor of 3 on Y would
    // fire on nearly every design. The thresholds are PLS1's, calibrated on
    // single-block designs and not re-derived for a two-block one — see
    // `signal_test::SPLIT_NB_GATE_MIN_N_EFF` for their provenance.
    let mut stable_rank_out: Option<f64> = None;
    if let ConfirmatoryArgs::Auto { .. } = opts.args {
        // Same X-only resolution, with the same unweighted n_eff, but without
        // the width ceiling: PLS3's `split_exact` always refits. `split_nb`
        // here is the resolved method, so it runs without a second gate.
        let (method, sr) = resolve_auto(x, None, n as f64, false);
        stable_rank_out = sr;
        if method == ConfirmatoryMethod::SplitNb {
            n_perm = None;
        }
    } else if n_perm.is_none() {
        // No weights on this family, so Kish n_eff is exactly the row count.
        let gate = resolve_split_nb(x, None, n as f64, force);
        stable_rank_out = Some(gate.stable_rank);
        if gate.reroute {
            n_perm = Some(SPLIT_NB_REROUTE_N_PERM);
        }
    }

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;

    // The J splits are drawn once and held fixed across all B permutation
    // replicates: redrawing per replicate would fold split-to-split scatter
    // into the null.
    let splits = draw_splits(n, k, n_splits, &mut rng)?;

    let (n_tr, n_te) = crate::resample::split_sizes(n, k);

    let Some(n_perm) = n_perm else {
        // No permutation loop: the J held-out correlations go straight to the
        // NB t-reference. The statistic is the one split_exact reports —
        // `tanh(z̄)` over the same fixed splits — so only the reference
        // changes. Dense blocks are read from X block by block; a sparse
        // side refits on materialized halves. At a shared seed the two
        // methods agree on the statistic to the route tolerance (~1e-10
        // relative), and bit for bit when both fit materialized halves (a
        // sparse side on split_exact's primal route).
        let dense = opts.keep_x.is_none_or(|kx| kx >= x.ncols())
            && opts.keep_y.is_none_or(|ky| ky >= y.ncols());
        let r_obs = if dense {
            pls3_split_lv_correlations_streamed(x, y, &splits, &opts)
        } else {
            pls3_split_lv_correlations(x, y, &splits, &opts)
        };
        let (p, statistic, _t, _df) = nb_test(&r_obs, n_tr, n_te);
        return Ok(ConfirmatoryTestOutput {
            pvalue: p,
            statistic,
            test_method: ConfirmatoryMethod::SplitNb.as_str().to_owned(),
            k,
            n_perm: None,
            n_splits: Some(n_splits),
            seed: seed_used,
            ci: None,
            // No weights on this family, so Kish n_eff is exactly the row count.
            n_eff: n as f64,
            rho_hat: nb_rho_hat(&r_obs, n_te),
            stable_rank: stable_rank_out,
        });
    };

    // Route choice: see `dual_route_eligible`, which owns the rule. Both
    // routes get the same splits and the same permutations, so the choice is
    // invisible in the output up to the crate's bit-near tolerance.
    let dual = dual_route_eligible(opts.keep_x, n_tr, x.ncols(), n_perm, y.ncols());

    // The B permutations, drawn once and shared by both routes. Permute the
    // ROWS of Y as units against X: that breaks the cross-block association
    // while leaving Y's within-row structure intact. A row permutation
    // reorders each column, so the full matrix's column moments are
    // unchanged, but nothing here depends on that: every split half standardizes on
    // its own rows, and those half-level moments do move with the
    // permutation.
    //
    // This is drawn *above* the route branch and is the only draw either
    // route makes, so the parent's seed budget is exactly
    // `resolve_seed` → `draw_splits(n_splits)` → `child_seeds(n_perm)` on
    // both arms. Replicate `b` gets the permutation of `seeds[b]` alone,
    // whichever route runs and however Rayon schedules it.
    let seeds = crate::rng::child_seeds(&mut rng, n_perm);
    let perms: Vec<Vec<usize>> = seeds
        .iter()
        .map(|s| crate::resample::permute_indices(n, &mut crate::rng::child_rng(*s)))
        .collect();

    // Both routes return `n_perm + 1` per-column z̄ values: column 0 is the
    // observed row map, columns `1..=n_perm` are the permutation nulls.
    let z = if dual {
        crate::dual_route::pls3_split_zbars_columns(
            x,
            y,
            &splits,
            &perms,
            opts.keep_y,
            opts.max_iter,
            opts.tol,
        )
    } else {
        pls3_split_zbars_columns_primal(x, y, &splits, &perms, &opts)
    };
    let (z_bar_obs, null_zbars) = (z[0], &z[1..]);

    // A non-finite null statistic counts as an exceedance so p is biased
    // upward, never downward. Mirrors run_split_perm / run_split_perm_nr in
    // signal_test.rs — change together.
    let exceedances = null_zbars
        .iter()
        .filter(|v| !v.is_finite() || **v >= z_bar_obs)
        .count();
    // `cast_precision_loss` is allowed crate-wide in lib.rs:10 — sample-size
    // usize → f64 casts are routine here.
    let p = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);
    // No weights on this family, so Kish n_eff is exactly the row count.
    let n_eff = n as f64;

    Ok(ConfirmatoryTestOutput {
        pvalue: p,
        statistic: z_bar_obs.tanh(),
        test_method: ConfirmatoryMethod::SplitExact.as_str().to_owned(),
        k,
        n_perm: Some(n_perm),
        n_splits: Some(n_splits),
        seed: seed_used,
        ci: None,
        n_eff,
        rho_hat: None,
        // `Some` only when split_nb was requested and rerouted here, or when
        // `auto` resolved here after the stable-rank check — it is what the
        // gate saw.
        stable_rank: stable_rank_out,
    })
}

/// The large buffers of one split on the primal route when `n_cols`
/// replicate columns of it are fitted: the standardized halves of X
/// (`signal_test::split_blocks_bytes`) and one LV1 fit's buffers
/// (`pls3::lv1_bytes`) per column that runs at once
/// (`signal_test::columns_bytes`).
pub(crate) fn primal_split_unit_bytes(
    splits: &[SplitIdx],
    p: usize,
    q: usize,
    n_cols: usize,
) -> usize {
    crate::signal_test::split_blocks_bytes(splits, p).saturating_add(
        crate::signal_test::columns_bytes(crate::pls3::lv1_bytes(p, q), n_cols),
    )
}

/// Held-out LV correlation on each supplied split, computed from an honest
/// per-split PLS3 refit (the primal route).
///
/// Takes the splits rather than drawing them so the permutation loop can
/// hold one set fixed across all B replicates. Nothing here consumes
/// randomness: given `(x, y, split)` the fit and the r are deterministic.
/// The J splits run in batches of `resample::map_units` by the memory
/// budget, one unit being the split's standardized halves of X and one LV1
/// fit's buffers ([`primal_split_unit_bytes`] with one column; the halves
/// of Y are counted as small); the result does not depend on the batch
/// size.
///
/// Both halves of both blocks are standardized with **training-half**
/// moments: `Ỹ_te v₁` mixes Y columns, so the test half must sit on the
/// scale `v₁` was estimated on. The fit and the score on those blocks are
/// [`pls3_split_column_r_primal`], the same body every `split_exact` route
/// evaluates per `(split, replicate column)`.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn pls3_split_lv_correlations(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[SplitIdx],
    opts: &Pls3ConfirmatoryTestOpts,
) -> Col<f64> {
    use crate::linalg::{standardize_apply_rows, standardize_rows};

    let per_split = |sp: &SplitIdx| -> f64 {
        let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
        let (xs_tr, x_mean, x_scale) = standardize_rows(x, tr, None, None);
        let xs_te = standardize_apply_rows(x, te, x_mean.as_ref(), x_scale.as_ref(), None);
        // A non-finite held-out block gives `r = 0`, as in
        // `pls3_split_columns_z_primal`, which explains it (change together).
        if crate::fit::check_finite_mat(xs_te.as_ref()).is_err() {
            return 0.0;
        }
        let (ys_tr, y_mean, y_scale) = standardize_rows(y, tr, None, None);
        let ys_te = standardize_apply_rows(y, te, y_mean.as_ref(), y_scale.as_ref(), None);
        // Likewise a non-finite held-out Y block; `pls3_split_columns_z_primal`
        // and `dual_route::pls3_split_zbars_columns` check it too (change
        // together).
        if crate::fit::check_finite_mat(ys_te.as_ref()).is_err() {
            return 0.0;
        }

        pls3_split_column_r_primal(
            xs_tr.as_ref(),
            xs_tr.norm_l2(),
            xs_te.as_ref(),
            ys_tr.as_ref(),
            ys_te.as_ref(),
            opts.keep_x,
            opts.keep_y,
            opts.max_iter,
            opts.tol,
        )
    };

    // The splits run in batches by the memory budget, collected in split order.
    let unit = primal_split_unit_bytes(splits, x.ncols(), y.ncols(), 1);
    let r_vec = crate::resample::map_units(splits.len(), unit, |j| per_split(&splits[j]));
    Col::<f64>::from_fn(r_vec.len(), |i| r_vec[i])
}

/// Held-out LV1 correlation of one split on dense blocks, read from `X`
/// one block of columns at a time, or `None` when this route cannot decide
/// the split and the caller must run the materialized fit.
///
/// # What it computes
/// With `a` and `c` the standardized training and held-out blocks of a
/// column range, one pass forms `A = X̃_tr'Ỹ_tr` (`p × q`, a block of rows
/// at a time) and sums `T = X̃_te A` (`n_te × q`) over the blocks. The
/// component stage of `pls3::pls3_lv1_prestd` gives `v₁` and `σ₁` from `A`,
/// and since `A v₁ = σ₁ u₁` with `σ₁ > 0`, `T v₁` is `X̃_te u₁` times `σ₁`:
/// the correlation with `Ỹ_te v₁` does not see the factor, and `u₁` is
/// never applied to `X̃_te`.
///
/// # Gates (decision gates)
/// Each row block of `A` is one product of the columns the materialized
/// fit multiplies, so it can differ from that fit's one-call `X̃_tr'Ỹ_tr`
/// only where faer evaluates a product of another shape differently. `T`
/// is summed over blocks, where the materialized fit applies `u₁` to
/// `X̃_te` in one product, and `‖X̃_tr‖_F` comes from the column moments,
/// not from `norm_l2`. So the floor and, apart from its factor `σ₁`, the
/// X score can differ from the materialized fit's in the last bits, and
/// where `A` differs so can `σ₁`, `v₁` and the Y score. The split keeps
/// its `r` only when `σ₁` is at least `dual_route::FLOOR_BAND` times the
/// relative floor and `dual_route::ABS_BAND` times `pls3::SIGMA_FLOOR`,
/// so the materialized fit keeps the component too, and when neither
/// score is constant to rounding and each has a centered norm of at least
/// `dual_route::SCORE_BAND` times its norm, clear of `guarded_pearson`'s
/// constant test. `a >= b` conjunctions, so a NaN anywhere fails. A
/// near-tie `σ₁ ≈ σ₂` is not gated, as on the dense arm of
/// `dual_route::pls3_split_zbars_columns`.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn pls3_split_r_streamed(x: MatRef<'_, f64>, y: MatRef<'_, f64>, sp: &SplitIdx) -> Option<f64> {
    use crate::dual_route::{ABS_BAND, FLOOR_BAND, SCORE_BAND};
    use crate::linalg::{ScaledMoments, scaled_moments, standardize_apply_rows, standardize_rows};
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
    let (p, q) = (x.ncols(), y.ncols());
    let (n_tr, n_te) = (tr.len(), te.len());
    let (ys_tr, y_mean, y_scale) = standardize_rows(y, tr, None, None);
    let ys_te = standardize_apply_rows(y, te, y_mean.as_ref(), y_scale.as_ref(), None);

    // Seq inside the per-split worker: outer Rayon owns the pool.
    let seq = faer::Par::Seq;
    let mut a_mat = Mat::<f64>::zeros(p, q);
    let mut t_mat = Mat::<f64>::zeros(n_te, q);
    let x_fro = crate::linalg::standardized_blocks(
        x,
        tr,
        te,
        None,
        None,
        None,
        // The width linalg::stream_blocks_bytes counts (change together).
        crate::linalg::stream_block_width(n_tr.max(n_te)),
        |j0, a, c| {
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
    // A non-finite product is the materialized fit's to answer.
    if !(a_mat.is_all_finite() && t_mat.is_all_finite() && ys_te.is_all_finite()) {
        return None;
    }

    let rel_floor = crate::pls3::sigma_rel_floor(n_tr, p, q, x_fro, ys_tr.norm_l2());
    let Ok(Some((v, sigma))) = crate::pls3::pls3_lv1_from_cross_cov(a_mat.as_ref(), rel_floor)
    else {
        return None;
    };
    if !(sigma >= FLOOR_BAND * rel_floor && sigma >= ABS_BAND * crate::pls3::SIGMA_FLOOR) {
        return None;
    }

    let s = crate::linalg::mat_vec(t_mat.as_ref(), v.as_ref(), seq);
    let t = crate::linalg::mat_vec(ys_te.as_ref(), v.as_ref(), seq);
    // `ss` and `sq` are the centered and raw sums of squares on one scale.
    let clear = |m: &ScaledMoments| !m.is_constant(n_te) && m.ss.sqrt() >= SCORE_BAND * m.sq.sqrt();
    if !(clear(&scaled_moments(n_te, |i| s[i], None))
        && clear(&scaled_moments(n_te, |i| t[i], None)))
    {
        return None;
    }
    Some(pearson_r_guarded(&s, &t))
}

/// [`pls3_split_lv_correlations`] for dense X and dense Y: each split is
/// [`pls3_split_r_streamed`], and a split that route cannot decide is the
/// materialized fit's to the bit. Both passes run the splits in batches of
/// `resample::map_units`.
fn pls3_split_lv_correlations_streamed(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[SplitIdx],
    opts: &Pls3ConfirmatoryTestOpts,
) -> Col<f64> {
    // Pass 1: every split the streamed body decides. A split holds two
    // stream blocks and one LV1 fit's buffers (`A`, its SVD's `U`, `V` and
    // scratch).
    let unit = crate::pls3::lv1_bytes(x.ncols(), y.ncols()).saturating_add(
        splits.first().map_or(0, |sp| {
            crate::linalg::stream_blocks_bytes(sp.tr.len(), sp.te.len())
        }),
    );
    let decided: Vec<Option<f64>> = crate::resample::map_units(splits.len(), unit, |j| {
        pls3_split_r_streamed(x, y, &splits[j])
    });
    // Pass 2: the undecided splits run in batches of `resample::map_units`,
    // the standardized halves of X and one LV1 fit counted per split. An
    // undecided split is the materialized fit's, which standardizes both
    // halves of X.
    let rows: Vec<Vec<Option<f64>>> = decided.into_iter().map(|r| vec![r]).collect();
    let rows = crate::resample::fill_pending(
        rows,
        primal_split_unit_bytes(splits, x.ncols(), y.ncols(), 1),
        |j, _| vec![pls3_split_lv_correlations(x, y, std::slice::from_ref(&splits[j]), opts)[0]],
    );
    Col::<f64>::from_fn(rows.len(), |j| rows[j][0])
}

/// The clamped Fisher z of the columns `cols` of one split on the primal
/// route, in the order of `cols`: [`pls3_split_column_r_primal`] per column
/// on halves of X standardized once for the split. Column 0 is the
/// identity row map and column `c > 0` applies `perms[c − 1]`. The columns
/// are independent refits sharing only the read-only X side, so they map in
/// parallel; `map_indexed`'s `collect` keeps their order, and each value
/// does not depend on which worker computes it.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pls3_split_columns_z_primal(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    sp: &SplitIdx,
    perms: &[Vec<usize>],
    keep_x: Option<usize>,
    keep_y: Option<usize>,
    max_iter: usize,
    tol: f64,
    cols: &[usize],
) -> Vec<f64> {
    use crate::linalg::{standardize, standardize_apply, standardize_apply_rows, standardize_rows};

    let q = y.ncols();
    let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());

    // Built once per split: invariant across replicates, because only Y
    // is permuted.
    let (xs_tr, x_mean, x_scale) = standardize_rows(x, tr, None, None);
    let xs_te = standardize_apply_rows(x, te, x_mean.as_ref(), x_scale.as_ref(), None);
    // The X-side input of the relative floor, likewise per split.
    let xs_tr_fro = xs_tr.norm_l2();
    // A held-out block with a non-finite entry has no finite score, so every
    // column gets `r = 0` (`atanh(0) = 0`), the value a non-finite training
    // block gets in `pls3_split_column_r_primal`. Checked here once per
    // split rather than in that per-column body, which would rescan `X̃_te`
    // for every replicate column. `pls3_split_lv_correlations` makes the
    // same check, and `signal_test::prepare_split` makes it for PLS1 as its
    // `x_finite` (change together).
    if crate::fit::check_finite_mat(xs_te.as_ref()).is_err() {
        return vec![0.0; cols.len()];
    }

    crate::resample::map_indexed(cols.len(), |u| {
        let col = cols[u];
        let row_of = |i: usize| if col == 0 { i } else { perms[col - 1][i] };
        let y_tr = Mat::<f64>::from_fn(tr.len(), q, |i, j| y[(row_of(tr[i]), j)]);
        let y_te = Mat::<f64>::from_fn(te.len(), q, |i, j| y[(row_of(te[i]), j)]);
        let (ys_tr, y_mean, y_scale) = standardize(y_tr.as_ref());
        let ys_te = standardize_apply(y_te.as_ref(), y_mean.as_ref(), y_scale.as_ref());
        // A non-finite held-out Y block gives `r = 0` as a non-finite `X̃_te`
        // does above; it is checked per column because Y is permuted per
        // column. `pls3_split_lv_correlations` and the per-column body of
        // `dual_route::pls3_split_zbars_columns` make the same check
        // (change together).
        if crate::fit::check_finite_mat(ys_te.as_ref()).is_err() {
            return 0.0;
        }

        let r = pls3_split_column_r_primal(
            xs_tr.as_ref(),
            xs_tr_fro,
            xs_te.as_ref(),
            ys_tr.as_ref(),
            ys_te.as_ref(),
            keep_x,
            keep_y,
            max_iter,
            tol,
        );
        // ±0.9999 pre-atanh clamp applied per (split, column), exactly
        // where `mean_fisher_z` applies it before summing.
        r.clamp(-0.9999, 0.9999).atanh()
    })
}

/// Per-column z̄ over the J splits on the primal route: `perms.len() + 1`
/// values, column 0 the observed row map and column `c > 0` permutation
/// `c − 1`. The primal sibling of `dual_route::pls3_split_zbars_columns`,
/// and shaped like it on purpose.
///
/// # What it is
/// Exactly `mean_fisher_z(pls3_split_lv_correlations(..))` evaluated once per
/// replicate, with the two loops interchanged: splits outer, replicates
/// inner. Only Y is permuted, so a split's X side (`standardize_rows` on the
/// training rows, `standardize_apply_rows` on the test rows) is a pure
/// function of `(x, split)` and is built once per split rather than
/// `n_perm + 1` times. That is the same hoist the Gram
/// route already makes ("Built once per split; neither carries p into the
/// replicate loop"), and it matters here because a sparse X side takes the
/// Gram route off the menu (`dual_route_eligible`): the `p ≫ n` shapes that
/// most need the hoist are routed to *this* function, not to that one.
///
/// # Why it is byte-identical, not merely close
/// - The permutations are materialized by the caller above the route branch,
///   so nothing about parent-seed consumption moves, and replicate `b`'s
///   permutation stays a pure function of `seeds[b]`.
/// - The per-`(split, column)` body is [`pls3_split_column_r_primal`],
///   the one `pls3_split_lv_correlations` calls, here reached through
///   [`pls3_split_columns_z_primal`], on bit-identical inputs: the
///   hoisted X quantities were already recomputed to the same bits every
///   replicate, and `y[(perm[tr[i]], j)]` is the same load
///   `standardize_rows(y_perm, tr, ..)` makes.
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
/// replicate-outer loop out of it (row-subsetting a permuted Y, X
/// re-standardized per replicate) and assert p-value equality and
/// `statistic.to_bits()` against what this function produces. They check
/// the hoist, the column parallelism and the accumulation, not the shared
/// per-column body.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn pls3_split_zbars_columns_primal(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[SplitIdx],
    perms: &[Vec<usize>],
    opts: &Pls3ConfirmatoryTestOpts,
) -> Vec<f64> {
    let n_cols = perms.len() + 1;
    let cols: Vec<usize> = (0..n_cols).collect();
    let per_split = |sp: &SplitIdx| -> Vec<f64> {
        pls3_split_columns_z_primal(
            x,
            y,
            sp,
            perms,
            opts.keep_x,
            opts.keep_y,
            opts.max_iter,
            opts.tol,
            &cols,
        )
    };

    // Splits in batches of `resample::map_units`; the columns of a split
    // map in parallel, so a split counts its halves of X and one LV1 fit
    // per column that runs at once.
    let unit_bytes = primal_split_unit_bytes(splits, x.ncols(), y.ncols(), n_cols);
    crate::signal_test::zbars_over_splits(splits, n_cols, unit_bytes, per_split)
}

/// Held-out LV1 correlation for one `(split, replicate column)` on the
/// primal route: an honest `pls3_fit` / `spls3_fit` refit on the
/// standardized training half, scored on the standardized test half.
///
/// The one per-column body of every PLS3 split statistic computed on
/// materialized halves: `pls3_split_lv_correlations` (`split_nb` with a
/// sparse side, and each dense `split_nb` split `pls3_split_r_streamed`
/// does not decide), the `split_exact` primal route
/// (`pls3_split_zbars_columns_primal`), and the second pass of
/// `dual_route::pls3_split_zbars_columns` on a column its truncation gates
/// cannot decide or whose sparse alternation comes within a guard band of
/// a discontinuity (see "Truncation" and "Near-ties" there). Keeping one
/// body is what keeps those paths bit-equal. Returns `r` before the `atanh` clamp.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
pub(crate) fn pls3_split_column_r_primal(
    xs_tr: MatRef<'_, f64>,
    xs_tr_fro: f64,
    xs_te: MatRef<'_, f64>,
    ys_tr: MatRef<'_, f64>,
    ys_te: MatRef<'_, f64>,
    keep_x: Option<usize>,
    keep_y: Option<usize>,
    max_iter: usize,
    tol: f64,
) -> f64 {
    // Selection is per training half, on that half's own
    // standardized block. The held-out half below is only ever
    // scored: a support chosen on all n rows and then scored on a
    // subset of them would leak the test half into the statistic.
    //
    // `pls3_lv1_prestd` is the LV1 of `pls3_fit` (dense request, or a keep
    // at its dense endpoint) or of `spls3_fit`, pre-standardized and with
    // `ParChoice::Seq` inside the per-split or per-column worker (outer
    // Rayon owns the pool), without the score matmuls those fits would build
    // and this loop would discard. `Seq` governs the SVD too
    // (`linalg::thin_svd`), and the two score products below are `Seq`:
    // nothing here reads faer's global parallelism, whose degree is the
    // pool size.
    match crate::pls3::pls3_lv1_prestd(xs_tr, xs_tr_fro, ys_tr, keep_x, keep_y, max_iter, tol) {
        // `None` is a half with no first component: a zero X̃ or Ỹ block,
        // or a Ỹ orthogonal to X̃ up to rounding (`σ₁` under the relative
        // floor). No direction exists, so the honest answer is no
        // association, not NaN. An
        // `Err` is an inner-fit failure (e.g. SVD non-convergence) on this
        // split, swallowed to the same `0.0`, the swallow
        // `signal_test::split_half_correlations` documents ("A failed
        // per-half fit already degrades to r = 0").
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
    use crate::pls3::spls3_fit;
    use crate::signal_test::ConfirmatoryArgs;

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

    #[test]
    fn strong_shared_factor_rejects() {
        let (x, y) = linked_blocks(80, 8, 4, 4.0, 3);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(199, 10, 42)).unwrap();
        assert_eq!(r.test_method, "split_exact");
        assert_eq!(r.k, 1);
        assert_eq!(r.n_perm, Some(199));
        assert_eq!(r.n_splits, Some(10));
        assert_eq!(r.seed, 42);
        assert!(r.rho_hat.is_none());
        assert!(r.stable_rank.is_none());
        assert!(r.ci.is_none());
        assert!(r.pvalue <= 0.01, "p = {}", r.pvalue);
        assert!(r.statistic > 0.3, "statistic = {}", r.statistic);
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

    /// Every split route reads X and Y only through row-subset
    /// standardization into owned matrices or column blocks, so a padded
    /// submatrix, a row-major view and a negative-stride view of X, or of
    /// Y, give the owned matrix's result to the bit: on `split_exact`'s
    /// primal route (dense 48x7, or any `keep_x`) and Gram route (wide
    /// 30x400, `keep_x` None), on `split_nb`'s streamed route (no keep) and
    /// on its materialized refit (any keep).
    #[test]
    fn confirmatory_output_is_bit_identical_across_x_and_y_layouts() {
        use crate::test_support::{assert_layout_invariant, copy_free_families};
        assert!(
            crate::dual_route::use_dual_route(crate::resample::split_sizes(30, 1).0, 400, 10, 4),
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
                for (keep_x, keep_y) in [
                    (None, None),
                    (Some(3), None),
                    (None, Some(2)),
                    (Some(3), Some(2)),
                ] {
                    let o = Pls3ConfirmatoryTestOpts {
                        args,
                        keep_x,
                        keep_y,
                        seed: Some(6),
                        ..Pls3ConfirmatoryTestOpts::default()
                    };
                    let what = format!("{} {args:?} keep=({keep_x:?},{keep_y:?})", f.name);
                    assert_layout_invariant(&f.x, &format!("X {what}"), |xv| {
                        pls3_confirmatory_test(xv, y.as_ref(), 1, o).unwrap()
                    });
                    assert_layout_invariant(&y, &format!("Y {what}"), |yv| {
                        pls3_confirmatory_test(f.x.as_ref(), yv, 1, o).unwrap()
                    });
                }
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

    /// A held-out half that overflows in standardization while its training
    /// half does not gives `r = 0`, the value a non-finite training half
    /// gives. Column 0's training rows are scaled by 1e-3 (a training scale
    /// below 1) and its held-out rows hold ±1.7e308, which standardize to
    /// ±∞.
    #[test]
    fn a_held_out_half_that_overflows_gives_r_zero() {
        let (mut x, y) = linked_blocks(12, 3, 2, 1.0, 5);
        for i in 0..6 {
            x[(i, 0)] *= 1e-3;
        }
        for i in 6..12 {
            x[(i, 0)] = if i % 2 == 0 { 1.7e308 } else { -1.7e308 };
        }
        let splits = vec![SplitIdx {
            tr: (0..6).collect(),
            te: (6..12).collect(),
        }];
        let opts = Pls3ConfirmatoryTestOpts::default();
        let r = pls3_split_lv_correlations(x.as_ref(), y.as_ref(), &splits, &opts);
        assert_eq!(r[0].to_bits(), 0.0_f64.to_bits(), "r = {}", r[0]);
        let z = pls3_split_columns_z_primal(
            x.as_ref(),
            y.as_ref(),
            &splits[0],
            &[],
            None,
            None,
            opts.max_iter,
            opts.tol,
            &[0],
        );
        assert_eq!(z[0].to_bits(), 0.0_f64.to_bits(), "z = {}", z[0]);
    }

    /// The same overflow on the Y side gives `r = 0` (`z = 0`) on every
    /// route: Y column 0's training rows scaled by 1e-3 and its held-out
    /// rows at ±1.7e308. `split_nb` on the materialized halves and on the
    /// streamed body (which hands the split to them), and `split_exact` on
    /// the primal and the Gram route.
    #[test]
    fn a_held_out_y_half_that_overflows_gives_r_zero() {
        let (x, mut y) = linked_blocks(12, 3, 2, 1.0, 5);
        for i in 0..6 {
            y[(i, 0)] *= 1e-3;
        }
        for i in 6..12 {
            y[(i, 0)] = if i % 2 == 0 { 1.7e308 } else { -1.7e308 };
        }
        let splits = vec![SplitIdx {
            tr: (0..6).collect(),
            te: (6..12).collect(),
        }];
        let opts = Pls3ConfirmatoryTestOpts::default();
        let r = pls3_split_lv_correlations(x.as_ref(), y.as_ref(), &splits, &opts);
        assert_eq!(r[0].to_bits(), 0.0_f64.to_bits(), "r = {}", r[0]);
        let r = pls3_split_lv_correlations_streamed(x.as_ref(), y.as_ref(), &splits, &opts);
        assert_eq!(r[0].to_bits(), 0.0_f64.to_bits(), "streamed r = {}", r[0]);
        let z = pls3_split_columns_z_primal(
            x.as_ref(),
            y.as_ref(),
            &splits[0],
            &[],
            None,
            None,
            opts.max_iter,
            opts.tol,
            &[0],
        );
        assert_eq!(z[0].to_bits(), 0.0_f64.to_bits(), "primal z = {}", z[0]);
        let z = crate::dual_route::pls3_split_zbars_columns(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &[],
            None,
            opts.max_iter,
            opts.tol,
        );
        assert_eq!(z[0].to_bits(), 0.0_f64.to_bits(), "gram z = {}", z[0]);
    }

    /// Every PLS3 split statistic on the materialized halves stays finite on
    /// an X that overflows only in held-out halves (column 0 scaled by 1e-3,
    /// row 0 of it at 1e306): dense `split_nb` (the streamed body, which
    /// hands such a split to the materialized fit), sparse `split_nb`, and
    /// `split_exact` on the primal route, dense and sparse.
    #[test]
    fn split_statistics_stay_finite_when_a_held_out_half_overflows() {
        let (mut x, y) = linked_blocks(40, 6, 3, 1.0, 11);
        for i in 0..40 {
            x[(i, 0)] *= 1e-3;
        }
        x[(0, 0)] = 1e306;
        let (n_tr, _) = crate::resample::split_sizes(40, 1);
        assert!(
            !dual_route_eligible(None, n_tr, 6, 19, 3),
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
        for (args, keep_x, keep_y) in [
            (nb, None, None),
            (nb, None, Some(1)),
            (exact, None, None),
            (exact, Some(5), None),
        ] {
            let r = pls3_confirmatory_test(
                x.as_ref(),
                y.as_ref(),
                1,
                Pls3ConfirmatoryTestOpts {
                    args,
                    keep_x,
                    keep_y,
                    seed: Some(7),
                    ..Default::default()
                },
            )
            .expect("a result");
            assert!(
                r.statistic.is_finite(),
                "{args:?} {keep_x:?} {keep_y:?}: {}",
                r.statistic
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
            ConfirmatoryArgs::Score,
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
        assert!(r.statistic.is_finite());
        assert!(r.statistic.abs() < 1e-12, "statistic = {}", r.statistic);
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
        assert!(r.statistic.is_finite());
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
            (nb.statistic - ex.statistic).abs() <= 1e-10,
            "{} vs {}",
            nb.statistic,
            ex.statistic
        );

        // With a sparse X side both methods fit materialized halves
        // (`split_exact` on its primal route), so the two agree to the bit.
        let sparse = |o: Pls3ConfirmatoryTestOpts| Pls3ConfirmatoryTestOpts {
            keep_x: Some(3),
            ..o
        };
        let nb = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sparse(nb_opts(8, false, 42)))
            .unwrap();
        let ex =
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, sparse(opts(99, 8, 42))).unwrap();
        assert_eq!(nb.test_method, "split_nb");
        assert_eq!(ex.test_method, "split_exact");
        assert_eq!(
            nb.statistic.to_bits(),
            ex.statistic.to_bits(),
            "keep_x=3: {} vs {}",
            nb.statistic,
            ex.statistic
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
            let (_, mut rng) = crate::rng::resolve_seed(Some(7)).unwrap();
            let splits = draw_splits(n, 1, 6, &mut rng).unwrap();
            let o = nb_opts(6, false, 7);
            let got = pls3_split_lv_correlations_streamed(x.as_ref(), y.as_ref(), &splits, &o);
            let want = pls3_split_lv_correlations(x.as_ref(), y.as_ref(), &splits, &o);
            for j in 0..splits.len() {
                assert!(
                    pls3_split_r_streamed(x.as_ref(), y.as_ref(), &splits[j]).is_some(),
                    "n={n} p={p}: split {j} should be decided without a copy"
                );
                assert!(
                    (got[j] - want[j]).abs() <= 1e-10,
                    "n={n} p={p} split {j}: {} vs {}",
                    got[j],
                    want[j]
                );
            }
        }
    }

    /// A split the streamed body cannot decide is the materialized fit's to
    /// the bit: a constant X has no first component.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_split_nb_streamed_hands_a_degenerate_split_to_the_materialized_fit() {
        use crate::signal_test::draw_splits;
        let (n, p, q) = (40_usize, 9_usize, 3_usize);
        let x = Mat::<f64>::from_fn(n, p, |_, j| (j + 1) as f64);
        let (_, y) = linked_blocks(n, p, q, 2.0, 5);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let splits = draw_splits(n, 1, 4, &mut rng).unwrap();
        let o = nb_opts(4, true, 3);
        let got = pls3_split_lv_correlations_streamed(x.as_ref(), y.as_ref(), &splits, &o);
        let want = pls3_split_lv_correlations(x.as_ref(), y.as_ref(), &splits, &o);
        for j in 0..splits.len() {
            assert!(pls3_split_r_streamed(x.as_ref(), y.as_ref(), &splits[j]).is_none());
            assert_eq!(got[j].to_bits(), want[j].to_bits(), "split {j}");
        }
    }

    /// A constant X leaves every split undecided, so the second pass of
    /// `pls3_split_lv_correlations_streamed` fills all of them. Both passes
    /// run one split at a time at budget 0 and all in one batch with no
    /// limit, and both give the materialized correlations to the bit.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn pls3_split_nb_streamed_fills_several_undecided_splits_in_one_batch() {
        use crate::resample::{copy_bytes, units_at_once};
        use crate::signal_test::draw_splits;
        let (n, p, q) = (40_usize, 9_usize, 3_usize);
        let x = Mat::<f64>::from_fn(n, p, |_, j| (j + 1) as f64);
        let (_, y) = linked_blocks(n, p, q, 2.0, 5);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let splits = draw_splits(n, 1, 4, &mut rng).unwrap();
        let o = nb_opts(4, true, 3);
        let unit = copy_bytes(n, p);
        assert_eq!(crate::fit::with_budget(0, || units_at_once(4, unit)), 1);
        assert_eq!(
            crate::fit::with_budget(u64::MAX, || units_at_once(4, unit)),
            4
        );
        for sp in &splits {
            assert!(pls3_split_r_streamed(x.as_ref(), y.as_ref(), sp).is_none());
        }
        let run = |b| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(b, || {
                    pls3_split_lv_correlations_streamed(x.as_ref(), y.as_ref(), &splits, &o)
                })
            })
        };
        let ((one, log_one), (all, log_all)) = (run(0), run(u64::MAX));
        crate::resample::assert_batches(&log_one, 4, 1, 2);
        crate::resample::assert_batches(&log_all, 4, 4, 2);
        let want = pls3_split_lv_correlations(x.as_ref(), y.as_ref(), &splits, &o);
        for j in 0..splits.len() {
            assert_eq!(one[j].to_bits(), all[j].to_bits(), "split {j}");
            assert_eq!(one[j].to_bits(), want[j].to_bits(), "split {j}");
        }
    }

    /// `pls3_split_lv_correlations` with a sparse X side gives the same bits
    /// with its splits one at a time (budget 0) and all in one batch (no
    /// limit).
    #[test]
    fn pls3_split_lv_correlations_do_not_depend_on_the_budget() {
        use crate::signal_test::draw_splits;
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let (_, mut rng) = crate::rng::resolve_seed(Some(42)).unwrap();
        let splits = draw_splits(60, 1, 8, &mut rng).unwrap();
        let o = Pls3ConfirmatoryTestOpts {
            keep_x: Some(3),
            ..nb_opts(8, false, 42)
        };
        let run = |b| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(b, || {
                    pls3_split_lv_correlations(x.as_ref(), y.as_ref(), &splits, &o)
                })
            })
        };
        let ((one, log_one), (all, log_all)) = (run(0), run(u64::MAX));
        crate::resample::assert_batches(&log_one, 8, 1, 1);
        crate::resample::assert_batches(&log_all, 8, 8, 1);
        assert!(
            (0..one.nrows()).any(|j| one[j] != 0.0),
            "premise: nonzero r"
        );
        for j in 0..splits.len() {
            assert_eq!(one[j].to_bits(), all[j].to_bits(), "split {j}");
        }
    }

    /// A split the score gate of `pls3_split_r_streamed` rejects is the
    /// materialized fit's to the bit, on an `r` that is not zero. Column 0
    /// of X carries the shared factor and is shifted by 1e6 on the held-out
    /// rows only, so the training half is ordinary and `σ₁` clears its
    /// floors, while the held-out X score is about 5e5 on every row plus
    /// variation of order one. Its centered norm is about 2e-6 of its
    /// norm: below `dual_route::SCORE_BAND`, so the gate rejects the split,
    /// yet far above the constant-to-rounding test, so the materialized fit
    /// correlates it.
    #[test]
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    fn pls3_split_nb_streamed_hands_a_score_gated_split_to_the_materialized_fit() {
        use crate::linalg::{scaled_moments, standardize_apply_rows, standardize_rows};
        let (n, p, q) = (60_usize, 6_usize, 3_usize);
        let (base, y) = linked_blocks(n, p, q, 2.0, 5);
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
        let o = nb_opts(2, true, 3);

        let (xs_tr, x_mean, x_scale) = standardize_rows(x.as_ref(), &sp.tr, None, None);
        let xs_te =
            standardize_apply_rows(x.as_ref(), &sp.te, x_mean.as_ref(), x_scale.as_ref(), None);
        let (ys_tr, _, _) = standardize_rows(y.as_ref(), &sp.tr, None, None);
        let (u, _) = crate::pls3::pls3_lv1_prestd(
            xs_tr.as_ref(),
            xs_tr.norm_l2(),
            ys_tr.as_ref(),
            None,
            None,
            o.max_iter,
            o.tol,
        )
        .unwrap()
        .expect("premise: the training half keeps LV1");
        let s = crate::linalg::mat_vec(xs_te.as_ref(), u.as_ref(), faer::Par::Seq);
        let ms = scaled_moments(30, |i| s[i], None);
        assert!(
            !ms.is_constant(30) && ms.ss.sqrt() < crate::dual_route::SCORE_BAND * ms.sq.sqrt(),
            "premise: the held-out X score is below the score band but not constant"
        );

        let one = std::slice::from_ref(&sp);
        let want = pls3_split_lv_correlations(x.as_ref(), y.as_ref(), one, &o);
        assert!(
            want[0].is_finite() && want[0] != 0.0,
            "premise: the materialized fit reports a nonzero r, got {}",
            want[0]
        );
        assert!(pls3_split_r_streamed(x.as_ref(), y.as_ref(), &sp).is_none());
        let got = pls3_split_lv_correlations_streamed(x.as_ref(), y.as_ref(), one, &o);
        assert_eq!(got[0].to_bits(), want[0].to_bits());
    }

    // ── `test_method = "auto"` ──────────────────────────────────────────

    /// At one seed `Auto` returns what an explicit call of the resolved
    /// method returns, field by field except `stable_rank`: n = 60 resolves
    /// to `split_exact` on the sample-size clause, n = 300 with eight
    /// mostly-iid X columns to `split_nb`.
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
        for (data, explicit, has_rank) in [
            (
                linked_blocks(60, 6, 3, 1.0, 5),
                ConfirmatoryArgs::SplitExact {
                    n_perm: 19,
                    n_splits: 4,
                },
                false,
            ),
            (
                linked_blocks(300, 8, 3, 0.5, 9),
                ConfirmatoryArgs::SplitNb {
                    n_splits: 4,
                    force: false,
                },
                true,
            ),
        ] {
            let tag = format!("{explicit:?}");
            let mut got = run(&data, auto);
            let mut want = run(&data, explicit);
            assert_eq!(got.test_method, explicit.method().as_str(), "{tag}");
            assert_eq!(got.stable_rank.is_some(), has_rank, "{tag}");
            got.stable_rank = None;
            want.stable_rank = None;
            assert_eq!(format!("{got:?}"), format!("{want:?}"), "{tag}");
        }
    }

    /// The count floors hold under `Auto` on a design that resolves to
    /// `split_nb`, where `n_perm` is otherwise unused.
    #[test]
    fn auto_count_floors_hold_on_a_split_nb_design() {
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

    #[test]
    fn sparse_knobs_are_validated_up_front() {
        // The per-split fits do not re-validate these, so a bad `max_iter`
        // / `tol` must be refused before any split runs, on either route.
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let sparse = |max_iter: usize, tol: f64| Pls3ConfirmatoryTestOpts {
            keep_x: Some(3),
            max_iter,
            tol,
            ..opts(9, 4, 7)
        };
        for o in [sparse(0, 1e-8), sparse(100, f64::NAN), sparse(100, -1.0)] {
            assert!(matches!(
                pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o),
                Err(PlsKitError::InvalidArgument(_))
            ));
        }
    }

    // ── split_exact dual (Gram) route tests ─────────────────────────────

    /// The equivalence test the PLS3 dual route rests on. Route A refits
    /// PLS3 honestly on each training half for every replicate column
    /// (`pls3_split_lv_correlations`, the shipped primal path); route B is
    /// `dual_route::pls3_split_zbars_columns`. Both get the same splits and
    /// the same permutations, so every one of the B+1 z̄ columns must agree.
    #[allow(clippy::many_single_char_names)]
    fn assert_pls3_split_exact_dual_route_matches_honest_refit(p: usize, keep_y: Option<usize>) {
        use crate::signal_test::{draw_splits, mean_fisher_z};

        // n=60 ⇒ n_tr=30; q=3, B+1=50 ⇒ Bq=150. Dual is live when
        // 30·(150+p) < p·150, which the `dual_route_eligible` assert
        // below checks.
        let (n, q) = (60_usize, 3_usize);
        let (n_perm, n_splits, seed) = (49_usize, 6_usize, 11_u64);
        let (x, y) = linked_blocks(n, p, q, 3.0, 7);
        let n_cols = n_perm + 1;
        let o = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            keep_x: None,
            keep_y,
            ..Pls3ConfirmatoryTestOpts::default()
        };

        // Rebuild the runner's split draw and child seeds off the same seed.
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(n, 1, n_splits, &mut rng).unwrap();
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let perms: Vec<Vec<usize>> = seeds
            .iter()
            .map(|s| crate::resample::permute_indices(n, &mut crate::rng::child_rng(*s)))
            .collect();

        let (n_tr, _) = crate::resample::split_sizes(n, 1);
        assert!(
            dual_route_eligible(None, n_tr, p, n_perm, q),
            "test premise: the runner must send this shape to the dual route \
             (n_tr={n_tr}, p={p}, q={q}, n_perm={n_perm})"
        );

        // Route A: honest per-split, per-column PLS3 refits.
        let mut a = Vec::with_capacity(n_cols);
        a.push(mean_fisher_z(&pls3_split_lv_correlations(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &o,
        )));
        for perm in &perms {
            let y_perm = Mat::<f64>::from_fn(n, q, |i, j| y[(perm[i], j)]);
            a.push(mean_fisher_z(&pls3_split_lv_correlations(
                x.as_ref(),
                y_perm.as_ref(),
                &splits,
                &o,
            )));
        }

        // Route B: the Gram path, its splits one at a time (budget 0) and
        // all in one batch (no limit): the same bits.
        let run = |budget| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(budget, || {
                    crate::dual_route::pls3_split_zbars_columns(
                        x.as_ref(),
                        y.as_ref(),
                        &splits,
                        &perms,
                        keep_y,
                        o.max_iter,
                        o.tol,
                    )
                })
            })
        };
        let ((b, log_one), (b_all, log_all)) = (run(0), run(u64::MAX));
        crate::resample::assert_batches(&log_one, n_splits, 1, 1);
        crate::resample::assert_batches(&log_all, n_splits, n_splits, 1);
        assert_eq!(
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b_all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "keep_y={keep_y:?}: the Gram route's bits depend on the budget"
        );

        assert_eq!(a.len(), b.len());
        for (col, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            let scale = av.abs().max(bv.abs());
            let rel = if scale == 0.0 {
                0.0
            } else {
                (av - bv).abs() / scale
            };
            assert!(
                rel < 1e-10,
                "keep_y={keep_y:?}, col {col}: primal={av} dual={bv} rel={rel}"
            );
        }

        let count = |v: &[f64]| {
            v[1..]
                .iter()
                .filter(|z| !z.is_finite() || **z >= v[0])
                .count()
        };
        assert_eq!(
            count(&a),
            count(&b),
            "keep_y={keep_y:?}: exceedance counts split on a tie"
        );
    }

    #[test]
    fn pls3_split_exact_dual_route_matches_honest_refit_when_selected() {
        for keep_y in [
            None,
            // Sparse `v`: the Gram path equals the primal refit.
            Some(2),
            // `ky == q` takes the dual route's dense arm.
            Some(3),
        ] {
            assert_pls3_split_exact_dual_route_matches_honest_refit(100, keep_y);
        }
    }

    #[test]
    fn pls3_split_exact_dual_route_matches_honest_refit_over_several_blocks() {
        // 30-row halves give a block of 8,736 columns: p=20,000 is three.
        assert_eq!(crate::linalg::stream_block_width(30), 8736);
        assert_pls3_split_exact_dual_route_matches_honest_refit(20_000, None);
        assert_pls3_split_exact_dual_route_matches_honest_refit(20_000, Some(2));
    }

    /// A sparse X side removes the Gram route from the menu, and the
    /// runner must then reproduce its own primal branch exactly. If the
    /// refusal were dropped, the dual kernel would answer with a dense `u`
    /// and neither the statistic nor the p-value would match.
    // The p-value is a count over a fixed denominator and must match
    // exactly; see the `float_cmp` note on the wiring test below.
    #[allow(clippy::float_cmp)]
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn sparse_x_side_forces_the_primal_route_in_the_runner() {
        use crate::signal_test::{draw_splits, mean_fisher_z};

        let (n, p, q) = (60_usize, 100_usize, 3_usize);
        // Same counts as the dense equivalence case, whose shape the
        // routing rule sends dual.
        let (n_perm, n_splits, seed) = (49_usize, 6_usize, 11_u64);
        let (x, y) = linked_blocks(n, p, q, 3.0, 7);

        let (n_tr, _) = crate::resample::split_sizes(n, 1);
        assert!(
            crate::dual_route::use_dual_route(n_tr, p, n_perm + 1, q),
            "test premise: the shape alone would take the dual route"
        );

        // The runner's own decision function, not a copy of it, so a change
        // to the rule fails here. The two boundaries are pinned side by
        // side: `keep_x = Some(p)` is the dense endpoint and leaves the
        // route free, `keep_x = Some(kx < p)` takes it off the menu.
        assert!(
            dual_route_eligible(Some(p), n_tr, p, n_perm, q),
            "keep_x = Some(n_features) is the dense endpoint: the route must stay free"
        );
        assert!(
            dual_route_eligible(None, n_tr, p, n_perm, q),
            "keep_x = None must decide exactly as it always has"
        );
        assert!(
            !dual_route_eligible(Some(10), n_tr, p, n_perm, q),
            "keep_x below n_features must force the primal route"
        );

        let o = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            keep_x: Some(10),
            keep_y: Some(2),
            ..Pls3ConfirmatoryTestOpts::default()
        };

        // Reference: the runner's primal branch, statement for statement.
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(n, 1, n_splits, &mut rng).unwrap();
        let z_obs = mean_fisher_z(&pls3_split_lv_correlations(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &o,
        ));
        let nulls = crate::resample::parallel_for_each_seeded(&mut rng, n_perm, |_, child| {
            let perm = crate::resample::permute_indices(n, child);
            let y_perm = Mat::<f64>::from_fn(n, q, |i, j| y[(perm[i], j)]);
            mean_fisher_z(&pls3_split_lv_correlations(
                x.as_ref(),
                y_perm.as_ref(),
                &splits,
                &o,
            ))
        });
        #[allow(clippy::cast_precision_loss)]
        let expected_p = (nulls
            .iter()
            .filter(|v| !v.is_finite() || **v >= z_obs)
            .count() as f64
            + 1.0)
            / (n_perm as f64 + 1.0);

        let out = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, o).unwrap();
        assert_eq!(out.pvalue, expected_p);
        assert_eq!(out.statistic.to_bits(), z_obs.tanh().to_bits());
    }

    /// The primal route's split driver on the inputs of
    /// `sparse_x_side_forces_the_primal_route_in_the_runner` (six splits, a
    /// sparse X side), its splits one at a time (budget 0) and all in one
    /// batch (no limit): the same bits, column 0 being the observed z̄ of the
    /// split-by-split reference.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn primal_split_driver_gives_the_same_bits_at_every_budget() {
        use crate::signal_test::{draw_splits, mean_fisher_z};
        let (n, p, q) = (60_usize, 100_usize, 3_usize);
        let (n_perm, n_splits, seed) = (49_usize, 6_usize, 11_u64);
        let (x, y) = linked_blocks(n, p, q, 3.0, 7);
        let o = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            keep_x: Some(10),
            keep_y: Some(2),
            ..Pls3ConfirmatoryTestOpts::default()
        };
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(n, 1, n_splits, &mut rng).unwrap();
        let perms: Vec<Vec<usize>> = crate::rng::child_seeds(&mut rng, n_perm)
            .into_iter()
            .map(|s| crate::resample::permutation_from_seed(n, s))
            .collect();
        let run = |b| {
            crate::resample::with_batch_log(|| {
                crate::fit::with_budget(b, || {
                    pls3_split_zbars_columns_primal(x.as_ref(), y.as_ref(), &splits, &perms, &o)
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
        let z_obs = mean_fisher_z(&pls3_split_lv_correlations(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &o,
        ));
        assert_eq!(one[0].to_bits(), z_obs.to_bits());
    }

    /// A training half whose `Ỹ_tr` is orthogonal to `X̃_tr` up to rounding
    /// has no first component: `σ₁` of `X̃_tr'Ỹ_tr` is rounding noise under
    /// the relative floor, so the primal fit keeps nothing and `r = 0`. The
    /// Gram route cannot see that (its `λ_max` carries rounding many orders
    /// above the floor), so it must hand such a column to the primal rather
    /// than correlate a noise eigenvector. X is rank 3 with `p ≫ n_tr` (the
    /// only way a training `Ỹ` can be orthogonal to it there), and column 0's
    /// training rows are residualized on `[1, B_tr]`; the permutation columns
    /// are ordinary and stay on the Gram formula.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_dual_route_hands_an_orthogonal_training_half_to_the_primal() {
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
        let splits = vec![SplitIdx { tr, te }];
        let perms: Vec<Vec<usize>> = (0..9)
            .map(|_| crate::resample::permute_indices(n, &mut rng))
            .collect();
        assert!(crate::dual_route::use_dual_route(
            n / 2,
            p,
            perms.len() + 1,
            q
        ));

        for keep_y in [None, Some(2)] {
            let o = Pls3ConfirmatoryTestOpts {
                keep_y,
                ..Pls3ConfirmatoryTestOpts::default()
            };
            let primal =
                pls3_split_zbars_columns_primal(x.as_ref(), y.as_ref(), &splits, &perms, &o);
            let dual = crate::dual_route::pls3_split_zbars_columns(
                x.as_ref(),
                y.as_ref(),
                &splits,
                &perms,
                keep_y,
                o.max_iter,
                o.tol,
            );
            // Premise: the primal kept no component on column 0, and did on
            // the others.
            assert_eq!(primal[0].to_bits(), 0.0_f64.to_bits(), "keep_y={keep_y:?}");
            assert!(primal[1..].iter().all(|z| *z != 0.0), "keep_y={keep_y:?}");
            assert_eq!(
                dual[0].to_bits(),
                primal[0].to_bits(),
                "keep_y={keep_y:?}: dual={} primal={}",
                dual[0],
                primal[0]
            );
            for col in 1..dual.len() {
                let (a, d) = (primal[col], dual[col]);
                let rel = (a - d).abs() / a.abs().max(d.abs());
                assert!(
                    rel < 1e-10,
                    "keep_y={keep_y:?} col {col}: primal={a} dual={d}"
                );
            }
        }
    }

    /// Constant X drives every split's `X̃_tr` to the exact zero matrix, so
    /// `G` and `M` are the exact zero matrix too. On a fully degenerate X,
    /// both routes must agree and report exactly zero rather than NaN: the
    /// dual side reaches it because `M` is zero (independent of whatever
    /// eigenvector the degenerate eigendecomposition returns), the primal
    /// side through its own `k_used == 0` branch. The comparison against
    /// the honest per-split refit below checks that route agreement.
    ///
    /// This test does not pin the primal fallback of the Gram route's
    /// truncation gate: with `‖X̃_tr‖_F = 0` every comparison of the gate
    /// reads `0 >= 0`, so this `λ_max = 0` column passes the gate and stays
    /// on the dual formula, and `M` being exactly zero gives it `r = 0`, the
    /// primal route's value. With a sparse `keep_y`, every column stops at
    /// the keep-boundary band and is computed by the second pass, so the
    /// "stays on the dual formula" text holds only for `keep_y = None`.
    /// `pls3_dual_route_hands_an_orthogonal_training_half_to_the_primal` is
    /// the test that pins the gate.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_dual_route_degenerate_x_gives_zero_not_nan() {
        use crate::signal_test::{draw_splits, mean_fisher_z};
        // 6 splits of 30 training rows at p = 5 and p = 400, one at a time
        // (budget 0) and all in one batch (no limit), with the same bits.
        for p in [5_usize, 400] {
            let x = Mat::<f64>::from_fn(60, p, |_, _| 1.25);
            let (_, y) = linked_blocks(60, p, 3, 3.0, 7);
            for keep_y in [None, Some(2_usize)] {
                let (_, mut rng) = crate::rng::resolve_seed(Some(4)).unwrap();
                let splits = draw_splits(60, 1, 6, &mut rng).unwrap();
                let perms: Vec<Vec<usize>> = vec![crate::resample::permute_indices(60, &mut rng)];
                let run = |b| {
                    crate::resample::with_batch_log(|| {
                        crate::fit::with_budget(b, || {
                            crate::dual_route::pls3_split_zbars_columns(
                                x.as_ref(),
                                y.as_ref(),
                                &splits,
                                &perms,
                                keep_y,
                                100,
                                1e-8,
                            )
                        })
                    })
                };
                let ((z, log_one), (all, log_all)) = (run(0), run(u64::MAX));
                // With a sparse `keep_y` every column goes to the second
                // pass, so both passes run over the six splits.
                let passes = if keep_y.is_some() { 2 } else { 1 };
                crate::resample::assert_batches(&log_one, 6, 1, passes);
                crate::resample::assert_batches(&log_all, 6, 6, passes);
                assert_eq!(
                    z.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "p={p} keep_y={keep_y:?}"
                );
                for (col, v) in z.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "p={p} keep_y={keep_y:?}: col {col} not finite: {v}"
                    );
                    assert!(v.abs() < 1e-12, "p={p} keep_y={keep_y:?}: col {col}: {v}");
                }

                // Route agreement, not just finiteness: the same honest per-split
                // refit the equivalence test uses, on the same splits and the same
                // permutation.
                let o = Pls3ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 1,
                        n_splits: 6,
                    },
                    seed: Some(4),
                    keep_y,
                    ..Pls3ConfirmatoryTestOpts::default()
                };
                let mut a = Vec::with_capacity(z.len());
                a.push(mean_fisher_z(&pls3_split_lv_correlations(
                    x.as_ref(),
                    y.as_ref(),
                    &splits,
                    &o,
                )));
                for perm in &perms {
                    let y_perm = Mat::<f64>::from_fn(60, 3, |i, j| y[(perm[i], j)]);
                    a.push(mean_fisher_z(&pls3_split_lv_correlations(
                        x.as_ref(),
                        y_perm.as_ref(),
                        &splits,
                        &o,
                    )));
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
                        "p={p} keep_y={keep_y:?}: col {col}: primal={av} dual={bv} rel={rel}"
                    );
                }
            }
        }
    }

    /// Wiring test: the public runner, on an input the rule sends *dual*,
    /// must return exactly the p-value and statistic the primal branch
    /// would have returned at the same seed.
    ///
    /// The kernel-level equivalence test hands both routes the same
    /// locally-built permutations, so it cannot catch a column-0 convention
    /// slip or a child-seed derivation that drifts from what
    /// `parallel_for_each_seeded` actually draws. Here the reference comes
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

        let (n_tr, _) = crate::resample::split_sizes(n, 1);
        assert!(
            crate::dual_route::use_dual_route(n_tr, p, n_perm + 1, q),
            "test premise: this shape must take the dual route"
        );

        let o = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            ..Pls3ConfirmatoryTestOpts::default()
        };

        // Reference: the runner's pre-branch split draw, then its primal
        // branch, statement for statement.
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(n, 1, n_splits, &mut rng).unwrap();
        let z_obs = mean_fisher_z(&pls3_split_lv_correlations(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &o,
        ));
        let nulls = crate::resample::parallel_for_each_seeded(&mut rng, n_perm, |_, child| {
            let perm = crate::resample::permute_indices(n, child);
            let y_perm = Mat::<f64>::from_fn(n, q, |i, j| y[(perm[i], j)]);
            mean_fisher_z(&pls3_split_lv_correlations(
                x.as_ref(),
                y_perm.as_ref(),
                &splits,
                &o,
            ))
        });
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
        let scale = r.statistic.abs().max(expected_stat.abs());
        let rel = if scale == 0.0 {
            0.0
        } else {
            (r.statistic - expected_stat).abs() / scale
        };
        assert!(
            rel < 1e-10,
            "statistic: dual={} primal={expected_stat} rel={rel}",
            r.statistic
        );

        // Dense split_nb reads X block by block; `force` keeps the gate
        // from deciding what runs.
        let nb = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(n_splits, true, seed))
            .unwrap();
        assert_eq!(nb.test_method, "split_nb");
        let rel = (nb.statistic - r.statistic).abs() / r.statistic.abs().max(1e-300);
        assert!(
            rel < 1e-10,
            "split_nb={} split_exact(dual)={} rel={rel}",
            nb.statistic,
            r.statistic
        );
    }

    /// Regression for the dual route's near-tie fallback (see "Near-ties" in
    /// `dual_route::pls3_split_zbars_columns`). Y column 3 loads on the
    /// shared factor with strength `t`; somewhere in `(0, 4)` it overtakes
    /// column 1 at the `keep_y = 2` boundary and the primal support flips.
    /// The flip point is bisected at test time on the primal route's own
    /// support, so it adapts to whatever last bits this platform's faer
    /// produces, and then every `t` within ±32 ulps of it is run through
    /// both routes. Without the fallback the dual route picks the other
    /// support on some of those `t` and `z` separates by 1e-3 to 0.5
    /// (0 to 24 of 129 ulps per seed hit on aarch64-apple-darwin, every
    /// seed in 0..20 hit at least once); with it, every column inside the
    /// band is the primal computation and agrees to the bit.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn dual_route_matches_primal_across_a_keep_y_support_flip() {
        use crate::linalg::{row_subset, standardize};
        use rand::RngExt;
        use rand::SeedableRng;

        const ULPS: u64 = 32;
        let (n, p, q, keep_y, max_iter, tol) = (40usize, 200usize, 4usize, 2usize, 100usize, 1e-8);
        let tr: Vec<usize> = (0..20).collect();
        let splits = vec![SplitIdx {
            tr: tr.clone(),
            te: (20..40).collect(),
        }];
        let o = Pls3ConfirmatoryTestOpts {
            keep_y: Some(keep_y),
            max_iter,
            tol,
            ..Pls3ConfirmatoryTestOpts::default()
        };
        let fit_opts = Pls3FitOpts {
            pre_standardized_x: true,
            pre_standardized_y: true,
            par: crate::fit::ParChoice::Seq,
            max_iter,
            tol,
        };

        for seed in 0..4u64 {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let f: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let x = Mat::<f64>::from_fn(n, p, |i, j| {
                let e = rng.random_range(-1.0..1.0);
                if j < 5 { 1.5 * f[i] + e } else { e }
            });
            let y0 = Mat::<f64>::from_fn(n, q, |_, _| rng.random_range(-1.0..1.0));
            let make_y = |t: f64| {
                Mat::<f64>::from_fn(n, q, |i, j| match j {
                    0 => 2.0 * f[i] + y0[(i, 0)],
                    1 => f[i] + y0[(i, 1)],
                    3 => t * f[i] + y0[(i, 3)],
                    _ => y0[(i, j)],
                })
            };
            let (xs, _, _) = standardize(row_subset(x.as_ref(), &tr).as_ref());
            let primal_support = |t: f64| -> Vec<bool> {
                let (ys, _, _) = standardize(row_subset(make_y(t).as_ref(), &tr).as_ref());
                let m = spls3_fit(xs.as_ref(), ys.as_ref(), 1, p, keep_y, None, fit_opts).unwrap();
                (0..q).map(|j| m.v_saliences[(j, 0)] != 0.0).collect()
            };

            let (mut lo, mut hi) = (0.0_f64, 4.0_f64);
            let s_lo = primal_support(lo);
            assert_ne!(
                primal_support(hi),
                s_lo,
                "seed {seed}: premise, no support flip in (0, 4)"
            );
            loop {
                let mid = f64::midpoint(lo, hi);
                if mid <= lo || mid >= hi {
                    break;
                }
                if primal_support(mid) == s_lo {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            let mut t = f64::from_bits(lo.to_bits() - ULPS);
            let (mut z_min, mut z_max) = (f64::INFINITY, f64::NEG_INFINITY);
            for _ in 0..=2 * ULPS {
                let y = make_y(t);
                let a =
                    pls3_split_zbars_columns_primal(x.as_ref(), y.as_ref(), &splits, &[], &o)[0];
                let b = crate::dual_route::pls3_split_zbars_columns(
                    x.as_ref(),
                    y.as_ref(),
                    &splits,
                    &[],
                    Some(keep_y),
                    max_iter,
                    tol,
                )[0];
                // Every `t` of the window takes the fallback: it sits within
                // 33 ulps (a relative 8e-15 at most) of the point where the
                // two magnitudes at the keep boundary cross, and the route's
                // selection band is a relative 1e-10. The fallback refits on
                // the halves the route materializes for it, so its value is
                // the primal route's to the bit.
                assert!(a.is_finite(), "seed {seed}, t = {t:e}: primal = {a}");
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "seed {seed}, t = {t:e}: primal = {a}, dual = {b}"
                );
                z_min = z_min.min(a);
                z_max = z_max.max(a);
                t = f64::from_bits(t.to_bits() + 1);
            }
            // The window straddles a real discontinuity, not a smooth stretch.
            assert!(
                z_max - z_min > 1e-4,
                "seed {seed}: premise, z jump {:e}",
                z_max - z_min
            );
        }
    }

    /// The open case in `dual_route::pls3_split_zbars_columns`' near-tie
    /// doc: a nearly degenerate *restricted* spectrum. The bands guard the
    /// full spectrum (`GAP_BAND`), the support boundary (`SEL_BAND`) and
    /// the stopping sweep (`STOP_BAND_*`), but not how well separated the
    /// top two eigenvalues of `C` are once restricted to the selected
    /// support, which is what the alternation actually iterates on.
    ///
    /// The design pins that restriction exactly. On the one training half,
    /// `X̃` has orthonormal-times-`√n` columns and each `Ỹ` column is
    /// `√n·(X̃-part + orthogonal noise)` with unit variance, so
    /// `A = X̃'Ỹ = n·B` for a chosen `B` and `C = A'A = n²·B'B`:
    ///
    /// ```text
    /// B'B = [ β²      0          γβ          ]
    ///       [ 0       β²(1+δ)    γβ√(1+δ)    ]
    ///       [ γβ      γβ√(1+δ)   2γ²         ]
    /// ```
    ///
    /// With `β = 0.9`, `γ = 0.2` the full spectrum has a relative top gap
    /// near 9% (far outside `GAP_BAND`), the initializer is about
    /// `(1, 1, 0.45)`, and `keep_y = 2` selects `{y₁, y₂}`, on which `C`
    /// restricts to `n²β²·diag(1, 1 + δ)`: relative gap `δ`. The alternation
    /// is then power iteration at rate `1/(1 + δ)`, and with `tol = 0` it
    /// runs every one of `max_iter` sweeps, the regime where the two
    /// routes' per-sweep rounding (different floats for `C v` vs `A'(Av)`)
    /// would have to accumulate or amplify to separate them. `max_iter·δ ≤
    /// 0.3` keeps `y₃` out of the support throughout (it would enter once
    /// `v₁/v₂` fell below about `0.29`), so no `SEL_BAND` or support flip is
    /// involved; `δ = 0` with the default `tol` is the exactly degenerate
    /// case, which stops on the second sweep.
    ///
    /// Without the drift gate there is no `1/δ` amplification, but there is
    /// accumulation. The ungated routes' `|Δz|` grows
    /// linearly in the sweep count at about `2.5e-16` per sweep, the same
    /// for `δ = 1e-6` and `1e-9` (3e-14 at 100 sweeps, 2.8e-13 at 1e3,
    /// 2.8e-12 at 1e4, 2.8e-11 at 1e5; aarch64-apple-darwin), past the
    /// corpus `1e-12` beyond about 4000 sweeps. A near-degenerate
    /// restricted map damps a per-sweep discrepancy only at rate `δ`, so it
    /// builds up as about `ε·min(sweeps, 1/δ)`; `DRIFT_SWEEPS` in
    /// `dual_route::pls3_split_zbars_columns` bounds that by falling
    /// back to the primal. Short runs (the gate is not evaluated) must
    /// agree to `1e-12`; long near-degenerate runs must fall back, which is
    /// checked to the bit.
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    #[allow(clippy::too_many_lines)]
    #[test]
    fn dual_route_matches_primal_on_a_near_degenerate_restricted_spectrum() {
        use crate::linalg::{row_subset, standardize};
        use rand::RngExt;
        use rand::SeedableRng;

        let (n_tr, n_te, p, q, keep_y) = (30usize, 20usize, 3usize, 3usize, 2usize);
        let n = n_tr + n_te;
        let (beta, gamma) = (0.9_f64, 0.2_f64);
        let tr: Vec<usize> = (0..n_tr).collect();
        let splits = vec![SplitIdx {
            tr: tr.clone(),
            te: (n_tr..n).collect(),
        }];

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        let raw = Mat::<f64>::from_fn(n_tr, p + q, |_, _| rng.random_range(-1.0..1.0));
        let f = crate::test_support::orthonormal_basis(
            Col::<f64>::from_fn(n_tr, |_| 1.0).as_ref(),
            raw.as_ref(),
            0.0,
        );
        assert_eq!(f.len(), 1 + p + q, "premise: full-rank basis");
        #[allow(clippy::cast_precision_loss)]
        let rt = (n_tr as f64).sqrt();
        let te_x = Mat::<f64>::from_fn(n_te, p, |_, _| rng.random_range(-1.0..1.0));
        let te_y = Mat::<f64>::from_fn(n_te, q, |_, _| rng.random_range(-1.0..1.0));
        let x = Mat::<f64>::from_fn(n, p, |i, k| {
            if i < n_tr {
                rt * f[1 + k][i]
            } else {
                te_x[(i - n_tr, k)]
            }
        });

        // `(δ, max_iter, tol)`. With `tol = 0` every sweep runs, so the
        // drift gate fires exactly when `max_iter` and `1/δ` both exceed
        // `DRIFT_SWEEPS` (400).
        for (delta, max_iter, tol) in [
            (1e-3_f64, 300usize, 0.0_f64),
            (1e-6, 100, 0.0),
            (0.0, 100, 1e-8),
            (1e-3, 1000, 0.0),
            (1e-6, 100_000, 0.0),
            (1e-9, 100_000, 0.0),
        ] {
            let b: [[f64; 3]; 3] = [
                [beta, 0.0, 0.0],
                [0.0, beta * (1.0 + delta).sqrt(), 0.0],
                [gamma, gamma, 0.0],
            ];
            let y = Mat::<f64>::from_fn(n, q, |i, j| {
                if i < n_tr {
                    let b2: f64 = b[j].iter().map(|v| v * v).sum();
                    let noise = (1.0 - b2).sqrt();
                    rt * ((0..p).map(|k| b[j][k] * f[1 + k][i]).sum::<f64>()
                        + noise * f[1 + p + j][i])
                } else {
                    te_y[(i - n_tr, j)]
                }
            });

            // Premises, on the blocks both routes actually see.
            let (xs, _, _) = standardize(row_subset(x.as_ref(), &tr).as_ref());
            let (ys, _, _) = standardize(row_subset(y.as_ref(), &tr).as_ref());
            let a: Mat<f64> = xs.transpose() * ys.as_ref();
            let c: Mat<f64> = a.transpose() * a.as_ref();
            let rel_gap = |m: MatRef<'_, f64>| {
                let s = m.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let l = s.S().column_vector();
                let k = l.nrows();
                (l[k - 1] - l[k - 2]) / l[k - 1]
            };
            let full_gap = rel_gap(c.as_ref());
            assert!(
                full_gap > 1e-2,
                "δ={delta:e}: premise, full gap {full_gap:e}"
            );
            let restricted_gap = rel_gap(c.as_ref().submatrix(0, 0, 2, 2));
            assert!(
                (restricted_gap - delta).abs() <= 1e-3 * delta + 1e-12,
                "δ={delta:e}: premise, restricted gap {restricted_gap:e}"
            );
            let fit_opts = Pls3FitOpts {
                pre_standardized_x: true,
                pre_standardized_y: true,
                par: crate::fit::ParChoice::Seq,
                max_iter,
                tol,
            };
            let m = spls3_fit(xs.as_ref(), ys.as_ref(), 1, p, keep_y, None, fit_opts).unwrap();
            assert_eq!(
                (0..q)
                    .map(|j| m.v_saliences[(j, 0)] != 0.0)
                    .collect::<Vec<_>>(),
                vec![true, true, false],
                "δ={delta:e}: premise, support stays {{y₁, y₂}}"
            );

            let o = Pls3ConfirmatoryTestOpts {
                keep_y: Some(keep_y),
                max_iter,
                tol,
                ..Pls3ConfirmatoryTestOpts::default()
            };
            let za = pls3_split_zbars_columns_primal(x.as_ref(), y.as_ref(), &splits, &[], &o)[0];
            let zb = crate::dual_route::pls3_split_zbars_columns(
                x.as_ref(),
                y.as_ref(),
                &splits,
                &[],
                Some(keep_y),
                max_iter,
                tol,
            )[0];
            let gate_fires = max_iter > 400 && delta < 1.0 / 400.0;
            if gate_fires {
                // The fallback is the primal computation, to the bit.
                assert_eq!(
                    za.to_bits(),
                    zb.to_bits(),
                    "δ={delta:e}, max_iter={max_iter}: drift gate did not fall back \
                     (primal={za} dual={zb})"
                );
            } else {
                // The Gram arm's own answer, held to the corpus's scalar
                // tolerance: agreement at that level is what keeps the
                // route choice unobservable.
                assert!(
                    (za - zb).abs() < 1e-12,
                    "δ={delta:e}, max_iter={max_iter}: primal={za} dual={zb}"
                );
            }
        }
    }

    /// A split on the primal route counts its standardized halves and one
    /// LV1 fit's buffers per replicate column that can run at once: one per
    /// thread, at most the column count. A budget of two such units runs
    /// two splits at once, on a pool of 1 thread and on a pool of 4, for
    /// `split_exact`'s splits (8 columns) and for `split_nb`'s (1 column).
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn primal_splits_count_one_lv1_fit_per_running_column() {
        use crate::signal_test::{draw_splits, split_blocks_bytes};
        let (n, p, q) = (40_usize, 9_usize, 3_usize);
        let (x, y) = linked_blocks(n, p, q, 2.0, 5);
        let (_, mut rng) = crate::rng::resolve_seed(Some(3)).unwrap();
        let splits = draw_splits(n, 1, 4, &mut rng).unwrap();
        let perms: Vec<Vec<usize>> = (0..7)
            .map(|b| crate::resample::permutation_from_seed(n, b))
            .collect();
        let o = Pls3ConfirmatoryTestOpts {
            keep_x: Some(4),
            ..opts(7, 4, 3)
        };
        let lv1 = crate::pls3::lv1_bytes(p, q);
        let budget =
            |live: usize| u64::try_from(2 * (split_blocks_bytes(&splits, p) + live * lv1)).unwrap();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (_, log) = pool.install(|| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(budget(threads), || {
                        pls3_split_zbars_columns_primal(x.as_ref(), y.as_ref(), &splits, &perms, &o)
                    })
                })
            });
            crate::resample::assert_batches(&log, 4, 2, 1);
            let (_, log) = pool.install(|| {
                crate::resample::with_batch_log(|| {
                    crate::fit::with_budget(budget(1), || {
                        pls3_split_lv_correlations(x.as_ref(), y.as_ref(), &splits, &o)
                    })
                })
            });
            crate::resample::assert_batches(&log, 4, 2, 1);
        }
    }
}
