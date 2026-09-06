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
//! enters here — unlike the per-coordinate CIs (feature-matrix §2 notes).
//!
//! # `split_nb` here versus for PLS1
//! Both sides of the correlation are directions estimated on the training
//! half, where PLS1 has an observed outcome on one side. That costs the NB
//! reference nothing: conditional on the training half, `X̃_te u₁` and
//! `Ỹ_te v₁` are fixed linear combinations of test-half rows, the test half
//! is independent of the training half, and under block independence the two
//! projections are independent — so the conditional null of `r` is the
//! ordinary null correlation law. PLS1 is the same construction with
//! `v₁ = 1`. Measured on iid Gaussian, heavy-tailed, low-stable-rank and
//! real two-block designs, the ruler holds (`atanh(r)·√(n_test−3)` has unit
//! spread and no excess kurtosis) and the resulting test is conservative,
//! never anti-conservative. `split_exact` stays the recommendation because
//! it holds its level on any design; `split_nb` is the cheap opt-in.
//!
//! # Why only at k = 1
//! Above k = 1 the training-half component *ordering* need not survive to
//! the test half when singular values are close, and whether the statistic
//! should then be per-component or subspace-level is undecided — so k > 1
//! errors rather than guessing.

use faer::{Col, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::pls3::{pls3_fit, Pls3FitOpts};
use crate::signal_test::{
    draw_splits, mean_fisher_z, nb_rho_hat, nb_test, split_nb_gate_rule, ConfirmatoryArgs,
    ConfirmatoryMethod, ConfirmatoryTestOutput, SplitIdx,
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
    /// Method dispatch + per-method args. [`ConfirmatoryArgs::SplitExact`]
    /// and [`ConfirmatoryArgs::SplitNb`] are accepted; the other three
    /// variants error.
    pub args: ConfirmatoryArgs,
    /// Caller asserts X is already standardized.
    ///
    /// **No effect on either method** — `pls3_split_lv_correlations`
    /// re-standardizes every training half with that half's own moments by
    /// design, so there is nothing for the flag to skip. The `split_nb`
    /// auto-gate likewise standardizes its own copy regardless. It is kept on the
    /// struct so the signature does not change when a method that reads it
    /// lands. PLS1's split path does the same —
    /// `signal_test::split_half_correlations` standardizes each half and
    /// passes `pre_standardized: true` inward regardless of the caller's
    /// flag.
    pub pre_standardized_x: bool,
    /// Caller asserts Y is already standardized. **No effect on either
    /// method** — see `pre_standardized_x`.
    pub pre_standardized_y: bool,
    /// RNG seed; `None` draws from OS entropy and the drawn value is
    /// recorded on the result.
    pub seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution).
    pub disable_parallelism: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
}

impl Default for Pls3ConfirmatoryTestOpts {
    fn default() -> Self {
        Self {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 1000,
                n_splits: 50,
            },
            pre_standardized_x: false,
            pre_standardized_y: false,
            seed: None,
            disable_parallelism: false,
            verbose: false,
        }
    }
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
/// - `PlsKitError::InvalidArgument` for `k != 1`, for any method other than
///   `split_exact` or `split_nb`, for `n_splits < 2`, for `n_perm < 1`, or
///   for `n < k + 5` (the split floor `draw_splits` enforces)
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
    let n = x.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x.ncols()),
            y: y.nrows(),
        });
    }
    crate::fit::check_finite_mat(x)?;
    crate::fit::check_finite_mat(y)?;

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
    // loop, keeps a 0-column X or Y from silently degrading every split's
    // fit to `KExceedsMax { k: 1, k_max: 0 }` and being swallowed by
    // `pls3_split_lv_correlations`'s per-split fail-soft convention below —
    // that convention is for genuine per-split degeneracy, not malformed
    // input reaching every split alike.
    let k_max = x.ncols().min(y.ncols());
    if k > k_max {
        return Err(PlsKitError::KExceedsMax { k, k_max });
    }

    // `n_perm` doubles as the mode flag from here down: `Some` means a
    // permutation reference runs — either because split_exact was asked for,
    // or because the gate below rerouted split_nb into it — and `None` means
    // the NB t-reference runs.
    let (mut n_perm, n_splits, force) = match opts.args {
        ConfirmatoryArgs::SplitExact { n_perm, n_splits } => (Some(n_perm), n_splits, false),
        ConfirmatoryArgs::SplitNb { n_splits, force } => (None, n_splits, force),
        other => {
            return Err(PlsKitError::InvalidArgument(format!(
                "pls3_confirmatory_test supports method='split_exact' or 'split_nb' (got \
                 '{}'): 'raw_perm' needs a CV statistic PLS3 does not have (there is no \
                 pls3_predict), and 'score' / 'e' have no symmetric formulation",
                other.method().as_str()
            )))
        }
    };
    // Same per-method count floors split_exact enforces in
    // `signal_test::confirmatory_test_impl` — change together.
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

    // The `split_nb` auto-gate, on X only. Y is deliberately never gated: q
    // is 3-10 in ordinary PLSC use, so a stable-rank floor of 3 on Y would
    // fire on nearly every design. The thresholds are PLS1's, calibrated on
    // single-block designs and not re-derived for a two-block one — see
    // `signal_test::SPLIT_NB_GATE_MIN_N_EFF` for their provenance.
    let mut stable_rank_out: Option<f64> = None;
    if n_perm.is_none() {
        // Standardizes its own copy unconditionally, including under
        // `pre_standardized_x`: re-standardizing an already-standardized
        // matrix is the identity up to fp rounding, and stable rank is
        // scale-invariant on top of that, so the extra sweep cannot move the
        // decision. Mirrors the same choice in
        // `signal_test::confirmatory_test_impl`.
        let (xs, _, _) = crate::linalg::standardize_weighted(x, None);
        // No weights on this family, so Kish n_eff is exactly the row count.
        let (fires, sr) = split_nb_gate_rule(xs.as_ref(), n as f64);
        stable_rank_out = Some(sr);
        if !force && fires {
            // 1000 is split_exact's own default; the requested `n_splits`
            // carries over untouched. Mirrors the reroute in
            // `signal_test::confirmatory_test_impl`, the sequence-level
            // reroute in `sequential::run_incremental_sequence`, and
            // `_REROUTE_FALLBACK_N_PERM` in plskit-py/python/plskit/_api.py —
            // all four change together.
            n_perm = Some(1000);
        }
    }

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;

    // The J splits are drawn once and held fixed across all B permutation
    // replicates: redrawing per replicate would fold split-to-split scatter
    // into the null (the bug `run_split_perm` was fixed for in 0.4.0).
    let splits = draw_splits(n, k, n_splits, opts.disable_parallelism, &mut rng)?;

    let (n_tr, n_te) = crate::resample::split_sizes(n, k);

    let Some(n_perm) = n_perm else {
        // No permutation loop: the J held-out correlations go straight to the
        // NB t-reference. The statistic is the one split_exact reports —
        // `tanh(z̄)` over the same fixed splits — so only the reference
        // changes, and at a shared seed the two methods agree on it exactly.
        let r_obs = pls3_split_lv_correlations(x, y, &splits, &opts);
        let (p, statistic, _t, _df) = nb_test(&r_obs, n_tr, n_te);
        return Ok(ConfirmatoryTestOutput {
            pvalue: p,
            statistic,
            method: ConfirmatoryMethod::SplitNb.as_str().to_owned(),
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

    // Route choice: internal, silent, decided from shape alone. Both routes
    // get the same splits and the same permutations, so the choice is
    // invisible in the output up to the crate's bit-near tolerance.
    let dual = crate::dual_route::use_dual_route(n_tr, x.ncols(), n_perm + 1, y.ncols());

    let (z_bar_obs, null_zbars) = if dual {
        // Re-derive exactly the child-seed sequence
        // `parallel_for_each_seeded` would draw below, so the two routes
        // permute identically at the same seed.
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let perms: Vec<Vec<usize>> = seeds
            .iter()
            .map(|s| crate::resample::permute_indices(n, &mut crate::rng::child_rng(*s)))
            .collect();
        let z = crate::dual_route::pls3_split_zbars_columns(
            x,
            y,
            &splits,
            &perms,
            opts.disable_parallelism,
        );
        (z[0], z[1..].to_vec())
    } else {
        let r_obs = pls3_split_lv_correlations(x, y, &splits, &opts);
        let z_obs = mean_fisher_z(&r_obs);
        let nulls = crate::resample::parallel_for_each_seeded(
            &mut rng,
            n_perm,
            opts.disable_parallelism,
            |_, child| {
                // Permute the ROWS of Y as units against X: that breaks the
                // cross-block association while leaving Y's within-row
                // structure intact. A row permutation reorders each column, so
                // the full matrix's column moments are unchanged — but nothing
                // here depends on that: every split half standardizes on its
                // own rows, and those half-level moments do move with the
                // permutation.
                let perm = crate::resample::permute_indices(n, child);
                let y_perm = Mat::<f64>::from_fn(n, y.ncols(), |i, j| y[(perm[i], j)]);
                let r_null = pls3_split_lv_correlations(x, y_perm.as_ref(), &splits, &opts);
                mean_fisher_z(&r_null)
            },
        );
        (z_obs, nulls)
    };

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
        method: ConfirmatoryMethod::SplitExact.as_str().to_owned(),
        k,
        n_perm: Some(n_perm),
        n_splits: Some(n_splits),
        seed: seed_used,
        ci: None,
        n_eff,
        rho_hat: None,
        // `Some` only when split_nb was requested and rerouted here — it is
        // what the gate saw.
        stable_rank: stable_rank_out,
    })
}

/// Held-out LV correlation on each supplied split, computed from an honest
/// per-split PLS3 refit (the primal route).
///
/// Takes the splits rather than drawing them so the permutation loop can
/// hold one set fixed across all B replicates. Nothing here consumes
/// randomness — given `(x, y, split)` the fit and the r are deterministic —
/// so the J splits map in parallel directly rather than through
/// `parallel_for_each_seeded`, exactly as `split_half_correlations` does.
///
/// Both halves of both blocks are standardized with **training-half**
/// moments: `Ỹ_te v₁` mixes Y columns, so the test half must sit on the
/// scale `v₁` was estimated on.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn pls3_split_lv_correlations(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    splits: &[SplitIdx],
    opts: &Pls3ConfirmatoryTestOpts,
) -> Col<f64> {
    use crate::linalg::{row_subset, standardize, standardize_apply};

    let per_split = |sp: &SplitIdx| -> f64 {
        let (tr, te) = (sp.tr.as_slice(), sp.te.as_slice());
        let x_tr = row_subset(x, tr);
        let x_te = row_subset(x, te);
        let y_tr = row_subset(y, tr);
        let y_te = row_subset(y, te);

        let (xs_tr, x_mean, x_scale) = standardize(x_tr.as_ref());
        let xs_te = standardize_apply(x_te.as_ref(), x_mean.as_ref(), x_scale.as_ref());
        let (ys_tr, y_mean, y_scale) = standardize(y_tr.as_ref());
        let ys_te = standardize_apply(y_te.as_ref(), y_mean.as_ref(), y_scale.as_ref());

        let Ok(m) = pls3_fit(
            xs_tr.as_ref(),
            ys_tr.as_ref(),
            1,
            None,
            Pls3FitOpts {
                pre_standardized_x: true,
                pre_standardized_y: true,
                // Seq inside the per-split worker — outer Rayon owns the
                // pool. This governs the surrounding matmuls only: faer's
                // `thin_svd` reads `get_global_parallelism()` internally
                // and ignores `par` entirely. Harmless here (both arms
                // take the same SVD path, so byte-equality holds), but do
                // not read this as full control over the decomposition.
                par: crate::fit::ParChoice::Seq,
            },
        ) else {
            // Inner-fit failure (e.g. SVD non-convergence) on this split: no
            // association on this split, not a propagated error — mirrors
            // the swallow `signal_test::split_half_correlations` documents
            // ("A failed per-half fit already degrades to r = 0").
            return 0.0;
        };
        if m.k_used == 0 {
            // Degenerate half (a zero X̃ or Ỹ block): no direction exists, so
            // the honest answer is no association, not NaN.
            return 0.0;
        }

        let s: Col<f64> = &xs_te * m.u_saliences.col(0);
        let t: Col<f64> = &ys_te * m.v_saliences.col(0);
        pearson_r_guarded(&s, &t)
    };

    let r_vec: Vec<f64> = if opts.disable_parallelism {
        splits.iter().map(per_split).collect()
    } else {
        use rayon::prelude::*;
        splits.par_iter().map(per_split).collect()
    };
    Col::<f64>::from_fn(r_vec.len(), |i| r_vec[i])
}

/// Pearson r with the crate's degenerate-input convention: a constant input
/// vector yields exactly `0.0`, never NaN, and the result is clamped into
/// `[-1, 1]` against fp overshoot.
///
/// Mirrors the arithmetic in `signal_test::split_half_correlations` and
/// `signal_test::split_perm_nr_zbars` — change together; the latter owns the
/// explanation.
/// `dual_route::pls3_split_zbars_columns` calls this one rather than
/// carrying a third copy.
pub(crate) fn pearson_r_guarded(a: &Col<f64>, b: &Col<f64>) -> f64 {
    let n = a.nrows();
    if n == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;
    let a_mean: f64 = (0..n).map(|i| a[i]).sum::<f64>() / n_f;
    let b_mean: f64 = (0..n).map(|i| b[i]).sum::<f64>() / n_f;
    let ss_a: f64 = (0..n).map(|i| (a[i] - a_mean).powi(2)).sum();
    let ss_b: f64 = (0..n).map(|i| (b[i] - b_mean).powi(2)).sum();
    if ss_a < 1e-15 || ss_b < 1e-15 {
        return 0.0;
    }
    let cross: f64 = (0..n)
        .map(|i| (a[i] - a_mean) * (b[i] - b_mean))
        .sum::<f64>();
    (cross / (ss_a * ss_b).sqrt()).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
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
            if j < 2 {
                snr * f[i] + e
            } else {
                e
            }
        });
        let y = Mat::<f64>::from_fn(n, q, |i, j| {
            let e = rng.random_range(-1.0..1.0);
            if j == 0 {
                snr * f[i] + e
            } else {
                e
            }
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
        assert_eq!(r.method, "split_exact");
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

    #[test]
    fn same_seed_reproduces_bit_for_bit() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let a = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(99, 8, 7)).unwrap();
        let b = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(99, 8, 7)).unwrap();
        assert_eq!(a.pvalue.to_bits(), b.pvalue.to_bits());
        assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
    }

    #[test]
    fn serial_and_parallel_are_byte_identical() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let mut serial_opts = opts(99, 8, 7);
        serial_opts.disable_parallelism = true;
        let a = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(99, 8, 7)).unwrap();
        let b = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, serial_opts).unwrap();
        assert_eq!(a.pvalue.to_bits(), b.pvalue.to_bits());
        assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
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

    #[test]
    fn zero_column_x_errors_instead_of_reporting_no_association() {
        let x = Mat::<f64>::from_fn(60, 0, |_, _| 0.0);
        let (_, y) = linked_blocks(60, 5, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(49, 6, 7));
        assert!(matches!(
            r,
            Err(PlsKitError::KExceedsMax { k: 1, k_max: 0 })
        ));
    }

    #[test]
    fn zero_column_y_errors_instead_of_reporting_no_association() {
        let (x, _) = linked_blocks(60, 5, 3, 2.0, 5);
        let y = Mat::<f64>::from_fn(60, 0, |_, _| 0.0);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(49, 6, 7));
        assert!(matches!(
            r,
            Err(PlsKitError::KExceedsMax { k: 1, k_max: 0 })
        ));
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
        // floors, so the request is not rerouted.
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(10, false, 7)).unwrap();
        assert_eq!(r.method, "split_nb");
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
        // from the seed, so only the reference differs. Bit equality holds
        // because `nb_test` and `mean_fisher_z` sum the same clamped atanh
        // values in the same order.
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let nb = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(8, false, 42)).unwrap();
        let ex = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, opts(99, 8, 42)).unwrap();
        assert_eq!(nb.method, "split_nb");
        assert_eq!(ex.method, "split_exact");
        assert_eq!(nb.statistic.to_bits(), ex.statistic.to_bits());
    }

    #[test]
    fn split_nb_gate_reroutes_a_flagged_design() {
        // 3 columns trips the gate's column precheck, so the run is rerouted
        // to split_exact at that method's own default n_perm.
        let (x, y) = linked_blocks(60, 3, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(4, false, 7)).unwrap();
        assert_eq!(r.method, "split_exact");
        assert_eq!(r.n_perm, Some(1000));
        assert_eq!(r.n_splits, Some(4), "the requested n_splits carries over");
        assert!(r.stable_rank.is_some(), "the gate reports what it saw");
    }

    #[test]
    fn split_nb_force_overrides_the_gate() {
        let (x, y) = linked_blocks(60, 3, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(4, true, 7)).unwrap();
        assert_eq!(r.method, "split_nb");
        assert!(r.n_perm.is_none());
    }

    #[test]
    fn split_nb_is_not_gated_on_y() {
        // q = 3 sits on the stable-rank floor the gate applies to X. Y is
        // never gated, so a design that clears the floor on X must run.
        let (x, y) = linked_blocks(60, 8, 3, 2.0, 5);
        let r = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(8, false, 7)).unwrap();
        assert_eq!(r.method, "split_nb");
    }

    #[test]
    fn split_nb_same_seed_reproduces_bit_for_bit() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        let a = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(8, false, 7)).unwrap();
        let b = pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(8, false, 7)).unwrap();
        assert_eq!(a.pvalue.to_bits(), b.pvalue.to_bits());
        assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
    }

    #[test]
    fn split_nb_rejects_k_above_one_and_tiny_n_splits() {
        let (x, y) = linked_blocks(60, 6, 3, 2.0, 5);
        assert!(matches!(
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), 2, nb_opts(8, false, 7)),
            Err(PlsKitError::InvalidArgument(_))
        ));
        assert!(matches!(
            pls3_confirmatory_test(x.as_ref(), y.as_ref(), 1, nb_opts(1, false, 7)),
            Err(PlsKitError::InvalidArgument(_))
        ));
    }

    // ── split_exact dual (Gram) route tests ─────────────────────────────

    /// The equivalence test the PLS3 dual route rests on. Route A refits
    /// PLS3 honestly on each training half for every replicate column
    /// (`pls3_split_lv_correlations`, the shipped primal path); route B is
    /// `dual_route::pls3_split_zbars_columns`. Both get the same splits and
    /// the same permutations, so every one of the B+1 z̄ columns must agree.
    ///
    /// `expect_dual` re-derives the routing rule and asserts which branch
    /// the configuration lands on, so the coverage claim is enforced.
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::too_many_arguments)]
    fn assert_pls3_split_exact_dual_route_matches_honest_refit(
        n: usize,
        p: usize,
        q: usize,
        snr: f64,
        data_seed: u64,
        n_perm: usize,
        n_splits: usize,
        seed: u64,
        expect_dual: bool,
    ) {
        use crate::signal_test::{draw_splits, mean_fisher_z};

        let (x, y) = linked_blocks(n, p, q, snr, data_seed);
        let n_cols = n_perm + 1;
        let o = Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm, n_splits },
            seed: Some(seed),
            ..Pls3ConfirmatoryTestOpts::default()
        };

        // Rebuild the runner's split draw and child seeds off the same seed.
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(n, 1, n_splits, false, &mut rng).unwrap();
        let seeds = crate::rng::child_seeds(&mut rng, n_perm);
        let perms: Vec<Vec<usize>> = seeds
            .iter()
            .map(|s| crate::resample::permute_indices(n, &mut crate::rng::child_rng(*s)))
            .collect();

        let (n_tr, _) = crate::resample::split_sizes(n, 1);
        let dual = crate::dual_route::use_dual_route(n_tr, p, n_cols, q);
        assert_eq!(
            dual, expect_dual,
            "routing check: expected dual={expect_dual}, computed={dual} \
             (n_tr={n_tr}, p={p}, q={q}, B+1={n_cols})"
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

        // Route B: the Gram path.
        let b = crate::dual_route::pls3_split_zbars_columns(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &perms,
            false,
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
    fn pls3_split_exact_dual_route_matches_honest_refit_when_selected() {
        // n=60 ⇒ n_tr=30; p=100, q=3, B+1=50 ⇒ Bq=150:
        // 30·(150+100) = 7,500 < 100·150 = 15,000 ⇒ dual is live.
        assert_pls3_split_exact_dual_route_matches_honest_refit(
            60, 100, 3, 3.0, 7, 49, 6, 11, true,
        );
    }

    #[test]
    fn pls3_split_exact_primal_route_is_the_default_on_narrow_p() {
        // n=60 ⇒ n_tr=30; p=6, q=3, B+1=50: 30·(150+6) = 4,680 > 6·150 = 900
        // ⇒ primal. The kernels must still agree — the rule only picks one.
        assert_pls3_split_exact_dual_route_matches_honest_refit(60, 6, 3, 3.0, 7, 49, 6, 11, false);
    }

    #[test]
    fn pls3_dual_route_serial_and_parallel_are_byte_equal() {
        use crate::signal_test::draw_splits;
        let (x, y) = linked_blocks(60, 100, 3, 3.0, 7);
        let (_, mut rng) = crate::rng::resolve_seed(Some(4)).unwrap();
        let splits = draw_splits(60, 1, 6, false, &mut rng).unwrap();
        let perms: Vec<Vec<usize>> = (0..4)
            .map(|_| crate::resample::permute_indices(60, &mut rng))
            .collect();
        let par = crate::dual_route::pls3_split_zbars_columns(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &perms,
            false,
        );
        let ser = crate::dual_route::pls3_split_zbars_columns(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &perms,
            true,
        );
        for (a, b) in par.iter().zip(ser.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
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
    /// This test is insensitive to `LAMBDA_FLOOR` entirely, not merely to
    /// its exact boundary: `M` is exactly zero here regardless of what the
    /// floor guard decides, so flipping the guard's comparison operator,
    /// mis-setting the constant, or deleting the guard outright would not
    /// change this test's outcome.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn pls3_dual_route_degenerate_x_gives_zero_not_nan() {
        use crate::signal_test::{draw_splits, mean_fisher_z};
        let x = Mat::<f64>::from_fn(60, 5, |_, _| 1.25);
        let (_, y) = linked_blocks(60, 5, 3, 3.0, 7);
        let (_, mut rng) = crate::rng::resolve_seed(Some(4)).unwrap();
        let splits = draw_splits(60, 1, 6, false, &mut rng).unwrap();
        let perms: Vec<Vec<usize>> = vec![crate::resample::permute_indices(60, &mut rng)];
        let z = crate::dual_route::pls3_split_zbars_columns(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &perms,
            true,
        );
        for (col, v) in z.iter().enumerate() {
            assert!(v.is_finite(), "col {col} not finite: {v}");
            assert!(v.abs() < 1e-12, "col {col}: {v}");
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
            disable_parallelism: true,
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
            assert!(rel < 1e-10, "col {col}: primal={av} dual={bv} rel={rel}");
        }
    }

    /// Wiring test: the public runner, on an input the rule sends *dual*,
    /// must return exactly the p-value and statistic the primal branch
    /// would have returned at the same seed.
    ///
    /// Same reasoning as `raw_perm_dual_route_runner_matches_the_primal_reference`
    /// in `signal_test.rs` (change together): the kernel-level equivalence
    /// test hands both routes the same locally-built permutations, so it
    /// cannot catch a column-0 convention slip or a child-seed derivation
    /// that drifts from what `parallel_for_each_seeded` actually draws.
    /// Here the reference comes from the runner's own primal machinery, so
    /// both of those separate the two p-values. A wrong routing argument
    /// (`n_tr`, `n_perm + 1`, `q`) flips the runner to primal instead,
    /// which the precondition assert below covers.
    // See the `float_cmp` note on the raw_perm wiring test — the p-value is
    // a count over a fixed denominator and must match exactly.
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
            disable_parallelism: true,
            ..Pls3ConfirmatoryTestOpts::default()
        };

        // Reference: the runner's pre-branch split draw, then its primal
        // branch, statement for statement.
        let (_, mut rng) = crate::rng::resolve_seed(Some(seed)).unwrap();
        let splits = draw_splits(n, 1, n_splits, false, &mut rng).unwrap();
        let z_obs = mean_fisher_z(&pls3_split_lv_correlations(
            x.as_ref(),
            y.as_ref(),
            &splits,
            &o,
        ));
        let nulls =
            crate::resample::parallel_for_each_seeded(&mut rng, n_perm, true, |_, child| {
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
    }
}
