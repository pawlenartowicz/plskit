//! Incremental per-component PLS1 test. Crate-internal helper used by
//! `pls1_find_k_sequence` and the diagnostic path of `pls1_find_k_optimal`.

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::signal_test::ConfirmatoryMethod;

/// Fold count the sequential `raw_perm` step uses at every `h`. Fixed
/// (this path has no caller-facing `n_folds`; only `n_perm` is an allowed
/// `args` key), so `run_incremental_sequence` validates against it up front
/// instead of letting the per-step `confirmatory_test_impl` call reject
/// with a message naming a parameter the caller never set. `pub(crate)` so
/// `find_k.rs` can hoist the same check ahead of K-selection for
/// `pls1_find_k_optimal(diagnostic="raw_perm")`, where it must not depend
/// on whatever K* selection happens to land on.
pub(crate) const SEQUENTIAL_RAW_PERM_N_FOLDS: usize = 5;

/// Method-specific arguments. `Score` has no sequential variant — it cannot
/// be constructed for this function.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SequentialArgs {
    /// Raw permutation CV R² test per component.
    RawPerm {
        /// Number of permutations.
        n_perm: usize,
    },
    /// Split-half NB test per component.
    SplitNb {
        /// Number of split-half repetitions.
        n_splits: usize,
        /// Run NB even on a design the hoisted auto-gate flags. Default
        /// `false`: a flagged design reroutes the WHOLE sequence to
        /// `split_exact` (see `run_incremental_sequence`).
        force: bool,
    },
    /// Permutation-calibrated split-half test per component (`split_exact`).
    ///
    /// Every step tests at k = 1 on the deflated residual (`p_for_incremental`
    /// passes a literal 1), so the confirmatory route split inside
    /// `split_exact` is decided by `keep` alone, never by the step index:
    /// `keep = None` takes the no-refit route at every step, a set `keep`
    /// takes the refit route at every step. Both routes report `tanh(z̄)`, so
    /// the statistic is uniform down the chain even when `keep` mixes them —
    /// which is what closed testing needs.
    SplitExact {
        /// Number of permutations.
        n_perm: usize,
        /// Number of split-half repetitions.
        n_splits: usize,
    },
    /// Universal-inference split-LR e-value per component.
    E,
    /// `split_exact` or `split_nb`, resolved once on the full X by
    /// `run_incremental_sequence` (see `signal_test::resolve_auto`).
    Auto {
        /// Number of permutations if `split_exact` runs.
        n_perm: usize,
        /// Number of split-half repetitions.
        n_splits: usize,
    },
}

impl SequentialArgs {
    /// The confirmatory-test method tag this variant maps onto.
    #[must_use]
    pub(crate) fn method(&self) -> ConfirmatoryMethod {
        match self {
            SequentialArgs::RawPerm { .. } => ConfirmatoryMethod::RawPerm,
            SequentialArgs::SplitNb { .. } => ConfirmatoryMethod::SplitNb,
            SequentialArgs::SplitExact { .. } => ConfirmatoryMethod::SplitExact,
            SequentialArgs::E => ConfirmatoryMethod::E,
            SequentialArgs::Auto { .. } => ConfirmatoryMethod::Auto,
        }
    }

    /// Default args for a given method. Returns `None` for `Score` and `SvPerm` (rejected
    /// at the dispatch boundary in the wrapper — neither has a per-component
    /// reading).
    #[must_use]
    pub(crate) fn defaults_for(method: ConfirmatoryMethod) -> Option<Self> {
        Some(match method {
            ConfirmatoryMethod::RawPerm => SequentialArgs::RawPerm { n_perm: 1000 },
            ConfirmatoryMethod::SplitNb => SequentialArgs::SplitNb {
                n_splits: 32,
                force: false,
            },
            ConfirmatoryMethod::SplitExact => SequentialArgs::SplitExact {
                n_perm: 1000,
                n_splits: 32,
            },
            ConfirmatoryMethod::E => SequentialArgs::E,
            ConfirmatoryMethod::Auto => SequentialArgs::Auto {
                n_perm: 1000,
                n_splits: 32,
            },
            ConfirmatoryMethod::Score | ConfirmatoryMethod::SvPerm => return None,
        })
    }

    /// Translate to the corresponding [`crate::signal_test::ConfirmatoryArgs`] for the per-step call.
    #[must_use]
    pub(crate) fn to_confirmatory_args(self) -> crate::signal_test::ConfirmatoryArgs {
        use crate::signal_test::ConfirmatoryArgs;
        match self {
            SequentialArgs::RawPerm { n_perm } => ConfirmatoryArgs::RawPerm {
                n_perm,
                n_folds: SEQUENTIAL_RAW_PERM_N_FOLDS,
            },
            // `force` is unread on this path: steps run under
            // `GateMode::Decided`, which skips the gate outright. A fired gate
            // never reaches here as SplitNb at all — it arrives already
            // rewritten to SplitExact by `run_incremental_sequence`.
            SequentialArgs::SplitNb { n_splits, force } => {
                ConfirmatoryArgs::SplitNb { n_splits, force }
            }
            SequentialArgs::SplitExact { n_perm, n_splits } => {
                ConfirmatoryArgs::SplitExact { n_perm, n_splits }
            }
            SequentialArgs::E => ConfirmatoryArgs::E,
            // Never reaches a step: `run_incremental_sequence` resolves it
            // first.
            SequentialArgs::Auto { n_perm, n_splits } => {
                ConfirmatoryArgs::Auto { n_perm, n_splits }
            }
        }
    }
}

/// Cross-cutting tuning knobs for [`run_incremental_sequence`].
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct IncrementalSequenceOpts {
    /// Method dispatch + per-method args.
    pub(crate) args: SequentialArgs,
    /// Significance threshold alpha.
    pub(crate) alpha: f64,
    /// Force-disable stop-early. Public sequence API hard-codes `false`
    /// here; the diagnostic path of `pls1_find_k_optimal` sets `true` so it
    /// can collect the full p-value vector.
    pub(crate) stop_early_override: bool,
    /// Skip the one standardization of X and y over all rows: the gate and
    /// step 1 read X as given, each part only centred. Steps from 2 on test
    /// the deflated residual on the scale step 1 used.
    pub(crate) pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub(crate) seed: Option<u64>,
    /// Sparse keep-count plumbing (spls1 family): threads into BOTH fit
    /// sites per step — the deflation `pls1_fit` and the per-component
    /// confirmatory test — so the test is coherent with the sparse residual.
    pub(crate) keep: Option<usize>,
}

/// Result of [`run_incremental_sequence`].
#[derive(Debug, Clone)]
pub(crate) struct IncrementalSequenceOutput {
    /// p-values per component, length `k_max`. NaN past the early-stop point.
    pub(crate) pvalues: Col<f64>,
    /// Largest `k` with `p_k` < alpha, or `None` if no rejection.
    pub(crate) last_significant_k: Option<usize>,
    /// Method name as a lowercase string (e.g. `"split_nb"`, `"raw_perm"`).
    /// Reports what actually RAN: a `split_nb` request that the hoisted gate
    /// flagged reads `"split_exact"` here.
    pub(crate) method: String,
    /// RNG seed actually used.
    pub(crate) seed: u64,
    /// Stable rank of X, as the hoisted gate saw it: standardized over all
    /// rows, or as given under `pre_standardized`.
    /// `Some` on an explicit `split_nb` request, and on an `"auto"` request
    /// that reached the stable-rank check (p > 4, `n_eff` above 1000 without `keep` (at least 250 with it), and
    /// p ≤ 100·n without `keep`); `None` otherwise. On a `split_nb` request it is set whether the gate
    /// fired or not, and `None` under `force`, matching the same field on
    /// `ConfirmatoryTestOutput`.
    pub(crate) stable_rank: Option<f64>,
    /// Each step's `ConfirmatoryTestOutput::n_failed_splits`, one entry per
    /// step that ran; `None` when the method has no splits.
    pub(crate) n_failed_splits: Option<Vec<usize>>,
    /// Each step's `ConfirmatoryTestOutput::n_failed_perms`, one entry per
    /// step that ran; `None` when the method has no permutation null.
    pub(crate) n_failed_perms: Option<Vec<usize>>,
}

/// Run the incremental sequence on raw data. Stops at the first
/// non-rejection unless `stop_early_override` is true.
///
/// `weights` and `n_eff` are the validated pair
/// `fit::validate_and_normalize_weights` returns (normalized weights, and the
/// Kish `n_eff` of the weights as the public caller handed them; the row
/// count when the weights are absent or all-equal). `n_eff` feeds the `split_nb` gate, so the
/// sequence decides on the same number its result reports.
///
/// With `cov`, `x` and `y` are the residuals on `[1, Z]` of the standardized
/// data and `opts.pre_standardized` is set. Every step then runs
/// `confirmatory_test_core` with the basis, so each permutation null column
/// is residualized on `[1, Z]` (Freedman–Lane); the deflated X and y stay
/// orthogonal to `[1, Z]`, so the same map serves every step. `auto` resolves
/// to `split_exact`. The caller has refused `split_nb`, `score` and `e`.
///
/// # Errors
/// `PlsKitError::InvalidArgument` when `k_max == 0`, `PlsKitError::KExceedsMax`
/// when `k_max > n_features`; the steps' argument and data errors; then
/// `MemoryBudget` when the gate or the copies of X the steps hold do not fit.
#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
pub(crate) fn run_incremental_sequence(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    weights: Option<ColRef<'_, f64>>,
    n_eff: f64,
    mut opts: IncrementalSequenceOpts,
    cov: Option<&crate::covariates::CovariateBasis>,
) -> PlsKitResult<IncrementalSequenceOutput> {
    if k_max == 0 {
        return Err(PlsKitError::InvalidArgument("k_max must be >= 1".into()));
    }
    let max_allowed = x.ncols();
    if k_max > max_allowed {
        return Err(crate::error::PlsKitError::KExceedsMax {
            k: k_max,
            k_max: max_allowed,
        });
    }

    // `test_method="raw_perm"` uses a fixed 5-fold CV at every step (see
    // `SEQUENTIAL_RAW_PERM_N_FOLDS`); reject before the per-step call so the
    // message names the actual constraint (n > 5) rather than a `n_folds`
    // the caller has no way to set on this path.
    let n = x.nrows();
    if matches!(opts.args, SequentialArgs::RawPerm { .. }) && n <= SEQUENTIAL_RAW_PERM_N_FOLDS {
        return Err(PlsKitError::InvalidArgument(format!(
            "test_method='raw_perm' uses {SEQUENTIAL_RAW_PERM_N_FOLDS}-fold CV and needs \
             n > {SEQUENTIAL_RAW_PERM_N_FOLDS} (got n={n}): with n_folds >= n every \
             validation fold is a single row, so the pooled CV R² is undefined"
        )));
    }
    // The argument and data checks of the steps (each tests at k = 1), before
    // any memory check: the gate's below and the copies'.
    crate::fit::check_finite_mat(x)?;
    crate::fit::check_finite_col(y)?;
    crate::signal_test::check_method_counts(opts.args.to_confirmatory_args(), n)?;
    if matches!(
        opts.args,
        SequentialArgs::SplitNb { .. }
            | SequentialArgs::SplitExact { .. }
            | SequentialArgs::Auto { .. }
    ) {
        crate::signal_test::check_split_rows(n, 1)?;
    }

    // ── hoisted `split_nb` auto-gate ────────────────────────────────────────
    // Decided ONCE here, on the full X, before any deflation, and then frozen
    // into `opts.args` for every step. Two reasons it cannot live in the
    // per-step confirmatory call: each step passes the deflated residual,
    // whose spectrum is not X's (deflation removes the y-correlated direction
    // extracted so far, which need not be PC1), and a per-step decision could
    // flip the method mid-sequence, which closed testing cannot use.
    //
    // Rewriting `opts.args` is also what makes the reported method honest —
    // `IncrementalSequenceOutput.method` is read off the resolved args below,
    // exactly as `result.test_method` is read off `args_resolved` in
    // `pls1_confirmatory_test`.
    //
    // `Auto` resolves here too, once on the full X, for the same two reasons.
    //
    // The gate's Gram matrix (wide X) is the one step 1 needs: step 1 tests
    // X on the scale the gate read it, only centred. Later steps test the
    // deflated residual and build their own.
    let mut stable_rank_out = None;
    let mut gate_gram = None;
    if let SequentialArgs::Auto { n_perm, n_splits } = opts.args {
        // Every step tests at k = 1, so `keep` alone decides the route.
        let no_refit = crate::signal_test::split_exact_no_refit_route(1, opts.keep);
        let scale = crate::linalg::held_out_scale(x, weights, opts.pre_standardized);
        let (method, sr, gram) = crate::signal_test::resolve_auto_on_x(
            x,
            weights,
            n_eff,
            no_refit,
            false,
            cov.is_some(),
            false,
            scale.as_ref(),
        )?;
        stable_rank_out = sr;
        gate_gram = gram;
        opts.args = if method == ConfirmatoryMethod::SplitNb {
            SequentialArgs::SplitNb {
                n_splits,
                force: false,
            }
        } else {
            SequentialArgs::SplitExact { n_perm, n_splits }
        };
    } else if let SequentialArgs::SplitNb { n_splits, force } = opts.args {
        // Under `force` the gate builds only the wide Gram matrix step 1
        // reads, and `stable_rank` stays `None`, as on every result type.
        let scale = crate::linalg::held_out_scale(x, weights, opts.pre_standardized);
        let gate = crate::signal_test::resolve_split_nb(x, weights, n_eff, force, scale.as_ref())?;
        stable_rank_out = gate.stable_rank;
        if gate.reroute {
            opts.args = SequentialArgs::SplitExact {
                n_perm: crate::signal_test::SPLIT_NB_REROUTE_N_PERM,
                n_splits,
            };
        }
        gate_gram = gate.gram;
    }

    let seed_used = crate::rng::resolve_seed(opts.seed, n)?;
    let mut pvalues_vec: Vec<f64> = vec![f64::NAN; k_max];
    let mut last_sig: Option<usize> = None;
    let mut failed_splits: Vec<Option<usize>> = Vec::with_capacity(k_max);
    let mut failed_perms: Vec<Option<usize>> = Vec::with_capacity(k_max);

    // The copy of X the steps read lives beside the gate's matrix, which only
    // step 1 reads, and from step 2 on beside the deflated residual and, while
    // that is formed, `T·P'`: checked before it is written, then held.
    // shortcut: X used as given under `pre_standardized` is counted as copied (it is copied only off column-major layout or far from unit scale), and the deflation fit's own buffers are not counted; count them exactly if a sequence runs too few lanes or a capped run is killed in `deflate`.
    let copy = crate::resample::copy_bytes(n, x.ncols());
    let gram = gate_gram
        .as_ref()
        .map_or(0, |_| crate::signal_test::block_gram_bytes(n));
    let deflation = if k_max > 1 { copy.saturating_mul(2) } else { 0 };
    crate::resample::units_at_once(1, copy.saturating_add(gram.max(deflation)), 0)?;
    // Every step starts from the same X and y, standardized once unless flagged.
    let (xs_owned, ys_full) = sequence_inputs(x, y, weights, opts.pre_standardized);
    let _xs = crate::fit::hold_bytes(
        crate::fit::held_bytes().saturating_add(xs_owned.as_ref().map_or(0, |_| copy)),
    );
    let xs_full = xs_owned.as_ref().map_or(x, Mat::as_ref);
    for h in 1..=k_max {
        let step = p_for_incremental(
            xs_full,
            ys_full.as_ref(),
            h,
            weights,
            &opts,
            // Freed after step 1, the only step that reads it.
            gate_gram.take().as_ref(),
            // Step h's test seed, on its own stream.
            crate::rng::child_seed(seed_used, h as u64, crate::rng::TAG_STEPS),
            cov,
        )?;
        let p = step.pvalue;
        pvalues_vec[h - 1] = p;
        failed_splits.push(step.n_failed_splits);
        failed_perms.push(step.n_failed_perms);
        if p < opts.alpha {
            last_sig = Some(h);
        }
        if !opts.stop_early_override && (p.is_nan() || p >= opts.alpha) {
            break;
        }
    }

    let pvalues = Col::<f64>::from_fn(k_max, |i| pvalues_vec[i]);
    Ok(IncrementalSequenceOutput {
        pvalues,
        last_significant_k: last_sig,
        method: opts.args.method().as_str().to_owned(),
        seed: seed_used,
        stable_rank: stable_rank_out,
        // The method is fixed for the run, so every step sets a count or none does.
        n_failed_splits: failed_splits.into_iter().collect(),
        n_failed_perms: failed_perms.into_iter().collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn p_for_confirmatory_at_k(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: &IncrementalSequenceOpts,
    gram: Option<&crate::held_out_gram::HeldOutGram>,
    seed: u64,
    cov: Option<&crate::covariates::CovariateBasis>,
) -> PlsKitResult<crate::signal_test::ConfirmatoryTestOutput> {
    use crate::signal_test::{
        ConfirmatoryTestInput, ConfirmatoryTestOpts, GateMode, confirmatory_test_core,
        confirmatory_test_impl,
    };
    let test_opts = ConfirmatoryTestOpts {
        args: opts.args.to_confirmatory_args(),
        pre_standardized: opts.pre_standardized,
        seed: Some(seed),
        ci: None,
        // `IncrementalSequenceOpts` has no `max_skip_rate`; with
        // `ci: None` this field is unread.
        max_skip_rate: 0.01,
        keep: opts.keep,
    };
    if cov.is_some() {
        confirmatory_test_core(x, y, k, weights, test_opts, GateMode::Decided(gram), cov)
    } else {
        confirmatory_test_impl(
            ConfirmatoryTestInput::Raw {
                x,
                y,
                k: Some(&[k]),
                weights,
                covariates: None,
                null_y: None,
            },
            test_opts,
            GateMode::Decided(gram),
        )
    }
}

/// X and y as step 1 tests them and every later step deflates them, formed
/// once per run: used as given under `pre_standardized` (X copied when
/// `linalg::col_major_or_copy` says it must be: a view that is not
/// column-major, or one with a negative column stride; divided by a power
/// of two when far from unit scale, which moves no p-value,
/// `linalg::pre_standardized_input`); otherwise
/// standardized once over all rows, with weighted moments when weights are
/// present and with the plain `standardize` / `standardize1` when they are
/// not. The first value is `None` when X is read as given. Each step's test
/// only centres its parts; from step 2 on it tests the deflated residual on
/// this scale (see [`p_for_incremental`]).
fn sequence_inputs(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    pre_standardized: bool,
) -> (Option<Mat<f64>>, Col<f64>) {
    use crate::linalg::{standardize, standardize_weighted, standardize1, standardize1_weighted};
    if pre_standardized {
        (
            crate::linalg::pre_standardized_input(x, weights).0,
            Col::<f64>::from_fn(y.nrows(), |i| y[i]),
        )
    } else if weights.is_some() {
        let (xs, _, _) = standardize_weighted(x, weights);
        let (ys, _, _) = standardize1_weighted(y, weights);
        (Some(xs), ys)
    } else {
        let (xs, _, _) = standardize(x);
        let (ys, _, _) = standardize1(y);
        (Some(xs), ys)
    }
}

/// Step `h`'s `k = 1` confirmatory test. At `h = 1` it
/// reads X and y as [`sequence_inputs`] formed them. For `h > 1` both are
/// first deflated by a fit to `h − 1` components, and the test reads the
/// deflated residual on the scale step 1 used. Either way each part is only
/// centred, never rescaled. `gram` is the hoisted gate's Gram matrix of X,
/// which step 1 reads; later steps do not. The deflated residual is held
/// (`fit::hold_bytes`) beside the step's test.
#[allow(clippy::too_many_arguments)]
fn p_for_incremental(
    xs_full: MatRef<'_, f64>,
    ys_full: ColRef<'_, f64>,
    h: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: &IncrementalSequenceOpts,
    gram: Option<&crate::held_out_gram::HeldOutGram>,
    seed: u64,
    cov: Option<&crate::covariates::CovariateBasis>,
) -> PlsKitResult<crate::signal_test::ConfirmatoryTestOutput> {
    let mut sub_opts = *opts;
    sub_opts.pre_standardized = true;
    if h == 1 {
        return p_for_confirmatory_at_k(xs_full, ys_full, 1, weights, &sub_opts, gram, seed, cov);
    }
    let (xd, yd) = deflate(xs_full, ys_full, h - 1, weights, opts.keep)?;
    let _xd = crate::fit::hold_bytes(
        crate::fit::held_bytes()
            .saturating_add(crate::resample::copy_bytes(xd.nrows(), xd.ncols())),
    );
    p_for_confirmatory_at_k(
        xd.as_ref(),
        yd.as_ref(),
        1,
        weights,
        &sub_opts,
        None,
        seed,
        cov,
    )
}

/// X and y with `n_comp` components removed, both on the scale they came in.
fn deflate(
    xs_full: MatRef<'_, f64>,
    ys_full: ColRef<'_, f64>,
    n_comp: usize,
    weights: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
) -> PlsKitResult<(Mat<f64>, Col<f64>)> {
    use crate::fit::{FitOpts, InternalFitOpts, pls1_fit_impl};

    // Deflation components are fit with the same weights as the per-step
    // test so that the deflated residual matches the weighted model.
    let prev = pls1_fit_impl(
        xs_full,
        ys_full,
        n_comp,
        weights,
        None,
        FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        },
        InternalFitOpts {
            // check_n_eff: false: internal deflation refit; n_eff was
            // already validated at the top-level entry, and truncation is
            // tolerated by design (deflate by whatever was extracted).
            check_n_eff: false,
            keep,
        },
    )?;
    // Once per step, outside any replicate loop: the crate's fixed-degree
    // split, like the `Auto` fit above.
    let par = crate::fit::par_fixed();
    let tp = crate::linalg::mat_mul(prev.t_scores.as_ref(), prev.p_loadings.transpose(), par);
    let tq = crate::linalg::mat_vec(prev.t_scores.as_ref(), prev.q_loadings.as_ref(), par);
    // T, P′, q live on the √w′-row-scaled problem (see `pls1_fit` in fit.rs:
    // row-scaling runs even at pre_standardized=true), so T·P′ ≈ √W·Xs.
    // Deflate the UNscaled standardized data: Xs_d = Xs − √W⁻¹·T·P′ (same
    // for y). prev.weights holds the exact normalized weights the fit
    // row-scaled with (None when absent or uniform; that branch must stay
    // bit-identical to the unweighted fit, as the corpus fixtures pin). A zero weight
    // zeroes the score row (t = √w·xs·w_vec), so its deflation
    // contribution is 0, not 0·∞.
    Ok(match prev.weights.as_ref() {
        None => (
            Mat::<f64>::from_fn(xs_full.nrows(), xs_full.ncols(), |i, j| {
                xs_full[(i, j)] - tp[(i, j)]
            }),
            Col::<f64>::from_fn(ys_full.nrows(), |i| ys_full[i] - tq[i]),
        ),
        Some(w) => {
            let inv_sqw: Vec<f64> = (0..xs_full.nrows())
                .map(|i| if w[i] > 0.0 { 1.0 / w[i].sqrt() } else { 0.0 })
                .collect();
            (
                Mat::<f64>::from_fn(xs_full.nrows(), xs_full.ncols(), |i, j| {
                    xs_full[(i, j)] - inv_sqw[i] * tp[(i, j)]
                }),
                Col::<f64>::from_fn(ys_full.nrows(), |i| ys_full[i] - inv_sqw[i] * tq[i]),
            )
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::synth;

    /// `test_method="raw_perm"` uses a fixed 5-fold CV at every step, so
    /// `n <= 5` must be rejected before the per-step call, with a message
    /// naming the actual constraint (`n > 5`), not `n_folds` (which this
    /// path never exposes to the caller).
    #[test]
    fn raw_perm_rejects_n_at_or_below_five() {
        for n in [3_usize, 4, 5] {
            let (x, y) = synth(n, 3, 1, 4.0, 5);
            let r = run_incremental_sequence(
                x.as_ref(),
                y.as_ref(),
                1,
                None,
                n as f64,
                IncrementalSequenceOpts {
                    args: SequentialArgs::RawPerm { n_perm: 20 },
                    alpha: 0.05,
                    stop_early_override: true,
                    pre_standardized: false,
                    seed: Some(99),
                    keep: None,
                },
                None,
            );
            match r {
                Err(PlsKitError::InvalidArgument(msg)) => {
                    assert!(msg.contains("n > 5"), "n={n}: {msg}");
                }
                other => panic!("n={n}: expected InvalidArgument, got {other:?}"),
            }
        }
    }

    /// The steps' argument and data errors come before any memory check, and
    /// the copies of X the steps hold are checked before the first is
    /// written: at budget 1 a NaN in X gives `NonFiniteInput`, one step names
    /// the copy of X and two steps also the deflated residual and `T·P'`.
    #[test]
    fn copies_are_checked_after_the_argument_checks() {
        let (x, y) = synth(60, 5, 1, 4.0, 7);
        let run = |x: MatRef<'_, f64>, k_max| {
            crate::fit::with_budget(1, || {
                run_incremental_sequence(
                    x,
                    y.as_ref(),
                    k_max,
                    None,
                    60.0,
                    IncrementalSequenceOpts {
                        args: SequentialArgs::SplitExact {
                            n_perm: 19,
                            n_splits: 4,
                        },
                        alpha: 0.05,
                        stop_early_override: true,
                        pre_standardized: false,
                        seed: Some(99),
                        keep: None,
                    },
                    None,
                )
            })
        };
        let mut bad = x.clone();
        bad[(3, 2)] = f64::NAN;
        let r = run(bad.as_ref(), 2);
        assert!(matches!(r, Err(PlsKitError::NonFiniteInput)), "{r:?}");
        let copy = crate::resample::copy_bytes(60, 5) as u64;
        for (k_max, want) in [(1, copy), (2, 3 * copy)] {
            match run(x.as_ref(), k_max) {
                Err(PlsKitError::MemoryBudget { needed, .. }) => {
                    assert_eq!(needed, want, "k_max {k_max}");
                }
                other => panic!("k_max {k_max}: {other:?}"),
            }
        }
    }

    // ── hoisted split_nb auto-gate ───────────────────────────────────────────

    /// `pre_standardized = true` still evaluates (and here fires) the hoisted
    /// gate; n = 20 fires on the `n_eff` floor, which does not depend on
    /// standardization, so this pins that the gate is not skipped, not the
    /// scale it reads.
    #[test]
    fn gate_runs_under_pre_standardized() {
        use crate::linalg::{standardize, standardize1};
        let (x, y) = synth(20, 5, 1, 4.0, 5);
        let (xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let r = run_incremental_sequence(
            xs.as_ref(),
            ys.as_ref(),
            2,
            None,
            xs.nrows() as f64,
            IncrementalSequenceOpts {
                args: SequentialArgs::SplitNb {
                    n_splits: 10,
                    force: false,
                },
                alpha: 0.05,
                stop_early_override: true,
                pre_standardized: true,
                seed: Some(99),
                keep: None,
            },
            None,
        )
        .unwrap();
        assert_eq!(r.method, "split_exact");
        assert!(r.stable_rank.is_some());
    }

    /// The no-per-step-re-gate guarantee is structural, so pin the structure
    /// rather than trying to build a design whose deflated residual would gate
    /// differently from X: steps run under `GateMode::Decided`, so a
    /// `split_nb` step never evaluates the gate and never reports a
    /// `stable_rank` of its own, whatever the sequence-level `force` was.
    #[test]
    fn steps_never_re_gate() {
        use crate::signal_test::{
            ConfirmatoryTestInput, ConfirmatoryTestOpts, GateMode, confirmatory_test_impl,
        };
        let (x, y) = synth(60, 5, 1, 4.0, 7);
        let r = confirmatory_test_impl(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[1]),
                weights: None,
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args: SequentialArgs::SplitNb {
                    n_splits: 10,
                    force: false,
                }
                .to_confirmatory_args(),
                seed: Some(7),
                ..Default::default()
            },
            GateMode::Decided(None),
        )
        .unwrap();
        assert_eq!(r.test_method, "split_nb");
        assert!(r.stable_rank.is_none(), "step evaluated the gate");
    }

    /// With `stop_early_override` every step up to `k_max` gets a p-value;
    /// without it the sequence stops at the first non-rejection and leaves
    /// the later steps NaN.
    #[test]
    fn override_runs_all_k() {
        let (x, y) = synth(60, 5, 1, 4.0, 1);
        let run = |stop_early_override: bool| {
            run_incremental_sequence(
                x.as_ref(),
                y.as_ref(),
                4,
                None,
                x.nrows() as f64,
                IncrementalSequenceOpts {
                    args: SequentialArgs::SplitNb {
                        n_splits: 30,
                        force: false,
                    },
                    alpha: 0.05,
                    stop_early_override,
                    pre_standardized: false,
                    seed: Some(7),
                    keep: None,
                },
                None,
            )
            .unwrap()
            .pvalues
        };
        let stopped = run(false);
        assert!(
            stopped.iter().any(|p| p.is_nan()),
            "premise: the early stop fires without the override, got {stopped:?}"
        );
        let all = run(true);
        assert_eq!(all.nrows(), 4);
        assert!(all.iter().all(|p| !p.is_nan()), "{all:?}");
    }

    /// Layout invariance: every step reads the X that `sequence_inputs`
    /// formed through `standardize_weighted`, `standardize` or
    /// `col_major_or_copy`, so padded, row-major and reversed-column views
    /// give bit-identical p-values to the owned matrix, for all four
    /// sequential methods, dense and `keep = 3`, over the copy-free families
    /// with and without `pre_standardized`. `stop_early_override` runs
    /// h = 1, 2, 3 everywhere, so the deflation (weighted branch included) is
    /// exercised under every layout.
    #[test]
    fn incremental_sequence_is_layout_invariant() {
        crate::test_support::assert_families_layout_invariant(
            "incremental_sequence",
            |c| {
                let (w_norm, n_eff) = crate::fit::validate_and_normalize_weights(c.w, c.x.nrows())?;
                let wr = w_norm.as_ref().map(Col::as_ref);
                let mut v = Vec::new();
                for args in [
                    SequentialArgs::SplitNb {
                        n_splits: 10,
                        force: false,
                    },
                    SequentialArgs::SplitExact {
                        n_perm: 49,
                        n_splits: 10,
                    },
                    SequentialArgs::RawPerm { n_perm: 19 },
                    SequentialArgs::E,
                ] {
                    for keep in [None, Some(3)] {
                        let r = run_incremental_sequence(
                            c.x,
                            c.y,
                            3,
                            wr,
                            n_eff,
                            IncrementalSequenceOpts {
                                args,
                                alpha: 0.05,
                                stop_early_override: true,
                                pre_standardized: c.pre,
                                seed: Some(23),
                                keep,
                            },
                            None,
                        )?;
                        v.extend((0..3).map(|i| r.pvalues[i]));
                        v.push(r.stable_rank.unwrap_or(f64::NAN));
                        v.push(if r.method == args.method().as_str() {
                            0.0
                        } else {
                            1.0
                        });
                    }
                }
                Ok(v)
            },
            Vec::clone,
        );
    }

    /// Each step's `n_failed_splits` and `n_failed_perms` are the counts of
    /// the confirmatory test on the step's data at the step's seed: X and y
    /// as `sequence_inputs` forms them at h = 1, deflated by `h - 1`
    /// components after that, read under `pre_standardized`. y is the
    /// indicator of two rows, so step 1 has splits with no positive row in a
    /// half. A run that stops early has one entry per step run.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn sequence_reports_failed_splits_per_step() {
        use crate::signal_test::{
            ConfirmatoryTestInput, ConfirmatoryTestOpts, pls1_confirmatory_test,
        };
        const SEED: u64 = 23;
        let n = 40;
        let (x, y_cont) = synth(n, 6, 2, 1.0, 5);
        let y = Col::<f64>::from_fn(n, |i| if i == 3 || i == 17 { 1.0 } else { 0.0 });
        let opts = |args, stop_early_override| IncrementalSequenceOpts {
            args,
            alpha: 0.05,
            stop_early_override,
            pre_standardized: false,
            seed: Some(SEED),
            keep: None,
        };
        let exact = SequentialArgs::SplitExact {
            n_perm: 19,
            n_splits: 32,
        };
        let nb = SequentialArgs::SplitNb {
            n_splits: 32,
            force: true,
        };
        let (xs, ys) = sequence_inputs(x.as_ref(), y.as_ref(), None, false);
        let xs = xs.unwrap();
        for args in [exact, nb] {
            let r = run_incremental_sequence(
                x.as_ref(),
                y.as_ref(),
                3,
                None,
                n as f64,
                opts(args, true),
                None,
            )
            .unwrap();
            let splits = r
                .n_failed_splits
                .expect("split methods set n_failed_splits");
            assert_eq!(splits.len(), 3, "{args:?}");
            assert_eq!(
                r.n_failed_perms.as_ref().map(Vec::len),
                matches!(args, SequentialArgs::SplitExact { .. }).then_some(3),
                "{args:?}"
            );
            for h in 1..=3 {
                let (xd, yd) = if h == 1 {
                    (xs.clone(), ys.clone())
                } else {
                    deflate(xs.as_ref(), ys.as_ref(), h - 1, None, None).unwrap()
                };
                let direct = pls1_confirmatory_test(
                    ConfirmatoryTestInput::Raw {
                        x: xd.as_ref(),
                        y: yd.as_ref(),
                        k: Some(&[1]),
                        weights: None,
                        covariates: None,
                        null_y: None,
                    },
                    ConfirmatoryTestOpts {
                        args: args.to_confirmatory_args(),
                        pre_standardized: true,
                        seed: Some(crate::rng::child_seed(
                            SEED,
                            h as u64,
                            crate::rng::TAG_STEPS,
                        )),
                        ..Default::default()
                    },
                )
                .unwrap();
                if h == 1 {
                    assert!(
                        direct.n_failed_splits.unwrap() > 0,
                        "{args:?}: premise: step 1 has failed splits"
                    );
                }
                assert_eq!(
                    Some(splits[h - 1]),
                    direct.n_failed_splits,
                    "{args:?} step {h}"
                );
                assert_eq!(
                    r.n_failed_perms.as_ref().map(|v| v[h - 1]),
                    direct.n_failed_perms,
                    "{args:?} step {h}"
                );
            }
        }

        // At `n_perm` 19 no p is below 0.05, so the run stops after step 1.
        let y_noise = Col::<f64>::from_fn(n, |i| y_cont[(i * 7 + 3) % n]);
        let r = run_incremental_sequence(
            x.as_ref(),
            y_noise.as_ref(),
            3,
            None,
            n as f64,
            opts(exact, false),
            None,
        )
        .unwrap();
        let ran = r.pvalues.iter().filter(|p| p.is_finite()).count();
        assert!(
            ran < 3,
            "premise: the sequence stops early, {:?}",
            r.pvalues
        );
        assert_eq!(r.n_failed_splits.map(|v| v.len()), Some(ran));
        assert_eq!(r.n_failed_perms.map(|v| v.len()), Some(ran));
    }

    /// Relative closeness at `tol`, NaN matching NaN (steps past the stop).
    fn assert_rel(a: f64, b: f64, tol: f64, what: &str) {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan(), "{what}: {a:e} vs {b:e}");
            return;
        }
        let d = (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
        assert!(d <= tol, "{what}: {a:e} vs {b:e} (rel {d:e})");
    }

    /// The hoisted gate reads `pre_standardized` as the per-step tests do:
    /// under the flag X is used as given, so a column scale moves
    /// `stable_rank`, and a call on X standardized by hand under the flag
    /// matches the call on X without it.
    #[test]
    fn gate_reads_the_flag() {
        use crate::find_k::{FindKSequenceOpts, pls1_find_k_sequence};
        use crate::linalg::{standardize_weighted, standardize1_weighted};
        let (x, y) = synth(60, 5, 1, 4.0, 3);
        let d = [1.0, 10.0, 0.1];
        let xd = Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * d[j % 3]);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 2.0 } else { 0.5 });
        for wr in [None, Some(w.as_ref())] {
            let run = |x: MatRef<'_, f64>, y: ColRef<'_, f64>, pre: bool| {
                pls1_find_k_sequence(
                    x,
                    y,
                    3,
                    wr,
                    None,
                    FindKSequenceOpts {
                        test_method: ConfirmatoryMethod::SplitNb,
                        n_splits: 10,
                        pre_standardized: pre,
                        seed: Some(5),
                        ..Default::default()
                    },
                )
                .unwrap()
            };
            let tag = if wr.is_some() {
                "weighted"
            } else {
                "unweighted"
            };
            let plain = run(x.as_ref(), y.as_ref(), true).stable_rank.unwrap();
            let scaled = run(xd.as_ref(), y.as_ref(), true).stable_rank.unwrap();
            assert!(
                (plain - scaled).abs() > 1e-6,
                "{tag}: stable_rank {plain} did not move with the column scale"
            );
            let raw = run(x.as_ref(), y.as_ref(), false);
            let xs = standardize_weighted(x.as_ref(), wr).0;
            let ys = standardize1_weighted(y.as_ref(), wr).0;
            let by_hand = run(xs.as_ref(), ys.as_ref(), true);
            assert_eq!(raw.test_method, by_hand.test_method, "{tag}");
            assert_rel(
                raw.stable_rank.unwrap(),
                by_hand.stable_rank.unwrap(),
                1e-12,
                &format!("{tag} stable_rank"),
            );
            for h in 0..3 {
                assert_rel(
                    raw.pvalues[h],
                    by_hand.pvalues[h],
                    1e-12,
                    &format!("{tag} p[{h}]"),
                );
            }
        }
    }

    /// A step from h = 2 on tests the deflated residual as deflation left
    /// it, on the scale step 1 used: the step's p-value is bit-equal to a
    /// confirmatory call on `deflate(.., 1, ..)` read under
    /// `pre_standardized`, for every sequential method, weighted and not.
    #[test]
    fn step_two_tests_the_residual_as_deflated() {
        let (x, y) = synth(60, 5, 1, 4.0, 11);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 2.0 } else { 0.25 });
        let (w_norm, _) = crate::fit::validate_and_normalize_weights(Some(w.as_ref()), 60).unwrap();
        for wr in [None, w_norm.as_ref().map(Col::as_ref)] {
            let (xs, ys) = sequence_inputs(x.as_ref(), y.as_ref(), wr, false);
            let xs = xs.unwrap();
            let (xd, yd) = deflate(xs.as_ref(), ys.as_ref(), 1, wr, None).unwrap();
            for args in [
                SequentialArgs::SplitExact {
                    n_perm: 99,
                    n_splits: 10,
                },
                SequentialArgs::SplitNb {
                    n_splits: 10,
                    force: true,
                },
                SequentialArgs::RawPerm { n_perm: 99 },
                SequentialArgs::E,
            ] {
                let opts = IncrementalSequenceOpts {
                    args,
                    alpha: 0.05,
                    stop_early_override: true,
                    pre_standardized: false,
                    seed: None,
                    keep: None,
                };
                let step =
                    p_for_incremental(xs.as_ref(), ys.as_ref(), 2, wr, &opts, None, 17, None)
                        .unwrap()
                        .pvalue;
                let direct = {
                    let mut flagged = opts;
                    flagged.pre_standardized = true;
                    p_for_confirmatory_at_k(
                        xd.as_ref(),
                        yd.as_ref(),
                        1,
                        wr,
                        &flagged,
                        None,
                        17,
                        None,
                    )
                    .unwrap()
                    .pvalue
                };
                let tag = format!("{:?} weighted={}", args.method(), wr.is_some());
                assert_eq!(
                    step.to_bits(),
                    direct.to_bits(),
                    "{tag}: {step:e} vs {direct:e}"
                );
            }
        }
    }

    /// The sequence keeps the caller's column scale at every step: under
    /// `pre_standardized` a column rescale moves the step-2 p-value, and a
    /// flagged call on X and y standardized by hand matches the unflagged
    /// call at every step, for every sequential method, weighted and not.
    /// A step-2 p at its cap of 1 cannot move, so seed 17, the first from 0
    /// where every case's step-2 p is below 1, is used; the premise assert
    /// checks it.
    #[test]
    fn sequence_keeps_user_scaling_at_every_step() {
        use crate::linalg::{standardize_weighted, standardize1_weighted};
        let (x, y) = synth(60, 5, 1, 2.0, 5);
        let d = [1.0, 10.0, 0.1];
        let xd = Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * d[j % 3]);
        let w = Col::<f64>::from_fn(60, |i| if i % 2 == 0 { 2.0 } else { 0.5 });
        let (w_norm, n_eff) =
            crate::fit::validate_and_normalize_weights(Some(w.as_ref()), 60).unwrap();
        for (wr, n_eff) in [(None, 60.0), (w_norm.as_ref().map(Col::as_ref), n_eff)] {
            let xs = standardize_weighted(x.as_ref(), wr).0;
            let ys = standardize1_weighted(y.as_ref(), wr).0;
            for args in [
                SequentialArgs::SplitExact {
                    n_perm: 49,
                    n_splits: 10,
                },
                SequentialArgs::SplitNb {
                    n_splits: 10,
                    force: true,
                },
                SequentialArgs::RawPerm { n_perm: 49 },
                SequentialArgs::E,
            ] {
                let run = |x: MatRef<'_, f64>, y: ColRef<'_, f64>, pre: bool| {
                    run_incremental_sequence(
                        x,
                        y,
                        3,
                        wr,
                        n_eff,
                        IncrementalSequenceOpts {
                            args,
                            alpha: 0.05,
                            stop_early_override: true,
                            pre_standardized: pre,
                            seed: Some(17),
                            keep: None,
                        },
                        None,
                    )
                    .unwrap()
                    .pvalues
                };
                let tag = format!("{:?} weighted={}", args.method(), wr.is_some());
                let plain = run(x.as_ref(), y.as_ref(), true);
                let scaled = run(xd.as_ref(), y.as_ref(), true);
                assert!(plain[1] < 1.0, "{tag}: premise: step-2 p is below its cap");
                assert!(
                    (plain[1] - scaled[1]).abs() > 1e-9,
                    "{tag}: step-2 p {} did not move with the column scale",
                    plain[1]
                );
                let raw = run(x.as_ref(), y.as_ref(), false);
                let by_hand = run(xs.as_ref(), ys.as_ref(), true);
                for h in 0..3 {
                    assert_rel(raw[h], by_hand[h], 1e-12, &format!("{tag} p[{h}]"));
                }
            }
        }
    }
}
