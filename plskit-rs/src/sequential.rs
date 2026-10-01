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
        }
    }

    /// Default args for a given method. Returns `None` for `Score` (rejected
    /// at the dispatch boundary in the wrapper — score has no per-component
    /// reading).
    #[must_use]
    pub(crate) fn defaults_for(method: ConfirmatoryMethod) -> Option<Self> {
        Some(match method {
            ConfirmatoryMethod::RawPerm => SequentialArgs::RawPerm { n_perm: 1000 },
            ConfirmatoryMethod::SplitNb => SequentialArgs::SplitNb {
                n_splits: 50,
                force: false,
            },
            ConfirmatoryMethod::SplitExact => SequentialArgs::SplitExact {
                n_perm: 1000,
                n_splits: 50,
            },
            ConfirmatoryMethod::E => SequentialArgs::E,
            ConfirmatoryMethod::Score => return None,
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
    /// Caller asserts X and y are already standardized; skips centering/scaling.
    pub(crate) pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub(crate) seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
    pub(crate) disable_parallelism: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub(crate) verbose: bool,
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
    /// Stable rank of the standardized X, as the hoisted gate saw it.
    /// `Some` whenever `split_nb` was the REQUESTED method — whether the gate
    /// fired or not, and also under `force` — matching the same field on
    /// `ConfirmatoryTestOutput`. `None` for every other requested method,
    /// which never evaluates the gate.
    pub(crate) stable_rank: Option<f64>,
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
/// # Errors
/// `PlsKitError::InvalidArgument` when `k_max == 0`, `PlsKitError::KExceedsMax`
/// when `k_max > n_features`.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run_incremental_sequence(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    weights: Option<ColRef<'_, f64>>,
    n_eff: f64,
    mut opts: IncrementalSequenceOpts,
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
    // exactly as `result.method` is read off `args_resolved` in
    // `pls1_confirmatory_test`.
    let mut stable_rank_out = None;
    if let SequentialArgs::SplitNb { n_splits, force } = opts.args {
        // Evaluated even under `force`, whose only effect is to skip the
        // reroute below: `stable_rank` means the same thing on every result
        // type that carries it (what the gate saw on a `split_nb` request),
        // and `pls1_confirmatory_test` populates it under `force` too. The
        // price is one SVD on a forced run.
        //
        // The helper restandardizes with the run's weights unconditionally,
        // ignoring `pre_standardized`, exactly as `pls1_confirmatory_test`
        // does; honouring the flag would make the two sites disagree
        // whenever the caller standardized with unweighted moments.
        let gate = crate::signal_test::resolve_split_nb(x, weights, n_eff, force);
        stable_rank_out = Some(gate.stable_rank);
        if gate.reroute {
            opts.args = SequentialArgs::SplitExact {
                n_perm: crate::signal_test::SPLIT_NB_REROUTE_N_PERM,
                n_splits,
            };
        }
    }

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;
    let mut pvalues_vec: Vec<f64> = vec![f64::NAN; k_max];
    let mut last_sig: Option<usize> = None;

    for h in 1..=k_max {
        let p = p_for_incremental(x, y, h, weights, &opts, &mut rng)?;
        pvalues_vec[h - 1] = p;
        if p < opts.alpha {
            last_sig = Some(h);
        }
        if !opts.stop_early_override && p >= opts.alpha {
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
    })
}

fn p_for_confirmatory_at_k(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: &IncrementalSequenceOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<f64> {
    use crate::signal_test::{
        confirmatory_test_impl, ConfirmatoryTestInput, ConfirmatoryTestOpts, GateMode,
    };
    // Burn one RNG advance so the per-step seed stream stays bit-stable
    // across `pls1_find_k_sequence` revisions. DO NOT remove without regen
    // of testdata/ — see byte_parity tests for the sentinel.
    let _: u64 = {
        use rand::Rng;
        rng.next_u64()
    };
    let r = confirmatory_test_impl(
        ConfirmatoryTestInput::Raw { x, y, k, weights },
        ConfirmatoryTestOpts {
            args: opts.args.to_confirmatory_args(),
            pre_standardized: opts.pre_standardized,
            seed: Some({
                use rand::Rng;
                rng.next_u64()
            }),
            disable_parallelism: opts.disable_parallelism,
            verbose: opts.verbose,
            ci: None,
            // `IncrementalSequenceOpts` does not expose `max_skip_rate` yet, and
            // `ci: None` makes this field dead until it does.
            max_skip_rate: 0.01,
            keep: opts.keep,
        },
        GateMode::Decided,
    )?;
    Ok(r.pvalue)
}

fn p_for_incremental(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    h: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: &IncrementalSequenceOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<f64> {
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{standardize, standardize1, standardize1_weighted, standardize_weighted};

    // Standardize the same way the per-step fit will: weighted moments when
    // weights are present, and skip standardization entirely when the caller
    // asserts pre-standardized inputs (IncrementalSequenceOpts.pre_standardized
    // contract). The weights=None, pre_standardized=false path must stay
    // bit-identical — it resolves to the plain standardize/standardize1 calls.
    //
    // Pre-standardized input is used as given, copied only when it is not
    // column-major (`linalg::col_major_or_copy`).
    let x_copy = if opts.pre_standardized {
        crate::linalg::col_major_or_copy(x)
    } else {
        None
    };
    let (xs_std, ys_full): (Option<Mat<f64>>, Col<f64>) = if opts.pre_standardized {
        (None, Col::<f64>::from_fn(y.nrows(), |i| y[i]))
    } else if weights.is_some() {
        let (xs, _, _) = standardize_weighted(x, weights);
        let (ys, _, _) = standardize1_weighted(y, weights);
        (Some(xs), ys)
    } else {
        let (xs, _, _) = standardize(x);
        let (ys, _, _) = standardize1(y);
        (Some(xs), ys)
    };
    let xs_full: MatRef<'_, f64> = match (&xs_std, &x_copy) {
        (Some(m), _) | (None, Some(m)) => m.as_ref(),
        (None, None) => x,
    };

    let deflated: Option<(Mat<f64>, Col<f64>)> = if h == 1 {
        None
    } else {
        // Deflation components are fit with the same weights as the per-step
        // test so that the deflated residual matches the weighted model.
        let prev = pls1_fit(
            xs_full,
            ys_full.as_ref(),
            KSpec::Fixed(h - 1),
            weights,
            FitOpts {
                pre_standardized: true,
                // check_n_eff: false: internal deflation refit; n_eff was
                // already validated at the top-level entry, and truncation is
                // tolerated by design (deflate by whatever was extracted).
                check_n_eff: false,
                keep: opts.keep,
                ..FitOpts::default()
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
        // bit-identical to the historical unweighted path). A zero weight
        // zeroes the score row (t = √w·xs·w_vec), so its deflation
        // contribution is 0, not 0·∞.
        Some(match prev.weights.as_ref() {
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
    };
    let (xs_def, ys_def): (MatRef<'_, f64>, ColRef<'_, f64>) = match &deflated {
        Some((xd, yd)) => (xd.as_ref(), yd.as_ref()),
        None => (xs_full, ys_full.as_ref()),
    };
    let mut sub_opts = *opts;
    sub_opts.pre_standardized = true;
    p_for_confirmatory_at_k(xs_def, ys_def, 1, weights, &sub_opts, rng)
}

// ── Tests ─────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn synth(
        n: usize,
        d: usize,
        k_signal: usize,
        snr: f64,
        seed: u64,
    ) -> (faer::Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = faer::Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < k_signal { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + rng.random_range(-1.0..1.0));
        (x, y)
    }

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
                    disable_parallelism: false,
                    verbose: false,
                    keep: None,
                },
            );
            match r {
                Err(PlsKitError::InvalidArgument(msg)) => {
                    assert!(msg.contains("n > 5"), "n={n}: {msg}");
                }
                other => panic!("n={n}: expected InvalidArgument, got {other:?}"),
            }
        }
    }

    // ── hoisted split_nb auto-gate ───────────────────────────────────────────

    /// `pre_standardized = true` still evaluates (and here fires) the hoisted
    /// gate; n = 20 fires on the `n_eff` floor, which does not depend on
    /// standardization, so this pins that the gate is not skipped, not what
    /// it restandardizes.
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
                disable_parallelism: false,
                verbose: false,
                keep: None,
            },
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
            confirmatory_test_impl, ConfirmatoryTestInput, ConfirmatoryTestOpts, GateMode,
        };
        let (x, y) = synth(60, 5, 1, 4.0, 7);
        let r = confirmatory_test_impl(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
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
            GateMode::Decided,
        )
        .unwrap();
        assert_eq!(r.method, "split_nb");
        assert!(r.stable_rank.is_none(), "step evaluated the gate");
    }

    #[test]
    fn override_runs_all_k() {
        let (x, y) = synth(60, 5, 1, 4.0, 1);
        let r = run_incremental_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
            x.nrows() as f64,
            IncrementalSequenceOpts {
                args: SequentialArgs::SplitNb {
                    n_splits: 30,
                    force: false,
                },
                alpha: 0.05,
                stop_early_override: true,
                pre_standardized: false,
                seed: Some(7),
                disable_parallelism: false,
                verbose: false,
                keep: None,
            },
        )
        .unwrap();
        assert_eq!(r.pvalues.nrows(), 3);
        assert!((0..3).all(|i| !r.pvalues[i].is_nan()));
    }

    /// Layout invariance (D1): every step reads X through `standardize` /
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
                let (w_norm, n_eff) =
                    crate::fit::validate_and_normalize_weights(c.w, c.x.nrows(), 3)?;
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
                                disable_parallelism: false,
                                verbose: false,
                                keep,
                            },
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
}
