//! Incremental per-component PLS1 test. Crate-internal helper used by
//! `pls1_find_k_sequence` and the diagnostic path of `pls1_find_k_optimal`.

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::PlsKitResult;
use crate::signal_test::ConfirmatoryMethod;

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
            SequentialArgs::RawPerm { n_perm } => ConfirmatoryArgs::RawPerm { n_perm, n_folds: 5 },
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
    /// Significance threshold alpha used.
    #[allow(dead_code)]
    pub(crate) alpha: f64,
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
/// # Errors
/// `PlsKitError::KExceedsMax` when `k_max == 0` or `k_max > n_features`.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run_incremental_sequence(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    weights: Option<ColRef<'_, f64>>,
    mut opts: IncrementalSequenceOpts,
) -> PlsKitResult<IncrementalSequenceOutput> {
    let max_allowed = x.ncols();
    if k_max == 0 || k_max > max_allowed {
        return Err(crate::error::PlsKitError::KExceedsMax {
            k: k_max,
            k_max: max_allowed,
        });
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
        // type that carries it — what the gate saw on a `split_nb` request —
        // and `pls1_confirmatory_test` populates it under `force` too. The
        // price is one SVD on a forced run.
        //
        // Restandardize with the run's weights unconditionally, ignoring
        // `pre_standardized`. The gate rule is one rule shared with
        // `pls1_confirmatory_test`, which also standardizes here regardless
        // of the flag; honouring the flag would make the two sites disagree
        // whenever the caller standardized with unweighted moments. The
        // cost is one owned copy of X — already paid on the other branch.
        let (owned, _, _) = crate::linalg::standardize_weighted(x, weights);
        let xs = owned.as_ref();
        // n_gate: Kish n_eff under weights, raw row count without —
        // matching `pls1_confirmatory_test`, which passes the `n_eff_val`
        // that `validate_and_normalize_weights` defines the same way.
        #[allow(clippy::cast_precision_loss)]
        let n_gate = weights.map_or(x.nrows() as f64, crate::linalg::compute_n_eff);
        let (fires, sr) = crate::signal_test::split_nb_gate_rule(xs, n_gate);
        stable_rank_out = Some(sr);
        if !force && fires {
            // `n_perm` is split_exact's own default; the requested
            // `n_splits` carries over untouched. Mirrored by the
            // confirmatory reroute in signal_test.rs and by
            // `_REROUTE_FALLBACK_N_PERM` in
            // plskit-py/python/plskit/_api.py — all three change together.
            opts.args = SequentialArgs::SplitExact {
                n_perm: 1000,
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
        alpha: opts.alpha,
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
            // TODO: forward `max_skip_rate` from IncrementalSequenceOpts when Task 10/11
            //       wires that knob through. Currently `ci: None` makes this field dead.
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
    let (xs_full, ys_full) = if opts.pre_standardized {
        (
            Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)]),
            Col::<f64>::from_fn(y.nrows(), |i| y[i]),
        )
    } else if weights.is_some() {
        let (xs, _, _) = standardize_weighted(x, weights);
        let (ys, _, _) = standardize1_weighted(y, weights);
        (xs, ys)
    } else {
        let (xs, _, _) = standardize(x);
        let (ys, _, _) = standardize1(y);
        (xs, ys)
    };

    let (xs_def, ys_def) = if h == 1 {
        (xs_full, ys_full)
    } else {
        // Deflation components are fit with the same weights as the per-step
        // test so that the deflated residual matches the weighted model.
        let prev = pls1_fit(
            xs_full.as_ref(),
            ys_full.as_ref(),
            KSpec::Fixed(h - 1),
            weights,
            FitOpts {
                pre_standardized: true,
                // check_n_eff: false — internal deflation refit; n_eff was
                // already validated at the top-level entry, and truncation is
                // tolerated by design (deflate by whatever was extracted).
                check_n_eff: false,
                keep: opts.keep,
                ..FitOpts::default()
            },
        )?;
        let tp: Mat<f64> = prev.t_scores.as_ref() * prev.p_loadings.transpose();
        let tq: Col<f64> = prev.t_scores.as_ref() * prev.q_loadings.as_ref();
        // T, P′, q live on the √w′-row-scaled problem (fit.rs spec §4.2:
        // row-scaling runs even at pre_standardized=true), so T·P′ ≈ √W·Xs.
        // Deflate the UNscaled standardized data: Xs_d = Xs − √W⁻¹·T·P′ (same
        // for y). prev.weights holds the exact normalized weights the fit
        // row-scaled with (None when absent or uniform — that branch must stay
        // bit-identical to the historical unweighted path). A zero weight
        // zeroes the score row (t = √w·xs·w_vec), so its deflation
        // contribution is 0, not 0·∞.
        let (xs_d, ys_d) = match prev.weights.as_ref() {
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
        };
        (xs_d, ys_d)
    };
    let mut sub_opts = *opts;
    sub_opts.pre_standardized = true;
    p_for_confirmatory_at_k(xs_def.as_ref(), ys_def.as_ref(), 1, weights, &sub_opts, rng)
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

    #[test]
    fn score_unrepresentable() {
        assert!(SequentialArgs::defaults_for(ConfirmatoryMethod::Score).is_none());
    }

    // ── hoisted split_nb auto-gate ───────────────────────────────────────────

    /// `stop_early_override: true` so the whole p-value vector is filled —
    /// these tests read every step, not just the first.
    fn seq_run(
        x: &faer::Mat<f64>,
        y: &Col<f64>,
        k_max: usize,
        args: SequentialArgs,
    ) -> IncrementalSequenceOutput {
        run_incremental_sequence(
            x.as_ref(),
            y.as_ref(),
            k_max,
            None,
            IncrementalSequenceOpts {
                args,
                alpha: 0.05,
                stop_early_override: true,
                pre_standardized: false,
                seed: Some(99),
                disable_parallelism: false,
                verbose: false,
                keep: None,
            },
        )
        .unwrap()
    }

    /// n = 20 sits below the gate's effective-sample floor of 25 while the
    /// iid-uniform spectrum stays well clear of the rank floor, so the size
    /// half of the rule is what fires here.
    #[test]
    fn gate_reroutes_whole_sequence_on_flagged_design() {
        let (x, y) = synth(20, 5, 1, 4.0, 5);
        let r = seq_run(
            &x,
            &y,
            2,
            SequentialArgs::SplitNb {
                n_splits: 10,
                force: false,
            },
        );
        assert_eq!(r.method, "split_exact");
        // Every step ran: the reroute rewrites `opts.args` once, before the
        // loop, so there is no per-step branch that could disagree.
        assert!((0..2).all(|i| !r.pvalues[i].is_nan()), "{:?}", r.pvalues);
    }

    /// The gate restandardizes regardless of `pre_standardized`. Feeding it an
    /// already-standardized flagged design and asserting the same reroute pins
    /// that restandardizing standardized data is the identity for the gate.
    #[test]
    fn gate_reroutes_on_pre_standardized_input() {
        use crate::linalg::{standardize, standardize1};
        let (x, y) = synth(20, 5, 1, 4.0, 5);
        let (xs, _, _) = standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let r = run_incremental_sequence(
            xs.as_ref(),
            ys.as_ref(),
            2,
            None,
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
    }

    /// `force` suppresses the reroute only — the rule is still evaluated, so
    /// the caller can see what it would have decided. Same as
    /// `pls1_confirmatory_test` under `force`.
    #[test]
    fn gate_force_keeps_nb_on_flagged_design() {
        let (x, y) = synth(20, 5, 1, 4.0, 5);
        let r = seq_run(
            &x,
            &y,
            2,
            SequentialArgs::SplitNb {
                n_splits: 10,
                force: true,
            },
        );
        assert_eq!(r.method, "split_nb");
        assert!(r.stable_rank.is_some());
    }

    #[test]
    fn gate_clears_on_adequate_design() {
        let (x, y) = synth(60, 5, 1, 4.0, 2);
        let r = seq_run(
            &x,
            &y,
            3,
            SequentialArgs::SplitNb {
                n_splits: 10,
                force: false,
            },
        );
        assert_eq!(r.method, "split_nb");
        // Populated on the pass path too — it reports what the gate saw, not
        // whether it fired.
        assert!(r.stable_rank.is_some());
    }

    /// Only a `split_nb` request evaluates the rule, so nothing else has a
    /// rank to report.
    #[test]
    fn gate_not_evaluated_for_other_methods() {
        let (x, y) = synth(60, 5, 1, 4.0, 2);
        let r = seq_run(&x, &y, 2, SequentialArgs::RawPerm { n_perm: 20 });
        assert_eq!(r.method, "raw_perm");
        assert!(r.stable_rank.is_none());
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
    fn split_exact_runs_as_a_sequential_method() {
        let (x, y) = synth(60, 5, 1, 4.0, 2);
        let r = seq_run(
            &x,
            &y,
            3,
            SequentialArgs::SplitExact {
                n_perm: 199,
                n_splits: 10,
            },
        );
        assert_eq!(r.method, "split_exact");
        assert!(
            (0..3).all(|i| r.pvalues[i] >= 0.0 && r.pvalues[i] <= 1.0),
            "{:?}",
            r.pvalues
        );
        assert!(r.pvalues[0] < 0.05, "signal component not detected");
    }

    #[test]
    fn incremental_stops_early_at_first_nonrejection() {
        let (x, y) = synth(60, 5, 1, 4.0, 2);
        let r = run_incremental_sequence(
            x.as_ref(),
            y.as_ref(),
            5,
            None,
            IncrementalSequenceOpts {
                args: SequentialArgs::SplitNb {
                    n_splits: 30,
                    force: false,
                },
                alpha: 0.05,
                stop_early_override: false,
                pre_standardized: false,
                seed: Some(11),
                disable_parallelism: false,
                verbose: false,
                keep: None,
            },
        )
        .unwrap();
        let n_filled = (0..r.pvalues.nrows())
            .filter(|i| !r.pvalues[*i].is_nan())
            .count();
        assert!(
            n_filled < 5,
            "stop-early did not trigger; pvalues={:?}",
            r.pvalues
        );
        assert!(r.pvalues[0] < 0.05);
    }

    #[test]
    fn override_runs_all_k() {
        let (x, y) = synth(60, 5, 1, 4.0, 1);
        let r = run_incremental_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
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
}
