//! Incremental per-component PLS1 test. Crate-internal helper used by
//! `pls1_find_k_sequence` and the diagnostic path of `pls1_find_k_optimal`.

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::PlsKitResult;
use crate::signal_test::ConfirmatoryMethod;

/// Method-specific arguments. `Score` has no sequential variant —
/// it cannot be constructed for this function.
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
    },
    /// Permutation-calibrated split-half test per component.
    SplitPerm {
        /// Number of permutations.
        n_perm: usize,
        /// Number of split-half repetitions per permutation.
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
            SequentialArgs::SplitPerm { .. } => ConfirmatoryMethod::SplitPerm,
            SequentialArgs::E => ConfirmatoryMethod::E,
        }
    }

    /// Default args for a given method. Returns `None` for `Score`
    /// (rejected at the dispatch boundary in the wrapper).
    #[must_use]
    pub(crate) fn defaults_for(method: ConfirmatoryMethod) -> Option<Self> {
        Some(match method {
            ConfirmatoryMethod::RawPerm => SequentialArgs::RawPerm { n_perm: 1000 },
            ConfirmatoryMethod::SplitNb => SequentialArgs::SplitNb { n_splits: 50 },
            ConfirmatoryMethod::SplitPerm => SequentialArgs::SplitPerm {
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
            SequentialArgs::SplitNb { n_splits } => ConfirmatoryArgs::SplitNb { n_splits },
            SequentialArgs::SplitPerm { n_perm, n_splits } => {
                ConfirmatoryArgs::SplitPerm { n_perm, n_splits }
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
}

/// Result of [`run_incremental_sequence`].
#[derive(Debug, Clone)]
pub(crate) struct IncrementalSequenceOutput {
    /// p-values per component, length `k_max`. NaN past the early-stop point.
    pub(crate) pvalues: Col<f64>,
    /// Largest `k` with `p_k` < alpha, or `None` if no rejection.
    pub(crate) last_significant_k: Option<usize>,
    /// Method name as a lowercase string (e.g. `"split_nb"`, `"raw_perm"`).
    #[allow(dead_code)]
    pub(crate) method: String,
    /// Significance threshold alpha used.
    #[allow(dead_code)]
    pub(crate) alpha: f64,
    /// RNG seed actually used.
    pub(crate) seed: u64,
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
    opts: IncrementalSequenceOpts,
) -> PlsKitResult<IncrementalSequenceOutput> {
    let max_allowed = x.ncols();
    if k_max == 0 || k_max > max_allowed {
        return Err(crate::error::PlsKitError::KExceedsMax {
            k: k_max,
            k_max: max_allowed,
        });
    }
    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed);
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
    use crate::signal_test::{pls1_confirmatory_test, ConfirmatoryTestInput, ConfirmatoryTestOpts};
    // Burn one RNG advance so the per-step seed stream stays bit-stable
    // across `pls1_find_k_sequence` revisions. DO NOT remove without regen
    // of testdata/ — see byte_parity tests for the sentinel.
    let _: u64 = {
        use rand::Rng;
        rng.next_u64()
    };
    let r = pls1_confirmatory_test(
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
        },
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
    use crate::linalg::{standardize, standardize1};
    let (xs_full, _, _) = standardize(x);
    let (ys_full, _, _) = standardize1(y);

    let (xs_def, ys_def) = if h == 1 {
        (xs_full, ys_full)
    } else {
        let prev = pls1_fit(
            xs_full.as_ref(),
            ys_full.as_ref(),
            KSpec::Fixed(h - 1),
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
        )?;
        let tp: Mat<f64> = prev.t_scores.as_ref() * prev.p_loadings.transpose();
        let xs_d = Mat::<f64>::from_fn(xs_full.nrows(), xs_full.ncols(), |i, j| {
            xs_full[(i, j)] - tp[(i, j)]
        });
        let tq: Col<f64> = prev.t_scores.as_ref() * prev.q_loadings.as_ref();
        let ys_d = Col::<f64>::from_fn(ys_full.nrows(), |i| ys_full[i] - tq[i]);
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

    #[test]
    fn incremental_stops_early_at_first_nonrejection() {
        let (x, y) = synth(60, 5, 1, 4.0, 2);
        let r = run_incremental_sequence(
            x.as_ref(),
            y.as_ref(),
            5,
            None,
            IncrementalSequenceOpts {
                args: SequentialArgs::SplitNb { n_splits: 30 },
                alpha: 0.05,
                stop_early_override: false,
                pre_standardized: false,
                seed: Some(11),
                disable_parallelism: false,
                verbose: false,
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
                args: SequentialArgs::SplitNb { n_splits: 30 },
                alpha: 0.05,
                stop_early_override: true,
                pre_standardized: false,
                seed: Some(7),
                disable_parallelism: false,
                verbose: false,
            },
        )
        .unwrap();
        assert_eq!(r.pvalues.nrows(), 3);
        assert!((0..3).all(|i| !r.pvalues[i].is_nan()));
    }
}
