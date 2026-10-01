//! K-selection: optimal (CV / BIC) and sequence (sequential test) methods.

use std::collections::BTreeMap;

use rand::seq::SliceRandom;

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::{pls1_fit, FitOpts, KSpec};
use crate::linalg::{
    col_row_subset, normalize_weights, standardize1_weighted, standardize_apply_rows,
    standardize_rows, standardize_weighted,
};
use crate::sequential::{run_incremental_sequence, IncrementalSequenceOpts, SequentialArgs};
use crate::signal_test::ConfirmatoryMethod;

use faer::{Col, ColRef, Mat, MatRef};

/// Selector used in `pls1_find_k_optimal` to pick K* from the candidate range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selector {
    /// Maximize CV R² with the 1-SE rule (default).
    R2Se,
    /// Maximize CV R² without the 1-SE rule.
    R2Max,
    /// Minimize `BIC(k) = n · log(SSR / n) + k · log(n)`.
    Bic,
}

/// Opts for `pls1_find_k_optimal`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct FindKOptimalOpts {
    /// Which selection criterion to use (CV R² with 1-SE, CV R² max, or BIC).
    pub selector: Selector,
    /// Number of CV folds. Used by `R2Se` and `R2Max` selectors. The wrapper
    /// is responsible for rejecting non-default `n_folds` with `selector=Bic`.
    pub n_folds: usize,
    /// Optional same-sample sequential diagnostic to run on K*. `None` disables
    /// the diagnostic; `Some(method)` enables it (any `ConfirmatoryMethod`
    /// except `Score`). Selection and test share data, so the resulting
    /// pvalues are diagnostic only and not honest inference.
    pub diagnostic: Option<ConfirmatoryMethod>,
    /// Number of permutations for `raw_perm` / `split_exact` diagnostic. Inert
    /// when `diagnostic is None`.
    pub n_perm: usize,
    /// Number of split-half repetitions for `split_nb` / `split_exact`
    /// diagnostic. Inert when `diagnostic is None`.
    pub n_splits: usize,
    /// `split_nb` diagnostic only: run NB even on a design the sequence-level
    /// auto-gate flags. Default `false`, which reroutes the diagnostic to
    /// `split_exact`. Inert unless `diagnostic == Some(SplitNb)`.
    pub force: bool,
    /// Caller asserts X and y are already standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
    ///
    /// Serial replicate loops only: single top-level products (a reference
    /// fit under `ParChoice::Auto`, a one-off scoring product or
    /// decomposition) keep the crate's fixed parallel split, so results
    /// match the parallel run bit for bit.
    pub disable_parallelism: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
}

impl Default for FindKOptimalOpts {
    fn default() -> Self {
        Self {
            selector: Selector::R2Se,
            n_folds: 5,
            diagnostic: None,
            n_perm: 1000,
            n_splits: 50,
            force: false,
            pre_standardized: false,
            seed: None,
            disable_parallelism: false,
            verbose: false,
        }
    }
}

/// Result of `pls1_find_k_optimal`.
#[derive(Debug, Clone)]
pub struct FindKOptimalOutput {
    /// The selected number of components. `0` when the full-data fit cannot
    /// extract a first component (`y` constant, or orthogonal to `X` up to
    /// rounding: the `k_used = 0` zero model of `pls1_fit`); the score maps
    /// are then empty and the diagnostic, if requested, tests nothing.
    pub k_star: usize,
    /// Selection criterion used (`"r2_se"`, `"r2_max"`, or `"bic"`).
    pub selector: String,
    /// CV R² per K (present for `R2Se` and `R2Max` selectors).
    pub cv_scores: Option<BTreeMap<usize, f64>>,
    /// SE of CV R² per K (present only for `R2Se` selector).
    pub cv_scores_se: Option<BTreeMap<usize, f64>>,
    /// BIC scores per K (present only for `Bic` selector).
    pub bic_scores: Option<BTreeMap<usize, f64>>,
    /// Per-component p-values from the same-sample sequential diagnostic up to
    /// K* (present when `diagnostic.is_some()`). Same-sample → diagnostic only,
    /// not honest inference.
    pub pvalues: Option<Col<f64>>,
    /// Name of the diagnostic method that actually ran (present when
    /// `diagnostic.is_some()`). A `split_nb` request the sequence-level
    /// auto-gate flagged reads `"split_exact"`.
    pub diagnostic: Option<String>,
    /// RNG seed actually used.
    pub seed: u64,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
    /// Stable rank of the standardized X, as the sequence-level auto-gate saw
    /// it. `Some` whenever `diagnostic == Some(SplitNb)` — whether the gate
    /// fired or not, and also under `force`. `None` otherwise, including when
    /// no diagnostic was requested.
    pub stable_rank: Option<f64>,
}

impl FindKOptimalOutput {
    /// The component count a fit at the selected K uses: `k_star`.
    ///
    /// This is the policy of `pls1_fit(k = "optimal")` in every wrapper,
    /// kept here so they share it: selection and fit are separate calls,
    /// and a wrapper that composes them asks this rather than reading
    /// `k_star` itself.
    ///
    /// # Errors
    /// [`PlsKitError::OptimalNoComponent`] when `k_star == 0` (no first
    /// component can be extracted). An explicit `k` still fits the
    /// `k_used = 0` zero model; this refuses only to present that model as
    /// the selected one.
    pub fn k_to_fit(&self) -> PlsKitResult<usize> {
        if self.k_star == 0 {
            Err(PlsKitError::OptimalNoComponent)
        } else {
            Ok(self.k_star)
        }
    }
}

/// Optimal-K selection on full data. Optionally attaches a same-sample
/// sequential diagnostic when `diagnostic.is_some()` — selection and test
/// share data, so the diagnostic pvalues are not honest inference.
///
/// # K-range cap (CV selectors)
/// `R2Se` and `R2Max` evaluate components `k = 1..=max_comp` where
/// `max_comp = min(k_max, n − n_folds − 2).max(1)`. This cap prevents CV
/// folds from having fewer training rows than components. `Bic` fits once at
/// `k_max` and sweeps; it is not capped. `cv_scores` (present for CV
/// selectors) covers exactly `1..=max_comp`.
///
/// # No extractable component
/// When the full-data fit truncates at the first component (`y` constant,
/// or orthogonal to the columns of `X` up to rounding), every selector
/// returns `k_star = 0` with an empty score map rather than a `k` the fit
/// cannot produce. This is the same `0` that [`pls1_find_k_sequence`]
/// reports when no component rejects. A requested diagnostic then runs no
/// step: `pvalues` is empty and `diagnostic` names the method the
/// `split_nb` auto-gate resolved to.
///
/// # Errors
/// - `InvalidArgument` for `k_max == 0`
/// - `KExceedsMax` for `k_max > n_features`
/// - `DimensionMismatch` for shape disagreements
/// - `InvalidArgument` if `diagnostic == Some(Score)`
/// - `NonFiniteInput` when X, y, or weights contain NaN/inf
/// - `InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
pub fn pls1_find_k_optimal(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: FindKOptimalOpts,
) -> PlsKitResult<FindKOptimalOutput> {
    find_k_optimal_impl(x, y, k_max, None, weights, opts)
}

/// Sparse counterpart of [`pls1_find_k_optimal`] (mode 3 of the `spls1`
/// family): identical CV / selector machinery with the inner fitter swapped
/// to the sparse fit at the caller's fixed `keep`. Same selectors, options,
/// and output shape; `keep = n_features` reproduces the dense function
/// bit-exactly.
///
/// The `Bic` selector reuses the dense complexity penalty `k · ln(n_eff)` —
/// NOT a keep-aware sparse BIC: effective complexity scales with `keep` per
/// component, so the dense penalty underpenalizes added components and
/// biases the selected `k` upward under sparsity. Deliberate v1
/// simplification (not a consequence of `keep` being fixed); BIC values are
/// not comparable across `keep`. A keep-aware penalty is deferred.
///
/// # Errors
/// Everything `pls1_find_k_optimal` returns, plus `InvalidArgument` for
/// `keep == 0` or `keep > n_features`.
pub fn spls1_find_k_optimal(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    keep: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: FindKOptimalOpts,
) -> PlsKitResult<FindKOptimalOutput> {
    crate::fit::validate_keep(keep, x.ncols())?;
    find_k_optimal_impl(x, y, k_max, Some(keep), weights, opts)
}

#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::type_complexity)]
fn find_k_optimal_impl(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    keep: Option<usize>,
    weights: Option<ColRef<'_, f64>>,
    opts: FindKOptimalOpts,
) -> PlsKitResult<FindKOptimalOutput> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x.ncols()),
            y: y.nrows(),
        });
    }
    if k_max == 0 {
        return Err(PlsKitError::InvalidArgument("k_max must be >= 1".into()));
    }
    let max_allowed = x.ncols();
    if k_max > max_allowed {
        return Err(PlsKitError::KExceedsMax {
            k: k_max,
            k_max: max_allowed,
        });
    }
    if matches!(opts.diagnostic, Some(ConfirmatoryMethod::Score)) {
        return Err(PlsKitError::InvalidArgument(
            "score has no sequential variant".into(),
        ));
    }
    // Hoisted ahead of selection: `diagnostic="raw_perm"` uses a fixed
    // 5-fold CV at every sequential step (`SEQUENTIAL_RAW_PERM_N_FOLDS`),
    // same floor `run_incremental_sequence` enforces for
    // `pls1_find_k_sequence`. Checking here, not only inside the branch
    // that calls `run_incremental_sequence`, keeps the outcome independent
    // of where selection lands K*: a `k_star = 0` selection (e.g. a
    // constant `y`) must not skip this check, whatever `n` is.
    if matches!(opts.diagnostic, Some(ConfirmatoryMethod::RawPerm))
        && n <= crate::sequential::SEQUENTIAL_RAW_PERM_N_FOLDS
    {
        return Err(PlsKitError::InvalidArgument(format!(
            "diagnostic='raw_perm' uses {f}-fold CV and needs n > {f} (got n={n}): with \
             n_folds >= n every validation fold is a single row, so the pooled CV R² is \
             undefined",
            f = crate::sequential::SEQUENTIAL_RAW_PERM_N_FOLDS
        )));
    }

    let (w_norm, n_eff_val) = crate::fit::validate_weights_for_k(weights, n, k_max)?;

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;

    // Selector dispatch. A `y` with no extractable first component gives
    // `k_star = 0` with empty score maps on every selector: the CV selectors
    // get that from the shared CV layer (`cv_select`), the BIC selector from
    // its own full-data fit (an empty sweep, see `select_bic`).
    let (k_star, cv_scores, cv_scores_se, bic_scores, selector_str): (
        usize,
        Option<BTreeMap<usize, f64>>,
        Option<BTreeMap<usize, f64>>,
        Option<BTreeMap<usize, f64>>,
        &'static str,
    ) = match opts.selector {
        Selector::R2Se => {
            let (k, scores, se) = select_cv(
                x,
                y,
                k_max,
                opts.n_folds,
                true,
                &opts,
                w_norm.as_ref().map(Col::as_ref),
                keep,
                &mut rng,
            )?;
            (k, Some(scores), se, None, "r2_se")
        }
        Selector::R2Max => {
            let (k, scores, _) = select_cv(
                x,
                y,
                k_max,
                opts.n_folds,
                false,
                &opts,
                w_norm.as_ref().map(Col::as_ref),
                keep,
                &mut rng,
            )?;
            (k, Some(scores), None, None, "r2_max")
        }
        Selector::Bic => {
            let (k, scores) = select_bic(
                x,
                y,
                k_max,
                w_norm.as_ref().map(Col::as_ref),
                n_eff_val,
                keep,
            )?;
            (k, None, None, Some(scores), "bic")
        }
    };

    let (pvalues, diagnostic_str, stable_rank) = if let (Some(method), 0) =
        (opts.diagnostic, k_star)
    {
        // No component to test. Report the method the sequence would have
        // run, with the same gate evaluation `run_incremental_sequence`
        // makes, so `diagnostic` and `stable_rank` keep their meaning.
        if matches!(method, ConfirmatoryMethod::SplitNb) {
            let gate = crate::signal_test::resolve_split_nb(
                x,
                w_norm.as_ref().map(Col::as_ref),
                n_eff_val,
                opts.force,
            );
            (
                Some(Col::zeros(0)),
                Some(gate.method().as_str().to_owned()),
                Some(gate.stable_rank),
            )
        } else {
            (Some(Col::zeros(0)), Some(method.as_str().to_owned()), None)
        }
    } else if let Some(method) = opts.diagnostic {
        // stop_early_override=true so we collect the full p-value vector up to K*.
        let seq_args = SequentialArgs::defaults_for(method).ok_or_else(|| {
            PlsKitError::InvalidArgument(format!("{} has no sequential variant", method.as_str()))
        })?;
        let seq_args = match seq_args {
            SequentialArgs::RawPerm { .. } => SequentialArgs::RawPerm {
                n_perm: opts.n_perm,
            },
            SequentialArgs::SplitNb { .. } => SequentialArgs::SplitNb {
                n_splits: opts.n_splits,
                force: opts.force,
            },
            SequentialArgs::SplitExact { .. } => SequentialArgs::SplitExact {
                n_perm: opts.n_perm,
                n_splits: opts.n_splits,
            },
            SequentialArgs::E => SequentialArgs::E,
        };
        let r = run_incremental_sequence(
            x,
            y,
            k_star,
            w_norm.as_ref().map(Col::as_ref),
            n_eff_val,
            IncrementalSequenceOpts {
                args: seq_args,
                alpha: 0.05,
                stop_early_override: true, // collect ALL p-values up to K*
                pre_standardized: opts.pre_standardized,
                seed: Some({
                    use rand::Rng;
                    rng.next_u64()
                }),
                disable_parallelism: opts.disable_parallelism,
                verbose: opts.verbose,
                keep,
            },
        )?;
        // Echo what RAN, not what was asked for: the hoisted `split_nb`
        // auto-gate can reroute the whole sequence to `split_exact`.
        (Some(r.pvalues), Some(r.method), r.stable_rank)
    } else {
        (None, None, None)
    };

    Ok(FindKOptimalOutput {
        k_star,
        selector: selector_str.to_owned(),
        cv_scores,
        cv_scores_se,
        bic_scores,
        pvalues,
        diagnostic: diagnostic_str,
        seed: seed_used,
        n_eff: n_eff_val,
        stable_rank,
    })
}

/// Opts for `pls1_find_k_sequence`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct FindKSequenceOpts {
    /// Per-step test method (any `ConfirmatoryMethod` except `Score`).
    /// `SplitNb` is subject to the sequence-level auto-gate:
    /// a flagged design runs `split_exact` instead, and
    /// `FindKSequenceOutput.test_method` says so.
    pub test_method: ConfirmatoryMethod,
    /// Significance threshold for sequential rejection.
    pub alpha: f64,
    /// Number of permutations for `raw_perm` / `split_exact`.
    pub n_perm: usize,
    /// Number of split-half repetitions for `split_nb` / `split_exact`.
    pub n_splits: usize,
    /// `split_nb` only: run NB even on a design the sequence-level auto-gate
    /// flags. Default `false`, which reroutes the whole sequence to
    /// `split_exact`. Inert for every other `test_method`.
    pub force: bool,
    /// Caller asserts X and y are already standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
    ///
    /// Serial replicate loops only: single top-level products (a reference
    /// fit under `ParChoice::Auto`, a one-off scoring product or
    /// decomposition) keep the crate's fixed parallel split, so results
    /// match the parallel run bit for bit.
    pub disable_parallelism: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
}

impl Default for FindKSequenceOpts {
    fn default() -> Self {
        Self {
            test_method: ConfirmatoryMethod::SplitNb,
            alpha: 0.05,
            n_perm: 1000,
            n_splits: 50,
            force: false,
            pre_standardized: false,
            seed: None,
            disable_parallelism: false,
            verbose: false,
        }
    }
}

/// Result of `pls1_find_k_sequence`.
#[derive(Debug, Clone)]
pub struct FindKSequenceOutput {
    /// The selected number of components (0 if no rejection).
    pub k_star: usize,
    /// Per-component p-values, length `k_max`. NaN past the stop point.
    pub pvalues: Col<f64>,
    /// Name of the test method that actually ran. A `split_nb` request the
    /// sequence-level auto-gate flagged reads `"split_exact"`.
    pub test_method: String,
    /// Significance threshold used.
    pub alpha: f64,
    /// RNG seed actually used.
    pub seed: u64,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
    /// Stable rank of the standardized X, as the sequence-level auto-gate saw
    /// it. `Some` whenever `test_method == SplitNb` — whether the gate fired
    /// or not, and also under `force`. `None` for every other test method,
    /// which never evaluates the gate.
    pub stable_rank: Option<f64>,
}

impl FindKSequenceOutput {
    /// The component count a fit at the selected K uses: `k_star`.
    ///
    /// The policy of `pls1_fit(k = "sequence")` in every wrapper; see
    /// [`FindKOptimalOutput::k_to_fit`].
    ///
    /// # Errors
    /// [`PlsKitError::SequenceNoRejection`] when `k_star == 0` (no
    /// component rejected at `alpha`).
    pub fn k_to_fit(&self) -> PlsKitResult<usize> {
        if self.k_star == 0 {
            Err(PlsKitError::SequenceNoRejection { alpha: self.alpha })
        } else {
            Ok(self.k_star)
        }
    }
}

/// Sequence-based K selection on full data. Stop-early is
/// always on — call this when you want a frequentist guarantee on the
/// nested chain.
///
/// `k_star = 0` when the first step does not reject, which includes a
/// `y` from which no first component can be extracted (constant, or
/// orthogonal to `X` up to rounding), matching [`pls1_find_k_optimal`].
///
/// # Errors
/// - `InvalidArgument` for `k_max == 0`
/// - `KExceedsMax` for `k_max > n_features`
/// - `DimensionMismatch` for shape disagreements
/// - `InvalidArgument` for `Score` test method (no sequential variant)
/// - `NonFiniteInput` when X, y, or weights contain NaN/inf
/// - `InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
pub fn pls1_find_k_sequence(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: FindKSequenceOpts,
) -> PlsKitResult<FindKSequenceOutput> {
    find_k_sequence_impl(x, y, k_max, None, weights, opts)
}

/// Sparse counterpart of [`pls1_find_k_sequence`] (mode 4 of the `spls1`
/// family): the same sequential closed test with the inner fitter swapped
/// to the sparse fit at the caller's fixed `keep`. `keep` threads through
/// BOTH per-step fit sites — the deflation fit and the per-component
/// confirmatory test — so each step tests the sparse marginal component
/// against the sparse residual. `keep = n_features` reproduces the dense
/// function bit-exactly.
///
/// # Errors
/// Everything `pls1_find_k_sequence` returns, plus `InvalidArgument` for
/// `keep == 0` or `keep > n_features`.
pub fn spls1_find_k_sequence(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    keep: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: FindKSequenceOpts,
) -> PlsKitResult<FindKSequenceOutput> {
    crate::fit::validate_keep(keep, x.ncols())?;
    find_k_sequence_impl(x, y, k_max, Some(keep), weights, opts)
}

#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::many_single_char_names)]
fn find_k_sequence_impl(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    keep: Option<usize>,
    weights: Option<ColRef<'_, f64>>,
    opts: FindKSequenceOpts,
) -> PlsKitResult<FindKSequenceOutput> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x.ncols()),
            y: y.nrows(),
        });
    }
    if k_max == 0 {
        return Err(PlsKitError::InvalidArgument("k_max must be >= 1".into()));
    }
    let max_allowed = x.ncols();
    if k_max > max_allowed {
        return Err(PlsKitError::KExceedsMax {
            k: k_max,
            k_max: max_allowed,
        });
    }

    let (w_norm, n_eff_val) = crate::fit::validate_weights_for_k(weights, n, k_max)?;

    let seq_args = SequentialArgs::defaults_for(opts.test_method).ok_or_else(|| {
        PlsKitError::InvalidArgument(format!(
            "test_method='{}' has no sequential variant",
            opts.test_method.as_str()
        ))
    })?;
    let seq_args = match seq_args {
        SequentialArgs::RawPerm { .. } => SequentialArgs::RawPerm {
            n_perm: opts.n_perm,
        },
        SequentialArgs::SplitNb { .. } => SequentialArgs::SplitNb {
            n_splits: opts.n_splits,
            force: opts.force,
        },
        SequentialArgs::SplitExact { .. } => SequentialArgs::SplitExact {
            n_perm: opts.n_perm,
            n_splits: opts.n_splits,
        },
        SequentialArgs::E => SequentialArgs::E,
    };
    let r = run_incremental_sequence(
        x,
        y,
        k_max,
        w_norm.as_ref().map(Col::as_ref),
        n_eff_val,
        IncrementalSequenceOpts {
            args: seq_args,
            alpha: opts.alpha,
            stop_early_override: false,
            pre_standardized: opts.pre_standardized,
            seed: opts.seed,
            disable_parallelism: opts.disable_parallelism,
            verbose: opts.verbose,
            keep,
        },
    )?;
    let k_star = r.last_significant_k.unwrap_or(0);
    Ok(FindKSequenceOutput {
        k_star,
        pvalues: r.pvalues,
        // What RAN, not what was asked for — see `run_incremental_sequence`'s
        // hoisted `split_nb` auto-gate.
        test_method: r.method,
        alpha: opts.alpha,
        seed: r.seed,
        n_eff: n_eff_val,
        stable_rank: r.stable_rank,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal selectors
// ──────────────────────────────────────────────────────────────────────────────

/// Per-fold standardized train/val slices shared by `select_cv` and
/// `spls1_find_keep_optimal`. Train fold standardized with weighted
/// moments; val fold standardized with the train fold's parameters.
/// Per-fold weights re-normalized to mean 1; an unnormalizable slice
/// yields `None` → that fold runs unweighted (kept rather than an
/// explicit uniform fill — the two paths are not provably bit-identical;
/// see the original note at the `select_cv` call site).
struct FoldData {
    xs_tr: Mat<f64>,
    ys_tr: Col<f64>,
    xs_val: Mat<f64>,
    ys_val: Col<f64>,
    train_w: Option<Col<f64>>,
    val_w: Option<Col<f64>>,
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn prep_fold(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    folds: &[Vec<usize>],
    fi: usize,
    val_idx: &[usize],
) -> FoldData {
    let train_idx: Vec<usize> = folds
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != fi)
        .flat_map(|(_, f)| f.iter().copied())
        .collect();
    let y_tr = col_row_subset(y, &train_idx);
    let y_val = col_row_subset(y, val_idx);
    let nv = y_val.nrows();

    // Slice and re-normalize weights for train fold (if weights are provided).
    // An unnormalizable slice yields None → that fold runs unweighted.
    let train_w_full: Option<Col<f64>> = weights.map(|w| col_row_subset(w, &train_idx));
    let train_w_norm: Option<Col<f64>> = train_w_full
        .as_ref()
        .and_then(|w| normalize_weights(w.as_ref()));
    let train_wref = train_w_norm.as_ref().map(Col::as_ref);

    // Slice and re-normalize weights for val fold.
    let val_w_full: Option<Col<f64>> = weights.map(|w| col_row_subset(w, val_idx));
    let val_w_norm: Option<Col<f64>> = val_w_full
        .as_ref()
        .and_then(|w| normalize_weights(w.as_ref()));

    let (xs_tr, x_mean, x_scale) = standardize_rows(x, &train_idx, train_wref, None);
    let xs_val = standardize_apply_rows(x, val_idx, x_mean.as_ref(), x_scale.as_ref(), None);
    let (ys_tr, ym, ys) = standardize1_weighted(y_tr.as_ref(), train_wref);
    let ys_val = Col::<f64>::from_fn(nv, |i| (y_val[i] - ym) / ys);

    FoldData {
        xs_tr,
        ys_tr,
        xs_val,
        ys_val,
        train_w: train_w_norm,
        val_w: val_w_norm,
    }
}

impl FoldData {
    /// Weighted total sum of squares of the validation `y` about its
    /// (weighted) mean: the denominator of [`FoldData::r2`].
    fn ss_tot(&self) -> f64 {
        let nv = self.ys_val.nrows();
        let val_wref = self.val_w.as_ref().map(Col::as_ref);
        let mean_val = match val_wref {
            None => {
                if nv > 0 {
                    (0..nv).map(|i| self.ys_val[i]).sum::<f64>() / nv as f64
                } else {
                    0.0
                }
            }
            Some(wv) => (0..nv).map(|i| wv[i] * self.ys_val[i]).sum::<f64>() / nv as f64,
        };
        match val_wref {
            None => (0..nv).map(|i| (self.ys_val[i] - mean_val).powi(2)).sum(),
            Some(wv) => (0..nv)
                .map(|i| wv[i] * (self.ys_val[i] - mean_val).powi(2))
                .sum(),
        }
    }

    /// Weighted validation R² of `y_pred` against `ss_tot` (from
    /// [`FoldData::ss_tot`]); `0` on a validation `y` with no spread.
    fn r2(&self, y_pred: &Col<f64>, ss_tot: f64) -> f64 {
        let nv = self.ys_val.nrows();
        let ss_res: f64 = match self.val_w.as_ref() {
            None => (0..nv).map(|i| (y_pred[i] - self.ys_val[i]).powi(2)).sum(),
            Some(wv) => (0..nv)
                .map(|i| wv[i] * (y_pred[i] - self.ys_val[i]).powi(2))
                .sum(),
        };
        if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        }
    }
}

/// Fold count the CV layer runs: the caller's `n_folds`, capped at `n − 2`
/// and floored at 2. The floor can push the effective count back up to (or,
/// for `n <= 1`, past) `n` at very small `n`; see
/// [`reject_cv_leave_one_out`], which every caller of this function must
/// check the result against.
fn cv_n_folds(n: usize, requested: usize) -> usize {
    requested.min(n.saturating_sub(2)).max(2)
}

/// Reject the same leave-one-out degeneracy `raw_perm` rejects, for the CV
/// layer `cv_select` shares between `select_cv` (backs
/// `pls1_find_k_optimal`'s `r2_se` / `r2_max`) and `spls1_find_keep_optimal`.
/// `cv_n_folds` floors the effective fold count at 2, which for `n <= 2`
/// gives `effective_n_folds >= n`: every validation fold is then a single
/// row (`linalg::fold_split` puts at most `ceil(n / n_folds)` rows in any
/// fold, and `n_folds >= n` makes that `<= 1`), so a training fold has too
/// few rows to fit anything and every per-fold score is `NaN` or an
/// uninformative constant: the same shape of degeneracy
/// `signal_test::confirmatory_test_impl` rejects for `raw_perm`, just
/// surfacing through the per-fold-averaged convention instead of the
/// pooled `ss_res` / `ss_tot` one. `n / 2 < effective_n_folds < n` (some,
/// not all, folds singleton) is unaffected: at least one fold still has a
/// usable training set, so the mean CV score stays informative, just
/// noisier. `requested` is the caller's own `n_folds` (before the
/// `cv_n_folds` cap/floor), reported in the error so a caller who asked
/// for a fold count nowhere near the floor still sees what they passed,
/// not the capped value they cannot change.
fn reject_cv_leave_one_out(n: usize, requested: usize, effective: usize) -> PlsKitResult<()> {
    if effective >= n {
        return Err(PlsKitError::InvalidArgument(format!(
            "cross-validation needs n >= 3 (got n={n}): the fold count (requested \
             {requested}, capped at max(2, n - 2) = {effective}) would make every validation \
             fold a single row (leave-one-out)"
        )));
    }
    Ok(())
}

/// The K-fold CV layer shared by `select_cv` (candidates are component
/// counts `1..=max_comp`) and `spls1_find_keep_optimal` (candidates are the
/// keep grid): shuffle, split into `n_folds` folds, run `fold_work` on each
/// fold's standardized slices ([`prep_fold`]), and pick through
/// [`aggregate_and_pick`]. `fold_work` returns one validation R² per
/// candidate.
///
/// # No extractable component
/// Before any fold runs, it asks the full data whether a first component
/// exists at all ([`first_component_exhausted`], at `check_keep`). When it
/// does not (`y` constant, or orthogonal to the columns of `X` up to
/// rounding), it returns pick `0` with empty score maps: the `k_star = 0`
/// convention. The folds would still produce scores (0 on a constant `y`,
/// below 0 on an orthogonal one) and a pick, and a fold may even extract a
/// component the full data cannot, so a pick there names a model the
/// full-data fit returns as the `k_used = 0` zero model. The check draws
/// nothing from `rng`.
///
/// Each fold is independent and RNG-free (the only RNG use is the shuffle),
/// so byte-parity holds for both serial and parallel execution.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn cv_select<F>(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    check_keep: Option<usize>,
    n_folds: usize,
    candidates: &[usize],
    use_1se: bool,
    disable_parallelism: bool,
    rng: &mut crate::rng::Rng,
    fold_work: F,
) -> PlsKitResult<(usize, BTreeMap<usize, f64>, BTreeMap<usize, f64>)>
where
    F: Fn(&FoldData) -> PlsKitResult<Vec<f64>> + Sync,
{
    if first_component_exhausted(x, y, weights, check_keep)? {
        return Ok((0, BTreeMap::new(), BTreeMap::new()));
    }
    let n = x.nrows();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    let folds = crate::linalg::fold_split(&indices, n_folds);
    let run = |fi: usize, val_idx: &Vec<usize>| {
        fold_work(&prep_fold(x, y, weights, &folds, fi, val_idx.as_slice()))
    };
    let r2_matrix: Vec<Vec<f64>> = if disable_parallelism {
        folds
            .iter()
            .enumerate()
            .map(|(fi, val_idx)| run(fi, val_idx))
            .collect::<PlsKitResult<Vec<_>>>()?
    } else {
        use rayon::prelude::*;
        folds
            .par_iter()
            .enumerate()
            .map(|(fi, val_idx)| run(fi, val_idx))
            .collect::<PlsKitResult<Vec<_>>>()?
    };
    Ok(aggregate_and_pick(&r2_matrix, candidates, use_1se))
}

/// Aggregate per-fold R² rows into mean/SE maps keyed by `candidates`,
/// then pick: the best finite mean, or — under the 1-SE rule — the
/// SMALLEST candidate whose mean is within one SE of the best. Shared by
/// `select_cv` (candidates = component counts `1..=max_comp`) and
/// `spls1_find_keep_optimal` (candidates = the keep grid): "smallest" is
/// the parsimony order on both axes.
fn aggregate_and_pick(
    r2_matrix: &[Vec<f64>],
    candidates: &[usize],
    use_1se: bool,
) -> (usize, BTreeMap<usize, f64>, BTreeMap<usize, f64>) {
    let mut cv_scores = BTreeMap::new();
    let mut cv_scores_se = BTreeMap::new();
    for (ci, &cand) in candidates.iter().enumerate() {
        let finite: Vec<f64> = r2_matrix
            .iter()
            .map(|row| row[ci])
            .filter(|v| v.is_finite())
            .collect();
        if finite.is_empty() {
            cv_scores.insert(cand, f64::NAN);
            cv_scores_se.insert(cand, f64::NAN);
        } else {
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (finite.len() - 1).max(1) as f64;
            let se = var.sqrt() / (finite.len() as f64).sqrt();
            cv_scores.insert(cand, mean);
            cv_scores_se.insert(cand, se);
        }
    }
    let best = *cv_scores
        .iter()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
        .map_or(&candidates[0], |(c, _)| c);
    let pick = if use_1se {
        let threshold = cv_scores[&best] - cv_scores_se.get(&best).copied().unwrap_or(0.0);
        cv_scores
            .iter()
            .filter(|(_, v)| v.is_finite() && **v >= threshold)
            .map(|(c, _)| *c)
            .min()
            .unwrap_or(candidates[0])
    } else {
        best
    };
    (pick, cv_scores, cv_scores_se)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::type_complexity)]
fn select_cv(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    n_folds: usize,
    use_1se: bool,
    opts: &FindKOptimalOpts,
    weights: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<(usize, BTreeMap<usize, f64>, Option<BTreeMap<usize, f64>>)> {
    // k-fold CV with optional 1-SE rule.
    // CV-R² convention (cites the "CV-R² convention" anchor on `pooled_cv_r2_columns` in
    // signal_test.rs): this function uses per-fold *weighted-validation* R² averaged across
    // folds, which differs from raw_perm's pooled SS_res/SS_tot convention. The two are not
    // interchangeable; each method owns the convention appropriate to its statistic.
    let n = x.nrows();
    let requested_n_folds = n_folds;
    let n_folds = cv_n_folds(n, n_folds);
    reject_cv_leave_one_out(n, requested_n_folds, n_folds)?;
    let max_comp = k_max.min(n.saturating_sub(n_folds + 2)).max(1);

    // row[k-1] = CV R² for this fold and k components.
    let fold_work = |fd: &FoldData| -> PlsKitResult<Vec<f64>> {
        let train_wref = fd.train_w.as_ref().map(Col::as_ref);
        let m = pls1_fit(
            fd.xs_tr.as_ref(),
            fd.ys_tr.as_ref(),
            KSpec::Fixed(max_comp),
            train_wref,
            FitOpts {
                pre_standardized: true,
                // check_n_eff: false — per-fold slice may have low n_eff; let the math degrade
                // and rely on the parent statistic to absorb noise.
                check_n_eff: false,
                // Seq inside the per-fold worker — outer Rayon owns the threadpool.
                par: crate::fit::ParChoice::Seq,
                keep,
            },
        )?;
        let actual = m.w_star.ncols();
        let ss_tot = fd.ss_tot();
        let mut row = vec![f64::NAN; max_comp];
        for k in 1..=actual {
            let coef_k = crate::fit::pls1_coef_at_k(
                &m.w_star,
                &m.p_loadings,
                &m.q_loadings,
                k,
                faer::Par::Seq,
            );
            let y_pred =
                crate::linalg::mat_vec(fd.xs_val.as_ref(), coef_k.as_ref(), faer::Par::Seq);
            row[k - 1] = fd.r2(&y_pred, ss_tot);
        }
        Ok(row)
    };

    let candidates: Vec<usize> = (1..=max_comp).collect();
    let (k_star, cv_scores, cv_scores_se) = cv_select(
        x,
        y,
        weights,
        keep,
        n_folds,
        &candidates,
        use_1se,
        opts.disable_parallelism,
        rng,
        fold_work,
    )?;
    Ok((
        k_star,
        cv_scores,
        if use_1se { Some(cv_scores_se) } else { None },
    ))
}

#[allow(clippy::many_single_char_names)]
fn select_bic(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
    weights: Option<ColRef<'_, f64>>,
    n_eff: f64,
    keep: Option<usize>,
) -> PlsKitResult<(usize, BTreeMap<usize, f64>)> {
    let (xs, _, _) = standardize_weighted(x, weights);
    let (ys, _, _) = standardize1_weighted(y, weights);
    let m = pls1_fit(
        xs.as_ref(),
        ys.as_ref(),
        KSpec::Fixed(k_max),
        weights,
        FitOpts {
            pre_standardized: true,
            // check_n_eff: false — the BIC sweep below iterates 1..=k_used and
            // is truncation-tolerant by design (rank-deficient X just shortens
            // the sweep); n_eff was already validated at the top-level entry.
            check_n_eff: false,
            keep,
            ..FitOpts::default()
        },
    )?;
    // A first-component truncation (constant y, or y orthogonal to X up to
    // rounding) leaves nothing to sweep: no component, no score.
    if m.w_star.ncols() == 0 {
        return Ok((0, BTreeMap::new()));
    }
    let mut best_k = 1;
    let mut best_bic = f64::INFINITY;
    let mut bic_scores = BTreeMap::<usize, f64>::new();
    let nv = ys.nrows();
    for k in 1..=m.w_star.ncols() {
        let coef_k =
            crate::fit::pls1_coef_at_k(&m.w_star, &m.p_loadings, &m.q_loadings, k, faer::Par::Seq);
        // Outside any replicate loop, like the `Auto` fit above.
        let y_pred = crate::linalg::mat_vec(xs.as_ref(), coef_k.as_ref(), crate::fit::par_fixed());
        let ssr_w: f64 = match weights {
            None => (0..nv).map(|i| (y_pred[i] - ys[i]).powi(2)).sum(),
            Some(w) => (0..nv).map(|i| w[i] * (y_pred[i] - ys[i]).powi(2)).sum(),
        };
        let bic = n_eff * (ssr_w / n_eff).ln() + k as f64 * n_eff.ln();
        bic_scores.insert(k, bic);
        if bic < best_bic {
            best_bic = bic;
            best_k = k;
        }
    }
    Ok((best_k, bic_scores))
}

/// Whether the full-data fit truncates at the first component: `y` constant,
/// or orthogonal to the columns of `X` up to rounding, which `pls1_fit`
/// returns as the `k_used = 0` zero model. Standardizes exactly as
/// `select_bic` does (weighted moments, whatever `pre_standardized` says) and
/// threads `keep`, so the sparse entry points ask the sparse fit.
fn first_component_exhausted(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    weights: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
) -> PlsKitResult<bool> {
    let (xs, _, _) = standardize_weighted(x, weights);
    let (ys, _, _) = standardize1_weighted(y, weights);
    let m = pls1_fit(
        xs.as_ref(),
        ys.as_ref(),
        KSpec::Fixed(1),
        weights,
        FitOpts {
            pre_standardized: true,
            // check_n_eff: false: the truncation is the answer sought here,
            // not an error, and n_eff was validated at the entry.
            check_n_eff: false,
            keep,
            ..FitOpts::default()
        },
    )?;
    Ok(m.k_used == 0)
}

/// Logged geometric keep grid for `spls1_find_keep_optimal`: powers of two
/// in `[1, n_features)` plus the dense endpoint `n_features`. Both endpoints
/// are always included — the dense endpoint keeps the bit-parity mirror
/// reachable. The 1-SE parsimony selector makes exact spacing non-critical.
fn keep_grid(n_features: usize) -> Vec<usize> {
    let mut grid = Vec::new();
    let mut v = 1usize;
    while v < n_features {
        grid.push(v);
        v = v.saturating_mul(2);
    }
    grid.push(n_features);
    grid
}

/// Opts for `spls1_find_keep_optimal`.
#[derive(Debug, Clone, Copy)]
pub struct FindKeepOptimalOpts {
    /// Number of CV folds.
    pub n_folds: usize,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
    ///
    /// Serial replicate loops only: single top-level products (a reference
    /// fit under `ParChoice::Auto`, a one-off scoring product or
    /// decomposition) keep the crate's fixed parallel split, so results
    /// match the parallel run bit for bit.
    pub disable_parallelism: bool,
    /// Print the swept keep grid to stderr.
    pub verbose: bool,
}

impl Default for FindKeepOptimalOpts {
    fn default() -> Self {
        Self {
            n_folds: 5,
            seed: None,
            disable_parallelism: false,
            verbose: false,
        }
    }
}

/// Result of `spls1_find_keep_optimal`.
#[derive(Debug, Clone)]
pub struct FindKeepOptimalOutput {
    /// Selected keep-count: the SPARSEST grid point whose mean CV R² is
    /// within 1 SE of the best. `0` when the full-data fit cannot extract a
    /// first component at any keep (`y` constant, or orthogonal to `X` up to
    /// rounding); the score maps and `keep_grid` are then empty. The same
    /// `0` that `spls1_find_k_optimal` reports as `k_star` on that input.
    pub keep_star: usize,
    /// The fixed component count the sweep ran at (echoes the caller's `k`).
    pub k: usize,
    /// Mean CV R² per swept keep.
    pub cv_scores: BTreeMap<usize, f64>,
    /// SE of CV R² per swept keep.
    pub cv_scores_se: BTreeMap<usize, f64>,
    /// The keep grid actually swept (no-silent-caps rule). Empty when
    /// `keep_star = 0`: nothing was swept.
    pub keep_grid: Vec<usize>,
    /// RNG seed actually used.
    pub seed: u64,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
}

/// Keep-count tuning at fixed `k` (mode 2 of the `spls1` family): sweep a
/// logged geometric keep grid over `[1, n_features]`, compute per-fold CV R²
/// at each grid point, and return the sparsest keep whose mean CV R² is
/// within 1 SE of the best. Sparsity is tuned inside the training split,
/// never on test data — same honest-inference contract as
/// `pls1_find_k_optimal`.
///
/// Runs through the CV layer `select_cv` uses (`cv_select`: folds,
/// aggregation, 1-SE band) with its own per-fold work: `keep` is a
/// fit-time hard threshold (not post-hoc truncatable like `k`), so each
/// grid point is an independent fit, `n_folds × |grid|` fits.
///
/// # No extractable component
/// When the full-data fit cannot extract a first component at any keep
/// (`y` constant, or orthogonal to the columns of `X` up to rounding), it
/// returns `keep_star = 0` with empty score maps and an empty `keep_grid`
/// rather than a keep that selects nothing, matching the `k_star = 0` of
/// [`spls1_find_k_optimal`] on the same input.
///
/// # Errors
/// - `DimensionMismatch` for shape disagreements
/// - `InvalidArgument` for `k == 0`
/// - `KExceedsMax` for `k > n_features`
/// - `NonFiniteInput` / `InvalidWeights` as per `pls1_fit`
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::many_single_char_names)]
pub fn spls1_find_keep_optimal(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    opts: FindKeepOptimalOpts,
) -> PlsKitResult<FindKeepOptimalOutput> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x.ncols()),
            y: y.nrows(),
        });
    }
    let d = x.ncols();
    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    if k > d {
        return Err(PlsKitError::KExceedsMax { k, k_max: d });
    }
    crate::fit::check_finite_mat(x)?;
    crate::fit::check_finite_col(y)?;
    let (w_norm, n_eff_val) = crate::fit::validate_weights_for_k(weights, n, k)?;

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;
    let grid = keep_grid(d);
    if opts.verbose {
        eprintln!("spls1_find_keep_optimal: sweeping keep grid {grid:?} at fixed k={k}");
    }

    let n_folds = cv_n_folds(n, opts.n_folds);
    reject_cv_leave_one_out(n, opts.n_folds, n_folds)?;
    let wref = w_norm.as_ref().map(Col::as_ref);

    let fold_work = |fd: &FoldData| -> PlsKitResult<Vec<f64>> {
        let train_wref = fd.train_w.as_ref().map(Col::as_ref);
        let ss_tot = fd.ss_tot();
        let mut row = vec![f64::NAN; grid.len()];
        for (gi, &kp) in grid.iter().enumerate() {
            let m = pls1_fit(
                fd.xs_tr.as_ref(),
                fd.ys_tr.as_ref(),
                KSpec::Fixed(k),
                train_wref,
                FitOpts {
                    pre_standardized: true,
                    check_n_eff: false,
                    par: crate::fit::ParChoice::Seq,
                    keep: Some(kp),
                },
            )?;
            let y_pred =
                crate::linalg::mat_vec(fd.xs_val.as_ref(), m.coef.as_ref(), faer::Par::Seq);
            row[gi] = fd.r2(&y_pred, ss_tot);
        }
        Ok(row)
    };

    // The first-component check runs at the dense endpoint: selection only
    // shrinks the first weight vector (`‖select(X'y)‖ ≤ ‖X'y‖`), so a `y` the
    // dense fit cannot extract a component from gives none at any keep, and
    // one it can gives a component at the dense endpoint of the grid.
    let (keep_star, cv_scores, cv_scores_se) = cv_select(
        x,
        y,
        wref,
        None,
        n_folds,
        &grid,
        true,
        opts.disable_parallelism,
        &mut rng,
        fold_work,
    )?;
    // Nothing was swept when there is no first component (`keep_star = 0`).
    let grid = if keep_star == 0 { Vec::new() } else { grid };

    Ok(FindKeepOptimalOutput {
        keep_star,
        k,
        cv_scores,
        cv_scores_se,
        keep_grid: grid,
        seed: seed_used,
        n_eff: n_eff_val,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use rand::SeedableRng;

    fn synth(n: usize, d: usize, k_signal: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < k_signal { 1.0 } else { 0.0 });
        let y_signal: Col<f64> = x.as_ref() * beta.as_ref();
        let y = Col::<f64>::from_fn(n, |i| y_signal[i] * snr + rng.random_range(-1.0..1.0));
        (x, y)
    }

    /// The two inputs whose full-data fit truncates at the first component:
    /// a constant `y`, and a `y` residualized on `[1, X]` (orthogonal to the
    /// standardized columns of `X` up to rounding).
    #[allow(clippy::many_single_char_names)]
    fn degenerate_ys(n: usize, d: usize, seed: u64) -> (Mat<f64>, [(&'static str, Col<f64>); 2]) {
        use rand::RngExt;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let e = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        // Gram-Schmidt basis of [1, X], orthogonalized twice for accuracy.
        let basis = crate::test_support::orthonormal_basis(
            Col::<f64>::from_fn(n, |_| 1.0).as_ref(),
            x.as_ref(),
            0.0,
        );
        let mut y_orth = e;
        crate::test_support::project_off(&basis, &mut y_orth);
        let y_const = Col::<f64>::from_fn(n, |_| 3.0);
        (x, [("constant", y_const), ("orthogonal", y_orth)])
    }

    /// A first-component truncation leaves no component to select, so every
    /// selector, dense and sparse, reports `k_star = 0` with an empty score
    /// map. Before the fix BIC returned `k_star = 1` with an empty map and the
    /// CV selectors `k_star = 1` with NaN (constant) or negative
    /// (orthogonal) scores.
    #[test]
    fn optimal_first_component_truncation_returns_k_star_zero() {
        let (x, ys) = degenerate_ys(60, 8, 5);
        for (name, y) in &ys {
            // The premise: pls1_fit gives the zero model on this y.
            let m = pls1_fit(
                x.as_ref(),
                y.as_ref(),
                KSpec::Fixed(2),
                None,
                FitOpts::default(),
            )
            .unwrap();
            assert_eq!(m.k_used, 0, "{name}: premise");
            for selector in [Selector::R2Se, Selector::R2Max, Selector::Bic] {
                let opts = FindKOptimalOpts {
                    selector,
                    seed: Some(3),
                    ..Default::default()
                };
                let dense = pls1_find_k_optimal(x.as_ref(), y.as_ref(), 3, None, opts).unwrap();
                let sparse =
                    spls1_find_k_optimal(x.as_ref(), y.as_ref(), 3, 4, None, opts).unwrap();
                for (path, r) in [("pls1", &dense), ("spls1", &sparse)] {
                    let tag = format!("{name} {path} {selector:?}");
                    assert_eq!(r.k_star, 0, "{tag}");
                    assert_eq!(r.seed, 3, "{tag}");
                    match selector {
                        Selector::R2Se => {
                            assert!(r.cv_scores.as_ref().unwrap().is_empty(), "{tag}");
                            assert!(r.cv_scores_se.as_ref().unwrap().is_empty(), "{tag}");
                            assert!(r.bic_scores.is_none(), "{tag}");
                        }
                        Selector::R2Max => {
                            assert!(r.cv_scores.as_ref().unwrap().is_empty(), "{tag}");
                            assert!(r.cv_scores_se.is_none(), "{tag}");
                            assert!(r.bic_scores.is_none(), "{tag}");
                        }
                        Selector::Bic => {
                            assert!(r.cv_scores.is_none(), "{tag}");
                            assert!(r.bic_scores.as_ref().unwrap().is_empty(), "{tag}");
                        }
                    }
                    assert!(r.pvalues.is_none() && r.diagnostic.is_none(), "{tag}");
                }
            }
        }
    }

    /// The "fit at the selected K" policy lives in the core, so every
    /// wrapper's `k = "optimal"` / `k = "sequence"` gets the same answer:
    /// `k_star` itself, or a typed error when there is no component to fit.
    #[test]
    fn k_to_fit_refuses_k_star_zero() {
        let (x, ys) = degenerate_ys(60, 8, 5);
        for (name, y) in &ys {
            let opt = pls1_find_k_optimal(
                x.as_ref(),
                y.as_ref(),
                3,
                None,
                FindKOptimalOpts {
                    seed: Some(3),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(opt.k_star, 0, "{name}");
            let e = opt.k_to_fit().unwrap_err();
            assert!(matches!(e, PlsKitError::OptimalNoComponent), "{name}: {e}");
            assert_eq!(e.code(), "optimal_no_component");

            let seq = pls1_find_k_sequence(
                x.as_ref(),
                y.as_ref(),
                3,
                None,
                FindKSequenceOpts {
                    test_method: ConfirmatoryMethod::E,
                    seed: Some(3),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(seq.k_star, 0, "{name}");
            let e = seq.k_to_fit().unwrap_err();
            assert!(
                matches!(e, PlsKitError::SequenceNoRejection { alpha } if alpha.to_bits() == 0.05_f64.to_bits()),
                "{name}: {e}"
            );
            assert_eq!(e.code(), "sequence_no_rejection");
        }

        let (x, y) = synth(80, 5, 1, 5.0, 1);
        let opt = pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            4,
            None,
            FindKOptimalOpts {
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(opt.k_star >= 1);
        assert_eq!(opt.k_to_fit().unwrap(), opt.k_star);
        let seq = pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::E,
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(seq.k_star >= 1);
        assert_eq!(seq.k_to_fit().unwrap(), seq.k_star);
    }

    /// The keep sweep answers the same question from the same shared CV
    /// layer: with no first component at any keep, `keep_star = 0` (the
    /// `k_star = 0` convention) with empty score maps and an empty swept
    /// grid, rather than `keep_star = 1` over scores of 0 (constant) or
    /// below 0 (orthogonal).
    #[test]
    fn keep_optimal_first_component_truncation_returns_keep_star_zero() {
        let (x, ys) = degenerate_ys(60, 8, 5);
        for (name, y) in &ys {
            for k in [1, 2] {
                let r = spls1_find_keep_optimal(
                    x.as_ref(),
                    y.as_ref(),
                    k,
                    None,
                    FindKeepOptimalOpts {
                        seed: Some(3),
                        ..Default::default()
                    },
                )
                .unwrap();
                let tag = format!("{name} k={k}");
                assert_eq!(r.keep_star, 0, "{tag}");
                assert_eq!(r.k, k, "{tag}");
                assert_eq!(r.seed, 3, "{tag}");
                assert!(r.cv_scores.is_empty(), "{tag}");
                assert!(r.cv_scores_se.is_empty(), "{tag}");
                assert!(r.keep_grid.is_empty(), "{tag}");
            }
        }
    }

    /// With `k_star = 0` a requested diagnostic runs no step: `pvalues` is
    /// empty (length `k_star`), and `diagnostic` / `stable_rank` report what
    /// the gate resolved to, as they would on a sequence that ran.
    #[test]
    fn optimal_diagnostic_at_k_star_zero_is_empty() {
        // n = 20 trips the split_nb gate's effective-sample floor.
        let (x, ys) = degenerate_ys(20, 5, 9);
        let gate = crate::signal_test::split_nb_gate(x.as_ref(), None).unwrap();
        assert!(gate.fires);
        for (name, y) in &ys {
            for (method, force, expect, has_rank) in [
                (ConfirmatoryMethod::E, false, "e", false),
                (ConfirmatoryMethod::RawPerm, false, "raw_perm", false),
                (ConfirmatoryMethod::SplitNb, false, "split_exact", true),
                (ConfirmatoryMethod::SplitNb, true, "split_nb", true),
            ] {
                for selector in [Selector::Bic, Selector::R2Se] {
                    let r = pls1_find_k_optimal(
                        x.as_ref(),
                        y.as_ref(),
                        2,
                        None,
                        FindKOptimalOpts {
                            selector,
                            diagnostic: Some(method),
                            force,
                            n_perm: 20,
                            n_splits: 5,
                            seed: Some(7),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let tag = format!("{name} {selector:?} {method:?} force={force}");
                    assert_eq!(r.k_star, 0, "{tag}");
                    assert_eq!(r.pvalues.as_ref().unwrap().nrows(), 0, "{tag}");
                    assert_eq!(r.diagnostic.as_deref(), Some(expect), "{tag}");
                    assert_eq!(r.stable_rank.is_some(), has_rank, "{tag}");
                    if has_rank {
                        assert_eq!(
                            r.stable_rank.unwrap().to_bits(),
                            gate.stable_rank.to_bits(),
                            "{tag}"
                        );
                    }
                }
            }
        }
    }

    /// `diagnostic="raw_perm"` needs `n > 5` (fixed 5-fold CV at every
    /// sequential step), rejected up front, before selection runs, so the
    /// outcome does not depend on where selection lands `k_star`: a
    /// non-constant `y` at n = 4 selects `k_star = 1` with the CV
    /// selectors (which would otherwise reach `run_incremental_sequence`
    /// and hit the error there instead), and a constant `y` selects
    /// `k_star = 0` (which runs no sequential step at all, so without the
    /// hoist it would report empty `pvalues` and no error, at any `n`).
    /// Both must raise the same way, with the `raw_perm` validator's message
    /// (`check_n_eff_for_k` also returns `InvalidArgument`, so the variant
    /// alone does not say which check fired), at every n <= 5.
    #[test]
    fn optimal_raw_perm_diagnostic_rejects_small_n_regardless_of_k_star() {
        for n in [3_usize, 4, 5] {
            let (x, y) = synth(n, 3, 1, 4.0, 5);
            let (_, ys) = degenerate_ys(n, 3, 5);
            for selector in [Selector::Bic, Selector::R2Se] {
                for (name, y) in [("signal", &y), (ys[0].0, &ys[0].1), (ys[1].0, &ys[1].1)] {
                    let r = pls1_find_k_optimal(
                        x.as_ref(),
                        y.as_ref(),
                        2,
                        None,
                        FindKOptimalOpts {
                            selector,
                            diagnostic: Some(ConfirmatoryMethod::RawPerm),
                            n_perm: 20,
                            seed: Some(7),
                            ..Default::default()
                        },
                    );
                    match r {
                        Err(PlsKitError::InvalidArgument(msg)) => {
                            assert!(msg.contains("n > 5"), "n={n} {selector:?} {name}: {msg}");
                        }
                        other => panic!(
                            "n={n} {selector:?} {name}: expected InvalidArgument, got {other:?}"
                        ),
                    }
                }
            }
        }
    }

    /// `n = 2` is leave-one-out: `cv_n_folds`'s floor of 2 pushes the
    /// effective fold count back up to `n`, so every validation fold is a
    /// single row and every training fold is too small to fit anything.
    /// The CV score for that fold is then `NaN` or an uninformative
    /// constant, depending on how a single-row training fold's
    /// zero-variance column happens to fit; either way `n = 2` must be
    /// rejected, the same degeneracy `raw_perm` rejects, just surfacing
    /// through the per-fold-averaged convention instead of the pooled one.
    #[test]
    fn optimal_rejects_leave_one_out_at_n_eq_2() {
        let (x, y) = synth(2, 3, 1, 4.0, 6);
        for selector in [Selector::R2Se, Selector::R2Max] {
            let r = pls1_find_k_optimal(
                x.as_ref(),
                y.as_ref(),
                1,
                None,
                FindKOptimalOpts {
                    selector,
                    ..Default::default()
                },
            );
            assert!(
                matches!(r, Err(PlsKitError::InvalidArgument(_))),
                "{selector:?}: expected InvalidArgument for n = 2, got {r:?}"
            );
        }
    }

    /// The `split_nb` gate reads its effective-sample input off ONE formula
    /// at every entry point. These weights sit on the `n_eff` floor of 25:
    /// Kish `n_eff` of the raw weights is `24.999999999999996` (fires), of
    /// their normalized copy `25.00000000000002` (does not). The sequence
    /// used to recompute it from the normalized weights it is handed, so it
    /// ran `split_nb` on a design `split_nb_gate` and
    /// `pls1_confirmatory_test` both reroute.
    #[test]
    fn split_nb_gate_agrees_across_entry_points_at_the_n_eff_floor() {
        let (x, y) = synth(26, 8, 1, 4.0, 3);
        let w = Col::<f64>::from_fn(26, |i| if i == 25 { 2.083_333_333_333_333 } else { 1.0 });
        // Without the weights n = 26 clears the floor, so the size half of
        // the rule is what decides below.
        assert!(
            !crate::signal_test::split_nb_gate(x.as_ref(), None)
                .unwrap()
                .fires
        );
        let gate = crate::signal_test::split_nb_gate(x.as_ref(), Some(w.as_ref())).unwrap();
        assert!(gate.n_eff < 25.0, "n_eff = {}", gate.n_eff);
        assert!(gate.fires);

        let conf = crate::signal_test::pls1_confirmatory_test(
            crate::signal_test::ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: Some(w.as_ref()),
            },
            crate::signal_test::ConfirmatoryTestOpts {
                args: crate::signal_test::ConfirmatoryArgs::SplitNb {
                    n_splits: 5,
                    force: false,
                },
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(conf.method, "split_exact");
        assert_eq!(
            conf.stable_rank.unwrap().to_bits(),
            gate.stable_rank.to_bits()
        );

        let seq = pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            2,
            Some(w.as_ref()),
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_perm: 20,
                n_splits: 5,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(seq.test_method, "split_exact");
        assert_eq!(seq.n_eff.to_bits(), gate.n_eff.to_bits());
        assert_eq!(
            seq.stable_rank.unwrap().to_bits(),
            gate.stable_rank.to_bits()
        );

        let opt = pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            2,
            Some(w.as_ref()),
            FindKOptimalOpts {
                selector: Selector::Bic,
                diagnostic: Some(ConfirmatoryMethod::SplitNb),
                n_perm: 20,
                n_splits: 5,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(opt.k_star >= 1);
        assert_eq!(opt.diagnostic.as_deref(), Some("split_exact"));
        assert_eq!(
            opt.stable_rank.unwrap().to_bits(),
            gate.stable_rank.to_bits()
        );
    }

    /// The sequence path already reports `k_star = 0` on these inputs (its
    /// first step does not reject); pinned so the `find_k` family stays
    /// consistent.
    #[test]
    fn sequence_first_component_truncation_returns_k_star_zero() {
        let (x, ys) = degenerate_ys(60, 8, 5);
        for (name, y) in &ys {
            for method in [
                ConfirmatoryMethod::E,
                ConfirmatoryMethod::RawPerm,
                ConfirmatoryMethod::SplitExact,
                ConfirmatoryMethod::SplitNb,
            ] {
                let opts = FindKSequenceOpts {
                    test_method: method,
                    n_perm: 50,
                    n_splits: 10,
                    force: true,
                    seed: Some(3),
                    ..Default::default()
                };
                let dense = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 3, None, opts).unwrap();
                let sparse =
                    spls1_find_k_sequence(x.as_ref(), y.as_ref(), 3, 4, None, opts).unwrap();
                for (path, r) in [("pls1", &dense), ("spls1", &sparse)] {
                    let tag = format!("{name} {path} {method:?}");
                    assert_eq!(r.k_star, 0, "{tag}");
                    assert_eq!(r.pvalues.nrows(), 3, "{tag}");
                    assert!(r.pvalues[0] >= r.alpha, "{tag}: p1={}", r.pvalues[0]);
                    assert!(r.pvalues[1].is_nan() && r.pvalues[2].is_nan(), "{tag}");
                }
            }
        }
    }

    #[test]
    fn optimal_score_diagnostic_rejected() {
        let (x, y) = synth(60, 5, 1, 4.0, 3);
        let err = pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
            FindKOptimalOpts {
                diagnostic: Some(ConfirmatoryMethod::Score),
                ..Default::default()
            },
        );
        assert!(matches!(err, Err(PlsKitError::InvalidArgument(_))));
    }

    /// `force` has to be reachable from the optimal entry point's diagnostic
    /// too, not only from the sequence API: n = 20 trips the gate, and only
    /// the opts field can hold NB in place. The gate is hoisted into
    /// `run_incremental_sequence`, so the diagnostic path inherits it and the
    /// echoed name says what ran (`"split_exact"` on the unforced call).
    #[test]
    fn optimal_diagnostic_force_is_settable_from_public_opts() {
        let (x, y) = synth(20, 5, 1, 5.0, 1);
        let opts = |force: bool| FindKOptimalOpts {
            selector: Selector::Bic,
            diagnostic: Some(ConfirmatoryMethod::SplitNb),
            n_splits: 10,
            force,
            seed: Some(7),
            ..Default::default()
        };
        let rerouted = pls1_find_k_optimal(x.as_ref(), y.as_ref(), 2, None, opts(false)).unwrap();
        assert_eq!(rerouted.diagnostic.as_deref(), Some("split_exact"));
        let forced = pls1_find_k_optimal(x.as_ref(), y.as_ref(), 2, None, opts(true)).unwrap();
        assert_eq!(forced.diagnostic.as_deref(), Some("split_nb"));
    }

    /// On an adequate design (n = 80) the `split_nb` gate clears, and the
    /// result still carries the rank it saw: `stable_rank` reports what the
    /// gate evaluated, not whether it fired.
    #[test]
    fn sequence_reports_pvalues_n_eff_and_gate_rank() {
        let (x, y) = synth(80, 5, 1, 5.0, 1);
        let r = pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            4,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitNb,
                n_splits: 30,
                alpha: 0.05,
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.pvalues.nrows(), 4);
        assert_eq!(r.test_method, "split_nb");
        assert!((r.n_eff - 80.0).abs() < 1e-9);
        assert!(r.stable_rank.is_some());
    }

    /// `force` has to be reachable from the public sequence API, not just from
    /// the crate-internal variant knob: n = 20 trips the gate, and only the
    /// opts field can hold NB in place.
    #[test]
    fn sequence_force_is_settable_from_public_opts() {
        let (x, y) = synth(20, 5, 1, 5.0, 1);
        let opts = |force: bool| FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitNb,
            n_splits: 10,
            alpha: 0.05,
            force,
            seed: Some(7),
            ..Default::default()
        };
        let rerouted = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 2, None, opts(false)).unwrap();
        assert_eq!(rerouted.test_method, "split_exact");
        let forced = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 2, None, opts(true)).unwrap();
        assert_eq!(forced.test_method, "split_nb");
        // Both carry what the gate saw — that is what makes the reroute
        // explainable to the caller, and `force` doesn't suppress it.
        assert_eq!(rerouted.stable_rank, forced.stable_rank);
        assert!(rerouted.stable_rank.is_some());
    }

    /// The sequence-level rank has to survive the trip out through
    /// `find_k_optimal`'s diagnostic branch, and only that branch produces it.
    #[test]
    fn optimal_diagnostic_carries_the_gate_rank() {
        let (x, y) = synth(20, 5, 1, 5.0, 1);
        let run = |diagnostic| {
            pls1_find_k_optimal(
                x.as_ref(),
                y.as_ref(),
                2,
                None,
                FindKOptimalOpts {
                    diagnostic,
                    n_splits: 10,
                    n_perm: 50,
                    seed: Some(7),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let gated = run(Some(ConfirmatoryMethod::SplitNb));
        assert_eq!(gated.diagnostic.as_deref(), Some("split_exact"));
        assert!(gated.stable_rank.is_some());
        assert!(run(Some(ConfirmatoryMethod::RawPerm)).stable_rank.is_none());
        assert!(run(None).stable_rank.is_none());
    }

    #[test]
    fn sequence_score_rejected() {
        let (x, y) = synth(60, 5, 1, 4.0, 3);
        let err = pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::Score,
                ..Default::default()
            },
        );
        assert!(matches!(err, Err(PlsKitError::InvalidArgument(_))));
    }

    #[test]
    fn spls1_find_k_optimal_dense_endpoint_bit_parity() {
        // keep = n_features must reproduce pls1_find_k_optimal exactly
        // (same seed → same RNG stream → identical floats).
        let (x, y) = synth(80, 5, 1, 5.0, 1);
        let opts = FindKOptimalOpts {
            selector: Selector::R2Se,
            seed: Some(7),
            ..Default::default()
        };
        let dense = pls1_find_k_optimal(x.as_ref(), y.as_ref(), 4, None, opts).unwrap();
        let sparse = spls1_find_k_optimal(x.as_ref(), y.as_ref(), 4, 5, None, opts).unwrap();
        assert_eq!(dense.k_star, sparse.k_star);
        assert_eq!(dense.seed, sparse.seed);
        let (dm, sm) = (dense.cv_scores.unwrap(), sparse.cv_scores.unwrap());
        assert_eq!(dm.len(), sm.len());
        for (k, v) in &dm {
            assert_eq!(v.to_bits(), sm[k].to_bits(), "cv_scores[{k}]");
        }
    }

    /// The sparse BIC sweep reports a score for every `k` in `1..=k_used`,
    /// selects its first minimum (as `select_bic`'s strict `<` does), and
    /// actually runs sparse: `keep = 2` changes the scores relative to the
    /// dense sweep on the same data.
    #[test]
    fn spls1_find_k_optimal_bic_runs_sparse() {
        let (x, y) = synth(60, 5, 2, 4.0, 3);
        let opts = FindKOptimalOpts {
            selector: Selector::Bic,
            seed: Some(13),
            ..Default::default()
        };
        let r = spls1_find_k_optimal(x.as_ref(), y.as_ref(), 4, 2, None, opts).unwrap();
        let bic = r.bic_scores.as_ref().expect("bic selector reports scores");
        let keys: Vec<usize> = bic.keys().copied().collect();
        assert!(
            !keys.is_empty() && keys == (1..=keys.len()).collect::<Vec<_>>(),
            "{keys:?}"
        );
        let argmin = bic
            .iter()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(k, _)| *k)
            .unwrap();
        assert_eq!(r.k_star, argmin);
        let dense = pls1_find_k_optimal(x.as_ref(), y.as_ref(), 4, None, opts).unwrap();
        let dense_bic = dense.bic_scores.unwrap();
        assert!(
            bic.iter()
                .any(|(k, v)| dense_bic.get(k).is_none_or(|d| d.to_bits() != v.to_bits())),
            "keep = 2 must change the BIC sweep"
        );
    }

    /// `keep` outside `1..=n_features` is rejected by both keep-taking
    /// `find_k` entry points.
    #[test]
    fn spls1_find_k_entries_reject_bad_keep() {
        let (x, y) = synth(60, 5, 1, 4.0, 3);
        for keep in [0_usize, 9] {
            let e = spls1_find_k_optimal(
                x.as_ref(),
                y.as_ref(),
                3,
                keep,
                None,
                FindKOptimalOpts::default(),
            );
            assert!(
                matches!(e, Err(PlsKitError::InvalidArgument(_))),
                "optimal keep={keep}: {e:?}"
            );
            let e = spls1_find_k_sequence(
                x.as_ref(),
                y.as_ref(),
                3,
                keep,
                None,
                FindKSequenceOpts::default(),
            );
            assert!(
                matches!(e, Err(PlsKitError::InvalidArgument(_))),
                "sequence keep={keep}: {e:?}"
            );
        }
    }

    #[test]
    fn spls1_find_k_sequence_dense_endpoint_bit_parity() {
        let (x, y) = synth(80, 5, 1, 5.0, 1);
        let opts = FindKSequenceOpts {
            test_method: ConfirmatoryMethod::SplitNb,
            n_splits: 30,
            alpha: 0.05,
            seed: Some(7),
            ..Default::default()
        };
        let dense = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 4, None, opts).unwrap();
        let sparse = spls1_find_k_sequence(x.as_ref(), y.as_ref(), 4, 5, None, opts).unwrap();
        assert_eq!(dense.k_star, sparse.k_star);
        assert_eq!(dense.seed, sparse.seed);
        for i in 0..4 {
            assert_eq!(
                dense.pvalues[i].to_bits(),
                sparse.pvalues[i].to_bits(),
                "pvalues[{i}]"
            );
        }
    }

    #[test]
    fn keep_grid_powers_of_two_with_endpoints() {
        assert_eq!(keep_grid(1), vec![1]);
        assert_eq!(keep_grid(6), vec![1, 2, 4, 6]);
        assert_eq!(keep_grid(8), vec![1, 2, 4, 8]);
        assert_eq!(keep_grid(100), vec![1, 2, 4, 8, 16, 32, 64, 100]);
    }

    /// The keep call site applies the 1-SE rule, not the argmax. The design
    /// (2 signal columns of 6, SNR 1.0) is chosen so the two differ: the
    /// best mean CV score sits at keep = 4, while keep = 2 is within one SE
    /// of it; on a strong-signal design both land on the same grid point and
    /// the rule would go unchecked.
    #[test]
    fn spls1_find_keep_optimal_selects_sparsest_within_1se() {
        let (x, y) = synth(80, 6, 2, 1.0, 1);
        let r = spls1_find_keep_optimal(
            x.as_ref(),
            y.as_ref(),
            1,
            None,
            FindKeepOptimalOpts {
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.keep_grid, vec![1, 2, 4, 6]);
        assert_eq!(r.k, 1);
        assert!((r.n_eff - 80.0).abs() < 1e-9);
        // Self-consistency of the 1-SE parsimony rule against the returned maps:
        // keep_star is the SMALLEST grid point within 1 SE of the best mean.
        let best = r
            .cv_scores
            .iter()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| *k)
            .unwrap();
        let threshold = r.cv_scores[&best] - r.cv_scores_se[&best];
        let expected = r
            .cv_scores
            .iter()
            .filter(|(_, v)| v.is_finite() && **v >= threshold)
            .map(|(k, _)| *k)
            .min()
            .unwrap();
        assert_ne!(
            best, r.keep_star,
            "fixture must separate the 1-SE pick from the argmax"
        );
        assert_eq!(r.keep_star, expected);
        assert!(r.keep_grid.contains(&r.keep_star));
    }

    /// `n = 2` is leave-one-out: `cv_n_folds`'s floor of 2 pushes the
    /// effective fold count back up to `n`, so every training fold has 1
    /// row, too few to fit anything. The CV score is then `NaN` or an
    /// uninformative constant (this fold's zero-variance training column
    /// happens to fit a trivial zero model, so the score comes out finite
    /// but still uninformative); either way `n = 2` must be rejected, the
    /// same degeneracy `raw_perm` rejects.
    #[test]
    fn spls1_find_keep_optimal_rejects_leave_one_out_at_n_eq_2() {
        let (x, y) = synth(2, 5, 1, 4.0, 2);
        let r = spls1_find_keep_optimal(
            x.as_ref(),
            y.as_ref(),
            1,
            None,
            FindKeepOptimalOpts::default(),
        );
        assert!(
            matches!(r, Err(PlsKitError::InvalidArgument(_))),
            "expected InvalidArgument for n = 2, got {r:?}"
        );
    }

    /// `k = 0` is `fit::boundary_tests::k_zero_is_invalid_argument_at_every_entry`.
    #[test]
    fn spls1_find_keep_optimal_rejects_k_above_n_features() {
        let (x, y) = synth(40, 5, 1, 4.0, 2);
        let e = spls1_find_keep_optimal(
            x.as_ref(),
            y.as_ref(),
            6,
            None,
            FindKeepOptimalOpts::default(),
        );
        assert!(matches!(e, Err(PlsKitError::KExceedsMax { .. })));
    }

    /// Layout invariance (D1): every `find_k` entry copies X column-major
    /// before it computes (CV folds through `prep_fold`, BIC and the gate
    /// through `standardize_weighted`, sequence steps through `standardize` /
    /// `col_major_or_copy`), so padded, row-major and reversed-column views
    /// give bit-identical results to the owned matrix, over the copy-free
    /// families (weighted and wide included) with and without
    /// `pre_standardized`.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn find_k_entries_are_layout_invariant() {
        fn push_map(v: &mut Vec<f64>, m: Option<&BTreeMap<usize, f64>>) {
            match m {
                Some(m) => {
                    v.push(m.len() as f64);
                    v.extend(m.values());
                }
                None => v.push(-1.0),
            }
        }
        fn push_optimal(v: &mut Vec<f64>, r: &FindKOptimalOutput) {
            v.extend([r.k_star as f64, r.n_eff, r.stable_rank.unwrap_or(f64::NAN)]);
            push_map(v, r.cv_scores.as_ref());
            push_map(v, r.cv_scores_se.as_ref());
            push_map(v, r.bic_scores.as_ref());
            if let Some(p) = &r.pvalues {
                v.extend((0..p.nrows()).map(|i| p[i]));
            }
        }
        crate::test_support::assert_families_layout_invariant(
            "find_k",
            |c| {
                let mut v = Vec::new();
                for (selector, diagnostic) in [
                    (Selector::R2Se, None),
                    (Selector::Bic, None),
                    (Selector::R2Se, Some(ConfirmatoryMethod::SplitNb)),
                    (Selector::Bic, Some(ConfirmatoryMethod::SplitExact)),
                ] {
                    let opts = FindKOptimalOpts {
                        selector,
                        diagnostic,
                        n_perm: 49,
                        n_splits: 10,
                        pre_standardized: c.pre,
                        seed: Some(7),
                        ..Default::default()
                    };
                    push_optimal(&mut v, &pls1_find_k_optimal(c.x, c.y, 3, c.w, opts)?);
                    push_optimal(&mut v, &spls1_find_k_optimal(c.x, c.y, 3, 3, c.w, opts)?);
                }
                let r = spls1_find_keep_optimal(
                    c.x,
                    c.y,
                    2,
                    c.w,
                    FindKeepOptimalOpts {
                        seed: Some(7),
                        ..Default::default()
                    },
                )?;
                v.extend([r.keep_star as f64, r.n_eff]);
                v.extend(r.cv_scores.values());
                v.extend(r.cv_scores_se.values());
                Ok(v)
            },
            Vec::clone,
        );
    }
}
