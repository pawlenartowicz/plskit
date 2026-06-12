//! Confirmatory PLS1 omnibus test at fixed K: five methods.

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::Pls1Model;

/// Which test statistic / resampling method to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmatoryMethod {
    /// Raw permutation CV R² test (`raw_perm`).
    RawPerm,
    /// Split-half NB test with Fisher-z correction (`split_nb`).
    SplitNb,
    /// Permutation-calibrated split-half test (`split_perm`).
    SplitPerm,
    /// Score test (closed-form, Welch-Satterthwaite χ² approximation).
    Score,
    /// Universal-inference split-LR e-value.
    E,
}

impl ConfirmatoryMethod {
    /// Public string identifier (`snake_case`) used in result objects and wrapper APIs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ConfirmatoryMethod::RawPerm => "raw_perm",
            ConfirmatoryMethod::SplitNb => "split_nb",
            ConfirmatoryMethod::SplitPerm => "split_perm",
            ConfirmatoryMethod::Score => "score",
            ConfirmatoryMethod::E => "e",
        }
    }
}

/// Input to `pls1_confirmatory_test`: either raw `(X, y, k)` or a pre-fitted model.
pub enum ConfirmatoryTestInput<'a> {
    // TODO: implement; until then, callers must use Raw
    /// Pre-fitted model. Core validates preconditions then returns `Internal`
    /// (wrappers must reassemble raw X+y before calling core).
    #[doc(hidden)]
    Model(&'a Pls1Model),
    /// Raw data with an explicit component count.
    Raw {
        /// Feature matrix `(n_samples, n_features)`.
        x: MatRef<'a, f64>,
        /// Target vector `(n_samples,)`.
        y: ColRef<'a, f64>,
        /// Number of components to test.
        k: usize,
        /// Optional per-observation weights. `None` means uniform weights.
        /// Normalized to mean 1 before use (spec §3.3–3.4).
        weights: Option<ColRef<'a, f64>>,
    },
}

/// Method-specific arguments for `pls1_confirmatory_test`.
///
/// The variant chosen *is* the method; per-method knobs live inside the variant.
/// Cross-cutting kwargs (`seed`, `pre_standardized`, …) live on
/// [`ConfirmatoryTestOpts`] alongside this enum.
#[derive(Debug, Clone, Copy)]
pub enum ConfirmatoryArgs {
    /// Raw permutation CV R² test.
    RawPerm {
        /// Number of permutations.
        n_perm: usize,
        /// Number of CV folds.
        n_folds: usize,
    },
    /// Split-half NB test with Fisher-z correction.
    SplitNb {
        /// Number of split-half repetitions.
        n_splits: usize,
    },
    /// Permutation-calibrated split-half test.
    SplitPerm {
        /// Number of permutations.
        n_perm: usize,
        /// Number of split-half repetitions per permutation.
        n_splits: usize,
    },
    /// Closed-form score test (Welch-Satterthwaite generalized χ²).
    Score,
    /// Universal-inference split-LR e-value.
    E,
}

impl ConfirmatoryArgs {
    /// The method tag this variant represents (e.g. `"raw_perm"`).
    #[must_use]
    pub fn method(&self) -> ConfirmatoryMethod {
        match self {
            ConfirmatoryArgs::RawPerm { .. } => ConfirmatoryMethod::RawPerm,
            ConfirmatoryArgs::SplitNb { .. } => ConfirmatoryMethod::SplitNb,
            ConfirmatoryArgs::SplitPerm { .. } => ConfirmatoryMethod::SplitPerm,
            ConfirmatoryArgs::Score => ConfirmatoryMethod::Score,
            ConfirmatoryArgs::E => ConfirmatoryMethod::E,
        }
    }

    /// Default args for a given method (used when the caller passes no
    /// method-specific kwargs).
    #[must_use]
    pub fn defaults_for(method: ConfirmatoryMethod) -> Self {
        match method {
            ConfirmatoryMethod::RawPerm => ConfirmatoryArgs::RawPerm {
                n_perm: 1000,
                n_folds: 5,
            },
            ConfirmatoryMethod::SplitNb => ConfirmatoryArgs::SplitNb { n_splits: 50 },
            ConfirmatoryMethod::SplitPerm => ConfirmatoryArgs::SplitPerm {
                n_perm: 1000,
                n_splits: 50,
            },
            ConfirmatoryMethod::Score => ConfirmatoryArgs::Score,
            ConfirmatoryMethod::E => ConfirmatoryArgs::E,
        }
    }
}

/// Knobs for the optional CI branch on `pls1_confirmatory_test`. When
/// `ConfirmatoryTestOpts.ci` is `Some`, the function runs an independent
/// subsampling pass (separate child-seed branch from the test pass) and
/// populates `ConfirmatoryTestOutput.ci`.
#[derive(Debug, Clone, Copy)]
pub struct CIOpts {
    /// Number of subsampling resamples. Must be ≥ 100.
    pub n_boot: usize,
    /// Subsample rate: `m = ceil(n^m_rate)`. Must satisfy `0.5 < m_rate < 0.95`.
    pub m_rate: f64,
    /// Nominal CI level (e.g. 0.95). Must satisfy `0.5 ≤ level ≤ 0.99`.
    pub level: f64,
    /// Maximum tolerable combined per-resample failure rate. Default `0.01`.
    /// Range `[0.0, 1.0]`. See `subsample::SubsampleOpts::max_failure_rate`.
    pub max_failure_rate: f64,
}

impl Default for CIOpts {
    fn default() -> Self {
        Self {
            n_boot: 1000,
            m_rate: 0.7,
            level: 0.95,
            max_failure_rate: 0.01,
        }
    }
}

/// Cross-cutting tuning knobs for `pls1_confirmatory_test`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct ConfirmatoryTestOpts {
    /// Method dispatch + per-method args.
    pub args: ConfirmatoryArgs,
    /// Caller asserts X and y are already standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
    pub disable_parallelism: bool,
    /// Print progress to stderr (reserved for future verbose mode).
    pub verbose: bool,
    /// Optional CI bundle. When `Some`, runs an independent subsampling pass
    /// after the test and populates `ConfirmatoryTestOutput.ci`.
    pub ci: Option<CIOpts>,
    /// Subsample-loop skip threshold for the `ci` branch (spec §6.3).
    /// Default `0.01`. The CI loop fails with `ResamplingDegenerate`
    /// if `skipped/total > max_skip_rate`.
    pub max_skip_rate: f64,
    /// Sparse keep-count plumbing for the `spls1` family: every inner fit
    /// (CV folds for `raw_perm`, split halves for `split_nb`/`split_perm`,
    /// the train half for `e`) runs sparse at this keep. `Score` ignores it —
    /// the score statistic `T = ‖X'y‖²` is fit-free. `None` (default) = dense.
    /// Wrapper surfaces do not expose this; it exists for
    /// `spls1_find_k_sequence`'s per-component tests (and Rust-level use).
    pub keep: Option<usize>,
}

impl Default for ConfirmatoryTestOpts {
    fn default() -> Self {
        Self {
            args: ConfirmatoryArgs::defaults_for(ConfirmatoryMethod::SplitNb),
            pre_standardized: false,
            seed: None,
            disable_parallelism: false,
            verbose: false,
            ci: None,
            max_skip_rate: 0.01,
            keep: None,
        }
    }
}

/// Result of `pls1_confirmatory_test`.
#[derive(Debug, Clone)]
pub struct ConfirmatoryTestOutput {
    /// p-value (or `min(1, 1/e)` for the `e` method).
    pub pvalue: f64,
    /// Observed test statistic (CV R² for `raw_perm`; mean Fisher-z back-transformed for `split_nb`;
    /// mean split-r for `split_perm`; `||X'y||²` for `score`; log-e for `e`).
    pub statistic: f64,
    /// Method name as a lowercase string (e.g. `"raw_perm"`, `"split_nb"`, `"e"`).
    pub method: String,
    /// Resolved number of components actually tested.
    pub k: usize,
    /// Number of `raw_perm` / `split_perm` iterations used. `None` when the method has no permutation count.
    pub n_perm: Option<usize>,
    /// Number of split-half repetitions used. `None` when the method has no split count.
    pub n_splits: Option<usize>,
    /// RNG seed actually used.
    pub seed: u64,
    /// CI bundle. `Some` when the caller passed `ConfirmatoryTestOpts.ci = Some(...)`.
    pub ci: Option<crate::subsample::ConfirmatoryCI>,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
}

/// Confirmatory PLS1 omnibus test at fixed K.
///
/// # Shapes
/// - `input` (Raw form): `x: (n_samples, n_features)`, `y: (n_samples,)`, `k: 1..=k_max`
///
/// # Errors
/// - `PlsKitError::Internal` for the `Model` input form (wrappers must reassemble raw X+y)
/// - `PlsKitError::DimensionMismatch` when row counts disagree
/// - `PlsKitError::KExceedsMax` when k > `d`
/// - `PlsKitError::NonFiniteInput` when X, y, or weights contain NaN/inf
/// - `PlsKitError::InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_pass_by_value)] // ConfirmatoryTestInput wraps non-Copy Pls1Model ref
pub fn pls1_confirmatory_test(
    input: ConfirmatoryTestInput<'_>,
    opts: ConfirmatoryTestOpts,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    let (x_ref, y_ref, k_resolved, weights_in) = match &input {
        ConfirmatoryTestInput::Raw { x, y, k, weights } => (*x, *y, *k, *weights),
        ConfirmatoryTestInput::Model(_) => {
            // Model form: wrappers must call us with Raw (they hold the original X reference).
            return Err(PlsKitError::Internal(
                "Model form not yet supported in core; wrapper must pass Raw".into(),
            ));
        }
    };

    let n = x_ref.nrows();
    if y_ref.nrows() != n {
        return Err(PlsKitError::DimensionMismatch {
            x: (n, x_ref.ncols()),
            y: y_ref.nrows(),
        });
    }
    crate::fit::check_finite_mat(x_ref)?;
    crate::fit::check_finite_col(y_ref)?;

    // Per-method count floors. Each method validates only the counts it uses:
    // resampling needs ≥1 permutation, split/CV calibration needs ≥2 of each.
    match opts.args {
        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => {
            if n_folds < 2 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_folds must be ≥ 2, got {n_folds}"
                )));
            }
            if n_perm < 1 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_perm must be ≥ 1, got {n_perm}"
                )));
            }
        }
        ConfirmatoryArgs::SplitNb { n_splits } => {
            if n_splits < 2 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_splits must be ≥ 2, got {n_splits}"
                )));
            }
        }
        ConfirmatoryArgs::SplitPerm { n_perm, n_splits } => {
            if n_splits < 2 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_splits must be ≥ 2, got {n_splits}"
                )));
            }
            if n_perm < 1 {
                return Err(PlsKitError::InvalidArgument(format!(
                    "n_perm must be ≥ 1, got {n_perm}"
                )));
            }
        }
        ConfirmatoryArgs::Score | ConfirmatoryArgs::E => {}
    }
    let k_max = x_ref.ncols();
    if k_resolved > k_max {
        return Err(PlsKitError::KExceedsMax {
            k: k_resolved,
            k_max,
        });
    }

    // Validate + normalize weights. Row-scaling pattern: materialize X̃ = √w' · X_std
    // and ỹ = √w' · y_std and run the unweighted statistics on (X̃, ỹ).
    let (w_norm, n_eff_val, _all_uniform) =
        crate::fit::validate_and_normalize_weights(weights_in, n, k_resolved)?;
    crate::fit::check_n_eff_for_k(n_eff_val, k_resolved, weights_in.is_some())?;

    if let Some(kp) = opts.keep {
        crate::fit::validate_keep(kp, x_ref.ncols())?;
        if opts.ci.is_some() {
            return Err(PlsKitError::InvalidArgument(
                "keep does not combine with ci: per-coordinate subsample CIs under \
                 selection are post-selection inference (see the spls1 spec)"
                    .into(),
            ));
        }
    }

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed)?;

    let (result, n_perm_out, n_splits_out) = match opts.args {
        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => (
            run_raw_perm(
                x_ref,
                y_ref,
                k_resolved,
                n_perm,
                n_folds,
                w_norm.as_ref().map(Col::as_ref),
                &opts,
                &mut rng,
            )?,
            Some(n_perm),
            None,
        ),
        ConfirmatoryArgs::SplitNb { n_splits } => (
            run_split_nb(
                x_ref,
                y_ref,
                k_resolved,
                n_splits,
                w_norm.as_ref().map(Col::as_ref),
                &opts,
                &mut rng,
            )?,
            None,
            Some(n_splits),
        ),
        ConfirmatoryArgs::SplitPerm { n_perm, n_splits } => (
            run_split_perm(
                x_ref,
                y_ref,
                k_resolved,
                n_perm,
                n_splits,
                w_norm.as_ref().map(Col::as_ref),
                &opts,
                &mut rng,
            )?,
            Some(n_perm),
            Some(n_splits),
        ),
        ConfirmatoryArgs::Score => (
            run_score(x_ref, y_ref, w_norm.as_ref().map(Col::as_ref), &opts)?,
            None,
            None,
        ),
        ConfirmatoryArgs::E => (
            run_e(
                x_ref,
                y_ref,
                k_resolved,
                w_norm.as_ref().map(Col::as_ref),
                &opts,
                &mut rng,
            )?,
            None,
            None,
        ),
    };

    let ci_payload = if let Some(ci_opts) = opts.ci {
        let sub_opts = crate::subsample::SubsampleOpts {
            n_boot: ci_opts.n_boot,
            m_rate: ci_opts.m_rate,
            level: ci_opts.level,
            pre_standardized: opts.pre_standardized,
            disable_parallelism: opts.disable_parallelism,
            max_failure_rate: ci_opts.max_failure_rate,
            max_skip_rate: opts.max_skip_rate,
        };
        sub_opts.validate()?;

        // Independent child-seed branch — derive a second child RNG from the
        // post-test parent state. This guarantees stream non-interference
        // between test path and CI path while keeping a single user-facing seed.
        let mut ci_rng = {
            use rand::Rng;
            crate::rng::child_rng(rng.next_u64())
        };

        // Reference fit on full data.
        let fit_ref = {
            use crate::fit::{pls1_fit, FitOpts, KSpec};
            pls1_fit(
                x_ref,
                y_ref,
                KSpec::Fixed(k_resolved),
                w_norm.as_ref().map(Col::as_ref),
                FitOpts {
                    pre_standardized: opts.pre_standardized,
                    ..FitOpts::default()
                },
            )?
        };

        // leverage_ref[j] = diag(W_ref (W_ref' W_ref)^-1 W_ref').
        let leverage_ref = crate::linalg::leverage_diag(fit_ref.w_star.as_ref());
        Some(crate::subsample::pls1_subsample_inference_confirmatory(
            x_ref,
            y_ref,
            k_resolved,
            fit_ref.w_star.as_ref(),
            fit_ref.beta.as_ref(),
            &leverage_ref,
            sub_opts,
            w_norm.as_ref().map(Col::as_ref),
            &mut ci_rng,
        )?)
    } else {
        None
    };

    Ok(ConfirmatoryTestOutput {
        pvalue: result.pvalue,
        statistic: result.statistic,
        method: opts.args.method().as_str().to_owned(),
        k: k_resolved,
        n_perm: n_perm_out,
        n_splits: n_splits_out,
        seed: seed_used,
        ci: ci_payload,
        n_eff: n_eff_val,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal result carrier (not pub)
// ──────────────────────────────────────────────────────────────────────────────

struct RunResult {
    pvalue: f64,
    statistic: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 4: `raw_perm` CV R² test
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
fn run_raw_perm(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_perm: usize,
    n_folds: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use rand::seq::SliceRandom;

    let n = x.nrows();

    // Fixed fold indices: shuffle once, then split.
    // Weights are passed through to pls1_cv_r2 which re-normalizes per fold.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    let folds = crate::linalg::fold_split(&indices, n_folds);

    let cv_r2_obs = pls1_cv_r2(x, y, k, &folds, w_norm, opts.keep)?;

    let nulls_vec = crate::resample::parallel_for_each_seeded(
        rng,
        n_perm,
        opts.disable_parallelism,
        |_, child| {
            // Permute y rows; weights stay tied to row indices (not permuted).
            let perm = crate::resample::permute_indices(n, child);
            let y_perm = Col::<f64>::from_fn(n, |i| y[perm[i]]);
            pls1_cv_r2(x, y_perm.as_ref(), k, &folds, w_norm, opts.keep).unwrap_or(f64::NAN)
        },
    );

    // A failed null fit surfaces as NaN (parallel_for_each_seeded has no error
    // channel); count it as an exceedance so it biases p upward, never downward.
    // That fail-soft rule is only sound for occasional failures: past half the
    // nulls, p saturates toward 1 and the test silently stops measuring
    // anything — error out instead (hardcoded 1/2; not worth a knob).
    let nan_nulls = nulls_vec.iter().filter(|v| v.is_nan()).count();
    if nan_nulls * 2 > n_perm {
        return Err(PlsKitError::PermNullDegenerate {
            failed: nan_nulls,
            total: n_perm,
        });
    }
    let exceedances = nulls_vec
        .iter()
        .filter(|v| v.is_nan() || **v >= cv_r2_obs)
        .count();
    let p = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);

    Ok(RunResult {
        pvalue: p,
        statistic: cv_r2_obs,
    })
}

/// K-fold cross-validated R² for PLS1.
///
/// CV-R² convention (anchor — `find_k::select_cv` cites this): this function
/// pools `SS_res` and `SS_tot` across all folds and forms a single R² from the
/// pooled sums, with **unweighted** validation residuals even when training is
/// weighted. `select_cv` in `find_k.rs` deliberately uses the other convention —
/// per-fold **weighted-validation** R² averaged across folds — because k-selection
/// wants each fold weighted on its own scale. The two are not interchangeable;
/// each method owns the convention appropriate to its statistic.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
fn pls1_cv_r2(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    folds: &[Vec<usize>],
    weights: Option<ColRef<'_, f64>>,
    keep: Option<usize>,
) -> PlsKitResult<f64> {
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{
        col_row_subset, normalize_weights, row_subset, standardize, standardize1,
        standardize1_weighted, standardize_apply, standardize_weighted,
    };

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for (fi, val_idx) in folds.iter().enumerate() {
        let train_idx: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != fi)
            .flat_map(|(_, f)| f.iter().copied())
            .collect();

        let x_tr = row_subset(x, &train_idx);
        let y_tr = col_row_subset(y, &train_idx);
        let x_val = row_subset(x, val_idx);
        let y_val = col_row_subset(y, val_idx);

        // Slice and re-normalize weights for the training fold.
        let w_tr_norm: Option<Col<f64>> = weights.map(|w| {
            let w_slice = col_row_subset(w, &train_idx);
            // Re-normalize so weights mean = 1 within this fold.
            normalize_weights(w_slice.as_ref())
                .unwrap_or_else(|| Col::from_fn(train_idx.len(), |_| 1.0))
        });
        let w_tr_ref: Option<ColRef<'_, f64>> = w_tr_norm.as_ref().map(Col::as_ref);

        let (xs_tr, x_mean, x_scale) = if let Some(w) = w_tr_ref {
            standardize_weighted(x_tr.as_ref(), Some(w))
        } else {
            standardize(x_tr.as_ref())
        };
        let xs_val = standardize_apply(x_val.as_ref(), x_mean.as_ref(), x_scale.as_ref());
        let (ys_tr, y_mean, y_scale) = if let Some(w) = w_tr_ref {
            standardize1_weighted(y_tr.as_ref(), Some(w))
        } else {
            standardize1(y_tr.as_ref())
        };
        let ys_val = Col::<f64>::from_fn(y_val.nrows(), |i| (y_val[i] - y_mean) / y_scale);

        let m = pls1_fit(
            xs_tr.as_ref(),
            ys_tr.as_ref(),
            KSpec::Fixed(k),
            w_tr_ref,
            FitOpts {
                pre_standardized: true,
                // check_n_eff: false — per-fold slice may have low n_eff; let the math degrade
                // and rely on the parent statistic to absorb noise (see Option B contract)
                check_n_eff: false,
                // Seq inside the per-fold worker — outer Rayon owns the threadpool.
                par: crate::fit::ParChoice::Seq,
                keep,
            },
        )?;

        let y_pred: Col<f64> = &xs_val * &m.coef;

        let n_val = ys_val.nrows();
        let mean_val: f64 = (0..n_val).map(|i| ys_val[i]).sum::<f64>() / n_val as f64;

        ss_res += (0..n_val)
            .map(|i| (y_pred[i] - ys_val[i]).powi(2))
            .sum::<f64>();
        ss_tot += (0..n_val)
            .map(|i| (ys_val[i] - mean_val).powi(2))
            .sum::<f64>();
    }

    Ok(if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 5: `split_nb` and `split_perm` split-half tests
// ──────────────────────────────────────────────────────────────────────────────

/// Compute J split-half Pearson r values. Port of `_tests.py:161-186`.
///
/// Split fraction is hardcoded 50/50: NB calibration assumes balanced
/// halves and there is no scientific reason to vary it.
///
/// `x`/`y` are the **raw** (un-√w-scaled) inputs; `w_norm` carries the
/// mean-1-normalized weights when present. Each half is standardized with
/// **weighted** moments and then √w' row-scaled (Convention A, matching
/// `pls1_fit`). The reported statistic is the unweighted Pearson r on the
/// resulting √w-scaled test-half scores and test-half y — *not* the weighted
/// Pearson r on the original data, because the test-half centering subtracts
/// the plain mean of the √w-scaled values rather than the weighted mean. The
/// identical transform is applied to observed and permuted data, and the
/// per-half weights stay tied to their rows (never permuted), so permutation
/// calibration still holds.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::unnecessary_wraps)] // signature must match callers that use ?
fn split_half_correlations(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    disable_parallelism: bool,
    keep: Option<usize>,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<Col<f64>> {
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{
        col_row_subset, normalize_weights, row_subset, standardize, standardize1,
        standardize1_weighted, standardize_apply, standardize_weighted,
    };
    use crate::resample::{one_split, split_sizes};

    let n = x.nrows();
    // n-3 test-floor in split_sizes can drop n_train below k+2; below that the
    // per-half fit degrades to a silent r=0. Reject up front. See split_sizes.
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, _) = split_sizes(n, k);

    let r_vec = crate::resample::parallel_for_each_seeded(
        rng,
        n_splits,
        disable_parallelism,
        |_, child| {
            let (tr, te) = one_split(n, n_train, child);
            let x_tr = row_subset(x, &tr);
            let y_tr = col_row_subset(y, &tr);
            let x_te = row_subset(x, &te);
            let y_te = col_row_subset(y, &te);

            // Per-half weights, re-normalized to mean 1 within the half (mirrors
            // pls1_cv_r2's per-fold renormalization). √w of each half is applied
            // *after* weighted standardization so the per-half fit matches what
            // pls1_fit would compute on that half's (X, y, w).
            let w_tr: Option<Col<f64>> = w_norm.map(|w| {
                let s = col_row_subset(w, &tr);
                normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(tr.len(), |_| 1.0))
            });
            let w_te: Option<Col<f64>> = w_norm.map(|w| {
                let s = col_row_subset(w, &te);
                normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(te.len(), |_| 1.0))
            });
            let w_tr_ref = w_tr.as_ref().map(Col::as_ref);

            let (xs_tr, x_mean, x_scale) = if let Some(w) = w_tr_ref {
                standardize_weighted(x_tr.as_ref(), Some(w))
            } else {
                standardize(x_tr.as_ref())
            };
            let xs_te = standardize_apply(x_te.as_ref(), x_mean.as_ref(), x_scale.as_ref());
            let (ys_tr, _, _) = if let Some(w) = w_tr_ref {
                standardize1_weighted(y_tr.as_ref(), Some(w))
            } else {
                standardize1(y_tr.as_ref())
            };

            // √w' row-scaling on top of weighted standardization (Convention A).
            let (xs_tr, ys_tr) = match w_tr_ref {
                Some(w) => (
                    Mat::<f64>::from_fn(xs_tr.nrows(), xs_tr.ncols(), |i, j| {
                        xs_tr[(i, j)] * w[i].sqrt()
                    }),
                    Col::<f64>::from_fn(ys_tr.nrows(), |i| ys_tr[i] * w[i].sqrt()),
                ),
                None => (xs_tr, ys_tr),
            };
            let xs_te = match w_te.as_ref() {
                Some(w) => Mat::<f64>::from_fn(xs_te.nrows(), xs_te.ncols(), |i, j| {
                    xs_te[(i, j)] * w[i].sqrt()
                }),
                None => xs_te,
            };

            let Ok(m) = pls1_fit(
                xs_tr.as_ref(),
                ys_tr.as_ref(),
                KSpec::Fixed(k),
                None,
                FitOpts {
                    pre_standardized: true,
                    // check_n_eff: false — per-half fit may truncate (small half,
                    // sparse keep); a truncated model still yields a valid r at
                    // k_used, which beats silently recording r=0 via the Err arm.
                    check_n_eff: false,
                    // Seq inside the per-half worker — outer Rayon owns the threadpool.
                    par: crate::fit::ParChoice::Seq,
                    keep,
                },
            ) else {
                return 0.0;
            };

            // scores on test half = X_te * coef. Both scores (via √w-scaled
            // xs_te) and y_te carry √w' so the Pearson r is taken on √w-scaled
            // data, matching Convention A.
            let scores_te: Col<f64> = &xs_te * &m.coef;
            let n_te = scores_te.nrows();
            let y_te: Col<f64> = match w_te.as_ref() {
                Some(w) => Col::<f64>::from_fn(n_te, |i| y_te[i] * w[i].sqrt()),
                None => y_te,
            };

            let s_mean: f64 = (0..n_te).map(|i| scores_te[i]).sum::<f64>() / n_te as f64;
            let y_mean: f64 = (0..n_te).map(|i| y_te[i]).sum::<f64>() / n_te as f64;

            let scores_c = Col::<f64>::from_fn(n_te, |i| scores_te[i] - s_mean);
            let y_c = Col::<f64>::from_fn(n_te, |i| y_te[i] - y_mean);

            let ss_s: f64 = (0..n_te).map(|i| scores_c[i] * scores_c[i]).sum();
            let ss_y: f64 = (0..n_te).map(|i| y_c[i] * y_c[i]).sum();

            if ss_s < 1e-15 || ss_y < 1e-15 {
                return 0.0;
            }

            let cross: f64 = (0..n_te).map(|i| scores_c[i] * y_c[i]).sum();
            let r = cross / (ss_s * ss_y).sqrt();
            r.clamp(-1.0, 1.0)
        },
    );

    Ok(Col::<f64>::from_fn(r_vec.len(), |i| r_vec[i]))
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn run_split_nb(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use crate::resample::split_sizes;
    let n = x.nrows();
    let (n_train, n_test) = split_sizes(n, k);
    // Raw (X, y) and weights flow into split_half_correlations, which does the
    // per-half weighted-standardize-then-√w (Convention A) internally.
    let r_splits = split_half_correlations(
        x,
        y,
        k,
        n_splits,
        w_norm,
        opts.disable_parallelism,
        opts.keep,
        rng,
    )?;
    let (p, mean_r, _t_stat, _df) = nb_test(&r_splits, n_train, n_test);
    Ok(RunResult {
        pvalue: p,
        statistic: mean_r,
    })
}

/// NB t-test on Fisher-z transforms. Port of `_core.py:32-89`.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn nb_test(stats: &Col<f64>, n_train: usize, n_test: usize) -> (f64, f64, f64, f64) {
    let j = stats.nrows() as f64;
    // Fisher-z transform
    let z_vec: Vec<f64> = (0..stats.nrows())
        .map(|i| stats[i].clamp(-0.9999, 0.9999).atanh())
        .collect();
    let z_mean: f64 = z_vec.iter().sum::<f64>() / j;
    let z_var: f64 = z_vec.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / (j - 1.0);
    let z_std = z_var.sqrt();
    // Clamp se to the degeneracy floor rather than branching to an exact p=0:
    // a vanishing spread (all split-r identical) still yields a finite, tiny p
    // through the t survival function instead of an unattainable exact zero.
    // The z_mean ≤ 0 side stays conservative — t ≤ 0 maps to p ≥ 0.5.
    let se = (z_std * (1.0 / j + n_test as f64 / n_train as f64).sqrt()).max(1e-15);
    let t = z_mean / se;
    let p = crate::linalg::t_sf(t, j - 1.0);
    (p, z_mean.tanh(), t, j - 1.0)
}

#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn run_split_perm(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    n_perm: usize,
    n_splits: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    let n = x.nrows();
    // Raw (X, y, weights) flow into split_half_correlations (Convention A
    // weighted-standardize-then-√w internally); the permutation loop reuses the
    // raw x and permutes raw y rows.
    let r_obs = split_half_correlations(
        x,
        y,
        k,
        n_splits,
        w_norm,
        opts.disable_parallelism,
        opts.keep,
        rng,
    )?;
    let j = r_obs.nrows();
    let mean_obs: f64 = if j > 0 {
        (0..j).map(|i| r_obs[i]).sum::<f64>() / j as f64
    } else {
        0.0
    };

    let null_means_vec = crate::resample::parallel_for_each_seeded(
        rng,
        n_perm,
        opts.disable_parallelism,
        |_, outer_rng| {
            let perm = crate::resample::permute_indices(n, outer_rng);
            // Permute raw y rows; weights stay tied to row positions (w_norm is
            // passed unpermuted), so w[i] always pairs with destination row i.
            let y_perm = Col::<f64>::from_fn(n, |i| y[perm[i]]);
            match split_half_correlations(
                x,
                y_perm.as_ref(),
                k,
                n_splits,
                w_norm,
                opts.disable_parallelism,
                opts.keep,
                outer_rng,
            ) {
                Ok(r_null) => {
                    let jn = r_null.nrows();
                    if jn > 0 {
                        (0..jn).map(|i| r_null[i]).sum::<f64>() / jn as f64
                    } else {
                        0.0
                    }
                }
                // A failed null replicate counts as an exceedance so it biases
                // p upward, never downward (mirrors run_raw_perm — change
                // together). Effectively unreachable: the observed call above
                // already failed identically behind the n < k+5 guard.
                Err(_) => f64::NAN,
            }
        },
    );

    let exceedances = null_means_vec
        .iter()
        .filter(|v| v.is_nan() || **v >= mean_obs)
        .count();
    let p = (exceedances as f64 + 1.0) / (n_perm as f64 + 1.0);

    Ok(RunResult {
        pvalue: p,
        statistic: mean_obs,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 6: score test (Welch-Satterthwaite generalized χ²)
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::many_single_char_names)]
#[allow(clippy::unnecessary_wraps)] // signature must match other run_* helpers returning PlsKitResult
fn run_score(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
) -> PlsKitResult<RunResult> {
    use crate::linalg::{standardize, standardize1, standardize1_weighted, standardize_weighted};

    let n = x.nrows();

    // Standardize with weighted moments when weights are present (matching
    // pls1_fit's path); the √w row-scaling below then sits on top.
    let (xs, _, _) = if opts.pre_standardized {
        let d = x.ncols();
        (
            Mat::<f64>::from_fn(n, d, |i, j| x[(i, j)]),
            Col::<f64>::zeros(d),
            Col::<f64>::from_fn(d, |_| 1.0),
        )
    } else if w_norm.is_some() {
        standardize_weighted(x, w_norm)
    } else {
        standardize(x)
    };

    let (ys, _, _) = if opts.pre_standardized {
        (Col::<f64>::from_fn(n, |i| y[i]), 0.0_f64, 1.0_f64)
    } else if w_norm.is_some() {
        standardize1_weighted(y, w_norm)
    } else {
        standardize1(y)
    };

    // When weights are present, further row-scale the standardized data by √w'.
    // T_w = ||X̃'ỹ||² where X̃ = diag(√w')·X_std, ỹ = diag(√w')·y_std.
    // This equals the unweighted T on (X̃, ỹ).
    let (xs_eff, ys_eff) = if let Some(w) = w_norm {
        let xs_w = Mat::<f64>::from_fn(n, xs.ncols(), |i, j| xs[(i, j)] * w[i].sqrt());
        let ys_w = Col::<f64>::from_fn(n, |i| ys[i] * w[i].sqrt());
        (xs_w, ys_w)
    } else {
        (xs, ys)
    };

    // T_obs = ||X'y||² = y'XX'y
    let xy: Col<f64> = xs_eff.transpose() * &ys_eff;
    let t_obs: f64 = (0..xy.nrows()).map(|i| xy[i].powi(2)).sum::<f64>();

    // Eigenvalues of the smaller Gram matrix (X'X for d≤n, XX' otherwise).
    let nn = xs_eff.nrows();
    let d = xs_eff.ncols();
    let lambdas: Col<f64> = if d <= nn {
        let gram: Mat<f64> = xs_eff.transpose() * xs_eff.as_ref();
        eigenvalues_symmetric(gram.as_ref())
    } else {
        let gram: Mat<f64> = xs_eff.as_ref() * xs_eff.transpose();
        eigenvalues_symmetric(gram.as_ref())
    };

    // Welch-Satterthwaite: T ~ a·χ²(df) approximately.
    let s1: f64 = (0..lambdas.nrows()).map(|i| lambdas[i]).sum();
    let s2: f64 = (0..lambdas.nrows()).map(|i| lambdas[i].powi(2)).sum();

    if s1.abs() < 1e-15 || s2 < 1e-30 {
        return Ok(RunResult {
            pvalue: 1.0,
            statistic: t_obs,
        });
    }

    let scale = s2 / s1;
    let df = s1 * s1 / s2;
    let p = chi2_sf(t_obs / scale, df);

    Ok(RunResult {
        pvalue: p,
        statistic: t_obs,
    })
}

/// Symmetric eigenvalues via faer's `self_adjoint_eigen`. Returns eigenvalues ascending.
/// `Side::Lower` is pinned for byte-parity stability.
fn eigenvalues_symmetric(a: MatRef<'_, f64>) -> Col<f64> {
    // Returns Result<SelfAdjointEigen<f64>, EvdError>; unwrap is safe for
    // PD/PSD Gram matrices. If the matrix is degenerate (n=0 or all-zero),
    // return a zero Col — the caller guards s1 < 1e-15.
    match a.self_adjoint_eigen(faer::Side::Lower) {
        Ok(eig) => eig.S().column_vector().to_owned(),
        Err(_) => Col::<f64>::zeros(a.nrows()),
    }
}

/// Survival function of `χ²(df)` at `x`. Uses the regularized upper incomplete
/// gamma Q(a, z) directly so extreme tails are not lost to a `1-(1-Q)` round trip.
#[allow(clippy::many_single_char_names)]
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let a = df / 2.0;
    let z = x / 2.0;
    gammainc_upper(a, z)
}

/// Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x). Numerical Recipes
/// §6.2. The continued-fraction branch returns Q directly (no `1−` round trip),
/// so χ² survival values below ~1e-16 stay representable instead of collapsing
/// to 0.0. The series branch evaluates P and returns its complement.
#[allow(clippy::many_single_char_names)]
fn gammainc_upper(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    let log_pref = a * x.ln() - x - crate::linalg::lgamma(a);
    if x < a + 1.0 {
        // Series expansion for the lower tail P(a, x); Q = 1 − P. Identical to
        // the prior series branch — only the final complement differs.
        let mut term = 1.0 / a;
        let mut sum = term;
        for i in 1_i32..200 {
            term *= x / (a + f64::from(i));
            sum += term;
            if term.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        1.0 - sum * log_pref.exp()
    } else {
        // Continued fraction evaluates Q(a, x) directly.
        let tiny = 1e-30;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1_i32..200 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-14 {
                break;
            }
        }
        h * log_pref.exp()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 7: universal-inference split-LR e-value
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
fn run_e(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    w_norm: Option<ColRef<'_, f64>>,
    opts: &ConfirmatoryTestOpts,
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<RunResult> {
    use crate::fit::{pls1_fit, FitOpts, KSpec};
    use crate::linalg::{
        col_row_subset, normalize_weights, row_subset, standardize, standardize1,
        standardize1_weighted, standardize_apply, standardize_weighted,
    };
    use crate::resample::{one_split, split_sizes};

    let n = x.nrows();
    // Mirror of split_half_correlations' guard — change together.
    if n < k + 5 {
        return Err(PlsKitError::InvalidArgument(format!(
            "n={n} too small for k={k} under split methods (need n ≥ k+5)"
        )));
    }
    let (n_train, _) = split_sizes(n, k);
    let (tr, te) = one_split(n, n_train, rng);

    // Split the raw data, then per-half weighted-standardize-then-√w
    // (Convention A, matching split_half_correlations / pls1_fit). Per-half
    // weights are re-normalized to mean 1 within the half (mirrors pls1_cv_r2).
    let x_tr = row_subset(x, &tr);
    let y_tr = col_row_subset(y, &tr);
    let x_te = row_subset(x, &te);
    let y_te = col_row_subset(y, &te);

    let w_tr: Option<Col<f64>> = w_norm.map(|w| {
        let s = col_row_subset(w, &tr);
        normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(tr.len(), |_| 1.0))
    });
    let w_te: Option<Col<f64>> = w_norm.map(|w| {
        let s = col_row_subset(w, &te);
        normalize_weights(s.as_ref()).unwrap_or_else(|| Col::from_fn(te.len(), |_| 1.0))
    });
    let w_tr_ref = w_tr.as_ref().map(Col::as_ref);

    let (xs_tr, x_mean, x_scale) = if let Some(w) = w_tr_ref {
        standardize_weighted(x_tr.as_ref(), Some(w))
    } else {
        standardize(x_tr.as_ref())
    };
    let xs_te = standardize_apply(x_te.as_ref(), x_mean.as_ref(), x_scale.as_ref());
    let (ys_tr, y_mean, y_scale) = if let Some(w) = w_tr_ref {
        standardize1_weighted(y_tr.as_ref(), Some(w))
    } else {
        standardize1(y_tr.as_ref())
    };
    let n_te = y_te.nrows();
    let ys_te = Col::<f64>::from_fn(n_te, |i| (y_te[i] - y_mean) / y_scale);

    // √w' row-scaling on top of weighted standardization, train and test halves.
    let (xs_tr, ys_tr) = match w_tr_ref {
        Some(w) => (
            Mat::<f64>::from_fn(xs_tr.nrows(), xs_tr.ncols(), |i, j| {
                xs_tr[(i, j)] * w[i].sqrt()
            }),
            Col::<f64>::from_fn(ys_tr.nrows(), |i| ys_tr[i] * w[i].sqrt()),
        ),
        None => (xs_tr, ys_tr),
    };
    let (xs_te, ys_te) = match w_te.as_ref() {
        Some(w) => (
            Mat::<f64>::from_fn(xs_te.nrows(), xs_te.ncols(), |i, j| {
                xs_te[(i, j)] * w[i].sqrt()
            }),
            Col::<f64>::from_fn(n_te, |i| ys_te[i] * w[i].sqrt()),
        ),
        None => (xs_te, ys_te),
    };

    let m = pls1_fit(
        xs_tr.as_ref(),
        ys_tr.as_ref(),
        KSpec::Fixed(k),
        None,
        FitOpts {
            pre_standardized: true,
            // check_n_eff: false — train-half refit; n_eff was validated at the
            // top-level entry, and the e-value remains valid at a truncated k_used.
            check_n_eff: false,
            keep: opts.keep,
            ..FitOpts::default()
        },
    )?;

    let y_pred: Col<f64> = &xs_te * &m.coef;

    // Universal inference fixes the numerator density on the training half:
    // σ²_alt is the residual MLE from predicting the training X with the fitted
    // model, evaluated against training y. Computing it on the test half would
    // make the likelihood ratio data-dependent and break the e-value guarantee.
    let n_tr = ys_tr.nrows();
    let y_pred_tr: Col<f64> = &xs_tr * &m.coef;
    let sigma2_alt: f64 = (0..n_tr)
        .map(|i| (ys_tr[i] - y_pred_tr[i]).powi(2))
        .sum::<f64>()
        / n_tr as f64;

    // σ² under null: variance of test y
    let mean_te: f64 = (0..n_te).map(|i| ys_te[i]).sum::<f64>() / n_te as f64;
    let sigma2_null: f64 =
        (0..n_te).map(|i| (ys_te[i] - mean_te).powi(2)).sum::<f64>() / n_te as f64;

    let n_te_f = n_te as f64;
    // Gaussian log-likelihoods with MLE variance
    let ll = |sigma2: f64, residuals_sq_sum: f64| -> f64 {
        let s = sigma2.max(1e-30);
        -0.5 * n_te_f * (2.0 * std::f64::consts::PI * s).ln() - 0.5 * residuals_sq_sum / s
    };

    let resid_alt_ss: f64 = (0..n_te).map(|i| (ys_te[i] - y_pred[i]).powi(2)).sum();
    let resid_null_ss: f64 = (0..n_te).map(|i| (ys_te[i] - mean_te).powi(2)).sum();

    let ll_alt = ll(sigma2_alt, resid_alt_ss);
    let ll_null = ll(sigma2_null, resid_null_ss);

    let log_e = ll_alt - ll_null;
    // Clip e below 1 so that p = 1/e ≤ 1.
    let e = log_e.exp().max(1.0);
    let p = (1.0 / e).min(1.0);

    // opts unused by e method; pre_standardized has no effect (re-standardizes each half by
    // design); disable_parallelism moot (single split, no inner loop).
    let _ = opts;

    Ok(RunResult {
        pvalue: p,
        statistic: log_e,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{pls1_fit, FitOpts, KSpec};

    fn synth_with_signal(n: usize, d: usize, snr: f64, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta = Col::<f64>::from_fn(d, |j| if j < 3 { 1.0 } else { 0.0 });
        let signal: Col<f64> = &x * &beta;
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |i| signal[i] * snr + noise[i]);
        (x, y)
    }

    fn synth_no_signal(n: usize, d: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
        (x, y)
    }

    // Smoke: ensure we can fit and immediately confirm.
    #[test]
    fn fit_then_confirm_smoke() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 1);
        let _ = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(2),
            None,
            FitOpts::default(),
        )
        .unwrap();
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!((0.0..=1.0).contains(&r.pvalue));
    }

    // ── raw_perm tests ───────────────────────────────────────────────────────

    #[test]
    fn raw_perm_calibration_under_h0() {
        let (x, y) = synth_no_signal(40, 5, 99);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 200,
                    n_folds: 5,
                },
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "raw_perm");
        assert!((0.0..=1.0).contains(&r.pvalue));
        // No assertion on p — calibration tested in plskit-py statistical suite.
    }

    #[test]
    fn raw_perm_rejects_under_signal() {
        let (x, y) = synth_with_signal(80, 6, 5.0, 11);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 3,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::RawPerm {
                    n_perm: 200,
                    n_folds: 5,
                },
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.pvalue < 0.05, "p={}", r.pvalue);
    }

    // ── split_nb and split_perm tests ───────────────────────────────────────

    #[test]
    fn split_nb_rejects_under_signal() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                seed: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "split_nb");
        assert!(r.pvalue < 0.1, "p={}", r.pvalue);
    }

    #[test]
    fn split_perm_runs_with_signal() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 23);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitPerm {
                    n_perm: 100,
                    n_splits: 20,
                },
                seed: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "split_perm");
        assert!((0.0..=1.0).contains(&r.pvalue));
    }

    // ── Score test ───────────────────────────────────────────────────────────

    #[test]
    fn score_returns_bounded_p() {
        let (x, y) = synth_no_signal(50, 6, 31);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::Score,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "score");
        assert!(r.pvalue >= 0.0 && r.pvalue <= 1.0);
    }

    // ── E-value test ─────────────────────────────────────────────────────────

    #[test]
    fn e_returns_bounded_p() {
        let (x, y) = synth_with_signal(80, 5, 3.0, 41);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::E,
                seed: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.method, "e");
        assert!(r.pvalue >= 0.0 && r.pvalue <= 1.0);
        // Universal inference always satisfies P(reject | H0) ≤ α exactly,
        // so under signal we expect p < 0.5 typically.
        assert!(r.pvalue < 0.5, "p={}", r.pvalue);
    }

    #[test]
    fn ci_branch_populates_ci_field_when_requested() {
        let (x, y) = synth_with_signal(80, 5, 4.0, 99);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                seed: Some(7),
                ci: Some(CIOpts {
                    n_boot: 200,
                    m_rate: 0.7,
                    level: 0.95,
                    max_failure_rate: 0.01,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let ci = r.ci.expect("ci should be populated");
        assert_eq!(ci.n_boot, 200);
        assert_eq!(ci.beta_sign_z.len(), 5);
    }

    #[test]
    fn ci_none_keeps_ci_field_none() {
        let (x, y) = synth_with_signal(80, 5, 4.0, 99);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                seed: Some(7),
                ci: None,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.ci.is_none());
    }

    #[test]
    fn confirmatory_keep_dense_endpoint_bit_parity() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let mk = |keep: Option<usize>| {
            pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: 2,
                    weights: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                    seed: Some(2),
                    keep,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let dense = mk(None);
        let endpoint = mk(Some(5));
        assert_eq!(dense.pvalue.to_bits(), endpoint.pvalue.to_bits());
        assert_eq!(dense.statistic.to_bits(), endpoint.statistic.to_bits());
    }

    #[test]
    fn confirmatory_sparse_split_nb_runs() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let r = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                seed: Some(2),
                keep: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert!((0.0..=1.0).contains(&r.pvalue));
    }

    #[test]
    fn confirmatory_rejects_keep_with_ci() {
        let (x, y) = synth_with_signal(80, 5, 4.0, 99);
        let e = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                seed: Some(7),
                keep: Some(2),
                ci: Some(CIOpts::default()),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(e.code(), "invalid_argument");
    }

    #[test]
    fn confirmatory_rejects_bad_keep() {
        let (x, y) = synth_with_signal(60, 5, 4.0, 17);
        let e = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 1,
                weights: None,
            },
            ConfirmatoryTestOpts {
                keep: Some(99),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(e.code(), "invalid_argument");
    }

    #[test]
    fn ci_branch_rejects_invalid_m_rate() {
        let (x, y) = synth_with_signal(80, 5, 4.0, 11);
        let err = pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: 2,
                weights: None,
            },
            ConfirmatoryTestOpts {
                args: ConfirmatoryArgs::SplitNb { n_splits: 30 },
                seed: Some(7),
                ci: Some(CIOpts {
                    n_boot: 200,
                    m_rate: 0.4,
                    level: 0.95,
                    max_failure_rate: 0.01,
                }),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
    }
}
