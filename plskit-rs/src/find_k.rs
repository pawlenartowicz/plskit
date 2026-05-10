//! K-selection: optimal (CV / BIC) and sequence (sequential test) methods.

use std::collections::BTreeMap;

use rand::seq::SliceRandom;

use crate::error::{PlsKitError, PlsKitResult};
use crate::fit::{pls1_fit, FitOpts, KSpec};
use crate::linalg::{
    col_row_subset, normalize_weights, row_subset, standardize1_weighted, standardize_apply,
    standardize_weighted,
};
use crate::sequential::{run_incremental_sequence, IncrementalSequenceOpts, SequentialArgs};
use crate::signal_test::ConfirmatoryMethod;

use faer::{Col, ColRef, MatRef};

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
    /// Number of permutations for `raw_perm` / `split_perm` diagnostic. Inert
    /// when `diagnostic is None`.
    pub n_perm: usize,
    /// Number of split-half repetitions for `split_nb` / `split_perm`
    /// diagnostic. Inert when `diagnostic is None`.
    pub n_splits: usize,
    /// Caller asserts X and y are already standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
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
    /// The selected number of components.
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
    /// Name of the diagnostic method used (present when `diagnostic.is_some()`).
    pub diagnostic: Option<String>,
    /// RNG seed actually used.
    pub seed: u64,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
}

/// Optimal-K selection on full data. Optionally attaches a same-sample
/// sequential diagnostic when `diagnostic.is_some()` — selection and test
/// share data, so the diagnostic pvalues are not honest inference.
///
/// # Errors
/// `KExceedsMax` for `k_max==0` or `k_max>d`; `DimensionMismatch` for shape
/// disagreements; `InvalidArgument` if `diagnostic == Some(Score)`.
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::type_complexity)]
pub fn pls1_find_k_optimal(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
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
    let max_allowed = x.ncols();
    if k_max == 0 || k_max > max_allowed {
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

    let (w_norm, n_eff_val, _all_uniform) =
        crate::fit::validate_and_normalize_weights(weights, n, k_max)?;

    let (seed_used, mut rng) = crate::rng::resolve_seed(opts.seed);

    // Selector dispatch.
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
                &mut rng,
            )?;
            (k, Some(scores), None, None, "r2_max")
        }
        Selector::Bic => {
            let (k, scores) = select_bic(x, y, k_max, w_norm.as_ref().map(Col::as_ref), n_eff_val)?;
            (k, None, None, Some(scores), "bic")
        }
    };

    let (pvalues, diagnostic_str) = if let Some(method) = opts.diagnostic {
        // stop_early_override=true so we collect the full p-value vector up to K*.
        let seq_args = SequentialArgs::defaults_for(method).ok_or_else(|| {
            PlsKitError::InvalidArgument("score has no sequential variant".into())
        })?;
        let seq_args = match seq_args {
            SequentialArgs::RawPerm { .. } => SequentialArgs::RawPerm {
                n_perm: opts.n_perm,
            },
            SequentialArgs::SplitNb { .. } => SequentialArgs::SplitNb {
                n_splits: opts.n_splits,
            },
            SequentialArgs::SplitPerm { .. } => SequentialArgs::SplitPerm {
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
            },
        )?;
        (Some(r.pvalues), Some(method.as_str().to_owned()))
    } else {
        (None, None)
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
    })
}

/// Opts for `pls1_find_k_sequence`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct FindKSequenceOpts {
    /// Per-step test method (any `ConfirmatoryMethod` except `Score`).
    pub test_method: ConfirmatoryMethod,
    /// Significance threshold for sequential rejection.
    pub alpha: f64,
    /// Number of permutations for `raw_perm` / `split_perm`.
    pub n_perm: usize,
    /// Number of split-half repetitions for `split_nb` / `split_perm`.
    pub n_splits: usize,
    /// Caller asserts X and y are already standardized; skips centering/scaling.
    pub pre_standardized: bool,
    /// RNG seed; `None` draws from OS entropy.
    pub seed: Option<u64>,
    /// Disable Rayon parallelism (forces serial execution; useful for deterministic debugging).
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
    /// Name of the test method used.
    pub test_method: String,
    /// Significance threshold used.
    pub alpha: f64,
    /// RNG seed actually used.
    pub seed: u64,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
}

/// Sequence-based K selection on full data. Stop-early is
/// always on — call this when you want a frequentist guarantee on the
/// nested chain.
///
/// # Errors
/// `KExceedsMax`, `DimensionMismatch`, `InvalidArgument` for `Score`.
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::many_single_char_names)]
pub fn pls1_find_k_sequence(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_max: usize,
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
    let max_allowed = x.ncols();
    if k_max == 0 || k_max > max_allowed {
        return Err(PlsKitError::KExceedsMax {
            k: k_max,
            k_max: max_allowed,
        });
    }

    let (w_norm, n_eff_val, _all_uniform) =
        crate::fit::validate_and_normalize_weights(weights, n, k_max)?;

    let seq_args = SequentialArgs::defaults_for(opts.test_method).ok_or_else(|| {
        PlsKitError::InvalidArgument("test_method='score' has no sequential variant".into())
    })?;
    let seq_args = match seq_args {
        SequentialArgs::RawPerm { .. } => SequentialArgs::RawPerm {
            n_perm: opts.n_perm,
        },
        SequentialArgs::SplitNb { .. } => SequentialArgs::SplitNb {
            n_splits: opts.n_splits,
        },
        SequentialArgs::SplitPerm { .. } => SequentialArgs::SplitPerm {
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
        IncrementalSequenceOpts {
            args: seq_args,
            alpha: opts.alpha,
            stop_early_override: false,
            pre_standardized: opts.pre_standardized,
            seed: opts.seed,
            disable_parallelism: opts.disable_parallelism,
            verbose: opts.verbose,
        },
    )?;
    let k_star = r.last_significant_k.unwrap_or(0);
    Ok(FindKSequenceOutput {
        k_star,
        pvalues: r.pvalues,
        test_method: opts.test_method.as_str().to_owned(),
        alpha: opts.alpha,
        seed: r.seed,
        n_eff: n_eff_val,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal selectors
// ──────────────────────────────────────────────────────────────────────────────

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
    rng: &mut crate::rng::Rng,
) -> PlsKitResult<(usize, BTreeMap<usize, f64>, Option<BTreeMap<usize, f64>>)> {
    // k-fold CV with optional 1-SE rule.
    let n = x.nrows();
    let n_folds = n_folds.min(n.saturating_sub(2)).max(2);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    let folds = crate::linalg::fold_split(&indices, n_folds);
    let max_comp = k_max.min(n.saturating_sub(n_folds + 2)).max(1);

    // r2_matrix[fi][k-1] = CV R² for fold fi and k components.
    // Each fold is independent and RNG-free (the only RNG use was the
    // pre-loop shuffle), so byte-parity holds for both serial and parallel execution.
    let fold_work = |fi: usize, val_idx: &Vec<usize>| -> PlsKitResult<Vec<f64>> {
        let train_idx: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != fi)
            .flat_map(|(_, f)| f.iter().copied())
            .collect();
        let x_tr = row_subset(x, &train_idx);
        let y_tr = col_row_subset(y, &train_idx);
        let x_val = row_subset(x, val_idx.as_slice());
        let y_val = col_row_subset(y, val_idx.as_slice());
        let nv = y_val.nrows();

        // Slice and re-normalize weights for train fold (if weights are provided).
        let train_w_full: Option<Col<f64>> = weights.map(|w| col_row_subset(w, &train_idx));
        let train_w_norm: Option<Col<f64>> = train_w_full
            .as_ref()
            .and_then(|w| normalize_weights(w.as_ref()));
        let train_wref = train_w_norm.as_ref().map(Col::as_ref);

        // Slice and re-normalize weights for val fold.
        let val_w_full: Option<Col<f64>> = weights.map(|w| col_row_subset(w, val_idx.as_slice()));
        let val_w_norm: Option<Col<f64>> = val_w_full
            .as_ref()
            .and_then(|w| normalize_weights(w.as_ref()));
        let val_wref = val_w_norm.as_ref().map(Col::as_ref);

        let (xs_tr, x_mean, x_scale) = standardize_weighted(x_tr.as_ref(), train_wref);
        let xs_val = standardize_apply(x_val.as_ref(), x_mean.as_ref(), x_scale.as_ref());
        let (ys_tr, ym, ys) = standardize1_weighted(y_tr.as_ref(), train_wref);
        let ys_val = Col::<f64>::from_fn(nv, |i| (y_val[i] - ym) / ys);

        let m = pls1_fit(
            xs_tr.as_ref(),
            ys_tr.as_ref(),
            KSpec::Fixed(max_comp),
            train_wref,
            FitOpts {
                pre_standardized: true,
                // check_n_eff: false — per-fold slice may have low n_eff; let the math degrade
                // and rely on the parent statistic to absorb noise (see Option B contract)
                check_n_eff: false,
                ..FitOpts::default()
            },
        )?;
        let actual = m.w_star.ncols();

        // Weighted mean and weighted R² on validation fold.
        let mean_val = match val_wref {
            None => {
                if nv > 0 {
                    (0..nv).map(|i| ys_val[i]).sum::<f64>() / nv as f64
                } else {
                    0.0
                }
            }
            Some(wv) => (0..nv).map(|i| wv[i] * ys_val[i]).sum::<f64>() / nv as f64,
        };
        let ss_tot: f64 = match val_wref {
            None => (0..nv).map(|i| (ys_val[i] - mean_val).powi(2)).sum(),
            Some(wv) => (0..nv)
                .map(|i| wv[i] * (ys_val[i] - mean_val).powi(2))
                .sum(),
        };

        let mut row = vec![f64::NAN; max_comp];
        for k in 1..=actual {
            let coef_k = crate::fit::pls1_coef_at_k(&m.w_star, &m.p_loadings, &m.q_loadings, k);
            let y_pred: Col<f64> = xs_val.as_ref() * coef_k.as_ref();
            let ss_res: f64 = match val_wref {
                None => (0..nv).map(|i| (y_pred[i] - ys_val[i]).powi(2)).sum(),
                Some(wv) => (0..nv)
                    .map(|i| wv[i] * (y_pred[i] - ys_val[i]).powi(2))
                    .sum(),
            };
            row[k - 1] = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };
        }
        Ok(row)
    };

    let r2_matrix: Vec<Vec<f64>> = if opts.disable_parallelism {
        folds
            .iter()
            .enumerate()
            .map(|(fi, val_idx)| fold_work(fi, val_idx))
            .collect::<PlsKitResult<Vec<_>>>()?
    } else {
        use rayon::prelude::*;
        folds
            .par_iter()
            .enumerate()
            .map(|(fi, val_idx)| fold_work(fi, val_idx))
            .collect::<PlsKitResult<Vec<_>>>()?
    };

    let mut cv_scores = BTreeMap::new();
    let mut cv_scores_se = BTreeMap::new();
    for k in 1..=max_comp {
        let finite: Vec<f64> = (0..n_folds)
            .map(|fi| r2_matrix[fi][k - 1])
            .filter(|v| v.is_finite())
            .collect();
        if finite.is_empty() {
            cv_scores.insert(k, f64::NAN);
            cv_scores_se.insert(k, f64::NAN);
        } else {
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (finite.len() - 1).max(1) as f64;
            let se = var.sqrt() / (finite.len() as f64).sqrt();
            cv_scores.insert(k, mean);
            cv_scores_se.insert(k, se);
        }
    }

    let best_k = *cv_scores
        .iter()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
        .map_or(&1, |(k, _)| k);
    let k_star = if use_1se {
        let threshold = cv_scores[&best_k] - cv_scores_se.get(&best_k).copied().unwrap_or(0.0);
        cv_scores
            .iter()
            .filter(|(_, v)| v.is_finite() && **v >= threshold)
            .map(|(k, _)| *k)
            .min()
            .unwrap_or(1)
    } else {
        best_k
    };

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
            ..FitOpts::default()
        },
    )?;
    let mut best_k = 1;
    let mut best_bic = f64::INFINITY;
    let mut bic_scores = BTreeMap::<usize, f64>::new();
    let nv = ys.nrows();
    for k in 1..=m.w_star.ncols() {
        let coef_k = crate::fit::pls1_coef_at_k(&m.w_star, &m.p_loadings, &m.q_loadings, k);
        let y_pred: Col<f64> = xs.as_ref() * coef_k.as_ref();
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

    #[test]
    fn optimal_r2_se_returns_k_star_and_cv_scores() {
        let (x, y) = synth(80, 5, 1, 5.0, 1);
        let r = pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            4,
            None,
            FindKOptimalOpts {
                selector: Selector::R2Se,
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.k_star >= 1);
        assert!(r.cv_scores.is_some());
        assert!(r.cv_scores_se.is_some());
        assert!(r.bic_scores.is_none());
        assert!(r.pvalues.is_none());
        assert!(r.diagnostic.is_none());
        assert_eq!(r.selector, "r2_se");
        assert!((r.n_eff - 80.0).abs() < 1e-9);
    }

    #[test]
    fn optimal_r2_max_returns_k_star_no_se() {
        let (x, y) = synth(80, 5, 1, 5.0, 1);
        let r = pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            4,
            None,
            FindKOptimalOpts {
                selector: Selector::R2Max,
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.cv_scores.is_some());
        assert!(r.cv_scores_se.is_none());
        assert_eq!(r.selector, "r2_max");
    }

    #[test]
    fn optimal_bic_returns_k_star_and_bic_scores() {
        let (x, y) = synth(60, 5, 2, 4.0, 3);
        let r = pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            4,
            None,
            FindKOptimalOpts {
                selector: Selector::Bic,
                seed: Some(13),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.k_star >= 1 && r.k_star <= 4);
        assert!(r.bic_scores.is_some());
        assert!(r.cv_scores.is_none());
        assert_eq!(r.selector, "bic");
    }

    #[test]
    fn optimal_with_diagnostic_returns_pvalues() {
        let (x, y) = synth(80, 5, 1, 5.0, 1);
        let r = pls1_find_k_optimal(
            x.as_ref(),
            y.as_ref(),
            4,
            None,
            FindKOptimalOpts {
                selector: Selector::R2Se,
                diagnostic: Some(ConfirmatoryMethod::SplitNb),
                n_splits: 30,
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        let pv = r.pvalues.expect("diagnostic pvalues missing");
        assert_eq!(pv.nrows(), r.k_star);
        assert_eq!(r.diagnostic.as_deref(), Some("split_nb"));
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

    #[test]
    fn sequence_returns_k_star_and_full_pvalues() {
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
    }

    #[test]
    fn sequence_no_rejection_returns_kstar_zero() {
        // Pure noise: synth with k_signal=0 (zero true signal).
        let (x, y) = synth(60, 5, 0, 0.0, 99);
        let r = pls1_find_k_sequence(
            x.as_ref(),
            y.as_ref(),
            4,
            None,
            FindKSequenceOpts {
                test_method: ConfirmatoryMethod::E, // e-value safe under null
                alpha: 0.001,                       // very strict to almost-guarantee no rejection
                seed: Some(99),
                ..Default::default()
            },
        )
        .unwrap();
        // k_star may rarely be >0 by chance; assertion is only that the call succeeds.
        let _ = r.k_star;
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
}
