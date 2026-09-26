# Observation weights

> Status: stub. Content TBD.

`plskit` supports observation weights on the PLS1 fitting, K-selection,
and inference entry points via the `weights` argument. This page covers
what they mean, when to use them, and the numerical recipe.

## What they mean

- WLS-style precision / sampling weights: a length-`n` non-negative
  vector that re-scales each row of `(X, y)` by `√w`
- Default `weights=None` is uniform (equivalent to all-ones)
- Weights are normalized to sum to `n` (i.e., `w / mean(w)`) before use

## When to use them

- Heteroscedastic noise: down-weight observations with higher noise variance
- Survey / sampling weights: re-weight to a target population
- Robust workflows: combine with iteratively reweighted PLS *(planned, `pls1_robust_fit`)*

## Which functions take them

- Take `weights`: `pls1_fit`, `pls1_confirmatory_test`,
  `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_rotation_stability`,
  `pls1_perm_null`, `split_nb_gate`, `preprocess`, and the sparse
  `spls1_*` family
- Do not take `weights`: `pls1_predict` (prediction is `intercept + X_new · beta`;
  the weights have already shaped `beta` and `intercept` at fit time)
  and `rotate`
- `pls3_fit` / `plssvd_fit` accept the argument but refuse anything other
  than `None`: weights are not implemented for PLS3

## How they propagate

- The PLS1 functions that take `weights` thread them through fit,
  K-selection, and inference consistently
- Effective sample size: `n_eff = (Σw)² / Σw²`, surfaced on
  `PreprocessResult`, `PLS1Result`, and the inference results
- Cache pattern: `plskit.preprocess(X, y, weights=...)` normalizes once;
  downstream calls pass `pre_standardized=True` to skip redundant work

## Cross-references

- Function signatures: [Python API](../../python/api.md)
- The `pre_standardized` flag covers X+Y centering / scaling AND weight normalization as a unit
