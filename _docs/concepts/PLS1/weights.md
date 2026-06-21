# Observation weights

> Status: stub. Content TBD.

`plskit` supports observation weights on every PLS1 entry point via the
`weights` argument. This page covers what they mean, when to use them,
and the numerical recipe.

## What they mean

- WLS-style precision / sampling weights — a length-`n` non-negative
  vector that re-scales each row of `(X, y)` by `√w`
- Default `weights=None` is uniform (equivalent to all-ones)
- Weights are normalized to sum to `n` (i.e., `w / mean(w)`) before use

## When to use them

- Heteroscedastic noise: down-weight observations with higher noise variance
- Survey / sampling weights: re-weight to a target population
- Robust workflows: combine with iteratively reweighted PLS (planned, `pls1_robust_fit`)

## How they propagate

- All PLS1 functions that take `weights` thread them through fit, predict, K-selection, and inference consistently
- Effective sample size: `n_eff = (Σw)² / Σw²`, surfaced on `PreprocessResult`
- Cache pattern: `plskit.preprocess(X, y, weights=...)` normalizes once; downstream calls pass `pre_standardized=True` to skip redundant work

## Cross-references

- Function signatures: [Python API](../../python/api.md)
- The `pre_standardized` flag covers X+Y centering / scaling AND weight normalization as a unit
