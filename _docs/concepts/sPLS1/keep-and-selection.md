# Keep and selection

How `plskit` chooses `keep` and `k` for [sparse PLS1](index.md), and what
you can and cannot infer once a selection has been made.

## Tuning `keep`: the 1-SE rule on a keep grid

`spls1_find_keep_optimal(X, y, k)` tunes `keep` at a fixed `k`.

1. **Grid.** A logged geometric grid over `[1, n_features]`: powers of two
   below `n_features`, plus `n_features` itself. For 100 features that is
   `1, 2, 4, 8, 16, 32, 64, 100`. The grid actually swept is returned on
   `result.keep_grid`.
2. **Cross-validation.** For each fold (`args={"n_folds": ...}`, default 5)
   the training part is standardized on its own rows, the validation part
   with the training moments, and a sparse fit is run at every grid point.
   Each fold yields one CV R² per `keep`. Sparsity is tuned inside the
   training part of each fold, never on validation rows.
3. **Selection.** Per `keep`, the mean CV R² across folds and its standard
   error (fold SD divided by √folds) are recorded on `cv_scores` /
   `cv_scores_se`. `keep_star` is the **smallest** `keep` whose mean is at
   least `best mean − SE(best)`.

This is the same 1-SE rule `pls1_find_k_optimal` applies on the `k` axis
(`selector="r2_se"`), with "fewer variables" in place of "fewer
components" as the parsimony order. The dense endpoint is always on the
grid, so if no sparser `keep` comes within one SE of the best,
`keep_star` is `n_features` and the selected model is dense PLS1.

The grid is coarse on purpose. Because the rule picks the sparsest
adequate point rather than the exact maximum, finer spacing buys little.

## Tuning `k` at fixed `keep`

`spls1_find_k_optimal` and `spls1_find_k_sequence` are
`pls1_find_k_optimal` and `pls1_find_k_sequence` (see
[Find K](../PLS1/find-k.md)) with every inner fit replaced by the sparse
fit at your `keep`. Same selectors, same options, same result types. In
the sequential version each step deflates on the sparse residual and tests
the sparse component, so the sequence tests the model you will actually
fit.

### `selector="bic"` is biased toward more components

The BIC selector reuses the dense complexity penalty, `k · ln(n_eff)`. It
does not account for `keep`. Under sparsity this under-penalizes each
added component, so BIC tends to select a larger `k` than a keep-aware
penalty would. BIC values are also not comparable across different `keep`
values. This is a deliberate first-version simplification. Prefer the
default `selector="r2_se"` for sparse fits.

## Why there are no per-coordinate β CIs

For dense PLS1, `pls1_confirmatory_test(ci=True)` reports subsample CIs on
each coordinate of β (`beta_ci_lower` / `beta_ci_upper`; see
[Confidence intervals](../PLS1/ci.md)). sPLS1 has no equivalent, by
design.

The zeros in a sparse `W` are **selection events**: the data decided which
variables survive. A variable can be selected in some subsamples and
dropped in others, so the subsample distribution of its β coordinate
mixes two different models, and an ordinary resampling interval cannot be
trusted to reach its nominal coverage. Valid intervals need
post-selection inference, which `plskit` does not implement.

Two practical consequences:

- `pls1_confirmatory_test` has no `keep` argument. Its tests and its
  `ci=True` intervals are for the **dense** model, not for your sparse fit.
- The honest-split rule from [Find K](../PLS1/find-k.md) applies to `keep`
  as well as `k`: a `keep` chosen on the same data as a subsequent test
  makes that test exploratory, not confirmatory.

## Cross-references

- [Sparse PLS1](index.md): what `keep` is, the one-axis-fixed rule
- [Find K](../PLS1/find-k.md): the dense selectors these functions mirror
- [Python API §2b](../../python/api.md): `spls1_find_keep_optimal`, `spls1_find_k_optimal`, `spls1_find_k_sequence`
- [Results](../../python/results.md): `FindKeepOptimalResult`, `FindKOptimalResult`, `FindKSequenceResult`
