# Sparse PLS1 (sPLS1)

Sparse PLS1 is [PLS1](index.md) with a hard limit on how many X
variables each component may use. Everything else (scores, loadings,
deflation, `coef`, `beta`, `intercept`, prediction) is the dense PLS1
machinery unchanged. The difference is one step: after each weight vector
is computed, all but its `keep` largest-magnitude entries are set to zero.

## What `keep` is

`keep` is an integer in `[1, n_features]`: the number of X variables each
latent direction may load on.

- **One value for all components.** `keep` is a scalar broadcast to every
  component; there is no per-component budget.
- **Hard selection, not shrinkage.** The surviving weights keep their
  magnitudes (the vector is then rescaled to unit length as usual). This is
  not soft thresholding, so there is no penalty parameter.
- **Deterministic ties.** When two `|w|` entries tie exactly, the lower
  column index survives, so the selected set does not depend on sort order
  or thread count.
- **Selection is per component.** Component 2 selects on the deflated
  residual, so its variables need not include component 1's. Tune the
  path; don't read it as a nested ranking of variables.
- Out-of-range values (`keep = 0`, `keep > n_features`) raise
  `PlsKitError(code="invalid_argument")`.

The fitted model is an ordinary `PLS1Result` with its `keep` field set
(it is `None` on a dense fit). Each column of `W` is zero outside its
`keep` selected rows. There is no `spls1_predict`: pass the model to
`pls1_predict`. Observation [weights](weights.md) and
`pre_standardized` work as in dense PLS1.

## `keep = n_features` is dense PLS1, bit for bit

At the dense endpoint the selection step is skipped entirely rather than
run as a no-op, so `spls1_fit(X, y, k, keep=X.shape[1])` returns the same
floats as `pls1_fit(X, y, k)`, not merely close ones. The same holds for
`spls1_find_k_optimal` and `spls1_find_k_sequence` against their dense
counterparts at the same seed. The engine's test suite checks this with
exact bit comparison.

## One axis is always fixed

sPLS1 has two tuning axes, `k` (components) and `keep` (variables per
component). `plskit` never searches them jointly:

| Function | Tunes | Holds fixed |
|---|---|---|
| `spls1_fit` | nothing | `k` and `keep` (both required) |
| `spls1_find_keep_optimal` | `keep` | `k` |
| `spls1_find_k_optimal` | `k` | `keep` |
| `spls1_find_k_sequence` | `k` | `keep` |

`spls1_fit` has no `'optimal'` / `'sequence'` string modes for `k`; call
the `spls1_find_*` functions directly. The sections below describe how
each selector works, and what inference is and is not available after
selection.

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
[Find K](find-k.md)) with every inner fit replaced by the sparse
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
[Confidence intervals](ci.md)). sPLS1 has no equivalent, by
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
- The honest-split rule from [Find K](find-k.md) applies to `keep`
  as well as `k`: a `keep` chosen on the same data as a subsequent test
  makes that test exploratory, not confirmatory.

## Cross-references

- [PLS1](index.md): the dense model sPLS1 reduces to
- [Find K](find-k.md): the dense K-selection paths the `spls1_find_k_*` functions mirror
- [Python API §2b](../../python/api.md): signatures and options
- [Results](../../python/results.md): `PLS1Result.keep`, `FindKeepOptimalResult`, `FindKOptimalResult`, `FindKSequenceResult`
