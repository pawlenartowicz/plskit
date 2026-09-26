# Sparse PLS1 (sPLS1)

Sparse PLS1 is [PLS1](../PLS1/index.md) with a hard limit on how many X
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
`pls1_predict`. Observation [weights](../PLS1/weights.md) and
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
the `spls1_find_*` functions directly. How each selector works, and what
inference is and is not available after selection, is on
[Keep and selection](keep-and-selection.md).

## Pages

- [Keep and selection](keep-and-selection.md): the 1-SE rule on the keep
  grid, why there are no per-coordinate β CIs, the BIC caveat

## Cross-references

- [PLS1](../PLS1/index.md): the dense model sPLS1 reduces to
- [Find K](../PLS1/find-k.md): the dense K-selection paths the `spls1_find_k_*` functions mirror
- [Python API §2b](../../python/api.md): signatures and options
- [Results](../../python/results.md): `PLS1Result.keep`, `FindKeepOptimalResult`
