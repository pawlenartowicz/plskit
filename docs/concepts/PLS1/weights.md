# Observation weights

The PLS1 fitting, K-selection and inference functions take observation
weights through the `weights` argument.

## What they mean

- A length-`n` non-negative vector of precision or sampling weights. Each
  row of `(X, y)` is scaled by `√w`. With as many components as `X` has
  independent columns, the weighted fit is weighted least squares.
- Weights are normalized to mean 1 (`w / mean(w)`) on every call, so their
  overall scale does not matter. The standardization of `X` and `y` uses
  the weighted mean and variance
  ([Preprocessing](../preprocessing.md#the-canonical-recipe)).
- `weights=None` is uniform. Weights whose normalized entries are all
  within `1e-12` of 1 are treated exactly as `None`: the call is
  bit-identical, the result echoes `weights` as `None`, and `n_eff = n`.
- The effective sample size `n_eff = (Σw)² / Σw²` is reported on
  `PreprocessResult`, `PLS1Result` and the inference results.

## What is accepted

- Zero weights are allowed; at least one weight must be positive.
- A wrong length, a negative entry or an all-zero vector raises
  `PlsKitInvalidWeights` (code `invalid_weights`), with `reason` set to
  `"length_mismatch"`, `"negative"` or `"all_zero"`.
- A NaN or an infinity raises `PlsKitError` with code `non_finite_input`.
- Weights that leave `n_eff < k + 1` raise `PlsKitInvalidWeights` with
  `reason="insufficient_effective_n"`
  ([Effective sample size](../effective-sample-size.md)).

## When to use them

- Heteroscedastic noise: down-weight observations with higher noise
  variance.
- Survey or sampling weights: re-weight to a target population.

## Which functions take them

- Take `weights`: `pls1_fit`, `pls1_confirmatory_test`,
  `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_rotation_stability`,
  `pls1_perm_null`, `split_nb_gate`, `preprocess`, and the sparse
  `spls1_*` family. They thread the weights through fit, K-selection and
  inference consistently.
- Do not take `weights`: `pls1_predict` (the weights have already shaped
  `beta` and `intercept` at fit time), `rotate`, and `pls1_perm_alpha` (it
  takes the Gram matrix of the caller's block in place of `X`, and the fit
  is unweighted).
- `pls3_fit` / `plssvd_fit` and `spls3_fit` accept the argument but refuse
  anything other than `None`: weights are not implemented for PLS3.

## Caching the standardization

`plskit.preprocess(X, y, weights=w)` standardizes `X` and `y` once, with
the weighted moments. Later calls take its arrays with
`pre_standardized=True` and the same weights:

```python
pre = plskit.preprocess(X, y, weights=w)
model = plskit.pls1_fit(
    pre.X_std, pre.Y_std, k=2, pre_standardized=True, weights=w,
)
```

`model.coef` equals the `coef` of `plskit.pls1_fit(X, y, k=2, weights=w)`
up to rounding. `pre_standardized` skips only the centering and scaling:
`model.beta` equals `model.coef` and `model.intercept` is `0`
([Preprocessing](../preprocessing.md#common-gotchas)).

## Cross-references

- [Effective sample size](../effective-sample-size.md): where the `n_eff` check runs and the errors it raises
- Function signatures: [Python API](../../python/api.md)
