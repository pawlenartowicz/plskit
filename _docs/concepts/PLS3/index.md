# PLS3 / PLSSVD

PLS3 is the symmetric two-block family: `X (n × p)` and `Y (n × q)` enter
on equal footing, and the question is which pattern of X covaries with
which pattern of Y. Neuroimaging and psychology call this **PLSC**
(PLS correlation, in the McIntosh tradition); chemometrics calls it
**SVD-PLS** or PLSSVD.

## One SVD, no deflation

The whole fit is one thin SVD of the standardized cross-product `X̃ᵀỸ`, a
`p × q` matrix:

```
X̃ᵀỸ = U · diag(singular_values) · Vᵀ
```

All `k` components come out of that single decomposition, so they are
orthogonal by construction rather than by sequential deflation. The SVD
never forms a `p × p` matrix, which is why `p ≫ n` is the ordinary case
here: a vectorized connectome with 18,528 features against 5 behavioural
measures is an 18,528 × 5 SVD.

## PLSC is not PLS2

Both take a matrix `Y`, but they answer different questions.

| | PLS3 / PLSC | PLS2 |
|---|---|---|
| Roles | symmetric, neither block is the outcome | X predicts Y |
| Components | one SVD of `X̃ᵀỸ`, no deflation | extracted one at a time with deflation |
| Output | paired saliences `U`, `V` and their singular values | a regression `Ŷ = X B` |
| `predict` | none | yes |

PLS2 is not implemented in `plskit`. If you have a single outcome, you
want [PLS1](../PLS1/index.md); `pls3_fit` rejects a 1-D `Y` and says so.

## No `predict`, by design

Because the family is symmetric there is nothing to predict: no `coef`,
no `beta`, no `intercept`. What you can do with new data is project it
onto the fitted latent variables with `pls3_transform`, which returns
X-side scores, Y-side scores, or both. See
[Fit and transform](fit-and-transform.md).

The missing `predict` also shapes inference: the tests that need a
cross-validated prediction statistic cannot exist here. See
[Inference](inference.md).

## The `plssvd_*` aliases

`plssvd_fit` and `plssvd_transform` are exact aliases of `pls3_fit` and
`pls3_transform`: same arguments, same result types (`PLS3Result`,
`PLS3Scores`), same numbers. They exist because the two literatures name
the method differently. Use whichever name matches your field. The
confirmatory test has only one name, `pls3_confirmatory_test`.

## Pages

- [Fit and transform](fit-and-transform.md): the sign convention, truncation and its traps, why weights are refused, projecting new data
- [Inference](inference.md): `pls3_confirmatory_test`, the held-out LV correlation, why `k = 1` only, which methods exist and why
- [sPLS3](../sPLS3/index.md): the sparse variant, `keep_X` / `keep_Y`, and [what sparsity costs](../sPLS3/sparse-saliences.md)

## Cross-references

- [PLS1](../PLS1/index.md): the single-response family
- [Preprocessing](../preprocessing.md): the standardization recipe both blocks go through
- [Python API §2c, §3.4](../../python/api.md): signatures and options
- [Results](../../python/results.md): `PLS3Result`, `PLS3Scores`, `ConfirmatoryTestResult`
