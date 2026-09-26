# Find K

> Status: stub. Content TBD.

Choosing the number of PLS1 components. `plskit` exposes two
distinct K-selection paths, each appropriate to a different question.

## `pls1_find_k_optimal`

- One-shot K-selection: pick the `k` that optimizes a criterion
- Selectors: `r2_se` (one-SE rule on CV R²), `r2_max` (raw CV R² maximum), `bic`
- Optional same-sample diagnostic (`diagnostic="split_nb"` etc.) — runs a sequential test up to the selected `k` and attaches `pvalues` to the result; same data → robustness check, not honest inference
- When to use: production model selection where the goal is to fit and ship

## `pls1_find_k_sequence`

- Sequential closed-test path: incrementally test `k = 1, 2, …` and stop at the first non-rejection
- Inner test driven by `test_method` (`raw_perm`, `split_nb`, `split_exact`, `e`)
- Strong control over the family-wise error rate by closed testing
- When to use: hypothesis-driven workflows where the question is "how many components are statistically supported"

## When no component exists

If `y` is constant, or orthogonal to the columns of `X` up to rounding
(for example an outcome residualized on covariates that span `X`), the
full-data fit cannot extract even a first component: `pls1_fit` returns
the `k_used = 0` zero model, which predicts the mean of `y`. Both paths
then report `k_star = 0`. `pls1_find_k_optimal` returns empty score maps
for every selector rather than recommending a component the fit cannot
produce; `pls1_find_k_sequence` does not reject at the first step. The
sparse `spls1_find_k_*` entry points behave the same way.

## Cross-references

- Confirmatory testing at a fixed `k` (no data reuse): see [inference](inference.md)
- The honest split: K-selection and confirmatory testing must not share data
- Function signatures: [Python API](../../python/api.md), [Rust API](../../rust/api.md)
