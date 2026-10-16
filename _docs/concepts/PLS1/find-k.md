# Find K

Choosing the number of PLS1 components. `plskit` has two K-selection
paths, each for a different question. Both take `X`, `y` and `k_max`, the
largest `k` to consider, and both report their choice as `k_star`.

## `pls1_find_k_optimal`

- Picks the `k` that optimizes a criterion.
- Selectors: `r2_se` (one-SE rule on CV R², the default), `r2_max` (CV R²
  maximum), `bic`. `bic` counts each component as one degree of freedom;
  on wide `X` (p much larger than n) a PLS fit's effective degrees of
  freedom are much larger, so BIC tends to select `k_max`, and `r2_se` is
  recommended there.
- An optional same-sample diagnostic (`diagnostic="split_nb"` etc.) runs a
  sequential test up to the selected `k` and attaches `pvalues` to the
  result. It uses the same data, so it is a robustness check, not
  inference.
- Use it for model selection when the goal is a model to predict with.

## `pls1_find_k_sequence`

- Tests `k = 1, 2, …` in order and stops at the first non-rejection.
- The inner test is `test_method` (`raw_perm`, `split_nb`, `split_exact`,
  `e`, `auto`; the default is [`auto`](inference.md#test_methodauto)).
- Each step is tested at level `alpha` (default `0.05`). `k_star` is the
  number of components that rejected before the first failure.
- Closed testing gives strong control of the family-wise error rate.
- Use it when the question is how many components are statistically
  supported.

## From `pls1_fit`

In Python, R and Julia, `pls1_fit(X, y, k="optimal", k_max=...)` and
`k="sequence"` run the matching function and fit at its `k_star`. The
selection result is kept on the model as `selection_result`. In Rust, call
the selection function and pass its `k_star` to `pls1_fit`
([Rust API](../../rust/api.md#side-by-side-example)).

## When no component exists

If `y` is constant, or orthogonal to the columns of `X` up to rounding
(for example an outcome residualized on covariates that span `X`), the
full-data fit cannot extract a first component: `pls1_fit` returns the
`k_used = 0` zero model, which predicts the mean of `y`. Both paths then
report `k_star = 0`. `pls1_find_k_optimal` returns empty score maps;
`pls1_find_k_sequence` does not reject at the first step. The sparse
`spls1_find_k_*` functions behave the same way.

`pls1_find_k_sequence` also reports `k_star = 0` when a first component
exists but its test does not reject.

The string modes of `pls1_fit` do not fit a zero model: with
`k_star = 0`, `k="optimal"` raises `optimal_no_component` and
`k="sequence"` raises `sequence_no_rejection`. Pass an explicit integer
`k` to fit anyway.

## The honest split

K-selection and confirmatory testing must not share data. A `k` chosen by
either path and then tested with `pls1_confirmatory_test` on the same data
gives an exploratory result, not a confirmatory one.
`pls1_confirmatory_test` cannot check this for you. For a confirmatory
claim, fix `k` in advance or test on a fresh sample.

The same holds for the `keep` of a sparse fit
([sparse PLS1](sparse.md)).

## Cross-references

- Confirmatory testing at a fixed `k`: [inference](inference.md)
- Sparse fits: the `spls1_find_k_*` functions run the same two paths at a fixed `keep` ([sparse PLS1](sparse.md#tuning-k-at-fixed-keep))
- Result fields: [`FindKOptimalResult`, `FindKSequenceResult`](../../python/results.md)
- Function signatures: [Python API](../../python/api.md), [Rust API](../../rust/api.md)
