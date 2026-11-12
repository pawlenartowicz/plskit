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
- The CV folds of `r2_se` and `r2_max` read `X` divided once by its column scales over all rows (weighted when `weights` are given; used as given under `pre_standardized`) and only centre: each fold centres its training rows with their own means and its held-out rows with those same means. `y` is standardized in each fold. `spls1_find_keep_optimal` scales its folds the same way.
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
- `X` and `y` are standardized once over all rows (used as given under `pre_standardized`), and each test only centres its parts. Steps from 2 on test the deflated residual on the scale step 1 used.
- Use it when the question is how many components are statistically
  supported.

## From `pls1_fit`

In Python, R and Julia, `pls1_fit(X, y)` chooses `k`: `k` defaults to `"optimal"`, which runs `pls1_find_k_optimal` (selector `r2_se` unless `find_k_args` says otherwise) and fits at its `k_star`. `k="sequence"` does the same with `pls1_find_k_sequence`. The selection result is kept on the model as `selection_result`.

Under either mode `k_max` defaults to `min(9, p)`, lowered until the weights' Kish effective sample size and the covariates' residual degrees of freedom both stay at least `k_max + 1`, never below 1. The cross-validation cap `min(k_max, n − n_folds − 2)` then applies, as it does to an explicit `k_max`. An explicit `k_max` is not lowered by the effective sample size or the residual degrees of freedom (above `p` it raises `k_exceeds_max`). With an integer `k`, `k_max` must be left unset.

Without a `seed`, the cross-validation folds are drawn at random, so the chosen `k` can differ between runs. A `seed` makes it repeatable; `seed=fit.selection_result.seed` repeats a run whose seed was drawn for you.

A `y` with no first component makes the default call raise `optimal_no_component` (see [below](#when-no-component-exists)).

In Rust, `pls1_fit` takes an explicit `KSpec::Fixed(k)`: call the selection function and pass its `k_star` to `pls1_fit` ([Rust API](../../rust/api.md#side-by-side-example)).

## When no component exists

If `y` is constant, or orthogonal to the columns of `X` up to rounding
(for example an outcome residualized on covariates that span `X`), the
full-data fit cannot extract a first component: `pls1_fit` at an integer `k` returns the
`k_used = 0` zero model, which predicts the mean of `y`. Both paths then
report `k_star = 0`. `pls1_find_k_optimal` returns empty score maps;
`pls1_find_k_sequence` does not reject at the first step. The sparse
`spls1_find_k_*` functions behave the same way.

`pls1_find_k_sequence` also reports `k_star = 0` when a first component
exists but its test does not reject.

The string modes of `pls1_fit` do not fit a zero model: with `k_star = 0`, `k="optimal"` (the default) raises `optimal_no_component` and `k="sequence"` raises `sequence_no_rejection`. Pass an explicit integer `k` to fit anyway.

## The honest split

K-selection and confirmatory testing must not share data. A `k` chosen by
either path and then tested with `pls1_confirmatory_test` on the same data
gives an exploratory result, not a confirmatory one.
`pls1_confirmatory_test` cannot check this for you. For a confirmatory claim, test the default set of `k` or a `k` (or set) fixed in advance ([A set of `k`](inference.md#a-set-of-k)), or test on a fresh sample.

The same holds for the `keep` of a sparse fit
([sparse PLS1](sparse.md)).

## Cross-references

- Confirmatory testing at a `k` or over a set of `k`: [inference](inference.md)
- Sparse fits: the `spls1_find_k_*` functions run the same two paths at a fixed `keep` ([sparse PLS1](sparse.md#tuning-k-at-fixed-keep))
- Result fields: [`FindKOptimalResult`, `FindKSequenceResult`](../../python/results.md)
- Function signatures: [Python API](../../python/api.md), [Rust API](../../rust/api.md)
