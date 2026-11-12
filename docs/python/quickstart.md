# Quickstart

Three steps on synthetic data: fit a PLS1 model, test it, then run PLS3 on
a two-block design. Every snippet below runs as written, in order, in one
Python session. See [installation](installation.md) to get `plskit`.

## Data

A single outcome `y` driven by the first three of 20 predictors, and a
four-column block `Y` that shares the same signal.

```python
import numpy as np
import plskit

rng = np.random.default_rng(0)
n, p = 200, 20
X = rng.standard_normal((n, p))
signal = X[:, :3] @ np.array([1.0, 0.5, -0.5])
y = signal + rng.standard_normal(n)
Y = np.column_stack([signal + rng.standard_normal(n) for _ in range(4)])
```

## 1. Fit PLS1

`pls1_fit` fits the PLS1 (NIPALS) model (one continuous outcome). `X` and `y` are
standardized internally. `coef` holds the coefficients on the
standardized scale; `beta` and `intercept` are back-projected to the raw
scale of `X` and `y`.

```python
model = plskit.pls1_fit(X, y, k=2)

model.W.shape        # (20, 2): X weights, one column per component
model.T.shape        # (200, 2): X scores
model.beta[:3]       # raw-scale regression coefficients
model.k_used         # 2

y_hat = plskit.pls1_predict(model, X)   # shape (200,)
```

Without `k`, `pls1_fit(X, y)` chooses the component count: the default `k="optimal"` runs `pls1_find_k_optimal` (cross-validation, `k_max` up to 9) and fits at its choice. Without a `seed` the folds are random, so the chosen `k` can differ between runs. `k="sequence"` chooses by a sequence of tests; see `pls1_find_k_optimal` and `pls1_find_k_sequence` in the [API reference](api.md).

## 2. Test for signal

`pls1_confirmatory_test` asks whether there is any predictive signal at some `k` in a set fixed before the test sees `y`. Without `k` it tests the default set `1..8` (cut by the design; here all eight) and returns one p-value for the whole set ([A set of `k`](../concepts/PLS1/inference.md#a-set-of-k)). `test_method` defaults to `"auto"`, which runs `split_exact` over a set of more than one `k` and, at one `k`, picks `split_exact` or
`split_nb` from `X` (see [`test_method="auto"`](../concepts/PLS1/inference.md#test_methodauto)); the example passes `"split_exact"` explicitly. The permutation and split
counts here are kept small so the example runs in seconds; the defaults
are `n_perm=1000`, `n_splits=32`.

```python
test = plskit.pls1_confirmatory_test(
    X, y,
    test_method="split_exact",
    args={"n_perm": 199, "n_splits": 20},
    seed=1,
)

test.pvalue           # one permutation p-value for the whole set
test.k                # [1, 2, 3, 4, 5, 6, 7, 8]: the tested set
test.pvalues_by_k     # each k's own p, not adjusted for the set
test.statistics_by_k  # each k's tanh of the mean Fisher-z held-out correlation
test.statistic        # None for a set; the statistic when k is one integer
test.test_method      # "split_exact"
```

A fixed `seed` makes the result reproducible, also across thread counts.
Pass `ci=True` to add rotation-invariant subsample CIs on `test.ci`
(see [results](results.md), `ConfirmatoryCI`).

The test names no `k` to fit at. Make the claim with the test and fit the model with `pls1_fit(X, y)`, which chooses its own `k`; pass no `k` from one to the other. Choosing `k` with `pls1_find_k_*` and then testing at that `k` on the same data is not a confirmatory test; test the default set, fix `k` in advance, or test on fresh data.

## 3. Two blocks: PLS3

When neither block is the outcome, `pls3_fit` (alias `plssvd_fit`) finds
paired X and Y directions that maximize covariance, from one SVD of the
standardized `X'Y`. `Y` must be 2-D.

```python
pls3 = plskit.pls3_fit(X, Y, k=2)

pls3.U.shape             # (20, 2): X saliences
pls3.V.shape             # (4, 2):  Y saliences
pls3.singular_values     # decreasing

scores = plskit.pls3_transform(pls3, X_new=X, Y_new=Y)
scores.x_scores.shape    # (200, 2)
```

PLS3 has no `predict`: it is symmetric, so there is nothing to predict.
`pls3_confirmatory_test` tests the first latent variable (`k=1` only) with
the same `test_method="split_exact"` / `"split_nb"` choice as PLS1.

## Next

- [API reference](api.md): every function, its options and error codes.
- [Result objects](results.md): every field on every result type.
