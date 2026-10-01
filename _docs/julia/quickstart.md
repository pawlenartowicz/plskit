# Quickstart

Three steps on synthetic data: fit a PLS1 model, test it, then run PLS3 on
a two-block design. Every snippet runs as written, in order, in one Julia
session. See [installation](installation.md) to get `PLSKit`.

## Data

A single outcome `y` driven by the first three of 20 predictors, and a
four-column block `Y` that shares the same signal.

```julia
using PLSKit
using Random

rng = Xoshiro(0)
n, p = 200, 20
X = randn(rng, n, p)
signal = X[:, 1:3] * [1.0, 0.5, -0.5]
y = signal .+ randn(rng, n)
Y = reduce(hcat, [signal .+ randn(rng, n) for _ in 1:4])
```

## 1. Fit PLS1

Required arguments are positional, everything with a default is a keyword.

```julia
model = pls1_fit(X, y; k=2)

size(model.W)       # (20, 2): X weights, one column per component
size(model.T)       # (200, 2): X scores
model.beta[1:3]     # raw-scale regression coefficients
model.k_used        # 2

ŷ = pls1_predict(model, X)   # 200-element Vector{Float64}
```

`k` can also be `"optimal"` or `"sequence"` (with `k_max`) to choose the
component count from the data; see `pls1_find_k_optimal` and
`pls1_find_k_sequence` in the [API reference](../python/api.md).

## 2. Test for signal

`test_method` has no default and must be passed. `args` takes a `NamedTuple`
(or a `Dict`). The counts are kept small so the example runs in seconds.

```julia
test = pls1_confirmatory_test(X, y; k=1, test_method="split_exact",
                              args=(n_perm=199, n_splits=20), seed=1)

test.pvalue        # permutation p-value
test.statistic     # tanh of the mean Fisher-z held-out correlation
test.test_method   # "split_exact"
test.seed          # 0x0000000000000001
```

Pass `seed=test.seed` to reproduce a run whose seed was drawn for you.

## 3. Two blocks: PLS3

```julia
pls3 = pls3_fit(X, Y; k=2)
pls3.singular_values
scores = pls3_transform(pls3; X_new=X, Y_new=Y)
size(scores.x_scores)    # (200, 2)
```

## Rotation and errors

```julia
rot = rotate(model; method=:varimax)
rot.rotation_spec.args   # (max_iter = 50, tol = 1.0e-8, kaiser_normalize = true)

try
    pls1_fit(X, y; weights=fill(-1.0, n))
catch e
    e isa PlsKitError || rethrow()
    e.code, e.details    # (:invalid_weights, (reason = :negative,))
end
```
