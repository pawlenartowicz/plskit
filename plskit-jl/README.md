# PLSKit.jl

Julia wrapper for [plskit](https://github.com/pawlenartowicz/plskit),
Partial Least Squares with modern inference. `PLSKit.jl` runs the Python
`plskit` package through PythonCall.jl, so every function, argument name
and result field is the Python one, and every number is the Python
wheel's.

```julia
using PLSKit
fit  = pls1_fit(X, y; k=3)
test = pls1_confirmatory_test(X, y; k=1, test_method="split_exact", seed=42)
ŷ    = pls1_predict(fit, X_new)
```

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/pawlenartowicz/plskit", subdir="plskit-jl")
```

The first `using PLSKit` sets up a private Python environment (a one-time
download). To use your own Python instead, see
[installation](../_docs/julia/installation.md).

## Documentation

- [Julia docs](../_docs/julia/index.md): installation, quickstart, differences from Python
- [API reference](../_docs/python/api.md) and [result objects](../_docs/python/results.md):
  canonical for every language

## What is plskit?

plskit is a Partial Least Squares (PLS) library built around three ideas:

- **Modern inference.** Confirmatory tests use split-half held-out
  prediction: `split_exact` is calibrated by permutation, and
  `split_nb` is its faster Fisher-z approximation. Confidence
  intervals are taken on rotation-invariant quantities (held-out
  correlation, per-variable leverage, a subsampling z for β), so no
  Procrustes alignment is needed. The tests are introduced in our
  [NeurIPS 2026 paper](https://openreview.net/forum?id=xb6CB7d9LO).
- **Same results in every language.** All numerical work runs in one
  Rust engine; every wrapper calls into it, so a fixed
  `(version, seed, X, y)` gives the same results, within a bit-near
  tolerance, in every language and on every supported platform.
- **Speed.** The engine is written in Rust with parallel resampling.

## Status

Version 0.7.0, running Python `plskit` 0.7.0. Not yet in the General
registry.
