# PLSKit.jl

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](../LICENSE)

Julia package for [plskit](https://github.com/pawlenartowicz/plskit):
Partial Least Squares with modern inference, running the Python package.

## What is plskit?

plskit is a Partial Least Squares (PLS) library built around four ideas:

- **Modern inference.** Confirmatory tests use split-half held-out
  prediction: `split_exact` is calibrated by permutation, and
  `split_nb` is its faster Fisher-z approximation. Confidence
  intervals are taken on rotation-invariant quantities (held-out
  correlation, per-variable leverage, a subsampling z for β), so no
  Procrustes alignment is needed. The tests are introduced in our
  [NeurIPS 2026 paper](https://openreview.net/forum?id=xb6CB7d9LO).
- **Same results in every language.** All numerical work runs in one
  Rust engine. The Python and R wrappers call into it, and the Julia
  wrapper runs the Python package, so a fixed `(version, seed, X, y)`
  gives the same results, within a bit-near tolerance, in every
  language and on every supported platform.
- **Speed.** The engine is written in Rust and fits PLS1 with
  [Improved Kernel PLS](../docs/concepts/PLS1/fit-and-predict.md#how-plskit-computes-it)
  (Dayal and MacGregor 1997), which makes plskit probably the fastest
  PLS package on a single core.
- **Parallel and RAM-safe.** Inference runs efficiently in parallel
  without blowing up RAM.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/pawlenartowicz/plskit", subdir="plskit-jl")
```

The first `using PLSKit` sets up a private Python environment (a one-time
download). To use your own Python instead, see
[installation](../docs/julia/installation.md).

## Quickstart

```julia
using PLSKit
using Random

rng   = Xoshiro(0)
X     = randn(rng, 200, 20)
y     = vec(sum(X[:, 1:3]; dims=2)) .+ randn(rng, 200)
X_new = randn(rng, 5, 20)

fit  = pls1_fit(X, y; k=3)
test = pls1_confirmatory_test(X, y; k=1, test_method="split_exact", seed=42)
ŷ    = pls1_predict(fit, X_new)
```

## How it works

`PLSKit.jl` runs the Python `plskit` package through
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl), so every
function, argument name and result field is the Python one, and every
number is the Python wheel's.

## Status

Version 0.7.2, running Python `plskit` 0.7.2. Not yet in the General
registry.

## Documentation

- [Julia pages](../docs/julia/index.md):
  installation, quickstart, differences from Python.
- [Python API](../docs/python/api.md)
  and [result objects](../docs/python/results.md):
  the canonical reference for every language.

## License

GPL-3.0-or-later.

## Citation

```bibtex
@inproceedings{lenartowicz2026cheap,
  title     = {Cheap and Powerful Tests for Supervised Subspaces: Per-Component Inference for {PLS}},
  author    = {Lenartowicz, Pawe{\l} and Plisiecki, Hubert},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=xb6CB7d9LO}
}
```

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
