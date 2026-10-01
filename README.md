# plskit

[![crates.io](https://img.shields.io/crates/v/plskit.svg)](https://crates.io/crates/plskit)
[![PyPI](https://img.shields.io/pypi/v/plskit.svg)](https://pypi.org/project/plskit/)
[![CI](https://github.com/pawlenartowicz/plskit/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/plskit/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

**plskit is a fast Partial Least Squares implementation** (a Rust
engine with Python bindings) that also implements faster and more
powerful statistical tests for PLS models, introduced in our
[NeurIPS 2026 paper](PAPER_URL): the `split_exact` and `split_nb`
confirmatory tests, plus rotation-invariant subsampling CIs.

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
  Rust engine. The Python and R wrappers call into it, and the Julia
  wrapper runs the Python package, so a fixed `(version, seed, X, y)`
  gives the same results, within a bit-near tolerance, in every
  language and on every supported platform.
- **Speed.** The engine is written in Rust with parallel resampling,
  which makes plskit probably the fastest PLS package available.

## Wrappers

| Language | Package              | Status                                          |
|----------|----------------------|-------------------------------------------------|
| Rust     | `cargo add plskit`   | available — see [`plskit-rs/`](plskit-rs/)      |
| Python   | `pip install plskit` | available — see [`plskit-py/`](plskit-py/)      |
| R        | —                    | implemented, not yet released; see [`plskit-r/`](plskit-r/) |
| Julia    | —                    | implemented, not yet registered; see [`plskit-jl/`](plskit-jl/) |

Installation, examples, and language-specific notes live in each
wrapper's README.

## A 30-second look

```python
import numpy as np
import plskit

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 20))
y = X[:, :3].sum(axis=1) + rng.standard_normal(200)

model = plskit.pls1_fit(X, y, k=1)
sig   = plskit.pls1_confirmatory_test(
    X, y, k=1, test_method="split_exact", seed=42,
)
sig.pvalue, sig.statistic
```

`test_method` has no default and must be passed. `k=1` with `split_exact`
is the recommended choice: a split-half test calibrated by
permutation, so it holds its level on any design (its p-value is
floored at `1/(n_perm + 1)`). `split_nb` is the faster asymptotic
alternative. See the [Python API](_docs/python/api.md) (§3.1) for all
five methods.

## Repository layout

```
plskit/
├── plskit-rs/     Rust crate — canonical implementation
├── plskit-py/     Python wrapper
├── plskit-r/      R wrapper
├── plskit-jl/     Julia wrapper (runs the Python package)
└── testdata/      Shared reference corpus, regenerated from the Rust core
```

## Monorepo and versioning

This is the single repository for the engine and every wrapper. The
Rust engine and each language wrapper carry their own version number,
and **the same version number always means the same features** —
`plskit-rs 0.6.0` and `plskit (Python) 0.6.0` ship the same API. The
Python, R, or Julia version may lag behind the Rust engine while its
surface is being built out, but it can never run ahead of it. Releases
use `vX.Y.Z` for the engine and `vX.Y.Z-py` / `-r` / `-jl` for the
wrappers. File pull requests and issues here.

API wiring (function names, argument names, result fields) is stable
across versions. Numerical reproducibility requires pinning the
version: changes to defaults, algorithms, or implementation details
may shift outputs between releases.

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

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
