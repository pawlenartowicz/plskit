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

## Wrappers

| Language | Package              | Status                                          |
|----------|----------------------|-------------------------------------------------|
| Rust     | `cargo add plskit`   | available — see [`plskit-rs/`](plskit-rs/)      |
| Python   | `pip install plskit` | available — see [`plskit-py/`](plskit-py/)      |
| R        | —                    | planned                                         |
| Julia    | —                    | planned                                         |

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
    X, y, k=1, method="split_exact", seed=42,
)
sig.pvalue, sig.statistic
```

`method` has no default and must be passed. `k=1` with `split_exact`
is the recommended choice: a split-half test calibrated by
permutation, so it holds its level on any design (its p-value is
floored at `1/(n_perm + 1)`). `split_nb` is the faster asymptotic
alternative. See the [Python API](_docs/python/api.md) (§3.1) for all
five methods.

## Why plskit?

- **Modern inference is canonical.** Confirmatory tests use
  split-half held-out prediction (`split_exact`, calibrated by
  permutation, or `split_nb`, its faster Fisher-z approximation).
  `ci=True` adds resampling readouts on rotation-invariant quantities
  (a CI on held-out correlation, per-variable bootstrap leverage CIs and
  a per-variable subsampling z for β), so no Procrustes alignment is
  needed.
- **One Rust engine, identical results across wrappers.** All
  numerical computation lives in `plskit-rs`. Python (and the
  forthcoming R and Julia wrappers) call into it via FFI, so a fixed
  `(version, seed, X, y)` reproduces within a bit-near tolerance on
  every supported platform.

## Repository layout

```
plskit/
├── plskit-rs/     Rust crate — canonical implementation
├── plskit-py/     Python wrapper
├── plskit-r/      R wrapper (planned)
├── plskit-jl/     Julia wrapper (planned)
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

Lenartowicz, P., Plisiecki, H. (2026). (Under Review).

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
