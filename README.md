# plskit

[![crates.io](https://img.shields.io/crates/v/plskit.svg)](https://crates.io/crates/plskit)
[![PyPI](https://img.shields.io/pypi/v/plskit.svg)](https://pypi.org/project/plskit/)
[![r-universe](https://pawlenartowicz.r-universe.dev/badges/plskit)](https://pawlenartowicz.r-universe.dev/plskit)
[![CI](https://github.com/pawlenartowicz/plskit/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/plskit/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Partial Least Squares with modern inference: one Rust engine, with
packages for Rust, Python, R and Julia.

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
  [Improved Kernel PLS](docs/concepts/PLS1/fit-and-predict.md#how-plskit-computes-it)
  (Dayal and MacGregor 1997), which makes plskit probably the fastest
  PLS package on a single core.
- **Parallel and RAM-safe.** Inference runs efficiently in parallel
  without blowing up RAM.

## Install

| Language | Install | Status |
|----------|---------|--------|
| Rust | `cargo add plskit` | available; see [`plskit-rs/`](plskit-rs/) |
| Python | `pip install plskit` | available; see [`plskit-py/`](plskit-py/) |
| R | `install.packages("plskit", repos = "https://pawlenartowicz.r-universe.dev")` | available; see [`plskit-r/`](plskit-r/) |
| Julia | `Pkg.add(url="https://github.com/pawlenartowicz/plskit", subdir="plskit-jl")` | implemented, not yet registered; see [`plskit-jl/`](plskit-jl/) |

Each package's README covers its install details, a quickstart, and
what is specific to that language.

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

`test_method` defaults to `"auto"`, which picks `split_exact` or `split_nb`
from `X`. `k=1` with `split_exact` is the recommended choice: a split-half test calibrated by
permutation, so it holds its level on any design (its p-value is
floored at `1/(n_perm + 1)`). `split_nb` is the faster asymptotic
alternative. See the [Python API](docs/python/api.md) (§3.1) for all
five methods.

## Documentation

The full documentation is at <https://pawlenartowicz.pl/plskit/>. Its
source is the [`docs/`](docs/) folder.

## Repository layout

```
plskit/
├── plskit-rs/            Rust crate — canonical implementation
├── plskit-bind/          Language-neutral layer shared by the Python and R wrappers
├── plskit-py/            Python wrapper
├── plskit-r/             R wrapper
├── plskit-jl/            Julia wrapper (runs the Python package)
├── testdata/             Shared reference corpus, regenerated from the Rust core
├── plskit-testdata-gen/  Generator for testdata/
├── docs/                 User documentation (source of the docs site)
├── validation/           Comparisons with other packages, for numbers and speed
├── examples/             Python example scripts
├── scripts/              CI and maintenance checks
└── vendor/               Patched third-party crates
```

## Versioning

This is the single repository for the engine and every wrapper. The
Rust engine and each language wrapper carry their own version number,
and **the same version number always means the same features** —
`plskit-rs 0.6.0` and `plskit (Python) 0.6.0` ship the same API. The
Python, R, or Julia version may lag behind the Rust engine while its
surface is being built out, but it can never run ahead of it. Releases
use `vX.Y.Z` for the engine and `vX.Y.Z-py` / `-r` / `-jl` for the
wrappers.

API wiring (function names, argument names, result fields) is stable
across versions. Numerical reproducibility requires pinning the
version: changes to defaults, algorithms, or implementation details
may shift outputs between releases.

## Contributing

File issues and pull requests here, against this repository; there are
no per-language mirror repositories. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).

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
