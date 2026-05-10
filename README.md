# plskit

[![crates.io](https://img.shields.io/crates/v/plskit.svg)](https://crates.io/crates/plskit)
[![PyPI](https://img.shields.io/pypi/v/plskit.svg)](https://pypi.org/project/plskit/)
[![CI](https://github.com/pawlenartowicz/plskit/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/plskit/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Cross-language Partial Least Squares with **modern inference**:
canonical percentile CIs, `split_perm`, and `split_nb` tests.

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
import plskit

model = plskit.pls1_fit(X, y, k=3, seed=42)
sig   = plskit.pls1_confirmatory_test(
    X, y, k=3, method="split_nb", args={"n_splits": 50}, seed=42,
)
sig.pvalue, sig.statistic
```

## Why plskit?

- **Modern inference is canonical.** Procrustes-aligned percentile CIs,
  `split_perm`, and `split_nb` are the default. Legacy outputs (BSR,
  VIP, jackknife) are planned for a future release.
- **One Rust engine, identical results across wrappers.** All
  numerical computation lives in `plskit-rs`. Python (and the
  forthcoming R and Julia wrappers) call into it via FFI, so a fixed
  `(version, seed, X, y)` reproduces within a bit-near tolerance on
  every supported platform.
- **Honest K-selection.** Confirmatory tests at a pre-specified `K`
  are separate from exploratory K-selection. Closed-testing sequences
  (`pls1_find_k_sequence`) carry exact FWER control; the optional
  same-sample diagnostic on `pls1_find_k_optimal(diagnostic=...)` is
  reported as `pvalues` / `diagnostic` and is not honest inference
  (selection and test reuse the same data).

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
`plskit-rs 0.5.0` and `plskit (Python) 0.5.0` ship the same API. The
Python, R, or Julia version may lag behind the Rust engine while its
surface is being built out, but it can never run ahead of it. Releases
use `vX.Y.Z` for the engine and `vX.Y.Z-py` / `-r` / `-jl` for the
wrappers. File pull requests and issues here.

API wiring (function names, argument names, result fields) is stable
across versions. Numerical reproducibility requires pinning the
version: changes to defaults, algorithms, or implementation details
may shift outputs between releases.

## Citation

Lenartowicz, P., Plisiecki, H. (2026). *Cheap Per-Component Testing for
PLS, Stable Under Rotation* (Under Review).

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
