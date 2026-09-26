# plskit (Python)

[![PyPI](https://img.shields.io/pypi/v/plskit.svg)](https://pypi.org/project/plskit/)
[![Python versions](https://img.shields.io/pypi/pyversions/plskit.svg)](https://pypi.org/project/plskit/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](../LICENSE)

Python wrapper for **plskit**: Partial Least Squares with modern
inference (`split_exact` and `split_nb` tests, rotation-invariant
subsampling CIs), backed
by a Rust engine.

> Part of the **[plskit project](https://github.com/pawlenartowicz/plskit)**:
> the Rust core, sibling wrappers (R and Julia, planned), shared test
> corpus, issues, and PRs all live there.

## Install

```
pip install plskit
```

Wheels are built for Linux, macOS, and Windows on recent Python
versions (`>=3.10`). When a wheel is unavailable for your platform,
pip falls back to building from the source distribution, which
requires a working Rust toolchain.

## A 60-second look

```python
import numpy as np
import plskit

X = np.random.default_rng(0).normal(size=(200, 20))
y = X[:, :3].sum(axis=1) + np.random.default_rng(1).normal(size=200)

# Fit at fixed K, or let plskit pick K via cross-validation.
model   = plskit.pls1_fit(X, y, k=3, seed=42)
optimal = plskit.pls1_fit(X, y, k="optimal", k_max=10, seed=42)

y_hat = plskit.pls1_predict(model, X_new=X[:5])

# Confirmatory test for any signal; k=1 with split_exact is recommended.
sig = plskit.pls1_confirmatory_test(
    X, y, k=1, method="split_exact", seed=42,
)
print(sig.pvalue, sig.statistic)
```

## Public surface

Three model families: PLS1 (single continuous `y`), sparse PLS1, and
PLS3 / PLSSVD (symmetric `X`/`Y` covariance analysis) with a sparse
variant. PLS2 and multi-block PLS are planned.

- **Preprocessing:** `preprocess`.
- **PLS1 fit / predict:** `pls1_fit`, `pls1_predict`; K-selection via
  `pls1_find_k_optimal`, `pls1_find_k_sequence`.
- **Sparse PLS1:** `spls1_fit`, `spls1_find_keep_optimal`,
  `spls1_find_k_optimal`, `spls1_find_k_sequence` (predict with
  `pls1_predict`).
- **PLS3 / PLSSVD:** `pls3_fit` (alias `plssvd_fit`), `pls3_transform`
  (alias `plssvd_transform`), sparse `spls3_fit` (new in 0.6.0).
- **Inference:** `pls1_confirmatory_test`, `pls3_confirmatory_test`,
  `split_nb_gate`, `pls1_perm_null`.
- **Interpretive:** `rotate`, `pls1_rotation_stability`.

The [Python API reference](https://github.com/pawlenartowicz/plskit/blob/main/_docs/python/api.md)
documents every function, and the
[results reference](https://github.com/pawlenartowicz/plskit/blob/main/_docs/python/results.md)
every result type. Results are frozen dataclasses and carry the inputs
and seed that produced them.

Errors raised by the Rust engine surface as `plskit.PlsKitError`
(with `PlsKitInvalidWeights` and `PlsKitResamplingDegenerate`
subclasses) and carry a stable `.code` for programmatic handling.

## This is a thin wrapper

All numerical work runs inside the `plskit` Rust crate. The Python
package converts inputs to `f64`, calls the engine, and wraps the
result. Bug reports and feature requests belong on the
[monorepo issue tracker](https://github.com/pawlenartowicz/plskit/issues).

## Citation

Lenartowicz, P., Plisiecki, H. (2026). *Cheap Per-Component Testing for
PLS, Stable Under Rotation* (Under Review).

## License

GPL-3.0-or-later.
