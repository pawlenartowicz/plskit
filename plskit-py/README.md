# plskit (Python)

[![PyPI](https://img.shields.io/pypi/v/plskit.svg)](https://pypi.org/project/plskit/)
[![Python versions](https://img.shields.io/pypi/pyversions/plskit.svg)](https://pypi.org/project/plskit/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](../LICENSE)

Python wrapper for **plskit** — Partial Least Squares with modern
inference (canonical percentile CIs, `split_perm`, `split_nb`), backed
by a Rust engine.

> Part of the **[plskit project](https://github.com/pawlenartowicz/plskit)** —
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

# Confirmatory test at the pre-specified K.
sig = plskit.pls1_confirmatory_test(
    X, y, k=3,
    method="split_nb", args={"n_splits": 50}, seed=42,
)
print(sig.pvalue, sig.statistic)
```

PLS1 (single continuous `y`) is the only family available in v0.1;
PLS2, PLS3 / PLSSVD, and multi-block follow.

## Public surface

Functions: `preprocess`, `pls1_fit`, `pls1_predict`,
`pls1_confirmatory_test`, `pls1_find_k_optimal`,
`pls1_find_k_sequence`, `pls1_perm_null`, `pls1_rotation_stability`,
`rotate`.

Result types: `PreprocessResult`, `PLS1Result`,
`ConfirmatoryTestResult`, `FindKOptimalResult`, `FindKSequenceResult`,
`PermNullResult`, `RotationStabilityResult`, `RotateResult`,
`RotationSpec`, `CIScalar`, `ConfirmatoryCI`. All are frozen
dataclasses and carry the inputs and seed that produced them.

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
