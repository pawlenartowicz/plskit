# Citation & reproducibility

Citing `plskit` and pinning a version are the same workflow: a paper that
cites this library should also pin the version (and seed) so a reader can
reproduce the numbers exactly.

## Citing plskit

Lenartowicz, P., Plisiecki, H. (2026). *Cheap Per-Component Testing for PLS,
Stable Under Rotation* (Under Review).

## Reproducibility

- The bit-near contract: same `(X, y, seed, version)` reproduces results across platforms within tolerance
- Why version pinning matters — across versions, results may change
- Seeds: how each entry point consumes its seed; deterministic vs stochastic operations
- Cross-language reproducibility: Rust / Python / R / Julia at the same version produce numerically equivalent outputs (within tolerance)
- Tolerance specifics: bit-for-bit on deterministic ops, statistical-equivalence on stochastic ops
