# Citation & reproducibility

Citing `plskit` and pinning a version are the same workflow: a paper that
cites this library should also pin the version (and seed) so a reader can
reproduce the numbers exactly.

## Citing plskit

```bibtex
@inproceedings{lenartowicz2026cheap,
  title     = {Cheap and Powerful Tests for Supervised Subspaces: Per-Component Inference for {PLS}},
  author    = {Lenartowicz, Pawe{\l} and Plisiecki, Hubert},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=xb6CB7d9LO}
}
```

## Reproducibility

- The bit-near contract: same `(X, y, seed, version)` reproduces results across platforms within tolerance
- Why version pinning matters — across versions, results may change
- Seeds: how each entry point consumes its seed; deterministic vs stochastic operations
- Cross-language reproducibility: Rust / Python / R / Julia at the same version produce numerically equivalent outputs (within tolerance)
- Tolerance specifics: bit-for-bit on deterministic ops, statistical-equivalence on stochastic ops
