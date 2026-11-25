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
- Seeds: random draws use Philox4x32-10 through `commonstats`, and a fit at a fixed `k`, a prediction or a rotation draws nothing. Permutation nulls follow commonstats' draw convention: at seed `s` on `n` rows, null column `c` takes row `perm[i]` of the outcome as its row `i`, with `perm` from `commonstats::gen_resample_indices(Scheme::Permutation, n, c, s)`. A neurostats test at the same seed and rows therefore uses the same row shuffles, and a plskit call with `n_perm = B − 1` sees neurostats' draws `1..B` (neurostats counts the unshuffled draw 0 in `B`). With `covariates`, plskit permutes `ỹ`, the residual of y on `[1, Z]`, and residualises the permuted `ỹ` again on the unpermuted `Z` (Freedman–Lane); weights stay on their rows. With weights, the draw permutes only the `m` rows of positive weight, taken in increasing order, with `perm` from `gen_resample_indices(Scheme::Permutation, m, c, s)`, and a row of weight 0 keeps its place; without a weight of 0 this is the draw above.
- Cross-language reproducibility: Rust / Python / R / Julia at the same version produce numerically equivalent outputs (within tolerance)
- Tolerance specifics: bit-for-bit on deterministic ops, statistical-equivalence on stochastic ops
