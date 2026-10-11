# Limitations

What `plskit` does not do, and where its functions stop. Each item links
to the page that explains it.

## Not implemented

| Missing | What to do instead |
|---|---|
| PLS2, robust PLS1, multi-block PLS | one outcome: [PLS1](PLS1/index.md); two blocks and no outcome: [PLS3](PLS3/index.md) |
| PLS-DA or any classification variant | — |
| VIP scores, bootstrap ratios, jackknife standard errors | the [confidence intervals](PLS1/ci.md) of `pls1_confirmatory_test(ci=True)`; `pls1_perm_null` for per-variable permutation tests |
| Rotations other than varimax | — |
| Observation weights for PLS3 and sparse PLS3 | none; the functions refuse weights ([why](PLS3/fit-and-transform.md#observation-weights-are-refused)) |
| A test of a PLS3 component beyond the first, and intervals for PLS3 | none; `pls3_confirmatory_test` accepts `k = 1` only and its `ci` is always `None` ([PLS3 inference](PLS3/inference.md)) |
| A confirmatory test at a fixed `k`, or intervals, for a sparse PLS1 fit | `pls1_confirmatory_test` has no `keep`; `spls1_find_k_sequence` tests sparse components in sequence ([sparse PLS1](PLS1/sparse.md)) |
| Selection of `keep_X` / `keep_Y` for sparse PLS3 | choose them yourself |
| A joint search over `k` and `keep` | fix one and tune the other ([sPLS1](PLS1/sparse.md#one-axis-is-always-fixed)) |
| Tests for clustered or repeated-measures rows | aggregate to one row per independent unit first |

## Input data

- **Dense, real, finite.** Inputs are dense `float64` arrays. A NaN or an
  infinity raises `non_finite_input`. Missing values are not handled.
- **In memory.** There is no streaming or out-of-core mode.
  `PLSKIT_MEMORY_BUDGET` bounds the working buffers of the resampling
  loops, not `X` itself ([Memory budget](threads.md#memory-budget)).
- **One standardization recipe.** Columns are centered and scaled to unit
  population variance. Any other scaling goes in with
  `pre_standardized=True` ([Preprocessing](preprocessing.md)).
- **Rows are treated as exchangeable.** No function takes a grouping
  argument, so splits, folds and permutations are drawn over single rows
  ([clustered rows](PLS3/inference.md#clustered-rows-neither-method-is-valid)).

## Sample size

- A fit with `k` components needs `n_eff ≥ k + 1`
  ([Effective sample size](effective-sample-size.md)).
- `split_nb` is not reliable when `n_eff < 25`, when `X` has 4 columns or
  fewer, or when the stable rank of `X` is below 3. Such a request runs
  `split_exact` instead, unless you force it
  ([`test_method="auto"`](PLS1/inference.md#test_methodauto)).
- `pls3_confirmatory_test` needs `n ≥ 6`.
- The `raw_perm` test needs `n > 5`. The cross-validated selectors of
  `pls1_find_k_optimal` need `n ≥ 3`.
- `n_boot`, and the `n_perm` of `pls1_perm_null`, must be at least 100.

## Numerical limits

- **Fewer components than requested.** A fit stops when the next component
  is rounding noise and reports the count in `k_used`. It is `0` when `y`
  is constant or orthogonal to `X`
  ([Fit and predict](PLS1/fit-and-predict.md#the-pls1-model),
  [PLS3 truncation](PLS3/fit-and-transform.md#truncation)).
- **Near-constant columns.** A column whose spread is within rounding of
  its own mean is treated as constant and contributes nothing
  ([Preprocessing](preprocessing.md#the-canonical-recipe)).
- **Signs of PLS3 saliences** can flip across machines when the two
  largest entries of a salience vector are nearly equal
  ([Signs are pinned](PLS3/fit-and-transform.md#signs-are-pinned)).
- **Mis-scaled `pre_standardized` blocks.** A PLS3 block passed as
  standardized but on a much smaller scale can lose real components
  without an error ([PLS3 truncation](PLS3/fit-and-transform.md#truncation)).
- **Platforms agree within a tolerance, not bit for bit.** The same holds
  for C- and Fortran-ordered copies of one array
  ([Python API, Conventions](../python/api.md#conventions)).
- **Results can change between versions.** Pin the version to reproduce
  numbers ([Citation & reproducibility](citation.md)).

## Cost

- `split_exact` refits for every permutation at `k ≥ 2`, with a sparse
  `keep`, and in `pls3_confirmatory_test`: about `n_perm × n_splits` fits.
- `pls1_rotation_stability` accepts `2 ≤ k ≤ 7`. Its alignment step
  enumerates `2^k · k!` candidates per resample.
- `pls1_perm_alpha` works at `k = 1` only.
- A copy of `X` larger than half the memory budget makes the units that
  copy `X` run one at a time ([when to change it](threads.md#when-to-change-it)).

## Sparse fits

- Selection is not nested: component 2 need not keep the variables of
  component 1.
- `selector="bic"` does not account for `keep` and tends to select a
  larger `k` ([sparse PLS1](PLS1/sparse.md#selectorbic-is-biased-toward-more-components)).
- Sparse PLS3 saliences are not orthogonal, its `singular_values` are not
  singular values, and its alternation has no convergence guarantee. The
  fit reports `converged` per component and does not raise
  ([sPLS3](PLS3/sparse.md#how-it-is-computed),
  [what sparsity costs](PLS3/sparse.md#what-sparsity-costs)).
- Intervals after sparse selection need post-selection inference, which is
  not implemented.

## Language wrappers

- R: [known limitations](../r/differences.md#known-limitations).
- Julia: [limitations](../julia/differences.md#limitations).

## Open questions

- **PLS3 beyond the first latent variable.** Whether the statistic should
  be per-component or subspace-level when singular values are close
  ([PLS3 inference](PLS3/inference.md)).
- **`split_nb` for two blocks.** Its between-split correction is the PLS1
  heuristic, and its thresholds were not re-derived for two blocks
  ([PLS3 inference](PLS3/inference.md)).
- **Per-variable z for β at `k ≥ 2`.** `beta_sign_z` is not calibrated
  there ([Python API](../python/api.md)).
