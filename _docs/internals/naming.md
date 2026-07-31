# Naming conventions

Authoritative argument-name and option-value table for the public surface
of `plskit`. Argument names are identical across language wrappers; every
public arg has exactly one entry here.

## Scope

- **In scope:** names of arguments and string-valued option enums on the
  public surface of every function in the [Python API](../python/api.md)
  and [Rust API](../rust/api.md).
- **Out of scope:** result-object field names (covered by
  [results](../python/results.md)), internal Rust struct fields, and language-specific
  ergonomic layers (`pls1.fit(...)` Python class methods, R formula
  interfaces, Julia keyword-only sugar).

---

## Cross-language ground rules

1. **`X` / `Y` (capital) in user-facing wrappers; `x` / `y` (lowercase)
   in Rust core.** Capitals are the universal stats convention for
   matrices in user docs (sklearn, R `pls`, Julia `MultivariateStats`).
   Rust core uses lowercase per language convention. The *name* is the
   same letter; case alone is a language-bound rendering, not a renaming.
2. **`snake_case` everywhere** for multi-word arg names *and* for string
   option values (`split_nb`, `r2_se`, `two_sided`). Single-letter math
   symbols stay as-is (`X`, `Y`, `W`, `L`, `k`).
3. **R wrappers must use `k` / `k_max`, not `ncomp`.** This breaks the R
   `pls` package convention but is forced by the cross-language rule.
   Document the deviation in the R README; do not paper over it with aliases.
4. **No abbreviations from paper notation in the public API.** `n_perm`,
   not `B`; `n_splits`, not `J`; `n_boot`, not `B`. The methods paper uses
   `B` / `J`; the API does not.

---

## Convention table

Format: `name` | description | functions where used.

### 1. Data inputs

| name | description | functions |
|---|---|---|
| `X` | predictor matrix `(n, p)`, float64 | `pls1_fit`, `pls2_fit`, `pls3_fit`, `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_confirmatory_test`, `nested_cv_r2_ci`, `anisotropic_null`, `bootstrap_saliences`, `grassmannian_alignment_test` |
| `y` | target vector `(n,)`, float64 | `pls1_fit`, `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_confirmatory_test`, `nested_cv_r2_ci`, `grassmannian_alignment_test` |
| `Y` | target matrix `(n, q)`, float64 | `pls2_fit`, `pls3_fit`, `bootstrap_saliences` (PLS2/3 path) |
| `X_new` | held-out predictor matrix | `pls1_predict`, `pls2_predict`, `pls3_transform` |
| `Y_new` | held-out Y for symmetric transform | `pls3_transform` |
| `model` | fitted result object | `pls1_predict`, `pls2_predict`, `pls3_transform`, `rotate` (model overload) |
| `boot_cache` | aligned bootstrap cache | `percentile_ci`, `bsr` |
| `W` | weight matrix `(p, k)` for standalone rotation | `rotate` (array overload) |
| `L` | loading basis on which simple-structure is computed (default identity) | `rotate`, `pls1_rotation_stability` |
| `v0` | target direction `(p,)` for direction-level test | `grassmannian_alignment_test` |
| `weights` | length-n float64 vector; optional; default `None` = uniform; row-scales `(X, y/Y)` by `√w` (WLS-style precision/sampling weights) | `pls1_fit`, `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_confirmatory_test`, `pls1_rotation_stability`, `pls1_perm_null` |

`pls1_confirmatory_test` takes `(X, y, k)` directly — there is no `model` overload (the test refuses to consume a model whose K was selected on the same data).

**`plskit.preprocess(X, Y=None, weights=None)`** — public cache-pattern helper. Standardizes X (and optionally Y), normalizes weights, and returns a `PreprocessResult`. Pass `pre_standardized=True` on the downstream entry points to skip redundant rescaling.

### 2. Component counts

| name | description | functions |
|---|---|---|
| `k` | number of components to fit/test, or `"optimal"` / `"sequence"` (string modes on `pls1_fit` only) | `pls1_fit`, `pls2_fit`, `pls3_fit`, `pls1_confirmatory_test` |
| `k_max` | upper bound on k for K-selection | `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_fit(k="optimal" \| "sequence")`, `nested_cv_r2_ci` |

### 3. Resampling counts

| name | description | functions |
|---|---|---|
| `n_perm` | permutations for `raw_perm` / `split_perm` / `split_perm_nr` / Grassmannian (in `pls1_find_k_optimal`, lives in the shared `args` dict and applies to the diagnostic) | `pls1_confirmatory_test`, `pls1_find_k_optimal` (diagnostic), `pls1_find_k_sequence`, `grassmannian_alignment_test` |
| `n_splits` | split-half repetitions for `split_nb` / `split_perm` / `split_perm_nr` (in `pls1_find_k_optimal`, lives in the shared `args` dict and applies to the diagnostic) | `pls1_confirmatory_test`, `pls1_find_k_optimal` (diagnostic), `pls1_find_k_sequence` |
| `n_boot` | bootstrap / subsampling iterations | `bootstrap_saliences`, `pls1_confirmatory_test` (`ci=True`), `pls1_rotation_stability` |
| `m_rate` | subsampling exponent: resolved subsample size is `m = ceil(n^m_rate)` (default `0.7`; must satisfy `0.5 < m_rate < 0.95`) | `pls1_confirmatory_test` (`ci=True`), `pls1_rotation_stability` |
| `n_folds` | CV folds | `pls1_find_k_optimal` (`r2_se` / `r2_max`), `pls1_confirmatory_test` (`raw_perm`) |
| `outer_folds` | outer CV folds for nested CV | `nested_cv_r2_ci` |
| `inner_folds` | inner CV folds for nested CV | `nested_cv_r2_ci` |

### 4. Method dispatch

| name | description | functions |
|---|---|---|
| `method` | algorithm tag (string) | `pls1_confirmatory_test`, `rotate`, `grassmannian_alignment_test` |
| `args` | method-specific kwargs (dict on the wrapper boundary, `enum` in Rust) | every method-axis function |
| `rotation_method` | algorithm tag for the rotation applied inside the diagnostic (parallel to `method` for `rotate`) | `pls1_rotation_stability` |
| `rotation_args` | method-specific kwargs for the inner rotation (parallel to `args` for `rotate`) | `pls1_rotation_stability` |
| `find_k_args` | method-specific kwargs forwarded by `pls1_fit(k="optimal" \| "sequence")` to the underlying `pls1_find_k_*` call. Allowed keys are the public params of the target function except `seed` / `pre_standardized` / `weights` / `disable_parallelism` / `verbose`, which live on `pls1_fit`. Unknown keys raise `PlsKitError(code="invalid_args")` listing the allowed set. | `pls1_fit` |
| `selector` | K-selection criterion (`"r2_se"` / `"r2_max"` / `"bic"`) | `pls1_find_k_optimal` |
| `test_method` | inner test for the sequential closed-test path (`"raw_perm"` / `"split_nb"` / `"split_perm"` / `"e"`; `"score"` and `"split_perm_nr"` rejected — no sequential variant) | `pls1_find_k_sequence` |
| `diagnostic` | optional same-sample sequential diagnostic on `pls1_find_k_optimal`; same enum as `test_method` but `None` disables it. Distinct param name encodes "diagnostic, not confirmatory inference." | `pls1_find_k_optimal` |
| `quantity` | CI/BSR target (`"salience"` / `"loading"` / `"beta"` / `"score_loading"`) | `percentile_ci`, `bsr` |
| `which` | scores to project (`"x_scores"` / `"y_scores"` / `"both"`) | `pls3_transform` |
| `statistic` | Grassmannian statistic (`"cosine"` / `"sin2"` / `"procrustes"`) | `grassmannian_alignment_test` |
| `null` | null kind (`"y_perm"` / `"anisotropic"`) | `grassmannian_alignment_test` |

There is no `at` argument (legacy `"fitted_k"` / `"first_k"` / `"postselection"` is replaced by the explicit confirmatory-vs-exploratory split — confirmatory tests live in `pls1_confirmatory_test`, exploratory K-selection lives in `pls1_find_k_optimal(diagnostic=...)`). There is no `mode` argument (the cumulative sequential path was dropped — incremental closed testing is the only sequential).

### 5. Inference scalars

| name | description | functions |
|---|---|---|
| `alpha` | significance threshold for the closed-test path | `pls1_find_k_sequence` |
| `level` | CI confidence level (default 0.95) | `percentile_ci`, `nested_cv_r2_ci`, `pls1_confirmatory_test` (`ci=True`), `pls1_rotation_stability` |
| `strata` | block labels for blocked resampling | `bootstrap_saliences` |

### 6. Reproducibility

| name | description | functions |
|---|---|---|
| `seed` | RNG seed; `None` → OS entropy, recorded as `result.seed` | `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_confirmatory_test`, `pls1_rotation_stability` |

### 7. Pre-standardization flags

| name | description | functions |
|---|---|---|
| `pre_standardized` | skip X+Y centering/scaling AND weight normalization (unified PLS1 flag) | `pls1_fit`, `pls1_find_k_optimal`, `pls1_find_k_sequence`, `pls1_confirmatory_test`, `pls1_rotation_stability`, `pls1_perm_null` |
| `pre_standardized_X` | skip X centering/scaling | `pls2_fit`, `pls3_fit` |
| `pre_standardized_Y` | skip Y centering/scaling | `pls2_fit`, `pls3_fit` |

### 8. Algorithm internals

| name | description | functions |
|---|---|---|
| `tol` | convergence tolerance | `pls1_fit`, `pls2_fit`, `rotate` (`varimax` args) |
| `max_iter` | iteration cap | `pls1_fit`, `pls2_fit`, `rotate` (`varimax` args) |
| `kaiser_normalize` | Kaiser row-normalize before varimax | `rotate` (`varimax` args) |
| `estimator` | CI estimator (`"percentile"` / `"bca"`) | `percentile_ci` |

### 9. Operational

| name | description | functions |
|---|---|---|
| `disable_parallelism` | force serial execution | every long-running core function |
| `verbose` | progress to stderr | every long-running core function |
| `max_skip_rate` | float in [0, 1]; default `0.01`; subsample loop raises `PlsKitResamplingDegenerate` if the fraction of skipped resamples exceeds this threshold | `pls1_confirmatory_test(ci=True)`, `pls1_rotation_stability` |

### 10. Rotation-invariant subsample readouts (result-only; not args)

These names appear only on result objects (`ConfirmatoryCI`,
`RotationStabilityResult`); the user does not pass them in. They are
listed here so language wrappers stay aligned on the names. These are
computed in Rust core and surface verbatim across wrappers.

| name | shape / type | description |
|---|---|---|
| `m` | `int` | resolved subsample size, `m = ceil(n^m_rate)` |
| `agreement` | `CIScalar` | scalar summary of post-procrustes Frobenius agreement (rotation-stability output) |
| `subspace_cos` | `CIScalar` | composite: cosine of subspace angle between resampled and full-fit `W` |
| `cos_beta` | `CIScalar` | composite: cosine between resampled and full-fit `β` |
| `beta_norm` | `CIScalar` | composite: `‖β_resample‖` distribution |
| `holdout_corr` | `CIScalar` | composite: NB-adjusted Wald CI on out-of-sample correlation |
| `beta_sign_z` | `(D,)` array | per-variable: uncorrected sign-stability z |
| `leverage_ci_lower` | `(D,)` array | per-variable: subsampling CI lower bound on leverage |
| `leverage_ci_upper` | `(D,)` array | per-variable: subsampling CI upper bound on leverage |
| `leverage_se` | `(D,)` array | per-variable: subsampling SE of leverage |

### Preprocess helper output (`PreprocessResult` fields)

`PreprocessResult` is returned by `plskit.preprocess(...)`. Field names below are binding across language wrappers.

| name | shape / type | description |
|---|---|---|
| `X_std` | `(n, p)` array | standardized X (zero mean, unit SD per column) |
| `X_mean` | `(p,)` array | per-column mean subtracted from X |
| `X_scale` | `(p,)` array | per-column SD used to scale X |
| `Y_std` | `(n,)` or `(n, q)` array; `None` if Y not supplied | standardized Y |
| `Y_mean` | `(q,)` array; `None` if Y not supplied | per-column mean subtracted from Y |
| `Y_scale` | `(q,)` array; `None` if Y not supplied | per-column SD used to scale Y |
| `weights_normalized` | `(n,)` array; `None` if weights not supplied | weights normalized to sum to `n` (i.e., `w / mean(w)`) |
| `n_eff` | `float` | effective sample size: `(sum(w))² / sum(w²)` (equals `n` for uniform weights) |

### Dropped from earlier drafts

- `split_ratio` — train fraction is fixed at 0.5 internally; never reach the public surface.
- `at` (`"fitted_k"` / `"first_k"` / `"postselection"`) — replaced by the confirmatory-vs-exploratory split.
- `mode` (`"incremental"` / `"cumulative"`) — only the incremental closed-test path survives, and it lives inside `pls1_find_k_sequence`.
- `stop_early` — sequential closed testing always stops at the first non-rejection.
- `selector="sequence"` / `"cv_q2"` strings — replaced by `"r2_se"` / `"r2_max"` / `"bic"` on `pls1_find_k_optimal`.

---

## Status

The 2026-04 confirmatory-vs-exploratory overhaul is implemented. Argument names
in the implemented surface (`pls1_fit`, `pls1_predict`,
`pls1_confirmatory_test`, `pls1_find_k_optimal`, `pls1_find_k_sequence`,
`rotate`) match this convention. The 2026-05 observation-weights work
uses `pre_standardized` (not `pre_standardized_X`) for all PLS1 entry
points and added `weights`, `max_skip_rate`, and `plskit.preprocess`.

What is **not yet implemented**, and where this convention is
forward-looking, is the Pillar-1 / Pillar-3 surface: `bootstrap_saliences`,
`percentile_ci`, `bsr`, `nested_cv_r2_ci`, `grassmannian_alignment_test`,
`anisotropic_null`, `rotation_stability`, plus `pls2_fit`, `pls3_fit`,
`pls2_predict`, `pls3_transform`. The argument names listed for those
functions in the tables above are binding on first implementation; gate
via PR review.

---

## Live tensions

The four points where this convention chose against an alternative.
Override by editing the table; do not ship code that mixes both.

1. **`n_perm` / `n_splits` / `n_boot` over `B_perm` / `J` / `B_boot`.**
   Code uses `n_*`. Override only if a future paper figure forces the
   methods-paper notation back into the API.
2. **`pre_standardized` over `pre_standardized_X`.** A single flag
   covers standardize-X, standardize-Y, and weight-normalization as a
   unit; the √w row-scaling stays in the fit, not preprocessing. The
   earlier `_X` / `_Y` split was motivated by chemometrics partial
   standardization, but pre-treated data round-trips at negligible cost
   and non-default chemometrics recipes need a "skip everything" route
   the split form never provided. Loop-cached workflows go through
   `plskit.preprocess`. Locked.
3. **`"two_sided"` over `"two-sided"`.** Hyphens are the only non-snake
   string values. Locked.
4. **`estimator` over `type` on `percentile_ci`.** `type` is awkward in
   every target language. Override only to copy sklearn's habit of
   using `type` as a kwarg.
