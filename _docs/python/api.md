# Python API

> Argument names follow [naming](../internals/naming.md). Result-field
> shapes live in [results](results.md).

The Python wrapper exposes the PLS1 family of plskit. Every function
listed here is reachable from `import plskit`. Each entry documents
what the function does, what it takes, and what it returns; brief
design rationale is omitted here and lives in the internal API surface
contract (maintainer-facing).

## Conventions

- **Method-axis dispatch** uses `(method, args)`: a `method` string +
  an `args` dict for method-specific kwargs. Cross-cutting kwargs
  (`seed`, `weights`, `pre_standardized`) live at the top level.
- Type hints use NumPy notation: `np.ndarray, shape (n, d)` is a 2-D
  array; `np.ndarray, shape (n,)` is a 1-D vector.
- Per-function entry format:

  ```
  function: <name>
  need: <what problem it solves>
  arguments: <required positional inputs>
  options: <keyword-only switches with their defaults>
  args by method: <when method-axis dispatch — args dict per method tag>
  returns: <result type and the high-bit fields>
  ```

---

## 1. Preprocessing

**function:** `preprocess`
**need:** standardize `X` / `Y` and normalize `weights` using plskit's
canonical recipe — useful when calling several plskit functions
back-to-back on the same data, to avoid recomputing the standardization.
All arguments are optional; only the fields matching passed inputs are
populated.
**arguments:** `X` (optional), `Y` (optional), `weights` (optional)
**options:** —
**returns:** `PreprocessResult` with `X_std` / `X_mean` / `X_scale`,
`Y_std` / `Y_mean` / `Y_scale`, `weights_normalized`, `n_eff`. Pass
the standardized arrays to subsequent calls with `pre_standardized=True`.

---

## 2. Core fit / predict

### 2.1 Fit

**function:** `pls1_fit`
**need:** NIPALS PLS1 — single continuous `y`, asymmetric X→y predictive.
**arguments:** `X`, `y`, `k` (`int | "optimal" | "sequence"`, default `1`)
**options:** `k_max` (required when `k` is a string); `find_k_args`
(dict of method-specific kwargs forwarded to `pls1_find_k_optimal` /
`pls1_find_k_sequence`; allowed keys are the public params of the
target function except `seed` / `pre_standardized` / `weights` /
`disable_parallelism` / `verbose`, which live on `pls1_fit` itself;
unknown keys raise `PlsKitError(code="invalid_args")`);
`pre_standardized` (bool, default `False`); `tol` (float, default
`1e-9`); `max_iter` (int, default `500`); `seed` (`int | None`);
`weights` (length-`n` vector; default `None` = uniform).
Rotation is post-fit; see `rotate`.
**returns:** `PLS1Result`. When `k="sequence"` and
`pls1_find_k_sequence` returns `k_star=0` (no component rejected at
`alpha`), raises `PlsKitError(code="sequence_no_rejection")`. Callers
that want to fit anyway must call `pls1_find_k_sequence` directly,
inspect the result, and pass an explicit `int` k.

### 2.2 Predict

**function:** `pls1_predict`
**need:** apply a fitted PLS1 model to new `X` → ŷ.
**arguments:** `model`, `X_new`
**options:** —
**returns:** `np.ndarray` of predictions, shape `(n_new,)`.

### 2.3 K-selection

Two distinct entry points for the two distinct workflows. The split
makes the exploratory-vs-confirmatory boundary visible in the function
name itself.

**function:** `pls1_find_k_optimal`
**need:** point estimate of `K*` from a selector criterion on the full
data; optionally a per-component same-sample diagnostic.
**arguments:** `X`, `y`, `k_max`
**options:** `selector` (`"r2_se"` | `"r2_max"` | `"bic"`; default
`"r2_se"`); `diagnostic` (`"raw_perm"` | `"split_nb"` | `"split_perm"`
| `"e"` | `None`; default `None` — `None` disables the diagnostic;
`"score"` is rejected, has no sequential variant); `args` (dict —
selector key `n_folds`; diagnostic keys `n_perm` for `raw_perm` /
`split_perm`, `n_splits` for `split_nb` / `split_perm`. Diagnostic
keys require `diagnostic` to be set.); `pre_standardized`; `weights`;
`seed`; `disable_parallelism`; `verbose`.
**returns:** `FindKOptimalResult`. When `diagnostic` is set, the
`pvalues` and `diagnostic` fields are populated; selection and the
diagnostic reuse the same data, so the pvalues are a robustness
check, not honest inference. The `diagnostic=` parameter name (vs.
`test_method=` on `pls1_find_k_sequence`) is the structural signal —
same enum, different inferential weight.

**function:** `pls1_find_k_sequence`
**need:** sequential closed-test on nested hypotheses — "how many
components carry signal at α?" with exact FWER control.
**arguments:** `X`, `y`, `k_max`
**options:** `test_method` (`"raw_perm"` | `"split_nb"` |
`"split_perm"` | `"e"`; default `"split_nb"`); `args` (dict of
method-specific kwargs: `n_perm`, `n_splits`); `alpha` (default
`0.05`); `pre_standardized`; `weights`; `seed`;
`disable_parallelism`; `verbose`. Stop-early at the first
non-rejection is hard-coded on; `K*` is the count of components that
rejected before the first failure.
**returns:** `FindKSequenceResult`. Closed testing on nested H is
exact, so the per-step pvalues form an honest FWER-controlled
sequence. To recover the path-max p-value, compute
`np.nanmax(r.pvalues[:r.k_star])`.

---

## 2b. Sparse PLS1 (sPLS1)

Sparse PLS1 fits a NIPALS model with a hard keep-count constraint: each
latent direction loads on at most `keep` X variables. `keep` is a scalar
integer broadcast to all components; `keep ∈ [1, n_features]`.
`keep = n_features` reproduces the dense functions bit-exactly. One axis
is always fixed — `spls1_find_keep_optimal` tunes `keep` at fixed `k`,
and `spls1_find_k_optimal` / `spls1_find_k_sequence` tune `k` at fixed
`keep`. There is no joint `(k, keep)` 2-D search. Prediction uses
`pls1_predict` on the spls1 model (the result object is a `PLS1Result`
with `keep` set); there is no separate `spls1_predict`. Per-coordinate β
CIs are **not** offered under selection: the zeros in the weight vector
are selection events, so subsample CIs on the selected β require
post-selection inference and are deferred to a separate spec.

### 2b.1 Sparse fit

**function:** `spls1_fit`
**need:** NIPALS PLS1 with hard keep-count selection — each LV direction
retains the `keep` largest-magnitude X loadings and zeros the rest.
**arguments:** `X`, `y`, `k`, `keep`
**options:** `pre_standardized` (bool, default `False`);
`weights` (length-`n` vector; default `None` = uniform).
No `'optimal'` / `'sequence'` string modes for `k` — use the
`spls1_find_*` entry points directly.
**returns:** `PLS1Result` with `keep` populated; exactly `keep` nonzeros
per `W` column.

### 2b.2 Keep-count tuning

**function:** `spls1_find_keep_optimal`
**need:** select the sparsest `keep` whose CV R² is within 1 SE of the
best — the standard 1-SE rule applied to the keep axis instead of the k
axis.
**arguments:** `X`, `y`, `k` (fixed component count for every fit in the
sweep)
**options:** `args` (`{'n_folds': int}`, default 5); `seed`;
`disable_parallelism`; `verbose`; `weights`.
**selection method:** logged geometric grid over `[1, n_features]`
(powers of two, endpoints always included); the swept grid is reported on
`result.keep_grid`. Sparsest-within-1-SE selection on mean CV R²; ties
broken toward sparser. Sparsity is tuned inside the training split, never
on test data.
**returns:** `FindKeepOptimalResult`.

### 2b.3 K-selection at fixed keep

**function:** `spls1_find_k_optimal`
**need:** `pls1_find_k_optimal` with the inner fitter swapped to the
sparse fit at the caller's fixed `keep`. Same selectors, diagnostic, and
result shape as the dense version. `keep = n_features` reproduces the
dense function bit-exactly.
**arguments:** `X`, `y`, `k_max`, `keep`
**options:** `selector` (`"r2_se"` | `"r2_max"` | `"bic"`; default
`"r2_se"`); `diagnostic` (`"raw_perm"` | `"split_nb"` | `"split_perm"`
| `"e"` | `None`; default `None`); `args`; `pre_standardized`; `seed`;
`disable_parallelism`; `verbose`; `weights`.
**Dense-BIC caveat:** `selector='bic'` reuses the dense complexity
penalty — it does not account for `keep`; under sparsity this
under-penalizes added components and biases the selected k upward.
Deliberate v1 simplification.
**returns:** `FindKOptimalResult` (same type as `pls1_find_k_optimal`).

**function:** `spls1_find_k_sequence`
**need:** `pls1_find_k_sequence` with the inner fitter swapped to the
sparse fit at the caller's fixed `keep`. Each step deflates on the sparse
residual and tests the sparse marginal component — a coherent sequential
test. `keep = n_features` reproduces the dense function bit-exactly.
**arguments:** `X`, `y`, `k_max`, `keep`
**options:** `test_method` (`"raw_perm"` | `"split_nb"` | `"split_perm"`
| `"e"`; default `"split_nb"`); `alpha` (default `0.05`); `args`;
`pre_standardized`; `seed`; `disable_parallelism`; `verbose`; `weights`.
**returns:** `FindKSequenceResult` (same type as `pls1_find_k_sequence`).

---

## 3. Inference

### 3.1 Confirmatory omnibus test

**Five test methods.** `raw_perm` / `split_nb` / `split_perm` are the
predictive-validity split-resampling family (the methods paper's core);
`score` is closed-form on `T = ‖X′y‖²` (generalized χ² under Gaussian
y, anisotropy-aware by construction, K-free); `e` is universal
inference (split-LR e-value, calibration-free, non-asymptotic α bound
— `P(reject | H₀) ≤ α` exactly, with a power tax of ~30–50% vs.
`split_nb` as the validity cost). `score` and `split_nb` complement
rather than replace each other — `score` wins on diffuse signal across
many directions, `split_nb` wins on signal concentrated in a few
directions; reporting both side-by-side is the canonical pattern.

**function:** `pls1_confirmatory_test`
**need:** omnibus null test at a pre-specified `k` ("is there signal
at K?"). Optionally runs an independent subsample pass for
rotation-invariant CIs.
**arguments:** `X`, `y`, `k` (default `1`)
**options:**

- `method` (`"raw_perm"` | `"split_nb"` | `"split_perm"` | `"score"` |
  `"e"`)
- `args` (dict of method-specific kwargs)
- `ci` (bool, default `False`) — when `True`, runs an independent
  subsample pass after the headline test and populates `result.ci`.
- `n_boot` (int, default `1000`; must be `≥ 100`); `m_rate` (float,
  default `0.7`; `0.5 < m_rate < 0.95`); `level` (float, default
  `0.95`; `0.5 ≤ level ≤ 0.99`); `max_skip_rate` (float, default
  `0.01`); `max_failure_rate` (float, default `0.01`).
- `pre_standardized`; `weights`; `seed`; `disable_parallelism`;
  `verbose`.

**args by method:**

- `"raw_perm"` — `n_perm`, `n_folds`
- `"split_nb"` — `n_splits`
- `"split_perm"` — `n_perm`, `n_splits`
- `"score"` — none (anisotropy handled internally by Welch–Satterthwaite)
- `"e"` — none

**default `k=1`:** the omnibus question "is there *any* signal?" is
power-optimized at `k=1`. All `k ≥ 1` are exact under the null, but
power generally falls as `k` grows (more nuisance directions diluting
the signal). Pass an explicit `k` only when you have a prior reason to
fix it higher.

**Honest use:** `pls1_confirmatory_test` does not pre-validate that
you didn't pick `k` from the same data via `pls1_find_k_*`. Honest
confirmatory inference means either fixing `k` from prior knowledge or
holding out a fresh sample.

**returns:** `ConfirmatoryTestResult`. When `ci=True`, the `.ci` field
is a `ConfirmatoryCI` bundle: a Fisher z-transformed Wald CI on
holdout correlation (`holdout_corr`, a `CIScalar`), per-variable
sign-stability z (`beta_sign_z` / `beta_sign_z_signed`), per-variable
leverage CI (`leverage_ci_lower` / `leverage_ci_upper` /
`leverage_ci_se`), and per-coordinate β CIs (`beta_ci_lower` /
`beta_ci_upper` / `beta_ci_se`). The per-coordinate `beta_ci_*` arrays
are a regression-style diagnostic; canonical inference is
`beta_sign_z` + `leverage_ci_*` + `holdout_corr`. See
[results](results.md) for field shapes and the
shrinkage / multiplicity / standardization caveats that apply to
`beta_ci_*`.

### 3.2 Permutation-null engine

**function:** `pls1_perm_null`
**need:** signed per-voxel z + (optional) full perm matrix, suitable
for downstream FWER correction (TFCE / max-stat / cluster-mass) at
fMRI / NIRS scale.
**arguments:** `X`, `y`, `k`
**options:** `n_perm` (int, default `1000`); `return_perm_matrix`
(bool, default `False`); `pre_standardized`; `seed`;
`disable_parallelism`; `verbose`; `weights`.
**returns:** `PermNullResult`. Pair with
`pls1_confirmatory_test(method="split_nb")` as an omnibus gate before
spending the `n_perm` permutation budget.

---

## 4. Interpretive

### 4.1 Rotation

**function:** `rotate`
**need:** simple-structure rotation of `W`. Can be called on a fitted
result to stamp a `RotationSpec`, or on a bare `W` matrix.
**arguments:** `model_or_W` (a PLS1Result or a 2-D np.ndarray)
**options:** `method` (`"varimax"`; default `"varimax"`); `args` (dict
of method-specific kwargs); `L` (loading basis on which simplicity is
computed; default identity → varimax on `W` directly).

**args by method:**

- `"varimax"` — `max_iter` (default `50`), `tol` (default `1e-8`),
  `kaiser_normalize` (default `True`).

**Pluggable `L`:** the loading basis on which simple-structure is
computed. Default identity rotates `W` directly; passing an alternative
`L` is the strict-superset extension over SSD's `mpls_fit`.

**returns:** `RotateResult` when called on `np.ndarray`; a new
`PLS1Result` (with `.rotation_spec` populated) when called on
`PLS1Result`. Re-rotation of an already-rotated `PLS1Result` raises
`PlsKitError(code="already_rotated")` in v0.1.x.

### 4.2 Rotation-stability diagnostic

**function:** `pls1_rotation_stability`
**need:** standalone subsampling diagnostic — does the chosen rotation
converge to the same axis permutation across resamples? Output is a
single post-procrustes Frobenius `CIScalar` summarizing whether
rotated-basis labels are stable.
**arguments:** `X`, `y`, `k`
**options:** `rotation_method` (`"varimax"`; default `"varimax"`);
`rotation_args` (dict); `L` (loading basis; default identity);
`n_boot` (int, default `1000`, `≥ 100`); `m_rate` (float, default
`0.7`, `0.5 < m_rate < 0.95`); `level` (float, default `0.95`,
`0.5 ≤ level ≤ 0.99`); `pre_standardized`; `weights`;
`max_skip_rate` (float, default `0.01`); `seed`;
`disable_parallelism`; `verbose`.

**Constraints on k:** `2 ≤ k ≤ 7`. `k = 1` is rejected because
rotation is the identity on a 1-D subspace, making the diagnostic
meaningless. `k > 7` is rejected because the discrete
signed-permutation alignment used internally enumerates `2^k * k!`
candidates per replicate; at `k = 8` that is 10,321,920 candidates
per replicate, which is not tractable within the bootstrap loop.

**returns:** `RotationStabilityResult`. See [results](results.md).

---

## Result objects

See [results](results.md) for full field shapes:

- `PreprocessResult`
- `PLS1Result` (`keep: int | None` — populated by `spls1_fit`, `None` for dense fits)
- `ConfirmatoryTestResult` (with optional `ConfirmatoryCI`)
- `FindKOptimalResult`, `FindKSequenceResult`
- `FindKeepOptimalResult` — returned by `spls1_find_keep_optimal`
- `PermNullResult`
- `RotateResult`, `RotationSpec`
- `RotationStabilityResult`
- `CIScalar` — the rotation-invariant CI primitive used throughout

## Errors

- `PlsKitError` — base error, with `code` for programmatic handling.
- `PlsKitInvalidWeights` — weights vector failed validation.
- `PlsKitResamplingDegenerate` — subsample loop exceeded
  `max_skip_rate` or `max_failure_rate`.
