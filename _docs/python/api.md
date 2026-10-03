# Python API

> Argument names follow [naming](../internals/naming.md). Result-field
> shapes live in [results](results.md).

The Python wrapper exposes the PLS1, sparse PLS1, and PLS3 / PLSSVD
families of plskit. Every function
listed here is reachable from `import plskit`. Each entry documents
what the function does, what it takes, and what it returns. Design
rationale is kept short and appears only where it changes how you call
a function or read its result.

## Conventions

- To cap the thread count, set `PLSKIT_NUM_THREADS`; see
  [Threads](../concepts/threads.md).
- **Method-axis dispatch** uses `(method, args)`: a `method` string +
  an `args` dict for method-specific kwargs. Cross-cutting kwargs
  (`seed`, `weights`, `pre_standardized`) live at the top level.
- Type hints use NumPy notation: `np.ndarray, shape (n, d)` is a 2-D
  array; `np.ndarray, shape (n,)` is a 1-D vector.
- **Arrays** are read as `float64` in their own memory layout. A C- or
  F-ordered `float64` `X` reaches the engine without a copy, and so does
  an F-ordered `float64` `Y`, `Y_new`, `W` or `L` (a C-ordered one is
  copied column-major, which gives the same bits as F order); other
  dtypes are converted, and a strided view is copied by the extension.
  Results for two layouts of the same values agree to rounding (the
  corpus tolerance, `1e-10 + 1e-14·|value|`), not bit for bit, and a
  fit sitting on its truncation floor can stop at a different
  `k_used`. The last bits move with the layout in `pls1_fit`,
  `spls1_fit`, `pls1_predict` and `pls1_rotation_stability`; the others
  currently give C and F order the same bits. The same array in the
  same layout gives byte-identical results on every run and at every
  thread count. An array numpy cannot read as real numbers (strings,
  numeric strings included, ragged nested lists, complex values,
  non-numeric objects) raises `PlsKitError(code="invalid_argument")`;
  booleans (read as `1`/`0`) and integers are converted, and `None` or
  `pd.NA` in an object array reads as NaN (`non_finite_input`).
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
**arguments:** `X` (optional), `Y` (optional, 1-D or 2-D), `weights` (optional)
**options:** —
**returns:** `PreprocessResult` with `X_std` / `X_mean` / `X_scale`,
`Y_std` / `Y_mean` / `Y_scale`, `weights_normalized`, `n_eff`. Pass
the standardized arrays to subsequent calls with `pre_standardized=True`.
A 2-D `Y` is validated exactly as a 1-D one (row counts, finiteness,
weights; NaN or infinity raises `code="non_finite_input"`) and each of its
columns is standardized as a 1-D `Y` would be. An `X` or `Y` with zero
rows raises `code="invalid_argument"` (an empty column has no mean).

---

## 2. Core fit / predict

### 2.1 Fit

**function:** `pls1_fit`
**need:** the PLS1 (NIPALS) model: single continuous `y`, asymmetric X→y predictive.
**arguments:** `X`, `y`, `k` (`int | "optimal" | "sequence"`, default `1`)
**options:** `k_max` (required when `k` is a string); `find_k_args`
(dict of method-specific kwargs forwarded to `pls1_find_k_optimal` /
`pls1_find_k_sequence`; allowed keys are the public params of the
target function except `seed` / `pre_standardized` / `weights` /
`verbose`, which live on `pls1_fit` itself;
unknown keys raise `PlsKitError(code="invalid_args")`);
`pre_standardized` (bool, default `False`); `seed` (`int | None`,
forwarded to the k-selection call when `k` is a string);
`weights` (length-`n` vector; default `None` = uniform). There is no
convergence tolerance or iteration cap to set: each component of the
PLS1 (NIPALS) model comes from a single pass.
Rotation is post-fit; see `rotate`.
**returns:** `PLS1Result`. When `k="sequence"` and
`pls1_find_k_sequence` returns `k_star=0` (no component rejected at
`alpha`), raises `PlsKitError(code="sequence_no_rejection")`. Callers
that want to fit anyway must call `pls1_find_k_sequence` directly,
inspect the result, and pass an explicit `int` k. Likewise, when
`k="optimal"` and `pls1_find_k_optimal` returns `k_star=0` (`y` is
constant or orthogonal to `X`, so no first component exists; at `n >=
3`, since the CV selectors reject `n <= 2` first), raises
`PlsKitError(code="optimal_no_component")`; an explicit `int` k returns
the `k_used=0` zero model.

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
`"r2_se"`); `diagnostic` (`"raw_perm"` | `"split_nb"` | `"split_exact"`
| `"e"` | `None`; default `None` — `None` disables the diagnostic;
`"score"` is rejected, has no sequential variant); `args` (dict —
selector key `n_folds`; diagnostic keys `n_perm` for `raw_perm` /
`split_exact`, `n_splits` for `split_nb` / `split_exact`, `force` for
`split_nb`. Diagnostic keys require `diagnostic` to be set.);
`pre_standardized`; `weights`; `seed`; `verbose`.
`diagnostic="raw_perm"` uses a fixed 5-fold CV at every step and needs
`n > 5`; below that it raises `invalid_argument` (leave-one-out would
otherwise make every validation fold a single row). For `selector`
`"r2_se"` / `"r2_max"`, the CV layer's own `n_folds` (`args`, default 5)
is capped at `n - 2` and floored at 2; at very small `n` (`n <= 2`) that
floor pushes the effective fold count back up to `n`, which is the same
leave-one-out degeneracy and also raises `invalid_argument`.
**returns:** `FindKOptimalResult`. A `"split_nb"` diagnostic the
auto-gate flags reroutes to `"split_exact"`; `result.diagnostic` says
so and Python warns. Pass `args={'force': True}` to run `"split_nb"`
anyway. When `diagnostic` is set, the `pvalues` and `diagnostic`
fields are populated; selection and the diagnostic reuse the same
data, so the pvalues are a robustness check, not honest inference. The `diagnostic=` parameter name (vs.
`test_method=` on `pls1_find_k_sequence`) is the structural signal —
same enum, different inferential weight. When the full-data fit
cannot extract a first component (`y` constant, or orthogonal to `X` up
to rounding), every selector returns `k_star=0` with an empty score
dict, matching `pls1_find_k_sequence` (for `r2_se` / `r2_max` only at
`n >= 3`, since those CV selectors reject `n <= 2` first; `bic` is
unaffected).

**function:** `pls1_find_k_sequence`
**need:** sequential closed-test on nested hypotheses — "how many
components carry signal at α?" with exact FWER control.
**arguments:** `X`, `y`, `k_max`
**options:** `test_method` (`"raw_perm"` | `"split_nb"` |
`"split_exact"` | `"e"` | `"auto"`; default `"auto"`, see
[`test_method="auto"`](../concepts/PLS1/inference.md#test_methodauto)); `args` (dict of
method-specific kwargs: `n_perm`, `n_splits`, and `force` for
`split_nb`; `"auto"` takes `n_perm` and `n_splits`, not `force`); `alpha` (default `None`: the engine default,
`0.05`); `pre_standardized`; `weights`; `seed`;
`verbose`. `test_method="raw_perm"` uses a
fixed 5-fold CV at every step and needs `n > 5`; below that it raises
`invalid_argument` (leave-one-out would otherwise make every
validation fold a single row). Stop-early at the first
non-rejection is hard-coded on; `K*` is the count of components that
rejected before the first failure. The `split_nb` auto-gate is
evaluated once for the whole sequence: a flagged explicit `split_nb` request runs
`split_exact` for every step and `result.test_method` says so. `"auto"`
resolves once on the full `X` and runs that method for every step.
**returns:** `FindKSequenceResult`. Closed testing on nested H is
exact, so the per-step pvalues form an honest FWER-controlled
sequence. To recover the path-max p-value, compute
`np.nanmax(r.pvalues[:r.k_star])`.

---

## 2b. Sparse PLS1 (sPLS1)

Concepts: [sPLS1](../concepts/sPLS1/index.md).

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
**need:** the PLS1 (NIPALS) model with hard keep-count selection: each LV
direction retains the `keep` largest-magnitude X loadings and zeros the rest.
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
`verbose`; `weights`. `n_folds` is capped at
`n - 2` and floored at 2; at very small `n` (`n <= 2`) that floor pushes
the effective fold count back up to `n`, leave-one-out (every validation
fold a single row), which raises `invalid_argument`.
**selection method:** logged geometric grid over `[1, n_features]`
(powers of two, endpoints always included); the swept grid is reported on
`result.keep_grid`. Sparsest-within-1-SE selection on mean CV R²; ties
broken toward sparser. Sparsity is tuned inside the training split, never
on test data.
**returns:** `FindKeepOptimalResult`. When `y` admits no first component
(constant, or orthogonal to `X` up to rounding), `keep_star` is `0` and
the score maps and `keep_grid` are empty, the same `0` convention as
`k_star` on `spls1_find_k_optimal`.

### 2b.3 K-selection at fixed keep

**function:** `spls1_find_k_optimal`
**need:** `pls1_find_k_optimal` with the inner fitter swapped to the
sparse fit at the caller's fixed `keep`. Same selectors, diagnostic, and
result shape as the dense version. `keep = n_features` reproduces the
dense function bit-exactly.
**arguments:** `X`, `y`, `k_max`, `keep`
**options:** `selector` (`"r2_se"` | `"r2_max"` | `"bic"`; default
`"r2_se"`); `diagnostic` (`"raw_perm"` | `"split_nb"` | `"split_exact"`
| `"e"` | `None`; default `None`); `args`; `pre_standardized`; `seed`;
`verbose`; `weights`.
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
**options:** `test_method` (`"raw_perm"` | `"split_nb"` | `"split_exact"`
| `"e"` | `"auto"`; default `"auto"`; `"raw_perm"` uses a fixed 5-fold CV at
every step and needs `n > 5`, else `invalid_argument`); `alpha`
(default `None`: the engine default, `0.05`); `args`;
`pre_standardized`; `seed`; `verbose`; `weights`.
**returns:** `FindKSequenceResult` (same type as `pls1_find_k_sequence`).

---

## 2c. PLS3 / PLSSVD

Symmetric X↔Y covariance analysis. Neither block is the outcome: the
question is which pattern of X covaries with which pattern of Y. Psychology
and neuroimaging call this PLSC.

### 2c.1 Fit

**function:** `pls3_fit`
**function:** `plssvd_fit`
**need:** SVD-PLS / PLSC — one SVD of the standardized `X'Y`, no deflation,
so all `k` components come out orthogonal from a single decomposition. The
SVD acts on a `p × q` matrix and never on a `p × p` one, so `p ≫ n` is the
ordinary case.
**arguments:** `X` (shape `(n, p)`), `Y` (shape `(n, q)`, must be 2-D),
`k` (`int`, default `1`, `k ≤ min(p, q)`)
**options:** `pre_standardized_X` (bool, default `False`);
`pre_standardized_Y` (bool, default `False`); `weights` (length-`n` vector —
**not implemented for this family**; anything other than `None` raises
`PlsKitError(code="invalid_argument")`). Zero rows raise
`invalid_argument`, as in `pls1_fit`; `k` is not bounded by `n`, so a
single row is not an error (it truncates to `k_used = 0` unless both
blocks are `pre_standardized_*`; see Truncation in
[fit and transform](../concepts/PLS3/fit-and-transform.md)). `spls3_fit`
follows the same rule.
**returns:** `PLS3Result` with `U`, `V`, `singular_values`, `x_scores`,
`y_scores`, the four standardization moment arrays, `k_used` and the two
`pre_standardized_*` echoes. `U` and `V` have a pinned sign (largest-|.|
entry of each `U` column positive, `V` flipped with it), so repeated fits
agree exactly.

### 2c.2 Transform

**function:** `pls3_transform`
**function:** `plssvd_transform`
**need:** project new data onto a fitted PLS3's LV scores. There is no
`pls3_predict` — PLS3 is symmetric, so there is nothing to predict.
**arguments:** `model`, `X_new` (optional), `Y_new` (optional)
**options:** `which` (`"x_scores"` | `"y_scores"` | `"both"`, default
`"both"`). A block `which` asks for must be supplied.
**returns:** `PLS3Scores`; a field is `None` exactly when `which` did not
ask for it.

### 2c.3 Sparse fit

**function:** `spls3_fit`
**need:** PLS3 with a hard keep-count on each side. `keep_Y < q` is the
reason it exists: it forces each latent dimension onto a few outcomes, so
the outcomes separate into groups.
**arguments:** `X`, `Y`, `k`, `keep_X`, `keep_Y`
**options:** `pre_standardized_X` / `pre_standardized_Y` (bool, default
`False`); `max_iter` (default `None`: the engine default, 100); `tol`
(default `None`: the engine default, 1e-8); `weights` (refused by this
family). Unless both keeps are at their full
dimension, `max_iter = 0` and a NaN, infinite or negative `tol` raise
`invalid_argument`; at the dense endpoint neither is read.
**returns:** `PLS3Result` with `keep_X`, `keep_Y`, `converged` and
`n_iter` populated. `keep_X = p` and `keep_Y = q` reproduce `pls3_fit`
bit for bit. The saliences are **not** orthogonal and `singular_values`
are not singular values of `X'Y`.

---

## 3. Inference

### 3.1 Confirmatory omnibus test

**Five test methods.** `raw_perm` / `split_nb` / `split_exact` are the
predictive-validity split-resampling family (`split_exact` is the same
held-out-correlation statistic as `split_nb`, calibrated by permutation
instead of the Fisher-z t approximation, so it holds its level on any
design; it is the recommended method at `k=1`);
`score` is closed-form on `T = ‖X′y‖²` (generalized χ² under Gaussian
y, anisotropy-aware by construction, K-free); `e` is universal
inference (split-LR e-value, calibration-free, non-asymptotic α bound
— `P(reject | H₀) ≤ α` exactly, with a power tax of ~30–50% vs.
`split_exact` as the validity cost). `score` and `split_exact` complement
rather than replace each other — `score` wins on diffuse signal across
many directions, `split_exact` wins on signal concentrated in a few
directions; reporting both side-by-side is the canonical pattern.

**`split_nb` auto-gate.** `split_nb`'s Fisher-z correction is exact at
ρ = ½ and drifts off level when `n_eff < 25` or the stable rank of the
standardized `X` is `< 3`. Since stable rank can never exceed the column
count, `X` with 4 columns or fewer is rerouted outright, without
consulting the computed rank. An explicit `split_nb` request on a design that trips
any of these reroutes to `split_exact` at `n_perm=1000`;
`result.test_method` reports `"split_exact"` and Python raises a
`UserWarning`. Pass `args={'force': True}` to run `split_nb` anyway.
Call `split_nb_gate` (§3.2) to ask the same question in advance.

**function:** `pls1_confirmatory_test`
**need:** omnibus null test at a pre-specified `k` ("is there signal
at K?"). Optionally runs an independent resampling pass for
rotation-invariant CIs.
**arguments:** `X`, `y`, `k` (default `1`)
**options:**

- `test_method` (`"raw_perm"` | `"split_nb"` | `"split_exact"` |
  `"score"` | `"e"` | `"auto"`; keyword-only, default `"auto"`, which picks
  `"split_exact"` or `"split_nb"` from `X`; see
  [`test_method="auto"`](../concepts/PLS1/inference.md#test_methodauto))
- `args` (dict of method-specific kwargs)
- `ci` (bool, default `False`) — when `True`, runs an independent
  resampling pass after the headline test and populates `result.ci`.
  Each of the `n_boot` replicates refits on a size-`m` subsample (for
  `holdout_corr` and β) and on an n-out-of-n bootstrap resample (for
  leverage).
- `n_boot` (int, default `None`: the engine default, `1000`; must be
  `≥ 100`); `m_rate` (float, default `None`: the engine default, `0.7`;
  `0.5 < m_rate < 0.95`); `level` (float, default `None`: the engine
  default, `0.95`; `0.5 ≤ level ≤ 0.99`); `max_skip_rate` (float,
  default `None`: the engine default, `0.01`); `max_failure_rate`
  (float, default `None`: the engine default, `0.01`). Every one of
  these five is inert when `ci=False`. When `ci=True`, `n_boot`,
  `m_rate` and `level` are recorded on `result.ci` as the resolved
  value, not `None`; `max_skip_rate` and `max_failure_rate` are not
  carried on any result field (they only gate the resampling loop).
- `pre_standardized`; `weights`; `seed`; `verbose`.

**args by method:**

- `"raw_perm"`: `n_perm`, `n_folds` (default `5`; must be `≥ 2` and
  `< n`: `n_folds >= n` is leave-one-out (an `n_folds` above `n` adds
  empty folds on top of the one-row folds, still leave-one-out), where
  every validation fold is a single row with zero spread, so the
  pooled CV R² is undefined; raises `invalid_argument`. `n/2 < n_folds
  < n` is not rejected: some folds hold a single row, which weakens
  power but keeps the permutation p-value valid since the same
  statistic is used on the nulls.)
- `"split_nb"`: `n_splits`, `force` (bool, default `False`; run `split_nb`
  even on a design the auto-gate flags)
- `"split_exact"`: `n_perm`, `n_splits`
- `"auto"`: `n_perm`, `n_splits` (no `force`)
- `"score"`: none (anisotropy handled internally by Welch–Satterthwaite)
- `"e"`: none

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
holdout correlation (`holdout_corr`, a `CIScalar`), a per-variable
subsampling z for β (`beta_sign_z` / `beta_sign_z_signed`), per-variable
bootstrap leverage CI (`leverage_ci_lower` / `leverage_ci_upper` /
`leverage_se`, bounds clamped to `[0, 1]`), and per-coordinate β CIs
(`beta_ci_lower` / `beta_ci_upper` / `beta_se`). The per-coordinate
`beta_ci_*` arrays are a regression-style diagnostic. `holdout_corr` is
the calibrated composite readout. `beta_sign_z` is `|β_ref[j]| / beta_se[j]`
with a subsampling SE corrected for sampling without replacement and
for the shrinkage of subsample fits, and `beta_ci_*` is built from the
same corrected replicates; at `k=1` it is roughly half-normal
for a variable whose population PLS coefficient is 0, so it can be read
against 1.96, without multiplicity correction. It is not calibrated at
`k ≥ 2`; `pls1_perm_null` (§3.3) gives permutation-calibrated
per-variable tests at any `k`. `leverage_ci_*` describes how much each
variable contributes to the fitted subspace; it brackets the expected
leverage of a size-`n` fit, which at small `n/D` sits below the
large-sample leverage of signal variables. See [results](results.md)
for field shapes, the leverage and z definitions with their simulated
calibration, and the shrinkage / multiplicity / standardization caveats
that apply to `beta_ci_*`.

### 3.2 Auto-gate query

**function:** `split_nb_gate`
**need:** find out whether the `split_nb` auto-gate (§3.1) flags your
design, before spending a test run to discover it from
`result.test_method`. Reports the engine's own decision — it evaluates the
one rule, it does not restate it.
**arguments:** `X`
**options:** `weights`.
**returns:** `SplitNbGateResult` — `fires` (would an explicit
`"split_nb"` request reroute?), plus the `stable_rank` and `n_eff` the rule read.
`y` is not an argument: only `X` and the weights enter the rule.

### 3.3 Permutation-null engine

**function:** `pls1_perm_null`
**need:** signed per-voxel z + (optional) full perm matrix, suitable
for downstream FWER correction (TFCE / max-stat / cluster-mass) at
fMRI / NIRS scale.
**arguments:** `X`, `y`, `k`
**options:** `n_perm` (int, default `None`: the engine default,
`1000`; must be `≥ 100`, recorded on `result.n_perm` as the resolved
value, not `None`); `return_perm_matrix`
(bool, default `False`); `pre_standardized`; `seed`;
`verbose`; `weights`.
**returns:** `PermNullResult`. Pair with
`pls1_confirmatory_test(test_method="split_exact")` as an omnibus gate before
spending the `n_perm` permutation budget.

### 3.4 Confirmatory PLS3 omnibus test

**function:** `pls3_confirmatory_test`
**need:** "is there a real X↔Y association at LV1?" — a test on the held-out
latent-variable correlation, calibrated by permutation or against a t
reference.
**arguments:** `X`, `Y`, `k` (must be `1`)
**options:** `test_method` (`"split_exact"` | `"split_nb"` | `"auto"`;
keyword-only, default `"auto"`, see
[`test_method="auto"`](../concepts/PLS1/inference.md#test_methodauto)); `args`
(`"split_exact"`: `{"n_perm": int, "n_splits": int}`, defaults `1000` / `50`;
`"split_nb"`: `{"n_splits": int, "force": bool}`, defaults `50` / `False`;
`"auto"`: `{"n_perm": int, "n_splits": int}`, defaults `1000` / `50`);
`pre_standardized_X`; `pre_standardized_Y`; `seed`; `verbose`. `pre_standardized_X` / `pre_standardized_Y` are accepted but have
no effect on either method: each training half is re-standardized with its
own moments regardless, and the flags exist only so the signature does not
change when a method that reads them lands.
**returns:** `ConfirmatoryTestResult` — the same object
`pls1_confirmatory_test` returns. `ci` is always `None` here and `n_eff`
equals `n`. `rho_hat` is populated for `split_nb` only (and only when the
test half has at least 4 rows); `stable_rank` is set on an explicit `split_nb`
request, and on an `"auto"` request that reached the stable-rank check
(p > 4, n_eff ≥ 250); `None` otherwise. `n_perm` is `None` for `split_nb`.

Concepts: [PLS3 inference](../concepts/PLS3/inference.md).

**The statistic.** Fit PLS3 on the training half for `(u1, v1)`, then take
`r = cor(X_te @ u1, Y_te @ v1)` on the held-out half. Fisher-z average
across the splits; the reported `statistic` is `tanh(z_bar)`, matching what
`split_nb` and `split_exact` report for PLS1. The p-value compares that
against a reference built by permuting the rows of Y against X, with the
splits drawn once and held fixed across all permutations.

**Sign indeterminacy costs nothing.** An SVD fixes `(u1, v1)` only up to a
simultaneous flip, and a flip negates both held-out score vectors at once,
so `r` is unchanged. No alignment step is needed.

**Which method.** `split_exact` is the recommendation: it is exact whenever
the rows are exchangeable under the null. `split_nb` compares the same statistic against a t reference
instead of permuting, costing `n_splits` fits in total rather than
`n_perm * n_splits`. Both sides of the correlation are estimated on the
training half, where PLS1 has an observed outcome on one side. The
per-split null still carries over: conditional on the training half the two
held-out score vectors are fixed linear combinations of independent
test-half rows, so under the null `r` on one split follows the ordinary
null correlation law. The between-split correction behind the p-value is
PLS1's Nadeau-Bengio heuristic, not derived for two blocks, so its transfer
is supported empirically only. Measured on Gaussian, heavy-tailed,
low-stable-rank and real two-block designs, `split_nb` came out
conservative, never anti-conservative.

**Clustered rows.** Neither method is valid when rows are clustered (e.g.
repeated scans per subject): random splits leak subjects across the halves,
and permuting Y rows individually breaks within-subject exchangeability.
Blocked splits and blocked permutation are not implemented.

**The `split_nb` auto-gate.** Identical to `pls1_confirmatory_test`'s and
applied to X only: a flagged design on an explicit `split_nb` request runs
`split_exact` instead (`result.test_method` says so, and Python warns), and `args={"force": True}`
overrides it. Y never enters the gate — `q` is small by construction in
PLSC, so a stable-rank floor on Y would flag almost every design. The gate
thresholds are the PLS1 ones and have not been re-derived for a two-block
design.

**The three methods that are not available.** `raw_perm` needs a
cross-validated R², which a method with no `predict` does not have. `score`
is not implemented: its symmetric analog is an RV-type test on `‖X'Y‖_F²`,
which tests a different estimand from the LV1 held-out correlation. `e`
needs a generative model that symmetric cross-decomposition does not
supply.

**Why only `k=1`.** Above LV1 neither the component ordering nor the
individual directions need survive to the test half when singular values
are close, and whether the statistic should then be per-component or
subspace-level is not settled.

---

## 4. Interpretive

### 4.1 Rotation

**function:** `rotate`
**need:** simple-structure rotation of `W`. Can be called on a fitted
result to stamp a `RotationSpec`, or on a bare `W` matrix.
**arguments:** `model_or_W` (a PLS1Result, or a 2-D `W` read like any
other array argument: an `np.ndarray`, a nested list, ...)
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
`PlsKitError(code="already_rotated")`; re-rotation is not supported.

### 4.2 Rotation-stability diagnostic

**function:** `pls1_rotation_stability`
**need:** standalone subsampling diagnostic: does rotating `W` make the
axes more or less replicable across resamples? Each resample's `W`
(unrotated, and rotated) is aligned to the full-data reference by a
signed permutation of its columns, and the squared residuals give an
axis variance `V_unrot` and `V_rot`. The headline is
`variance_ratio = V_rot / V_unrot` (a `CIScalar` with a paired-bootstrap
percentile CI), plus one ratio per axis in `variance_ratio_per_axis`.
A ratio below 1 means rotation made the axes more stable.
**arguments:** `X`, `y`, `k`
**options:** `rotation_method` (`"varimax"`; default `"varimax"`);
`rotation_args` (dict); `L` (loading basis; default identity);
`n_boot` (int, default `None`: the engine default, `1000`, `≥ 100`);
`m_rate` (float, default `None`: the engine default, `0.7`,
`0.5 < m_rate < 0.95`); `level` (float, default `None`: the engine
default, `0.95`, `0.5 ≤ level ≤ 0.99`); `pre_standardized`; `weights`;
`max_skip_rate` (float, default `None`: the engine default, `0.01`);
`seed`; `verbose`. `n_boot`, `m_rate` and
`level` are recorded on the result as the resolved value, not `None`.

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
- `PLS3Result`, `PLS3Scores`
- `ConfirmatoryTestResult` (with optional `ConfirmatoryCI`)
- `SplitNbGateResult`
- `FindKOptimalResult`, `FindKSequenceResult`
- `FindKeepOptimalResult` — returned by `spls1_find_keep_optimal`
- `PermNullResult`
- `RotateResult`, `RotationSpec`
- `RotationStabilityResult`
- `CIScalar` — the rotation-invariant CI primitive used throughout

## Errors

- `PlsKitError`: base error, with `code` for programmatic handling.
- `PlsKitInvalidWeights` (`code="invalid_weights"`): weights vector failed
  validation; `reason` is `"negative"`, `"all_zero"`,
  `"insufficient_effective_n"` or `"length_mismatch"` (the weights length
  differs from the row count of `X` or `y` / `Y`). A NaN or infinite
  weight raises `code="non_finite_input"` instead.
- `PlsKitResamplingDegenerate` (`code="resampling_degenerate"`): the
  resampling loop skipped more than `max_skip_rate` of its draws. Carries
  `skipped`, `total`, `skip_rate`, `threshold`.

**`invalid_args` vs. `invalid_argument`.** Two different codes, easy to
confuse:

- `invalid_args`: a method string (`method=`, `selector=`, `diagnostic=`,
  `test_method=`, `which=`) is unknown, or a method-specific args dict
  (`args=`, `rotation_args=`, `find_k_args=`) had an unknown key, a
  missing key, or a value of the wrong type or sign (a count that is
  negative or not a whole number, a non-number `tol`, a non-bool
  `force`). Raised when the extension parses the dict for the chosen
  `method`, and on the Python side for `find_k_args` on `pls1_fit`. A key
  set to `None` takes the engine default, like an absent key.
- `invalid_argument`: a top-level argument has a bad value (for example
  `k="optimal"` without `k_max`, a count such as `k`, `n_boot` or
  `n_perm` that is negative or not a whole number, `k = 0` or
  `k_max = 0` at every function that takes one ("k must be >= 1",
  "k_max must be >= 1"), a keep-count of `0`, an array argument numpy
  cannot read as real numbers (strings, ragged nested lists, complex
  values), a `seed` outside
  `[0, 2^64)`, a non-number `level`, a flag such as `ci` or `verbose`
  that is not a bool, a method name that is not a string, an `args` /
  `rotation_args` / `find_k_args` that is not a dict, an array with the
  wrong number of dimensions, a `model` that is not the result type the
  function takes, such as a `PLS3Result` passed to `pls1_predict` or
  `rotate`, or `weights` passed to a PLS3 function).
  `None` for an optional argument means its default.

**All codes.** `PlsKitError.code` is one of:

| `code` | Meaning |
|---|---|
| `dimension_mismatch` | `X` and `y` have different row counts |
| `k_exceeds_max` | requested `k` (or `k_max`) is above the maximum for this data; `k = 0` is `invalid_argument` |
| `non_finite_input` | an input contains NaN or infinity |
| `convergence_failure` | reserved; no current code path raises it |
| `invalid_argument` | bad value for a top-level argument (see above) |
| `invalid_args` | bad method-specific args dict (see above) |
| `invalid_input` | invalid input content (shape, finiteness, a `rotate` loadings matrix with no columns, ...) |
| `shape_mismatch` | two arrays have incompatible shapes |
| `rotation_method_not_implemented` | the requested rotation method does not exist in this version |
| `already_rotated` | `rotate` was called on a `PLS1Result` that already has a `rotation_spec` |
| `invalid_weights` | raised as `PlsKitInvalidWeights` |
| `resampling_degenerate` | raised as `PlsKitResamplingDegenerate` |
| `resample_failure_rate_exceeded` | the `ci=True` resampling pass had more failed replicates than `max_failure_rate` allows |
| `perm_null_degenerate` | more than half the permutation null fits failed |
| `internal` | a plskit bug; please report it |
| `sequence_no_rejection` | `pls1_fit(k="sequence")` found no component at `alpha` (raised by the engine: `FindKSequenceOutput::k_to_fit`) |
| `optimal_no_component` | `pls1_fit(k="optimal")` got `k_star=0`, no first component can be extracted (raised by the engine: `FindKOptimalOutput::k_to_fit`) |
