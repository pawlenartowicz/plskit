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
  [Threads](../concepts/threads.md). To bound the memory of the resampling
  loops (default 2 GiB), set `PLSKIT_MEMORY_BUDGET`; see
  [Memory budget](../concepts/threads.md#memory-budget).
- **Method-axis dispatch** uses `(method, args)`: a `method` string +
  an `args` dict for method-specific kwargs. Cross-cutting kwargs
  (`seed`, `weights`, `pre_standardized`) live at the top level.
- Type hints use NumPy notation: `np.ndarray, shape (n, d)` is a 2-D
  array; `np.ndarray, shape (n,)` is a 1-D vector.
- **Arrays** are read as `float64` in their own memory layout. A C- or
  F-ordered `float64` `X` reaches the engine without a copy, and so does
  an F-ordered `float64` `Y`, `Y_new`, `W`, `L` or `gram` (a C-ordered one is
  copied column-major, which gives the same bits as F order); other
  dtypes are converted, and a strided view is copied by the extension.
  Results for two layouts of the same values agree to rounding (the
  corpus tolerance, `1e-10 + 1e-14·|value|`), not bit for bit, and a
  fit sitting on its truncation floor can stop at a different
  `k_used`. The last bits move with the layout in `pls1_fit`,
  `spls1_fit`, `pls1_predict`,
  `pls1_confirmatory_test` with `ci`, `pls1_find_k_optimal` and
  `spls1_find_k_optimal` with every selector, `spls1_find_keep_optimal`,
  the `selection` of `pls1_fit(k="optimal")` and, without
  `pre_standardized_X`, in `pls3_fit`, `plssvd_fit` and `spls3_fit`; the
  others currently give C and F order the same bits. In the selections the
  `cv_scores`, `cv_scores_se` and `bic_scores` move, and a near tie (two
  BIC scores, two CV scores at the best, or for `r2_se` a CV score at the
  one-SE threshold) or a fit at its truncation floor can select another
  `k_star` or `keep_star`. Where the first component
  sits at its truncation floor, the selections by cross-validation
  (`selector="r2_se"` or `"r2_max"`, and `spls1_find_keep_optimal`) can
  return `k_star` (or `keep_star`) 0 on one layout and not on the other.
  The same array in the same layout gives byte-identical results on every
  run and at every thread count. An array numpy cannot read as real
  numbers (strings, numeric strings included, ragged nested lists,
  complex values, non-numeric objects) raises
  `PlsKitError(code="invalid_argument")`; booleans (read as `1`/`0`) and
  integers are converted, and `None` or `pd.NA` in an object array reads
  as NaN (`non_finite_input`).
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
**arguments:** `X`, `y`, `k` (`int | "optimal" | "sequence"`, default `"optimal"`: choose `k` with `pls1_find_k_optimal`, selector `r2_se` unless `find_k_args` says otherwise, and fit at its `k_star`)
**options:** `k_max` (default `None`; with `k` a string, `None` means `min(9, p)`, lowered until the weights' Kish effective sample size and the covariates' residual degrees of freedom both stay at least `k_max + 1`, never below 1; an explicit `k_max` is not lowered by the effective sample size or the residual degrees of freedom; must be `None` when `k` is an int); `find_k_args`
(dict of method-specific kwargs forwarded to `pls1_find_k_optimal` /
`pls1_find_k_sequence`; allowed keys are the public params of the
target function except `seed` / `pre_standardized` / `weights` / `covariates`,
which live on `pls1_fit` itself, and `verbose`, which the selection
run does not take;
unknown keys raise `PlsKitError(code="invalid_args")`);
`pre_standardized` (bool, default `False`); `seed` (`int | None`, forwarded to the k-selection call when `k` is a string; without it, the default call included, the CV folds are drawn at random, so the chosen `k` can differ between runs, and `seed=model.selection_result.seed` repeats a run);
`weights` (length-`n` vector; default `None` = uniform);
`covariates` (`(n, p_Z)` matrix or length-`n` vector; default `None`; see
[Covariates](../concepts/PLS1/covariates.md)). There is no
convergence tolerance or iteration cap to set: each component of the
PLS1 (NIPALS) model comes from a single pass.
Rotation is post-fit; see `rotate`.
The columns of `W` are orthonormal only up to rounding, and the
rounding grows with `k` on wide `X`, as in any Krylov method without
reorthogonalization: on random data `max|WᵀW − I|` was about 5e-11 at
`n = 100, D = 300, k = 20`, 2e-4 at `n = 100, D = 800, k = 25` and 5e-7 at
`n = 300, D = 800, k = 40`. `coef`, `beta` and predictions are not
affected (relative error within 2e-14 of an extended-precision fit in
those cells). Reorthogonalizing would make `WᵀW = I` hold to 1e-15 but
would not bring `W` closer to the exact one: at `D = 800, k = 25` it is
off by 2e-5 to 4e-5 either way, because the late components are that
sensitive to rounding. Leverage
(`W(WᵀW)⁻¹Wᵀ`) does not assume `WᵀW = I`; anything that does, such as
`rotate(m.W, ...)`, inherits the drift.
**returns:** `PLS1Result`; under a string `k`, `selection_result` holds the selection. When `k="sequence"` and
`pls1_find_k_sequence` returns `k_star=0` (no component rejected at
`alpha`), raises `PlsKitError(code="sequence_no_rejection")`. Callers
that want to fit anyway must call `pls1_find_k_sequence` directly,
inspect the result, and pass an explicit `int` k. Likewise, when
`k="optimal"` (the default) and `pls1_find_k_optimal` returns `k_star=0` (`y` is
constant or orthogonal to `X`, so no first component exists; at `n >=
3`, since the CV selectors reject `n <= 2` first), raises
`PlsKitError(code="optimal_no_component")`; an explicit `int` k returns
the `k_used=0` zero model.

### 2.2 Predict

**function:** `pls1_predict`
**need:** apply a fitted PLS1 model to new `X` → ŷ.
**arguments:** `model`, `X_new`
**options:** `covariates` (the covariates of the new rows, `(m, p_Z)` or length `m`: required for a model fitted with covariates, refused otherwise; the prediction is `intercept + X_new·beta + Z_new·covariate_coef`)
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
`pre_standardized`; `weights`; `covariates` (see [Covariates](../concepts/PLS1/covariates.md); the selection runs on the adjusted data, and a `diagnostic` follows the `pls1_confirmatory_test` rules for covariates); `seed`; `verbose`.
The CV folds of `"r2_se"` / `"r2_max"` read `X` divided once by its column scales over all rows (as given under `pre_standardized`) and only centre each part; `y` is standardized in each fold.
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
**BIC caveat:** `selector="bic"` counts each component as one degree
of freedom. On wide `X` (p much larger than n) the effective degrees of
freedom of a PLS fit are much larger, so BIC tends to select `k_max`;
`r2_se` (the default) is recommended there.

**function:** `pls1_find_k_sequence`
**need:** sequential closed-test on nested hypotheses — "how many
components carry signal at α?" with exact FWER control.
**arguments:** `X`, `y`, `k_max`
**options:** `test_method` (`"raw_perm"` | `"split_nb"` |
`"split_exact"` | `"e"` | `"auto"`; default `"auto"`, see
[`test_method="auto"`](../concepts/PLS1/inference.md#test_methodauto)); `args` (dict of
method-specific kwargs: `n_perm`, `n_splits`, and `force` for
`split_nb`; `"auto"` takes `n_perm` and `n_splits`, not `force`); `alpha` (default `None`: the engine default,
`0.05`); `pre_standardized`; `weights`; `covariates` (see [Covariates](../concepts/PLS1/covariates.md)); `seed`;
`verbose`.
`X` and `y` are standardized once over all rows (used as given under `pre_standardized`) and each test only centres its parts; steps from 2 on test the deflated residual on the scale step 1 used.
`test_method="raw_perm"` uses a
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

Concepts: [sPLS1](../concepts/PLS1/sparse.md).

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
`weights` (length-`n` vector; default `None` = uniform); `covariates` (see [Covariates](../concepts/PLS1/covariates.md)).
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
`verbose`; `weights`; `covariates` (see [Covariates](../concepts/PLS1/covariates.md)). `n_folds` is capped at
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
`verbose`; `weights`; `covariates` (see [Covariates](../concepts/PLS1/covariates.md)).
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
`pre_standardized`; `seed`; `verbose`; `weights`; `covariates` (see [Covariates](../concepts/PLS1/covariates.md)).
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

**Six test methods.** `raw_perm` / `split_nb` / `split_exact` are the
predictive-validity split-resampling family (`split_exact` is the same
held-out-correlation statistic as `split_nb`, calibrated by permutation
instead of the Fisher-z t approximation, so it holds its level on any
design; it is the recommended method);
`score` is closed-form on `T = ‖X′y‖²` (generalized χ² under Gaussian
y, anisotropy-aware by construction, K-free); `e` is universal
inference (split-LR e-value, calibration-free, non-asymptotic α bound
— `P(reject | H₀) ≤ α` exactly, with a power tax of ~30–50% vs.
`split_exact` as the validity cost). `sv_perm` is a permutation test on `d₁ = ‖X̃ᵀỹ‖` at `k = 1`; `score` with
`args={"n_perm": ...}` runs the same permutation test and keeps `T = ‖X̃ᵀỹ‖²` as
its statistic. `score` and `split_exact` complement
rather than replace each other — `score` wins on diffuse signal across
many directions, `split_exact` wins on signal concentrated in a few
directions; reporting both side-by-side is the canonical pattern.

**`split_nb` auto-gate.** `split_nb`'s Fisher-z correction is exact at ρ = ½ and drifts off level when `n_eff < 25` or the stable rank of `X` as the tests read it (standardized once over all rows, or only centred under `pre_standardized`) is `< 3`. Since stable rank can never exceed the column count, `X` with 4 columns or fewer is rerouted outright, without consulting the computed rank. An explicit `split_nb` request on a design that trips any of these reroutes to `split_exact` at `n_perm=1000`; `result.test_method` reports `"split_exact"` and Python raises a `UserWarning`. Pass `args={'force': True}` to run `split_nb` anyway. Call `split_nb_gate` (§3.2) to ask the same question in advance.

**function:** `pls1_confirmatory_test`
**need:** omnibus null test over a set of `k` fixed before the test sees `y` ("is there signal at some `k` in the set?"), or at one `k`. Optionally runs an independent resampling pass for rotation-invariant CIs.
**arguments:** `X` (or `gram`, below), `y`, `k` (an `int`, or a set: a list, a `range` or an integer array; default `None`, the set `1..8` cut by the design, see **default `k`** below)
**options:**

- `test_method` (`"raw_perm"` | `"split_nb"` | `"split_exact"` |
  `"score"` | `"sv_perm"` | `"e"` | `"auto"`; keyword-only, default `"auto"`, which picks
  `"split_exact"` or `"split_nb"` from `X` at one `k` and runs `"split_exact"` over a set of more than one `k`; see
  [`test_method="auto"`](../concepts/PLS1/inference.md#test_methodauto))
- `args` (dict of method-specific kwargs)
- `ci` (bool, default `False`) — when `True`, runs an independent
  resampling pass after the headline test and populates `result.ci`.
  Each of the `n_boot` replicates refits on a size-`m` subsample (for
  β), on an n-out-of-n bootstrap resample (for leverage) and on a
  random half of the rows (for `holdout_corr`, scored on the other
  half). Refused (`invalid_argument`) when `m` or `⌊n/2⌋` is below `k + 2`, or below `k + 1 + rank([1, Z])` with covariates.
  `ci` is beta, not to be trusted yet. It scales X and y once over all rows, as the held-out tests do (not under `pre_standardized`). Its `holdout_corr` half split centres the training half with its own means and the held-out half with the training means, as the held-out tests do. The β and leverage resamples are centred with their own means too, except under `pre_standardized`, where they use X and y as given, as the reference fit does.
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
- `covariates` (`(n, p_Z)` matrix or length-`n` vector; see [Covariates](../concepts/PLS1/covariates.md)): `split_exact` and `raw_perm` take it, with Freedman–Lane null columns (approximately exact, not exact). `"auto"` runs `split_exact`. `split_nb`, `score` without `n_perm` and `e` raise `invalid_argument`; `score` with `n_perm` and `sv_perm` take it, with Freedman–Lane null columns. With `gram`, the call adjusts the Gram matrix itself and Z has one row per row of `gram`. The result records `n_covariates`.
- `null_y` (`(n, B)` matrix): your own null outcomes, on the scale of `y`, for `split_exact` only (the default `"auto"` is refused: pass `test_method="split_exact"`). The statistic of each column, on the same splits as `y`, makes the null distribution, and `p = (#{null ≥ observed} + 1) / (B + 1)` at one `k`; over a set the reported p is the min-p p over the set (per-k p in `pvalues_by_k`). An `n_perm` in `args` raises `invalid_args`; `covariates` and any other method raise `invalid_argument`. It works with `gram`.
- `gram` (keyword-only, in place of `X`; passing both raises `invalid_argument`): a `GramMatrixResult` from `gram_matrix` (§3.6), or an `(n, n)` array `X̃ @ X̃.T` with `n_features` (required with an array, refused with the result, which records it). Every method runs from it: the call centres it over its rows, so `result.subset(rows)` is accepted, and checks its entries (a non-negative diagonal, symmetry and the Cauchy-Schwarz bound; `invalid_input` otherwise). On `gram_matrix(X, pre_standardized=flag)` the result equals the call on `X` with `pre_standardized=flag` at the same seed, to rounding (except as §3.6 says for a pre-standardized `X` beyond `2**±256`); `"score"` reads the centred matrix, so it equals the call on `X` centred over all rows. `pre_standardized` is read from the result (`True` with it raises `invalid_argument`) and says how `X̃` was scaled with an array. `weights`, `ci=True` and `k` above 10 raise `invalid_argument` with `gram`.

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
- `"score"`: `n_perm` (default `None`: the closed form, anisotropy handled internally by Welch–Satterthwaite; an integer runs the permutation test of `sv_perm` and reports `T = ‖X̃ᵀỹ‖²`)
- `"sv_perm"`: `n_perm` (default `1000`), `rotate` (must stay `False`; `rotate=True` raises `invalid_argument`)
- `"e"`: none

**A set of `k`:** with more than one `k` the test asks whether `X` predicts `y` at some `k` in the set and returns one p-value, combined over the set by min-p (each column's smallest per-`k` permutation p, compared with the observed one; Westfall and Young 1993), which keeps the permutation p exact. A set is sorted; an empty set, a duplicate or a `k` of 0 raises `invalid_argument`, and a set past a limit of the method (`k ≤ d`, the split sizes, `k ≤ 10` with `gram`) is an error, never cut. `split_exact` and `raw_perm` take a set; `split_nb`, `score`, `sv_perm`, `e` and `ci=True` raise `invalid_argument` on a set of more than one `k`. `"auto"` with more than one `k` runs `split_exact`, which refits for every permutation up to the largest `k`: on data where `"auto"` at one `k` picks `split_nb`, the default call costs much more than `k=1, test_method="split_nb"`, the fast choice. See [A set of `k`](../concepts/PLS1/inference.md#a-set-of-k).

**default `k`:** `None` is the set `1..8`, cut at `min(8, d, n // 2 - 2)`, then lowered until the weights' Kish effective sample size (`n_eff ≥ k + 1`) and the covariates' residual degrees of freedom (`n_pos - rank([1, Z]) ≥ k + 1`, `n_pos` the rows with positive weight) allow it, never below 1. It is `[1]` for `split_nb`, `score`, `sv_perm`, `e` and `ci=True`.

**No `k` to fit at:** the result names no `k` to fit at. The documented pairing is `r = pls1_confirmatory_test(X, y)` for the claim, then `fit = pls1_fit(X, y)` for the model, which chooses its own `k` by cross-validation; pass no `k` from one to the other.

**Honest use:** `pls1_confirmatory_test` does not pre-validate that you didn't pick `k`, or the set, from the same data via `pls1_find_k_*`. Honest confirmatory inference means testing the default set or a `k` (or set) fixed from prior knowledge, or holding out a fresh sample.

**returns:** `ConfirmatoryTestResult`: `k` is the tested set (a sorted list), `pvalue` the p for the whole set, `statistic` `None` for a set of more than one `k`, and `pvalues_by_k` / `statistics_by_k` each `k`'s own p (unadjusted) and statistic. When `ci=True`, the `.ci` field
is a `ConfirmatoryCI` bundle: a Fisher z-transformed Wald CI on
holdout correlation (`holdout_corr`, a `CIScalar`), a per-variable
subsampling z for β (`beta_sign_z` / `beta_sign_z_signed`), per-variable
bootstrap leverage CI (`leverage_ci_lower` / `leverage_ci_upper` /
`leverage_se`, bounds clamped to `[0, 1]`), and per-coordinate β CIs
(`beta_ci_lower` / `beta_ci_upper` / `beta_se`). The per-coordinate
`beta_ci_*` arrays are a regression-style diagnostic. `holdout_corr` is
the composite readout (beta, see `ci` above). `beta_sign_z` is `|β_ref[j]| / beta_se[j]`
with a subsampling SE corrected for sampling without replacement and
for the shrinkage of subsample fits, and `beta_ci_*` is built from the
same corrected replicates; at `k=1` it is roughly half-normal
for a variable whose population PLS coefficient is 0, so it can be read
against 1.96, without multiplicity correction. It is not calibrated at
`k ≥ 2`; `pls1_perm_null` (§3.3) gives permutation-calibrated
per-variable tests at any `k` (at `k ≥ 2` conservative and low-powered
on collinear `X`). `leverage_ci_*` describes how much each
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
**options:** `weights`; `pre_standardized` (default `False`: `X` is
standardized once over all rows; with it, `X` is read as given and only
centred, as the tests read it under the same flag).
**returns:** `SplitNbGateResult` — `fires` (would an explicit
`"split_nb"` request with the same `pre_standardized` reroute?), plus the
`stable_rank` and `n_eff` the rule read. `stable_rank` is NaN when the Gram
matrix of `X` so read is not finite. The gate then fires.
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
`verbose`; `weights`; `covariates` (null columns are Freedman–Lane residuals and the β maps are computed on the adjusted X; see [Covariates](../concepts/PLS1/covariates.md)).
At `k = 1` the per-variable null is that of `X_jᵀy` (see
[results](results.md), `PermNullResult`); at `k ≥ 2` it is a refit on
permuted `y`, which keeps its level but loses power on collinear `X`.
**returns:** `PermNullResult`. Pair with
`pls1_confirmatory_test(test_method="split_exact")` as an omnibus gate before
spending the `n_perm` permutation budget.
There is no `gram`: the per-feature outputs need `X`. A caller who holds only the Gram matrix gets the dual coefficients of the fit from `pls1_perm_alpha` (§3.5).

### 3.4 Confirmatory PLS3 omnibus test

**function:** `pls3_confirmatory_test`
**need:** "is there a real X↔Y association at LV1?" — a test on the held-out
latent-variable correlation, calibrated by permutation or against a t
reference.
**arguments:** `X` (or `gram`, below), `Y`, `k` (must be `1`)
**options:** `test_method` (`"split_exact"` | `"split_nb"` | `"sv_perm"` | `"auto"`;
keyword-only, default `"auto"`, which runs `"split_exact"` on every
design and never `"sv_perm"`, see
[`test_method="auto"`](../concepts/PLS1/inference.md#test_methodauto)); `args`
(`"split_exact"`: `{"n_perm": int, "n_splits": int}`, defaults `1000` / `32`;
`"split_nb"`: `{"n_splits": int, "force": bool}`, defaults `32` / `False`;
`"sv_perm"`: `{"n_perm": int, "rotate": bool}`, defaults `1000` / `False`;
`"auto"`: `{"n_perm": int, "n_splits": int}`, defaults `1000` / `32`);
`pre_standardized_X`; `pre_standardized_Y`; `seed`; `verbose`. Each block is
standardized once over all rows and each split only centres its halves with
the training means; `pre_standardized_X` / `pre_standardized_Y` skip that one
standardization for their block, which is then used as given.
`gram` and `n_features` as in §3.1, in place of `X`; `pre_standardized_X` is read from the result. Both methods run on the Gram route, so the result equals the call on `X` to rounding, except at the truncation floor, where the call on `X` can take the primal route.
**returns:** `ConfirmatoryTestResult` — the same object
`pls1_confirmatory_test` returns. `ci` is always `None` here and `n_eff`
equals `n`. `rho_hat` is populated for `split_nb` only (and only when the
test half has at least 4 rows); `stable_rank` is set on an explicit `split_nb`
request and `None` otherwise. `n_perm` is `None` for `split_nb`.

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
per-split null still carries over: conditional on X, the training half and
the all-row scales, one held-out score vector is fixed and the rows of the
other are exchangeable, so under the null `r` on one split follows the
ordinary null correlation law to the same approximation as in PLS1. With
both `pre_standardized_X` and `pre_standardized_Y` set the statement is
exact. The between-split correction behind the p-value is
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

**`sv_perm`.** A permutation test on `d₁`, the largest singular value of `X̃ᵀỸ`, in-sample, at `k=1` only. The rows of Y are shuffled against X `n_perm` times and `p = (1 + #{null ≥ observed}) / (n_perm + 1)`. `rotate=True` Procrustes-rotates each permuted singular vector set to the observed one before its singular value is taken (McIntosh and Lobaugh 2004); `rotate` is a PLS3 option, and `result.rotate` records which one ran. On the same permutations, rotation can only lower the LV1 p-value, so a rotated p is not exact. `covariates` and `keep_x` / `keep_y` below full are refused. plskit's `d₁` is `n` times the first singular value pyls reports. See [PLS3 inference](../concepts/PLS3/inference.md#sv_perm).

**The three methods that are not available.** `raw_perm` needs a
cross-validated R², which a method with no `predict` does not have. `score`
is not implemented: its symmetric analog is an RV-type test on `‖X'Y‖_F²`,
which tests a different estimand from the LV1 held-out correlation.
`e` needs a generative model that symmetric cross-decomposition does not
supply.

**Why only `k=1`.** Above LV1 neither the component ordering nor the
individual directions need survive to the test half when singular values
are close, and whether the statistic should then be per-component or
subspace-level is not settled.

### 3.5 Dual coefficients from a Gram matrix

**function:** `pls1_perm_alpha`
**need:** PLS1 fits of many outcomes against one standardized block `X̃`
that the caller holds and plskit never sees, such as a permutation null
whose per-feature maps the caller computes itself. For each outcome the
n-vector `alpha` with `coef = X̃.T @ alpha`, where `coef` is
`PLS1Result.coef` at `k` components.
**arguments:** `gram` (a `GramMatrixResult` from `gram_matrix` (§3.6), or an array `X̃ @ X̃.T` of shape `(n, n)`), `Y` (shape `(n, m)`, one raw outcome per column, each standardized inside), `k` (default `1`; `1` to `10`)
**options:** `n_features` (int, keyword-only; in R pass it by name, `n_features = p`, since the third positional argument is `k`): the number of columns of `X̃` summed into `gram`, constant columns included. Required with an array, refused with a `GramMatrixResult`, which records it. There is no `pre_standardized`: the fit does not read how `X̃` was scaled. There is no `seed`: nothing is random.
**returns:** `PermAlphaResult`: `alpha` `(n, m)`, `k_used` `(m,)`, `k`, `n_features`.
`k = 0`, `k` above 10, `n_features = 0` or above `2^53`, and a 1-D `Y` raise `invalid_argument`; a non-square `gram` and a `Y` whose row count differs from an array `gram`'s raise `shape_mismatch` (with a `GramMatrixResult` the row count raises `invalid_argument`, as in the tests); `n <= 1` raises `k_exceeds_max`; NaN or infinity in `gram` or `Y` raises `non_finite_input`; a `gram` that fails the checks below raises `invalid_input`.

**The result object.** A `GramMatrixResult` (or its `subset(rows)`) is centred over its rows first and then gets only the first three checks below (diagonal, symmetry, Cauchy-Schwarz), so a subset, whose rows do not sum to zero, and a matrix built under `pre_standardized`, whose trace can exceed `n * n_features`, are accepted. `X̃` is then the rows of `gram_matrix`'s `X̃` behind `gram`, centred by their own means. The contract and checks below are those of an array `gram`, which is not centred.

**The contract.** `gram` is the Gram matrix of the same `X̃` the caller
multiplies `alpha` by. The columns of `X̃` are centred and scaled as
`preprocess` scales them (population standard deviation, so each column has
sum of squares `n`; a constant or zeroed column is zero to rounding and
still counts in `n_features`, and a smaller count passes the checks
whenever the columns it leaves out are zero). `gram` is
computed in float64, each entry within the rounding of one
length-`n_features` dot product; sums over column blocks, and one final
scaling such as `n * D @ D.T` for a store of unit-norm columns `D`, stay
within it. Then `X̃.T @ alpha[:, j]` equals `pls1_fit(X, Y[:, j], k=1).coef`
to rounding. For a store `D` passed as `n * D @ D.T`, `X̃ = sqrt(n) * D`, so
the coefficient is `sqrt(n) * D.T @ alpha[:, j]`.

**The checks.** `gram` must have a non-negative diagonal (checked
exactly), be symmetric, have every off-diagonal entry within the
Cauchy-Schwarz bound `|gram[i, j]| <= sqrt(gram[i, i] * gram[j, j])`, have
`trace(gram) <= n * n_features`, and have row sums near zero, the last four
within four times the worst-case rounding the contract allows; otherwise
`invalid_input`. A zero diagonal entry therefore admits only exact zeros in
its row and column. The centring tolerance admits raw columns whose mean lies
up to a million standard deviations from zero. The trace check catches a
raw `X @ X.T` only when the raw columns' mean square, averaged over
columns, exceeds 1. The checks do not catch a `gram` of another block; a
block on a smaller common scale (sample instead of population standard
deviation, which R's `scale()` and pandas' `.std()` compute with `n - 1`;
unit-norm columns without the factor `n`; `gram / n` or
`gram / n_features`); columns scaled unequally whose sums of squares add up
to at most `n * n_features`; an `n_features` that leaves out only zero
columns, which passes the trace check exactly and lowers the relative
truncation floor; or an `n_features` above the true count (up to `2^53`),
which passes and raises it. On a block of
another common scale `X̃.T @ alpha` is the PLS1 coefficient of the caller's
scaling; on unequally scaled columns it is the PLS1 fit of a differently
weighted block. On a `gram` scaled by a common factor below 1, the
absolute truncation floor (`1e-14` on `‖X̃ᵀz‖`) can stop a fit that the
correctly scaled `gram` keeps.

**Truncation.** Every column is fitted from `gram` alone. The fit stops at the first component whose `‖X̃ᵀz‖`, computed from `gram`, falls under `pls1_fit`'s floors, and `k_used[j]` is then below `k`. `gram` resolves `‖X̃ᵀz‖` only to its own rounding, about `sqrt(n_features * eps)` relative to `‖X̃‖·‖z‖`, far above those floors, so near them (an outcome orthogonal to `X̃` up to rounding) the component count can differ from that of `pls1_fit` on `X`, and `alpha[:, j]` from its coefficient. An outcome that standardizes to exact zeros (all zeros, or a constant whose computed mean is exact) is the zero model: `alpha[:, j]` is `+0.0` and `k_used[j]` is `0`. A `gram` whose entries are all exactly zero (every column of `X̃` zero) gives the zero model for every outcome.

**Reproducibility.** Column `j` depends only on `gram`, `Y[:, j]`, `k` and
`n_features`: the same bits whatever the other columns of `Y`, the thread
count and the array layouts. Across machines results agree to rounding,
not bit for bit.

### 3.6 The held-out Gram matrix

**function:** `gram_matrix`
**need:** the all-row Gram matrix of `X` that the held-out tests compute, once, to reuse it across calls on the same `X` (several outcomes, or row subsamples).
**arguments:** `X` (at least 2 rows)
**options:** `pre_standardized` (default `False`: `X` is centred over all rows and divided by its column standard deviations, population, as `preprocess` scales them; with it, `X` is read as given and only centred, as the tests read it under the same flag).
**returns:** `GramMatrixResult`: `gram` `(n, n)` (`X̃ @ X̃.T` for that `X̃`, the bits the held-out tests compute from `X` without weights), `n_features` (the column count of `X`) and `pre_standardized`. `result.subset(rows)` returns a `GramMatrixResult` holding `gram[rows][:, rows]` with the same `n_features` and `pre_standardized`: the Gram matrix of those rows of `X̃`, which keep the all-row centring and scale. Either is passed as `gram=` to `pls1_confirmatory_test` (§3.1), `pls3_confirmatory_test` (§3.4) and `pls1_perm_alpha` (§3.5) in place of `X`. `rows` are row indices in the order given (0-based in Python; R and Julia take 1-based row numbers), and a repeated row is repeated.
Compute it once and pass it to every call on the same `X`, or take a row subsample from it:

```python
import numpy as np
import plskit

g = plskit.gram_matrix(X)                                   # once per X
tests = [plskit.pls1_confirmatory_test(gram=g, y=y, seed=1) for y in outcomes]
pls3 = plskit.pls3_confirmatory_test(gram=g, Y=Y, seed=1)
alpha = plskit.pls1_perm_alpha(g, Y_perm, k=2)

rows = np.arange(0, X.shape[0], 2)                          # every other row, 0-based
half = plskit.pls1_confirmatory_test(gram=g.subset(rows), y=y[rows], seed=1)
```

In R and Julia `rows` are 1-based:

```r
g <- gram_matrix(X)
test <- pls1_confirmatory_test(gram = g, y = y)
rows <- seq(1, nrow(X), by = 2)
half <- pls1_confirmatory_test(gram = subset(g, rows), y = y[rows])
```

```julia
g = gram_matrix(X)
test = pls1_confirmatory_test(; gram=g, y=y)
rows = 1:2:size(X, 1)
half = pls1_confirmatory_test(; gram=PLSKit.subset(g, rows), y=y[rows])
```

Fewer than 2 rows of `X` raise `invalid_argument`; NaN or infinity in `X` raises `non_finite_input`; `X` with no columns raises `invalid_input`. `subset` raises `invalid_argument` when `rows` is not a 1-D sequence of integer row indices of `gram`, or holds fewer than 2 entries. There is no `weights`.

Under `pre_standardized`, an `X` whose largest entry is outside `2**-256 .. 2**256` gives the Gram matrix of `X` divided by a power of two `c` near that entry, and the result does not keep `c`: p-values computed from it are those of `X`, but the `"score"` statistic of `pls1_confirmatory_test` and the `alpha` of `pls1_perm_alpha` are those of `X / c`.

**function:** `gram_matrix_blocks`
**need:** the `gram_matrix` of an `X` too wide to hold at once, given as column blocks (for example read from disk one block at a time).
**arguments:** `blocks` (an iterable of matrices, each with the same number of rows; in R a function returning the next matrix, or `NULL` when there are none left)
**options:** `pre_standardized` (as in `gram_matrix`, applied to each block).
**returns:** the `GramMatrixResult` of `gram_matrix` on the blocks side by side, with `n_features` the sum of the blocks' column counts. It is the sum of the blocks' Gram matrices, computed one block at a time, so only one block is in memory at once.

```python
X = np.load("X.npy", mmap_mode="r")                         # n rows, many columns
cuts = [*range(0, X.shape[1], 5000), X.shape[1]]
blocks = (X[:, a:b] for a, b in zip(cuts, cuts[1:]))        # a generator: one block at a time
g = plskit.gram_matrix_blocks(blocks)
```

No blocks raises `invalid_input`; a block with a different row count from the first raises `shape_mismatch`; any other problem in a block is raised as `gram_matrix` raises it. The result equals `gram_matrix` on the joined `X` to rounding, not bit for bit. Under `pre_standardized` each block is scaled on its own, so an `X` beyond `2**-256 .. 2**256` must use `gram_matrix` on the whole `X`.

---

## 4. Interpretive

### 4.1 Rotation

**function:** `rotate`
**need:** rotate a `FactorsResult`, a `PLS3Result` or a bare matrix toward simple structure. See [Profiles](../concepts/PLS1/profiles.md) for factors and [PLS3 fit and transform](../concepts/PLS3/fit-and-transform.md#after-rotate) for a `PLS3Result`.

| Argument | Type | Meaning |
|---|---|---|
| `model_or_W` | `FactorsResult`, `PLS3Result` or matrix | what gets rotated |
| `X` | matrix, optional | the data that gives the target `L = Xs W`. For a `FactorsResult` it must be the rows the factors came from. For a `PLS3Result` it selects the fit's stored `x_scores` as the target |
| `covariates` | matrix or vector, optional | removed from the target: `M_Z Xs W`. Required with `X` for a `FactorsResult` made with covariates, refused for one made without them and for a `PLS3Result`, optional for a matrix. Refused without `X` |
| `method` | `"varimax"` | the only method; default `"varimax"` |
| `args` | dict | varimax settings: `max_iter` (default `50`), `tol` (default `1e-8`), `kaiser_normalize` (default `True`) |
| `L` | matrix, optional | a target of your own |
| `pre_standardized` | bool or `None` | how `X` is standardized. `None` follows the `FactorsResult`, or is `False` for a matrix. Refused for a `PLS3Result` |

With neither `X` nor `L`, the target is the weights themselves (`W`, or `U` of a `PLS3Result`).

**returns:** `RotateResult` for a matrix. For a `FactorsResult` or a `PLS3Result`, a new result of the same type with `rotation_spec` set. See [results](results.md).

**refused:** `X` together with `L`, `pre_standardized` without `X`, and `pre_standardized` with a `PLS3Result` raise `invalid_argument`. A result that already has a `rotation_spec` raises `already_rotated`. A `PLS1Result` raises `invalid_argument`: use `profiles`, or `rotate(m.W, ...)` for the weights alone. `X` whose column count is not the row count of `W` raises `shape_mismatch`, and `X` with other rows than a `PLS3Result` was fit on raises `invalid_input`. A sparse `PLS3Result` raises a `UserWarning`.

### 4.2 PCA basis

**function:** `pca_basis`
**need:** the right singular vectors of standardized `X`, as the basis that `factors` splits a coefficient vector in.
**arguments:** `X`
**options:** `pre_standardized` (default `False`; if `True`, `X` is used as given).

It keeps the components with singular value `s > 1e-10 · s[0]`, so `V` has one column per unit of rank of `X`. Column signs are arbitrary, and `factors` does not depend on them. Non-finite values in `X`, every column of `X` constant (unless `pre_standardized`), or all singular values zero raise `invalid_input`.

**returns:** `PCABasisResult`. See [results](results.md).

### 4.3 Factors

**function:** `factors`
**need:** split one PLS1 coefficient vector into `n` orthogonal parts that add up to it. See [Profiles](../concepts/PLS1/profiles.md).

| Argument | Type | Meaning |
|---|---|---|
| `model_or_coef` | `PLS1Result` or vector `(D,)` | a fit from `pls1_fit` or `spls1_fit`, or `model.coef`. Not `model.beta` |
| `X` | matrix `(N, D)` | the data the fit came from |
| `n` | int | number of factors, from `1` to `K`, the rank of `X` after the variables with a zero `coef` are left out, plus one when `coef` has a part outside its row space (see below) |
| `covariates` | matrix or vector, optional | removed from `X` (standardized without weights, or used as given under `pre_standardized`, then least-squares residuals on `[1, Z]`) before the basis is taken. Required for a fit with covariates, refused for a fit without them, optional for a vector |
| `pre_standardized` | bool or `None` | `True` uses `X` as given, `False` standardizes it. `None` follows the fit, or is `False` for a vector |

```python
m = plskit.pls1_fit(X, y, k=3)
f = plskit.factors(m, X, n=3)
f = plskit.rotate(f, X)
f.F.sum(axis=1)                  # equals m.coef
```

When `coef` has a part outside the row space of the standardized kept columns of `X` (for example a sparse fit with more kept columns than `N − 1`), `factors` appends that part, normalized, as one extra last direction of the basis, so it lands in the last factor.

**returns:** `FactorsResult`, not rotated, in block order. See [results](results.md).

**refused:** these raise `invalid_input`: non-finite values in `X` or `coef`; every kept column of `X` constant (unless `pre_standardized`); `n` outside `[1, K]`; a `coef` that is all zero; a `PLS1Result` with `weights`, with no component, or fitted on a different number of rows than `X` has; a block with no signal, where the message names the block's components. A `PLS3Result` raises `invalid_argument`. A `coef` whose length differs from the columns of `X` raises `shape_mismatch`.

### 4.4 Choosing n

**function:** `profiles_find_n`
**need:** choose `n` by repeated cross-validation. See [Profiles](../concepts/PLS1/profiles.md#choosing-n).

| Argument | Type | Meaning |
|---|---|---|
| `X`, `y` | matrix, vector | the data |
| `k` | int | PLS components used in every fold |
| `n_max` | int, default `9` | upper limit of the search, not of `n` |
| `args` | dict | `n_folds` (default `5`), `n_repeats` (default `2`), `n_perm` (default `20`, shuffles for the parallel analysis) |
| `pre_standardized` | bool, default `False` | use `X` as given |
| `seed` | int or `None` | random seed |

Parallel analysis on standardized `X` gives `K_star`, the number of leading eigenvalues above the 95th percentile of the shuffled ones. In each fold the leading `K_star` axes are rotated with varimax, ordered by their share of `coef`, and `y` is regressed on the top `j` of them for `j` from 1 to `min(K_star, n_max)`. `n_star` is the smallest `j` whose mean held-out R² is within 2 SE of the best, or `0` when the best is not above 0 by 2 SE.

**returns:** `ProfilesFindNResult`. See [results](results.md).

**refused:** fewer than two usable folds, or `n_max` below 1, raises `invalid_input`. Folds with a constant training column, or with no finite held-out R², are not used and are counted in `n_failed_folds`.

### 4.5 Profiles

**function:** `profiles`
**need:** `factors`, then `rotate(f, X)`, then the scores `Xs F`, in one call. See [Profiles](../concepts/PLS1/profiles.md).

| Argument | Type | Meaning |
|---|---|---|
| `model_or_coef`, `X`, `n`, `pre_standardized` | | as in `factors` |
| `rotation` | `"varimax"` | passed to `rotate` as `method`. Any other value raises `rotation_method_not_implemented` |
| `args` | dict | passed to `rotate` as `args` |

```python
r = plskit.profiles(m, X, n=3)
r.factors.F.sum(axis=1)          # equals m.coef
```

**returns:** `ProfilesResult`. With a vector, `model` is `None`. See [results](results.md).

**refused:** as `factors`, and a model fitted with covariates (`profiles` has no `covariates` argument: use `factors` and `rotate`).

**function:** `profiles_search`
**need:** choose `n` with `profiles_find_n`, run `profiles` at that `n`, and keep the search.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `PLS1Result` | a dense, unweighted fit, not a vector |
| `X`, `y` | matrix, vector | the data of the fit |
| `n_max` | int, default `9` | as in `profiles_find_n` |
| `rotation`, `args` | | as in `profiles` |
| `find_n_args` | dict | passed to `profiles_find_n` as `args` |
| `seed` | int or `None` | random seed, kept on the result |

The search runs at `k = model.k_used`. When it returns `n_star = 0`, `profiles` runs with `n = 1` and `n_search.n_star` stays `0`.

```python
r = plskit.profiles_search(m, X, y)
r.n_search.r2_mean, r.n_search.r2_se, r.n_search.n_star
```

**returns:** `ProfilesSearchResult`. See [results](results.md).

**refused:** a sparse fit (one with `keep`) raises `invalid_input`; a model fitted with covariates raises `invalid_argument`; otherwise as `factors`.

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
- `PermAlphaResult`
- `GramMatrixResult` (with the method `subset(rows)`)
- `RotateResult`, `RotationSpec`
- `PCABasisResult`
- `FactorsResult`
- `ProfilesFindNResult`
- `ProfilesResult`, `ProfilesSearchResult`
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
  (`args=`, `find_k_args=`, `find_n_args=`) had an unknown key, a
  missing key, or a value of the wrong type or sign (a count that is
  negative or not a whole number, a non-number `tol`, a non-bool
  `force`). Raised when the extension parses the dict for the chosen
  `method`. A key set to `None` takes the engine default, like an absent
  key.
- `invalid_argument`: a top-level argument has a bad value (for example
  `k="optimal"` without `k_max`, a count such as `k`, `n_boot` or
  `n_perm` that is negative or not a whole number, `k = 0` or
  `k_max = 0` at every function that takes one ("k must be >= 1",
  "k_max must be >= 1"), a keep-count of `0`, an array argument numpy
  cannot read as real numbers (strings, ragged nested lists, complex
  values), a `seed` outside
  `[0, 2^64)`, a non-number `level`, a flag such as `ci` or `verbose`
  that is not a bool, a method name that is not a string, an `args` /
  `find_k_args` / `find_n_args` that is not a dict, an array with the
  wrong number of dimensions, a `model` that is not the result type the
  function takes, such as a `PLS3Result` passed to `pls1_predict`,
  a `PLS1Result` passed to `rotate`, or `weights` passed to a PLS3 function).
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
| `already_rotated` | `rotate` was called on a `PLS3Result` or `FactorsResult` that already has a `rotation_spec` |
| `invalid_weights` | raised as `PlsKitInvalidWeights` |
| `resampling_degenerate` | raised as `PlsKitResamplingDegenerate` |
| `resample_failure_rate_exceeded` | the `ci=True` resampling pass had more failed replicates than `max_failure_rate` allows |
| `perm_null_degenerate` | more than half the permutation null fits failed |
| `insufficient_residual_df` | with `covariates`, `n - rank([1, Z])` is below `k + 1`, with `n` the rows with positive weight: too few rows are left after the covariates are removed |
| `internal` | a plskit bug; please report it |
| `sequence_no_rejection` | `pls1_fit(k="sequence")` found no component at `alpha` (raised by the engine: `FindKSequenceOutput::k_to_fit`) |
| `optimal_no_component` | `pls1_fit(k="optimal")` got `k_star=0`, no first component can be extracted (raised by the engine: `FindKOptimalOutput::k_to_fit`) |
