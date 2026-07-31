# Result objects (Python)

User-facing field shapes for each public Python result class. All fields
are immutable (`@dataclass(frozen=True)`), and field names match across
language wrappers.

## `PLS1Result` — what `pls1_fit` returns

| Python field | Rust core field | numpy type | Shape |
|---|---|---|---|
| `T` | `t_scores` | `np.ndarray` | `(n, K)` |
| `P` | `p_loadings` | `np.ndarray` | `(D, K)` |
| `W` | `w_star` | `np.ndarray` | `(D, K)` |
| `Q` | `q_loadings` | `np.ndarray` | `(K,)` |
| `coef` | `coef` | `np.ndarray` | `(D,)` |
| `beta` | `beta` | `np.ndarray` | `(D,)` |
| `intercept` | `intercept` | `float` | scalar |
| `k_used` | `k_used` | `int` | scalar |
| `pre_standardized` | `pre_standardized` | `bool` | scalar |
| `rotation_spec` | `rotation_spec` | `RotationSpec \| None` | — |
| `keep` | — | `int \| None` | `int` for `spls1_fit`; `None` for `pls1_fit` |

There is no `k_was_auto` flag and no `find_k_certificate` field. The
2026-04 confirmatory-vs-exploratory overhaul moved K-selection
diagnostics onto the K-selection result objects themselves, where
they originate. `rotation_spec` is `None` until `rotate(model, ...)`
stamps it onto a copy of the model.

## `FindKOptimalResult` — what `pls1_find_k_optimal` returns

| Field | Python type | When populated |
|---|---|---|
| `k_star` | `int` | always |
| `selector` | `str` (`"r2_se"` / `"r2_max"` / `"bic"`) | always |
| `cv_scores` | `dict[int, float] \| None` | `selector ∈ {r2_se, r2_max}` |
| `cv_scores_se` | `dict[int, float] \| None` | `selector="r2_se"` only |
| `bic_scores` | `dict[int, float] \| None` | `selector="bic"` |
| `pvalues` | `np.ndarray \| None` | `diagnostic` set |
| `diagnostic` | `str \| None` | `diagnostic` set |
| `seed` | `int` | always |

When `diagnostic=` is set, `pvalues` carries the per-component p-values
of a same-sample sequential test up to `k_star`, and `diagnostic`
echoes the method name. Selection and the diagnostic share the same
data, so the pvalues are a robustness check, not honest inference; a
fresh sample is required for a confirmatory claim. To get the
worst-case p-value along the path, compute `np.nanmax(pvalues)`.

## `FindKSequenceResult` — what `pls1_find_k_sequence` returns

| Field | Python type | When populated |
|---|---|---|
| `k_star` | `int` | always (0 if no component rejects at α) |
| `pvalues` | `np.ndarray` (k_max,) | always; trailing entries are `nan` if stop-early kicked in |
| `test_method` | `str` (`"raw_perm"` / `"split_nb"` / `"split_perm"` / `"e"`) | always |
| `alpha` | `float` | always |
| `seed` | `int` | always |

Closed testing on nested H is exact, so `pvalues[:k_star]` is an
honest FWER-controlled sequence. To get the path-max p-value
along the rejected chain, compute `np.nanmax(pvalues[:k_star])`.

## `FindKeepOptimalResult` — what `spls1_find_keep_optimal` returns

| Field | Python type | Notes |
|---|---|---|
| `keep_star` | `int` | sparsest `keep` within 1 SE of the best mean CV R² |
| `k` | `int` | the fixed component count the sweep ran at |
| `cv_scores` | `dict[int, float]` | keep → mean CV R² across folds |
| `cv_scores_se` | `dict[int, float]` | keep → SE of the CV R² |
| `keep_grid` | `list[int]` | the logged geometric grid swept (powers of two; endpoints 1 and n_features always included) |
| `seed` | `int` | always |
| `n_eff` | `float` | effective sample size (from weights; `nan` if unavailable) |

Selection criterion: the 1-SE rule on mean CV R² — `keep_star` is the
sparsest keep whose mean CV R² is within 1 SE of the maximum. Ties
broken toward sparser. The `keep_grid` field records exactly which
candidates were evaluated.

## `ConfirmatoryTestResult` — what `pls1_confirmatory_test` returns

| Field | Python type | Notes |
|---|---|---|
| `pvalue` | `float` | always |
| `statistic` | `float` | always |
| `method` | `str` | one of `"raw_perm"` / `"split_nb"` / `"split_perm_nr"` / `"split_perm"` / `"score"` / `"e"` |
| `k` | `int` | the K tested (echoed from the input) |
| `n_perm` | `int \| None` | not None for resampling-family methods, None for `score` / `e` |
| `n_splits` | `int \| None` | not None for `split_*` methods, None for `raw_perm` / `score` / `e` |
| `seed` | `int` | always |
| `rho_hat` | `float \| None` | `split_nb` only, and only when unweighted with a test half of at least 4 rows; `None` for every other method |
| `ci` | `ConfirmatoryCI \| None` | not None when called with `ci=True`; carries the rotation-invariant subsample CIs |

There is no `at` field (legacy concept dropped). There is no
`null_distribution` or `split_mean_r` slot — those internals were
dropped from the public surface; only the headline result and
(optionally) the `ConfirmatoryCI` bundle survive.

## `CIScalar` — scalar subsample CI

Centered-scaled subsampling CI for a scalar functional, plus its SD.

| Field | Python type | Notes |
|---|---|---|
| `point` | `float` | full-data point estimate of the functional |
| `lower` | `float` | CI lower bound at the requested `level` |
| `upper` | `float` | CI upper bound at the requested `level` |
| `sd` | `float` | subsampling SD of the functional |

## `ConfirmatoryCI` — what `pls1_confirmatory_test(ci=True)` adds

Rotation-invariant readouts only. Per-axis CIs are intentionally absent
(see [api](api.md) §2.4 rationale).

| Field | Python type | Shape | Description |
|---|---|---|---|
| `n_boot` | `int` | scalar | resolved subsample iterations |
| `m` | `int` | scalar | resolved subsample size, `m = ceil(n^m_rate)` |
| `m_rate` | `float` | scalar | echoed from the input |
| `level` | `float` | scalar | echoed from the input |
| `beta_sign_z` | `np.ndarray` | `(D,)` | per-variable folded sign-stability z; canonical for hypothesis tests |
| `beta_sign_z_signed` | `np.ndarray` | `(D,)` | per-variable signed = `sign(β_ref[j]) · |beta_sign_z[j]|`; descriptive directional map |
| `leverage_ci_lower` | `np.ndarray` | `(D,)` | per-variable subsampling CI lower bound on leverage |
| `leverage_ci_upper` | `np.ndarray` | `(D,)` | per-variable subsampling CI upper bound on leverage |
| `leverage_se` | `np.ndarray` | `(D,)` | per-variable subsampling SE of leverage |
| `beta_ci_lower` | `np.ndarray` | `(D,)` | per-coordinate centered-scaled CI on β; PLS1-only diagnostic — see caveats below |
| `beta_ci_upper` | `np.ndarray` | `(D,)` | per-coordinate |
| `beta_se` | `np.ndarray` | `(D,)` | `= √(m/n) · sd(β_b[j])` |
| `holdout_corr` | `CIScalar` | scalar | Fisher z-transformed NB-Wald CI on out-of-sample predictive correlation |
| `n_boot_finite` | `int` | scalar | resamples whose worker fit succeeded (≤ `n_boot`) |
| `n_boot_finite_holdout_corr` | `int` | scalar | subset whose holdout_corr is finite (≤ `n_boot_finite`); resamples with `|r_b| ≥ 1` (degenerate) are also excluded from the Fisher pool |

The `holdout_corr` CI is built on the variance-stabilized Fisher z-scale
(`ζ = atanh(r)`), with the same NB inflation factor `(1/B + (n−m)/m)`
applied to z-scale variance, then back-transformed via `tanh`. Bounds
are guaranteed to lie strictly in `(−1, 1)` and are **asymmetric** on the
r-scale. `point` is the subsample mean on the r-scale (textbook plug-in
estimate of ρ); `sd` is on the z-scale, so `point ± Φ⁻¹(1−α/2) · sd` does
**not** reconstruct the CI — read the bounds directly. To recover the
z-scale interval: `ci_z = (atanh(point) − Φ⁻¹(1−α/2)·sd, atanh(point) +
Φ⁻¹(1−α/2)·sd)`; the reported bounds equal `tanh(ci_z)`.

### Per-coordinate β CIs — diagnostic, with caveats

`beta_ci_lower / beta_ci_upper / beta_se` are a **regression-style
diagnostic for downstream pipelines** (one-line β_j ± SE tables in the
OLS reporting shape — psychometric reports, fMRI thresholding,
supplementary tables). They are not a primary inferential output;
canonical inference on `ConfirmatoryCI` is `beta_sign_z` (omnibus sign
reliability) plus `leverage_ci_*` (subspace importance) plus
`holdout_corr` (out-of-sample predictive). Math: same `reduce_centered_scaled`
reduction used for `leverage_ci_*`, applied per coordinate to β_b values
versus β_ref; β being unbounded means no transform is needed.

PLS1 only. In PLS2/PLSC the analogue β is a matrix that inherits W's
rotation/sign indeterminacy and requires procrustes alignment; per-β CIs
for those families are out of scope and will be specced separately.

Caveats — part of the contract, not optional commentary:

1. **PLS shrinkage bias on small m.** β_b on subsamples of size m is
   shrunk more aggressively than β_ref (PLS effective DoF scales
   sub-linearly in n; Krämer & Sugiyama 2011). The centered-scaled
   formula treats bias and sampling variance as sharing a √(n/m) rate.
   Well-calibrated in the easy regime (large n, low D, strong signal,
   mild `m_rate`); degrades as the regime gets hard — the bias can
   dominate and produce CIs whose midpoint sits well above or below
   the point estimate, particularly at aggressive `m_rate`, low SNR,
   or D approaching n. Treat as a directional sanity check otherwise.
2. **No multiple-comparison correction.** Per-coordinate CIs at level
   α are individual, not simultaneous. For omnibus claims about which
   coordinates are nonzero, use `beta_sign_z`. For coordinate-wise
   FWER control, apply Bonferroni / Westfall–Young externally.
3. **Theoretical caveat: PLS1 β is biased by Krylov shrinkage.**
   Asymptotic normality of β̂ is established only for K=1 in the
   high-dimensional regime (Basa, Cook, Forzani & Marcos 2024). For
   K ≥ 2 the per-coordinate centered-scaled CI is a useful diagnostic,
   not a calibrated inferential tool.
4. **Standardization mode matters.** Per-coordinate CIs are reported
   on the same scale as `β_ref`: raw X → raw y units when
   `pre_standardized=False` (`pls1_fit` back-projects internally),
   standardized scale when `pre_standardized=True`. The subsample
   worker matches that scale via the same back-projection using the
   subsample's own `y_scale / x_scale[j]`; subsample-vs-full-data
   stats differ slightly, but converge as `m` grows.
5. **Three-way distinction (sign-z ↔ leverage_ci ↔ beta_ci).** A
   coordinate with `|beta_sign_z[j]| ≫ 2` whose `beta_ci_*[j]` straddles
   zero is **not** a contradiction — sign-z reflects sign reliability
   across resamples, β CI reflects regression-coefficient magnitude,
   and `leverage_ci_*[j]` measures subspace contribution (always ≥ 0).
   A coordinate can have a confidently nonzero `leverage_ci` with a
   β CI through zero (the variable shapes the latent direction without
   contributing a stable regression coefficient on its own).

Memory: storing per-resample β adds `8 · D · n_boot_finite` bytes —
~8 MB at D=1000, B=1000 (trivial); ~400 MB at D=50_000, B=1000 (fMRI
scale). Brain-scale users typically reach for `pls1_perm_null` (sparse
z-map output) instead of the confirmatory CI bundle.

## `RotationStabilityResult` — what `pls1_rotation_stability` returns

| Field | Python type | Notes |
|---|---|---|
| `method` | `str` | the rotation method used (e.g. `"varimax"`) |
| `n_boot` | `int` | resolved subsample iterations |
| `m` | `int` | resolved subsample size |
| `m_rate` | `float` | echoed from the input |
| `level` | `float` | echoed from the input |
| `seed` | `int` | always |
| `agreement` | `CIScalar` | post-procrustes Frobenius CI; `agreement.point` is `0` by construction (full-data fit aligns to itself), so the CI width is the diagnostic |

## `RotationSpec` — stamped by `rotate(model, ...)`

| Field | Python type | Notes |
|---|---|---|
| `method` | `str` | `"varimax"` today; future `"promax"` / `"oblimin"` / `"geomin"` |
| `args` | `Mapping` (frozen via `MappingProxyType`) | method-specific kwargs used at rotate-time |
| `R` | `np.ndarray` `(K, K)` | rotation matrix; `W_rot = W @ R` |
| `sweeps` | `int` | varimax iterations to convergence |
| `V_converged` | `float` | final varimax criterion value |
| `L_was_provided` | `bool` | whether caller passed a loading basis |

`rotation_spec` is `None` until `rotate(model, ...)` stamps it on a
copy of the model. When the model flows back into a follow-up call
(e.g. `pls1_predict`, future `bootstrap_saliences`), the spec is
reconstructed verbatim — bit-exact, no rounding.
