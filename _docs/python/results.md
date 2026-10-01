# Result objects (Python)

User-facing field shapes for each public Python result class. All fields
are immutable (`@dataclass(frozen=True)`), and field names match across
language wrappers.

## `PreprocessResult` — what `preprocess` returns

| Field | Python type | Notes |
|---|---|---|
| `X_std` | `np.ndarray \| None` | standardized `X`; populated when `X` was passed |
| `X_mean` | `np.ndarray \| None` | per-column mean of `X`; populated when `X` was passed |
| `X_scale` | `np.ndarray \| None` | per-column scale of `X`; populated when `X` was passed |
| `Y_std` | `np.ndarray \| None` | standardized `Y`; shape matches the input (1-D or 2-D); populated when `Y` was passed |
| `Y_mean` | `float \| np.ndarray \| None` | scalar for 1-D `Y`, one entry per column for 2-D `Y`; populated when `Y` was passed |
| `Y_scale` | `float \| np.ndarray \| None` | scalar for 1-D `Y`, one entry per column for 2-D `Y`; populated when `Y` was passed |
| `weights_normalized` | `np.ndarray \| None` | weights normalized to mean 1; populated when `weights` was passed |
| `n_eff` | `float \| None` | Kish's effective sample size; populated when `weights` was passed; exactly `n` when the weights are all equal, as every fit reports |

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
| `k_used` | `k_used` | `int` | scalar; `< k` when a component's `‖X_a'y_a‖` or `t't` fell below `1e-14`, or `‖X_a'y_a‖` fell below `max(n, D)·ε·‖X‖_F·‖y‖`, with `X`, `y` the inputs the PLS1 kernel runs on (standardized unless `pre_standardized`, rows scaled by `√w`): `y` is then exhausted and what is left of `X_a'y_a` is rounding noise. `0` when this happens at the first component (`y` constant, or orthogonal to `X` up to rounding): the zero model, `coef = beta = 0` and `intercept` the (weighted) mean of `y` |
| `pre_standardized` | `pre_standardized` | `bool` | scalar |
| `weights` | `weights` | `np.ndarray \| None` | `(n,)` when present; `None` for uniform/absent weights |
| `n_eff` | `n_eff` | `float` | scalar; Kish's effective sample size, equals `n` for uniform/absent weights |
| `rotation_spec` | `rotation_spec` | `RotationSpec \| None` | — |
| `selection_result` | — | `FindKOptimalResult \| FindKSequenceResult \| None` | `None` unless the model came from `pls1_fit(X, y, k="optimal" \| "sequence")` |
| `keep` | — | `int \| None` | `int` for `spls1_fit`; `None` for `pls1_fit` |

There is no `k_was_auto` flag and no `find_k_certificate` field. The
2026-04 confirmatory-vs-exploratory overhaul moved K-selection
diagnostics onto the K-selection result objects themselves, where
they originate. `rotation_spec` is `None` until `rotate(model, ...)`
stamps it onto a copy of the model.

## `PLS3Result`, what `pls3_fit` / `plssvd_fit` / `spls3_fit` returns

| Python field | Rust core field | numpy type | Shape |
|---|---|---|---|
| `U` | `u_saliences` | `np.ndarray` | `(p, k_used)`; orthonormal columns on a dense fit, unit-norm but not orthogonal on an `spls3_fit` |
| `V` | `v_saliences` | `np.ndarray` | `(q, k_used)`; same dense-only orthonormality condition |
| `singular_values` | `singular_values` | `np.ndarray` | `(k_used,)`; singular values of `X̃'Ỹ`, descending, on a dense fit. On a sparse fit, `σ_a = u_a'A_a v_a` on the deflated `A_a`: not singular values, not ordered, and `Σσ²/‖A‖_F²` is not an explained share |
| `x_scores` | `x_scores` | `np.ndarray` | `(n, k_used)` |
| `y_scores` | `y_scores` | `np.ndarray` | `(n, k_used)` |
| `X_mean` | `x_mean` | `np.ndarray` | `(p,)` |
| `X_scale` | `x_scale` | `np.ndarray` | `(p,)` |
| `Y_mean` | `y_mean` | `np.ndarray` | `(q,)` |
| `Y_scale` | `y_scale` | `np.ndarray` | `(q,)` |
| `k_used` | `k_used` | `int` | scalar; `< k` when a component's `σ` fell below `1e-14` or below `max(n, p, q)·ε·‖X̃‖_F·‖Ỹ‖_F`. `0` when this happens at the first component (a Y orthogonal to X up to rounding, or a constant block), matching `pls1_fit` |
| `pre_standardized_X` | `pre_standardized_x` | `bool` | scalar |
| `pre_standardized_Y` | `pre_standardized_y` | `bool` | scalar |
| `keep_X` | `keep_x` | `int \| None` | scalar; `None` on a dense fit |
| `keep_Y` | `keep_y` | `int \| None` | scalar; `None` on a dense fit |
| `converged` | `converged` | `np.ndarray \| None` | `(k_used,)` bool; `None` on a dense fit |
| `n_iter` | `n_iter` | `np.ndarray \| None` | `(k_used,)` int; `None` on a dense fit |

PLS3 is symmetric, so there is no `coef`, no `beta`, no `intercept` and no
`predict`. On a dense fit `U` and `V` have pinned signs (largest-magnitude
entry of each `U` column positive, the matching `V` column flipped with
it) and orthonormal columns, so repeated fits on the same data agree
exactly. On a sparse fit (`spls3_fit`) the columns are unit-norm but not
orthogonal, and `singular_values` are the per-component `u'Av` on the
deflated `A`, not singular values of `X'Y`: they need not descend and
their squares do not partition anything. Observation weights are not
implemented for this family, so there is no `weights` and no `n_eff`.

## `PLS3Scores` — what `pls3_transform` / `plssvd_transform` returns

| Python field | Rust core field | numpy type | Shape |
|---|---|---|---|
| `x_scores` | `x_scores` | `np.ndarray \| None` | `(n_new, k_used)`; `None` when `which` did not ask for it |
| `y_scores` | `y_scores` | `np.ndarray \| None` | `(n_new, k_used)`; `None` when `which` did not ask for it |

## `FindKOptimalResult` — what `pls1_find_k_optimal` returns

| Field | Python type | When populated |
|---|---|---|
| `k_star` | `int` | always (0 if the data admit no first component; see below) |
| `selector` | `str` (`"r2_se"` / `"r2_max"` / `"bic"`) | always |
| `cv_scores` | `dict[int, float] \| None` | `selector ∈ {r2_se, r2_max}` |
| `cv_scores_se` | `dict[int, float] \| None` | `selector="r2_se"` only |
| `bic_scores` | `dict[int, float] \| None` | `selector="bic"` |
| `pvalues` | `np.ndarray \| None` | `diagnostic` set |
| `diagnostic` | `str \| None` | `diagnostic` set |
| `seed` | `int` | always |
| `n_eff` | `float` | always; equals `n` for uniform/absent weights |
| `stable_rank` | `float \| None` | `diagnostic="split_nb"` requested — fired or not, including under `force` |

When `diagnostic=` is set, `pvalues` carries the per-component p-values
of a same-sample sequential test up to `k_star`, and `diagnostic`
echoes the method name. Selection and the diagnostic share the same
data, so the pvalues are a robustness check, not honest inference; a
fresh sample is required for a confirmatory claim. To get the
worst-case p-value along the path, compute `np.nanmax(pvalues)`.

`k_star` is `0` when the full-data fit cannot extract a first component:
`y` is constant, or orthogonal to the columns of `X` up to rounding
(`pls1_fit` returns the `k_used=0` zero model on such data). Every
selector then returns an empty score dict (`cv_scores`, `cv_scores_se`,
or `bic_scores`, whichever the selector populates) instead of
recommending a component the fit cannot produce. A requested diagnostic
runs no step: `pvalues` has shape `(0,)` and `diagnostic` names the
method the auto-gate resolved to.

## `FindKSequenceResult` — what `pls1_find_k_sequence` returns

| Field | Python type | When populated |
|---|---|---|
| `k_star` | `int` | always (0 if no component rejects at α, which includes a `y` with no first component) |
| `pvalues` | `np.ndarray` (k_max,) | always; trailing entries are `nan` if stop-early kicked in |
| `test_method` | `str` (`"raw_perm"` / `"split_nb"` / `"split_exact"` / `"e"`) | always |
| `alpha` | `float` | always |
| `seed` | `int` | always |
| `n_eff` | `float` | always; equals `n` for uniform/absent weights |
| `stable_rank` | `float \| None` | `test_method="split_nb"` requested — fired or not, including under `force` |

Closed testing on nested H is exact, so `pvalues[:k_star]` is an
honest FWER-controlled sequence. To get the path-max p-value
along the rejected chain, compute `np.nanmax(pvalues[:k_star])`.

On both result types `stable_rank` is what the auto-gate saw on the
undeflated `X`, so when `test_method` / `diagnostic` reads
`"split_exact"` after a `"split_nb"` request, it and `n_eff` are the
two numbers that explain why. Use `split_nb_gate` to get them without
running a test.

## `FindKeepOptimalResult` — what `spls1_find_keep_optimal` returns

| Field | Python type | Notes |
|---|---|---|
| `keep_star` | `int` | sparsest `keep` within 1 SE of the best mean CV R²; `0` if the data admit no first component (see below) |
| `k` | `int` | the fixed component count the sweep ran at |
| `cv_scores` | `dict[int, float]` | keep → mean CV R² across folds |
| `cv_scores_se` | `dict[int, float]` | keep → SE of the CV R² |
| `keep_grid` | `list[int]` | the logged geometric grid swept (powers of two; endpoints 1 and n_features always included); empty when `keep_star` is `0` |
| `seed` | `int` | always |
| `n_eff` | `float` | effective sample size (from weights; `nan` if unavailable) |

Selection criterion: the 1-SE rule on mean CV R² — `keep_star` is the
sparsest keep whose mean CV R² is within 1 SE of the maximum. Ties
broken toward sparser. The `keep_grid` field records exactly which
candidates were evaluated.

`keep_star` is `0` when the full-data fit cannot extract a first
component at any keep: `y` constant, or orthogonal to the columns of `X`
up to rounding (for instance an outcome residualized on covariates that
span `X`). `cv_scores`, `cv_scores_se` and `keep_grid` are then empty.
This is the `k_star = 0` convention of `FindKOptimalResult`, and the same
input gives `k_star = 0` from `spls1_find_k_optimal`.

## `ConfirmatoryTestResult` — what `pls1_confirmatory_test` returns

| Field | Python type | Notes |
|---|---|---|
| `pvalue` | `float` | always |
| `statistic` | `float` | always |
| `test_method` | `str` | one of `"raw_perm"` / `"split_nb"` / `"split_exact"` / `"score"` / `"e"`; `"split_exact"` when a `"split_nb"` request was rerouted by the auto-gate |
| `k` | `int` | the K tested (echoed from the input) |
| `n_perm` | `int \| None` | not None for resampling-family methods, None for `score` / `e` |
| `n_splits` | `int \| None` | not None for `split_*` methods, None for `raw_perm` / `score` / `e` |
| `seed` | `int` | always |
| `n_eff` | `float` | always; Kish effective sample size, equals `n` for uniform/absent weights |
| `rho_hat` | `float \| None` | `split_nb` only, and only when unweighted (no weights, or all-equal weights) with a test half of at least 4 rows; `None` for every other method, including `split_exact` |
| `stable_rank` | `float \| None` | stable rank of the standardized `X`, as seen by the `split_nb` auto-gate; populated whenever `"split_nb"` was requested (fired or not, including under `force`), `None` for every other requested method |
| `ci` | `ConfirmatoryCI \| None` | not None when called with `ci=True`; carries the rotation-invariant resampling CIs (subsampling for β and `holdout_corr`, bootstrap for leverage) |

There is no `at` field (legacy concept dropped). There is no
`null_distribution` or `split_mean_r` slot — those internals were
dropped from the public surface; only the headline result and
(optionally) the `ConfirmatoryCI` bundle survive.

## `SplitNbGateResult` — what `split_nb_gate` returns

| Field | Python type | Notes |
|---|---|---|
| `fires` | `bool` | `True` → a `"split_nb"` request on this `X` reroutes to `"split_exact"` |
| `stable_rank` | `float` | stable rank of the standardized `X` |
| `n_eff` | `float` | Kish effective sample size; equals `n` for uniform/absent weights |

Every field is always populated. This is the same rule the test
functions apply internally, evaluated on the same standardized `X` —
querying it costs one SVD and no resampling.

## `PermNullResult` — what `pls1_perm_null` returns

| Field | Python type | Notes |
|---|---|---|
| `n_perm` | `int` | number of permutations actually run |
| `k` | `int` | K used for fitting |
| `seed` | `int` | RNG seed actually used |
| `beta_ref` | `np.ndarray` | shape `(D,)`; full-data β |
| `beta_perm_mean` | `np.ndarray` | shape `(D,)`; ≈ 0 under H0 (calibration diagnostic) |
| `beta_perm_sd` | `np.ndarray` | shape `(D,)`; SD of β under permuted y |
| `beta_perm_z` | `np.ndarray` | shape `(D,)`; signed = β_ref / β_perm_sd |
| `beta_perm_matrix` | `np.ndarray \| None` | shape `(n_perm, D)` when `return_perm_matrix=True` |
| `n_eff` | `float` | effective sample size; `nan` if unavailable |

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
(see [api](api.md) §3.1).

| Field | Python type | Shape | Description |
|---|---|---|---|
| `n_boot` | `int` | scalar | resolved resampling replicates (each: one size-`m` subsample and one n-out-of-n bootstrap resample) |
| `m` | `int` | scalar | resolved subsample size, `m = ceil(n^m_rate)` |
| `m_rate` | `float` | scalar | echoed from the input |
| `level` | `float` | scalar | echoed from the input |
| `beta_sign_z` | `np.ndarray` | `(D,)` | per-variable folded subsampling z for β_j, `|β_ref[j]| / se_j`; roughly half-normal when β_j = 0 at K = 1 (see below) |
| `beta_sign_z_signed` | `np.ndarray` | `(D,)` | per-variable signed z = `sign(β_ref[j]) · beta_sign_z[j]`; roughly `N(0, 1)` when β_j = 0 at K = 1 |
| `leverage_ci_lower` | `np.ndarray` | `(D,)` | per-variable bootstrap CI lower bound on leverage, `h − Φ⁻¹(1−α/2)·leverage_se`, clamped to `[0, 1]` (see below) |
| `leverage_ci_upper` | `np.ndarray` | `(D,)` | per-variable bootstrap CI upper bound on leverage, `h + Φ⁻¹(1−α/2)·leverage_se`, clamped to `[0, 1]` |
| `leverage_se` | `np.ndarray` | `(D,)` | per-variable bootstrap SE of leverage, `sd(h_b)` over n-out-of-n bootstrap refits |
| `beta_ci_lower` | `np.ndarray` | `(D,)` | per-coordinate subsampling CI on β, shrinkage-corrected and finite-population-scaled; PLS1-only, calibrated at K = 1 (see caveats below) |
| `beta_ci_upper` | `np.ndarray` | `(D,)` | per-coordinate |
| `beta_se` | `np.ndarray` | `(D,)` | `= √(m/(n − m)) · sd(β_b[j]) / κ̂`, the SE `beta_sign_z` divides by (see below) |
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

### Per-variable readouts: leverage CIs and the subsampling z for β

**Leverage.** Variable `j`'s leverage is `h_j = [W (WᵀW)⁻¹ Wᵀ]_jj`, the
diagonal of the projection onto the span of the `K` weight vectors. It
does not change under sign flips or rotations of `W`, lies in `[0, 1]`
and sums to `K` over variables; at `K = 1` it is `w_j² / ‖w‖²`.
`leverage_ci_*` is a normal-theory bootstrap interval centered on the
full-data leverage `h`. Each replicate refits on an n-out-of-n bootstrap
resample (rows drawn with replacement, same standardization regime as
the full fit), giving leverages `h_b`; then
`leverage_se = sd(h_b)`, `lower = h − Φ⁻¹(1−α/2)·leverage_se` and
`upper = h + Φ⁻¹(1−α/2)·leverage_se`, both clamped to `[0, 1]`.

*What it covers.* The leverage estimate is biased at finite `n`, and the
bias depends on the sample size: noise in `Xᵀy` spreads weight from the
signal variables onto the noise variables, more so on fewer rows. The
interval targets `E[h]`, the expected leverage of a size-`n` fit, and
assumes `h` is roughly normal around it with an SD the bootstrap spread
estimates. It does not correct the bias toward the large-sample
leverage; at small `n/D` that value lies outside the interval more often
than `1 − level` for signal variables (see the numbers below). Two
choices follow from the bias. The replicates are size-`n` fits, because
a size-`m` subsample measures the spread and the bias at the wrong
sample size: the subsampling interval used before, reflected around `h`
at the rate `√(m/(n − m))`, inherited the gap between size-`m` and
size-`n` leverage as an offset and covered `E[h]` 40–65% of the time at
`D = 20`. And the interval is centered on `h` without a bootstrap bias
correction (not the basic `2h − q` or the percentile form), because
the bootstrap mean of `h_b` carries a further, resampling-induced share
of the same bias, and moving the interval by it traded coverage of
`E[h]` for partial coverage of the large-sample value.

*Simulated calibration* (`m_rate = 0.7`, `n_boot = 300`,
`y = s·(x₀ + x₁) + e` with uniform columns and noise, `s ∈ {1, 4}`,
`n ∈ {100, 200, 500}`, `D ∈ {6, 20}`; 1000 datasets per cell at `K = 1`,
500 at `K = 2, 3`; target `E[h]` from 2000 to 4000 fresh datasets per cell). At
`K = 1` the interval covers `E[h]` 0.92–0.97 of the time for the signal
variables (MC SE ≤ 0.007), and the between-dataset SD of `h` is 0.93 to
1.11 times `leverage_se`. At `K = 2, 3` coverage is 0.92–0.98 for signal
and 0.93–0.97 for noise variables. The leverage of a `K = 1` noise
variable is of order `1/n` and sits at the boundary, so its interval
is conservative (coverage near 1.00, lower bound 0 in nearly every run).
The large-sample leverage of the signal variables (0.5) is covered
0.93–0.95 of the time at `D = 6` but only 0.49–0.87 at `D = 20`, where
`E[h]` is 0.39 to 0.48 (`n = 100` to `500`). Treat the interval as the
sampling uncertainty of this fit's leverage, not as a bias-corrected
interval for the leverage of the population weights.

**Subsampling z for β.** `beta_sign_z_signed[j] = z_j = β_ref[j] / se_j`
and `beta_sign_z[j] = |z_j|`, where

```
se_j = beta_se[j] = √(m/(n − m)) · sd(β_b[j]) / κ̂
```

is the spread of the subsample replicates `β_b[j]` with two corrections
to plain centered-scaled subsampling (`√(m/n) · sd(β_b[j])`), which
`beta_ci_*` carries too:

1. **Finite-population factor `√(1 − m/n)`.** Subsamples are drawn
   without replacement, so `sd(β_b[j])` is the spread at size `m`
   shrunk by `1 − m/n`. Plain `√(m/n)` leaves it out; at the default
   `m_rate = 0.7` the factor is 1.16 at `n = 100`, 1.12 at `n = 200` and
   1.07 at `n = 1000`.
2. **Shrinkage factor `κ̂`.** A PLS fit on `m < n` rows shrinks the whole
   coefficient vector toward 0 more than the full fit does (at `K = 1`,
   `β = w·(wᵀXᵀy)/(wᵀXᵀXw)` with `w ∝ Xᵀy`, and the scalar factor falls
   as the sample shrinks), so `sd(β_b[j])` understates the spread of a
   size-`n` fit by the same factor. `κ̂` is the precision-weighted slope
   of the subsample means on the full-data fit,
   `κ̂ = Σ_j t̄_j·t_j / Σ_j t_j²` with `t_j = β_ref[j] / sd(β_b[j])` and
   `t̄_j = mean(β_b[j]) / sd(β_b[j])`, clamped to `[0, 1]`. It is one
   number per run and does not depend on the units of the columns of X.
   If `κ̂ = 0` (the subsample fits do not reproduce the full fit),
   `beta_se` is `inf`, every `beta_ci_*` interval is `(−inf, inf)` and
   every `z_j` is 0.

So `z_j = β_ref[j] / beta_se[j]` exactly, and ranking variables by
`beta_sign_z` is ranking them by `|β_ref| / beta_se`. `z_j` has no ceiling and does not grow with
`n_boot`: more resamples only reduce Monte Carlo noise in `se_j`. It is
NaN when `β_ref[j] ≠ 0` and the subsample spread of `β_b[j]` is zero or
undefined (fewer than two successful resamples).

When the population PLS coefficient `β_j` at the chosen `K` is 0,
`z_j` is roughly `N(0, 1)` at `K = 1`. In simulation (`D = 8`, `K = 1`,
`m_rate = 0.7`, 250 datasets per cell, `n_boot = 100`, `500` and `2000`
within a percentage point of each other), the share of null
coordinates with `|z| > 1.96` was:

| design | `n = 100` | `n = 200` | `n = 1000` |
|---|---|---|---|
| pure null (y independent of X) | 3.5–3.7% | 4.8–5.0% | 3.5–4.3% |
| `examples/` design (`y = 4(x₀ + x₁) + e`) | 5.9–6.3% | 5.7–6.3% | 4.5–4.6% |
| two correlated blocks (ρ = 0.6), signal in one | 5.6–5.8% | 6.3–6.4% | 4.8–5.5% |

with mean `|z|` between 0.76 and 0.85 (half-normal: 0.80), and signal
coordinates at mean `|z|` of about 10, 17 and 51 on the `examples/`
design. At `D = 50` and `D = 200` with `n ≤ 200` the pure-null and
`examples/` designs stay between 4.2% and 5.2%. Without `κ̂` the same
ratio over-rejects wherever the subsample fits shrink: 12–14% on the
`examples/` design at `n ≤ 200` (`κ̂` about 0.8) and 25–50% at
`D ≥ 50` (`κ̂` between 0.33 and 0.56).

Limits:

- **`K ≥ 2` is not calibrated.** `κ̂` models the shrinkage of a `K = 1`
  fit, which is one scalar; at `K ≥ 2` shrinkage differs by component. With one
  true direction and `K = 2`, null coordinates rarely exceed 1.96
  (under 1%); when a weak second direction competes with a strong noise
  factor, the subsampling distribution changes shape between `m` and
  `n`, and one simulated design reached 38%. `beta_ci_*` has the same limitation
  (caveat 3 below).
- **What `β_j = 0` means.** It is the population PLS coefficient at the
  chosen `K`, not a causal or OLS coefficient. A variable that does not
  enter `y` but is correlated with one that does has a nonzero PLS1
  coefficient, and `beta_sign_z` flags it, correctly for that estimand.
- **Pure null.** Slightly conservative (3.5–5.0%), since `κ̂` then
  slightly over-corrects.
- **No multiple-comparison correction**, and the reference is `N(0, 1)`
  from subsampling, not an exact null. For per-variable claims with
  exact error control, use `pls1_perm_null` (a permutation z per
  variable) after the omnibus test.

### Per-coordinate β CIs — diagnostic, with caveats

`beta_ci_lower / beta_ci_upper / beta_se` are a **regression-style
diagnostic for downstream pipelines** (one-line β_j ± SE tables in the
OLS reporting shape — psychometric reports, fMRI thresholding,
supplementary tables). They are not a primary inferential output;
the calibrated inferential outputs on `ConfirmatoryCI` are `holdout_corr`
(out-of-sample predictive) and, at `K = 1`, `beta_sign_z` (per-variable
z for β_j); `leverage_ci_*` (subspace importance) is a per-variable
readout on the sample fit (see above). Math: centered-scaled
subsampling on the replicates rescaled by `1/κ̂`, at the
finite-population rate `√(m/(n − m))` (the two corrections described
for `beta_sign_z` above):

```
Δ_b   = β_b[j] / κ̂ − β_ref[j]
lower = β_ref[j] − √(m/(n − m)) · q_{1−α/2}(Δ)
upper = β_ref[j] − √(m/(n − m)) · q_{α/2}(Δ)
```

Dividing by `κ̂` removes the shrinkage offset `(κ − 1)·β_ref` that would
otherwise move the midpoint away from 0, and restores the spread the
shrinkage takes out. β being unbounded means no transform is needed.
In simulation (`D = 8`, `K = 1`, `m_rate = 0.7`, `y = s·(x₀ + x₁) + e`,
300 datasets per cell, `n_boot = 300`; cells `n = 100` and `200` at
`s = 0.5`, `n = 200` at `s = 1`, `n = 500` at `s = 0.3`), coverage of
the population `K = 1` coefficient was 91–96% at the nominal 95%, and
the root-mean-square of `(β̂_j − β_j) / beta_se[j]` was 0.98–1.10.
Without the two corrections the same cells covered 77–89% and that
ratio was 1.17–1.44.

PLS1 only. In PLS2/PLSC the analogue β is a matrix that inherits W's
rotation/sign indeterminacy and requires procrustes alignment; per-β CIs
for those families are out of scope and will be documented separately.

Caveats — part of the contract, not optional commentary:

1. **PLS shrinkage bias on small m.** β_b on subsamples of size m is
   shrunk more aggressively than β_ref (PLS effective DoF scales
   sub-linearly in n; Krämer & Sugiyama 2011). At `K = 1` that
   shrinkage is one scalar, and `κ̂` corrects it (see the math above).
   `κ̂` is estimated once per run from all coordinates, so the
   correction is only as good as that fit of subsample means on
   `β_ref`; at `K ≥ 2` shrinkage differs by component and a single
   `κ̂` does not model it (caveat 3).
2. **No multiple-comparison correction.** Per-coordinate CIs at level
   α are individual, not simultaneous, and so is `beta_sign_z`. For
   claims about which coordinates are
   nonzero, use `pls1_perm_null` with an external FWER correction
   (Bonferroni, Westfall–Young, max-stat).
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
5. **Three-way distinction (sign-z ↔ leverage_ci ↔ beta_ci).**
   `beta_sign_z` and `beta_ci_*` read the same corrected replicates:
   the z uses their standard deviation (`beta_se`), the CI their
   quantiles. With a roughly normal subsampling distribution the CI is
   close to `β_ref ± 1.96·beta_se`, and it excludes zero exactly when
   `beta_sign_z[j] > 1.96`. With a skewed or heavy-tailed one they can
   disagree near the threshold; neither is then the better reading. `leverage_ci_*[j]` measures subspace
   contribution (leverage and its clamped CI lie in `[0, 1]`).
   A coordinate can have a confidently nonzero `leverage_ci` with a
   β CI through zero (the variable shapes the latent direction without
   contributing a stable regression coefficient on its own).

Memory: storing per-resample β adds `8 · D · n_boot_finite` bytes —
~8 MB at D=1000, B=1000 (trivial); ~400 MB at D=50_000, B=1000 (fMRI
scale). Brain-scale users typically reach for `pls1_perm_null` (sparse
z-map output) instead of the confirmatory CI bundle.

## `RotateResult` — what `rotate(W: np.ndarray, ...)` returns

| Field | Python type | Notes |
|---|---|---|
| `W_rot` | `np.ndarray` | rotated weights `W @ R` |
| `spec` | `RotationSpec` | the stamped rotation spec (method, args, `R`, sweeps, convergence value) |

## `RotationStabilityResult` — what `pls1_rotation_stability` returns

| Field | Python type | Notes |
|---|---|---|
| `method` | `str` | the rotation method used (e.g. `"varimax"`) |
| `n_boot` | `int` | resolved subsample iterations |
| `m` | `int` | resolved subsample size |
| `m_rate` | `float` | echoed from the input |
| `level` | `float` | echoed from the input |
| `seed` | `int` | always |
| `variance_ratio` | `CIScalar` | headline aggregate variance ratio `ρ = V_rot / V_unrot` with paired-bootstrap percentile CI |
| `variance_ratio_per_axis` | `list[CIScalar]` | per-axis ratio `ρ_k = V_rot,k / V_unrot,k`, length K, reference-axis order |
| `variance_unrot` | `float` | aggregate `V_unrot = (1/B) Σ_b Σ_k α²_unrot,b,k` |
| `variance_rot` | `float` | aggregate `V_rot = (1/B) Σ_b Σ_k α²_rot,b,k` |
| `variance_unrot_per_axis` | `np.ndarray` | per-axis `V_unrot,k`, length K, reference-axis order |
| `variance_rot_per_axis` | `np.ndarray` | per-axis `V_rot,k`, length K, reference-axis order |
| `degenerate_baseline` | `bool` | `True` iff `V_unrot = 0` on the engine pass or more than 5% of bootstrap iterations had `V_unrot* = 0`; only the first trigger makes `variance_ratio.point` `NaN`, under the second alone the point stays the finite `V_rot / V_unrot` |
| `n_boot_finite` | `int` | number of resamples that produced finite per-axis squared residuals (≤ `n_boot`) |
| `n_eff` | `float` | effective sample size `(Σ wᵢ)² / Σ wᵢ²` from the full normalized weight vector; equals `n` for uniform weights |

Interpretation: ratio < 1 means rotation reduced axis variance
(rotated axes are more replicable than unrotated ones); ratio ≈ 1
means rotation did not change axis replicability; ratio > 1 means
rotation increased axis variance (rotated axes are less replicable —
suspect a local-optimum varimax convergence issue).

## `RotationSpec` — stamped by `rotate(model, ...)`

| Field | Python type | Notes |
|---|---|---|
| `method` | `str` | `"varimax"` today; future `"promax"` / `"oblimin"` / `"geomin"` |
| `args` | `Mapping` (frozen via `MappingProxyType`) | method-specific kwargs as the engine resolved them (defaults filled, counts as `int`) |
| `R` | `np.ndarray` `(K, K)` | rotation matrix; `W_rot = W @ R` |
| `sweeps` | `int` | varimax iterations to convergence |
| `V_converged` | `float` | final varimax criterion value |
| `L_was_provided` | `bool` | whether caller passed a loading basis |

`rotation_spec` is `None` until `rotate(model, ...)` stamps it on a
copy of the model. That copy carries the rotated `T`, `P`, `W` and `Q`;
`beta`, `coef` and `intercept` are unchanged, so `pls1_predict` gives the
same predictions before and after rotation. `pls1_predict` does not read
`rotation_spec`.
