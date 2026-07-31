# Fit and predict

PLS1 fits a single-response regression `y ≈ X β` by extracting `K`
components that maximize the covariance between `X` and `y` at each step.
The β coefficient vector is reconstructed from the `K`-component
decomposition; predictions on new data follow from the same β with the
training-time centering and scaling re-applied.

## The PLS1 model

Inputs:

- `X (n × p)` — predictor matrix, real-valued
- `y (n,)` — response vector, real-valued
- `k` — number of components, `1 ≤ k ≤ min(n−1, p)`

Outputs (returned on the `PLS1Result` object):

| Field | Shape | Meaning |
|---|---|---|
| `W` | `p × k` | PLS weights — directions in `X`-space, unit norm |
| `T` | `n × k` | Component scores — projections of `X` onto `W` |
| `P` | `p × k` | `X`-loadings — regression of `X` on `T` |
| `q` | `k` | `y`-loadings — regression of `y` on `T` |
| `coef` | `p` | Regression coefficient β at the requested `k` |
| `intercept` | scalar | Intercept `α` so that `ŷ = α + X β` on the original scale |

The β vector is the headline output. `W`, `T`, `P`, `q` are exposed for
diagnostics, rotations, and downstream tools (CIs, stability tests).

## Standardization

By default, `pls1_fit` standardizes columns of `X` to zero mean and unit
sample variance, and centers `y` to zero mean. Standardization is part
of the contract — β returned in `coef` is on the original scale of `X`
and `y`, with the centering and scaling absorbed into `intercept`.

Two opt-outs:

- `pre_standardized=True` — skip both steps; the caller asserts that
  `X` and `y` are already centered and scaled. Used when the same
  preprocessing is shared across many fits and you want to avoid
  recomputing it.
- `plskit.preprocess` — a cache helper that performs the standardization
  once and reuses it across `pls1_fit`, `pls1_confirmatory_test`, and
  `pls1_find_k_*` calls on the same data.

## NIPALS — the algorithm

PLS1 uses **NIPALS** (Nonlinear Iterative Partial Least Squares),
attributable to Wold's group in the 1970s. The single-`y` case has a
particularly clean form: components are extracted **non-iteratively**,
one at a time, by repeated deflation.

For each component `k = 1, …, K`, given residuals `X_{k−1}` and
`y_{k−1}` from the previous step (with `X_0 = X`, `y_0 = y` after
standardization):

1. **Weight.** `w_k = X_{k−1}ᵀ y_{k−1} / ‖X_{k−1}ᵀ y_{k−1}‖` —
   the unit direction in `X`-space most correlated with the current
   `y` residual. This is the defining choice that distinguishes PLS
   from PCA (which would pick the leading eigenvector of `X_{k−1}ᵀ
   X_{k−1}`, ignoring `y`) and from OLS (which would solve for β
   directly, ignoring rank).
2. **Score.** `t_k = X_{k−1} w_k` — project `X_{k−1}` onto `w_k`.
3. **Loadings.** `p_k = X_{k−1}ᵀ t_k / (t_kᵀ t_k)` and
   `q_k = y_{k−1}ᵀ t_k / (t_kᵀ t_k)` — regress `X_{k−1}` and
   `y_{k−1}` on `t_k`.
4. **Deflate.** `X_k = X_{k−1} − t_k p_kᵀ` and
   `y_k = y_{k−1} − q_k t_k` — remove the component-`k` signal
   before the next iteration.

After `K` components, the regression coefficient is reconstructed:

```
β = W (PᵀW)⁻¹ q
```

For PLS1, this single-pass loop converges by construction — there is
no iterative refinement, no convergence tolerance to tune. The `tol`
and `max_iter` options on `Pls1FitOpts` are reserved for the symmetric
variants (PLS2, PLS3/PLSSVD) where component extraction is genuinely
iterative.

## Predicting on new data

`pls1_predict(model, X_new)` applies β to new observations:

```
ŷ_new = intercept + X_new · coef
```

What `pls1_predict` re-applies for you:

- **Centering and scaling** of `X_new` using the **training-time**
  column means and standard deviations (stored on the `PLS1Result`).
  Re-standardizing on `X_new` alone would silently change the
  coefficient interpretation.

What it does not re-apply:

- Any preprocessing the caller did *before* `pls1_fit` (transforms,
  derivatives, smoothing). Reapply those manually to `X_new` before
  calling `pls1_predict`.

Edge cases handled by the function (rather than the caller): `X_new`
with `n_new = 1` row, columns whose training-time standard deviation
was zero (treated as constant; their coefficient is exactly zero).

## Confirmatory testing — picking a method

After fitting at a chosen `k`, the natural next question is: **is this
model statistically supported, or could the apparent fit be noise?**
`pls1_confirmatory_test` provides six methods. The honest split rule
applies (see [Find K](find-k.md)): if `k` was chosen on the same data,
the inference is exploratory, not confirmatory.

| Method | When to use | Cost |
|---|---|---|
| `split_nb` | **Default.** Calibrated split test on Fisher-z of held-out correlation. Robust to non-Gaussian `y` and to non-iid `X`. | `O(n_splits)` fits |
| `split_perm` | More robust variant of the split test — exact calibration via permutation rather than Fisher-z asymptotics. Slower than `split_nb`; pick when you do not want to rely on the Gaussian approximation in the calibration step. | `O(n_splits × n_perm)` |
| `split_perm_nr` | Same statistic as `split_nb`, calibrated by permutation instead of the Fisher-z t approximation, and with no inner refits — at K = 1 the fitted direction is a fixed linear map of `y`, so every permutation reuses it. Much cheaper than `split_perm`. **K = 1 and unweighted input only**; errors on anything else. | `O(n_splits)` GEMM pairs of width `n_perm`, no fits |
| `score` | Fast pre-fit screening test on `‖X′ y‖²`. **Detects signal in `span(X)`; does not validate the PLS fit at your chosen `k`.** Faster and more powerful than the split tests when its assumptions hold, but sensitive to heavy tails / outliers in `y`. Use as a cheap omnibus check, not as a fit-validation test. | One matvec + one eigendecomp |
| `e` | Universal-inference e-value. Run only when you specifically need an **e-value** — for anytime-valid sequential testing, optional-stopping inference, or composition with other e-processes. Substantially less powerful than `split_nb` for the omnibus K-fixed test. | One PLS fit |
| `raw_perm` | **Legacy. Do not use for new analyses.** Implemented for compatibility with the chemometrics permutation-Q² convention; included so users porting workflows from older tools can reproduce historical numbers. Power and calibration are uniformly worse than `split_nb`. | `O(n_perm)` fits |

For a deeper treatment of the split-half construction, the Fisher-z
calibration, the e-process, and how the score test relates to the
Rao / Lagrange-Multiplier framework, see [Inference](inference.md).

## Cross-references

- [Find K](find-k.md) — choosing `k` (`pls1_find_k_optimal`,
  `pls1_find_k_sequence`)
- [Inference](inference.md) — full treatment of the five confirmatory
  tests, with the split-half construction and validity proofs
- [Confidence intervals](ci.md) — rotation-invariant subsample CIs
  via `pls1_confirmatory_test(ci=True)`
- [Weights](weights.md) — observation weights for WLS-style fits
- [Python API → `pls1_fit` / `pls1_predict`](../../python/api.md)
- [Rust API](../../rust/api.md)
