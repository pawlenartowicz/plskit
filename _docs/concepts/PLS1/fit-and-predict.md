# Fit and predict

PLS1 fits a single-response regression `y ≈ X β` by extracting `K`
components that maximize the covariance between `X` and `y` at each step.
The β coefficient vector is reconstructed from the `K`-component
decomposition; predictions on new data follow from the same β, with the
training-time centering and scaling already folded into β and the
intercept.

## The PLS1 model

Inputs:

- `X (n × p)`: predictor matrix, real-valued
- `y (n,)`: response vector, real-valued
- `k`: number of components, `1 ≤ k ≤ p`, with at least `k + 1`
  observations (`n_eff ≥ k + 1` under weights). The fit may retain fewer
  components than requested: on rank-deficient `X`, or once the part of
  `y` that `X` can explain has been fitted and the remaining
  cross-product `X'y` is rounding noise. The result's `k_used` says how
  many it kept. It is `0` when `y` is constant or orthogonal to the
  columns of `X` up to rounding: `beta` is then zero and the model
  predicts the mean of `y`. With `pre_standardized=true`, a top-level fit
  that keeps fewer than `k` components raises instead.

`pls1_fit` returns a `PLS1Result`. The full field list, with shapes and
Rust core names, is in [`PLS1Result`](../../python/results.md). The
three fields this page is about:

| Field | Shape | Meaning |
|---|---|---|
| `beta` | `p` | Regression coefficient β on the original (raw) scale of `X` and `y` |
| `intercept` | scalar | Intercept `α` so that `ŷ = α + X β` on the original scale |
| `coef` | `p` | The same coefficients on the standardized scale, before back-projection |

`beta` is the headline output. The NIPALS quantities `W`, `T`, `P`, `Q`
(described below) are exposed for diagnostics and downstream tools.
Rotating a model with `rotate` replaces `T`, `P`, `W`, `Q` and leaves
`beta`, `coef`, and `intercept` unchanged, so predictions from a rotated
model are identical to those from the original.

## Standardization

By default, `pls1_fit` standardizes each column of `X` and the response
`y` to zero mean and unit **population** variance (`ddof=0`); see
[Preprocessing](../preprocessing.md) for the exact recipe, including
weighted moments and zero-variance columns. Standardization is part of
the contract. The PLS1 kernel runs on the standardized data and produces `coef`;
the fit then back-projects it to `beta = coef · s_y / s_x` on the
original scale and absorbs the means into `intercept`.

Two opt-outs:

- `pre_standardized=True`: skip standardization (and weight
  normalization); the caller asserts that `X` and `y` are already
  centered and scaled. The fit then reports `beta` equal to `coef` and
  `intercept = 0`. Used when the same preprocessing is shared across
  many fits and you want to avoid recomputing it.
- `plskit.preprocess`: a cache helper that performs the canonical
  standardization once; pass its standardized arrays with
  `pre_standardized=True` to `pls1_fit`, `pls1_confirmatory_test`, and
  `pls1_find_k_*` calls on the same data.

## NIPALS: the algorithm

The model `pls1_fit` fits is the **NIPALS** PLS1 model (Nonlinear
Iterative Partial Least Squares), attributable to Wold's group in the
1970s. The single-`y` case has a particularly clean form: components are
extracted **non-iteratively**, one at a time, by repeated deflation.

For each component `k = 1, …, K`, given residuals `X_{k−1}` and
`y_{k−1}` from the previous step (with `X_0 = X`, `y_0 = y` after
standardization):

1. **Weight.** `w_k = X_{k−1}ᵀ y_{k−1} / ‖X_{k−1}ᵀ y_{k−1}‖`:
   the unit direction in `X`-space most correlated with the current
   `y` residual. This is the defining choice that distinguishes PLS
   from PCA (which would pick the leading eigenvector of `X_{k−1}ᵀ
   X_{k−1}`, ignoring `y`) and from OLS (which would solve for β
   directly, ignoring rank).
2. **Score.** `t_k = X_{k−1} w_k`: project `X_{k−1}` onto `w_k`.
3. **Loadings.** `p_k = X_{k−1}ᵀ t_k / (t_kᵀ t_k)` and
   `q_k = y_{k−1}ᵀ t_k / (t_kᵀ t_k)`: regress `X_{k−1}` and
   `y_{k−1}` on `t_k`.
4. **Deflate.** `X_k = X_{k−1} − t_k p_kᵀ` and
   `y_k = y_{k−1} − q_k t_k`: remove the component-`k` signal
   before the next iteration.

The columns `w_k`, `t_k`, `p_k` and the entries `q_k` are what the
result reports as `W`, `T`, `P`, `Q`. After `K` components, the
standardized-scale coefficient is reconstructed:

```
coef = W (PᵀW)⁻¹ Q
```

For PLS1, this single-pass loop converges by construction: there is
no iterative refinement and no convergence tolerance to tune.

### How plskit computes it

`plskit` does not run the deflation above literally. It computes the
same model with **Improved Kernel PLS** (Dayal and MacGregor 1997,
Algorithm 1), which never forms the deflated matrices `X_k` and never
deflates `y`. Write `s_k = X_{k−1}ᵀ y_{k−1}`, so that
`w_k = s_k / ‖s_k‖`. In exact arithmetic, by induction on `k`:

- **The cross-product is updated, not recomputed.**
  `s_{k+1} = s_k − p_k (q_k t_kᵀ t_k)`: deflating `X` removes `t_k`
  from every later score, so `y` needs no deflation, and
  `t_kᵀ y_{k−1} = q_k t_kᵀ t_k`.
- **Scores come from the undeflated `X`.** `t_k = X r_k` with
  `r_k = w_k − Σ_{j<k} r_j (p_jᵀ w_k)`: the deflated matrix is
  `X_{k−1} = X M_k` with `M_k = I − Σ_{j<k} r_j p_jᵀ`, and
  `r_k = M_k w_k`.
- **Loadings come from the undeflated `X`.** `p_k = Xᵀ t_k / (t_kᵀ t_k)`:
  the scores are mutually orthogonal, so deflation does not change the
  loadings on `t_k`.
- **`y` loadings come from the undeflated `y`.** `q_k = yᵀ t_k / (t_kᵀ t_k)`:
  the fitted part `y − y_{k−1}` is a combination of earlier scores, all
  orthogonal to `t_k`.

So `W`, `T`, `P`, `Q`, `coef` and `beta` mean exactly what the steps
above say, and the rules that stop the fit early (the result's `k_used`)
test the same quantities, `‖s_k‖` and `t_kᵀ t_k`. The two computations
differ only in rounding: fits with two or more components agree with the
literal deflation well within the corpus tolerance, and a one-component
fit is the same sequence of floating-point operations on both. Each
component reads `X` twice and never writes it, and the fit keeps no
working copy of `X`.

Reference: Dayal, B. S. and MacGregor, J. F. (1997). Improved PLS
algorithms. *Journal of Chemometrics* 11(1), 73-85.

## Predicting on new data

`pls1_predict(model, X_new)` applies β to new observations:

```
ŷ_new = intercept + X_new · beta
```

`pls1_predict` does not standardize `X_new` itself. It does not need
to: the training-time column means and scales were folded into `beta`
and `intercept` at fit time, so raw `X_new` on the training scale gives
predictions on the original `y` scale. Re-standardizing `X_new` on its
own moments would silently change the coefficient interpretation.

If the model was fitted with `pre_standardized=True`, `intercept` is `0`
and `beta` equals `coef`: standardize `X_new` yourself with the
**training** means and scales (for example `X_mean` / `X_scale` from
`plskit.preprocess`), and the predictions come out on the standardized
`y` scale.

What it does not re-apply:

- Any preprocessing the caller did *before* `pls1_fit` (transforms,
  derivatives, smoothing). Reapply those manually to `X_new` before
  calling `pls1_predict`.

Edge cases: `X_new` with `n_new = 1` row works like any other. A column
whose training-time values were constant up to rounding (see
[Preprocessing](../preprocessing.md) for the exact rule) standardizes to
a column of zeros, so its fitted coefficient is zero up to
floating-point rounding and its value in `X_new` has no practical
effect on `ŷ`. This is a property of the fit, not something
`pls1_predict` does.

## Confirmatory testing: picking a method

After fitting at a chosen `k`, the natural next question is: **is this
model statistically supported, or could the apparent fit be noise?**
`pls1_confirmatory_test` provides five methods; `test_method` is a keyword
argument that defaults to `"auto"` (see
[`test_method="auto"`](inference.md#test_methodauto)). The honest split rule applies (see
[Find K](find-k.md)): if `k` was chosen on the same data, the inference
is exploratory, not confirmatory.

| Method | When to use | Cost |
|---|---|---|
| `split_exact` | **Recommended, especially at k = 1.** Split-half test on Fisher-z of held-out correlation, calibrated by permutation, so it holds its level on any design. At K = 1 the engine uses a no-refit route: the fitted direction is a fixed linear map of `y`, so every permutation reuses it, which keeps this route cheap; K ≥ 2 falls back to a per-permutation refit. | `O(n_splits)` GEMM pairs of width `n_perm` at K = 1; `O(n_splits × n_perm)` fits at K ≥ 2 |
| `split_nb` | Same statistic as `split_exact`, calibrated by a Fisher-z t approximation instead of permutation: cheaper, but only appropriate for `n` large relative to `p` with a flat X spectrum. A design with `n_eff < 25`, 4 columns or fewer, or a stable rank `< 3` is auto-gated: an explicit `split_nb` request reroutes to `split_exact` (at `n_perm=1000`) unless you pass `args={"force": True}`. | `O(n_splits)` fits |
| `score` | Fast pre-fit screening test on `‖X′ y‖²`. **Detects signal in `span(X)`; does not validate the PLS fit at your chosen `k`.** Faster and more powerful than the split tests when its assumptions hold, but sensitive to heavy tails / outliers in `y`. Use as a cheap omnibus check, not as a fit-validation test. | One matvec + one eigendecomp |
| `e` | Universal-inference e-value. Run only when you specifically need an **e-value**: for anytime-valid sequential testing, optional-stopping inference, or composition with other e-processes. Substantially less powerful than `split_exact` for the omnibus K-fixed test. | One PLS fit |
| `raw_perm` | **Legacy. Do not use for new analyses.** Implemented for compatibility with the chemometrics permutation-Q² convention; included so users porting workflows from older tools can reproduce historical numbers. Power and calibration are uniformly worse than `split_exact`. | `O(n_perm)` fits |

For a deeper treatment of the split-half construction, the Fisher-z
calibration, the e-process, and how the score test relates to the
Rao / Lagrange-Multiplier framework, see [Inference](inference.md).

## Cross-references

- [Find K](find-k.md): choosing `k` (`pls1_find_k_optimal`,
  `pls1_find_k_sequence`)
- [Inference](inference.md): full treatment of the five confirmatory
  tests, with the split-half construction and validity proofs
- [Confidence intervals](ci.md): rotation-invariant subsample CIs
  via `pls1_confirmatory_test(ci=True)`
- [Weights](weights.md): observation weights for WLS-style fits
- [Preprocessing](../preprocessing.md): the canonical standardization
  recipe and the `pre_standardized` flag
- [Python API → `pls1_fit` / `pls1_predict`](../../python/api.md)
- [Python results → `PLS1Result`](../../python/results.md)
- [Rust API](../../rust/api.md)
