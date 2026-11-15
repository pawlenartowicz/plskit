# Covariates

`covariates` adjusts a PLS1 analysis for variables you do not want the X→y relation to carry: age, sex, a site factor. The fit, the prediction, the permutation tests, the choice of k, `pls1_perm_null`, `factors` and `rotate` all take the same argument, so the fitted model, its β map and its tests describe one adjusted model.

## Tutorial

A study of 300 people with 40 predictors. Age and a two-level site affect y. Age is also related to the first five predictors. Three other predictors carry the effect we want to test.

```python
import numpy as np
import plskit

rng = np.random.default_rng(0)
n, p = 300, 40
age = rng.normal(40, 10, n)
site = rng.integers(0, 2, n).astype(float)    # already dummy-coded: 0 or 1
X = rng.standard_normal((n, p))
X[:, :5] += 0.05 * (age[:, None] - 40)         # age shows up in X
y = X[:, 5:8] @ np.array([0.4, 0.3, -0.3]) + 0.08 * age + 1.0 * site + rng.standard_normal(n)

Z = np.column_stack([age, site])               # no intercept column: plskit adds it
```

Fit with the covariates. `beta` is the effect of X with age and site held out, and `covariate_coef` holds one coefficient per column of `Z`.

```python
model = plskit.pls1_fit(X, y, k=1, covariates=Z)

model.n_covariates      # 2
model.covariate_coef    # one entry for age, one for site
model.beta.shape        # (40,): partial β on the original scale of X
```

Test whether X predicts y beyond age and site. Pass the same `covariates`. The default `test_method="auto"` runs `split_exact` with Freedman–Lane null columns.

```python
test = plskit.pls1_confirmatory_test(X, y, k=1, covariates=Z, seed=1)

test.test_method        # "split_exact"
test.pvalue             # p-value for X beyond Z
test.n_covariates       # 2
```

Predict new rows. The new rows need their own covariate values.

```python
y_hat = plskit.pls1_predict(model, X[:5], covariates=Z[:5])
```

The choice of k, `pls1_perm_null`, `factors` and `rotate` take the same `covariates=Z` (see "Which methods take covariates" below).

## Caveat: let plskit remove the covariates

There are two common ways to handle covariates by hand: ignore them, or regress them out of X and y first and run the usual test on the residuals (the Kennedy method). Both give false positives in the setting covariates are for: Z affects y and is related to X. `covariates=` uses the Freedman–Lane method, which held the 5% rate in every setting below.

The setting: two real X matrices, word embeddings (n = 200, p = 300) and Tecator NIR spectra (n = 215, p = 100). X has no effect on y. There are 1, 10, 20 or 40 covariates, each correlated 0.5 with a direction in X. Z explains either none or half of the variance of y. Each setting has 500 replicates, tested with `split_exact` at k = 1 and α = 0.05. A correct test rejects 5% of the time, so with 500 replicates a rate between 3.2% and 7.0% is expected.

Rejection rates under the null, lowest to highest over the two data sets and four covariate counts:

| Method | Z does not affect y | Z explains half of y |
|---|---|---|
| Covariates ignored | 4.2% to 6.6% | 29% to 96% |
| Kennedy: residualize X and y by hand, then test | 5.4% to 68% | 4.0% to 65% |
| plskit `covariates=` (Freedman–Lane) | 4.6% to 6.0% | 4.0% to 6.4% |

Ignoring Z fails whenever Z affects y: the test then finds the path from X to y through Z. Kennedy fails as the number of covariates grows on the embeddings, where p is larger than n: 9 to 13% at 10 covariates, 23 to 25% at 20 and 65 to 68% at 40, whether or not Z affects y. On the spectra it stayed at or below 7.0%.

## The argument

`covariates` is an `n × p_Z` float matrix, or a vector of length `n` for one covariate. The default is none. Dummy-code factors yourself (`model.matrix(~ age + site)[, -1]` in R, `pandas.get_dummies(..., drop_first=True)` in Python).

An intercept is always added, so you never pass one. A column of Z that is constant, or a linear combination of the intercept and the columns before it, is dropped, as R's `lm` does. A passed intercept column, a constant column or a duplicate is therefore harmless: the fit is the same as without it, and its entry in `covariate_coef` is 0. Columns are checked in order. Each column is projected on the intercept and the columns kept before it, and it is dropped when what is left has a (weighted) norm of at most `1e-7` times its norm before, which is `lm.fit`'s default tolerance. There is no pivoting.

A `covariates` with the wrong row count raises `shape_mismatch`, one with a non-finite entry raises `non_finite_input`, and one that leaves too few rows raises `insufficient_residual_df` (`n − rank([1, Z])` must be at least `k + 1`, with `n` the rows with positive weight).

## What is removed, and in what order

1. X is standardized by plskit's recipe (weighted moments when weights are given), or used as given under `pre_standardized=True`.
2. Z is removed from the standardized X and from y, over all rows: `X̃ = M_Z·X_std` and `ỹ = M_Z·y`, where `M_Z` takes the residual of a regression on `[1, Z]`. With weights it is the weighted least-squares residual.
3. PLS1 runs on `(X̃, ỹ)` as it runs on any data.

Standardizing first keeps plskit's usual column weights: a column that Z mostly explains is left with little weight. Removing Z from the raw X and then standardizing would put every leftover column back at unit variance. That order is not offered.

Every field of the fitted model other than `beta`, `intercept` and `covariate_coef` describes the adjusted data: the scores `T` are `X̃·W`, not `X_std·W`.

## Prediction

A model fitted with covariates predicts

```
y_hat = intercept + X_new · beta + Z_new · covariate_coef
```

`beta` is the partial β on X's original scale, `covariate_coef` has one entry per column of the `covariates` you passed (0 at a dropped column), and `intercept` is the fit's intercept, also under `pre_standardized=True`. `pls1_predict` needs `covariates` for such a model and refuses it for a model fitted without them. Its column count must equal that of the fit.

## Permutation tests: Freedman–Lane

Residualizing y and X on Z and then permuting the residuals as if they were free values (Kennedy) rejects too often (see the caveat above). Freedman–Lane builds each null outcome as the part of y fitted on Z plus the permuted residuals, `y* = Z·γ̂ + π(e)`, with `e` the residual of y on `[1, Z]` and `π` a permutation. After Z is removed again, the null column is `M_Z·π(e)`.

Freedman–Lane is approximately exact, not exact. The null columns are orthogonal to `[1, Z]`, but a permutation of `e` is not a draw from the exact null distribution of the residual.

With weights, `e` is the raw-scale weighted residual of y, `π(e)` is permuted, and each weight stays on its row position, as in every weighted permutation test of plskit. The null column is the weighted residual of `π(e)` on `[1, Z]`.

## Which methods take covariates

| Function or method | With `covariates` |
|---|---|
| `pls1_fit`, `spls1_fit`, `pls1_predict` | adjusted fit and prediction |
| `pls1_confirmatory_test`, `test_method="split_exact"` | Freedman–Lane null columns |
| `pls1_confirmatory_test`, `test_method="raw_perm"` | Freedman–Lane null columns |
| `pls1_confirmatory_test`, `test_method="auto"` | runs `split_exact` |
| `pls1_confirmatory_test`, `test_method="sv_perm"`, and `"score"` with `n_perm` | Freedman–Lane null columns |
| `pls1_confirmatory_test`, `test_method="split_nb"`, `"score"` without `n_perm`, `"e"` | refused (`invalid_argument`) |
| `pls1_find_k_optimal`, `pls1_find_k_sequence`, `spls1_find_k_optimal`, `spls1_find_k_sequence`, `spls1_find_keep_optimal` | selection on the adjusted data; their test methods follow the rules above |
| `pls1_perm_null` | Freedman–Lane null columns, β maps on `X̃` |
| `factors`, `rotate` | X is adjusted before use |
| `profiles`, `profiles_search` | refused for a model fitted with covariates |

`keep` and `ci=True` work with covariates. With `ci=True` each subsample redoes the whole adjustment on its own rows, so the intervals include the sampling noise of the covariate coefficients. With `gram=`, the call applies `M_Z·G·M_Z` itself, and `covariates` has one row per row of the Gram matrix; a `result.subset(rows)` Gram matrix needs a Z with those rows.

The result of every test and selection records `n_covariates`, the rank of `[1, Z]` minus 1, or `None` without covariates. The model records it too.

## `factors` and `rotate`

`factors(model, X, n, covariates=Z)` and `rotate(f, X, covariates=Z)` use `X̃` where they use X without covariates. For a model fitted with covariates they require `covariates` and raise an error without it, so they never run on unadjusted X. For a model fitted without covariates they refuse it. A `FactorsResult` records whether covariates were removed in `covariate_adjusted`, and `rotate(f, X)` applies the same two rules to it. For a raw coefficient vector or weight matrix `covariates` is optional: you say whether it came from an adjusted fit. `rotate` without `X` refuses `covariates`.

`profiles` and `profiles_search` take a model and X but have no `covariates` argument. They refuse a model fitted with covariates rather than run it on unadjusted X; use `factors` and `rotate`.

## `null_y`: your own nulls

For a null plskit cannot build (a mixed model, a GLMM), pass your own null outcomes: `pls1_confirmatory_test(X, y, k=1, test_method="split_exact", null_y=Ynull)` with `Ynull` of shape `n × B`. plskit draws its splits once, computes the statistic on `y` and on each column of `Ynull` with the same splits, and reports `p = (#{null ≥ observed} + 1) / (B + 1)`. Over a set of k (the default without `k`), that is the p at each k (in `pvalues_by_k`) and the reported p is the min-p p over the set.

The columns of `Ynull` are on the scale of y. `null_y` works with `split_exact` only, so the default `test_method="auto"` is refused: pass `test_method="split_exact"`. It refuses `covariates` (adjust X and y yourself) and an explicit `n_perm` in `args` (B is the number of columns). It works with `gram=`. Memory for the columns is `B·n` doubles.
