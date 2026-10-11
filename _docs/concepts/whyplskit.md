# Why plskit?

`plskit` fits PLS models and tests them. This page lists what it covers,
what it leaves out, and how it relates to other PLS packages.

## What it covers

- **Model families.** [PLS1](PLS1/index.md) for one outcome,
  [PLS3 / PLSSVD](PLS3/index.md) for two blocks (PLSC in neuroimaging and
  psychology), and sparse variants of both ([sPLS1](PLS1/sparse.md),
  [sPLS3](PLS3/sparse.md)).
- **A test of the model.** `pls1_confirmatory_test` asks whether there is
  signal at a chosen `k`. The recommended method, `split_exact`, is a
  split-half test calibrated by permutation
  ([picking a method](PLS1/fit-and-predict.md#confirmatory-testing-picking-a-method)).
  `pls3_confirmatory_test` asks the same of the first latent variable of a
  two-block model ([PLS3 inference](PLS3/inference.md)).
- **A choice of `k`** by cross-validation, BIC, or a sequence of tests
  ([Find K](PLS1/find-k.md)).
- **Confidence intervals** on the held-out correlation, per-variable
  leverage and β, from `pls1_confirmatory_test(ci=True)`
  ([Confidence intervals](PLS1/ci.md)).
- **Observation weights** for PLS1 and sparse PLS1
  ([Weights](PLS1/weights.md)).
- **One engine, four languages.** Rust, Python, R and Julia share function
  names, argument names and result fields. A fixed version, seed and input
  give the same numbers in each, within the tolerance of the shared test
  corpus, and the same bytes at any thread count.

## What it leaves out

- PLS2 (multi-response regression) and PLS-DA. A matrix `Y` goes to PLS3,
  which is symmetric and has no `predict`
  ([PLSC is not PLS2](PLS3/index.md#plsc-is-not-pls2)).
- VIP scores, bootstrap ratios and jackknife standard errors.
- Mean-centred (task) PLS.
- Estimator classes and formula interfaces. The API is a set of functions
  that return immutable result objects.
- Penalty-based sparsity. Sparsity here is a fixed count of variables.

[Limitations](limitations.md) has the full list.

## Other packages

- **scikit-learn** (`PLSRegression`, `PLSSVD`) also fits a multi-column
  `Y` and works as a pipeline estimator. The repository's `validation/`
  tool compares `plskit` with both: PLS1 predictions on new rows, and PLS3
  weights and scores up to sign. scikit-learn standardizes with the
  `n − 1` standard deviation and `plskit` with the `n` one, so PLS3 scores
  differ by a factor `sqrt(n / (n − 1))`. `PLSSVD` does not expose the
  singular values; `pls3_fit` returns them.
- **ikpls** implements Improved Kernel PLS algorithms 1 and 2, with sample
  weights. `plskit` computes PLS1 with the same algorithm family
  ([how plskit computes it](PLS1/fit-and-predict.md#how-plskit-computes-it)),
  and `validation/` compares the two with and without weights.
- **R `pls`** fits PLS regression, principal component regression and
  CPPLS through a formula interface. The `plskit` R package takes matrices
  and names the component count `k`, not `ncomp`
  ([R: differences from Python](../r/differences.md)).
- **pyls** implements behavioural and mean-centred PLSC and PLS
  regression, with permutation testing and bootstrap resampling.
  `pls3_fit` is the behavioural-PLSC model: one SVD of the cross-product
  of two standardized blocks. `plskit` tests only the first latent
  variable, by held-out correlation, and reports no bootstrap ratios
  ([PLS3 inference](PLS3/inference.md)).
- **pyChemometrics** offers PCA, PLS regression and PLS-DA with
  mean-centring, Pareto and unit-variance scaling, cross-validation,
  permutation tests and VIP. `plskit` has no PLS-DA and no VIP, and one
  standardization recipe. Data scaled another way goes in with
  `pre_standardized=True`
  ([Preprocessing](preprocessing.md#chemometrics-quick-reference)).
