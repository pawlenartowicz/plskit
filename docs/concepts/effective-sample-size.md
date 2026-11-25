# Effective sample size and the n_eff check

## What `n_eff` is

`n_eff` is Kish's effective sample size: `(Σwᵢ)² / Σwᵢ²`, computed on the mean-one weights (`wᵢ · n / Σw`). Kish's ratio does not depend on the scale of the weights, so this gives the value of the raw weights up to rounding. For uniform or absent weights, every term equals 1 and `n_eff = n`.

`n_eff` captures how much information the weighted sample contains: a
sample where one observation has all the weight has `n_eff ≈ 1`; a
sample with perfectly balanced weights has `n_eff = n`. The check
`n_eff < k + 1` is the PLS analogue of the OLS rule that you need at
least as many observations as parameters. With weighted data, the
relevant count is the effective number of observations, not the raw
count.

Every PLS1 and sPLS1 result reports the `n_eff` it was computed with
(except `pls1_perm_alpha`'s, which has no `n_eff` field), and
`preprocess` and `split_nb_gate` report it too.

## Where the check fires

The check runs once, on the full data you pass in, at the start of each
public function that fits PLS1 components. If it fails, the function
errors before any fitting.

| Function | Checked against |
|---|---|
| `pls1_fit`, `spls1_fit` | `k` |
| `pls1_confirmatory_test` (every method, with or without `ci`) | the largest `k` of the set |
| `pls1_perm_null` | `k` |
| `spls1_find_keep_optimal` | `k` |
| `pls1_find_k_optimal`, `pls1_find_k_sequence`, `spls1_find_k_optimal`, `spls1_find_k_sequence` | `k_max` |

The K-selection functions check against `k_max`, the largest model they
will try, so a feasible `k_max` guarantees every candidate `k` is feasible
on the full data.

No check runs in `preprocess` and `split_nb_gate` (they take no `k`), in
`pls1_predict` and `rotate` (they take no weights), in `pls1_perm_alpha`
(it fits PLS1 but takes no `X` and no weights, so `n_eff = n`; its own
check refuses `k > n - 1`), or in the PLS3 family
(which takes no weights, so `n_eff = n`).

### Inside resampling loops

Resampling loops fit on pieces of the data: CV folds, split halves,
permuted copies, subsamples. A piece has fewer rows than the full data, and
with weights its own `n_eff` can be much smaller. The full-data check has
already established that your `(weights, k)` request is feasible, so a
piece with low `n_eff` is sampling variance, not a user error. How each
loop treats it:

- **CV folds, split halves, permutation refits, and the K-selection sweeps**
  do not check. The fit on a low-`n_eff` piece is overfit but valid, and
  its noise is absorbed by the statistic being averaged (a noisier fold
  R² or split correlation).
- **Subsamples in the `ci` branch of `pls1_confirmatory_test`** do check, per subsample. A subsample that
  fails is skipped, and the skips are counted. If the fraction of skipped
  subsamples exceeds `max_skip_rate` (default `0.01`), the call errors
  with `ResamplingDegenerate` rather than return a CI built from the
  surviving, unrepresentative draws.
  A replicate's half split (for `holdout_corr`) whose fit fails the same check, or has all its weights zero, costs that replicate its `holdout_corr` only, and counts as a skip too.

## What users see

- **`PlsKitInvalidWeights` with `reason="insufficient_effective_n"`**
  (error code `invalid_weights`) from a top-level call with non-uniform
  weights: your `(weights, k)` combination is infeasible. The effective
  sample size is too small for the number of components requested.
  Actions: lower `k` (or `k_max`), flatten extreme weights, or add
  observations.

- **The same error from the cross-validated selection** (`pls1_find_k_optimal` and `spls1_find_k_optimal` with `r2_se` or `r2_max`, `spls1_find_keep_optimal`, `pls1_fit(k="optimal")`) can also mean that all the positive weight sits in one CV fold, so no fold has weight on both its training and its validation rows. Lowering `k` does not help there. The same holds for the folds of `raw_perm` in `pls1_confirmatory_test`, for its `split_exact` and `split_nb` when fewer than 2 of the drawn splits have weight on both halves (also in the steps of a `k` sequence that run them), and for its `e` when a half of its one split has zero total weight.

- **`PlsKitError` with code `invalid_argument`** ("insufficient n for
  k=...") from a top-level call without weights, or with all-equal weights
  (which are no weights: the call is identical to one without them): the
  same check, but with no weights in play it is a plain data-size problem,
  `n < k + 1`. Lower `k` or add observations.

- **`PlsKitResamplingDegenerate`** (error code `resampling_degenerate`)
  from `pls1_confirmatory_test(ci=True)`:
  your weights are healthy on the full data, but too many subsamples
  failed individually (each subsample draws fewer rows, so its `n_eff` is
  smaller). A half split whose fit fails on the weights counts as a skip of its `holdout_corr`, so raising `max_skip_rate` covers it too. The exception carries `skipped`, `total`, `skip_rate` and
  `threshold`. The call also returns it when no draw is left after skips and failures, and raising `max_skip_rate` does not help then. The same holds when every draw left has a NaN holdout correlation. Actions: lower `k`, soften the weight distribution, raise
  `m_rate` to draw larger subsamples (`m = ceil(n^m_rate)`), or raise
  `max_skip_rate` to accept the dropped draws knowingly.

The wrappers and the Rust API always run the check. How the check is wired through the engine is described in [the n_eff check (internals)](../internals/n-eff-check.md).
