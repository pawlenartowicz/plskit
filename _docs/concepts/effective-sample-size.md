# Effective sample size and the n_eff check

## What `n_eff` is

`n_eff` is Kish's effective sample size: `(Σwᵢ)² / Σwᵢ²`, computed on
the raw (pre-normalization) weight vector. For uniform or absent weights,
every term equals 1 and `n_eff = n`.

`n_eff` captures how much information the weighted sample contains: a
sample where one observation has all the weight has `n_eff ≈ 1`; a
sample with perfectly balanced weights has `n_eff = n`. The check
`n_eff < k + 1` is the PLS analogue of the OLS rule that you need at
least as many observations as parameters. With weighted data, the
relevant count is the effective number of observations, not the raw
count.

Every PLS1 and sPLS1 result reports the `n_eff` it was computed with, and
`preprocess` and `split_nb_gate` report it too.

## Where the check fires

The check runs once, on the full data you pass in, at the start of each
public function that fits PLS1 components. If it fails, the function
errors before any fitting.

| Function | Checked against |
|---|---|
| `pls1_fit`, `spls1_fit` | `k` |
| `pls1_confirmatory_test` (every method, with or without `ci`) | `k` |
| `pls1_perm_null` | `k` |
| `pls1_rotation_stability` | `k` |
| `spls1_find_keep_optimal` | `k` |
| `pls1_find_k_optimal`, `pls1_find_k_sequence`, `spls1_find_k_optimal`, `spls1_find_k_sequence` | `k_max` |

The K-selection functions check against `k_max`, the largest model they
will try, so a feasible `k_max` guarantees every candidate `k` is feasible
on the full data.

No check runs in `preprocess` and `split_nb_gate` (they take no `k`), in
`pls1_predict` and `rotate` (they take no weights), or in the PLS3 family
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
- **Subsamples in the `ci` branch of `pls1_confirmatory_test` and in
  `pls1_rotation_stability`** do check, per subsample. A subsample that
  fails is skipped, and the skips are counted. If the fraction of skipped
  subsamples exceeds `max_skip_rate` (default `0.01`), the call errors
  with `ResamplingDegenerate` rather than return a CI built from the
  surviving, unrepresentative draws. (In `pls1_rotation_stability`, any
  failed subsample counts as a skip, not only an `n_eff` failure.)

## What users see

- **`PlsKitInvalidWeights` with `reason="insufficient_effective_n"`**
  (error code `invalid_weights`) from a top-level call with non-uniform
  weights: your `(weights, k)` combination is infeasible. The effective
  sample size is too small for the number of components requested.
  Actions: lower `k` (or `k_max`), flatten extreme weights, or add
  observations.

- **`PlsKitError` with code `invalid_argument`** ("insufficient n for
  k=...") from a top-level call without weights, or with all-equal weights
  (which are no weights: the call is identical to one without them): the
  same check, but with no weights in play it is a plain data-size problem,
  `n < k + 1`. Lower `k` or add observations.

- **`PlsKitResamplingDegenerate`** (error code `resampling_degenerate`)
  from `pls1_confirmatory_test(ci=True)` or `pls1_rotation_stability`:
  your weights are healthy on the full data, but too many subsamples
  failed individually (each subsample draws fewer rows, so its `n_eff` is
  smaller). The exception carries `skipped`, `total`, `skip_rate` and
  `threshold`. Actions: lower `k`, soften the weight distribution, raise
  `m_rate` to draw larger subsamples (`m = ceil(n^m_rate)`), or raise
  `max_skip_rate` to accept the dropped draws knowingly.

The wrappers always run the check. Rust callers can turn it off for
`pls1_fit` / `spls1_fit`; see
[Bypassing the `n_eff` check](../rust/api.md#bypassing-the-n_eff-check-rust-only).
How the check is wired through the engine is described in
[the n_eff check (internals)](../internals/n-eff-check.md).
