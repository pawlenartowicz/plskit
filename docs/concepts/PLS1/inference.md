# Inference

> Status: placeholder. The full treatment will land with publication of
> the methods paper introducing `split_nb` and `split_exact`. Until then,
> see the [Python API → pls1_confirmatory_test](../../python/api.md) for
> the implemented surface.

## A set of `k`

`pls1_confirmatory_test` tests one `k` or a set of `k`. With a set, it asks whether `X` predicts `y` at some `k` in the set and returns one p-value for the whole set. The set must be fixed before the test sees `y`, as a single `k` must: a set or a `k` chosen from `y` on the same rows makes the test exploratory.

A test at one `k` is weak when the signal needs another `k`. `k = 1` has power only when one component already predicts `y`, and on spectra the signal often needs several components. A larger fixed `k` loses power where one component is enough. A set covers both without choosing `k` from `y`.

**The default set** is `1..8`, cut by the design: at `min(8, p, ⌊n/2⌋ − 2)`, then lowered until the weights' Kish effective sample size and the covariates' residual degrees of freedom both stay at least `k + 1` (`n_eff ≥ k + 1`, and `n_pos − rank([1, Z]) ≥ k + 1` with `n_pos` the rows with positive weight), never below 1. `⌊n/2⌋ − 2` is the largest `k` whose split halves have the same sizes as at `k = 1`. The cut applies only to the default: an explicit set is sorted, and an empty set, a duplicate, a `k` of 0 or a set past a limit of the method (`k ≤ p`, the split sizes, `k ≤ 10` with `gram`) is an error, never cut.

**Methods.** `split_exact` and `raw_perm` take a set. `split_nb`, `score`, `sv_perm`, `e` and `ci=True` take one `k` (in Rust also `keep`): there the default is `{1}`, and an explicit set of more than one `k` raises `invalid_argument`. `split_nb` tests at one fixed `k`, the `score` statistic does not depend on `k`, and `e` has no combination over a set. With more than one `k`, the default set included, `"auto"` runs `split_exact` (see [`test_method="auto"`](#test_methodauto)). Over a set `split_exact` refits for every permutation, up to the largest `k`, so on data where `"auto"` with one `k` picks `split_nb` the default call costs much more. `k=1, test_method="split_nb"` is the fast choice there.

**The combination (min-p).** `split_exact` and `raw_perm` draw their splits or folds once, for the largest `k`, and keep them for every `k` and every null column. On the observed `y` and on each of the `n_perm` null columns they compute the method's statistic at every `k` in the set, from one fit per split or fold and column up to the largest `k`. Each `k` gets its own permutation p over the `n_perm + 1` columns, and each column's smallest p over the set is its min-p. The reported p is the share of columns whose min-p is at or below the observed one (Westfall and Young 1993, *Resampling-Based Multiple Testing*). When `y` has no relation to `X` the columns are exchangeable, so this p is exact, as at one `k`. With one `k` it is the usual permutation p. A null statistic that is not finite counts as an exceedance, as at one `k`. For `raw_perm`, a `k` at which more than half the null fits fail fails the call (`perm_null_degenerate`), as a single call at that `k` does.

**The result.** `result.k` is the tested set, sorted, always a list (one `k` gives a one-element list). `result.pvalue` is the p for the whole set. `result.statistic` is set for one `k` and null for a set of more than one. `result.pvalues_by_k` and `result.statistics_by_k` map each tested `k` to its own p and its own statistic; a per-`k` p is not adjusted for the set.

**No `k` to fit at.** The test answers yes or no; it names no `k` to fit at. The `k` with the smallest per-`k` p is a poor `k` to predict with: it tends to be smaller than the `k` cross-validation picks, and to predict worse on new rows. Make the claim with the test and fit the model with `pls1_fit`, which chooses its own `k` ([Find K](find-k.md#from-pls1_fit)), and pass no `k` from one to the other:

```python
r = plskit.pls1_confirmatory_test(X, y)   # is there signal at some k in 1..8?
fit = plskit.pls1_fit(X, y)               # the model, k chosen by cross-validation
```

## `test_method="auto"`

`test_method="auto"` is the default of `pls1_confirmatory_test`,
`pls3_confirmatory_test`, `pls1_find_k_sequence` and
`spls1_find_k_sequence`. It picks `split_exact` or `split_nb` once per
call. The choice uses only `X` and the weights, never `y`.
`result.test_method` reports the method that ran; it is never `"auto"`.

In `pls3_confirmatory_test`, and in `pls1_confirmatory_test` with a set of more than one `k` (the default set included), `"auto"` runs `split_exact` on every design. Elsewhere it runs `split_exact` when any of these holds, and `split_nb` otherwise:

1. The `split_nb` auto-gate fires: `X` has 4 columns or fewer, `n_eff < 25`, or the stable rank of `X` as the tests read it (standardized once over all rows, or only centred under `pre_standardized`) is `< 3`. Purpose: validity. `split_nb` is not reliable on such designs.
2. The Kish effective sample size `n_eff` (the row count when there are no weights) is at or below `1000` at `k = 1` without `keep`, or below `250` otherwise. Purpose: power. On smaller samples `split_nb` can lose power against `split_exact`, and at `k = 1` without `keep` `split_exact` is cheap enough to keep up to `n_eff = 1000`.
3. At `k = 1` without `keep`, the number of columns exceeds `100` times
   the number of rows. Purpose: cost. There `split_exact` costs about as
   much as `split_nb` on very wide `X`. With `k ≥ 2` or a sparse `keep`,
   `split_exact` refits for every permutation and costs more, so this clause
   does not apply. In `pls1_find_k_sequence`
   every step tests at `k = 1`, so the clause applies; in
   `spls1_find_k_sequence` it applies only at `keep = n_features`, which
   runs the dense path.

These thresholds may change between versions.

Above the floor, `"auto"` takes `split_nb`. On the designs measured it lost a little power against `split_exact` and took about a tenth of the time at `k = 1` and a fortieth at `k ≥ 2`. A caller who wants the power passes `test_method="split_exact"`.

The `split_nb` standard error is too large where the correlation between its splits is below 1/2. That holds on every `pls3_confirmatory_test` design measured, and in `pls1_confirmatory_test` on `X` of low effective rank. Under the null its rejection rate at α = 0.05 is 0 to 2.5% for `pls3_confirmatory_test` at `n ≥ 100` with `p ≥ n/2`, and 4.0% to 6.2% on `X` of low effective rank (PLS1 and PLS3).

`args` under `"auto"` takes `n_perm` and `n_splits`, with the `split_exact` defaults (`1000` and `32`). `n_perm` is checked on every call but ignored when `split_nb` is chosen, and `result.n_perm` is then `None`. `force` is not accepted. `result.stable_rank` is set on an explicit `split_nb` request, and on an `"auto"` request with one `k` that reached the stable-rank check (p > 4, `n_eff` above 1000 at `k = 1` without `keep`, at least 250 otherwise, and p ≤ 100·n where clause 3 applies); `None` otherwise, and on every `"auto"` request of `pls3_confirmatory_test`. It is NaN when the Gram matrix of `X` so read is not finite. The gate then fires.

To repeat a run with the method that ran, pass `test_method=result.test_method, seed=result.seed` and the same `k` and `args`. When `split_nb` ran, drop `n_perm` from `args`: an explicit `split_nb` call rejects it. The statistic and p-value match the `"auto"` call at the same seed, but `stable_rank` can differ, because an explicit `split_exact` leaves it unset.

## Covariates and `null_y`

`split_exact` and `raw_perm` take `covariates` and use Freedman–Lane null columns; `"auto"` then runs `split_exact`. `split_nb`, `score` without `n_perm` and `e` refuse them; `score` with `n_perm` and `sv_perm` take them (Freedman–Lane). `null_y` replaces the permutation nulls of `split_exact` with null outcomes you supply. See [Covariates](covariates.md).

## `raw_perm`'s CV R²

`raw_perm` reads `X` scaled once over all rows, held-out rows included, and each fold only centres. Its CV R² is therefore slightly optimistic at small n, against folds that scale with their own training rows: +0.034 at n = 40, p = 20 in simulation. The permutation p-value stays exact, since the scaling reads `X` only, never `y`.

Topics this page will cover:

- Confirmatory vs exploratory: why a `k`, or a set of `k`, chosen on the same data makes the test exploratory
- The confirmatory test methods: `split_exact`, `split_nb`, `raw_perm`, `score`, `sv_perm`, `e`
- The recommended method: `split_exact`, a split-half test calibrated by permutation, so it holds its level on any design (`test_method` defaults to `"auto"`, which runs `split_exact` over a set of `k`, the default set included, and chooses between `split_exact` and `split_nb` at one `k`; see [`test_method="auto"`](#test_methodauto))
- `split_nb` as the faster asymptotic alternative: same split-half statistic, calibrated by a Fisher-z correction instead of permutation. Appropriate when `n` is large relative to `p` and `X`'s spectrum is flat. Designs where `n_eff < 25`, where `X` has 4 columns or fewer, or where the stable rank of `X` is `< 3` are auto-gated: an explicit `split_nb` request on such a design reroutes to `split_exact` (at `n_perm=1000`) unless the caller passes `args={"force": True}`. `split_nb_gate` reports that decision, plus the `stable_rank` and `n_eff` behind it, without running a test
- Power vs validity tradeoffs across methods
- The split-half construction underlying `split_nb` / `split_exact`, and the K = 1 identity that lets `split_exact`'s no-refit route permute without refitting
- Universal inference (`e`-values) and when it is the right tool
- Closed-form `score` test (PLS1 single-`y` only)
