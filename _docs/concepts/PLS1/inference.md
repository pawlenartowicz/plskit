# Inference

> Status: placeholder. The full treatment will land with publication of
> the methods paper introducing `split_nb` and `split_exact`. Until then,
> see the [Python API → pls1_confirmatory_test](../../python/api.md) for
> the implemented surface.

## `test_method="auto"`

`test_method="auto"` is the default of `pls1_confirmatory_test`,
`pls3_confirmatory_test`, `pls1_find_k_sequence` and
`spls1_find_k_sequence`. It picks `split_exact` or `split_nb` once per
call. The choice uses only `X` and the weights, never `y`.
`result.test_method` reports the method that ran; it is never `"auto"`.

`"auto"` runs `split_exact` when any of these holds, and `split_nb`
otherwise:

1. The `split_nb` auto-gate fires: `X` has 4 columns or fewer, `n_eff < 25`,
   or the stable rank of the standardized `X` is `< 3`. Purpose: validity.
   `split_nb` is not reliable on such designs.
2. `n_eff < 250` (the Kish effective sample size; the row count when there
   are no weights). Purpose: power. On smaller samples `split_nb` can lose
   power against `split_exact`.
3. At `k = 1` without `keep`, the number of columns exceeds `100` times
   the number of rows. Purpose: cost. There `split_exact` costs about as
   much as `split_nb` on very wide `X`. With `k ≥ 2`, a sparse `keep`, or in
   `pls3_confirmatory_test`, `split_exact` refits for every permutation and
   costs more, so this clause does not apply. In `pls1_find_k_sequence`
   every step tests at `k = 1`, so the clause applies; it does not apply in
   `spls1_find_k_sequence`, which always has a `keep`.

These thresholds may change between versions.

`args` under `"auto"` takes `n_perm` and `n_splits`, with the `split_exact`
defaults (`1000` and `50`). `n_perm` is checked on every call but ignored
when `split_nb` is chosen, and `result.n_perm` is then `None`. `force` is
not accepted. `result.stable_rank` is set on an explicit `split_nb`
request, and on an `"auto"` request that reached the stable-rank check
(p > 4, n_eff ≥ 250, and p ≤ 100·n where clause 3 applies); `None` otherwise.

To repeat a run with the method that ran, pass
`test_method=result.test_method, seed=result.seed` and the same `args`. When
`split_nb` ran, drop `n_perm` from `args`: an explicit `split_nb` call
rejects it. The statistic and p-value match the `"auto"` call at the same
seed, but `stable_rank` can differ, because an explicit `split_exact` leaves
it unset.

Topics this page will cover:

- Confirmatory vs exploratory: why `pls1_confirmatory_test` takes only an explicit integer `k`, and why a `k` chosen on the same data makes the test exploratory
- The five confirmatory test methods: `split_exact`, `split_nb`, `raw_perm`, `score`, `e`
- The recommended choice: `k=1` with `split_exact`, a split-half test calibrated by permutation, so it holds its level on any design (`test_method` defaults to `"auto"`, which chooses between `split_exact` and `split_nb`; see [`test_method="auto"`](#test_methodauto))
- `split_nb` as the faster asymptotic alternative: same split-half statistic, calibrated by a Fisher-z correction instead of permutation. Appropriate when `n` is large relative to `p` and `X`'s spectrum is flat. Designs where `n_eff < 25`, where `X` has 4 columns or fewer, or where the stable rank of `X` is `< 3` are auto-gated: an explicit `split_nb` request on such a design reroutes to `split_exact` (at `n_perm=1000`) unless the caller passes `args={"force": True}`. `split_nb_gate` reports that decision, plus the `stable_rank` and `n_eff` behind it, without running a test
- Power vs validity tradeoffs across methods
- The split-half construction underlying `split_nb` / `split_exact`, and the K = 1 identity that lets `split_exact`'s no-refit route permute without refitting
- Universal inference (`e`-values) and when it is the right tool
- Closed-form `score` test (PLS1 single-`y` only)
