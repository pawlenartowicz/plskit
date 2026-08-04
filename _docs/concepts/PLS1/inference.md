# Inference

> Status: placeholder. The full treatment will land with publication of
> the methods paper introducing `split_nb` and `split_exact`. Until then,
> see the [Python API → pls1_confirmatory_test](../../python/api.md) for
> the implemented surface.

Topics this page will cover:

- Confirmatory vs exploratory: why `pls1_confirmatory_test` refuses a `k` chosen on the same data
- The five confirmatory test methods: `split_exact`, `split_nb`, `raw_perm`, `score`, `e`
- The recommended default: `k=1` with `split_exact` — a split-half test calibrated by permutation, so it holds its level on any design
- `split_nb` as the faster asymptotic alternative: same split-half statistic, calibrated by a Fisher-z correction instead of permutation. Appropriate when `n` is large relative to `p` and `X`'s spectrum is flat. Designs where `n_eff < 25`, where `X` has 4 columns or fewer, or where the stable rank of `X` is `< 3` are auto-gated: a `split_nb` request on such a design reroutes to `split_exact` (at `n_perm=1000`) unless the caller passes `force=True`. `split_nb_gate` reports that decision, plus the `stable_rank` and `n_eff` behind it, without running a test
- Power vs validity tradeoffs across methods
- The split-half construction underlying `split_nb` / `split_exact`, and the K = 1 identity that lets `split_exact`'s no-refit route permute without refitting
- Universal inference (`e`-values) and when it is the right tool
- Closed-form `score` test (PLS1 single-`y` only)
