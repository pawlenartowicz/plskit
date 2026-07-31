# Inference

> Status: placeholder. The full treatment will land with publication of
> the methods paper introducing `split_nb` and `split_perm`. Until then,
> see the [Python API → pls1_confirmatory_test](../../python/api.md) for
> the implemented surface.

Topics this page will cover:

- Confirmatory vs exploratory: why `pls1_confirmatory_test` refuses a `k` chosen on the same data
- The six confirmatory test methods: `split_nb`, `split_perm`, `split_perm_nr`, `raw_perm`, `score`, `e`
- Power vs validity tradeoffs across methods
- The split-half construction underlying `split_nb` / `split_perm` / `split_perm_nr`, and the K = 1 identity that lets `split_perm_nr` permute without refitting
- Universal inference (`e`-values) and when it is the right tool
- Closed-form `score` test (PLS1 single-`y` only)
