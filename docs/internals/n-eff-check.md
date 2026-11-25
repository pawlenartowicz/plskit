# The n_eff check: implementation

Contributor notes on how the `n_eff < k + 1` check is wired through
`plskit-rs`. The user-facing behaviour (what `n_eff` is, which public
functions check it, which errors users see) is in
[Effective sample size](../concepts/effective-sample-size.md); this page
covers the internal call sites.

## The pieces

All in `plskit-rs/src/fit.rs`:

- `validate_and_normalize_weights(weights, n)` validates
  length, finiteness and sign, normalizes to mean 1, and computes `n_eff`
  with `linalg::compute_n_eff` on the mean-one weights. All-equal weights
  (every normalized entry 1 within `1e-12`) come back as `None` with
  `n_eff = n` exactly, so every downstream `w_norm.is_some()` branch
  (standardization, route choice, `rho_hat`, the `split_nb` gate) takes the
  unweighted path and the call is bit-identical to one without weights. It
  does **not** check `n_eff` against `k` (see the history note below).
- `validate_weights_for_k(weights, n, k)` is that validation followed by the
  check, and the one call a top-level entry makes. The private
  `check_n_eff_for_k(n_eff, k, weighted)` returns
  `InvalidWeights { reason: "insufficient_effective_n" }` for non-uniform
  weights, and `InvalidArgument("insufficient n for k=...")` for uniform or
  absent ones (no weights in play, so it is a data-size error, and callers
  branch on `code()`). `weighted` is read off the normalized vector, never
  off the caller's `Option`.
- `check_n_eff: bool` (default `true`) is a field of the crate-internal `InternalFitOpts`, the second options argument of `pls1_fit_impl`, not of the public `FitOpts`. It controls whether the fit calls `validate_weights_for_k` or the bare validation. The same flag also gates the truncation guard for `pre_standardized: true` fits: with both flags set, a NIPALS short-circuit below the requested `k` returns `InvalidInput`; with `check_n_eff: false` the truncated model is returned and callers read `k_used`.

`pls1_fit` and `spls1_fit` pass `check_n_eff: true`, so the public API and the wrappers always run the check; only the crate's internal call sites turn it off.

## Top-level entries

Every public entry that takes weights and a component count calls
`validate_weights_for_k` on the full data, before any fitting:

| Entry | File | Checked against |
|---|---|---|
| `pls1_fit` (and `spls1_fit`, which delegates) | `fit.rs` | `k` |
| `pls1_confirmatory_test` | `signal_test.rs` (`confirmatory_test_run`) | the largest `k` of the set |
| `pls1_perm_null` | `perm_null.rs` | `k` |
| `pls1_find_k_optimal`, `spls1_find_k_optimal` | `find_k.rs` (`find_k_optimal_impl`) | `k_max` |
| `pls1_find_k_sequence`, `spls1_find_k_sequence` | `find_k.rs` (`find_k_sequence_impl`) | `k_max` |
| `spls1_find_keep_optimal` | `find_k.rs` | `k` |

`split_nb_gate` calls `validate_and_normalize_weights` and runs no check;
it has no `k`.
`preprocess` computes `n_eff` without a check for the same reason. The
PLS3 family refuses weights, so `n_eff = n` and there is no check.

## Internal call sites that turn the check off

These call `pls1_fit_impl` with `check_n_eff: false`. The full-data check has
already run at the entry, and each site tolerates a low-`n_eff` or
truncated fit by design (the comment at each call site says why):

- `signal_test.rs`: the `e`-value train-half refit in `run_e`. The per-half fits of `split_nb` (`split_column_r`, at `k ≥ 2` or with `keep` set) do not call `pls1_fit`: they run its kernel tail, `fit::pls1_fit_prepared_fro`. The `raw_perm` CV folds (`fold_columns_lanes`) and the `split_exact` refit route (`split_columns_lanes`) fit their columns in batches through `fit::lanes::pls1_fit_lanes`. Neither has the check, and neither takes weights (the weights are already in the row-scaled data).
- `find_k.rs`: `select_cv` fold fits; the `select_bic` full-`k_max` fit
  (the BIC sweep is truncation-tolerant); the `first_component_exhausted`
  full-data `k = 1` fit ahead of the CV selectors (a truncation is the
  answer it looks for, not an error); the `spls1_find_keep_optimal`
  fold fits.
- `perm_null.rs`: the reference fit and every per-permutation refit.
- `sequential.rs`: the deflation refit in the step-down sequence.

The dual (Gram) routes in `dual_route.rs` (`pls3_split_zbars_columns`, and the n-space PLS1 kernel `pls1_nspace_kernel`, whose unit bodies `nspace_perm_row`, `fold_column_nspace` and `split_columns_r_nspace` serve `pls1_perm_null`, raw_perm at every `k` and the split_exact refit route at `k ≥ 2`) do not call `pls1_fit` and run no check, matching the primal sites they replace.

The p-space Gram backend in `gram_p.rs` (`GramPBlock::fit`, behind
`pls1_perm_null`, raw_perm and the split_exact refit route on tall data,
where `n` is well above `p`) does not call `pls1_fit` and runs no check
either, matching the primal units of its sites, which run with the check
off. It decides every replicate itself; nothing is refitted on X.

## Internal call sites that keep the check on

- **Reference fits** in the `ci` branch of `pls1_confirmatory_test`
  (`signal_test.rs`) leaves
  `check_n_eff` at its default. On the full data this repeats the entry check and cannot fail.
- **CI subsamples** (`subsample.rs`, `run_one_confirmatory`): the
  per-subsample fit leaves `check_n_eff` at its default, so the check fires on the
  subsample's own `n_eff`. The driver
  (`pls1_subsample_inference_confirmatory`) maps `InvalidWeights` with reason `"insufficient_effective_n"` or `"all_zero"` (a subsample whose weights are all zero) to `WorkerOutcome::Skipped` and every other error to `Failed`. `run_one_confirmatory` marks a half split whose fit fails with those reasons as `holdout_skipped`: the replicate keeps its β and leverage rows, and its NaN `holdout_corr` counts as a skip. Skips are compared with `max_skip_rate` (`ResamplingDegenerate`) and dropped; failures alone are compared with `max_failure_rate` (`ResampleFailureRateExceeded`). When no draw is left after skips and failures, or every draw left has a NaN holdout correlation, the call returns `ResamplingDegenerate` whatever the two rates. Without weights the check cannot fire here, because the engine rejects `m < k + 2` up front.

## History note

Earlier versions enforced the `n_eff` check inside
`validate_and_normalize_weights`, which runs at every call site,
including per-fold CV. That coupled two distinct failure modes: "user
request infeasible on the full dataset" and "one unlucky fold has low
`n_eff` by construction." The current contract separates them:
`validate_and_normalize_weights` no longer checks `n_eff`, and
`validate_weights_for_k` is called explicitly at the top-level entries (and
from `pls1_fit_impl` via the internal `check_n_eff` flag).
