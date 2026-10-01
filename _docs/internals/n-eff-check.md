# The n_eff check: implementation

Contributor notes on how the `n_eff < k + 1` check is wired through
`plskit-rs`. The user-facing behaviour (what `n_eff` is, which public
functions check it, which errors users see) is in
[Effective sample size](../concepts/effective-sample-size.md); this page
covers the internal call sites.

## The pieces

All in `plskit-rs/src/fit.rs`:

- `validate_and_normalize_weights(weights, n, k_requested)` validates
  length, finiteness and sign, normalizes to mean 1, and computes `n_eff`
  with `linalg::compute_n_eff` on the raw weights. All-equal weights
  (every normalized entry 1 within `1e-12`) come back as `None` with
  `n_eff = n` exactly, so every downstream `w_norm.is_some()` branch
  (standardization, route choice, `rho_hat`, the `split_nb` gate) takes the
  unweighted path and the call is bit-identical to one without weights. It
  does **not** check `n_eff` against `k`; its `k_requested` argument is
  unused (see the history note below).
- `validate_weights_for_k(weights, n, k)` is that validation followed by the
  check, and the one call a top-level entry makes. The private
  `check_n_eff_for_k(n_eff, k, weighted)` returns
  `InvalidWeights { reason: "insufficient_effective_n" }` for non-uniform
  weights, and `InvalidArgument("insufficient n for k=...")` for uniform or
  absent ones (no weights in play, so it is a data-size error, and callers
  branch on `code()`). `weighted` is read off the normalized vector, never
  off the caller's `Option`.
- `FitOpts::check_n_eff: bool` (default `true`) controls whether
  `pls1_fit` calls `validate_weights_for_k` or the bare validation. The same flag also gates the
  truncation guard for `pre_standardized: true` fits: with both flags set,
  a NIPALS short-circuit below the requested `k` returns `InvalidInput`;
  with `check_n_eff: false` the truncated model is returned and callers
  read `k_used`.

`check_n_eff` is the only Rust-only knob here that is public; the wrappers
never set it. The Rust bypass is documented in
[`rust/api.md`](../rust/api.md#bypassing-the-n_eff-check-rust-only).

## Top-level entries

Every public entry that takes weights and a component count calls
`validate_weights_for_k` on the full data, before any fitting:

| Entry | File | Checked against |
|---|---|---|
| `pls1_fit` (and `spls1_fit`, which delegates) | `fit.rs` | `k`, only when `opts.check_n_eff` |
| `pls1_confirmatory_test` | `signal_test.rs` (`confirmatory_test_impl`) | resolved `k` |
| `pls1_perm_null` | `perm_null.rs` | `k` |
| `pls1_rotation_stability` | `rotation_stability.rs` | `k` |
| `pls1_find_k_optimal`, `spls1_find_k_optimal` | `find_k.rs` (`find_k_optimal_impl`) | `k_max` |
| `pls1_find_k_sequence`, `spls1_find_k_sequence` | `find_k.rs` (`find_k_sequence_impl`) | `k_max` |
| `spls1_find_keep_optimal` | `find_k.rs` | `k` |

`split_nb_gate` calls `validate_and_normalize_weights` with
`k_requested = 0` and runs no check; it has no `k`.
`preprocess` computes `n_eff` without a check for the same reason. The
PLS3 family refuses weights, so `n_eff = n` and there is no check.

## Internal call sites that turn the check off

These call `pls1_fit` with `check_n_eff: false`. The full-data check has
already run at the entry, and each site tolerates a low-`n_eff` or
truncated fit by design (the comment at each call site says why):

- `signal_test.rs`: `pls1_cv_r2` CV folds (`raw_perm`);
  `split_half_correlations` per-half fits (`split_nb`, `split_exact` refit
  route; these pass `weights = None` because the weights are already baked
  into the row-scaled data, so the check could not fire anyway); the
  `e`-value train-half refit in `run_e`.
- `find_k.rs`: `select_cv` fold fits; the `select_bic` full-`k_max` fit
  (the BIC sweep is truncation-tolerant); the `first_component_exhausted`
  full-data `k = 1` fit ahead of the CV selectors (a truncation is the
  answer it looks for, not an error); the `spls1_find_keep_optimal`
  fold fits.
- `perm_null.rs`: the reference fit and every per-permutation refit.
- `sequential.rs`: the deflation refit in the step-down sequence.

The dual (Gram) routes in `dual_route.rs` (`pls1_cv_r2_columns`,
`pls3_split_zbars_columns`, and the n-space PLS1 kernel
`pls1_nspace_kernel`, whose unit bodies `nspace_perm_row`,
`fold_column_nspace` and `split_column_r_nspace` serve `pls1_perm_null`,
raw_perm at `k ≥ 2` and the split_exact refit route at `k ≥ 2`) do not
call `pls1_fit` and run no check, matching the primal sites they replace.
Their primal fallbacks run with the check off, as those sites do.

The p-space Gram backend in `gram_p.rs` (`GramPBlock::fit_replicate`,
behind `pls1_perm_null`, raw_perm and the split_exact refit route on tall
data, where `n` is well above `p`) does not call `pls1_fit` and runs no
check either. A replicate it cannot certify falls back to the primal unit
of its site, which runs with the check off, as those sites do.

## Internal call sites that keep the check on

- **Reference fits** in the `ci` branch of `pls1_confirmatory_test`
  (`signal_test.rs`) and in `pls1_rotation_stability` use default
  `FitOpts`. On the full data this repeats the entry check and cannot fail.
- **CI subsamples** (`subsample.rs`, `run_one_confirmatory`): the
  per-subsample fit uses default `FitOpts`, so the check fires on the
  subsample's own `n_eff`. The driver
  (`pls1_subsample_inference_confirmatory`) maps
  `InvalidWeights { reason: "insufficient_effective_n" }` to
  `WorkerOutcome::Skipped` and every other error to `Failed`. Skips are
  compared with `max_skip_rate` (`ResamplingDegenerate`), failures with
  `max_failure_rate` (`ResampleFailureRateExceeded`). Without weights the
  check cannot fire here, because the engine rejects `m < k + 2` up front.
- **Rotation-stability subsamples** (`rotation_stability.rs`,
  `run_one_rotation_stability`): also default `FitOpts`. Any worker error,
  `n_eff` or numerical, becomes a NaN row; the NaN-row filter counts them
  as skips against `max_skip_rate`. This path deliberately does not split
  skips from failures. (The comment above the subsample's
  `validate_and_normalize_weights` call says it "propagates InvalidWeights
  (e.g. n_eff_sub < k+1)"; that function no longer checks `n_eff`, and the
  error actually comes from the `pls1_fit` call below it.)

## History note

Earlier versions enforced the `n_eff` check inside
`validate_and_normalize_weights`, which runs at every call site,
including per-fold CV. That coupled two distinct failure modes: "user
request infeasible on the full dataset" and "one unlucky fold has low
`n_eff` by construction." The current contract separates them:
`validate_and_normalize_weights` no longer checks `n_eff`, and
`validate_weights_for_k` is called explicitly at the top-level entries (and
from `pls1_fit` via the `FitOpts::check_n_eff` flag).
