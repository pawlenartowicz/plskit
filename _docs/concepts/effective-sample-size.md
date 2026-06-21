# Effective sample size and the n_eff check

## What `n_eff` is

`n_eff` is Kish's effective sample size: `(Σwᵢ)² / Σwᵢ²`, computed on
the raw (pre-normalization) weight vector. For uniform or absent weights,
every term equals 1 and `n_eff = n`.

`n_eff` captures how much information the weighted sample contains: a
sample where one observation has all the weight has `n_eff ≈ 1`; a
sample with perfectly balanced weights has `n_eff = n`. The check
`n_eff < k + 1` is the PLS analogue of the OLS rule that you need at
least as many observations as parameters — with weighted data, the
relevant count is the effective number of observations, not the raw
count.

## The contract: where the check fires

Top-level public entries run the check and error early when `n_eff <
k + 1`. This is a user-error signal: the `(weights, k)` combination is
infeasible on the full dataset as presented.

Per-iteration internal call sites — CV folds, split-halves, K-selector
folds — do not run the check. They let the NIPALS math proceed (NIPALS
produces overfit-but-valid coefficients for low-`n_eff` slices), and
rely on the parent statistic to absorb noise. A fold whose training
slice has low `n_eff` contributes a noisier fold R² or split-r;
that is sampling variance, not user error.

The mechanism: `FitOpts` carries a `check_n_eff: bool` field (default
`true`). Top-level entry points call `pls1_fit` with the default.
Per-iteration internal call sites set `check_n_eff: false` explicitly.

## Which top-level entries enforce the check

- `pls1_fit` — directly, via `check_n_eff_for_k` when `opts.check_n_eff
  = true` (the default).
- `pls1_perm_null` — transitively: its reference fit uses default
  `FitOpts`. The per-permutation refits also use default `FitOpts`, but
  they refit on permuted y with the same weights and n_eff as the
  reference, so the check is redundant-but-harmless there.
- `pls1_confirmatory_test` — transitively via the reference fit in the
  CI branch (`opts.ci = Some(...)`). The raw-data entry validation path
  goes through `validate_and_normalize_weights`, which checks dimension
  and sign but not n_eff; the reference fit's `check_n_eff = true`
  covers the n_eff gate.
- `pls1_find_k_optimal` / `pls1_find_k_sequence` — transitively via
  `validate_and_normalize_weights` at the top of each function (for the
  full-data n_eff gate) and via the BIC reference fit (`select_bic`),
  which uses default `FitOpts`.
- `pls1_rotation_stability` — transitively: the per-subsample call in
  `run_one_rotation_stability` goes through `validate_and_normalize_weights`
  directly (not via `pls1_fit`), so the n_eff check there fires on the
  subsample's own n_eff and is caught by the surrounding skip-rate
  accumulator (see below).

## Where the check is intentionally absent

**CV folds in `pls1_cv_r2` and `select_cv`.**  Each training fold is a
subset of the full data with its own re-normalized weights. Its n_eff is
smaller than the full n_eff. Requiring `n_eff_fold ≥ k + 1` would be
overly strict — the full-data gate already ensured feasibility. These
call sites set `check_n_eff: false`.

**Split-halves in `split_half_correlations`.**  Splits pass `weights =
None` (weights are baked into the row-scaled data before splitting), so
the check cannot fire regardless.

**Per-fold fits in K-selector CV (`select_cv` inside
`pls1_find_k_optimal`).**  Same reasoning as `pls1_cv_r2`. These set
`check_n_eff: false`.

**Bootstrap subsamples in `subsample.rs`.**  The per-subsample fit
calls `pls1_fit` with default `FitOpts` (check enabled), but the
surrounding loop catches `InvalidWeights { reason:
"insufficient_effective_n" }` explicitly and converts it to a soft skip.
Skips accumulate into a skip-rate; if `skip_rate > max_skip_rate`, the
loop escalates to `ResamplingDegenerate`. These call sites are left at
the default.

**Bootstrap subsamples in `rotation_stability.rs`.**  The per-subsample
call goes through `validate_and_normalize_weights` directly; errors
propagate to the caller, which maps them to NaN rows. The NaN-row filter
at the end of the loop applies the same skip-rate logic. Left at default.

## What users see

- `PlsKitInvalidWeights(reason="insufficient_effective_n")` from a
  top-level call — your `(weights, k)` combination is infeasible. The
  effective sample size is too small for the number of components
  requested. Actions: lower `k`, flatten extreme weights, or add
  observations.

- `PlsKitResamplingDegenerate` from a top-level call with CI or rotation
  stability — your weights are healthy globally but too many bootstrap
  subsamples failed individually (each subsample draws fewer rows, so
  n_eff shrinks per draw). Actions: raise `n_boot` to average over more
  draws, lower `m_rate` to draw larger subsamples, or raise
  `max_skip_rate` to tolerate a higher fraction of degraded subsamples.

## How to bypass the check (advanced)

Direct Rust users who know what they are doing can set:

```rust
pls1_fit(x, y, KSpec::Fixed(k), weights, FitOpts {
    check_n_eff: false,
    ..FitOpts::default()
})
```

Python, R, and Julia wrappers do not expose `check_n_eff`. It is an
internals-only opt-out. The public wrapper API always runs the check at
the top level.

## History note

Earlier versions enforced the n_eff check inside
`validate_and_normalize_weights`, which is called at every call site —
including per-fold CV. That coupled two distinct failure modes: "user
request infeasible on the full dataset" and "one unlucky fold has low
n_eff by construction." Option B (this contract, implemented in wave
F11+) separates them: `validate_and_normalize_weights` no longer does
the n_eff check; `check_n_eff_for_k` is called explicitly at top-level
entries via the `FitOpts.check_n_eff` flag.
