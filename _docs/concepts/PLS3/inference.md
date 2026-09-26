# Inference

`pls3_confirmatory_test(X, Y, k=1, *, method=...)` asks one question: **is
there a real X↔Y association on the first latent variable, or could the
fitted pattern be noise?** It returns the same `ConfirmatoryTestResult`
as `pls1_confirmatory_test`.

## The statistic

On each of `n_splits` random half-splits of the rows (default 50):

1. Standardize the training half on its own moments and the test half
   with the training moments.
2. Fit PLS3 at `k = 1` on the training half to get `(u₁, v₁)`.
3. Project the **held-out** rows and correlate the two score vectors:
   `r = cor(X̃_te u₁, Ỹ_te v₁)`.

The split correlations are averaged on the Fisher-z scale, and the
reported `statistic` is `tanh(z̄)`. This is the same held-out statistic
`split_exact` and `split_nb` use for PLS1, with the fixed outcome `y`
replaced by an estimated Y-side direction `v₁`. A split whose training
half has no first component (a constant block, or a Y half orthogonal to
the X half up to rounding, which the fit returns as `k_used = 0`)
contributes `r = 0`, not `NaN`.

**Sign indeterminacy costs nothing.** An SVD fixes `(u₁, v₁)` only up to a
simultaneous sign flip, and a flip negates both held-out score vectors at
once, so `r` is unchanged. No alignment step is needed. This protection is
specific to the correlation; it would not extend to per-coordinate
intervals on the saliences, which is one reason the family offers none
(`result.ci` is always `None`).

## Two methods

`method` is keyword-only and required; there is no default.

| | `split_exact` (recommended) | `split_nb` |
|---|---|---|
| Reference | permutation | Nadeau-Bengio corrected t |
| Validity | exact whenever rows are exchangeable under the null | asymptotic |
| Cost | up to `(n_perm + 1) × n_splits` fits (default `n_perm = 1000`) | `n_splits` fits |
| `args` | `{"n_perm": 1000, "n_splits": 50}` | `{"n_splits": 50, "force": False}` |

Both compute the same statistic on the same splits and differ only in the
reference they compare it against. Both p-values are one-sided: only a
positive held-out correlation counts as evidence.

**`split_exact`** builds its null by permuting the **rows of Y as whole
units** against X. That breaks the cross-block association while keeping
each row's within-Y structure, which matches the scientific question ("is
there *any* X↔Y association?"). The splits are drawn once and held fixed
across all permutations; redrawing them per permutation would fold
split-to-split scatter into the null. The p-value is
`(1 + #{null z̄ ≥ observed z̄}) / (n_perm + 1)`.

**`split_nb`** skips the permutations and compares the Fisher-z average
against a t reference whose spread is inflated by PLS1's Nadeau-Bengio
correction. Part of that transfers cleanly to two blocks: conditional on
the training half, `u₁` and `v₁` are fixed, so on one split the two
held-out score vectors are fixed linear combinations of independent
test-half rows and `r` follows the ordinary null correlation law. The
between-split correction does not follow from that argument; it is PLS1's
heuristic, and its use here rests on the developers' simulations
(Gaussian, heavy-tailed, low-stable-rank and real two-block designs), in
which `split_nb` came out conservative, never anti-conservative. Use it
when `split_exact` is too slow; prefer `split_exact` otherwise.

## The `split_nb` auto-gate runs on X only

A `split_nb` request first goes through the same auto-gate as
`pls1_confirmatory_test`, applied to the standardized X. The gate flags a
design when any of these holds:

- `n < 25` (with no weights, `n_eff` is the row count),
- the stable rank of X is below 3,
- X has 4 columns or fewer.

A flagged design runs `split_exact` instead, with `n_perm = 1000` and your
`n_splits`. `result.method` then reports `"split_exact"`, and Python emits
a `UserWarning`. `args={"force": True}` skips the reroute and runs `split_nb`
anyway. `result.stable_rank` is filled whenever `split_nb` was requested,
since it is what the gate saw.

Y is deliberately never gated. In ordinary PLSC `q` is small (a handful of
behavioural measures), so a stable-rank floor of 3 on Y would flag nearly
every legitimate design. The thresholds are PLS1's, calibrated on
single-block designs and not re-derived for two blocks; treat them as a
conservative guard, not a tuned boundary.

## Why only `k = 1`

`k` other than `1` raises `invalid_argument`. Above LV1, when singular
values are close, neither the component ordering nor the individual
directions need survive from the training half to the test half, and it is
not settled whether the statistic should then be per-component or
subspace-level. The function errors rather than guessing. (Sign flips are
not the issue; they cancel, as above.)

## Methods that are not available

`pls1_confirmatory_test` has five methods; this function accepts two. The
other three are refused, each for its own reason:

- **`raw_perm`** needs a cross-validated prediction statistic, and PLS3 has
  no `predict`.
- **`score`** is not implemented. Its symmetric analog would be an RV-type
  test on `‖X̃ᵀỸ‖²_F`, which tests a different estimand from the LV1
  held-out correlation this function reports.
- **`e`** needs a generative model, which symmetric cross-decomposition
  does not supply.

## Clustered rows: neither method is valid

Both methods assume the rows are exchangeable under the null. With
clustered rows (repeated scans per subject, several sessions per
participant, families) that fails twice: random splits put the same
subject in both halves, and permuting Y rows one at a time breaks
within-subject exchangeability, so `split_exact` loses its exactness too.
`split_nb` is worse off, because its between-split correction is least
trustworthy when the resampling unit is not the independent unit.
Blocked splits and blocked permutation are not implemented. Do not run
this test on clustered designs; aggregate to one row per independent unit
first.

## Other things to know

- `pre_standardized_X` / `pre_standardized_Y` are accepted but have **no
  effect** on either method: every training half is re-standardized on its
  own rows, and the gate standardizes its own copy of X.
- Minimum sizes: `n ≥ 6`, `n_splits ≥ 2`, `n_perm ≥ 1`; smaller values
  raise `invalid_argument`.
- On the result, `n_eff` equals `n`, `ci` is `None`, `n_perm` is `None`
  for `split_nb`, and `rho_hat` is filled for `split_nb` only (and only
  when the test half has at least 4 rows).
- Pass `seed` for a reproducible run; results are identical across thread
  counts for a fixed seed.

## Cross-references

- [PLS3 overview](index.md): why there is no `predict`
- [Fit and transform](fit-and-transform.md): the sign convention the statistic relies on
- [PLS1 inference](../PLS1/inference.md): the single-response tests this one extends
- [Python API §3.4](../../python/api.md): `pls3_confirmatory_test` signature and `args`
- [Results](../../python/results.md): `ConfirmatoryTestResult`
