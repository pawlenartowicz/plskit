# Confidence intervals

> Status: placeholder. The full treatment will land with publication of
> the methods paper. Until then, see the
> [Python API → pls1_confirmatory_test(ci=True)](../../python/api.md)
> and the [results](../../python/results.md) page for the implemented surface.

Topics this page will cover:

- Rotation-invariant subsample CIs: why naïve bootstrap CIs on `W`, `β`, or
  loadings fail under sign / rotation indeterminacy, and how
  `plskit` works around it
- Composite CI: `holdout_corr`, a Fisher-z Wald CI on the held-out correlation
- Per-variable CIs: `leverage_ci_lower` / `leverage_ci_upper` with `leverage_se` (a normal-theory bootstrap interval centered on the full-data leverage, clamped to [0, 1]), and the per-variable subsampling z for β, `beta_sign_z` / `beta_sign_z_signed`
- Per-coordinate β CI: `beta_ci_lower` / `beta_ci_upper` with `beta_se` (PLS1-only; β is invariant under PLS1 rotations)
- Why no alignment step is needed: leverage and β do not change under within-subspace rotation of `W`, so resampled fits are compared without Procrustes alignment
- Subsampling vs bootstrap: `m_rate`, the resolved subsample size `m = ceil(n^m_rate)`, which β and `holdout_corr` resample at; and why leverage is resampled n-out-of-n with replacement instead. Its finite-sample bias depends on the sample size, so size-`m` fits measure its spread and bias at the wrong size; the interval targets the expected leverage of a size-`n` fit, not its large-sample limit (see [results](../../python/results.md), "Per-variable readouts")
