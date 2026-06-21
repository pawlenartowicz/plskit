# Confidence intervals

> Status: placeholder. The full treatment will land with publication of
> the methods paper. Until then, see the
> [Python API → pls1_confirmatory_test(ci=True)](../../python/api.md)
> and the [results](../../python/results.md) page for the implemented surface.

Topics this page will cover:

- Rotation-invariant subsample CIs: why naïve bootstrap CIs on `W`, `β`, or
  loadings fail under sign / rotation indeterminacy, and how
  `plskit` works around it
- Composite (subspace-level) CIs: `subspace_cos`, `cos_β`, `beta_norm`, `holdout_corr`
- Per-variable CIs: `leverage_ci_*`, `beta_sign_z`
- Per-coordinate β CI (PLS1-only — β is invariant under PLS1 rotations)
- Subsampling vs bootstrap: `m_rate`, the resolved subsample size `m = ceil(n^m_rate)`
- Procrustes alignment: how each resampled `W` is aligned back to the full-fit basis
- Legacy compatibility: BSR (bootstrap ratio) — what it is, why we ship it, why we recommend the percentile CI instead
