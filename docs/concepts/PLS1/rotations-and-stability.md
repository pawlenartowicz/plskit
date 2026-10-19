# Rotations and stability

> Status: placeholder. The full treatment will land with publication of
> the methods paper. Until then, see the
> [Python API → rotate / pls1_rotation_stability](../../python/api.md)
> and the [results](../../python/results.md) page for the implemented surface.

Topics this page will cover:

- Why PLS components are not unique: sign indeterminacy and within-subspace rotation
- Post-fit rotations: `varimax` (implemented), `promax` / `oblimin` / `geomin` *(planned)*
- The pluggable loading basis `L`: running rotation in a basis other than `W` itself
- `pls1_rotation_stability`: a Politis–Romano subsampling diagnostic for rotation reliability
- The stability statistic: `variance_ratio`, the ratio `V_rot / V_unrot` of rotated to unrotated axis variance across subsamples (each resampled basis aligned to its reference by signed permutation), with a paired-bootstrap percentile CI, plus the per-axis `variance_ratio_per_axis` and the `degenerate_baseline` flag
- Reading the diagnostic: when a fitted rotation is trustworthy and when it isn't
- Cross-reference: rotation-invariant CIs on leverage and `β` live in [confidence intervals](ci.md)
