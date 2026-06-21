# Rotations and stability

> Status: placeholder. The full treatment will land with publication of
> the methods paper. Until then, see the
> [Python API → rotate / pls1_rotation_stability](../../python/api.md)
> for the implemented surface.

Topics this page will cover:

- Why PLS components are not unique: sign indeterminacy and within-subspace rotation
- Post-fit rotations: `varimax` (implemented), `promax` / `oblimin` / `geomin` (planned)
- The pluggable loading basis `L` — running rotation in a basis other than `W` itself
- `pls1_rotation_stability`: a Politis–Romano subsampling diagnostic for rotation reliability
- The composite stability statistics: `agreement` (post-procrustes Frobenius), `subspace_cos`, `cos_β`, `beta_norm`
- Reading the diagnostic: when a fitted rotation is trustworthy and when it isn't
- Cross-reference: rotation-invariant CIs on `W` / `β` live in [confidence intervals](ci.md)
