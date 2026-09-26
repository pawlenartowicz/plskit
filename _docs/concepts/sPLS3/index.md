# Sparse PLS3 (sPLS3)

Sparse PLS3 is [PLS3](../PLS3/index.md) with a hard limit on how many
variables each component may use, on *both* sides. `spls3_fit` takes two
counts: `keep_X`, the non-zeros allowed per column of `U`, and `keep_Y`,
the non-zeros allowed per column of `V`.

`keep_Y` is the one that motivated the method. A dense PLS3 dimension
loads a little on every outcome, which makes "which outcomes does this
dimension describe" a question about thresholds. With `keep_Y = 2` on a
four-outcome block, each dimension names exactly two outcomes and the
outcomes fall into hard groups.

## What the two counts are

- `keep_X` in `[1, n_features]`, `keep_Y` in `[1, n_targets]`. Out of
  range raises `PlsKitError(code="invalid_argument")`.
- **One value for all components.** Both are scalars broadcast to every
  component; there is no per-component budget.
- **Hard selection, not shrinkage.** Survivors keep their magnitudes and
  the vector is renormalized. There is no penalty parameter, and this is
  not the soft-thresholding sPLS of mixOmics.
- **Deterministic ties.** Exact ties in `|u|` or `|v|` break toward the
  lower index, so the selected set never depends on sort order or thread
  count.
- **Selection is per component,** on the deflated cross-covariance.
  Component 2's variables need not include component 1's.
- `keep_X = n_features` and `keep_Y = n_targets` together reproduce
  `pls3_fit` bit for bit; the engine delegates rather than re-deriving.

## How it is computed

The dense fit is one SVD of `A = X̃'Ỹ`. The sparse fit cannot be: there is
no decomposition that returns a sparsity-constrained factorization in one
shot. Instead each component runs an alternation, started from the leading
singular pair of the current `A`:

1. `u ← A v`, keep the `keep_X` largest by magnitude, renormalize.
2. `v ← A'u`, keep the `keep_Y` largest by magnitude, renormalize.
3. Repeat until `v` stops moving (`tol`, default 1e-8) or `max_iter`
   (default 100) sweeps are spent.
4. `σ = u'Av`, then deflate `A ← A − σuv'` and go to the next component.

Hard thresholding is not a projection onto a convex set, so this has no
monotone-convergence guarantee and the support can settle into a
two-cycle. That outcome is reported, not raised: `result.converged` is a
per-component boolean vector and `result.n_iter` the matching sweep
counts.

## Pages

- [What sparsity costs](sparse-saliences.md): non-orthogonality, what
  `singular_values` mean here, and why rotation and sparsity do not mix.
