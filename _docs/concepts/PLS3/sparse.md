# Sparse PLS3 (sPLS3)

Sparse PLS3 is [PLS3](index.md) with a hard limit on how many
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

## What sparsity costs

Three properties of `pls3_fit` do not survive `keep_X < p` or
`keep_Y < q`. None of them is a bug, and none of them is guarded against
in the library: they are consequences of asking for a constrained
factorization.

### The saliences are no longer orthogonal

`pls3_fit` gets all `k` components out of one SVD, so the columns of `U`
are orthonormal and so are the columns of `V`, by construction. `spls3_fit`
restricts each component to its own support. Two components may select
overlapping variables, and nothing re-orthogonalizes them afterwards. Each
column is still unit-norm.

What that changes for a reader: `x_scores` columns are correlated, so a
per-component "variance explained" does not add up, and the usual habit of
reading component 2 as "what is left after component 1" is only
approximately true (deflation removes the rank-one piece `σuv'`, but the
next component is not constrained to be orthogonal to it).

### `singular_values` are not singular values

The field keeps its name because the result type is shared, but on a
sparse fit it holds `σ_a = u_a' A_a v_a`, the bilinear form at the
converged sparse pair on the *deflated* `A_a`. These are not singular
values of `X̃'Ỹ`, they are not guaranteed to descend, and
`Σσ² / ‖X̃'Ỹ‖_F²` is not an explained share of anything. Do not report
one.

Truncation is also weaker here. The fit stops at the first component whose
`σ` falls below `1e-14` or below the dense fit's relative floor
`max(n, p, q)·ε·‖X̃‖_F·‖Ỹ‖_F` (`ε` is machine epsilon; see "Truncation" in
`concepts/PLS3/fit-and-transform.md`). Both apply from the first
component on, so a Y orthogonal to X gives `k_used = 0`, as on a dense
fit. The floor is the same as on a dense fit because the rounding it
guards against comes from forming `X̃'Ỹ`, which both fits share; the
first sparse `σ` would be no reference, since it can sit far below that
rounding.
On a dense fit the first value that fails bounds every later one (the
values descend). On a sparse fit it is a stopping heuristic kept
for shape-consistency with the dense path, not a proof that no later
component would have been larger.

### Rotation removes the zeros

`rotate` right-multiplies the weights by an orthogonal `R`. A rotated
sparse salience matrix is dense again: every column becomes a mixture of
all `k` supports. plskit does not guard against this. Sparsity and
simple-structure rotation are two answers to the same question (make the
loadings readable), and you pick one.

## What is unchanged

Sign pinning, standardization and its moments, `pls3_transform`, the
refusal of observation weights, and the `k ≤ min(p, q)` bound all behave
exactly as on a dense fit.
