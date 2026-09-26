# What sparsity costs

Three properties of `pls3_fit` do not survive `keep_X < p` or
`keep_Y < q`. None of them is a bug, and none of them is guarded against
in the library: they are consequences of asking for a constrained
factorization.

## The saliences are no longer orthogonal

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

## `singular_values` are not singular values

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

## Rotation removes the zeros

`rotate` right-multiplies the weights by an orthogonal `R`. A rotated
sparse salience matrix is dense again: every column becomes a mixture of
all `k` supports. plskit does not guard against this. Sparsity and
simple-structure rotation are two answers to the same question (make the
loadings readable), and you pick one.

## What is unchanged

Sign pinning, standardization and its moments, `pls3_transform`, the
refusal of observation weights, and the `k ≤ min(p, q)` bound all behave
exactly as on a dense fit.
