# Fit and transform

`pls3_fit(X, Y, k=1)` standardizes both blocks, takes one thin SVD of
`X̃ᵀỸ`, and keeps the first `k` components. `pls3_transform` projects new
rows onto them. (`plssvd_fit` / `plssvd_transform` are the same
functions.)

## Inputs

- `X (n × p)`, `Y (n × q)`: both 2-D, same number of rows, at least one.
  Zero rows raise `invalid_argument` (the standardization moments of an
  empty column are undefined), as in `pls1_fit`.
- `k`: `1 ≤ k ≤ min(p, q)`, default `1`. `k = 0` raises
  `invalid_argument`; `k > min(p, q)` raises `k_exceeds_max`. Note that the
  bound does not involve `n`; see Truncation below.
- `pre_standardized_X`, `pre_standardized_Y`: one flag per block, because
  the blocks are independent inputs. Each skips that block's
  standardization; the [canonical recipe](../preprocessing.md) (population
  variance, `ddof=0`) applies otherwise.
- `weights`: must be `None` (see below).

## Outputs

`PLS3Result` carries the saliences `U (p × k_used)` and `V (q × k_used)`
with orthonormal columns, `singular_values` in descending order, the
in-sample scores `x_scores = X̃U` and `y_scores = ỸV`, the four
standardization moment vectors, `k_used`, and both `pre_standardized_*`
flags. The fit draws no randomness, so there is no `seed`. Full field
list: [results](../../python/results.md).

## Signs are pinned

An SVD determines each pair `(uᵢ, vᵢ)` only up to a simultaneous sign
flip, and the linear-algebra backend makes no promise about which one it
returns. `plskit` fixes it: each pair is flipped **together** so that the
largest-magnitude entry of `uᵢ` is positive, with exact ties going to the
lowest row index.

Flipping the pair as a unit leaves `uᵢ σᵢ vᵢᵀ`, and therefore every
product and correlation between X-side and Y-side scores, unchanged. It
only makes the reported saliences reproducible, so two fits of the same
data agree and salience vectors can be compared across analyses without
spurious sign differences.

> **Near-tie caveat.** The pin is only as stable as the entry it keys on.
> If the two largest `|uᵢ|` entries are nearly (not exactly) equal, the
> tiny cross-platform differences `plskit` allows can move the argmax to
> the other entry, and the whole pair flips sign. Signs are reproducible
> except at near-ties. If you compare saliences across machines, compare
> up to sign when the top two `|uᵢ|` are close.

## Truncation

A component whose singular value is numerical dust is dropped, so `k_used`
can be smaller than the requested `k`. Two floors decide what counts as
dust, and truncation stops at the first singular value that fails either:

- an absolute floor of `1e-14`, applied to every component;
- a relative floor of `max(n, p, q)·ε·‖X̃‖_F·‖Ỹ‖_F`, also applied to every
  component (the first included), where `ε ≈ 2.2e-16` is machine epsilon and
  `‖X̃‖_F`, `‖Ỹ‖_F` are the Frobenius norms of the blocks `X̃ᵀỸ` is formed
  from (`n·√(pq)` for their product under the canonical recipe).

The relative floor is a numerical-rank tolerance, and it is the one that
matters on ordinary standardized input. When the true rank of `X̃ᵀỸ` is
below the requested `k`, the "zero" singular values come out as rounding
noise from forming `X̃ᵀỸ`, which scales with `‖X̃‖_F·‖Ỹ‖_F` and so grows
with `n`, `p` and `q`. That noise clears `1e-14` once `n` is in the
hundreds, so the absolute floor alone would keep pure-noise components
with arbitrary `U` / `V` columns. The relative floor sits well above the
noise and drops them. The rank falls below `k` when:

- `n − 1 < k`: after centering, `n` rows span at most `n − 1` dimensions,
  and the bound on `k` does not check this;
- columns are linearly dependent, e.g. a Y block holding several subscales
  plus their total score, where `k = q` asks for a component that does not
  exist.

For example, with `n = 10`, `p = 50`, `q = 20` (true rank 9) and `k = 15`,
the fit returns `k_used = 9`.

The reference is the size of the data, not the leading singular value
`σ₁`. When Y is nearly orthogonal to X, `σ₁` is small while the rounding
noise is not, and a floor proportional to `σ₁` would keep noise
components. Measured across rank-deficient designs with strong, null,
weak (down to `1e-8` of the noise scale) and exactly orthogonal signal,
`n` from 10 to 2e5, the noise components sat at most at 0.03 times the
floor, and every component carrying more than `1e-9` of
`‖X̃‖_F·‖Ỹ‖_F` sat at least 22 times above it.

`k_used` is a numerical rank, not a count of components that carry signal.
A component that clears both floors can still be noise; the confirmatory
test, not `k_used`, answers whether an LV is there. The first component
answers to the same two floors as the rest: if Y is orthogonal to X up to
rounding (for instance an outcome block residualized on covariates that
span X), every singular value is rounding noise and the fit returns
`k_used = 0`, with empty `U`, `V` and score matrices. This is the same
policy `pls1_fit` follows: at `q = 1` the two families return the same
`k_used` on such input.

**Mis-scaled pre-standardized blocks.** With `pre_standardized_X=True` (or
`_Y`), the engine takes your word that the block is zero-mean and
unit-variance. The relative floor does not care about scale, but the
absolute one does: if the block is on a much smaller scale, real
components can fall under `1e-14` and `k_used` silently comes back below
`k` instead of raising an error. PLS3 has no option to turn this
truncation into an error. If you pass a `pre_standardized_*` flag, make
sure the block really is standardized.

## Observation weights are refused

`pls3_fit` raises `PlsKitError(code="invalid_argument")` whenever
`weights` is not `None`. It is refused rather than ignored because a caller who ran
a weighted PLS1 analysis and passed the same weights here would otherwise
get an unweighted answer that looks weighted. `PLS3Result` has no `n_eff`
field, and the confirmatory test reports `n_eff` equal to `n`.

## Projecting new data

```python
scores = plskit.pls3_transform(model, X_new, Y_new, which="both")
```

- `which` is `"x_scores"`, `"y_scores"` or `"both"` (default). A block that
  `which` asks for must be supplied, or the call raises
  `invalid_argument`; it never returns a silent `None` for a requested
  block.
- In the returned `PLS3Scores`, a field is `None` exactly when `which` did
  not ask for it.
- New data is standardized with the **fit's** moments, not its own, just
  as `pls1_predict` does. On a `pre_standardized_*` fit those moments are
  the identity (mean 0, scale 1), so apply your own preprocessing to
  `X_new` / `Y_new` first.
- The column count must match the fit (`shape_mismatch` otherwise).

## The sparse variant drops the orthogonality

Everything above describes `pls3_fit`, which gets every component from one
SVD. `spls3_fit` constrains each component to `keep_X` X variables and
`keep_Y` Y variables, which cannot be done in one decomposition; it
alternates instead, and the resulting saliences are unit-norm but not
orthogonal, with `singular_values` that are not singular values. See
[what sparsity costs](../sPLS3/sparse-saliences.md) before reading a
sparse fit the way you would read a dense one.

## Cross-references

- [PLS3 overview](index.md): PLSC vs PLS2, the aliases
- [Inference](inference.md): testing the LV1 association
- [Preprocessing](../preprocessing.md): the standardization recipe and the `pre_standardized` decision
- [Python API §2c](../../python/api.md): `pls3_fit`, `pls3_transform`
- [sPLS3](../sPLS3/index.md): the sparse variant, `keep_X` / `keep_Y`
- [What sparsity costs](../sPLS3/sparse-saliences.md): non-orthogonality, `singular_values`, rotation
