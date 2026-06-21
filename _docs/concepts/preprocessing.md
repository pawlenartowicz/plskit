# Preprocessing

This page is built around `plskit.preprocess` — what it does, when to
call it explicitly, and when the default fit-time path is fine.

## What `preprocess` does

A single helper that runs plskit's canonical standardization on whichever
inputs you pass. All three arguments are independently optional:

- `X` *(2-D)* → standardize per column (zero mean, unit SD)
- `Y` *(1-D or 2-D, shape-polymorphic)* → standardize per column (or as
  a scalar for 1-D)
- `weights` *(1-D, optional)* → normalize to mean 1 (`Σ w'ᵢ = n`) and
  report `n_eff`

The result carries the standardized arrays plus the means / scales /
normalized weights that produced them, so you can reuse the same
transformation on fresh data and back-project predictions later. It is
the **same code path** the fit uses internally when
`pre_standardized=False` — there is no second recipe.

For the binding signature and field names, see the
[`PreprocessResult` table in the naming contract](../internals/naming.md#preprocess-helper-output-preprocessresult-fields).
For per-language signatures and examples, see the API reference under
your wrapper's docs.

## When to use `preprocess`

- **Loop over the same `(X, y, w)`** — `k`-sweep, bootstrap, permutation
  test, sensitivity analysis. Call `preprocess` once, then pass
  `pre_standardized=True` on every downstream fit so each iteration
  skips an `O(n·p)` standardization. NIPALS is `O(n·p·k)`; the speedup
  is `1 + 1/k`× — small but free.
- **Inspect the standardized data or the `(mean, scale)` plskit fits
  on.** Useful for sanity-checking units, plotting, or diagnosing why
  one column dominates a loading.
- **Apply the same transformation to a held-out split.** Standardize
  the train fold via `preprocess`, then apply the returned
  `X_mean` / `X_scale` to the test fold yourself before predicting —
  the result fields are exactly the parameters you need.

## When *not* to use `preprocess`

- **One-shot fit.** Call `pls1_fit(X, y)` with `pre_standardized=False`
  (the default). The fit standardizes internally; calling `preprocess`
  separately is redundant.
- **You already preprocessed with a non-default recipe** (mean-centering
  only, Pareto, range, robust median/MAD, SNV, MSC). Do **not** run
  `preprocess` over already-treated data — pass `pre_standardized=True`
  directly. See the [chemometrics quick reference](#chemometrics-quick-reference).

## The `pre_standardized` flag — quick decision

- **No preprocessing done →** `pre_standardized=False` (default). Safe.
- **Preprocessed with the canonical recipe** (mean-center + divide by
  `sqrt(var, ddof=0)`, weights normalized to mean 1) → either flag
  works; `True` skips one redundant pass.
- **Preprocessed with a non-default recipe →** `pre_standardized=True`.
  plskit will not re-process your data; you own the consequences.

## The canonical recipe

`preprocess` (and the fit's internal `pre_standardized=False` branch)
applies, per column on `X` and as a scalar on `y`:

    standardized = (raw − mean) / sqrt(var, ddof=0)

(Population variance, not the unbiased sample variance.) Columns whose
standard deviation is below `1e-12` are clamped to scale `1.0` instead
of dividing by near-zero — zero-variance columns become zero columns.

When you pass observation weights `w`, plskit first normalizes them to
mean 1 (`Σ w'ᵢ = n`), then uses weighted formulas:

    weighted_mean = (Σ w'ᵢ xᵢ) / n
    weighted_var  = (Σ w'ᵢ (xᵢ − weighted_mean)²) / n

The √w row-scaling is **not** preprocessing — it is the Cholesky factor
of the weighted least-squares problem and stays inside the fit, applied
even when `pre_standardized=True`.

## Common gotchas

- **Half-treated X (e.g. mean-centered but not scaled)** —
  `pre_standardized=False` will divide by std and *change* your data.
  Either skip your mean-centering and leave the flag at default, or
  finish the standardize and pass `True`.
- **Half-treated y** — same trap.
- **Weights not normalized** — `pre_standardized=True` skips weight
  normalization. If your weights are not already mean-1, your fit is
  on the wrong scale. Either pass raw weights and leave the flag at
  default, or normalize first (`w_norm = w * len(w) / w.sum()`) and
  pass `True`.

When `pre_standardized=True`, the fit's `coef` is in standardized space
and `beta` equals `coef` (no back-projection). The `intercept` is `0`.
Compare `coef` (not `beta`) to a from-raw fit if you want to verify
parity.

## Chemometrics quick reference

If you use a non-default scaling recipe in your pipeline, pass
`pre_standardized=True` after your own preprocessing — do **not** route
already-treated data through `preprocess`. Common recipes that are
*not* idempotent under plskit's standardize:

| Recipe | What it does | After this, pass |
|---|---|---|
| Mean-centering only | Subtract mean, leave scale | `pre_standardized=True` |
| Pareto scaling | Divide by `sqrt(std)`, not `std` | `pre_standardized=True` |
| Range scaling | Divide by `max − min` | `pre_standardized=True` |
| Robust (median/MAD) | Robust center + scale | `pre_standardized=True` |
| SNV | Per-row mean-center + scale | `pre_standardized=True` |
| MSC | Multiplicative scatter correction | `pre_standardized=True` |
