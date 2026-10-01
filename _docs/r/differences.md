# R: differences from Python

Names are identical across languages. What differs is how
values are represented.

## Arguments

- `k` and `k_max`, not `ncomp` (naming.md ground rule 3): this departs
  from the R `pls` package convention on purpose, and there is no alias.
- Whole-number doubles are accepted where an integer is expected
  (`k = 3` and `k = 3L` are the same); `k = 2.5` raises `invalid_argument`.
- `X` may be a numeric matrix or a `data.frame` whose columns are all
  numeric or logical (converted with `as.matrix`); a data.frame with any
  other column raises `invalid_argument`. Integer data is converted to
  double, and so is logical data (`TRUE`/`FALSE` become `1`/`0`) in every
  numeric-array argument (`X`, `y`, `Y`, `X_new`, `Y_new`, `weights`,
  `L`, a matrix `W`), the same way numpy casts a bool array to float. A
  scalar logical flag such as `pre_standardized` or `ci` is left as a
  flag, not promoted, since the declared argument kind tells the two
  apart.
- A length-1 array (`array(y)`, `dim` of length one) behaves like a
  plain vector, whether it holds doubles, integers, or logicals.
- A vector `y` is 1-D; a one-column matrix `Y` stays 2-D (as in Python's
  PLS3 functions).
- A single row of `X_new` needs `drop = FALSE`
  (`pls1_predict(fit, X[1, , drop = FALSE])`): `X[1, ]` drops to a
  plain (1-D) vector, which raises `invalid_argument` ("must be
  2-D, got 1-D").
- Names on data vectors are ignored (`y` or `weights` with names fit
  exactly like the unnamed vectors).
- `NA` and `NaN` in the data (numeric, integer or logical) reach the
  engine, which raises `non_finite_input`. `NA` in a logical or string
  option raises `invalid_argument`.
- Strings and names are re-encoded to UTF-8 before they reach the
  engine; a string that is not valid UTF-8 after that re-encoding
  (bytes `enc2utf8()` cannot fix up) raises `invalid_argument`.
- `args`, `find_k_args` and `rotation_args` are named lists; `NULL`
  (the whole argument or one entry) means the engine default. For
  `args` and `rotation_args`, `list()` behaves the same as `NULL`.
  `find_k_args` is the exception: with an integer `k` (not
  `"optimal"` or `"sequence"`), `find_k_args` must be `NULL`;
  `find_k_args = list()` still raises `invalid_argument`, because the
  check is "is it set at all", not "is it empty" (as in Python).
  Unknown keys inside any of the three raise `invalid_args`.
- `bit64::integer64` input is not read as numeric data: any
  `integer64` value other than `seed` (a vector, a matrix, or a
  `data.frame` column) raises `invalid_argument`; convert it with
  `as.numeric()` first. `seed` is the exception, since an
  `integer64` seed converts exactly with `as.character()` (below).

## Results

Results are named lists with an S3 class: the snake_case of the Python
type name, then `"plskit_result"`.

| Python | R |
|---|---|
| `PLS1Result` | `c("pls1_result", "plskit_result")` |
| `ConfirmatoryTestResult` | `c("confirmatory_test_result", "plskit_result")` |
| `CIScalar` (nested) | `c("ci_scalar", "plskit_result")` |
| every other type | the same rule |

| Python value | R value |
|---|---|
| 1-D `ndarray` | numeric vector |
| 2-D `ndarray` | numeric matrix |
| `None` | a retained `NULL` element |
| `dict[int, float]` (`cv_scores`, ...) | named numeric vector, names `"1"`, `"2"`, ... |
| `list[int]`, integer array | integer vector |
| bool array | logical vector |
| `list[CIScalar]` | unnamed list of `ci_scalar` lists |
| integer scalar | integer (double above 2^31 - 1) |
| `seed` | character string (below) |

`print()` shows a compact summary (field names, shapes, nested type
names). There are no `coef()` or `predict()` methods (wrappers add no
statistical methods of their own): call `pls1_predict(fit, X_new)`.

Results can be passed back: `pls1_predict(fit, X_new)`, `rotate(fit)`,
`pls3_transform(model, ...)`. `rotate` takes a fitted `pls1_result`
(model overload, returns a `pls1_result` with `rotation_spec` set) or a
weight matrix (returns a `rotate_result`). A result edited by hand so
that it no longer forms a valid model raises `invalid_argument`; an
unclassed copy (`unclass(fit)`) still works.

## Seeds

Seeds are unsigned 64-bit integers, and about half of all drawn seeds
exceed what R's integer or `bit64::integer64` can hold. R therefore
carries them as decimal strings: `result$seed` is a character scalar such
as `"13835058055282163712"`, and `seed =` accepts a string, an integer, a
whole double up to 2^53, or a `bit64::integer64` (converted with
`as.character()`, exact even above 2^53). `seed = result$seed`
reproduces the run exactly.

## Errors and warnings

Errors are R conditions carrying Python's error codes:

| Condition class | Fields |
|---|---|
| `c("plskit_invalid_weights", "plskit_error", "error", "condition")` | `code`, `reason` |
| `c("plskit_resampling_degenerate", "plskit_error", "error", "condition")` | `code`, `skipped`, `total`, `skip_rate`, `threshold` |
| `c("plskit_error", "error", "condition")` | `code` (every other code) |

```r
tryCatch(pls1_fit(X, y, weights = w),
  plskit_invalid_weights = function(e) e$reason,
  plskit_error = function(e) e$code)
```

When the `split_nb` auto-gate reroutes a test, a warning of class
`c("plskit_rerouted", "plskit_warning", "warning", "condition")` carries
`requested`, `actual`, `n_perm`, `stable_rank` and `n_eff`; silence it with
`suppressWarnings(..., classes = "plskit_rerouted")`.

## Known limitations

- A long call cannot be interrupted with Ctrl-C: the engine's worker
  threads do not poll R for interrupts.
- `verbose = TRUE` writes progress to the process's stderr. Terminal R
  shows it; RStudio does not display native stderr.
- Forking after a parallel plskit call (`parallel::mclapply`) can hang
  the child process, because the engine's thread pool does not survive
  `fork`. Use a socket cluster (`parallel::makePSOCKcluster`) instead.
- If R runs out of memory while plskit converts arguments or results (a
  very large result, for example `return_perm_matrix = TRUE` on a big
  problem, or a huge ALTREP input materializing on the way in), the R
  error is raised from inside extendr's allocation and skips plskit's
  cleanup, leaking that call's native memory. The session stays usable;
  only the memory for that one call is not reclaimed.
