# Differences from Python

`PLSKit.jl` calls the Python `plskit` package, so names, defaults,
validation, error codes, warnings and numbers are Python's. This page
lists what looks different from Julia.

## Calling

- **Positional or keyword.** A Python parameter that is required and not
  keyword-only is a positional argument; every other parameter is a
  keyword.

  ```julia
  pls1_fit(X, y; k=3, seed=42)
  spls1_fit(X, y, 3, 20)                                     # k and keep: positional
  pls1_confirmatory_test(X, y; k=1, method="split_exact")    # method: required keyword
  preprocess(; X=X, Y=y)                                     # every argument has a default
  pls3_transform(model; X_new=X2, which=:x_scores)
  ```

  Python's `pls3_transform(model, X_new)` is `pls3_transform(model; X_new=X_new)`
  in Julia. Leaving out `method=` raises Julia's own `UndefKeywordError`.
- **`args`, `find_k_args`, `rotation_args`** take a `NamedTuple` or a
  `Dict` with `Symbol` or `String` keys, nested like Python's dicts:
  `find_k_args=(selector=:bic, args=(n_folds=5,))`.
- **String options** (`method`, `selector`, `which`, ...) take a `String`
  or a `Symbol`: `method=:split_exact`.
- `nothing` is Python's `None`.
- **Arrays**: any `AbstractArray` of numbers. A dense `Array` reaches
  Python as a NumPy view; other arrays (`X'`, `view(...)`, `BitVector`)
  are collected first. Element types and memory layout follow the Python
  rules: integers and `Float32` are promoted to `Float64`, `Bool` data
  (`BitMatrix`, `BitVector`, `Matrix{Bool}`) reads as `1`/`0`, and a
  column-major `Float64` `X` (or `Y`, `Y_new`, `W`, `L`) is read in
  place, with no copy. On the same machine,
  results then match the Rust core bit for bit, and agree only to
  rounding with a Python call on a C-ordered copy of the same `X`. `missing` is passed as `NaN`, so it raises
  `:non_finite_input`, as `NaN` does.
- **`seed`** takes an `Int` or a `UInt64`. `result.seed` is a `UInt64`,
  shown in hex (`0x076988841aab76b2`), and spans the full 64-bit range;
  `seed=result.seed` reproduces the run exactly.
- A whole float count (`k=2.0`) is accepted as `2`. Every argument is
  validated by `plskit-py` exactly as it is for a Python caller: a
  fractional `k` (`k=2.5`) raises `PlsKitError` with code
  `:invalid_argument`.

## Results

- Every result is a `PlsKitResult{T}`, where `T` is the Python type name.
  Each Python result type has an alias, `PLS1Result === PlsKitResult{:PLS1Result}`,
  so `fit isa PLS1Result` and dispatch on the alias work. Nested results
  (`test.ci`, `test.ci.holdout_corr`, `fit.rotation_spec`,
  `fit.selection_result`) are `PlsKitResult`s too.
- Fields read as properties (`fit.W`, `fit.seed`), in the order of the
  [results reference](../python/results.md); `propertynames(fit)` lists them.
- **Fields are copies.** Editing `fit.W` in place changes your copy only:
  `pls1_predict(fit, X)`, `rotate(fit)` and `pls3_transform(model; ...)`
  hand Python the object the result holds, so they use the values the fit
  returned. `getfield(fit, :py)` is that Python object.
- Type mapping:

  | Python | Julia |
  |---|---|
  | `ndarray` of floats, 1-D / 2-D | `Vector{Float64}` / `Matrix{Float64}` (copied) |
  | `ndarray` of bools / ints | `Vector{Bool}` / `Vector{Int}` |
  | `None` | `nothing` |
  | `int` in a `seed` field, at any nesting depth | `UInt64` |
  | other `int`, `float`, `bool`, `str` | `Int`, `Float64`, `Bool`, `String` |
  | NumPy scalar (`np.float64`, `np.bool_`, ...) | its plain Julia value, by the rules above |
  | `dict[int, float]` (`cv_scores`, ...) | `Dict{Int,Float64}` (empty: `Dict{Int,Float64}()`) |
  | `RotationSpec.args` | `NamedTuple` |
  | `list[int]` (`keep_grid`) | `Vector{Int}` (empty: `Int[]`) |
  | `list[bool]` | `Vector{Bool}` |
  | `list[CIScalar]` | `Vector{PlsKitResult}` |
  | result dataclass | `PlsKitResult{T}` |

  An integer that does not fit the target width (an `Int` overflow, or an
  unsigned array with a value above `typemax(Int)`) raises an error naming
  the field, rather than wrapping silently.
- `pls1_predict` returns a plain `Vector{Float64}`.
- Display is a compact summary: the type, then each field with its value
  (scalars, strings) or a short description (arrays, dicts). Results have no
  methods (`coef`, `predict`, ...) beyond display.

## Errors

- One error type, `PlsKitError`, with `code::Symbol` (Python's
  `PlsKitError.code` spelling), `msg::String` and `details::NamedTuple`:

  ```julia
  try
      pls1_fit(X, y; weights=w)
  catch e
      e isa PlsKitError || rethrow()
      if e.code === :invalid_weights
          e.details.reason   # :negative, :all_zero, :length_mismatch or :insufficient_effective_n
      elseif e.code === :resampling_degenerate
          e.details          # (skipped=..., total=..., skip_rate=..., threshold=...)
      end
  end
  ```

  Julia cannot subclass a concrete type, so dispatch on `e.code` replaces
  Python's `PlsKitInvalidWeights` and `PlsKitResamplingDegenerate`.
  `details` is empty for every other code. The code list is in the
  [Python API reference](../python/api.md#errors).
- An argument of the wrong kind (a `PLS3Result` passed to `rotate`, a
  string where an array belongs) raises `PlsKitError` with code
  `:invalid_argument`, and an unknown `method` or `selector` string one
  with code `:invalid_args`. Any other Python exception (one Python
  itself raises, not plskit) reaches you unchanged as a PythonCall
  `PyException`.

## Warnings

Python warnings raised during a call (the `split_nb` reroute notice) are
re-emitted with `@warn`, message verbatim, `_group=:plskit`, once per call.
Silence them with Julia's logging (for example
`Logging.with_logger(Logging.NullLogger()) do ... end`) or match them in
tests with `@test_logs (:warn, r"rerouted", PLSKit, :plskit) ...`. The
reroute notice has no structured fields: it is a plain `UserWarning` in
Python.

## The Python underneath

- `PLSKit.jl` needs a Python with `plskit` at exactly the version it pins
  (`PLSKit.PLSKIT_PY_VERSION`). By default CondaPkg builds a private
  environment on first use; to use your own Python, see
  [installation](installation.md). Another `plskit` version stops
  `using PLSKit` with an error.
- The Julia package version equals the Python version it runs.

## Limitations

- plskit calls run on Julia's main thread (PythonCall's rule): calling a
  plskit function from any other thread is unsupported and raises an
  error naming the main-thread rule, rather than crashing the process
  (without that check, calling PythonCall off the main thread would
  segfault). Run plskit calls on Julia's main thread; a Julia task
  spawned with `@spawn` and pinned to thread 1, or the default task
  scheduler under `julia` without `-t`, both satisfy this. Parallelism
  comes from the engine's own threads, not from calling plskit
  concurrently: the Python wrapper holds the GIL for the whole engine
  call, so calls from several Julia tasks on the main thread still run
  one at a time, while the engine's own threads run in parallel inside
  each call. Under `julia -t N`, a garbage collection started on another
  thread waits until a long plskit call returns.
- A long call cannot be interrupted with Ctrl-C.
- `verbose=true` writes progress to the process's stderr: the terminal REPL
  shows it, some notebook front ends do not.
- The first `using PLSKit` downloads a private Python environment unless
  PythonCall points at your own Python.
