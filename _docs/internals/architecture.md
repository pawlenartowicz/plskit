# Architecture

All numerical work runs in one Rust crate. Every other package converts
values on the way in and on the way out.

## Packages

| Directory | What it is | Calls |
|---|---|---|
| `plskit-rs/` | the engine: Rust crate `plskit` | — |
| `plskit-bind/` | binding layer: Rust crate with one entry point, `call(name, inputs)` | the engine |
| `plskit-py/` | Python package: a PyO3 extension module plus Python code | `plskit-bind` |
| `plskit-r/` | R package: an extendr crate plus R code | `plskit-bind` |
| `plskit-jl/` | Julia package, no Rust code | the Python package, through PythonCall.jl |
| `plskit-testdata-gen/` | generator of the [test corpus](testdata.md) | the engine |

The cargo workspace holds `plskit-rs`, `plskit-bind`, `plskit-py` and
`plskit-testdata-gen`. The R crate (`plskit-r/src/rust/`) is its own
workspace root, because R's build runs cargo there.

## How a call travels

```
Python   plskit.pls1_fit(...)                  plskit-py/python/plskit/_api.py
           └─ _plskit.call(name, arguments, …)   plskit-py/src/lib.rs
R        pls1_fit(...)                         plskit-r/R/stubs.R
           └─ .plskit_call(fn, inputs)           plskit-r/R/call.R
                └─ plskit_call_impl              plskit-r/src/rust/src/lib.rs
Julia    pls1_fit(...)                         plskit-jl/src/generated.jl
           └─ _call(:pls1_fit; ...)              plskit-jl/src/call.jl
                └─ the Python function, through PythonCall.jl

Python and R then share one path:

plskit_bind::call(name, record)                plskit-bind/src/registry.rs
  └─ plskit::pls1_fit(...)                       plskit-rs/src/fit.rs
```

A Rust caller uses the functions of `plskit` directly.

## The engine (`plskit-rs`)

`src/lib.rs` declares the modules and re-exports the public surface.

| Module | Holds |
|---|---|
| `fit` | `pls1_fit`, `spls1_fit`, the PLS1 kernel, weight validation, the thread and memory settings |
| `predict` | `pls1_predict` |
| `preprocess` | `preprocess` and its multi-column form `preprocess_block` |
| `find_k` | `pls1_find_k_optimal`, `pls1_find_k_sequence` and the `spls1_find_*` functions |
| `signal_test` | `pls1_confirmatory_test`, `split_nb_gate` |
| `perm_null`, `perm_alpha` | `pls1_perm_null`, `pls1_perm_alpha` |
| `pls3`, `pls3_signal_test` | `pls3_fit`, `pls3_transform`, `spls3_fit`, the `plssvd_*` aliases, `pls3_confirmatory_test` |
| `rotate`, `rotation_stability` | `rotate`, `pls1_rotation_stability` |
| `error` | `PlsKitError`, with a stable `code()` string per variant |
| `linalg`, `resample`, `rng` | shared helpers: products and decompositions, split and permutation indices, seeded generators |
| `dual_route`, `gram_p`, `sequential`, `subsample` | crate-internal: Gram-matrix routes for the resampling loops, the sequential tests, the subsampling engine |

Rules that hold across the crate:

- **No `unsafe`.** The workspace lint is `unsafe_code = "forbid"`.
- **Borrowed inputs, owned outputs.** Functions take `MatRef` / `ColRef`
  views and return structs that own their matrices.
- **Errors, not panics, for bad input.** Functions return `PlsKitResult<T>`.
- **One thread-limit read per call.** A public function wraps its body in
  `fit::with_thread_limit`, which reads `PLSKIT_NUM_THREADS` and
  `PLSKIT_MEMORY_BUDGET` once. Inner fits call the `_impl` functions and
  never read them again ([Threads](../concepts/threads.md)).
- **Results do not depend on the thread count**
  ([Dependencies](dependencies.md) has the two rules behind this).
- **The route is not part of the API.** The engine picks the primal or
  the Gram route from the shape of the problem.

## The binding layer (`plskit-bind`)

A wrapper converts its values into a `Record`, calls `call` with a
function name, and converts the `Outcome` or the `BindError` back.
`plskit-bind` holds what would otherwise be written once per language:

- the registry of public functions with their parameters, kinds and
  defaults (`registry.rs`);
- argument checks and the parsing of method names and `args`;
- the declared shape of every result type (`types.rs`);
- warning texts, and the mapping from `PlsKitError` to a code, a message
  and details.

`call` catches a panic below it and returns it as an `internal` error.

`plskit-bind-dump` prints the registry as JSON, and three scripts read it:

| Script | What it does |
|---|---|
| `scripts/check-bind-registry.py` | checks the registry against the Python signatures in `_api.py` |
| `scripts/render-r-stubs.py` | writes `plskit-r/R/stubs.R` |
| `scripts/render-jl-stubs.py` | writes `plskit-jl/src/generated.jl` |

CI fails when a committed stub file differs from a fresh render.

## Result objects

| Layer | Form | Example |
|---|---|---|
| engine | a Rust struct with faer fields | `Pls1Model` |
| binding | a `Record` tagged with a result type name | `"PLS1Result"` |
| wrapper | the language's own object | Python frozen dataclass `PLS1Result`, R S3 class `pls1_result`, Julia `PLS1Result` |

Field names are the same in every language. The engine's struct fields can
differ; [Result objects (Rust)](../rust/results.md) maps them.

## Tests

- Unit tests sit next to the code, in each module.
- `plskit-rs/tests/` holds the integration tests: the corpus test, parity
  across thread counts and memory budgets, and weight identities.
- `plskit-bind/tests/` tests the binding layer.
- Each wrapper has its own suite, and each reads the shared corpus
  ([Test data](testdata.md)).
