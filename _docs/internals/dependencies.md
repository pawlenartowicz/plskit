# Dependencies

What each package depends on, and the rules the engine follows when it
uses them. Shared version requirements are declared once, in
`[workspace.dependencies]` of the root `Cargo.toml`.

## The engine (`plskit-rs`)

| Crate | Requirement | Used for |
|---|---|---|
| `faer` | `0.24` | dense linear algebra |
| `procrustes` | `0.1.0` | signed-permutation alignment in `pls1_rotation_stability` |
| `rand`, `rand_chacha` | `0.10` | `ChaCha8Rng`, the one generator the engine uses |
| `getrandom` | `0.4` | OS entropy when the caller passes no seed |
| `rayon` | `1.8` | thread pools and parallel loops |
| `thiserror` | `2` | the `PlsKitError` enum |

The crate has no feature flags and no optional dependencies. Tests also
use `approx`, `proptest`, `serde_json`, `ndarray`, `ndarray-npy` and `zip`;
the last four read the `.npz` fixtures of the [test corpus](testdata.md).

## faer

`faer` is the only linear-algebra dependency. `Cargo.lock` holds no BLAS or
LAPACK crate.

- **It is part of the public API.** Engine functions take and return
  `faer` matrices, and the crate re-exports `Mat`, `MatRef`, `Col` and
  `ColRef`. A `faer` minor bump is therefore a `plskit` major bump. The
  requirement `0.24` accepts patch releases only.
- **SIMD is chosen at run time.** `faer` dispatches its kernels through
  `pulp`, so one binary uses the instruction sets of the machine it runs
  on ([Building from source](building.md)).
- **The engine never reads faer's global parallelism.** That setting
  follows the size of the Rayon pool, and the degree changes how a
  reduction is split, so results would round differently at different
  thread counts. Engine code passes an explicit `Par` to the helpers in
  `linalg.rs`: `Par::Seq` inside resampling workers and a fixed degree of
  8 elsewhere. `clippy.toml` lists the `faer` entry points that read the
  global setting under `disallowed-methods`. The operator `*` on `faer`
  matrices reads it too and cannot be listed;
  `tests/thread_count_parity.rs` catches that at run time.

## Random numbers

All randomness comes from `ChaCha8Rng` (`plskit-rs/src/rng.rs`).

- A call with no seed draws one from the operating system through
  `getrandom`. The result reports the seed that was used.
- A resampling loop draws one child seed per iteration from the parent
  generator, in order, before any parallel work starts. Iteration `i` uses
  the same seed whichever thread runs it.

This is why a fixed seed gives the same bytes at any thread count
([Threads](../concepts/threads.md)). `tests/thread_count_parity.rs` runs
the same calls in pools of 1, 3 and 5 threads and requires bitwise
identical outputs.

## procrustes

`procrustes` is a separate crate, taken from crates.io. The engine uses
`procrustes::signed_permutation` to align each resampled weight matrix to
its reference in `rotation_stability.rs`. `procrustes::orthogonal` is on
the clippy `disallowed-methods` list, because its SVD reads faer's global
parallelism; `linalg::orthogonal_rotation` computes the same rotation on
`Par::Seq`.

## Wrappers and tools

| Package | Depends on |
|---|---|
| `plskit-bind` | `plskit` at exactly the workspace version (`=0.7.0`), `faer` |
| `plskit-py` (Rust side) | `plskit-bind`, `pyo3` 0.28 (`abi3-py311`), `numpy` 0.28, `faer`; built with `maturin` |
| `plskit` (Python package) | Python 3.11 or later, `numpy>=1.23` |
| `plskit-r` (Rust side) | `extendr-api` 0.9, `plskit` and `plskit-bind` at `=0.7.0`; its own lock file, kept in step by `scripts/check-profile-sync.py` |
| `plskit` (R package) | R 4.2 or later, Rust 1.85 or later to build; suggests `bit64`, `jsonlite`, `testthat` |
| `PLSKit.jl` | Julia 1.10 or later, `PythonCall` 0.9.36; `CondaPkg.toml` pins `plskit==0.7.0` from PyPI |
| `plskit-testdata-gen` | the engine and `.npz` writers; never published |

Python, R and Julia callers never see `faer`, `plskit-bind` or `pyo3`
types. The wrappers convert them to the arrays of their own language.

## Updates

- `Cargo.lock` is committed.
- Dependabot opens cargo update pull requests weekly, with the `faer*`
  crates grouped into one.
- The minimum Rust version, 1.85, is `rust-version` in the root
  `Cargo.toml`. A CI job builds, lints and tests the workspace on it.
