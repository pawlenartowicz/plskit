# plskit (Rust)

[![crates.io](https://img.shields.io/crates/v/plskit.svg)](https://crates.io/crates/plskit)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](../LICENSE)

The Rust engine of [plskit](https://github.com/pawlenartowicz/plskit):
Partial Least Squares with modern inference.

## What is plskit?

plskit is a Partial Least Squares (PLS) library built around four ideas:

- **Modern inference.** Confirmatory tests use split-half held-out
  prediction: `split_exact` is calibrated by permutation, and
  `split_nb` is its faster Fisher-z approximation. Confidence
  intervals are taken on rotation-invariant quantities (held-out
  correlation, per-variable leverage, a subsampling z for β), so no
  Procrustes alignment is needed. The tests are introduced in our
  [NeurIPS 2026 paper](https://openreview.net/forum?id=xb6CB7d9LO).
- **Same results in every language.** All numerical work runs in one
  Rust engine. The Python and R wrappers call into it, and the Julia
  wrapper runs the Python package, so a fixed `(version, seed, X, y)`
  gives the same results, within a bit-near tolerance, in every
  language and on every supported platform.
- **Speed.** The engine is written in Rust and fits PLS1 with
  [Improved Kernel PLS](../docs/concepts/PLS1/fit-and-predict.md#how-plskit-computes-it)
  (Dayal and MacGregor 1997), which makes plskit probably the fastest
  PLS package on a single core.
- **Parallel and RAM-safe.** Inference runs efficiently in parallel
  without blowing up RAM.

## Other languages

The Python, R and Julia packages run on this crate; see the
[root README](../README.md) for how to install them.

## Install

```
cargo add plskit
```

Minimum supported Rust version: **1.85**. Build in release mode
(`cargo run --release`): the numerical kernels are far slower under the
dev profile.

## Quickstart

```rust
use plskit::{
    pls1_confirmatory_test, pls1_fit, pls1_predict, Col, ConfirmatoryArgs,
    ConfirmatoryTestInput, ConfirmatoryTestOpts, FitOpts, KSpec, Mat, PlsKitError,
};

fn main() -> Result<(), PlsKitError> {
    // Toy data: replace with your own (n × p) X and length-n y.
    let noise = |i: usize, j: usize| ((i * 7919 + j * 104_729) as f64).sin();
    let x = Mat::<f64>::from_fn(200, 20, |i, j| noise(i, j));
    let y = Col::<f64>::from_fn(200, |i| x[(i, 0)] + x[(i, 1)] + 0.5 * noise(i, 99));

    // Fit at a fixed number of components and predict.
    let model = pls1_fit(x.as_ref(), y.as_ref(), KSpec::Fixed(3), None, FitOpts::default())?;
    let y_hat = pls1_predict(&model, x.as_ref())?;
    println!("beta[0] = {:.3}, y_hat[0] = {:.3}", model.beta[0], y_hat[0]);

    // Confirmatory test for any signal; k = 1 with split_exact is recommended.
    let test = pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw { x: x.as_ref(), y: y.as_ref(), k: 1, weights: None },
        ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact { n_perm: 1000, n_splits: 50 },
            seed: Some(42),
            ..ConfirmatoryTestOpts::default()
        },
    )?;
    println!("{}: p = {:.4}", test.test_method, test.pvalue);
    Ok(())
}
```

## Calling convention and public surface

Every function takes its data as `MatRef<f64>` / `ColRef<f64>` (call
`.as_ref()` on an owned `Mat` or `Col`), its settings as an options
struct that implements `Default`, and returns
`PlsKitResult<T>` with a plain output struct.

Three model families: PLS1 (single continuous `y`), sparse PLS1, and
PLS3 / PLSSVD (symmetric `X`/`Y` covariance analysis) with a sparse
variant. PLS2 and multi-block PLS are planned.

- **Preprocessing:** `preprocess`.
- **PLS1 fit / predict:** `pls1_fit`, `pls1_predict`; K-selection via
  `pls1_find_k_optimal`, `pls1_find_k_sequence`.
- **Sparse PLS1:** `spls1_fit`, `spls1_find_keep_optimal`,
  `spls1_find_k_optimal`, `spls1_find_k_sequence` (predict with
  `pls1_predict`).
- **PLS3 / PLSSVD:** `pls3_fit` (alias `plssvd_fit`), `pls3_transform`
  (alias `plssvd_transform`), sparse `spls3_fit` (new in 0.6.0).
- **Inference:** `pls1_confirmatory_test`, `pls3_confirmatory_test`,
  `split_nb_gate`, `pls1_perm_null`, `pls1_perm_alpha`.
- **Interpretive:** `rotate`, `pls1_rotation_stability`.

Each function exports its own options and output types alongside it
(`FitOpts`, `KSpec`, `Pls1Model`, `Pls3Model`, `ConfirmatoryArgs`,
`RotateOutput`, ...).

Every error is a `PlsKitError`. Its `.code()` is a stable string, the
same one the other languages report:

```rust
match pls1_fit(x.as_ref(), y.as_ref(), KSpec::Fixed(500), None, FitOpts::default()) {
    Err(e) if e.code() == "k_exceeds_max" => eprintln!("{e}"), // k=500 exceeds maximum 20
    Err(e) => return Err(e),
    Ok(model) => { /* ... */ }
}
```

## faer types in the public API

The public API takes and returns `faer` matrices and columns. The crate
re-exports `Mat`, `MatRef`, `Col`, and `ColRef`, so downstream code can
`use plskit::{Mat, Col}` without adding `faer` to its own
`Cargo.toml`. A `faer` minor bump is treated as a `plskit` major bump.

## Threads and memory

plskit runs its parallel work on the caller's current Rayon pool, so a
call made inside `rayon::ThreadPool::install` uses that pool's threads.
The environment variable `PLSKIT_NUM_THREADS` caps the thread count, and
`PLSKIT_MEMORY_BUDGET` sets the memory budget of the resampling loops
(default 2 GiB). Results depend on neither. See
[Threads and memory](../docs/concepts/threads.md).

## Versioning

A `faer` minor bump is a `plskit` major bump. Function names, argument
names and output fields are stable across versions, but defaults,
algorithms and implementation details may shift numbers between
releases: pin the version (`plskit = "=0.7.2"`) if you need numerical
reproducibility. The [root README](../README.md#versioning) explains how
the engine and wrapper versions relate.

## Documentation

- [docs.rs/plskit](https://docs.rs/plskit): every signature.
- [Rust pages](../docs/rust/index.md): installation, quickstart, and an
  [API overview](../docs/rust/api.md) that groups the surface as above
  and shows side-by-side Rust / Python calls.

## License

GPL-3.0-or-later.

## Citation

```bibtex
@inproceedings{lenartowicz2026cheap,
  title     = {Cheap and Powerful Tests for Supervised Subspaces: Per-Component Inference for {PLS}},
  author    = {Lenartowicz, Pawe{\l} and Plisiecki, Hubert},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=xb6CB7d9LO}
}
```

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
