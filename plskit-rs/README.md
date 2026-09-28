# plskit (Rust)

[![crates.io](https://img.shields.io/crates/v/plskit.svg)](https://crates.io/crates/plskit)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](../LICENSE)

Rust crate for **plskit**: Partial Least Squares with modern
inference (`split_exact` and `split_nb` tests, rotation-invariant
subsampling CIs). This
is the **canonical implementation**: every plskit language wrapper
(Python today; R and Julia planned) calls into this crate.

> Part of the **[plskit project](https://github.com/pawlenartowicz/plskit)**:
> sibling wrappers, shared test corpus, issues, and PRs all live there.

## What is plskit?

plskit is a Partial Least Squares (PLS) library built around three ideas:

- **Modern inference.** Confirmatory tests use split-half held-out
  prediction: `split_exact` is calibrated by permutation, and
  `split_nb` is its faster Fisher-z approximation. Confidence
  intervals are taken on rotation-invariant quantities (held-out
  correlation, per-variable leverage, a subsampling z for β), so no
  Procrustes alignment is needed. The tests are introduced in our
  [NeurIPS 2026 paper](https://openreview.net/forum?id=xb6CB7d9LO).
- **Same results in every language.** All numerical work runs in one
  Rust engine. Python (and the planned R and Julia wrappers) call
  into it, so a fixed `(version, seed, X, y)` gives the same results,
  within a bit-near tolerance, in every language and on every
  supported platform.
- **Speed.** The engine is written in Rust with parallel resampling,
  which makes plskit probably the fastest PLS package available.

## Install

```
cargo add plskit
```

Minimum supported Rust version: **1.85**.

## A 60-second look

```rust
use plskit::{pls1_fit, Col, FitOpts, KSpec, Mat};

// Toy data: replace with your own (n × p) X and length-n y.
let x = Mat::<f64>::from_fn(200, 20, |i, j| (i as f64).sin() + (j as f64).cos());
let y = Col::<f64>::from_fn(200, |i| x[(i, 0)] + x[(i, 1)]);

let model = pls1_fit(
    x.as_ref(),
    y.as_ref(),
    KSpec::Fixed(3),
    None,                  // no observation weights
    FitOpts::default(),
)
.expect("fit failed");

println!("β = {:?}", model.beta);
```

Confirmatory testing, K-selection, sparse fits, and rotation use the
same input shape; the public surface is summarized below.

## Public surface

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
  `split_nb_gate`, `pls1_perm_null`.
- **Interpretive:** `rotate`, `pls1_rotation_stability`.
- **Errors:** `PlsKitError` (with a stable `.code()` for programmatic
  handling) and the `PlsKitResult<T>` alias.

Each function exports its own options and output types alongside it
(`FitOpts`, `KSpec`, `Pls1Model`, `Pls3Model`, `ConfirmatoryArgs`,
`RotateOutput`, ...). The
[Rust API overview](https://github.com/pawlenartowicz/plskit/blob/main/_docs/rust/api.md)
groups the surface the same way and shows side-by-side Rust / Python
calls; [docs.rs/plskit](https://docs.rs/plskit) has every signature.

These names mirror the Python (and forthcoming R / Julia) wrappers, so
multi-language code is easy to read across the family.

## faer types in the public API

The public API takes and returns `faer` matrices and columns. The crate
re-exports `Mat`, `MatRef`, `Col`, and `ColRef`, so downstream code can
`use plskit::{Mat, Col}` without adding `faer` to its own
`Cargo.toml`. A `faer` minor bump is treated as a `plskit` major bump.

## Canonical implementation

All numerical computation, randomness, parallelism, and resampling
loops live in this crate. The wrappers are thin FFI shells: they
convert their language's array types to `f64` slices, call into the
engine, and wrap the result. If you want bit-near identical results
from Python, R, or Julia, this is what they are calling.

## Versioning

The engine and each language wrapper carry their own version number,
and **the same version number always means the same features**:
`plskit-rs 0.6.0` and `plskit (Python) 0.6.0` ship the same API. The
Python, R, or Julia version may lag behind the Rust engine while its
surface is being built out, but it can never run ahead of it. Releases
use `vX.Y.Z` for the engine and `vX.Y.Z-py` / `-r` / `-jl` for the
wrappers.

API wiring (function names, argument names, result fields) is stable
across versions; pin the version if you need numerical
reproducibility.

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

## License

GPL-3.0-or-later.
