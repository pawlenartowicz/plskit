# plskit (Rust)

[![crates.io](https://img.shields.io/crates/v/plskit.svg)](https://crates.io/crates/plskit)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](../LICENSE)

Rust crate for **plskit** — Partial Least Squares with modern
inference (canonical percentile CIs, `split_perm`, `split_nb`). This
is the **canonical implementation**: every plskit language wrapper
(Python today; R and Julia planned) calls into this crate.

> Part of the **[plskit project](https://github.com/pawlenartowicz/plskit)** —
> sibling wrappers, shared test corpus, issues, and PRs all live there.

## Install

```
cargo add plskit
```

Minimum supported Rust version: **1.85**.

## A 60-second look

```rust
use faer::{Col, Mat};
use plskit::{pls1_fit, FitOpts, KSpec};

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

Confirmatory testing, K-selection, and rotation use the same input
shape; the public surface is listed below.

## Public surface

- `preprocess`
- `pls1_fit`, `pls1_predict`
- `pls1_confirmatory_test`
- `pls1_find_k_optimal`, `pls1_find_k_sequence`
- `pls1_perm_null`
- `pls1_rotation_stability`
- `rotate`
- `PlsKitError` — error type with a stable `.code()` for programmatic
  handling

Each function exports its own opt and output types alongside it
(`FitOpts`, `KSpec`, `Pls1Model`, `ConfirmatoryArgs`, `RotateOutput`,
…).

These names mirror the Python (and forthcoming R / Julia) wrappers, so
multi-language code is easy to read across the family.

## faer types in the public API

The public API uses `faer::Mat` and `faer::MatRef` directly. The
crate's `Cargo.toml` pins `faer` so downstream consumers do not need
to manage that dependency themselves; a `faer` minor bump is treated
as a `plskit` major bump.

## Canonical implementation

All numerical computation, randomness, parallelism, and resampling
loops live in this crate. The wrappers are thin FFI shells: they
convert their language's array types to `f64` slices, call into the
engine, and wrap the result. If you want bit-near identical results
from Python, R, or Julia, this is what they are calling.

## Versioning

The engine and each language wrapper carry their own version number,
and **the same version number always means the same features** —
`plskit-rs 0.5.0` and `plskit (Python) 0.5.0` ship the same API. The
Python, R, or Julia version may lag behind the Rust engine while its
surface is being built out, but it can never run ahead of it. Releases
use `vX.Y.Z` for the engine and `vX.Y.Z-py` / `-r` / `-jl` for the
wrappers.

API wiring (function names, argument names, result fields) is stable
across versions; pin the version if you need numerical
reproducibility.

## Citation

Lenartowicz, P., Plisiecki, H. (2026). *Cheap Per-Component Testing for
PLS, Stable Under Rotation* (Under Review).

## License

GPL-3.0-or-later.
