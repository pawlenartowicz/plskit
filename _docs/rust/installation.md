# Installation

Rust 1.85 or later. That is the crate's `rust-version`, and CI builds and
tests on it.

```bash
cargo add plskit
```

Numerical results can change between versions. Pin the exact version when
you need to reproduce numbers (see
[Citation & reproducibility](../concepts/citation.md)):

```toml
[dependencies]
plskit = "=0.7.0"
```

- The crate has no feature flags.
- `plskit` re-exports faer's `Mat`, `MatRef`, `Col` and `ColRef`, so your
  own `Cargo.toml` does not need `faer`.
- No BLAS or LAPACK is needed. The linear algebra is `faer`'s.
- The crate exposes a Rust API only: there is no C ABI and no `cdylib`
  target. The Python and R wrappers reach it through `plskit-bind` (see
  [Architecture](../internals/architecture.md)).

## Check the install

```rust
use plskit::{pls1_fit, Col, FitOpts, KSpec, Mat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::<f64>::from_fn(30, 4, |i, j| ((i * 7919 + j * 104_729) as f64).sin());
    let y = Col::<f64>::from_fn(30, |i| x[(i, 0)] - 0.5 * x[(i, 1)]);
    let model = pls1_fit(x.as_ref(), y.as_ref(), KSpec::Fixed(1), None, FitOpts::default())?;
    println!("plskit {}, k_used = {}", plskit::version(), model.k_used);
    Ok(())
}
```

`cargo run --release` prints `plskit 0.7.0, k_used = 1`. Use release mode:
the numerical kernels are far slower under the dev profile.

## From a checkout

```bash
git clone https://github.com/pawlenartowicz/plskit
cd plskit
cargo test -p plskit --release
```

`rust-toolchain.toml` selects the `stable` channel for work inside the
repository. It does not pin the minimum version.
[Building from source](../internals/building.md) covers native-CPU builds.

## Next

- [Quickstart](quickstart.md): fit, test and PLS3 in one program
- [API reference](api.md), and <https://docs.rs/plskit> for every signature
- [Threads](../concepts/threads.md): `PLSKIT_NUM_THREADS` and `PLSKIT_MEMORY_BUDGET`
