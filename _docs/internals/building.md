# Building from source

plskit ships generic binaries: release profiles leave `target-cpu` unset so
crates and wheels run on any compatible CPU, and fast-math-style flags are
never used — IEEE 754 semantics are part of the reproducibility contract.

## Native-CPU builds

For a build that only ever runs on the machine compiling it, you can opt in
to native codegen:

```bash
# Rust crate
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Python wheel (writes the .whl to target/wheels/; pip install it)
RUSTFLAGS="-C target-cpu=native" maturin build --release
```

Measured effect on AVX2 hardware (Intel Arrow Lake-H, n=5000, d=50000):
about 14.6% faster single-threaded fits (`Par::Seq`) and 5.5% with
`Par::Auto`. The gain is modest because faer dispatches SIMD kernels at
runtime via `pulp`, so the static flag mostly accelerates non-faer code.
Expect a larger win on AVX-512 hardware, which the generic x86-64 baseline
cannot use at all.

Never distribute native-built artifacts: they fail with `SIGILL` on CPUs
older than the build machine.
