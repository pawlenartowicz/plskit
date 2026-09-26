# Installation

> Status: stub. Content TBD.

- `cargo add plskit` for the latest released version
- Rust toolchain requirements: MSRV is Rust 1.85, declared as `rust-version` in the workspace `Cargo.toml` and tested in CI. (`rust-toolchain.toml` only selects the `stable` channel for building the repo; it does not pin the MSRV.)
- Feature flags exposed by the crate and what each gates
- Linking and FFI considerations for downstream wrappers
- Verifying the install: a one-call doctest fitting a tiny model
