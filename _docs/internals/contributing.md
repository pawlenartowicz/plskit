# Contributing

The contributor guide is the top-level
[CONTRIBUTING.md](../../CONTRIBUTING.md). It says where pull requests and
issues go, how versions and release tags work, and which slow test to run
before a release. This page points to the rest.

## Where things are documented

| Task | Page |
|---|---|
| Build and test the Rust crate | [Rust installation](../rust/installation.md#from-a-checkout) |
| Build and test the Python package | [Python installation](../python/installation.md#from-a-checkout-contributors) |
| Build and test the R package | [R installation](../r/installation.md) |
| Build and test the Julia package | [Julia installation](../julia/installation.md#from-a-checkout-contributors) |
| Native-CPU builds | [Building from source](building.md) |
| How the crates and packages fit together | [Architecture](architecture.md) |
| Dependencies and the rules for using them | [Dependencies](dependencies.md) |
| Using and adding test fixtures | [Test data](testdata.md) |
| Argument names | [Naming conventions](naming.md) |

## What CI checks

Run these from the workspace root before opening a pull request:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
python3 scripts/check-docs-drift.py      # exported names and result fields against _docs/
python3 scripts/check_corpus_hash.py     # testdata/ files against manifest.json
```

After a change to the `plskit-bind` registry, render the R and Julia stubs
again. CI fails when a committed stub differs from a fresh render:

```bash
python3 scripts/render-r-stubs.py
python3 scripts/render-jl-stubs.py
```

CI also builds and tests the workspace on Rust 1.85, the minimum supported
version, and runs the Python, R and Julia test suites.
