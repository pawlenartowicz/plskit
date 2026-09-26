# Installation

> Status: stub. Content TBD.

- Wheel install via `pip install plskit` — supported Python versions and platforms
- Source build via `maturin develop` for contributors
- No optional extras: the only runtime dependency is `numpy>=1.23`; the test suite needs `pytest`, installed separately
- Verifying the install: a one-line smoke test importing `plskit` and printing `__version__`
- Troubleshooting common install issues (Rust toolchain absent, mismatched ABI)
