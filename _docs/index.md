# plskit

`plskit` is a cross-language Partial Least Squares library: a Rust core
(`cargo add plskit`) with thin wrappers for Python (`pip install plskit`),
R, and Julia. Its distinguishing feature is *modern inference* (the
`split_exact` and `split_nb` tests and rotation-invariant subsampling
CIs), alongside a compatibility layer for legacy outputs.

## Families

- [PLS1](concepts/PLS1/index.md): single-response regression (`pls1_*`)
- [sPLS1](concepts/sPLS1/index.md): sparse PLS1 with a per-component `keep` count (`spls1_*`)
- [sPLS3](concepts/sPLS3/index.md): sparse PLS3 with per-side `keep_X` / `keep_Y` counts (`spls3_*`)
- [PLS3 / PLSSVD](concepts/PLS3/index.md): symmetric two-block PLS, also known as PLSC (`pls3_*`, `plssvd_*`)

## Sections

- [Concepts](concepts/index.md): language-agnostic methods and methodology
- [Python](python/index.md): install, quickstart, API
- [Rust](rust/index.md): install, quickstart, API
- [R](r/index.md): placeholder (planned)
- [Julia](julia/index.md): placeholder (planned)
- [Internals](internals/index.md): contributor docs and `plskit-rs` internals
