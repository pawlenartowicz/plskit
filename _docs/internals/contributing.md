# Contributing

> Status: stub. Content TBD.

- Pointer to the top-level [CONTRIBUTING.md](../../CONTRIBUTING.md) for the canonical contributor guide
- Local dev workflow: cloning, building each wrapper, running cross-language tests
- PR conventions: branch naming; versioning is per-artifact, not lock-step. Each
  of the Rust engine, Python, R, and Julia packages carries its own version
  literal. A wrapper may lag the engine (ship an older feature set at a lower
  version) but may never lead it: no wrapper version exceeds the workspace
  engine version. Release tags route by suffix: `vX.Y.Z` for the engine,
  `vX.Y.Z-py` / `-r` / `-jl` for the wrappers, each matching that artifact's
  own version literal.
- Where to file issues: bug reports, feature requests, methodology questions
- Code-of-conduct pointer
