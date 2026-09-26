# testdata

Cross-language regression corpus for plskit. Frozen reference outputs
from the Rust core, consumed by every wrapper's parity tests.

## Layout

- `manifest.json` — manifest v2: `schema_version`, `producing_version`,
  and one entry per case with `inputs` / `outputs` / `kwargs` / `hashes`.
- `inputs/<name>.npz` — input arrays (`X`, `y`, …). Immutable per fixture id.
- `outputs/<function>/<name>.npz` — frozen outputs from the Rust core.
- `schema.json` — JSON-Schema v2 for the manifest.

## Regenerating

Run from the workspace root (the directory holding the top-level
`Cargo.toml`), on an x86_64 Linux host (see below):

```bash
cargo run -p plskit-testdata-gen -- --testdata-root testdata
```

Regeneration is PR-gated. The commit must explain *why* (bug fix,
algorithmic change, new fixture, new field).

## Regenerating: host requirement

**Regenerate on `x86_64-unknown-linux-gnu` only.** The corpus was frozen on
that target, and the blobs store exact bits. On any other host the LLVM
backend, FMA contraction and libm differ, so a regeneration rewrites fixtures
the change never touched: on `aarch64-apple-darwin` it rewrites 32
pre-existing files under `outputs/` across 15 function families. The drift
stays inside the tolerances below (worst measured: 5.68e-14 absolute,
1.42e-12 relative). That is the "bit-near across platforms" contract holding,
so *reading* the corpus is portable (CI reads it on `macos-latest` and
`windows-latest` and passes). Only *regenerating* is host-bound.

- Byte churn from a non-x86_64-Linux regeneration must not be committed, even
  though it is within tolerance. Commit only the files your change is
  supposed to move.
- A pristine checkout regenerated on x86_64 Linux reproduces every `.npz`
  file and `manifest.json` byte for byte. Use that as the sanity check before
  regenerating for real.
- Toolchain: `rust-toolchain.toml` pins `channel = "stable"` (not a specific
  version), so use a current stable `rustc` on the Linux host.
- `manifest.json` records `producing_version` only; it does not record the
  host triple or compiler version, so nothing enforces this rule
  automatically.

## Tolerance

Defaults: scalars `atol=1e-12`, arrays `atol=1e-10`.
