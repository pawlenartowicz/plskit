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

```bash
cargo run -p plskit-testdata-gen -- --testdata-root plskit/testdata
```

Regeneration is PR-gated. The commit must explain *why* (bug fix,
algorithmic change, new fixture, new field).

## Tolerance

Defaults: scalars `atol=1e-12`, arrays `atol=1e-10`.
