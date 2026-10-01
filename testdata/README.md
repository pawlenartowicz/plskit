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
`Cargo.toml`), on any host:

```bash
cargo run --release -p plskit-testdata-gen -- --testdata-root testdata
```

Regeneration is PR-gated. The commit must explain *why* (bug fix,
algorithmic change, new fixture, new field).

## Regenerating: settle mode

The generator writes every case into a scratch staging directory, then
settles it onto `--testdata-root` (`plskit-testdata-gen/src/settle.rs`). A
staged `.npz` replaces the committed file only when it is new or no longer
equivalent under the tolerances below: same entry names, dtypes and shapes,
`f64` entries within `atol_scalar` (0-D) or `atol_array` plus `rtol` times the
committed value, NaN equal to NaN, an infinity equal only to itself, integer
and string entries exactly equal. Otherwise the committed bytes stay. A case's
input goes in with its output: when the output moved, the staged input
replaces the committed one even if it alone is within tolerance, so no output
is committed next to an input it was not computed from. The generator prints
every file it added or replaced.

This makes regeneration host-independent. The LLVM backend, FMA contraction
and libm differ between hosts, so a raw regeneration on `aarch64-apple-darwin`
moves the last bits of 41 output files the change never touched, all well
inside tolerance (the "bit-near across platforms" contract). Settle mode keeps
those files byte-identical, so the diff holds only the fixtures a change
actually moved or added.

- A clean regeneration of an unchanged checkout writes no file and leaves
  `manifest.json` byte-identical, on any host. Use that as the sanity check.
- `producing_version` is bumped only when a settle writes a file or changes
  the case list.
- Drift inside tolerance is never written, so a behaviour change smaller
  than the tolerance does not update its fixture (the tests pass either way).
  Committed files can therefore come from different hosts and versions.

## Tolerance

A float field matches when `|actual - expected| <= atol + rtol * |expected|`
for a finite `expected` (numpy's `assert_allclose` rule), NaN equal to NaN
and an infinity equal only to itself. Defaults: scalars `atol=1e-12`, arrays
`atol=1e-10`, and `rtol=1e-14` for both. Integer and string fields match
exactly.

`atol` covers small and O(1) values, where observed cross-host drift is at
most `1.4e-14` (scalars) and `6e-14` (arrays) absolute, well inside it
(about 70x for scalars, over 1000x for arrays). `rtol` only matters where
`rtol * |expected|` exceeds `atol`: above about 100 for scalars and `1e4` for
arrays. There `atol` alone allows only a bit or two. One ulp of a float in
`[4096, 8192)` is 9.1e-13, so a score statistic of about 6126 fits one ulp
of drift inside `atol=1e-12` and not two, and `atol_array` falls below one
ulp from `2^19` (about 5.2e5). In an `aarch64-apple-darwin` regeneration
against the committed corpus, the scalars in that range (the score
statistics and `n_eff` values) are bit-equal, and array entries above 100
drift by at most one ulp (e.g. `bic_scores` around 274).
`rtol=1e-14` (about 45 machine epsilons, 68 ulps at 6126) absorbs that with
headroom and is still four orders of magnitude below the smallest change a
fixture exists to catch.

Every wrapper's corpus test applies these numbers (`plskit-rs/tests/corpus.rs`,
`plskit-py/tests/test_corpus.py`, `plskit-bind/tests/corpus.rs`,
`plskit-r/tests/testthat/test-corpus.R`, `plskit-jl/test/corpus.jl`), and the
settle step reads them from each case's manifest `tolerance`.
