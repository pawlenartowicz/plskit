# Test data

`testdata/`, at the workspace root, is the shared regression corpus: input
arrays and the outputs the Rust engine produced for them. Every test suite
replays the corpus and compares its own results with the stored ones. That
is how the repository checks that Rust, Python, R and Julia return the
same numbers.

[`testdata/README.md`](../../testdata/README.md) describes the file
layout, the regeneration command, settle mode and the tolerances. This
page covers how the test suites use the corpus and how to add to it.

## What a case is

`manifest.json` lists the cases. Each case names one public function, an
inputs file (`inputs/<name>.npz`), an outputs file
(`outputs/<function>/<name>.npz`), the `kwargs` of the call, and the
SHA-256 of both files.

The outputs come from the Rust engine only, through the generator crate
`plskit-testdata-gen`. No wrapper writes to the corpus. Comparisons with
other packages are not stored here; the `validation/` tool computes them
live.

## How each test suite reads it

| Suite | File | Cases run | Finds `testdata/` by | When it is missing |
|---|---|---|---|---|
| Rust | `plskit-rs/tests/corpus.rs` | all | the parent of the crate directory | the test fails |
| Python | `plskit-py/tests/test_corpus.py` | all | `../../testdata` from the test directory | calls `pytest.skip` |
| R | `plskit-r/tests/testthat/test-corpus.R` | all | `PLSKIT_TESTDATA`, else the first ancestor directory that holds `testdata/manifest.json` | skipped; an error when `PLSKIT_REQUIRE_CORPUS` is set |
| Julia | `plskit-jl/test/corpus.jl` | one per function | `PLSKIT_TESTDATA`, else `../../testdata` from the test directory | skipped; an error when `PLSKIT_REQUIRE_CORPUS` is set |

Each suite makes the call the generator made, with the options from the
case's `kwargs`, and compares every stored field at the corpus tolerances
(["Tolerance"](../../testdata/README.md#tolerance)). Julia runs one case
per function because its numbers are the Python wheel's; its test checks
the conversion.

Guards against a fixture that nothing checks:

- Rust: a case fails when it leaves a stored field unchecked, and
  `all_manifest_functions_are_covered` fails when the manifest names a
  function that `corpus.rs` has no arm for.
- Python: a case fails when the fixture holds a key no arm reads.
- R and Julia: every fixture key must resolve to a result field by name.
- CI sets `PLSKIT_REQUIRE_CORPUS=1` for the R and Julia jobs, and runs
  `scripts/check_corpus_hash.py`, which fails when a file's hash differs
  from the one in `manifest.json`.

## Adding a fixture

1. **Write the case.** Add a `pub fn` to the file of its function in
   `plskit-testdata-gen/src/cases/`; `pls1_predict.rs` is a short example.
   The function builds the inputs (`synth_data` and `synth_xy` draw seeded
   synthetic data), writes them with `NpzWriter`, calls the engine, writes
   the output fields, and returns a `Case`.
2. **Fill in `kwargs`** with the public argument names. They are the
   options the test suites pass to the wrapper function. Keys that only
   describe the data (`n`, `d` and the like) are dropped by each suite
   before the call; a new key of that kind has to be added to those
   filters.
3. **Register the case** in `all_cases` in
   `plskit-testdata-gen/src/cases/mod.rs`.
4. **Run the generator** from the workspace root:

   ```bash
   cargo run --release -p plskit-testdata-gen -- --testdata-root testdata
   ```

5. **Check the test suites.** A new case of an existing function with the
   same output fields needs no test change. A new output field needs a
   comparison in `corpus.rs` and `test_corpus.py`. A new function needs an
   arm in both, an entry in `COVERED_FUNCTIONS` in `corpus.rs`, and a case
   in `CORPUS_CASES` in `corpus.jl`.
6. **Commit** the new `.npz` files and `manifest.json` together with the
   generator change.

## Changing an existing fixture

- A fixture's inputs are immutable per fixture id.
- Outputs change only by running the generator. It rewrites a file only
  when the result moved beyond the tolerance, so unrelated fixtures stay
  byte-identical
  (["Regenerating: settle mode"](../../testdata/README.md#regenerating-settle-mode)).
- Regeneration goes through a pull request, and the commit must say why.
- `scripts/diff_corpus.py REFERENCE CANDIDATE` compares two corpus trees
  field by field and prints the largest difference per field.
