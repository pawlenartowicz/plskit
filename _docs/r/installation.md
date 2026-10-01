# Installing the R package

## Requirements

- R 4.2 or newer.
- Rust 1.85 or newer with `cargo` (install with [rustup](https://rustup.rs)).
- Windows: Rtools matching your R version, and the GNU Rust target:
  `rustup target add x86_64-pc-windows-gnu`.

## Building from a checkout (until the first release)

The package's Rust crate depends on the `plskit` and `plskit-bind` crates
from crates.io. Until `plskit-bind` is published there, the package builds
only in dev mode: a cargo config file outside the package patches both
crates to a checkout of the monorepo, and the environment variable
`PLSKIT_CARGO_CONFIG` names that file.

```sh
git clone https://github.com/pawlenartowicz/plskit
cd plskit
printf "[patch.crates-io]\nplskit = { path = '%s/plskit-rs' }\nplskit-bind = { path = '%s/plskit-bind' }\n" "$PWD" "$PWD" > "$HOME/plskit-dev.toml"
PLSKIT_CARGO_CONFIG="$HOME/plskit-dev.toml" R CMD INSTALL plskit-r
```

From an R session, set the variable first:

```r
Sys.setenv(PLSKIT_CARGO_CONFIG = path.expand("~/plskit-dev.toml"))
install.packages("plskit-r", repos = NULL, type = "source")
```

Keep the config file outside `plskit-r/`: `R CMD build` packs everything
in that directory, and the paths in the file are specific to your machine.
Use absolute paths; on Windows write them with forward slashes.

The first build compiles the engine and takes a minute or two; later
builds are incremental.

## What the build does

`src/Makevars` (`Makevars.win` on Windows) runs `tools/cargo-build.R`,
which in dev mode:

1. writes a copy of `src/rust/Cargo.toml` to `src/rust/target/dev-crate/`
   with the `plskit` and `plskit-bind` requirements relaxed, so the patch
   applies (the committed manifest is never edited);
2. starts the copy's lock file from the monorepo's `Cargo.lock`, so the
   engine's dependencies have the versions the workspace tests;
3. runs `cargo build --release --config $PLSKIT_CARGO_CONFIG`;
4. checks the lock file: the build fails if the patch went unused or if
   either crate came from crates.io instead of the checkout.

The release profile is the workspace's (`codegen-units = 1`,
`lto = "thin"`, `panic = "unwind"`), which the engine's determinism
contract relies on.

## When the build fails

| Message | Cause | Fix |
|---|---|---|
| `rust/Cargo.lock is not committed yet, so plskit builds only in dev mode` | `PLSKIT_CARGO_CONFIG` is not set | set it as above |
| `PLSKIT_CARGO_CONFIG names a missing file` | wrong path | use an absolute path |
| ``no matching package named `plskit-bind` found`` | the config patches only `plskit` | patch both crates |
| `plskit resolved from a registry, not from the patch` | the config patches only `plskit-bind` | patch both crates |
| `the [patch.crates-io] entries ... were not used` | the paths do not point at the monorepo crates | fix the paths |
| `cargo not found` | Rust is not installed or not on `PATH` | install with rustup |

## Running the tests

The test suite includes the shared `testdata/` corpus. From the monorepo
root, after installing:

```sh
Rscript -e 'testthat::test_dir("plskit-r/tests/testthat", package = "plskit", load_package = "installed")'
```

The corpus test finds `testdata/` by walking up from the test directory;
set `PLSKIT_TESTDATA` to point at it from elsewhere.
