# Installation

## Install

Julia 1.10 (the LTS) or later. `PLSKit` is not yet in the General
registry; until it is, install it from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/pawlenartowicz/plskit", subdir="plskit-jl")
```

## The Python environment

The first `using PLSKit` creates a private Python environment with CondaPkg:
a one-time download of Python, NumPy and the `plskit` wheel at the version
pinned in `plskit-jl/CondaPkg.toml`. Nothing else is needed.

PyPI carries `plskit` wheels for Linux x86_64, macOS arm64 and Windows
x86_64. On other platforms (macOS on Intel, Linux on ARM) pip builds
`plskit` from source, which needs a Rust toolchain (1.85 or later, from
[rustup](https://rustup.rs)).

## Using your own Python

Point PythonCall at a Python that has `plskit` installed, before
`using PLSKit` (in the shell, or with `ENV[...] = ...` before the first
`using`):

```bash
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/path/to/venv/bin/python   # has plskit==0.6.2
```

`using PLSKit` compares that Python's `plskit.__version__` with the version
it pins and stops with an error on a mismatch: a different `plskit` would
make the version number a false claim.

## Check the install

```julia
using PLSKit
PLSKit.PLSKIT_PY_VERSION                   # "0.6.2"
pls1_fit(randn(30, 4), randn(30)).k_used   # 1
```

## From a checkout (contributors)

Build `plskit-py` into a virtualenv and run the Julia tests against it,
from the workspace root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy && maturin develop --release
export JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE="$(which python)"
julia --project=plskit-jl -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

After changing the `plskit-bind` registry, re-render the stubs with
`python3 scripts/render-jl-stubs.py` (CI fails when the committed
`plskit-jl/src/generated.jl` differs from a fresh render).

### Running the tests

`Pkg.test()` needs the `testdata/` corpus, which lives in the monorepo
checkout and is not part of an installed `PLSKit` copy. Point
`PLSKIT_TESTDATA` at a corpus directory to run the corpus tests from
outside the checkout; when it is unset, `test/corpus.jl` falls back to
`../../testdata` relative to itself. If neither exists, the corpus
testset is skipped with an `@info` message rather than failing the whole
suite, unless `PLSKIT_REQUIRE_CORPUS` is set to `1` or `true`
(case-insensitive), in which case a missing corpus is an error instead of
a skip. The R wrapper's test suite uses the same two variable names.

### The Python pin

Each `PLSKit.jl` version pins one `plskit` release. `Project.toml`'s
`version`, the `plskit` pin in `CondaPkg.toml`, and `PLSKIT_PY_VERSION`
in `src/python.jl` are always equal; `scripts/julia-python-pin.py`
checks that. Julia may lag Python, never lead it: when `plskit-py` in
the checkout is the pinned version, CI builds it from source; when it is
newer, CI installs the pinned wheel from PyPI; when it is older, CI
fails. A `vX.Y.Z-jl` release is tested only against the PyPI wheel, so
the matching `vX.Y.Z-py` release must be published first.
