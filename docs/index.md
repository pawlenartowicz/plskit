# plskit

`plskit` is a cross-language Partial Least Squares library: a Rust core
(`cargo add plskit`) with thin wrappers for Python (`pip install plskit`),
R, and Julia. Its distinguishing feature is *modern inference* (the
`split_exact` and `split_nb` tests and rotation-invariant subsampling
CIs).

PLS fits one outcome from many variables, often more variables than
observations. Because its components are chosen to match the outcome, it
matches it to some degree even when the outcome is pure noise, so a good
in-sample fit is no evidence of a real effect. `plskit` is built around
the question of whether a fitted model found one. It implements the tests
from the NeurIPS 2026 paper
([Lenartowicz & Plisiecki, 2026](https://openreview.net/forum?id=xb6CB7d9LO)),
which are cheaper than the test most often used today and find real
effects with fewer observations.

## Install

**Python** 3.11 or later. Wheels for Linux x86_64, macOS arm64 and
Windows x86_64; elsewhere pip builds from source, which needs Rust.

```sh
pip install plskit
```

**Rust** 1.85 or later. No BLAS or LAPACK needed.

```sh
cargo add plskit
```

**R** 4.2 or later. r-universe serves binaries for Windows, macOS and
Linux.

```r
install.packages("plskit", repos = "https://pawlenartowicz.r-universe.dev")
```

**Julia** 1.10 or later. Not yet in the General registry, so it installs
from the repository. The first `using PLSKit` downloads its own Python
and the matching `plskit` wheel.

```julia
using Pkg
Pkg.add(url="https://github.com/pawlenartowicz/plskit", subdir="plskit-jl")
```

Each language has an installation page with platform details and a
check that the install works, and a quickstart with a first session
(fit PLS1, then test it for signal):

| Language | Installation | Quickstart |
|----------|--------------|------------|
| Python | [installation](python/installation.md) | [quickstart](python/quickstart.md) |
| Rust | [installation](rust/installation.md) | [quickstart](rust/quickstart.md) |
| R | [installation](r/installation.md) | [quickstart](r/quickstart.md) |
| Julia | [installation](julia/installation.md) | [quickstart](julia/quickstart.md) |

## How the test works

The rows are split at random into two halves, PLS is fitted on one half,
and the correlation between its predictions and the true outcome is
computed on the other. This is repeated over many splits, 50 by default,
and the correlations are averaged. `split_exact` calibrates that average
by shuffling the outcome on the same splits; `split_nb` uses the
Nadeau–Bengio corrected t-test instead. The default,
`test_method="auto"`, picks one of the two from `X` alone
([rules](concepts/PLS1/inference.md#test_methodauto)). Components are
tested one at a time, each on what the previous ones left, stopping at
the first that fails.

<img src="https://pawlenartowicz.pl/projects/plskit/inference-pipeline.svg" style="width:100%; box-sizing:border-box; background:#F1ECDF; padding:0.75rem; border-radius:8px;" alt="Diagram of the test: repeated half splits, testing components one by one, and rotation after testing">

*A:* one test, from many half splits to one number and its p-value.
*B:* components tested one by one. *C:* rotation comes after the test and
does not change its result.

[Picking a method](concepts/PLS1/fit-and-predict.md#confirmatory-testing-picking-a-method)
covers the choice between the two tests, and the
[project page](https://pawlenartowicz.pl/projects/plskit/) explains the
test step by step without formulas.

## Families

- [PLS1](concepts/PLS1/index.md): single-response regression (`pls1_*`)
- [sPLS1](concepts/PLS1/sparse.md): sparse PLS1 with a per-component `keep` count (`spls1_*`)
- [sPLS3](concepts/PLS3/sparse.md): sparse PLS3 with per-side `keep_X` / `keep_Y` counts (`spls3_*`)
- [PLS3 / PLSSVD](concepts/PLS3/index.md): symmetric two-block PLS, also known as PLSC (`pls3_*`, `plssvd_*`)

## Sections

- [Concepts](concepts/index.md): language-agnostic methods and methodology
- [Python](python/index.md): install, quickstart, API
- [Rust](rust/index.md): install, quickstart, API
- [R](r/index.md): install, quickstart, differences from Python (runs the Rust engine through `plskit-bind` and extendr)
- [Julia](julia/index.md): install, quickstart, differences from Python (implemented, not yet registered; runs the Python package through PythonCall.jl)
- [Internals](internals/index.md): contributor docs and `plskit-rs` internals

## Links

- [GitHub](https://github.com/pawlenartowicz/plskit)
- [PyPI](https://pypi.org/project/plskit/) · [crates.io](https://crates.io/crates/plskit)
- [The paper on OpenReview](https://openreview.net/forum?id=xb6CB7d9LO)
- [How to cite plskit](concepts/citation.md)
