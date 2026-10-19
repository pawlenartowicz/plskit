# plskit (R)

[![r-universe](https://pawlenartowicz.r-universe.dev/badges/plskit)](https://pawlenartowicz.r-universe.dev/plskit)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](../LICENSE)

R package for [plskit](https://github.com/pawlenartowicz/plskit):
Partial Least Squares with modern inference, backed by a Rust engine.

## What is plskit?

plskit is a Partial Least Squares (PLS) library built around four ideas:

- **Modern inference.** Confirmatory tests use split-half held-out
  prediction: `split_exact` is calibrated by permutation, and
  `split_nb` is its faster Fisher-z approximation. Confidence
  intervals are taken on rotation-invariant quantities (held-out
  correlation, per-variable leverage, a subsampling z for β), so no
  Procrustes alignment is needed. The tests are introduced in our
  [NeurIPS 2026 paper](https://openreview.net/forum?id=xb6CB7d9LO).
- **Same results in every language.** All numerical work runs in one
  Rust engine. The Python and R wrappers call into it, and the Julia
  wrapper runs the Python package, so a fixed `(version, seed, X, y)`
  gives the same results, within a bit-near tolerance, in every
  language and on every supported platform.
- **Speed.** The engine is written in Rust and fits PLS1 with
  [Improved Kernel PLS](../docs/concepts/PLS1/fit-and-predict.md#how-plskit-computes-it)
  (Dayal and MacGregor 1997), which makes plskit probably the fastest
  PLS package on a single core.
- **Parallel and RAM-safe.** Inference runs efficiently in parallel
  without blowing up RAM.

## Install

Requires R 4.2 or newer. Install a release from r-universe:

```r
install.packages("plskit", repos = "https://pawlenartowicz.r-universe.dev")
```

or from the `r-universe` branch of the GitHub repository:

```r
devtools::install_github("pawlenartowicz/plskit", ref = "r-universe")
```

r-universe serves binaries for Windows, macOS and Linux, which need no
Rust. A source install (from GitHub, or from r-universe where it has no
binary for your R version or platform) needs Rust 1.85 or newer with
`cargo`, from [rustup](https://rustup.rs); on Windows it also needs
Rtools and the GNU Rust target (`rustup target add x86_64-pc-windows-gnu`).

A checkout of `main` builds only in dev mode: a cargo config file
outside the package points the engine crates at the checkout.

```sh
git clone https://github.com/pawlenartowicz/plskit
cd plskit
printf "[patch.crates-io]\nplskit = { path = '%s/plskit-rs' }\nplskit-bind = { path = '%s/plskit-bind' }\n" "$PWD" "$PWD" > "$HOME/plskit-dev.toml"
PLSKIT_CARGO_CONFIG="$HOME/plskit-dev.toml" R CMD INSTALL plskit-r
```

See [installation](../docs/r/installation.md)
for what the dev-mode build does and how to fix a failed build.

## Quickstart

```r
library(plskit)

set.seed(1)
X <- matrix(rnorm(60 * 6), 60, 6)
y <- 4 * (X[, 1] + X[, 2]) + 0.5 * rnorm(60)

# Fit and predict
fit  <- pls1_fit(X, y, k = 3, seed = 42)
yhat <- pls1_predict(fit, X[1:5, ])

# Confirmatory test for any signal; k = 1 with split_exact is recommended.
test <- pls1_confirmatory_test(X, y, k = 1, test_method = "split_exact",
                               seed = 42)
test$pvalue
```

## Same names as Python

Function names, argument names and result fields are the Python ones.
The number of components is `k` (and `k_max`), as in every plskit
language, not `ncomp` as in the R `pls` package; there is no alias.

## R value types

- **Results** are named lists with an S3 class: `pls1_fit` returns a
  `c("pls1_result", "plskit_result")`. A 1-D `ndarray` becomes a numeric
  vector, a 2-D one a numeric matrix, and `None` a retained `NULL`
  element.
- **Seeds** are decimal strings, because R has no unsigned 64-bit
  integer: `result$seed` is a character scalar, and `seed = result$seed`
  reproduces the run exactly.
- **Errors** are R conditions carrying the Python error codes:
  `plskit_error`, with the subclasses `plskit_invalid_weights` and
  `plskit_resampling_degenerate`.

```r
tryCatch(pls1_fit(X, y, k = 2.5), plskit_error = function(e) e$code)
# "invalid_argument"
```

The full mapping is in
[Differences from Python](../docs/r/differences.md).

## Documentation

- [R pages](../docs/r/index.md):
  installation, quickstart, differences from Python.
- [Python API](../docs/python/api.md)
  and [result objects](../docs/python/results.md):
  the canonical reference for every function, argument and result field.

## License

GPL-3.0-or-later.

## Citation

```bibtex
@inproceedings{lenartowicz2026cheap,
  title     = {Cheap and Powerful Tests for Supervised Subspaces: Per-Component Inference for {PLS}},
  author    = {Lenartowicz, Pawe{\l} and Plisiecki, Hubert},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=xb6CB7d9LO}
}
```

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
