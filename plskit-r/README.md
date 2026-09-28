# plskit-r

R wrapper for the [plskit](https://github.com/pawlenartowicz/plskit)
Rust crate, planned via `extendr`.

## What is plskit?

plskit is a Partial Least Squares (PLS) library built around three ideas:

- **Modern inference.** Confirmatory tests use split-half held-out
  prediction: `split_exact` is calibrated by permutation, and
  `split_nb` is its faster Fisher-z approximation. Confidence
  intervals are taken on rotation-invariant quantities (held-out
  correlation, per-variable leverage, a subsampling z for β), so no
  Procrustes alignment is needed. The tests are introduced in our
  [NeurIPS 2026 paper](https://openreview.net/forum?id=xb6CB7d9LO).
- **Same results in every language.** All numerical work runs in one
  Rust engine. Python (and the planned R and Julia wrappers) call
  into it, so a fixed `(version, seed, X, y)` gives the same results,
  within a bit-near tolerance, in every language and on every
  supported platform.
- **Speed.** The engine is written in Rust with parallel resampling,
  which makes plskit probably the fastest PLS package available.

## Status

**Not functional in v0.1.0** — this directory is a placeholder. No
package is built, installed, or published yet. Track progress on the
[issue tracker](https://github.com/pawlenartowicz/plskit/issues).

For a working wrapper today, use the Python package
(`pip install plskit`) or the Rust crate (`cargo add plskit`).
