# R

The R package `plskit` wraps the Rust engine through
[extendr](https://extendr.github.io/). Every function has the name and the
arguments of its Python counterpart (RULE 1), and results carry the same
field names, so the canonical reference is the Python one:
[API](../python/api.md) and [result objects](../python/results.md). These
pages cover only what is specific to R.

> Status: implemented, not yet released. Until the first release the
> package builds from a checkout of the monorepo only; see
> [Installation](installation.md).

## Pages

- [Installation](installation.md): requirements and the dev-mode build
- [Quickstart](quickstart.md): a first session
- [Differences from Python](differences.md): type mapping, seeds,
  conditions, and the few places where R differs
