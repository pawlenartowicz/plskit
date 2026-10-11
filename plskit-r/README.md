# plskit-r

R package for [plskit](https://github.com/pawlenartowicz/plskit): PLS
regression with modern inference. It wraps the Rust engine through
extendr and the `plskit-bind` layer, with the same functions, argument
names and result fields as the Python package. The engine fits PLS1
with [Improved Kernel PLS](../_docs/concepts/PLS1/fit-and-predict.md#how-plskit-computes-it)
(Dayal and MacGregor 1997).

## Status

Implemented, not yet released. Until the first release it builds only
from a checkout of the monorepo, in dev mode: see
[`_docs/r/installation.md`](../_docs/r/installation.md).

## `k`, not `ncomp`

The number of components is `k` (and `k_max`), as in every plskit
language, not `ncomp` as in the R `pls` package. This is deliberate
(cross-language naming rule); there is no alias.

## Documentation

- [R pages](../_docs/r/index.md): installation, quickstart, differences
  from Python (type mapping, string seeds, condition classes)
- [Python API](../_docs/python/api.md): the canonical reference for every
  function and argument
