# Julia

`PLSKit.jl` is the Julia wrapper. It runs the Python `plskit` package
through [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl), so every
function, argument name and result field is the Python one, and every
number is the Python wheel's.

## Pages

- [Installation](installation.md): the private Python environment, or your own Python
- [Quickstart](quickstart.md): fit, test and PLS3 on synthetic data
- [Differences from Python](differences.md): call shape, results, errors, warnings, limits

The [Python API reference](../python/api.md) and the
[result objects](../python/results.md) are the reference for Julia too:
names, defaults and fields are identical.

## Version

`PLSKit.jl` carries the version of the Python `plskit` it runs
(`PLSKit.PLSKIT_PY_VERSION`, currently 0.6.2). A Julia release follows the
matching Python release and never leads it.
