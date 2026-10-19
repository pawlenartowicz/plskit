# Installation

Python 3.11 or later.

```bash
pip install plskit
```

The only runtime dependency is `numpy>=1.23`. There are no optional
extras.

PyPI carries wheels for Linux x86_64, macOS arm64 and Windows x86_64. They
are built against Python's stable ABI, so one wheel per platform serves
every supported Python version. On any other platform pip builds from the
source distribution, which needs Rust 1.85 or later
([rustup](https://rustup.rs)).

## Check the install

```python
import numpy as np
import plskit

print(plskit.__version__)

rng = np.random.default_rng(0)
X = rng.standard_normal((40, 5))
y = rng.standard_normal(40)
print(plskit.pls1_fit(X, y, k=2).k_used)     # 2
```

## From a checkout (contributors)

`pyproject.toml` sits at the workspace root and points maturin at
`plskit-py/`. Build from the root, into a virtualenv:

```bash
git clone https://github.com/pawlenartowicz/plskit
cd plskit
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop
(cd plskit-py && pytest)
```

- The build uses the release profile. Re-run `maturin develop` after
  changing Rust code.
- `tests/test_corpus.py` reads the shared corpus in `testdata/`; see
  [Test data](../internals/testdata.md).
- `plskit.__version__` reads `0.0.0+unknown` when Python finds the package
  files but no installed distribution, for example with `plskit-py/python`
  on `PYTHONPATH`. Install with `maturin develop` or `pip install .`.

## Next

- [Quickstart](quickstart.md): fit, test and PLS3 on synthetic data
- [API reference](api.md)
- [Threads](../concepts/threads.md): `PLSKIT_NUM_THREADS` and `PLSKIT_MEMORY_BUDGET`
