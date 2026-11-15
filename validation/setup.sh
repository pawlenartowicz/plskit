#!/usr/bin/env bash
# One-time per host: .venv with plskit-py (built from this checkout) + the reference packages, and the plskit-rs bench.
# On a shared box: NICE="nice -n 19" ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
NICE=${NICE:-}
[[ -x .venv/bin/python ]] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy scikit-learn ikpls h5py scipy tqdm "setuptools<81"   # pyls imports pkg_resources
# pyls is not on PyPI under this name, and its build-time versioneer fails on Python >= 3.12: clone and put it on the path, no build.
PYLS_COMMIT=d8a19d564cc5804249527b68c937df3a5fd8c7cc
[[ -d .venv/src/pyls ]] || git clone -q https://github.com/rmarkello/pyls .venv/src/pyls
git -C .venv/src/pyls checkout -q "$PYLS_COMMIT"
echo "$PWD/.venv/src/pyls" > "$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/pyls.pth"
$NICE .venv/bin/pip install -q --force-reinstall --no-deps ..   # release wheel via maturin
(cd rust && $NICE cargo build --release)
.venv/bin/python -c "import plskit, sklearn, ikpls, pyls; print('plskit-py', plskit.__version__, '| sklearn', sklearn.__version__)"
