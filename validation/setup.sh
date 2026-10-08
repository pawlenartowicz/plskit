#!/usr/bin/env bash
# One-time per host: .venv with plskit-py (built from this checkout) + the reference packages, and the plskit-rs bench.
# On a shared box: NICE="nice -n 19" ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
NICE=${NICE:-}
[[ -x .venv/bin/python ]] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy scikit-learn ikpls
$NICE .venv/bin/pip install -q --force-reinstall --no-deps ..   # release wheel via maturin
(cd rust && $NICE cargo build --release)
.venv/bin/python -c "import plskit, sklearn, ikpls; print('plskit-py', plskit.__version__, '| sklearn', sklearn.__version__)"
