#!/usr/bin/env bash
# ./run.sh [--save]
# Single-core fit timings for this host, printed as a report. Every library is timed on
# every run. --save also writes the run to results/<host>/.
set -euo pipefail
cd "$(dirname "$0")"
source single.env
PY=.venv/bin/python
HOST=$($PY -c 'import common; print(common.host())')
OUT=$(mktemp -d)
trap 'rm -r "$OUT"' EXIT
WHEN=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  echo "run started $WHEN"
  uname -srm
  sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo
  echo "load at start: $(uptime | sed 's/.*load average[s]*: //')"
  $PY -c "import sys, numpy; print('python', sys.version.split()[0], '| numpy', numpy.__version__)"
} > "$OUT/machine.txt"

rust/target/release/plskit-bench --out "$OUT/rs.csv" --when "$WHEN"
# bench.py exits non-zero on a numeric mismatch, after writing its CSV: the run is
# reported and saved all the same, and this script exits with that status. A run
# that wrote no CSV stops here.
STATUS=0
$PY bench.py --out "$OUT/py.csv" || STATUS=$?
[[ -f "$OUT/py.csv" ]] || exit "$STATUS"
echo "load at end: $(uptime | sed 's/.*load average[s]*: //')" >> "$OUT/machine.txt"
$PY report.py --host "$HOST" --dir "$OUT" | tee "$OUT/report.md"
if [[ ${1:-} == --save ]]; then
  mkdir -p "results/$HOST"
  cp "$OUT"/* "results/$HOST/"
fi
exit "$STATUS"
