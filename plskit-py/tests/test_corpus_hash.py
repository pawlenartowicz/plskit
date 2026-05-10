"""Verify the testdata manifest's recorded hashes match files on disk.

This test wraps `plskit/scripts/check_corpus_hash.py` so the staleness
check runs alongside the regular pytest suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_check_corpus_hash_passes_on_committed_state():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_corpus_hash.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
