#!/usr/bin/env python3
"""Verify each manifest entry's recorded sha256 matches the file on disk.

Exit 0 if every file's content hashes to the stored value, exit 1 otherwise.
Catches "edited a fixture but forgot to regenerate manifest.json".
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "testdata"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        print(
            f"manifest schema_version != 2 ({manifest.get('schema_version')!r})",
            file=sys.stderr,
        )
        return 1
    bad = []
    for case in manifest["cases"]:
        for kind in ("inputs", "outputs"):
            actual = sha256(ROOT / case[kind])
            expected = case["hashes"][f"{kind}_sha256"]
            if actual != expected:
                bad.append((case["name"], kind, expected, actual))
    if bad:
        for name, kind, e, a in bad:
            print(
                f"HASH MISMATCH {name} {kind}: expected {e[:16]}…, got {a[:16]}…",
                file=sys.stderr,
            )
        return 1
    print(f"OK: {len(manifest['cases'])} cases, all hashes match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
