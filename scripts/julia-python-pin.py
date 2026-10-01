#!/usr/bin/env python3
"""Decide which plskit Python package the Julia tests run against.

Every PLSKit.jl version pins one exact plskit-py release. The pin lives in
three places that must agree:

- `plskit-jl/Project.toml`        version = "X.Y.Z"
- `plskit-jl/CondaPkg.toml`       plskit = "==X.Y.Z"
- `plskit-jl/src/python.jl`       const PLSKIT_PY_VERSION = "X.Y.Z"

The working-tree plskit-py version (`plskit-py/Cargo.toml`) is compared
with the pin:

- equal:  mode=source (build plskit-py from this checkout)
- newer:  mode=pypi   (Python moved on; Julia still pins an older release,
                       so install that release from PyPI)
- older:  error       (a Julia release may not lead the Python release it wraps)

With --release the mode is always pypi: a Julia release is tested only
against the wheel users will get.

Prints `mode=...` and `pin=...` lines (append them to $GITHUB_OUTPUT).
Plain python3 (3.8+), no third-party imports.
"""

import argparse
import re
import sys
from pathlib import Path


def find(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        sys.exit(f"julia-python-pin: no match for {pattern!r} in {path}")
    return match.group(1)


def as_tuple(version: str) -> tuple:
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        sys.exit(f"julia-python-pin: expected a plain X.Y.Z version, got {version!r}")
    return tuple(int(p) for p in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release", action="store_true", help="always test against PyPI")
    args = parser.parse_args()
    jl = args.root / "plskit-jl"

    project = find(jl / "Project.toml", r'^version\s*=\s*"([^"]+)"')
    conda = find(jl / "CondaPkg.toml", r'^plskit\s*=\s*"==([^"]+)"')
    const = find(jl / "src" / "python.jl", r'PLSKIT_PY_VERSION\s*=\s*"([^"]+)"')
    if not project == conda == const:
        print(
            "julia-python-pin: the Julia pin disagrees with itself: "
            f"Project.toml {project}, CondaPkg.toml {conda}, PLSKIT_PY_VERSION {const}",
            file=sys.stderr,
        )
        return 1
    pin = project

    source = find(args.root / "plskit-py" / "Cargo.toml", r'^version\s*=\s*"([^"]+)"')
    if args.release:
        mode = "pypi"
    elif as_tuple(source) == as_tuple(pin):
        mode = "source"
    elif as_tuple(source) > as_tuple(pin):
        mode = "pypi"
    else:
        print(
            f"julia-python-pin: PLSKit.jl pins plskit {pin} but plskit-py is {source}; "
            "Julia may lag Python, never lead it",
            file=sys.stderr,
        )
        return 1

    print(f"mode={mode}")
    print(f"pin={pin}")
    print(f"julia-python-pin: pin {pin}, plskit-py {source} -> {mode}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
