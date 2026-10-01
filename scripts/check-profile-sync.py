#!/usr/bin/env python3
"""Keep the R crate's build in step with the workspace (spec sections 2.3, 8).

plskit-r/src/rust/ is outside the cargo workspace, so it repeats the
workspace [profile.release] block and resolves its own Cargo.lock. This
check fails when:

1. the R crate's [profile.release] differs from the workspace's, or its
   manifest lacks an empty [workspace] table;
2. a package in the dependency trees of plskit and plskit-bind (the two
   excluded) resolves to different versions in the R lock file than in
   the workspace Cargo.lock;
3. the R crate's pinned `plskit` / `plskit-bind` requirements
   (`= X.Y.Z` in plskit-r/src/rust/Cargo.toml) do not equal
   plskit-r/DESCRIPTION's `Version:` (Decision 4: both are pinned to
   the R package version until P4).

The committed R lock file starts in P4 (spec 2.3); until then check 2
is skipped unless --r-lock names a lock file, e.g. the dev-mode one at
plskit-r/src/rust/target/dev-crate/Cargo.lock. Plain python3 (3.11+,
for tomllib), no dependencies.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

R_MANIFEST = Path("plskit-r/src/rust/Cargo.toml")
R_LOCK = Path("plskit-r/src/rust/Cargo.lock")
R_DESCRIPTION = Path("plskit-r/DESCRIPTION")
SHARED_ROOTS = ("plskit", "plskit-bind")
RE_PINNED = re.compile(r"^=\s*([0-9][0-9A-Za-z.+-]*)$")
RE_DESCRIPTION_VERSION = re.compile(r"^Version:\s*(\S+)\s*$", re.MULTILINE)


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def check_profile(root: Path) -> list[str]:
    ws = load_toml(root / "Cargo.toml")
    r = load_toml(root / R_MANIFEST)
    issues = []
    if r.get("workspace") != {}:
        issues.append(f"{R_MANIFEST}: needs an empty [workspace] table")
    want = ws.get("profile", {}).get("release")
    got = r.get("profile", {}).get("release")
    if got != want:
        issues.append(f"{R_MANIFEST}: [profile.release] is {got}, the workspace has {want}")
    return issues


def check_version(root: Path) -> list[str]:
    """The R crate's pinned plskit / plskit-bind requirements equal
    DESCRIPTION's Version (Decision 4)."""
    r = load_toml(root / R_MANIFEST)
    deps = r.get("dependencies", {})
    description_text = (root / R_DESCRIPTION).read_text(encoding="utf-8")
    m = RE_DESCRIPTION_VERSION.search(description_text)
    if m is None:
        return [f"{R_DESCRIPTION}: no Version: line found"]
    description_version = m.group(1)
    issues = []
    for pkg in SHARED_ROOTS:
        req = deps.get(pkg)
        if not isinstance(req, str):
            issues.append(f"{R_MANIFEST}: {pkg} dependency is not a plain string requirement: {req!r}")
            continue
        pm = RE_PINNED.match(req.strip())
        if pm is None:
            issues.append(f"{R_MANIFEST}: {pkg} = {req!r} is not pinned with '='")
            continue
        pinned_version = pm.group(1)
        if pinned_version != description_version:
            issues.append(
                f"{R_MANIFEST}: {pkg} is pinned to {pinned_version}, "
                f"{R_DESCRIPTION} Version is {description_version}"
            )
    return issues


def dependency_versions(lock: dict, roots=SHARED_ROOTS) -> dict[str, set[str]]:
    """Name -> versions of every package reachable from `roots` (roots excluded)."""
    by_name: dict[str, list[dict]] = {}
    for pkg in lock.get("package", []):
        by_name.setdefault(pkg["name"], []).append(pkg)
    seen: dict[str, set[str]] = {}
    stack: list[tuple[str, str | None]] = [(name, None) for name in roots]
    while stack:
        name, version = stack.pop()
        for pkg in by_name.get(name, []):
            if version is not None and pkg["version"] != version:
                continue
            versions = seen.setdefault(name, set())
            if pkg["version"] in versions:
                continue
            versions.add(pkg["version"])
            for dep in pkg.get("dependencies", []):
                parts = dep.split(" ")
                stack.append((parts[0], parts[1] if len(parts) > 1 else None))
    for name in roots:
        seen.pop(name, None)
    return seen


def check_lock(root: Path, r_lock: Path) -> list[str]:
    """Every package the R crate compiles for plskit / plskit-bind has the
    workspace's version. Workspace-only packages (dev-dependencies, which
    Cargo.lock does not mark) are not compared."""
    ws = dependency_versions(load_toml(root / "Cargo.lock"))
    r = dependency_versions(load_toml(r_lock))
    return [
        f"{name}: {r_lock} has {sorted(r[name])}, the workspace has {sorted(ws.get(name, set()))}"
        for name in sorted(r)
        if r[name] != ws.get(name, set())
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="plskit monorepo root (default: inferred from this script's path)")
    ap.add_argument("--r-lock", type=Path, default=None,
                    help=f"R lock file to compare (default: {R_LOCK}, skipped when absent)")
    args = ap.parse_args()

    issues = check_profile(args.root)
    issues += check_version(args.root)
    r_lock = args.r_lock if args.r_lock is not None else args.root / R_LOCK
    if r_lock.exists():
        issues += check_lock(args.root, r_lock)
    elif args.r_lock is not None:
        issues.append(f"{r_lock}: not found")
    else:
        print(f"{R_LOCK} is not committed yet (spec 2.3: from P4); lock check skipped")

    for line in issues:
        print(f"  {line}")
    print("== profile sync: " + ("drift ==" if issues else "clean =="))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
