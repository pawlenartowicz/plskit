#!/usr/bin/env python3
"""Render plskit-r/R/stubs.R from the plskit-bind registry dump.

One exported R function per registry entry: formals in registry order with
the registry defaults, a body that forwards every formal to .plskit_call(),
and a short roxygen block that links to the canonical Python API entry
(_docs/python/api.md). Plain python3, no dependencies.

    python3 scripts/render-r-stubs.py                   # dump via cargo, write
    python3 scripts/render-r-stubs.py --dump reg.json   # use a saved dump
    python3 scripts/render-r-stubs.py --check           # exit 1 on drift
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUBS = ROOT / "plskit-r" / "R" / "stubs.R"
API_MD = ROOT / "_docs" / "python" / "api.md"
API_URL = "https://github.com/pawlenartowicz/plskit/blob/main/_docs/python/api.md"

# Rd titles. A registry function missing here fails the render, so a new
# function cannot ship without one.
TITLES = {
    "preprocess": "Standardize X and Y",
    "pls1_fit": "Fit a PLS1 model",
    "pls1_predict": "Predict from a fitted PLS1 model",
    "pls1_find_k_optimal": "Choose k for PLS1 by cross-validation or BIC",
    "pls1_find_k_sequence": "Choose k for PLS1 by sequential testing",
    "spls1_fit": "Fit a sparse PLS1 model",
    "spls1_find_keep_optimal": "Choose the sPLS1 keep-count by cross-validation",
    "spls1_find_k_optimal": "Choose k for sPLS1 by cross-validation or BIC",
    "spls1_find_k_sequence": "Choose k for sPLS1 by sequential testing",
    "pls3_fit": "Fit a PLS3 (PLS-SVD) model",
    "plssvd_fit": "Fit a PLS-SVD model (alias of pls3_fit)",
    "pls3_transform": "Project new data onto a fitted PLS3 model",
    "plssvd_transform": "Project new data onto a fitted PLS-SVD model",
    "spls3_fit": "Fit a sparse PLS3 model",
    "pls1_confirmatory_test": "Confirmatory test of a PLS1 signal",
    "split_nb_gate": "Query the split_nb auto-gate",
    "pls1_perm_null": "Permutation null of PLS1 coefficients",
    "pls3_confirmatory_test": "Confirmatory test of a PLS3 signal",
    "rotate": "Rotate PLS1 weights or a fitted PLS1 model",
    "pls1_rotation_stability": "Rotation-stability diagnostic for PLS1",
}

RE_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
RE_FUNCTION = re.compile(r"^\*\*function:\*\*\s*`(\w+)`")


def load_dump(path):
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        out = subprocess.run(
            ["cargo", "run", "-q", "-p", "plskit-bind", "--bin", "plskit-bind-dump", "--release"],
            cwd=ROOT, check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        raise SystemExit(f"render-r-stubs: plskit-bind-dump failed (exit {e.returncode})")
    return json.loads(out.stdout.decode("utf-8"))


def api_sections():
    """Function name -> the api.md heading its `**function:**` line sits under."""
    sections, heading = {}, None
    for line in API_MD.read_text(encoding="utf-8").splitlines():
        m = RE_HEADING.match(line)
        if m:
            heading = m.group(1)
            continue
        m = RE_FUNCTION.match(line)
        if m and heading is not None:
            sections[m.group(1)] = heading
    return sections


def r_default(param):
    if param["required"]:
        return None
    d = param["default"]
    if d is None:
        return "NULL"
    if isinstance(d, bool):
        return "TRUE" if d else "FALSE"
    if isinstance(d, int):
        return f"{d}L"
    if isinstance(d, str):
        return json.dumps(d)
    raise ValueError(f"unsupported default {d!r} for {param['name']}")


def return_doc(fn, r_class):
    types = fn["result_types"]
    if not types:
        return "A numeric vector."
    classes = [f'`c("{r_class[t]}", "plskit_result")`' for t in types]
    if len(classes) == 1:
        return f"A list of class {classes[0]}."
    return "A list of class " + ", or ".join(classes) + ", depending on the input."


def render_function(fn, r_class, sections):
    name = fn["name"]
    if name not in TITLES:
        raise SystemExit(f"render-r-stubs: no title for {name}; add it to TITLES")
    if name not in sections:
        raise SystemExit(f"render-r-stubs: {name} has no **function:** line in {API_MD}")
    params = fn["params"]
    formals = []
    for p in params:
        d = r_default(p)
        formals.append(p["name"] if d is None else f"{p['name']} = {d}")
    lines = [
        f"#' {TITLES[name]}",
        "#'",
        f"#' R binding of `{name}`. Arguments, defaults and result fields are",
        "#' documented once, in the canonical reference: section",
        f"#' \"{sections[name]}\" of <{API_URL}>.",
        "#'",
        f"#' @param {','.join(p['name'] for p in params)} See the canonical reference.",
        f"#' @return {return_doc(fn, r_class)}",
        "#' @export",
        f"{name} <- function(",
    ]
    lines += [f"  {f}," for f in formals[:-1]] + [f"  {formals[-1]}"]
    lines += [
        ") {",
        f'  .plskit_call("{name}", list(',
    ]
    lines += [f"    {p['name']} = {p['name']}," for p in params[:-1]]
    lines += [f"    {params[-1]['name']} = {params[-1]['name']}"]
    lines += ["  ))", "}"]
    return "\n".join(lines)


def render(dump):
    r_class = {t["name"]: t["r_class"] for t in dump["result_types"]}
    sections = api_sections()
    header = (
        "# Generated by scripts/render-r-stubs.py from the plskit-bind registry.\n"
        "# Do not edit by hand: re-run `python3 scripts/render-r-stubs.py`\n"
        "# after a registry change.\n"
    )
    body = "\n\n".join(render_function(f, r_class, sections) for f in dump["functions"])
    return header + "\n" + body + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", help="registry JSON from plskit-bind-dump (default: run it via cargo)")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed stubs differ")
    args = ap.parse_args()
    text = render(load_dump(args.dump))
    if args.check:
        current = STUBS.read_text(encoding="utf-8") if STUBS.exists() else ""
        if current != text:
            print(f"{STUBS.relative_to(ROOT)} differs from a fresh render; "
                  "run python3 scripts/render-r-stubs.py", file=sys.stderr)
            return 1
        print(f"{STUBS.relative_to(ROOT)}: up to date")
        return 0
    STUBS.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {STUBS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
