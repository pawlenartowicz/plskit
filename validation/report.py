#!/usr/bin/env python3
"""Markdown tables from a run's {py,rs}.csv. Ratios are relative to each group's first plskit routine (>1 = slower)."""
import argparse
import csv
import math
from pathlib import Path

from cases import GROUPS
from common import ROOT, host


def load(d):
    rows = {}
    for name in ("rs.csv", "py.csv"):
        path = Path(d) / name
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["bench"], f"{r['lib']}/{r['routine']}", int(r["n"]), int(r["p"]), int(r["k"]))] = r
    return rows


def val(r, metric):
    try:
        v = float(r[metric])
    except (TypeError, ValueError, KeyError):
        return math.nan
    return v


def ms(v):
    return f"{v:#.3g}" if v < 100 else f"{v:,.0f}"


def table(head, align, body):
    """Markdown table with cells padded to column width, so the raw text lines up too.
    align holds one letter per column: "r" right, "c" centre."""
    w = [max(len(r[i]) for r in [head, *body]) for i in range(len(head))]
    line = lambda cells: "| " + " | ".join(cells) + " |"
    pad = lambda r: [c.center(w[i]) if align[i] == "c" else c.rjust(w[i]) for i, c in enumerate(r)]
    rule = [(":" if a == "c" else "-") + "-" * (w[i] - 2) + ":" for i, a in enumerate(align)]
    return [line(pad(head)), line(rule), *(line(pad(r)) for r in body)]


def columns(rows, bench):
    """(header, [lib/routine, ...]) per column after the baseline: the group's other plskit
    routines, plskit-rs when it was timed, then one column per reference library. A library
    with several routines shows the fastest one timed at each shape, and a second column
    names it."""
    group = GROUPS[bench]
    cols = [(r.routine, [f"{r.lib}/{r.routine}"]) for r in group.plskit[1:]]
    rs = f"plskit-rs/{group.plskit[0].routine}"
    if any(k[:2] == (bench, rs) for k in rows):
        cols.append(("plskit-rs", [rs]))
    for lib in dict.fromkeys(r.lib for r in group.refs):
        cols.append((lib, [f"{lib}/{r.routine}" for r in group.refs if r.lib == lib]))
    return cols


def group_table(rows, bench, metric):
    """One table per n, one row per (p, k): the baseline in ms, the rest as multiples of it."""
    base_r = GROUPS[bench].plskit[0]
    get = lambda c, s: val(rows.get((bench, c, *s), {}), metric)
    cols = columns(rows, bench)
    head, align = ["p", "k", f"{base_r.routine} (ms)"], "rrr"
    for name, keys in cols:
        head += [name] if len(keys) == 1 else [name, f"{name} routine"]
        align += "r" if len(keys) == 1 else "rc"
    shapes = sorted({k[2:] for k in rows if k[0] == bench})
    out = [f"### {bench}, {metric.replace('_ms', '')} time (× {base_r.routine}, below 1 = faster)", ""]
    for n in sorted({s[0] for s in shapes}):
        body = []
        for s in (s for s in shapes if s[0] == n):
            base = get(f"{base_r.lib}/{base_r.routine}", s)
            ratio = lambda v: "—" if math.isnan(v) or math.isnan(base) else f"{v / base:.2f}"
            cells = ["—" if math.isnan(base) else ms(base)]
            for _, keys in cols:
                timed = [(get(c, s), c.split("/")[1]) for c in keys if not math.isnan(get(c, s))]
                if len(keys) == 1:
                    cells.append(ratio(timed[0][0]) if timed else "—")
                else:
                    cells += [ratio(min(timed)[0]), min(timed)[1]] if timed else ["—", ""]
            body.append([f"{s[1]:,}", str(s[2]), *cells])
        out += [f"**n = {n:,}**", "", *table(head, align, body), ""]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=host())
    ap.add_argument("--dir", help="directory holding py.csv and rs.csv (default results/<host>)")
    ap.add_argument("--metric", choices=["wall", "cpu"], nargs="*", default=["wall", "cpu"],
                    help="wall is fine on an idle machine; trust cpu on a loaded one")
    a = ap.parse_args()
    rows = load(a.dir or ROOT / "results" / a.host)
    print(f"## {a.host}\n")
    for m in a.metric:
        for bench in GROUPS:
            print("\n".join(group_table(rows, bench, m + "_ms")) + "\n")
    print("`pls1_fit_prestd` is `pls1_fit(..., pre_standardized=True)` on `plskit.preprocess` output. "
          "`*_endpoint` is the sparse fit with every variable kept. — means not timed: a "
          "pre-standardized fit raises `PlsKitError` when the data supports fewer than k "
          "components. ikpls is timed with `alg2` when n ≥ 100·p and with `alg1` otherwise.")


if __name__ == "__main__":
    main()
