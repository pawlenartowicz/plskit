#!/usr/bin/env python3
"""Check that the public API surface and result-object fields match the docs.

Two checks, run independently per language (never cross-language):

1. Function names: doc side vs the language's actual exported functions.
2. Result-type field names: doc side vs the language's actual struct /
   dataclass fields, matched by exact type name.

Exit 0 when both languages are clean, 1 with a per-language report otherwise.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

# Single constant so a docs-directory rename stays a one-line edit.
DOCS_DIRNAME = "_docs"

# --- doc-side patterns -------------------------------------------------
# api.md: "**function:** `name`" — one per documented function (Python only;
# rust/api.md has none of these by design, see parse_rust_doc_sections).
RE_DOC_FUNCTION = re.compile(r"^\*\*function:\*\*\s*`(\w+)`", re.MULTILINE)
# results.md: "## `TypeName` — ..." opens a section for that type.
RE_DOC_TYPE_HEADER = re.compile(r"^##\s+`(\w+)`")
# results.md: "| `field` | ... |" is a documented field row inside a section.
RE_DOC_FIELD_ROW = re.compile(r"^\|\s*`(\w+)`\s*\|")
# rust/api.md section headings that bound the "Surface at a glance" list and
# the "Planned — not yet implemented" list (both plain-text headings, not doc
# tables — see parse_rust_doc_sections).
RE_SURFACE_HEADING = re.compile(r"^## Surface at a glance\s*$", re.MULTILINE)
RE_PLANNED_HEADING = re.compile(r"^### Planned.*$", re.MULTILINE)
RE_NEXT_H2 = re.compile(r"^## ", re.MULTILINE)
# A whole backtick span that is exactly snake_case (letters/digits/underscore,
# lowercase) — the shape a function name takes; excludes camelCase type names
# (`Pls1Model`), method tags with punctuation (`ci=true`), and tuples.
RE_SNAKE_BACKTICK = re.compile(r"`([^`]+)`")
RE_SNAKE_CASE = re.compile(r"[a-z][a-z0-9_]*")

# --- Rust code-side patterns --------------------------------------------
# lib.rs: "pub use module::{a, b, c};" or "pub use module::single;"
RE_PUB_USE = re.compile(r"pub use (\w+)::(\{[^}]*\}|\w+);")
# A `pub field: Type,` line inside a struct body; group 2 is the type text,
# used (not the doc comments around it) to find other re-exported structs
# referenced as field types.
RE_PUB_FIELD = re.compile(r"^\s*pub (\w+):\s*(.+)$", re.MULTILINE)


def parse_doc_functions(path: Path) -> set[str]:
    return set(RE_DOC_FUNCTION.findall(path.read_text(encoding="utf-8")))


def parse_doc_type_fields(path: Path) -> dict[str, set[str]]:
    """Map type name -> set of documented field names, per `## \\`Type\\`` section."""
    sections: dict[str, set[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = RE_DOC_TYPE_HEADER.match(line)
        if header:
            current = header.group(1)
            sections.setdefault(current, set())
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is not None:
            row = RE_DOC_FIELD_ROW.match(line)
            if row:
                sections[current].add(row.group(1))
    return sections


def parse_rust_doc_sections(api_md: Path) -> tuple[set[str], set[str]]:
    """rust/api.md has no `**function:**` lines (it defers to the Python doc).

    Instead, "documented" is read off the plain-prose "Surface at a glance"
    list, and "planned" off the "Planned — not yet implemented" list — both
    give function names as snake_case backtick spans. Non-function backtick
    spans in the same prose (`Pls1Model`, `(k, keep)`, `ci=true`, `keep`,
    `raw_perm`) are filtered out downstream by only ever using this set for a
    one-directional "is this exported function mentioned here" check, never
    the reverse, so a stray snake_case tag someone was discussing (`keep`,
    `score`) never shows up as a false "exported but not documented".
    """
    text = api_md.read_text(encoding="utf-8")

    def snake_backticks(chunk: str) -> set[str]:
        return {
            tok for tok in RE_SNAKE_BACKTICK.findall(chunk)
            if RE_SNAKE_CASE.fullmatch(tok)
        }

    m_surface = RE_SURFACE_HEADING.search(text)
    m_planned = RE_PLANNED_HEADING.search(text)
    if not m_surface or not m_planned:
        return set(), set()
    surface = snake_backticks(text[m_surface.end():m_planned.start()])

    m_next = RE_NEXT_H2.search(text, m_planned.end())
    planned_end = m_next.start() if m_next else len(text)
    planned = snake_backticks(text[m_planned.end():planned_end])
    return surface, planned


def rust_reexported_items(rs_src: Path) -> dict[str, str]:
    """Map re-exported item name -> the module it's re-exported from."""
    text = (rs_src / "lib.rs").read_text(encoding="utf-8")
    items: dict[str, str] = {}
    for module, items_blob in RE_PUB_USE.findall(text):
        for item in items_blob.strip("{}").split(","):
            item = item.strip()
            if item:
                items[item] = module
    return items


def _struct_body(text: str, name: str) -> str | None:
    m = re.search(rf"^pub struct {re.escape(name)}\b", text, re.MULTILINE)
    if not m:
        return None
    # Take whichever of `{` / `;` comes first: a unit or tuple struct
    # (`pub struct Foo;`) has no brace body, and searching for `{` alone
    # would silently return the *next* item's. Stop instead — a drift gate
    # that reports another type's fields is worse than one that fails.
    m_open = re.search(r"[{;]", text[m.end():])
    if not m_open or m_open.group() == ";":
        raise ValueError(f"{name} is not a brace-form struct; this parser cannot read it")
    brace_start = m.end() + m_open.start()
    depth, end = 0, brace_start
    for end in range(brace_start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                break
    return text[brace_start:end]


def _fn_return_type(text: str, name: str) -> str | None:
    """Extract the return type identifier of `pub fn name(...)  -> Type {`.

    Params may span multiple lines, so match parens by depth rather than by
    regex. Unwraps `PlsKitResult<T>` to `T`; otherwise takes the leading
    identifier (e.g. `Col<f64>` -> `Col`).
    """
    m = re.search(rf"^pub fn {re.escape(name)}\b", text, re.MULTILINE)
    if not m:
        return None
    paren_start = text.find("(", m.end())
    if paren_start == -1:
        return None
    depth, i = 0, paren_start
    for i in range(paren_start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
    # Anchored at the closing paren: an unanchored search over the following
    # 400 chars can match a closure's `-> T {` inside the body of a fn that
    # declares no return type at all.
    m2 = re.match(r"\s*->\s*([^{;]+?)\s*\{", text[i + 1:i + 400], re.DOTALL)
    if not m2:
        return None
    return_text = m2.group(1).strip()
    wrapped = re.fullmatch(r"PlsKitResult<(\w+)>", return_text)
    if wrapped:
        return wrapped.group(1)
    bare = re.match(r"(\w+)", return_text)
    return bare.group(1) if bare else None


def rust_functions_and_result_types(rs_src: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Exported functions, and the result-type surface reachable from them.

    Code side = the return types of re-exported `pub fn`s, plus any
    re-exported struct that appears (transitively) as a field type of one of
    those — this pulls in `CIScalar` / `ConfirmatoryCI` via
    `ConfirmatoryTestOutput`'s `ci` field, and excludes every `*Opts` /
    `*Input` / `VarimaxArgs` input-side struct, since nothing reachable from
    a return type points at them.
    """
    items = rust_reexported_items(rs_src)
    module_text: dict[str, str] = {}
    for module in set(items.values()):
        f = rs_src / f"{module}.rs"
        if f.exists():
            module_text[module] = f.read_text(encoding="utf-8")

    functions: set[str] = set()
    struct_text: dict[str, str] = {}  # struct name -> its module's full text
    for name, module in items.items():
        if name.endswith("Error") or module not in module_text:
            continue
        text = module_text[module]
        if re.search(rf"^pub fn {re.escape(name)}\b", text, re.MULTILINE):
            functions.add(name)
        elif re.search(rf"^pub struct {re.escape(name)}\b", text, re.MULTILINE):
            struct_text[name] = text
    functions.discard("version")  # defined directly in lib.rs, not re-exported

    reachable: set[str] = set()
    worklist = [
        rt for name in functions
        if (rt := _fn_return_type(module_text[items[name]], name)) in struct_text
    ]
    while worklist:
        t = worklist.pop()
        if t in reachable:
            continue
        reachable.add(t)
        body = _struct_body(struct_text[t], t) or ""
        type_texts = "\n".join(m[1] for m in RE_PUB_FIELD.findall(body))
        for other in struct_text:
            if other not in reachable and re.search(rf"\b{re.escape(other)}\b", type_texts):
                worklist.append(other)

    result_types = {
        t: {m[0] for m in RE_PUB_FIELD.findall(_struct_body(struct_text[t], t) or "")}
        for t in reachable
    }
    return functions, result_types


def python_all_names(init_path: Path) -> list[str]:
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
    return []


def python_def_names(api_path: Path) -> set[str]:
    tree = ast.parse(api_path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _is_dataclass(node: ast.ClassDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "dataclass":
            return True
    return False


def python_dataclass_fields(results_path: Path) -> dict[str, set[str]]:
    tree = ast.parse(results_path.read_text(encoding="utf-8"))
    types: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_dataclass(node):
            fields = {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
            types[node.name] = fields
    return types


def _report_function_drift(lang: str, documented: set[str], exported: set[str]) -> list[str]:
    lines = []
    doc_only = sorted(documented - exported)
    code_only = sorted(exported - documented)
    if doc_only:
        lines.append(f"  documented but not exported: {', '.join(doc_only)}")
    if code_only:
        lines.append(f"  exported but not documented: {', '.join(code_only)}")
    return lines


def _report_type_drift(
    doc_types: dict[str, set[str]], code_types: dict[str, set[str]]
) -> list[str]:
    lines = []
    for name in sorted(set(code_types) - set(doc_types)):
        lines.append(f"  undocumented type: {name}")
    for name in sorted(set(doc_types) - set(code_types)):
        lines.append(f"  stale doc section (no matching type in code): {name}")
    for name in sorted(set(code_types) & set(doc_types)):
        doc_fields, code_fields = doc_types[name], code_types[name]
        missing_in_doc = sorted(code_fields - doc_fields)
        missing_in_code = sorted(doc_fields - code_fields)
        if missing_in_doc:
            lines.append(f"  {name}: field(s) in code but not documented: {', '.join(missing_in_doc)}")
        if missing_in_code:
            lines.append(f"  {name}: field(s) documented but not in code: {', '.join(missing_in_code)}")
    return lines


def check_rust(root: Path, docs_dir: Path) -> list[str]:
    rs_src = root / "plskit-rs" / "src"
    functions, result_types = rust_functions_and_result_types(rs_src)
    documented, planned = parse_rust_doc_sections(docs_dir / "rust" / "api.md")

    issues = []
    undocumented = sorted(functions - documented)
    if undocumented:
        issues.append(f"  exported but not documented: {', '.join(undocumented)}")
    stale_planned = sorted(planned & functions)
    if stale_planned:
        issues.append(f"  planned (not-yet-implemented) but now compiles: {', '.join(stale_planned)}")

    issues += _report_type_drift(
        parse_doc_type_fields(docs_dir / "rust" / "results.md"),
        result_types,
    )
    return issues


def check_python(root: Path, docs_dir: Path) -> list[str]:
    py_src = root / "plskit-py" / "python" / "plskit"
    def_names = python_def_names(py_src / "_api.py")
    exported = set(python_all_names(py_src / "__init__.py")) & def_names
    issues = _report_function_drift(
        "python",
        parse_doc_functions(docs_dir / "python" / "api.md"),
        exported,
    )
    issues += _report_type_drift(
        parse_doc_type_fields(docs_dir / "python" / "results.md"),
        python_dataclass_fields(py_src / "_results.py"),
    )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["rust", "python", "all"], default="all")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="plskit monorepo root (default: inferred from this script's path)",
    )
    args = parser.parse_args()

    docs_dir = args.root / DOCS_DIRNAME
    if not docs_dir.is_dir():
        print(f"{docs_dir} not found, skipping (docs dir not yet tracked)")
        return 0

    checks = {"rust": check_rust, "python": check_python}
    langs = checks if args.lang == "all" else {args.lang: checks[args.lang]}

    any_issues = False
    for lang, check_fn in langs.items():
        issues = check_fn(args.root, docs_dir)
        if issues:
            any_issues = True
            print(f"== {lang} drift ==")
            for line in issues:
                print(line)
        else:
            print(f"== {lang}: clean ==")

    return 1 if any_issues else 0


if __name__ == "__main__":
    sys.exit(main())
