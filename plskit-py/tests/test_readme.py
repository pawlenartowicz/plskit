"""Run the Python examples in the READMEs as written.

From the first ```python block that imports plskit, every later ```python
block runs in one shared namespace, so a block may use names an earlier one
defined. Blocks before it (install lines) are skipped. The same rule runs the
R and Julia READMEs (plskit-r/tests/testthat/test-readme.R,
plskit-jl/test/readme.jl): change together.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
READMES = [REPO / "README.md", REPO / "plskit-py" / "README.md"]


def example_blocks(path: Path) -> list[str]:
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(), re.M | re.S)
    start = next(i for i, b in enumerate(blocks) if "import plskit" in b)
    return blocks[start:]


@pytest.mark.parametrize("path", READMES, ids=lambda p: str(p.relative_to(REPO)))
def test_readme_examples_run(path):
    namespace: dict = {}
    for block in example_blocks(path):
        exec(compile(block, str(path), "exec"), namespace)
