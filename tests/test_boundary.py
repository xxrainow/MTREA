"""Enforce the one-way dependency boundary between research/ and lab/.

research/ must never import from lab/. The point is that research/ can be
lifted out into a standalone paper repo, so it may carry no dependency on the
hardware-interface code in lab/. The reverse direction (lab/ importing
research/) is expected and is deliberately not checked here.

A violation is a failing test, not a style opinion. See AGENTS.md.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"


def _imports_lab(module: str | None) -> bool:
    """True if `module` is the top-level `lab` package or a submodule of it."""
    return module == "lab" or (module is not None and module.startswith("lab."))


def _find_lab_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, imported name) for every import reaching into lab/."""
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _imports_lab(alias.name):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import; research/ and lab/ are sibling
            # top-level packages, so only absolute imports can reach lab/.
            if node.level == 0 and _imports_lab(node.module):
                offenders.append((node.lineno, node.module or "lab"))
    return offenders


def test_research_does_not_import_lab() -> None:
    violations: list[str] = []
    for path in sorted(RESEARCH_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, name in _find_lab_imports(tree):
            rel = path.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{rel}:{lineno} imports '{name}'")

    assert not violations, (
        "research/ must not import from lab/ (see AGENTS.md); found:\n  "
        + "\n  ".join(violations)
    )
