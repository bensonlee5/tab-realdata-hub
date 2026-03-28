from __future__ import annotations

from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+tab_foundry\b", re.MULTILINE)


def test_pyproject_has_no_tab_foundry_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"].get("dependencies", [])

    assert all("tab-foundry" not in str(dependency) for dependency in dependencies)


def test_source_tree_has_no_tab_foundry_imports() -> None:
    source_root = REPO_ROOT / "src" / "tab_realdata_hub"
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        contents = path.read_text(encoding="utf-8")
        if IMPORT_PATTERN.search(contents):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []
