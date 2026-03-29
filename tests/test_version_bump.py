from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "check_version_bump.py"
    spec = importlib.util.spec_from_file_location("check_version_bump", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


version_bump = _load_module()


def test_parse_semver_accepts_three_part_versions() -> None:
    assert version_bump.parse_semver("0.1.0") == (0, 1, 0)


@pytest.mark.parametrize("value", ["0.1", "1", "1.0.0.0", "v1.0.0", "1.a.0"])
def test_parse_semver_rejects_invalid_versions(value: str) -> None:
    with pytest.raises(ValueError):
        version_bump.parse_semver(value)


@pytest.mark.parametrize(
    ("base_version", "head_version", "expected"),
    [
        ((0, 1, 0), (0, 1, 1), "patch"),
        ((0, 1, 9), (0, 2, 0), "minor"),
        ((0, 9, 9), (1, 0, 0), "major"),
    ],
)
def test_classify_version_step_accepts_exactly_one_semver_step(
    base_version: tuple[int, int, int],
    head_version: tuple[int, int, int],
    expected: str,
) -> None:
    assert version_bump.classify_version_step(base_version, head_version) == expected


@pytest.mark.parametrize(
    ("base_version", "head_version"),
    [
        ((0, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 2)),
        ((0, 1, 0), (0, 2, 1)),
        ((0, 1, 0), (1, 0, 1)),
        ((0, 1, 0), (0, 0, 9)),
    ],
)
def test_classify_version_step_rejects_noops_multi_steps_and_downgrades(
    base_version: tuple[int, int, int],
    head_version: tuple[int, int, int],
) -> None:
    assert version_bump.classify_version_step(base_version, head_version) is None


def test_requires_version_bump_only_for_src_package_changes() -> None:
    assert version_bump.requires_version_bump(["src/tab_realdata_hub/openml.py"])
    assert version_bump.requires_version_bump(["src/tab_realdata_hub/bench/openml_classification_medium_v1.json"])
    assert not version_bump.requires_version_bump(["README.md", ".github/workflows/test.yml"])
