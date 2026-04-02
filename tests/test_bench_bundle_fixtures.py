from __future__ import annotations

from pathlib import Path

import tab_realdata_hub.openml as openml_module


def _bundle_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "tab_realdata_hub" / "bench" / name


def test_checked_in_classification_medium_bundle_matches_contract() -> None:
    bundle = openml_module.load_bundle(_bundle_path("openml_classification_medium_v1.json"), allow_missing_values=True)

    assert bundle["name"] == "openml_classification_medium"
    assert bundle["selection"] == {
        "new_instances": 200,
        "task_type": "supervised_classification",
        "max_features": 10,
        "min_classes": 2,
        "max_classes": 10,
        "max_missing_pct": 20.0,
        "min_minority_class_pct": 1.0,
    }
    assert len(bundle["task_ids"]) >= 10


def test_checked_in_classification_large_bundle_matches_contract() -> None:
    bundle = openml_module.load_bundle(_bundle_path("openml_classification_large_v1.json"), allow_missing_values=True)

    assert bundle["name"] == "openml_classification_large"
    assert bundle["selection"] == {
        "new_instances": 200,
        "task_type": "supervised_classification",
        "max_features": 20,
        "min_classes": 2,
        "max_classes": 10,
        "max_missing_pct": 20.0,
        "min_minority_class_pct": 1.0,
    }
    assert len(bundle["task_ids"]) >= 12
