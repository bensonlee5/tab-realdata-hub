from __future__ import annotations

from pathlib import Path

import pytest

import tab_realdata_hub.openml as openml_module


def _bundle_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "tab_realdata_hub" / "bench" / name


@pytest.mark.parametrize(
    ("filename", "expected_name", "expected_max_features", "expected_count"),
    [
        ("openml_classification_medium_v1.json", "openml_classification_medium", 10, 231),
        ("openml_classification_medium_top10_v1.json", "openml_classification_medium_top10", 10, 10),
        ("openml_classification_medium_top25_v1.json", "openml_classification_medium_top25", 10, 25),
        ("openml_classification_medium_top50_v1.json", "openml_classification_medium_top50", 10, 50),
        ("openml_classification_large_v1.json", "openml_classification_large", 20, 577),
        ("openml_classification_large_top10_v1.json", "openml_classification_large_top10", 20, 10),
        ("openml_classification_large_top25_v1.json", "openml_classification_large_top25", 20, 25),
        ("openml_classification_large_top50_v1.json", "openml_classification_large_top50", 20, 50),
    ],
)
def test_checked_in_classification_bundles_match_contract(
    filename: str,
    expected_name: str,
    expected_max_features: int,
    expected_count: int,
) -> None:
    bundle = openml_module.load_bundle(_bundle_path(filename), allow_missing_values=True)

    assert bundle["name"] == expected_name
    assert bundle["selection"] == {
        "task_type": "supervised_classification",
        "max_features": expected_max_features,
        "min_classes": 2,
        "max_classes": 10,
        "max_missing_pct": 20.0,
        "min_minority_class_pct": 1.0,
    }
    assert len(bundle["task_ids"]) == expected_count
    assert len(bundle["tasks"]) == expected_count
    assert all(
        not openml_module.is_synthetic_dataset_name(str(task["dataset_name"]))
        for task in bundle["tasks"]
    )


def test_subset_bundles_are_nested_prefixes_of_the_ranked_real_only_pools() -> None:
    medium_top10 = openml_module.load_bundle(_bundle_path("openml_classification_medium_top10_v1.json"), allow_missing_values=True)
    medium_top25 = openml_module.load_bundle(_bundle_path("openml_classification_medium_top25_v1.json"), allow_missing_values=True)
    medium_top50 = openml_module.load_bundle(_bundle_path("openml_classification_medium_top50_v1.json"), allow_missing_values=True)
    large_top10 = openml_module.load_bundle(_bundle_path("openml_classification_large_top10_v1.json"), allow_missing_values=True)
    large_top25 = openml_module.load_bundle(_bundle_path("openml_classification_large_top25_v1.json"), allow_missing_values=True)
    large_top50 = openml_module.load_bundle(_bundle_path("openml_classification_large_top50_v1.json"), allow_missing_values=True)

    assert medium_top25["task_ids"][:10] == medium_top10["task_ids"]
    assert medium_top50["task_ids"][:25] == medium_top25["task_ids"]
    assert large_top25["task_ids"][:10] == large_top10["task_ids"]
    assert large_top50["task_ids"][:25] == large_top25["task_ids"]
