from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tab_realdata_hub.openml as openml_module


class FakeDataset:
    def __init__(self, *, name: str, qualities: dict[str, float], frame: pd.DataFrame, target: pd.Series) -> None:
        self.name = name
        self.qualities = qualities
        self._frame = frame
        self._target = target

    def get_data(self, *, target: str, dataset_format: str) -> tuple[pd.DataFrame, pd.Series, list[bool], list[str]]:
        assert target == "target"
        assert dataset_format == "dataframe"
        return self._frame, self._target, [False] * self._frame.shape[1], list(self._frame.columns)


class FakeTask:
    def __init__(self, dataset: FakeDataset) -> None:
        self.task_type_id = openml_module.TaskType.SUPERVISED_CLASSIFICATION
        self.target_name = "target"
        self._dataset = dataset

    def get_dataset(self, *, download_data: bool) -> FakeDataset:
        assert download_data is False
        return self._dataset


def _prepared_task(
    *,
    task_id: int,
    dataset_name: str,
    n_rows: int,
    n_features: int,
    n_classes: int,
    raw_feature_count: int | None = None,
    missing_pct: float = 0.0,
    minority_class_pct: float = 25.0,
) -> openml_module.PreparedOpenMLTask:
    x = np.arange(n_rows * n_features, dtype=np.float32).reshape(n_rows, n_features)
    y = (np.arange(n_rows, dtype=np.int64) % n_classes).astype(np.int64, copy=False)
    return openml_module.PreparedOpenMLTask(
        task_id=task_id,
        dataset_name=dataset_name,
        x=x,
        y=y,
        observed_task={
            "task_id": int(task_id),
            "dataset_name": dataset_name,
            "n_rows": int(n_rows),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
        },
        qualities={
            "NumberOfFeatures": float(n_features if raw_feature_count is None else raw_feature_count),
            "NumberOfClasses": float(n_classes),
            "PercentageOfInstancesWithMissingValues": float(missing_pct),
            "MinorityClassPercentage": float(minority_class_pct),
        },
    )


def test_prepare_task_preserves_notebook_style_preprocessing() -> None:
    dataset = FakeDataset(
        name="mixed_columns",
        qualities={
            "NumberOfFeatures": 3.0,
            "PercentageOfInstancesWithMissingValues": 0.0,
            "NumberOfClasses": 2.0,
            "MinorityClassPercentage": 50.0,
        },
        frame=pd.DataFrame(
            {
                "num_a": [1.0, 2.0, 3.0, 4.0],
                "cat_b": ["x", "y", "x", "z"],
                "num_c": [0, 1, 0, 1],
            }
        ),
        target=pd.Series(["pos", "neg", "pos", "neg"]),
    )

    prepared = openml_module.prepare_task(
        123,
        new_instances=4,
        task_type="supervised_classification",
        get_task_fn=lambda *_args, **_kwargs: FakeTask(dataset),
    )

    assert prepared.x.dtype == np.float32
    assert prepared.x.shape == (4, 3)
    assert prepared.y.dtype == np.int64
    assert prepared.observed_task["n_classes"] == 2
    assert sorted(np.unique(prepared.y).tolist()) == [0, 1]


def test_build_bundle_from_pinned_task_ids_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dataset = FakeDataset(
        name="stable_pool",
        qualities={
            "NumberOfFeatures": 4.0,
            "PercentageOfInstancesWithMissingValues": 0.0,
            "NumberOfClasses": 2.0,
            "MinorityClassPercentage": 25.0,
        },
        frame=pd.DataFrame({"a": [0, 1], "b": [1, 0], "c": [0, 0], "d": [1, 1]}),
        target=pd.Series([0, 1]),
    )
    monkeypatch.setattr(openml_module.openml.tasks, "get_task", lambda *_args, **_kwargs: FakeTask(fake_dataset))
    monkeypatch.setattr(
        openml_module,
        "prepare_task",
        lambda task_id, *, new_instances, task_type: _prepared_task(
            task_id=int(task_id),
            dataset_name=f"task_{task_id}",
            n_rows=int(new_instances),
            n_features=4,
            n_classes=2,
        ),
    )

    bundle = openml_module.build_bundle(
        openml_module.OpenMLBundleConfig(
            bundle_name="stable_bundle",
            version=1,
            task_ids=(101, 102),
            task_source="binary_expanded_v1",
            discover_from_openml=False,
            max_features=10,
            max_classes=2,
            max_missing_pct=0.0,
            min_minority_class_pct=2.5,
        )
    )

    assert bundle["task_ids"] == [101, 102]
    assert [task["dataset_name"] for task in bundle["tasks"]] == ["task_101", "task_102"]
    assert bundle["selection"]["max_features"] == 10


def test_build_bundle_discovery_uses_bounded_filters_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_list_tasks(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "tid": 101,
                    "did": 10,
                    "name": "dup_dataset",
                    "NumberOfInstances": 300,
                    "NumberOfFeatures": 5,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 75,
                    "estimation_procedure": "holdout",
                },
                {
                    "tid": 102,
                    "did": 10,
                    "name": "dup_dataset",
                    "NumberOfInstances": 300,
                    "NumberOfFeatures": 5,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 75,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
                {
                    "tid": 201,
                    "did": 20,
                    "name": "too_wide",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 60,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 80,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
            ]
        )

    monkeypatch.setattr(openml_module.openml.tasks, "list_tasks", _fake_list_tasks)
    monkeypatch.setattr(
        openml_module,
        "prepare_task",
        lambda task_id, *, new_instances, task_type: _prepared_task(
            task_id=int(task_id),
            dataset_name="dup_dataset",
            n_rows=int(new_instances),
            n_features=5,
            n_classes=2,
        ),
    )

    result = openml_module.build_bundle_result(
        openml_module.OpenMLBundleConfig(
            bundle_name="binary_large_no_missing",
            version=1,
            discover_from_openml=True,
            min_instances=200,
            min_task_count=1,
            max_features=50,
            max_classes=2,
            max_missing_pct=0.0,
            min_minority_class_pct=2.5,
        )
    )

    assert result.bundle["task_ids"] == [102]
    assert captured["number_instances"] == "200..1000000000"
    assert captured["number_features"] == "0..50"
    assert captured["number_classes"] == 2
    assert captured["number_missing_values"] == 0
    report = openml_module.render_candidate_report(result.report_entries)
    assert "preferred task_id=102" in report
    assert "number_of_features=60 exceeds max_features=50" in report


def test_build_bundle_discovery_falls_back_when_filtered_listing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_list_tasks(**kwargs: object) -> pd.DataFrame:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("OpenML task-list filter request failed")
        return pd.DataFrame(
            [
                {
                    "tid": 10,
                    "did": 1,
                    "name": "fallback_ok",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 4,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 80,
                    "estimation_procedure": "10-fold Crossvalidation",
                }
            ]
        )

    monkeypatch.setattr(openml_module.openml.tasks, "list_tasks", _fake_list_tasks)
    monkeypatch.setattr(
        openml_module,
        "prepare_task",
        lambda task_id, *, new_instances, task_type: _prepared_task(
            task_id=int(task_id),
            dataset_name="fallback_ok",
            n_rows=int(new_instances),
            n_features=4,
            n_classes=2,
        ),
    )

    bundle = openml_module.build_bundle(
        openml_module.OpenMLBundleConfig(
            bundle_name="binary_large_no_missing",
            version=1,
            discover_from_openml=True,
            min_instances=200,
            min_task_count=1,
            max_features=50,
            max_classes=2,
            max_missing_pct=0.0,
            min_minority_class_pct=2.5,
        )
    )

    assert bundle["task_ids"] == [10]
    assert len(calls) == 2
    assert calls[0]["number_instances"] == "200..1000000000"
    assert calls[0]["number_features"] == "0..50"
    assert "number_instances" not in calls[1]


def test_write_bundle_round_trips_stably(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    payload = {
        "name": "manual_bundle",
        "version": 1,
        "selection": {
            "new_instances": 200,
            "task_type": "supervised_classification",
            "max_features": 10,
            "max_missing_pct": 0.0,
            "max_classes": 2,
            "min_minority_class_pct": 2.5,
        },
        "task_ids": [1],
        "tasks": [{"task_id": 1, "dataset_name": "a", "n_rows": 200, "n_features": 3, "n_classes": 2}],
    }

    openml_module.write_bundle(
        path,
        openml_module.OpenMLBundleConfig(bundle_name="manual_bundle", version=1),
        bundle=payload,
    )

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == payload
