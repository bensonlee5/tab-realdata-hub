from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import tab_realdata_hub.manifest as manifest_module
import tab_realdata_hub.openml as openml_module


def _build_split_table(rows: list[tuple[int, np.ndarray, np.ndarray]]) -> pa.Table:
    dataset_indices: list[int] = []
    row_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for dataset_index, x, y in rows:
        for row_index in range(int(x.shape[0])):
            dataset_indices.append(int(dataset_index))
            row_indices.append(int(row_index))
            x_rows.append(x[row_index].astype(np.float32, copy=False).tolist())
            y_rows.append(int(y[row_index]))
    return pa.table(
        {
            "dataset_index": pa.array(dataset_indices, type=pa.int64()),
            "row_index": pa.array(row_indices, type=pa.int64()),
            "x": pa.array(x_rows, type=pa.list_(pa.float32())),
            "y": pa.array(y_rows, type=pa.int64()),
        }
    )


def _write_dataset(shard_dir: Path, *, metadata: dict[str, Any]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    x_train = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    x_test = np.array([[2.0, 3.0]], dtype=np.float32)
    y_test = np.array([1], dtype=np.int64)
    pq.write_table(_build_split_table([(0, x_train, y_train)]), shard_dir / "train.parquet")
    pq.write_table(_build_split_table([(0, x_test, y_test)]), shard_dir / "test.parquet")
    payload = {
        "dataset_index": 0,
        "n_train": 2,
        "n_test": 1,
        "n_features": 2,
        "feature_types": ["floating", "floating"],
        "metadata": metadata,
    }
    with (shard_dir / "metadata.ndjson").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _prepared_task(
    *,
    task_id: int,
    dataset_name: str,
    n_rows: int,
    n_features: int,
    n_classes: int,
) -> openml_module.PreparedOpenMLTask:
    x = np.arange(n_rows * n_features, dtype=np.float32).reshape(n_rows, n_features)
    y = (np.arange(n_rows, dtype=np.int64) % n_classes).astype(np.int64, copy=False)
    return openml_module.PreparedOpenMLTask(
        task_id=task_id,
        dataset_name=dataset_name,
        x=x,
        y=y,
        observed_task={
            "task_id": task_id,
            "dataset_name": dataset_name,
            "n_rows": n_rows,
            "n_features": n_features,
            "n_classes": n_classes,
        },
        qualities={
            "NumberOfFeatures": float(n_features),
            "NumberOfClasses": float(n_classes),
            "PercentageOfInstancesWithMissingValues": 0.0,
            "MinorityClassPercentage": 25.0,
        },
    )


def test_split_prepared_task_falls_back_when_stratification_is_not_possible() -> None:
    prepared = openml_module.PreparedOpenMLTask(
        task_id=1,
        dataset_name="singleton_minority",
        x=np.arange(8, dtype=np.float32).reshape(4, 2),
        y=np.array([0, 0, 0, 1], dtype=np.int64),
        observed_task={"task_id": 1, "dataset_name": "singleton_minority", "n_rows": 4, "n_features": 2},
        qualities={"NumberOfFeatures": 2.0, "PercentageOfInstancesWithMissingValues": 0.0, "NumberOfClasses": 2.0},
    )

    x_train, x_test, y_train, y_test, split_mode = openml_module._split_prepared_task(
        prepared,
        split_seed=0,
        test_size=0.5,
    )

    assert split_mode == "unstratified_fallback"
    assert x_train.shape == (2, 2)
    assert x_test.shape == (2, 2)
    assert y_train.shape == (2,)
    assert y_test.shape == (2,)


def test_manifest_build_and_inspect_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "packed_shards"
    _write_dataset(
        root / "shard_00001_case",
        metadata={
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "deferred", "status": "not_run"},
        },
    )

    manifest_path = tmp_path / "manifest.parquet"
    summary = manifest_module.build_manifest([root], manifest_path)
    inspection = manifest_module.inspect_manifest(manifest_path)

    assert summary.total_records == 1
    assert inspection["total_records"] == 1
    assert inspection["task_counts"] == {"classification": 1}
    assert inspection["persisted_summary"]["total_records"] == 1


def test_materialize_bundle_writes_manifest_backed_shards(tmp_path: Path, monkeypatch) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "name": "many_class_v1",
                "version": 1,
                "selection": {
                    "new_instances": 10,
                    "task_type": "supervised_classification",
                    "max_features": 20,
                    "max_missing_pct": 5.0,
                    "max_classes": 2,
                    "min_minority_class_pct": 2.5,
                },
                "task_ids": [101, 102],
                "tasks": [
                    {"task_id": 101, "dataset_name": "first_dataset", "n_rows": 10, "n_features": 3, "n_classes": 2},
                    {"task_id": 102, "dataset_name": "second_dataset", "n_rows": 10, "n_features": 2, "n_classes": 2},
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prepared_tasks = {
        101: _prepared_task(task_id=101, dataset_name="first_dataset", n_rows=10, n_features=3, n_classes=2),
        102: _prepared_task(task_id=102, dataset_name="second_dataset", n_rows=10, n_features=2, n_classes=2),
    }
    monkeypatch.setattr(
        openml_module,
        "prepare_task",
        lambda task_id, *, new_instances, task_type: prepared_tasks[int(task_id)],
    )

    result = openml_module.materialize_bundle(bundle_path, tmp_path / "out")

    assert result.manifest_path.exists()
    assert result.data_root.exists()
    assert len(result.task_summaries) == 2

    manifest_rows = pq.read_table(result.manifest_path).to_pylist()
    assert len(manifest_rows) == 2

    metadata_path = result.data_root / "shard_00001_first_dataset" / "metadata.ndjson"
    payload = json.loads(metadata_path.read_text(encoding="utf-8").strip())
    assert payload["feature_types"] == ["floating", "floating", "floating"]
    assert payload["metadata"]["source_platform"] == "openml"
    assert payload["metadata"]["benchmark_bundle"]["source_path"] == str(bundle_path.resolve())
    assert payload["metadata"]["openml"]["task_id"] == 101
