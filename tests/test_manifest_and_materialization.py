from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import tab_realdata_hub.dagzoo_handoff as dagzoo_handoff_module
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


def _write_dataset(
    shard_dir: Path,
    *,
    metadata: dict[str, Any],
    legacy_catalog: bool = False,
) -> None:
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
    if legacy_catalog:
        with (shard_dir / "metadata.ndjson").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return
    manifest_module.write_dataset_catalog(
        shard_dir / manifest_module.DATASET_CATALOG_FILENAME,
        [payload],
    )


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
        observed_task={
            "task_id": 1,
            "dataset_name": "singleton_minority",
            "n_rows": 4,
            "n_features": 2,
        },
        qualities={
            "NumberOfFeatures": 2.0,
            "PercentageOfInstancesWithMissingValues": 0.0,
            "NumberOfClasses": 2.0,
        },
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
    assert inspection["manifest_contract"]["version"] == manifest_module.MANIFEST_CONTRACT_VERSION
    assert inspection["manifest_contract"]["stable_index_fields"] == list(
        manifest_module.MANIFEST_STABLE_INDEX_FIELDS
    )
    assert isinstance(inspection["manifest_sha256"], str)
    assert len(inspection["manifest_sha256"]) == 64
    assert inspection["task_counts"] == {"classification": 1}
    assert inspection["persisted_summary"]["total_records"] == 1


def test_build_manifest_persists_current_dagzoo_handoff_provenance(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    generate_run_id = "a" * 32
    dataset_id = "b" * 32
    _write_dataset(
        generated_root / "shard_00001_case",
        metadata={
            "dataset_id": dataset_id,
            "split_groups": {"request_run": generate_run_id},
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "generated", "status": "accepted", "accepted": True},
        },
    )
    handoff_manifest_path = tmp_path / "dagzoo_handoff.json"
    handoff_manifest_path.write_text(
        json.dumps(
            {
                "schema_name": dagzoo_handoff_module.DAGZOO_HANDOFF_SCHEMA_NAME,
                "schema_version": dagzoo_handoff_module.DAGZOO_HANDOFF_SCHEMA_VERSION,
                "identity": {
                    "source_family": "dagzoo.heterogeneous_scm",
                    "generate_run_id": generate_run_id,
                    "generated_corpus_id": dagzoo_handoff_module.stable_dagzoo_generated_corpus_id(
                        generate_run_id=generate_run_id,
                        dataset_ids=[dataset_id],
                    ),
                },
                "artifacts_relative": {
                    "generated_dir": "generated",
                    "curated_dir": "curated",
                },
                "provenance": {
                    "intervention": {"mode": "none", "signature": "c" * 32},
                    "target_derivation": "tabiclv2_latent_node",
                    "target_relevant_feature_count_range": {"min": 1, "max": 4},
                    "target_relevant_feature_fraction_range": {"min": 0.25, "max": 0.75},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.parquet"
    summary = manifest_module.build_manifest(
        [generated_root],
        manifest_path,
        dagzoo_handoff_manifest_path=handoff_manifest_path,
    )
    expected_provenance = {
        "intervention": {"mode": "none", "signature": "c" * 32},
        "target_derivation": "tabiclv2_latent_node",
        "target_relevant_feature_count_range": {"min": 1, "max": 4},
        "target_relevant_feature_fraction_range": {"min": 0.25, "max": 0.75},
    }

    assert summary.dagzoo_handoff is not None
    assert summary.dagzoo_handoff["provenance"] == expected_provenance
    assert "teacher_conditionals" not in summary.dagzoo_handoff
    assert summary.dagzoo_handoff["generated_dir"] == str(generated_root.resolve())
    assert summary.dagzoo_handoff["curated_dir"] == str((tmp_path / "curated").resolve())
    assert summary.dagzoo_handoff["handoff_manifest_path"] == str(handoff_manifest_path.resolve())
    assert len(summary.dagzoo_handoff["handoff_manifest_sha256"]) == 64

    inspection = manifest_module.inspect_manifest(manifest_path)
    assert inspection["persisted_summary"]["dagzoo_handoff"]["provenance"] == expected_provenance


def test_dagzoo_handoff_v4_remains_supported(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    generate_run_id = "a" * 32
    dataset_id = "b" * 32
    _write_dataset(
        generated_root / "shard_00001_case",
        metadata={
            "dataset_id": dataset_id,
            "split_groups": {"request_run": generate_run_id},
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "generated", "status": "accepted", "accepted": True},
        },
    )
    handoff_manifest_path = tmp_path / "dagzoo_handoff.json"
    handoff_manifest_path.write_text(
        json.dumps(
            {
                "schema_name": dagzoo_handoff_module.DAGZOO_HANDOFF_SCHEMA_NAME,
                "schema_version": 4,
                "identity": {
                    "source_family": "dagzoo",
                    "generate_run_id": generate_run_id,
                    "generated_corpus_id": dagzoo_handoff_module.stable_dagzoo_generated_corpus_id(
                        generate_run_id=generate_run_id,
                        dataset_ids=[dataset_id],
                    ),
                },
                "artifacts_relative": {"generated_dir": "generated"},
                "provenance": {
                    "target_derivation": "tabiclv2_latent_node",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = manifest_module.build_manifest(
        [generated_root],
        tmp_path / "manifest.parquet",
        dagzoo_handoff_manifest_path=handoff_manifest_path,
    )

    assert summary.dagzoo_handoff is not None
    assert summary.dagzoo_handoff["provenance"] == {"target_derivation": "tabiclv2_latent_node"}


def test_build_manifest_accepts_curated_root_without_embedded_filter_metadata(
    tmp_path: Path,
) -> None:
    curated_root = tmp_path / "curated"
    _write_dataset(
        curated_root / "shard_00001_case",
        metadata={
            "dataset_id": "c" * 32,
            "split_groups": {"request_run": "d" * 32},
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
        },
    )

    manifest_path = tmp_path / "manifest.parquet"
    summary = manifest_module.build_manifest(
        [curated_root],
        manifest_path,
        filter_policy="accepted_only",
    )
    inspection = manifest_module.inspect_manifest(manifest_path)

    assert summary.total_records == 1
    assert summary.filter_status_counts == {"accepted": 1}
    assert inspection["persisted_summary"]["filter_policy"] == "accepted_only"


def test_build_manifest_allow_any_skips_value_column_scans(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    curated_root = tmp_path / "curated"
    _write_dataset(
        curated_root / "shard_00001_case",
        metadata={
            "dataset_id": "c" * 32,
            "split_groups": {"request_run": "d" * 32},
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
        },
    )
    original_read_table = manifest_module.pq.read_table
    observed_paths: list[Path] = []

    def read_table_spy(*args: Any, **kwargs: Any) -> pa.Table:
        raw_path = args[0] if args else kwargs.get("source")
        if raw_path is not None:
            observed_paths.append(Path(str(raw_path)).resolve())
        columns = kwargs.get("columns")
        if columns is not None:
            normalized_columns = tuple(str(column) for column in columns)
            assert "x" not in normalized_columns
            assert "y" not in normalized_columns
        return original_read_table(*args, **kwargs)

    monkeypatch.setattr(manifest_module.pq, "read_table", read_table_spy)

    manifest_path = tmp_path / "manifest.parquet"
    summary = manifest_module.build_manifest(
        [curated_root],
        manifest_path,
        filter_policy="accepted_only",
        missing_value_policy="allow_any",
    )
    rows = original_read_table(manifest_path).to_pylist()
    inspection = manifest_module.inspect_manifest(manifest_path)
    captured = capsys.readouterr()

    assert all(path.name not in {"train.parquet", "test.parquet"} for path in observed_paths)
    assert summary.total_records == 1
    assert summary.missing_value_status_counts == {"not_checked": 1}
    assert rows[0]["missing_value_status"] == "not_checked"
    assert inspection["persisted_summary"]["missing_value_policy"] == "allow_any"
    assert inspection["persisted_summary"]["total_records"] == 1
    assert (
        inspection["persisted_summary"]["train_records"]
        + inspection["persisted_summary"]["val_records"]
        + inspection["persisted_summary"]["test_records"]
        == 1
    )
    assert inspection["missing_value_status_counts"] == {"not_checked": 1}
    assert "manifest build started:" in captured.err
    assert "manifest build writing manifest parquet:" in captured.err
    assert "manifest build manifest parquet written:" in captured.err


def test_build_manifest_allow_any_ignores_missing_split_dataset_indices(tmp_path: Path) -> None:
    root = tmp_path / "run"
    shard_dir = root / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    x_train = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    pq.write_table(_build_split_table([(0, x_train, y_train)]), shard_dir / "train.parquet")
    pq.write_table(_build_split_table([]), shard_dir / "test.parquet")
    payload = {
        "dataset_index": 0,
        "n_train": 2,
        "n_test": 0,
        "n_features": 2,
        "feature_types": ["floating", "floating"],
        "metadata": {
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "deferred", "status": "accepted", "accepted": True},
        },
    }
    manifest_module.write_dataset_catalog(
        shard_dir / manifest_module.DATASET_CATALOG_FILENAME,
        [payload],
    )

    summary = manifest_module.build_manifest(
        [root],
        tmp_path / "manifest.parquet",
        filter_policy="accepted_only",
        missing_value_policy="allow_any",
    )

    assert summary.total_records == 1
    assert summary.missing_value_status_counts == {"not_checked": 1}


def test_build_manifest_forbid_any_still_validates_split_dataset_indices(tmp_path: Path) -> None:
    root = tmp_path / "run"
    shard_dir = root / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    x_train = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    pq.write_table(_build_split_table([(0, x_train, y_train)]), shard_dir / "train.parquet")
    pq.write_table(_build_split_table([]), shard_dir / "test.parquet")
    payload = {
        "dataset_index": 0,
        "n_train": 2,
        "n_test": 0,
        "n_features": 2,
        "feature_types": ["floating", "floating"],
        "metadata": {
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "deferred", "status": "accepted", "accepted": True},
        },
    }
    manifest_module.write_dataset_catalog(
        shard_dir / manifest_module.DATASET_CATALOG_FILENAME,
        [payload],
    )

    with pytest.raises(RuntimeError, match="test.parquet"):
        manifest_module.build_manifest(
            [root],
            tmp_path / "manifest.parquet",
            filter_policy="accepted_only",
            missing_value_policy="forbid_any",
        )


def test_build_manifest_manifest_workers_matches_serial_output(tmp_path: Path) -> None:
    root = tmp_path / "packed_shards"
    for shard_index in range(3):
        _write_dataset(
            root / f"shard_{shard_index:05d}_case",
            metadata={
                "dataset_id": f"{shard_index + 1:032x}",
                "split_groups": {"request_run": "f" * 32},
                "n_features": 2,
                "n_classes": 2,
                "seed": shard_index,
                "config": {"dataset": {"task": "classification"}},
                "filter": {"mode": "curated", "status": "accepted", "accepted": True},
            },
        )

    serial_manifest_path = tmp_path / "serial.parquet"
    parallel_manifest_path = tmp_path / "parallel.parquet"
    _ = manifest_module.build_manifest(
        [root],
        serial_manifest_path,
        filter_policy="accepted_only",
        manifest_workers=1,
    )
    _ = manifest_module.build_manifest(
        [root],
        parallel_manifest_path,
        filter_policy="accepted_only",
        manifest_workers=4,
    )

    assert (
        pq.read_table(serial_manifest_path).to_pylist()
        == pq.read_table(parallel_manifest_path).to_pylist()
    )


def test_load_manifest_datasets_reads_generic_manifest_surface(tmp_path: Path) -> None:
    root = tmp_path / "packed_shards"
    _write_dataset(
        root / "shard_00001_case",
        metadata={
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "deferred", "status": "not_run"},
            "observed_task": {"dataset_name": "generic_case"},
        },
    )
    manifest_path = tmp_path / "manifest.parquet"
    _ = manifest_module.build_manifest([root], manifest_path)

    loaded = manifest_module.load_manifest_datasets(manifest_path)

    assert loaded.contract_version == manifest_module.MANIFEST_CONTRACT_VERSION
    assert len(loaded.manifest_sha256) == 64
    assert list(loaded.datasets) == ["generic_case"]
    x, y = loaded.datasets["generic_case"]
    assert x.shape == (3, 2)
    assert y.tolist() == [0, 1, 1]
    assert loaded.task_records[0]["row_order_mode"] == "split_concat"
    assert loaded.task_records[0]["metadata"]["observed_task"]["dataset_name"] == "generic_case"


def test_dataset_catalog_helpers_round_trip_parquet_and_legacy(tmp_path: Path) -> None:
    payload = {
        "dataset_index": 7,
        "dataset_id": "a" * 32,
        "group_ids": {"request_run": "b" * 32},
        "n_train": 4,
        "n_test": 2,
        "n_features": 3,
        "n_classes": 2,
        "feature_types": ["floating", "floating", "floating"],
        "metadata": {
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "curated", "status": "accepted", "accepted": True},
        },
    }
    parquet_path = tmp_path / "dataset_catalog.parquet"
    legacy_path = tmp_path / "metadata.ndjson"
    manifest_module.write_dataset_catalog(parquet_path, [payload])
    legacy_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    assert manifest_module.load_dataset_catalog_records(parquet_path) == [payload]
    assert manifest_module.load_dataset_catalog_records(legacy_path) == [payload]


def test_build_manifest_supports_mixed_parquet_and_legacy_catalog_roots(tmp_path: Path) -> None:
    parquet_root = tmp_path / "parquet_root"
    legacy_root = tmp_path / "legacy_root"
    _write_dataset(
        parquet_root / "shard_00001_case",
        metadata={
            "n_features": 2,
            "n_classes": 2,
            "seed": 7,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "curated", "status": "accepted", "accepted": True},
        },
    )
    _write_dataset(
        legacy_root / "shard_00002_case",
        metadata={
            "n_features": 2,
            "n_classes": 2,
            "seed": 8,
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "curated", "status": "accepted", "accepted": True},
        },
        legacy_catalog=True,
    )

    summary = manifest_module.build_manifest(
        [parquet_root, legacy_root],
        tmp_path / "manifest.parquet",
        filter_policy="accepted_only",
    )

    assert summary.total_records == 2


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
                    {
                        "task_id": 101,
                        "dataset_name": "first_dataset",
                        "n_rows": 10,
                        "n_features": 3,
                        "n_classes": 2,
                    },
                    {
                        "task_id": 102,
                        "dataset_name": "second_dataset",
                        "n_rows": 10,
                        "n_features": 2,
                        "n_classes": 2,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prepared_tasks = {
        101: _prepared_task(
            task_id=101, dataset_name="first_dataset", n_rows=10, n_features=3, n_classes=2
        ),
        102: _prepared_task(
            task_id=102, dataset_name="second_dataset", n_rows=10, n_features=2, n_classes=2
        ),
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

    catalog_path = (
        result.data_root / "shard_00001_first_dataset" / manifest_module.DATASET_CATALOG_FILENAME
    )
    payload = manifest_module.load_dataset_catalog_records(catalog_path)[0]
    assert payload["feature_types"] == ["floating", "floating", "floating"]
    assert payload["metadata"]["source_platform"] == "openml"
    assert payload["metadata"]["benchmark_bundle"]["source_path"] == str(bundle_path.resolve())
    assert payload["metadata"]["openml"]["task_id"] == 101

    loaded = manifest_module.load_manifest_datasets(result.manifest_path)
    first_x, first_y = loaded.datasets["first_dataset"]
    expected = prepared_tasks[101]
    assert np.array_equal(first_x, expected.x)
    assert np.array_equal(first_y, expected.y)
    assert len(loaded.manifest_sha256) == 64
    assert loaded.task_records[0]["row_order_mode"] == "global_row_index"
    assert loaded.task_records[0]["metadata"]["benchmark_bundle"]["source_path"] == str(
        bundle_path.resolve()
    )


def test_load_manifest_datasets_requires_contract_metadata(tmp_path: Path) -> None:
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
    _ = manifest_module.build_manifest([root], manifest_path)
    table = pq.read_table(manifest_path)
    pq.write_table(table.replace_schema_metadata(None), manifest_path)

    try:
        manifest_module.load_manifest_datasets(manifest_path)
    except RuntimeError as exc:
        assert "manifest contract metadata is missing" in str(exc)
    else:  # pragma: no cover - defensive failure
        raise AssertionError("expected missing contract metadata to raise")
