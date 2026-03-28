"""Manifest builder and inspection helpers for packed parquet shard outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import md5, sha1, sha256
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dagzoo_handoff import (
    DagzooGeneratedIdentityAccumulator,
    is_canonical_dagzoo_id,
    load_dagzoo_handoff_info,
    verify_dagzoo_handoff_matches_generated_corpus,
)
from .validation import (
    MISSING_VALUE_STATUS_CLEAN,
    MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF,
    SUPPORTED_MISSING_VALUE_POLICIES,
    missing_value_status,
)


MANIFEST_SUMMARY_METADATA_KEY = b"tab_foundry_manifest_summary"
HEX_DIGEST_RADIX = 16
SPLIT_BUCKET_COUNT = 10_000
SHORT_DIGEST_HEX_CHARS = 12
DATASET_INDEX_WIDTH = 6
SUPPORTED_FILTER_POLICIES = ("include_all", "accepted_only")


@dataclass(slots=True)
class ManifestSummary:
    """Build summary."""

    out_path: Path
    filter_policy: str
    discovered_records: int
    excluded_records: int
    total_records: int
    train_records: int
    val_records: int
    test_records: int
    missing_value_policy: str = "allow_any"
    excluded_for_missing_values: int = 0
    filter_status_counts: dict[str, int] = field(default_factory=dict)
    missing_value_status_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dagzoo_handoff: dict[str, Any] | None = None


def _stable_split(key: str, train_ratio: float, val_ratio: float) -> str:
    """Split by deterministic hash."""

    token = int(md5(key.encode("utf-8")).hexdigest(), HEX_DIGEST_RADIX) % SPLIT_BUCKET_COUNT
    p = token / float(SPLIT_BUCKET_COUNT)
    if p < train_ratio:
        return "train"
    if p < train_ratio + val_ratio:
        return "val"
    return "test"


def _root_id(root: Path) -> str:
    token = root.expanduser().resolve().as_posix().encode("utf-8")
    return sha1(token).hexdigest()[:SHORT_DIGEST_HEX_CHARS]


def _dataset_id(
    *,
    root_id: str,
    shard_relpath: str,
    dataset_index: int,
) -> str:
    """Stable dataset ID with root-level uniqueness."""

    normalized_relpath = shard_relpath.strip("/").replace(os.sep, "/")
    token = json.dumps(
        {
            "root_id": root_id,
            "shard_relpath": normalized_relpath,
            "dataset_index": int(dataset_index),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = md5(token.encode("utf-8")).hexdigest()[:SHORT_DIGEST_HEX_CHARS]
    return (
        f"root_{root_id}/{normalized_relpath}/"
        f"dataset_{dataset_index:0{DATASET_INDEX_WIDTH}d}_{digest}"
    )


def _canonical_dagzoo_dataset_identity_key(
    *,
    dataset_id: str,
    request_run: str,
) -> str:
    return f"dagzoo_request_{request_run}/dataset_{dataset_id}"


def _resolved_manifest_identity(
    *,
    metadata: dict[str, Any],
    root_id: str,
    shard_relpath: str,
    dataset_index: int,
) -> tuple[str, str]:
    canonical_dataset_id = metadata.get("dataset_id")
    split_groups = metadata.get("split_groups")
    request_run = split_groups.get("request_run") if isinstance(split_groups, dict) else None
    if is_canonical_dagzoo_id(canonical_dataset_id) and is_canonical_dagzoo_id(request_run):
        dataset_id = str(canonical_dataset_id)
        return dataset_id, _canonical_dagzoo_dataset_identity_key(
            dataset_id=dataset_id,
            request_run=str(request_run),
        )
    dataset_id = _dataset_id(
        root_id=root_id,
        shard_relpath=shard_relpath,
        dataset_index=dataset_index,
    )
    return dataset_id, dataset_id


def _manifest_relative_path(path: Path, *, manifest_dir: Path) -> str:
    """Serialize data path relative to manifest directory when possible."""

    absolute = path.expanduser().resolve()
    try:
        return os.path.relpath(absolute, start=manifest_dir)
    except ValueError:
        return absolute.as_posix()


def _infer_task(meta: dict[str, Any]) -> str:
    config_task = meta.get("config", {}).get("dataset", {}).get("task")
    if config_task in {"classification", "regression"}:
        return str(config_task)
    n_classes = meta.get("n_classes")
    return "classification" if n_classes is not None else "regression"


def _shard_relpath(root: Path, shard_dir: Path) -> str:
    try:
        relpath = shard_dir.relative_to(root)
    except ValueError:
        return os.path.relpath(shard_dir, start=root)
    return relpath.as_posix()


def _parse_filter_metadata(meta: dict[str, Any]) -> tuple[str | None, str | None, bool | None]:
    filter_raw = meta.get("filter")
    if not isinstance(filter_raw, dict):
        return None, None, None

    mode_raw = filter_raw.get("mode")
    status_raw = filter_raw.get("status")
    accepted_raw = filter_raw.get("accepted")

    mode = str(mode_raw) if isinstance(mode_raw, str) and mode_raw.strip() else None
    status = str(status_raw) if isinstance(status_raw, str) and status_raw.strip() else None
    accepted = accepted_raw if isinstance(accepted_raw, bool) else None
    return mode, status, accepted


def _status_bucket(status: str | None) -> str:
    return status if status is not None else "missing"


def _is_record_selected(
    *,
    filter_policy: str,
    filter_status: str | None,
    filter_accepted: bool | None,
) -> bool:
    if filter_policy == "include_all":
        return True
    if filter_policy == "accepted_only":
        return filter_status == "accepted" or filter_accepted is True
    raise ValueError(f"Unsupported filter_policy: {filter_policy!r}")


def _build_manifest_warnings(
    *,
    filter_policy: str,
    missing_value_policy: str,
    selected_records: int,
    excluded_records: int,
    excluded_for_missing_values: int,
    filter_status_counts: Counter[str],
) -> list[str]:
    warnings: list[str] = []
    if filter_policy == "include_all":
        included_unaccepted = sum(
            count
            for status, count in filter_status_counts.items()
            if status in {"missing", "not_run", "rejected"}
        )
        if included_unaccepted > 0:
            warnings.append(
                "Included datasets without accepted filter status "
                f"(count={included_unaccepted})."
            )
    elif excluded_records > 0:
        warnings.append(
            f"Excluded {excluded_records} dataset(s) under filter_policy={filter_policy}."
        )
    if missing_value_policy == "forbid_any" and excluded_for_missing_values > 0:
        warnings.append(
            "Excluded "
            f"{excluded_for_missing_values} dataset(s) containing NaN or Inf under "
            "missing_value_policy=forbid_any."
        )

    if selected_records <= 0:
        warnings.append("No records were selected into the output manifest.")
    return warnings


def _extract_shard_id(shard_dir: Path) -> int:
    name = shard_dir.name
    if not name.startswith("shard_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return -1


def _iter_shard_dirs(root: Path) -> list[Path]:
    """Return sorted shard directories."""

    shard_dirs: list[Path] = []
    for shard_dir in sorted(root.rglob("shard_*")):
        if shard_dir.is_dir():
            shard_dirs.append(shard_dir)
    return shard_dirs


def _read_metadata_records(metadata_path: Path) -> list[tuple[int, int, str, dict[str, Any]]]:
    """Read metadata.ndjson records and include byte offsets for random access."""

    records: list[tuple[int, int, str, dict[str, Any]]] = []
    with metadata_path.open("rb") as handle:
        while True:
            offset = int(handle.tell())
            line = handle.readline()
            if not line:
                break

            size = int(len(line))
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped.decode("utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"failed to parse NDJSON metadata record in {metadata_path} at byte offset {offset}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"metadata record must be a JSON object: path={metadata_path}, offset={offset}"
                )
            records.append((offset, size, sha256(line).hexdigest(), payload))
    return records


def _coerce_optional_int(value: Any, *, default: int, context: str) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be int-compatible or null, got {value!r}") from exc


def _missing_value_status_by_dataset(split_path: Path) -> dict[int, str]:
    """Summarize non-finite status for each dataset_index in one packed split parquet."""

    try:
        table = pq.read_table(split_path, columns=["dataset_index", "x", "y"])
    except Exception as exc:
        raise RuntimeError(f"failed to scan packed split for missing values: {split_path}") from exc

    if table.num_rows <= 0:
        return {}

    dataset_indices = table["dataset_index"].to_pylist()
    x_rows = table["x"].to_pylist()
    y_values = table["y"].to_pylist()
    status_by_dataset: dict[int, str] = {}
    for dataset_index, x_row, y_value in zip(dataset_indices, x_rows, y_values, strict=False):
        key = int(dataset_index)
        if status_by_dataset.get(key) == MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF:
            continue
        current = missing_value_status(
            {"x": x_row, "y": np.asarray([y_value])},
            context=f"packed split {split_path} dataset_index={key}",
        )
        status_by_dataset[key] = current
    return status_by_dataset


def _manifest_schema_metadata(*, summary: ManifestSummary) -> dict[bytes, bytes]:
    payload = {
        "filter_policy": summary.filter_policy,
        "missing_value_policy": summary.missing_value_policy,
        "discovered_records": int(summary.discovered_records),
        "excluded_records": int(summary.excluded_records),
        "excluded_for_missing_values": int(summary.excluded_for_missing_values),
        "total_records": int(summary.total_records),
        "train_records": int(summary.train_records),
        "val_records": int(summary.val_records),
        "test_records": int(summary.test_records),
        "filter_status_counts": dict(summary.filter_status_counts),
        "missing_value_status_counts": dict(summary.missing_value_status_counts),
    }
    if summary.dagzoo_handoff is not None:
        payload["dagzoo_handoff"] = dict(summary.dagzoo_handoff)
    return {MANIFEST_SUMMARY_METADATA_KEY: json.dumps(payload, sort_keys=True).encode("utf-8")}


def build_manifest(
    data_roots: list[Path],
    out_path: Path,
    *,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    filter_policy: str = "include_all",
    missing_value_policy: str = "allow_any",
    dagzoo_handoff_manifest_path: Path | None = None,
) -> ManifestSummary:
    """Scan parquet roots and persist manifest parquet."""

    if not data_roots:
        raise ValueError("data_roots must not be empty")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")
    if filter_policy not in SUPPORTED_FILTER_POLICIES:
        raise ValueError(
            f"filter_policy must be one of {SUPPORTED_FILTER_POLICIES}, got {filter_policy!r}"
        )
    if missing_value_policy not in SUPPORTED_MISSING_VALUE_POLICIES:
        raise ValueError(
            "missing_value_policy must be one of "
            f"{SUPPORTED_MISSING_VALUE_POLICIES}, got {missing_value_policy!r}"
        )

    out_path = out_path.expanduser().resolve()
    manifest_dir = out_path.parent
    roots = sorted({root.expanduser().resolve() for root in data_roots})

    records: list[dict[str, Any]] = []
    discovered_records = 0
    status_counts: Counter[str] = Counter()
    missing_value_status_counts: Counter[str] = Counter()
    excluded_for_missing_values = 0
    dagzoo_generated_identity = (
        None if dagzoo_handoff_manifest_path is None else DagzooGeneratedIdentityAccumulator()
    )
    for root in roots:
        if not root.exists():
            continue
        source_root_id = _root_id(root)
        for shard_dir in _iter_shard_dirs(root):
            train_path = shard_dir / "train.parquet"
            test_path = shard_dir / "test.parquet"
            metadata_path = shard_dir / "metadata.ndjson"
            if not (train_path.exists() and test_path.exists() and metadata_path.exists()):
                continue
            train_missing_status = _missing_value_status_by_dataset(train_path)
            test_missing_status = _missing_value_status_by_dataset(test_path)

            shard_id = _extract_shard_id(shard_dir)
            for offset, size, record_sha256, record in _read_metadata_records(metadata_path):
                if "dataset_index" not in record:
                    raise RuntimeError(
                        f"metadata record missing dataset_index: path={metadata_path}, offset={offset}"
                    )
                dataset_index = int(record["dataset_index"])
                meta_raw = record.get("metadata")
                if not isinstance(meta_raw, dict):
                    raise RuntimeError(
                        "metadata record missing object payload at key 'metadata': "
                        f"path={metadata_path}, dataset_index={dataset_index}"
                    )
                meta = meta_raw
                discovered_records += 1
                if dagzoo_generated_identity is not None:
                    dagzoo_generated_identity.add_metadata(
                        meta,
                        metadata_path=metadata_path,
                        dataset_index=dataset_index,
                    )

                source_shard_relpath = _shard_relpath(root, shard_dir)
                filter_mode, filter_status, filter_accepted = _parse_filter_metadata(meta)
                status_counts[_status_bucket(filter_status)] += 1
                if not _is_record_selected(
                    filter_policy=filter_policy,
                    filter_status=filter_status,
                    filter_accepted=filter_accepted,
                ):
                    continue
                missing_splits: list[str] = []
                if dataset_index not in train_missing_status:
                    missing_splits.append(train_path.name)
                if dataset_index not in test_missing_status:
                    missing_splits.append(test_path.name)
                if missing_splits:
                    raise RuntimeError(
                        "metadata dataset_index missing from packed split(s): "
                        f"shard={shard_dir}, path={metadata_path}, dataset_index={dataset_index}, "
                        f"missing_splits={','.join(missing_splits)}"
                    )
                train_missing_value_status = train_missing_status[dataset_index]
                test_missing_value_status = test_missing_status[dataset_index]
                record_missing_value_status = (
                    MISSING_VALUE_STATUS_CLEAN
                    if train_missing_value_status == MISSING_VALUE_STATUS_CLEAN
                    and test_missing_value_status == MISSING_VALUE_STATUS_CLEAN
                    else MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF
                )
                missing_value_status_counts[record_missing_value_status] += 1
                if (
                    missing_value_policy == "forbid_any"
                    and record_missing_value_status != MISSING_VALUE_STATUS_CLEAN
                ):
                    excluded_for_missing_values += 1
                    continue

                dsid, dataset_identity_key = _resolved_manifest_identity(
                    metadata=meta,
                    root_id=source_root_id,
                    shard_relpath=source_shard_relpath,
                    dataset_index=dataset_index,
                )
                split = _stable_split(dataset_identity_key, train_ratio, val_ratio)

                records.append(
                    {
                        "dataset_id": dsid,
                        "dataset_identity_key": dataset_identity_key,
                        "source_root_id": source_root_id,
                        "source_shard_relpath": source_shard_relpath,
                        "split": split,
                        "task": _infer_task(meta),
                        "shard_id": shard_id,
                        "dataset_index": dataset_index,
                        "train_path": _manifest_relative_path(train_path, manifest_dir=manifest_dir),
                        "test_path": _manifest_relative_path(test_path, manifest_dir=manifest_dir),
                        "metadata_path": _manifest_relative_path(metadata_path, manifest_dir=manifest_dir),
                        "metadata_offset_bytes": offset,
                        "metadata_size_bytes": size,
                        "metadata_sha256": record_sha256,
                        "n_train": int(record.get("n_train", -1)),
                        "n_test": int(record.get("n_test", -1)),
                        "n_features": _coerce_optional_int(
                            record.get("n_features", meta.get("n_features", -1)),
                            default=-1,
                            context=(
                                f"metadata.n_features path={metadata_path} "
                                f"dataset_index={dataset_index}"
                            ),
                        ),
                        "n_classes": (
                            int(meta["n_classes"]) if meta.get("n_classes") is not None else None
                        ),
                        "seed": int(meta.get("seed", -1)),
                        "filter_mode": filter_mode,
                        "filter_status": filter_status,
                        "filter_accepted": filter_accepted,
                        "missing_value_policy": missing_value_policy,
                        "missing_value_status": record_missing_value_status,
                    }
                )

    if discovered_records <= 0:
        raise RuntimeError("no datasets discovered while building manifest")
    if not records:
        raise RuntimeError(
            "no datasets matched "
            f"filter_policy={filter_policy!r} and missing_value_policy={missing_value_policy!r} "
            "while building manifest"
        )

    records.sort(
        key=lambda record: (
            str(record["source_root_id"]),
            str(record["source_shard_relpath"]),
            int(record["dataset_index"]),
            str(record["dataset_identity_key"]),
            str(record["dataset_id"]),
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_records = sum(1 for record in records if record["split"] == "train")
    val_records = sum(1 for record in records if record["split"] == "val")
    test_records = sum(1 for record in records if record["split"] == "test")
    excluded_records = discovered_records - len(records)
    dagzoo_handoff = (
        None
        if dagzoo_handoff_manifest_path is None
        else _verified_dagzoo_handoff_summary(
            dagzoo_handoff_manifest_path=dagzoo_handoff_manifest_path,
            dagzoo_generated_identity=dagzoo_generated_identity,
        )
    )
    summary = ManifestSummary(
        out_path=out_path,
        filter_policy=filter_policy,
        missing_value_policy=missing_value_policy,
        discovered_records=discovered_records,
        excluded_records=excluded_records,
        excluded_for_missing_values=excluded_for_missing_values,
        total_records=len(records),
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        filter_status_counts=dict(sorted(status_counts.items())),
        missing_value_status_counts=dict(sorted(missing_value_status_counts.items())),
        warnings=_build_manifest_warnings(
            filter_policy=filter_policy,
            missing_value_policy=missing_value_policy,
            selected_records=len(records),
            excluded_records=excluded_records,
            excluded_for_missing_values=excluded_for_missing_values,
            filter_status_counts=status_counts,
        ),
        dagzoo_handoff=dagzoo_handoff,
    )
    table = pa.Table.from_pylist(records).replace_schema_metadata(
        _manifest_schema_metadata(summary=summary)
    )
    pq.write_table(table, out_path, compression="zstd")
    return summary


def _verified_dagzoo_handoff_summary(
    *,
    dagzoo_handoff_manifest_path: Path,
    dagzoo_generated_identity: DagzooGeneratedIdentityAccumulator | None,
) -> dict[str, Any]:
    if dagzoo_generated_identity is None:
        raise RuntimeError("dagzoo handoff verification state was not initialized")
    handoff = load_dagzoo_handoff_info(dagzoo_handoff_manifest_path)
    verify_dagzoo_handoff_matches_generated_corpus(
        handoff,
        scanned_identity=dagzoo_generated_identity,
    )
    return handoff.to_summary_dict()


def _read_persisted_manifest_summary(manifest_path: Path) -> dict[str, Any] | None:
    metadata = pq.ParquetFile(manifest_path).schema_arrow.metadata or {}
    raw_summary = metadata.get(MANIFEST_SUMMARY_METADATA_KEY)
    if raw_summary is None:
        return None
    payload = json.loads(raw_summary.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"persisted manifest summary must be an object: {manifest_path}")
    return cast(dict[str, Any], payload)


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(sum(values) / float(len(values))),
    }


def inspect_manifest(manifest_path: Path) -> dict[str, Any]:
    """Inspect one manifest parquet file and summarize its contents."""

    resolved_manifest = manifest_path.expanduser().resolve()
    if not resolved_manifest.exists():
        raise RuntimeError(f"manifest does not exist: {resolved_manifest}")
    if not resolved_manifest.is_file():
        raise RuntimeError(f"manifest path is not a file: {resolved_manifest}")

    table = pq.read_table(resolved_manifest)
    records = cast(list[dict[str, Any]], table.to_pylist())
    if not records:
        raise RuntimeError(f"manifest has zero rows: {resolved_manifest}")

    split_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    task_split_counts: dict[str, Counter[str]] = {}
    task_train_test_record_counts: dict[str, Counter[str]] = {}
    filter_status_counts: Counter[str] = Counter()
    missing_value_status_counts: Counter[str] = Counter()
    task_missing_value_status_counts: dict[str, Counter[str]] = {}
    n_features_values: list[int] = []
    classification_n_classes: list[int] = []
    source_roots: set[str] = set()
    dataset_ids: set[str] = set()

    for record in records:
        split = str(record.get("split", "unknown"))
        task = str(record.get("task", "unknown"))
        split_counts[split] += 1
        task_counts[task] += 1
        task_split_counts.setdefault(task, Counter())[split] += 1
        if record.get("n_train") is not None and int(record["n_train"]) > 0:
            task_train_test_record_counts.setdefault(task, Counter())["train"] += 1
        if record.get("n_test") is not None and int(record["n_test"]) > 0:
            task_train_test_record_counts.setdefault(task, Counter())["test"] += 1

        raw_filter_status = record.get("filter_status")
        filter_status = "missing" if raw_filter_status is None else str(raw_filter_status)
        filter_status_counts[filter_status] += 1

        raw_missing_value_status = record.get("missing_value_status")
        missing_status = "missing" if raw_missing_value_status is None else str(raw_missing_value_status)
        missing_value_status_counts[missing_status] += 1
        task_missing_value_status_counts.setdefault(task, Counter())[missing_status] += 1

        raw_n_features = record.get("n_features")
        if raw_n_features is not None:
            n_features_values.append(int(raw_n_features))

        if task == "classification" and record.get("n_classes") is not None:
            classification_n_classes.append(int(record["n_classes"]))

        source_root = record.get("source_root_id")
        if isinstance(source_root, str) and source_root.strip():
            source_roots.add(source_root)
        dataset_id = record.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            dataset_ids.add(dataset_id)

    n_class_histogram = Counter(classification_n_classes)
    return {
        "manifest_path": str(resolved_manifest),
        "total_records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "task_split_counts": {
            str(task): dict(sorted(counts.items()))
            for task, counts in sorted(task_split_counts.items())
        },
        "task_train_test_record_counts": {
            str(task): dict(sorted(counts.items()))
            for task, counts in sorted(task_train_test_record_counts.items())
        },
        "filter_status_counts": dict(sorted(filter_status_counts.items())),
        "missing_value_status_counts": dict(sorted(missing_value_status_counts.items())),
        "task_missing_value_status_counts": {
            str(task): dict(sorted(counts.items()))
            for task, counts in sorted(task_missing_value_status_counts.items())
        },
        "n_features": (
            None
            if not n_features_values
            else {
                "min": int(min(n_features_values)),
                "max": int(max(n_features_values)),
            }
        ),
        "classification_n_classes": (
            None
            if not classification_n_classes
            else {
                "min": int(min(classification_n_classes)),
                "max": int(max(classification_n_classes)),
                "histogram": {
                    str(class_count): int(count)
                    for class_count, count in sorted(n_class_histogram.items())
                },
            }
        ),
        "unique_source_root_count": len(source_roots),
        "unique_dataset_id_count": len(dataset_ids),
        "persisted_summary": _read_persisted_manifest_summary(resolved_manifest),
    }


def manifest_characteristics(manifest_path: Path) -> dict[str, Any]:
    """Return a richer manifest summary for training and corpus surfaces."""

    resolved_manifest = manifest_path.expanduser().resolve()
    parquet_file = pq.ParquetFile(resolved_manifest)
    table = parquet_file.read()
    rows = cast(list[dict[str, Any]], table.to_pylist())
    missing_value_statuses = [
        str(status).strip()
        if isinstance((status := row.get("missing_value_status")), str) and str(status).strip()
        else None
        for row in rows
    ]
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    task_counts = Counter(str(row.get("task", "missing")) for row in rows)
    filter_status_counts = Counter(str(row.get("filter_status", "missing")) for row in rows)
    missing_value_status_counts = Counter(str(row.get("missing_value_status", "missing")) for row in rows)
    has_complete_missing_value_metadata = bool(rows) and all(
        status is not None for status in missing_value_statuses
    )
    missing_value_policies = sorted(
        {
            str(row["missing_value_policy"])
            for row in rows
            if isinstance(row.get("missing_value_policy"), str) and row["missing_value_policy"].strip()
        }
    )
    source_root_ids = sorted(
        {
            str(row["source_root_id"])
            for row in rows
            if isinstance(row.get("source_root_id"), str) and row["source_root_id"].strip()
        }
    )
    shard_counts = Counter(
        str(row["source_shard_relpath"])
        for row in rows
        if isinstance(row.get("source_shard_relpath"), str) and row["source_shard_relpath"].strip()
    )
    total_rows = [
        int(row["n_train"]) + int(row["n_test"])
        for row in rows
        if row.get("n_train") is not None and row.get("n_test") is not None
    ]
    n_features = [
        int(row["n_features"])
        for row in rows
        if row.get("n_features") is not None and int(row["n_features"]) >= 0
    ]
    n_classes = [
        int(row["n_classes"])
        for row in rows
        if row.get("n_classes") is not None
    ]
    raw_metadata = parquet_file.schema_arrow.metadata or {}
    persisted_summary = None
    raw_summary = raw_metadata.get(MANIFEST_SUMMARY_METADATA_KEY)
    if raw_summary is not None:
        persisted_summary = json.loads(raw_summary.decode("utf-8"))
    return {
        "record_count": int(len(rows)),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "row_count_distribution": _distribution(total_rows),
        "feature_count_distribution": _distribution(n_features),
        "class_count_distribution": _distribution(n_classes),
        "filter_status_counts": dict(sorted(filter_status_counts.items())),
        "missing_value_status_counts": dict(sorted(missing_value_status_counts.items())),
        "missing_value_policy": None if len(missing_value_policies) != 1 else missing_value_policies[0],
        "all_records_no_missing": (
            None
            if not has_complete_missing_value_metadata
            else missing_value_status_counts.get("contains_nan_or_inf", 0) == 0
        ),
        "persisted_summary": persisted_summary,
        "source_root_ids": source_root_ids,
        "source_shard_relpath_summary": {
            "unique_count": int(len(shard_counts)),
            "top_counts": [
                {"relpath": relpath, "count": int(count)}
                for relpath, count in shard_counts.most_common(10)
            ],
        },
    }


def compare_jsonlike_payloads(
    left: Any,
    right: Any,
    *,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """Return recursive differences between two JSON-like payloads."""

    if isinstance(left, dict) and isinstance(right, dict):
        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left.keys()) | set(right.keys())):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            differences.update(compare_jsonlike_payloads(left.get(key), right.get(key), prefix=next_prefix))
        return differences
    if left == right:
        return {}
    return {prefix: {"left": left, "right": right}}
