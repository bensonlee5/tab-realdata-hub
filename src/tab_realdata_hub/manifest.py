"""Manifest builder, inspection, and read helpers for packed parquet shard outputs."""

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
    DagzooHandoffInfo,
    is_canonical_dagzoo_id,
    load_dagzoo_handoff_info,
    verify_dagzoo_handoff_matches_generated_corpus,
)
from .validation import (
    MISSING_VALUE_STATUS_CLEAN,
    MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF,
    SUPPORTED_MISSING_VALUE_POLICIES,
    assert_no_non_finite_values,
    missing_value_status,
)


MANIFEST_CONTRACT_METADATA_KEY = b"tab_realdata_hub_manifest_contract"
MANIFEST_SUMMARY_METADATA_KEY = b"tab_foundry_manifest_summary"
MANIFEST_CONTRACT_VERSION = 2
SUPPORTED_MANIFEST_CONTRACT_VERSIONS = (1, 2)
MANIFEST_LOGICAL_LAYOUT = "parquet_index+dataset_catalog"
DATASET_CATALOG_FILENAME = "dataset_catalog.ndjson"
TEACHER_CONDITIONALS_FILENAME = "teacher_conditionals.parquet"
MANIFEST_STABLE_INDEX_FIELDS = (
    "dataset_id",
    "dataset_identity_key",
    "source_root_id",
    "source_shard_relpath",
    "split",
    "task",
    "shard_id",
    "dataset_index",
    "train_path",
    "test_path",
    "catalog_path",
    "catalog_offset_bytes",
    "catalog_size_bytes",
    "catalog_sha256",
    "teacher_conditionals_path",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "filter_mode",
    "filter_status",
    "filter_accepted",
    "missing_value_policy",
    "missing_value_status",
)
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


@dataclass(slots=True, frozen=True)
class LoadedManifestDatasets:
    """Canonical manifest-backed dataset payload for benchmark/helper execution."""

    manifest_path: Path
    contract_version: int
    manifest_sha256: str
    datasets: dict[str, tuple[np.ndarray, np.ndarray]]
    task_records: tuple[dict[str, Any], ...]
    persisted_summary: dict[str, Any] | None = None


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
    record_payload: dict[str, Any],
    root_id: str,
    shard_relpath: str,
    dataset_index: int,
) -> tuple[str, str]:
    canonical_dataset_id = record_payload.get("dataset_id")
    split_groups = record_payload.get("group_ids")
    if not isinstance(split_groups, dict):
        metadata = record_payload.get("metadata")
        split_groups = metadata.get("split_groups") if isinstance(metadata, dict) else None
        if canonical_dataset_id is None and isinstance(metadata, dict):
            canonical_dataset_id = metadata.get("dataset_id")
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


def _infer_task(record_payload: dict[str, Any]) -> str:
    task = record_payload.get("task")
    if task in {"classification", "regression"}:
        return str(task)
    metadata = record_payload.get("metadata")
    if isinstance(metadata, dict):
        config_task = metadata.get("config", {}).get("dataset", {}).get("task")
        if config_task in {"classification", "regression"}:
            return str(config_task)
        n_classes = metadata.get("n_classes")
        return "classification" if n_classes is not None else "regression"
    n_classes = record_payload.get("n_classes")
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
                f"(count={included_unaccepted}). Run `dagzoo filter --in <generated_dir> "
                "--out <filter_dir> --curated-out <curated_dir>` and rebuild the "
                "manifest from curated accepted-only shards at `<curated_dir>` with "
                "--filter-policy accepted_only."
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


def _read_ndjson_records(path: Path) -> list[tuple[int, int, str, dict[str, Any]]]:
    """Read NDJSON records and include byte offsets for random access."""

    records: list[tuple[int, int, str, dict[str, Any]]] = []
    with path.open("rb") as handle:
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
                    f"failed to parse NDJSON record in {path} at byte offset {offset}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"NDJSON record must be a JSON object: path={path}, offset={offset}"
                )
            records.append((offset, size, sha256(line).hexdigest(), payload))
    return records


def _catalog_path_for_shard(shard_dir: Path) -> Path | None:
    for filename in (DATASET_CATALOG_FILENAME, "metadata.ndjson"):
        candidate = shard_dir / filename
        if candidate.exists():
            return candidate
    return None


def _teacher_conditionals_path_for_shard(shard_dir: Path) -> Path | None:
    candidate = shard_dir / TEACHER_CONDITIONALS_FILENAME
    return candidate if candidate.exists() else None


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
    contract_payload = {
        "version": int(MANIFEST_CONTRACT_VERSION),
        "logical_layout": MANIFEST_LOGICAL_LAYOUT,
        "owner": "tab-realdata-hub",
        "stable_index_fields": list(MANIFEST_STABLE_INDEX_FIELDS),
    }
    payload = {
        "contract_version": int(MANIFEST_CONTRACT_VERSION),
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
    return {
        MANIFEST_CONTRACT_METADATA_KEY: json.dumps(contract_payload, sort_keys=True).encode("utf-8"),
        MANIFEST_SUMMARY_METADATA_KEY: json.dumps(payload, sort_keys=True).encode("utf-8"),
    }


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
    dagzoo_handoff = (
        None
        if dagzoo_handoff_manifest_path is None
        else load_dagzoo_handoff_info(dagzoo_handoff_manifest_path)
    )

    records: list[dict[str, Any]] = []
    discovered_records = 0
    status_counts: Counter[str] = Counter()
    missing_value_status_counts: Counter[str] = Counter()
    excluded_for_missing_values = 0
    dagzoo_generated_identity = None if dagzoo_handoff is None else DagzooGeneratedIdentityAccumulator()
    for root in roots:
        if not root.exists():
            continue
        source_root_id = _root_id(root)
        root_kind = None
        if dagzoo_handoff is not None:
            if root == dagzoo_handoff.generated_dir:
                root_kind = "generated"
            elif dagzoo_handoff.curated_dir is not None and root == dagzoo_handoff.curated_dir:
                root_kind = "curated"
        for shard_dir in _iter_shard_dirs(root):
            train_path = shard_dir / "train.parquet"
            test_path = shard_dir / "test.parquet"
            catalog_path = _catalog_path_for_shard(shard_dir)
            if catalog_path is None or not (train_path.exists() and test_path.exists()):
                continue
            teacher_conditionals_path = _teacher_conditionals_path_for_shard(shard_dir)
            train_missing_status = _missing_value_status_by_dataset(train_path)
            test_missing_status = _missing_value_status_by_dataset(test_path)

            shard_id = _extract_shard_id(shard_dir)
            for offset, size, record_sha256, record in _read_ndjson_records(catalog_path):
                if "dataset_index" not in record:
                    raise RuntimeError(
                        f"catalog record missing dataset_index: path={catalog_path}, offset={offset}"
                    )
                dataset_index = int(record["dataset_index"])
                meta = record.get("metadata")
                legacy_metadata = cast(dict[str, Any], meta) if isinstance(meta, dict) else None
                discovered_records += 1
                if dagzoo_generated_identity is not None:
                    dagzoo_generated_identity.add_record(
                        record,
                        record_path=catalog_path,
                        dataset_index=dataset_index,
                    )

                source_shard_relpath = _shard_relpath(root, shard_dir)
                if legacy_metadata is not None:
                    filter_mode, filter_status, filter_accepted = _parse_filter_metadata(legacy_metadata)
                elif root_kind == "curated":
                    filter_mode, filter_status, filter_accepted = ("curated", "accepted", True)
                else:
                    filter_mode, filter_status, filter_accepted = (None, None, None)
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
                        "catalog dataset_index missing from packed split(s): "
                        f"shard={shard_dir}, path={catalog_path}, dataset_index={dataset_index}, "
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
                    record_payload=record,
                    root_id=source_root_id,
                    shard_relpath=source_shard_relpath,
                    dataset_index=dataset_index,
                )
                split = _stable_split(dataset_identity_key, train_ratio, val_ratio)
                teacher_summary = (
                    record.get("teacher_conditionals")
                    if isinstance(record.get("teacher_conditionals"), dict)
                    else None
                )
                teacher_conditionals_manifest_path = (
                    _manifest_relative_path(teacher_conditionals_path, manifest_dir=manifest_dir)
                    if teacher_conditionals_path is not None
                    and isinstance(teacher_summary, dict)
                    and teacher_summary.get("available") is True
                    else None
                )
                n_classes_raw = (
                    record.get("n_classes")
                    if "n_classes" in record
                    else (legacy_metadata.get("n_classes") if legacy_metadata is not None else None)
                )

                records.append(
                    {
                        "dataset_id": dsid,
                        "dataset_identity_key": dataset_identity_key,
                        "source_root_id": source_root_id,
                        "source_shard_relpath": source_shard_relpath,
                        "split": split,
                        "task": _infer_task(record),
                        "shard_id": shard_id,
                        "dataset_index": dataset_index,
                        "train_path": _manifest_relative_path(train_path, manifest_dir=manifest_dir),
                        "test_path": _manifest_relative_path(test_path, manifest_dir=manifest_dir),
                        "catalog_path": _manifest_relative_path(catalog_path, manifest_dir=manifest_dir),
                        "catalog_offset_bytes": offset,
                        "catalog_size_bytes": size,
                        "catalog_sha256": record_sha256,
                        "teacher_conditionals_path": teacher_conditionals_manifest_path,
                        "n_train": int(record.get("n_train", -1)),
                        "n_test": int(record.get("n_test", -1)),
                        "n_features": _coerce_optional_int(
                            record.get(
                                "n_features",
                                legacy_metadata.get("n_features", -1)
                                if legacy_metadata is not None
                                else -1,
                            ),
                            default=-1,
                            context=(
                                f"catalog.n_features path={catalog_path} "
                                f"dataset_index={dataset_index}"
                            ),
                        ),
                        "n_classes": int(n_classes_raw) if n_classes_raw is not None else None,
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
    dagzoo_handoff_summary = (
        None
        if dagzoo_handoff is None
        else _verified_dagzoo_handoff_summary(
            handoff=dagzoo_handoff,
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
        dagzoo_handoff=dagzoo_handoff_summary,
    )
    table = pa.Table.from_pylist(records).replace_schema_metadata(
        _manifest_schema_metadata(summary=summary)
    )
    pq.write_table(table, out_path, compression="zstd")
    return summary


def _verified_dagzoo_handoff_summary(
    *,
    handoff: DagzooHandoffInfo,
    dagzoo_generated_identity: DagzooGeneratedIdentityAccumulator | None,
) -> dict[str, Any]:
    if dagzoo_generated_identity is None:
        raise RuntimeError("dagzoo handoff verification state was not initialized")
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


def manifest_sha256(manifest_path: Path) -> str:
    """Return the SHA-256 digest of one manifest parquet file."""

    resolved_manifest = manifest_path.expanduser().resolve()
    digest = sha256()
    with resolved_manifest.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest_contract_payload(manifest_path: Path) -> dict[str, Any] | None:
    metadata = pq.ParquetFile(manifest_path).schema_arrow.metadata or {}
    raw_contract = metadata.get(MANIFEST_CONTRACT_METADATA_KEY)
    if raw_contract is None:
        return None
    payload = json.loads(raw_contract.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"persisted manifest contract must be an object: {manifest_path}")
    return cast(dict[str, Any], payload)


def _require_manifest_contract(manifest_path: Path) -> dict[str, Any]:
    payload = _read_manifest_contract_payload(manifest_path)
    if payload is None:
        raise RuntimeError(
            "manifest contract metadata is missing; regenerate the manifest with tab-realdata-hub "
            f"before loading it: {manifest_path}"
        )
    raw_version = payload.get("version")
    if not isinstance(raw_version, int):
        raise RuntimeError(f"manifest contract version must be an int: {manifest_path}")
    if int(raw_version) not in SUPPORTED_MANIFEST_CONTRACT_VERSIONS:
        raise RuntimeError(
            "manifest contract version mismatch: "
            f"supported={SUPPORTED_MANIFEST_CONTRACT_VERSIONS}, actual={raw_version}, path={manifest_path}"
        )
    return payload


def _resolve_record_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _packed_x_to_matrix(x_column: Any) -> np.ndarray:
    rows = x_column.to_numpy(zero_copy_only=False)
    if rows.size == 0:
        raise RuntimeError("packed split has zero rows")
    try:
        x = np.vstack(rows).astype(np.float32, copy=False)
    except ValueError as exc:
        raise RuntimeError("packed x column has ragged row lengths") from exc
    if x.ndim != 2:
        raise RuntimeError(f"packed x column did not decode to rank-2 matrix, got shape={x.shape}")
    return x


def _read_packed_split(
    split_path: Path,
    *,
    dataset_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        table = pq.read_table(
            split_path,
            filters=[("dataset_index", "=", int(dataset_index))],
            columns=["row_index", "x", "y"],
        )
    except Exception as exc:  # pragma: no cover - pyarrow error typing is backend-specific
        raise RuntimeError(
            f"failed to read packed split parquet path={split_path}, dataset_index={dataset_index}"
        ) from exc
    if table.num_rows <= 0:
        raise RuntimeError(
            f"packed split has zero rows for dataset_index={dataset_index}: path={split_path}"
        )
    row_index = table["row_index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    x = _packed_x_to_matrix(table["x"])
    y = table["y"].to_numpy(zero_copy_only=False)
    if row_index.shape[0] != x.shape[0] or row_index.shape[0] != y.shape[0]:
        raise RuntimeError(
            "packed split row count mismatch: "
            f"path={split_path}, dataset_index={dataset_index}, "
            f"row_index={row_index.shape[0]}, x={x.shape[0]}, y={y.shape[0]}"
        )
    order = np.argsort(row_index, kind="stable")
    if not np.array_equal(order, np.arange(order.shape[0])):
        row_index = row_index[order]
        x = x[order]
        y = y[order]
    unique = np.unique(row_index)
    if unique.shape[0] != row_index.shape[0]:
        raise RuntimeError(
            f"packed split row_index values must be unique: path={split_path}, dataset_index={dataset_index}"
        )
    return row_index, x, y


def _read_ndjson_record_by_offset(
    ndjson_path: Path,
    *,
    offset_bytes: int,
    size_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    with ndjson_path.open("rb") as handle:
        handle.seek(offset_bytes)
        raw = handle.read(size_bytes)
    if len(raw) != size_bytes:
        raise RuntimeError(
            "failed to read full NDJSON slice: "
            f"path={ndjson_path}, offset={offset_bytes}, size={size_bytes}, got={len(raw)}"
        )
    actual_sha256 = sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "NDJSON checksum mismatch: "
            f"path={ndjson_path}, offset={offset_bytes}, size={size_bytes}, "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parse context
        raise RuntimeError(
            "failed to parse NDJSON record: "
            f"path={ndjson_path}, offset={offset_bytes}, size={size_bytes}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"NDJSON payload must be an object: path={ndjson_path}, offset={offset_bytes}"
        )
    return payload


def _manifest_row_catalog_locator(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    if "catalog_path" in row:
        return (
            str(row["catalog_path"]),
            int(row["catalog_offset_bytes"]),
            int(row["catalog_size_bytes"]),
            str(row["catalog_sha256"]),
        )
    return (
        str(row["metadata_path"]),
        int(row["metadata_offset_bytes"]),
        int(row["metadata_size_bytes"]),
        str(row["metadata_sha256"]),
    )


def load_manifest_record_catalog(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    raw_path, offset_bytes, size_bytes, expected_sha256 = _manifest_row_catalog_locator(record)
    catalog_path = _resolve_record_path(manifest_path, raw_path)
    return _read_ndjson_record_by_offset(
        catalog_path,
        offset_bytes=offset_bytes,
        size_bytes=size_bytes,
        expected_sha256=expected_sha256,
    )


def _teacher_probs_to_matrix(values: list[list[float]]) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def load_manifest_record_teacher_conditionals(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
) -> np.ndarray | None:
    raw_path = record.get("teacher_conditionals_path")
    if raw_path is None:
        return None
    teacher_path = _resolve_record_path(manifest_path, str(raw_path))
    dataset_index = int(record["dataset_index"])
    try:
        table = pq.read_table(
            teacher_path,
            filters=[("dataset_index", "=", dataset_index)],
            columns=["row_index", "class_probs"],
        )
    except Exception as exc:  # pragma: no cover - pyarrow error typing is backend-specific
        raise RuntimeError(
            "failed to read teacher_conditionals parquet: "
            f"path={teacher_path}, dataset_index={dataset_index}"
        ) from exc
    if table.num_rows <= 0:
        return np.empty((0, 0), dtype=np.float32)
    row_index = table["row_index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    order = np.argsort(row_index, kind="stable")
    probs = _teacher_probs_to_matrix(table["class_probs"].to_pylist())
    if not np.array_equal(order, np.arange(order.shape[0])):
        probs = probs[order]
    return probs


def _dataset_display_name(*, dataset_id: str, metadata: dict[str, Any]) -> str:
    openml_payload = metadata.get("openml")
    if isinstance(openml_payload, dict):
        dataset_name = openml_payload.get("dataset_name")
        if isinstance(dataset_name, str) and dataset_name.strip():
            return str(dataset_name)
    observed_task = metadata.get("observed_task")
    if isinstance(observed_task, dict):
        dataset_name = observed_task.get("dataset_name")
        if isinstance(dataset_name, str) and dataset_name.strip():
            return str(dataset_name)
    return str(dataset_id)


def _combine_dataset_splits(
    *,
    train_row_index: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    test_row_index: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    combined_row_index = np.concatenate([train_row_index, test_row_index], axis=0)
    combined_x = np.concatenate([x_train, x_test], axis=0)
    combined_y = np.concatenate([y_train, y_test], axis=0)
    if np.unique(combined_row_index).shape[0] == combined_row_index.shape[0]:
        order = np.argsort(combined_row_index, kind="stable")
        return combined_x[order], combined_y[order], "global_row_index"
    return combined_x, combined_y, "split_concat"


def load_manifest_datasets(
    manifest_path: Path,
    *,
    allow_missing_values: bool = False,
    expected_task: str | None = None,
) -> LoadedManifestDatasets:
    """Load manifest-backed datasets for benchmark/helper execution."""

    resolved_manifest = manifest_path.expanduser().resolve()
    if not resolved_manifest.exists():
        raise RuntimeError(f"manifest does not exist: {resolved_manifest}")
    contract = _require_manifest_contract(resolved_manifest)
    table = pq.read_table(resolved_manifest)
    rows = cast(list[dict[str, Any]], table.to_pylist())
    if not rows:
        raise RuntimeError(f"manifest has zero rows: {resolved_manifest}")

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    task_records: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task", "unknown"))
        if expected_task is not None and task != str(expected_task):
            raise RuntimeError(
                f"manifest row task mismatch: expected={expected_task!r}, actual={task!r}, path={resolved_manifest}"
            )
        dataset_index = int(row["dataset_index"])
        train_path = _resolve_record_path(resolved_manifest, str(row["train_path"]))
        test_path = _resolve_record_path(resolved_manifest, str(row["test_path"]))
        catalog_record = load_manifest_record_catalog(resolved_manifest, record=row)
        raw_metadata = catalog_record.get("metadata")
        metadata = (
            cast(dict[str, Any], raw_metadata)
            if isinstance(raw_metadata, dict)
            else cast(dict[str, Any], dict(catalog_record))
        )
        train_row_index, x_train, y_train = _read_packed_split(train_path, dataset_index=dataset_index)
        test_row_index, x_test, y_test = _read_packed_split(test_path, dataset_index=dataset_index)
        x, y, row_order_mode = _combine_dataset_splits(
            train_row_index=train_row_index,
            x_train=x_train,
            y_train=np.asarray(y_train),
            test_row_index=test_row_index,
            x_test=x_test,
            y_test=np.asarray(y_test),
        )
        dataset_id = str(row["dataset_id"])
        dataset_name = _dataset_display_name(dataset_id=dataset_id, metadata=metadata)
        if dataset_name in datasets:
            raise RuntimeError(
                f"duplicate dataset name in manifest-backed load result: {dataset_name!r}, path={resolved_manifest}"
            )
        if not allow_missing_values:
            assert_no_non_finite_values(
                {"x": x, "y": y},
                context=f"manifest dataset {dataset_name!r}",
            )
        datasets[dataset_name] = (np.asarray(x, dtype=np.float32), np.asarray(y))
        task_record = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "task": task,
            "n_rows": int(x.shape[0]),
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
            "n_features": int(row["n_features"]),
            "n_classes": None if row.get("n_classes") is None else int(row["n_classes"]),
            "row_order_mode": row_order_mode,
            "metadata": metadata,
            "manifest_record": dict(row),
        }
        task_records.append(task_record)
    return LoadedManifestDatasets(
        manifest_path=resolved_manifest,
        contract_version=int(contract["version"]),
        manifest_sha256=manifest_sha256(resolved_manifest),
        datasets=datasets,
        task_records=tuple(task_records),
        persisted_summary=_read_persisted_manifest_summary(resolved_manifest),
    )


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
        "manifest_sha256": manifest_sha256(resolved_manifest),
        "manifest_contract": _read_manifest_contract_payload(resolved_manifest),
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
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": manifest_sha256(resolved_manifest),
        "manifest_contract": _read_manifest_contract_payload(resolved_manifest),
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
