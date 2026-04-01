"""Helpers for consuming dagzoo handoff manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2s, sha256
import json
from pathlib import Path
from typing import Any, Mapping, cast


DAGZOO_HANDOFF_SCHEMA_NAME = "dagzoo_generate_handoff_manifest"
SUPPORTED_DAGZOO_HANDOFF_SCHEMA_VERSIONS = (1, 2, 3)
DAGZOO_HANDOFF_SCHEMA_VERSION = 3
_DAGZOO_ID_HEX_LENGTH = 32
_GENERATED_CORPUS_ID_DIGEST_BYTES = 16


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True, frozen=True)
class DagzooHandoffInfo:
    """Validated subset of one dagzoo handoff manifest."""

    handoff_manifest_path: Path
    handoff_manifest_sha256: str
    source_family: str
    generate_run_id: str
    generated_corpus_id: str
    generated_dir: Path
    curated_dir: Path | None = None
    provenance: dict[str, Any] | None = None
    teacher_conditionals: dict[str, Any] | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        payload = {
            "handoff_manifest_path": str(self.handoff_manifest_path),
            "handoff_manifest_sha256": self.handoff_manifest_sha256,
            "source_family": self.source_family,
            "generate_run_id": self.generate_run_id,
            "generated_corpus_id": self.generated_corpus_id,
            "generated_dir": str(self.generated_dir),
            "curated_dir": None if self.curated_dir is None else str(self.curated_dir),
        }
        if self.provenance is not None:
            payload["provenance"] = dict(self.provenance)
        if self.teacher_conditionals is not None:
            payload["teacher_conditionals"] = dict(self.teacher_conditionals)
        return payload


@dataclass(slots=True)
class DagzooGeneratedIdentityAccumulator:
    """Scan-time dagzoo identity derived from shard catalog records."""

    generate_run_id: str | None = None
    dataset_ids: list[str] = field(default_factory=list)

    def add_record(
        self,
        record: Mapping[str, Any],
        *,
        record_path: Path,
        dataset_index: int,
    ) -> None:
        if isinstance(record.get("metadata"), Mapping):
            metadata = cast(Mapping[str, Any], record["metadata"])
            group_payload = metadata.get("split_groups")
            dataset_id_value = metadata.get("dataset_id")
            request_run_context = "metadata.split_groups.request_run"
            dataset_id_context = "metadata.dataset_id"
        else:
            metadata = record
            group_payload = record.get("group_ids")
            dataset_id_value = record.get("dataset_id")
            request_run_context = "group_ids.request_run"
            dataset_id_context = "dataset_id"

        if not isinstance(group_payload, Mapping):
            raise RuntimeError(
                "dagzoo dataset catalog missing object payload for grouping keys: "
                f"path={record_path}, dataset_index={dataset_index}"
            )
        current_generate_run_id = _require_hex_string_value(
            group_payload.get("request_run"),
            context=(
                "dagzoo dataset identity field must be a "
                f"{_DAGZOO_ID_HEX_LENGTH}-character lowercase hex string: "
                f"path={record_path}, dataset_index={dataset_index}, key={request_run_context}"
            ),
        )
        dataset_id = _require_hex_string_value(
            dataset_id_value,
            context=(
                "dagzoo dataset identity field must be a "
                f"{_DAGZOO_ID_HEX_LENGTH}-character lowercase hex string: "
                f"path={record_path}, dataset_index={dataset_index}, key={dataset_id_context}"
            ),
        )
        if self.generate_run_id is None:
            self.generate_run_id = current_generate_run_id
        elif self.generate_run_id != current_generate_run_id:
            raise RuntimeError(
                "dagzoo generated corpus contains multiple request_run identities: "
                f"path={record_path}, dataset_index={dataset_index}, "
                f"expected={self.generate_run_id!r}, found={current_generate_run_id!r}"
            )
        self.dataset_ids.append(dataset_id)

    def add_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        metadata_path: Path,
        dataset_index: int,
    ) -> None:
        self.add_record(
            {"metadata": dict(metadata)},
            record_path=metadata_path,
            dataset_index=dataset_index,
        )

    def generated_corpus_id(self) -> str:
        if self.generate_run_id is None or not self.dataset_ids:
            raise RuntimeError(
                "dagzoo handoff verification requires at least one scanned dagzoo dataset record"
            )
        return stable_dagzoo_generated_corpus_id(
            generate_run_id=self.generate_run_id,
            dataset_ids=self.dataset_ids,
        )


def is_canonical_dagzoo_id(value: Any) -> bool:
    """Return whether a value matches the canonical dagzoo 32-char hex id shape."""

    return (
        isinstance(value, str)
        and len(value) == _DAGZOO_ID_HEX_LENGTH
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _read_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"dagzoo handoff manifest must contain a JSON object: path={path}")
    return cast(dict[str, Any], payload)


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"dagzoo handoff manifest field must be an object: path={path}, key={key}")
    return cast(Mapping[str, Any], value)


def _require_optional_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> Mapping[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"dagzoo handoff manifest field must be an object: path={path}, key={key}")
    return cast(Mapping[str, Any], value)


def _require_non_empty_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"dagzoo handoff manifest field must be a non-empty string: path={path}, key={key}"
        )
    return value


def _require_hex_string_value(value: Any, *, context: str) -> str:
    if not is_canonical_dagzoo_id(value):
        raise RuntimeError(context)
    return value


def _resolve_relative_path(raw: str, *, path: Path, field_key: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute():
        raise RuntimeError(
            "dagzoo handoff manifest field must be relative: "
            f"path={path}, key={field_key}, value={raw!r}"
        )
    resolved = (path.parent / relative).resolve()
    try:
        _ = resolved.relative_to(path.parent)
    except ValueError as exc:
        raise RuntimeError(
            "dagzoo handoff path escapes the handoff root: "
            f"path={path}, key={field_key}, value={raw!r}"
        ) from exc
    return resolved


def stable_dagzoo_generated_corpus_id(*, generate_run_id: str, dataset_ids: list[str]) -> str:
    payload = {
        "generate_run_id": str(generate_run_id),
        "dataset_ids": list(dataset_ids),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return blake2s(encoded, digest_size=_GENERATED_CORPUS_ID_DIGEST_BYTES).hexdigest()


def verify_dagzoo_handoff_matches_generated_corpus(
    handoff: DagzooHandoffInfo,
    *,
    scanned_identity: DagzooGeneratedIdentityAccumulator,
) -> None:
    if scanned_identity.generate_run_id is None or not scanned_identity.dataset_ids:
        raise RuntimeError(
            "dagzoo handoff verification requires at least one scanned dagzoo dataset record"
        )
    if handoff.generate_run_id != scanned_identity.generate_run_id:
        raise RuntimeError(
            "dagzoo handoff generate_run_id does not match scanned corpus metadata: "
            f"handoff={handoff.generate_run_id!r}, scanned={scanned_identity.generate_run_id!r}, "
            f"path={handoff.handoff_manifest_path}"
        )
    scanned_generated_corpus_id = scanned_identity.generated_corpus_id()
    if handoff.generated_corpus_id != scanned_generated_corpus_id:
        raise RuntimeError(
            "dagzoo handoff generated_corpus_id does not match scanned corpus metadata: "
            f"handoff={handoff.generated_corpus_id!r}, scanned={scanned_generated_corpus_id!r}, "
            f"path={handoff.handoff_manifest_path}"
        )


def _teacher_summary_from_v1(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    provenance = _require_optional_mapping(payload, "provenance", path=path)
    if provenance is None:
        return None
    enabled = provenance.get("teacher_conditional_export")
    if enabled is not True:
        return None
    metric_definition = provenance.get("teacher_conditional_metric_definition")
    target_split = provenance.get("target_split")
    if not isinstance(metric_definition, str) or not metric_definition.strip():
        return None
    if not isinstance(target_split, str) or not target_split.strip():
        target_split = "test"
    return {
        "enabled": True,
        "metric_definition": metric_definition,
        "target_split": target_split,
    }


def _teacher_summary_from_v2(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    teacher_conditionals = _require_optional_mapping(payload, "teacher_conditionals", path=path)
    if teacher_conditionals is None:
        return None
    enabled = teacher_conditionals.get("enabled")
    if enabled is not True:
        raise RuntimeError(
            "dagzoo handoff teacher_conditionals.enabled must equal true when present: "
            f"path={path}"
        )
    metric_definition = _require_non_empty_string(
        teacher_conditionals,
        "metric_definition",
        path=path,
    )
    target_split = _require_non_empty_string(
        teacher_conditionals,
        "target_split",
        path=path,
    )
    return {
        "enabled": True,
        "metric_definition": metric_definition,
        "target_split": target_split,
    }


def _normalized_range_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> dict[str, Any] | None:
    range_payload = _require_optional_mapping(payload, key, path=path)
    if range_payload is None:
        return None
    normalized: dict[str, Any] = {}
    minimum = range_payload.get("min")
    maximum = range_payload.get("max")
    if minimum is not None:
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise RuntimeError(
                "dagzoo handoff range bound must be numeric when present: "
                f"path={path}, key={key}.min"
            )
        normalized["min"] = float(minimum) if isinstance(minimum, float) else int(minimum)
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
            raise RuntimeError(
                "dagzoo handoff range bound must be numeric when present: "
                f"path={path}, key={key}.max"
            )
        normalized["max"] = float(maximum) if isinstance(maximum, float) else int(maximum)
    return normalized or None


def _provenance_summary_from_v3(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    provenance = _require_optional_mapping(payload, "provenance", path=path)
    if provenance is None:
        return None
    summary: dict[str, Any] = {}
    target_derivation = provenance.get("target_derivation")
    if target_derivation is not None:
        if not isinstance(target_derivation, str) or not target_derivation.strip():
            raise RuntimeError(
                "dagzoo handoff provenance.target_derivation must be a non-empty string "
                f"when present: path={path}"
            )
        summary["target_derivation"] = target_derivation
    for key in (
        "target_relevant_feature_count_range",
        "target_relevant_feature_fraction_range",
    ):
        normalized = _normalized_range_mapping(provenance, key, path=path)
        if normalized is not None:
            summary[key] = normalized
    return summary or None


def load_dagzoo_handoff_info(handoff_manifest_path: Path) -> DagzooHandoffInfo:
    """Load and validate the dagzoo handoff subset used by manifest builders."""

    path = handoff_manifest_path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"dagzoo handoff manifest not found: {path}")
    payload = _read_json_dict(path)

    schema_name = _require_non_empty_string(payload, "schema_name", path=path)
    if schema_name != DAGZOO_HANDOFF_SCHEMA_NAME:
        raise RuntimeError(
            "Unsupported dagzoo handoff schema_name: "
            f"path={path}, value={schema_name!r}, expected={DAGZOO_HANDOFF_SCHEMA_NAME!r}"
        )
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_DAGZOO_HANDOFF_SCHEMA_VERSIONS:
        raise RuntimeError(
            "Unsupported dagzoo handoff schema_version: "
            f"path={path}, value={schema_version!r}, expected one of "
            f"{SUPPORTED_DAGZOO_HANDOFF_SCHEMA_VERSIONS}"
        )

    identity = _require_mapping(payload, "identity", path=path)
    source_family = _require_non_empty_string(identity, "source_family", path=path)
    generate_run_id = _require_non_empty_string(identity, "generate_run_id", path=path)
    generated_corpus_id = _require_non_empty_string(identity, "generated_corpus_id", path=path)

    artifacts_relative = _require_mapping(payload, "artifacts_relative", path=path)
    if int(schema_version) == 1:
        run_root = _require_non_empty_string(artifacts_relative, "run_root", path=path)
        if run_root != ".":
            raise RuntimeError(
                "dagzoo handoff artifacts_relative.run_root must equal '.': "
                f"path={path}, value={run_root!r}"
            )
    generated_dir_rel = _require_non_empty_string(artifacts_relative, "generated_dir", path=path)
    generated_dir = _resolve_relative_path(
        generated_dir_rel,
        path=path,
        field_key="artifacts_relative.generated_dir",
    )
    curated_dir_raw = artifacts_relative.get("curated_dir")
    curated_dir = None
    if curated_dir_raw is not None:
        if not isinstance(curated_dir_raw, str) or not curated_dir_raw.strip():
            raise RuntimeError(
                "dagzoo handoff manifest field must be a non-empty string when present: "
                f"path={path}, key=artifacts_relative.curated_dir"
            )
        curated_dir = _resolve_relative_path(
            curated_dir_raw,
            path=path,
            field_key="artifacts_relative.curated_dir",
        )

    teacher_conditionals = (
        _teacher_summary_from_v1(payload, path=path)
        if int(schema_version) == 1
        else (_teacher_summary_from_v2(payload, path=path) if int(schema_version) == 2 else None)
    )
    provenance = _provenance_summary_from_v3(payload, path=path) if int(schema_version) >= 3 else None

    return DagzooHandoffInfo(
        handoff_manifest_path=path,
        handoff_manifest_sha256=_sha256_path(path),
        source_family=source_family,
        generate_run_id=generate_run_id,
        generated_corpus_id=generated_corpus_id,
        generated_dir=generated_dir,
        curated_dir=curated_dir,
        provenance=provenance,
        teacher_conditionals=teacher_conditionals,
    )
