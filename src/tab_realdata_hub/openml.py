"""OpenML bundle building and manifest materialization helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Protocol, cast

import numpy as np
import openml
from openml.tasks import TaskType
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

from ._json import write_json
from .manifest import build_manifest


_CLASSIFICATION_TASK_TYPE = "supervised_classification"
_REGRESSION_TASK_TYPE = "supervised_regression"
_ALLOWED_BUNDLE_SELECTION_TASK_TYPES = {
    _CLASSIFICATION_TASK_TYPE,
    _REGRESSION_TASK_TYPE,
}
_OPENML_TASK_LISTING_MAX_INSTANCES = 1_000_000_000
DEFAULT_OPENML_TASK_SOURCE = "tabarena_v0_1"
DEFAULT_COMPARATOR_SPLIT_SEED = 0
DEFAULT_COMPARATOR_TEST_SIZE = 0.20

TABARENA_V0_1_TASK_IDS: tuple[int, ...] = (
    363612,
    363613,
    363614,
    363615,
    363616,
    363618,
    363619,
    363620,
    363621,
    363623,
    363624,
    363625,
    363626,
    363627,
    363628,
    363629,
    363630,
    363631,
    363632,
    363671,
    363672,
    363673,
    363674,
    363675,
    363676,
    363677,
    363678,
    363679,
    363681,
    363682,
    363683,
    363684,
    363685,
    363686,
    363689,
    363691,
    363693,
    363694,
    363696,
    363697,
    363698,
    363699,
    363700,
    363702,
    363704,
    363705,
    363706,
    363707,
    363708,
    363711,
    363712,
)
BINARY_EXPANDED_V1_TASK_IDS: tuple[int, ...] = (
    42,
    3777,
    10091,
    10093,
    3638,
    9958,
    146230,
    363613,
    363621,
    363629,
)
BINARY_LARGE_NO_MISSING_V1_TASK_IDS: tuple[int, ...] = (
    3,
    31,
    37,
    42,
    49,
    52,
    57,
    134,
    135,
    139,
    146,
    147,
    148,
    150,
    152,
    154,
    157,
    206,
    208,
    209,
    211,
    212,
    215,
    219,
    220,
    221,
    229,
    230,
    2137,
    2142,
    2147,
    2148,
    2253,
    2255,
    2257,
    2262,
    2264,
    3484,
    3492,
    3493,
    3494,
    3495,
    3496,
    3539,
    3542,
    3555,
    3581,
    3583,
    3586,
    3587,
    3588,
    3589,
    3590,
    3591,
    3593,
    3594,
    3596,
    3599,
    3600,
    3601,
    3603,
    3606,
    3607,
    3609,
)
OPENML_TASK_SOURCE_REGISTRY: dict[str, tuple[int, ...]] = {
    "tabarena_v0_1": TABARENA_V0_1_TASK_IDS,
    "binary_expanded_v1": BINARY_EXPANDED_V1_TASK_IDS,
    "binary_large_no_missing_v1": BINARY_LARGE_NO_MISSING_V1_TASK_IDS,
}


@dataclass(slots=True)
class PreparedOpenMLTask:
    """Materialized OpenML task after notebook-style preprocessing."""

    task_id: int
    dataset_name: str
    x: np.ndarray
    y: np.ndarray
    observed_task: dict[str, Any]
    qualities: dict[str, float]
    task_type: str = _CLASSIFICATION_TASK_TYPE


@dataclass(slots=True, frozen=True)
class OpenMLBundleConfig:
    """Configuration for generating a pinned OpenML bundle."""

    bundle_name: str
    version: int
    task_source: str = DEFAULT_OPENML_TASK_SOURCE
    task_type: str = _CLASSIFICATION_TASK_TYPE
    new_instances: int = 200
    max_features: int = 10
    max_classes: int | None = 2
    max_missing_pct: float = 0.0
    min_minority_class_pct: float = 2.5
    task_ids: tuple[int, ...] | None = None
    discover_from_openml: bool = False
    min_instances: int = 1
    min_task_count: int = 1

    def resolved_task_ids(self) -> tuple[int, ...]:
        """Resolve custom task ids or fall back to the named pinned source pool."""

        if self.discover_from_openml:
            raise RuntimeError("OpenML discovery mode does not use pinned task ids")
        if self.task_ids is not None:
            return tuple(int(task_id) for task_id in self.task_ids)
        return task_ids_for_source(self.task_source)


@dataclass(slots=True, frozen=True)
class OpenMLTaskCandidate:
    """OpenML task metadata used for bundle filtering."""

    task_id: int
    number_of_features: float
    number_of_classes: float | None
    missing_pct: float
    minority_class_pct: float | None
    dataset_id: int | None = None
    dataset_name: str | None = None
    estimation_procedure: str | None = None
    number_of_instances: float | None = None


@dataclass(slots=True, frozen=True)
class OpenMLCandidateReportEntry:
    """One task-candidate decision recorded during discovery."""

    task_id: int
    status: str
    reason: str
    dataset_id: int | None = None
    dataset_name: str | None = None
    estimation_procedure: str | None = None


@dataclass(slots=True, frozen=True)
class OpenMLBundleBuildResult:
    """Final bundle payload and optional discovery report entries."""

    bundle: dict[str, Any]
    report_entries: tuple[OpenMLCandidateReportEntry, ...] = ()


@dataclass(slots=True, frozen=True)
class OpenMLMaterializationResult:
    """Materialized OpenML bundle surface."""

    bundle_summary: dict[str, Any]
    task_summaries: tuple[dict[str, Any], ...]
    allow_missing_values: bool
    data_root: Path
    manifest_path: Path


class _PreparedTaskProvider(Protocol):
    def __call__(self, task_id: int, *, new_instances: int, task_type: str) -> PreparedOpenMLTask: ...


def task_source_names() -> tuple[str, ...]:
    """Return stable CLI-facing task-source names."""

    return tuple(OPENML_TASK_SOURCE_REGISTRY.keys())


def task_ids_for_source(task_source: str) -> tuple[int, ...]:
    """Resolve one named task source into its pinned OpenML task ids."""

    try:
        return OPENML_TASK_SOURCE_REGISTRY[str(task_source)]
    except KeyError as exc:
        choices = ", ".join(repr(name) for name in task_source_names())
        raise ValueError(f"unknown OpenML task source {task_source!r}; expected one of: {choices}") from exc


def parse_max_classes_arg(raw_value: str) -> int | None:
    normalized = str(raw_value).strip().lower()
    if normalized == "auto":
        return None
    value = int(normalized)
    if value <= 0:
        raise ValueError("max_classes must be a positive int or 'auto'")
    return value


def task_type_value(task_type: TaskType | int) -> int:
    return int(task_type.value) if isinstance(task_type, TaskType) else int(task_type)


def get_feature_preprocessor(x: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """Replicate the nanoTabPFN notebook preprocessing logic."""

    frame = pd.DataFrame(x)
    num_mask: list[bool] = []
    cat_mask: list[bool] = []
    for column in frame:
        unique_non_nan_entries = frame[column].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = frame[column].notna().sum()
        numeric_entries = pd.to_numeric(frame[column], errors="coerce").notna().sum()
        num_mask.append(bool(non_nan_entries == numeric_entries))
        cat_mask.append(bool(non_nan_entries != numeric_entries))

    num_transformer = Pipeline(
        [
            (
                "to_pandas",
                FunctionTransformer(
                    lambda value: pd.DataFrame(value) if not isinstance(value, pd.DataFrame) else value
                ),
            ),
            (
                "to_numeric",
                FunctionTransformer(lambda value: value.apply(pd.to_numeric, errors="coerce").to_numpy()),
            ),
        ]
    )
    cat_transformer = Pipeline(
        [("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, np.asarray(num_mask)),
            ("cat", cat_transformer, np.asarray(cat_mask)),
        ]
    )


def read_required_quality(raw_qualities: Any, *, task_id: int, quality_name: str) -> float:
    """Read a numeric OpenML quality and raise a drift error if it is missing."""

    if not isinstance(raw_qualities, dict):
        raise RuntimeError(f"benchmark bundle drift: task {task_id} dataset qualities are missing")
    value = raw_qualities.get(quality_name)
    if not isinstance(value, (int, float)):
        raise RuntimeError(
            f"benchmark bundle drift: task {task_id} missing numeric quality {quality_name!r}"
        )
    return float(value)


def _openml_task_type_for_bundle_task_type(task_type: str) -> int:
    if task_type == _CLASSIFICATION_TASK_TYPE:
        return task_type_value(TaskType.SUPERVISED_CLASSIFICATION)
    if task_type == _REGRESSION_TASK_TYPE:
        return task_type_value(TaskType.SUPERVISED_REGRESSION)
    raise RuntimeError(f"unsupported benchmark bundle task_type: {task_type!r}")


def prepare_task(
    task_id: int,
    *,
    new_instances: int,
    task_type: str,
    get_task_fn: Any | None = None,
) -> PreparedOpenMLTask:
    """Load and preprocess one OpenML task using the notebook logic."""

    task = (openml.tasks.get_task if get_task_fn is None else get_task_fn)(
        task_id,
        download_splits=False,
    )
    expected_task_type_id = _openml_task_type_for_bundle_task_type(task_type)
    observed_task_type_id = task_type_value(cast(TaskType | int, task.task_type_id))
    if observed_task_type_id != expected_task_type_id:
        raise RuntimeError(f"benchmark bundle drift: task {task_id} is no longer {task_type}")
    task_any: Any = task
    dataset = task_any.get_dataset(download_data=False)
    dataset_any: Any = dataset
    raw_qualities = dataset_any.qualities
    number_of_features = read_required_quality(
        raw_qualities,
        task_id=int(task_id),
        quality_name="NumberOfFeatures",
    )
    missing_pct = read_required_quality(
        raw_qualities,
        task_id=int(task_id),
        quality_name="PercentageOfInstancesWithMissingValues",
    )
    number_of_classes = None
    minority_class_pct = None
    if task_type == _CLASSIFICATION_TASK_TYPE:
        number_of_classes = read_required_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="NumberOfClasses",
        )
        minority_class_pct = read_required_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="MinorityClassPercentage",
        )

    x_frame, y_raw, _categorical_indicator, _attribute_names = dataset_any.get_data(
        target=str(task_any.target_name),
        dataset_format="dataframe",
    )
    if new_instances < int(len(cast(Any, y_raw))):
        train_test_split_kwargs: dict[str, Any] = {
            "test_size": new_instances,
            "random_state": 0,
        }
        if task_type == _CLASSIFICATION_TASK_TYPE:
            train_test_split_kwargs["stratify"] = y_raw
        _x_unused, x_sub, _y_unused, y_sub = train_test_split(x_frame, y_raw, **train_test_split_kwargs)
    else:
        x_sub = x_frame
        y_sub = y_raw

    preprocessor = get_feature_preprocessor(x_sub)
    x = np.asarray(preprocessor.fit_transform(x_sub), dtype=np.float32)
    observed_task = {
        "task_id": int(task_id),
        "dataset_name": str(dataset.name),
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }
    if task_type == _CLASSIFICATION_TASK_TYPE:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_sub.to_numpy(copy=True)).astype(np.int64, copy=False)
        observed_task["n_classes"] = int(np.unique(y).size)
    else:
        y = np.asarray(pd.to_numeric(y_sub, errors="raise"), dtype=np.float32)
    return PreparedOpenMLTask(
        task_id=int(task_id),
        task_type=str(task_type),
        dataset_name=str(dataset.name),
        x=x,
        y=y,
        observed_task=observed_task,
        qualities={
            "NumberOfFeatures": float(number_of_features),
            "PercentageOfInstancesWithMissingValues": float(missing_pct),
            **({} if number_of_classes is None else {"NumberOfClasses": float(number_of_classes)}),
            **({} if minority_class_pct is None else {"MinorityClassPercentage": float(minority_class_pct)}),
        },
    )


def _normalize_selection(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("benchmark bundle selection must be an object")
    task_type = payload.get("task_type", _CLASSIFICATION_TASK_TYPE)
    if task_type not in _ALLOWED_BUNDLE_SELECTION_TASK_TYPES:
        raise RuntimeError(
            "benchmark bundle selection.task_type must be one of "
            f"{sorted(_ALLOWED_BUNDLE_SELECTION_TASK_TYPES)!r}"
        )
    expected_keys = (
        {
            "new_instances",
            "max_features",
            "max_classes",
            "max_missing_pct",
            "min_minority_class_pct",
        }
        | ({"task_type"} if "task_type" in payload else set())
        if task_type == _CLASSIFICATION_TASK_TYPE
        else {"new_instances", "task_type", "max_features", "max_missing_pct"}
    )
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            "benchmark bundle selection keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}"
        )

    new_instances = payload["new_instances"]
    max_features = payload["max_features"]
    max_missing_pct = payload["max_missing_pct"]
    if not isinstance(new_instances, int) or isinstance(new_instances, bool) or new_instances <= 0:
        raise RuntimeError("benchmark bundle selection.new_instances must be a positive int")
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features <= 0:
        raise RuntimeError("benchmark bundle selection.max_features must be a positive int")
    if not isinstance(max_missing_pct, (int, float)) or not 0 <= float(max_missing_pct) <= 100:
        raise RuntimeError("benchmark bundle selection.max_missing_pct must be a percentage between 0 and 100")
    normalized = {
        "new_instances": int(new_instances),
        "task_type": str(task_type),
        "max_features": int(max_features),
        "max_missing_pct": float(max_missing_pct),
    }
    if task_type == _CLASSIFICATION_TASK_TYPE:
        max_classes = payload["max_classes"]
        min_minority_class_pct = payload["min_minority_class_pct"]
        if not isinstance(max_classes, int) or isinstance(max_classes, bool) or max_classes <= 0:
            raise RuntimeError("benchmark bundle selection.max_classes must be a positive int")
        if not isinstance(min_minority_class_pct, (int, float)) or not 0 <= float(min_minority_class_pct) <= 100:
            raise RuntimeError(
                "benchmark bundle selection.min_minority_class_pct must be a percentage between 0 and 100"
            )
        normalized["max_classes"] = int(max_classes)
        normalized["min_minority_class_pct"] = float(min_minority_class_pct)
    return normalized


def normalize_bundle(payload: Any) -> dict[str, Any]:
    """Validate and normalize benchmark bundle metadata."""

    if not isinstance(payload, dict):
        raise RuntimeError("benchmark bundle must be a JSON object")
    expected_keys = {"name", "version", "selection", "task_ids", "tasks"}
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            "benchmark bundle keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}"
        )

    name = payload["name"]
    version = payload["version"]
    selection = payload["selection"]
    task_ids = payload["task_ids"]
    tasks = payload["tasks"]
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("benchmark bundle name must be a non-empty string")
    if not isinstance(version, int) or version <= 0:
        raise RuntimeError("benchmark bundle version must be a positive int")
    if not isinstance(task_ids, list) or not task_ids:
        raise RuntimeError("benchmark bundle task_ids must be a non-empty list")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("benchmark bundle tasks must be a non-empty list")

    normalized_selection = _normalize_selection(selection)
    selection_task_type = str(normalized_selection["task_type"])
    normalized_task_ids = [int(task_id) for task_id in task_ids]
    normalized_tasks: list[dict[str, Any]] = []
    for index, task_payload in enumerate(tasks):
        if not isinstance(task_payload, dict):
            raise RuntimeError(f"benchmark bundle task {index} must be an object")
        task_keys = (
            {"task_id", "dataset_name", "n_rows", "n_features", "n_classes"}
            if selection_task_type == _CLASSIFICATION_TASK_TYPE
            else {"task_id", "dataset_name", "n_rows", "n_features"}
        )
        actual_task_keys = set(task_payload.keys())
        if actual_task_keys != task_keys:
            raise RuntimeError(
                f"benchmark bundle task keys mismatch at index {index}: "
                f"expected={sorted(task_keys)}, actual={sorted(actual_task_keys)}"
            )
        dataset_name = task_payload["dataset_name"]
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise RuntimeError(f"benchmark bundle task dataset_name must be non-empty at index {index}")
        normalized_task = {
            "task_id": int(task_payload["task_id"]),
            "dataset_name": str(dataset_name),
            "n_rows": int(task_payload["n_rows"]),
            "n_features": int(task_payload["n_features"]),
        }
        if selection_task_type == _CLASSIFICATION_TASK_TYPE:
            normalized_task["n_classes"] = int(task_payload["n_classes"])
        normalized_tasks.append(normalized_task)

    if normalized_task_ids != [int(task["task_id"]) for task in normalized_tasks]:
        raise RuntimeError("benchmark bundle task_ids must match tasks[].task_id order exactly")

    return {
        "name": str(name),
        "version": int(version),
        "selection": normalized_selection,
        "task_ids": normalized_task_ids,
        "tasks": normalized_tasks,
    }


def bundle_allows_missing_values(bundle: Mapping[str, Any]) -> bool:
    selection = cast(dict[str, Any], bundle["selection"])
    raw_max_missing_pct = selection.get("max_missing_pct")
    if not isinstance(raw_max_missing_pct, (int, float)):
        return False
    return bool(float(raw_max_missing_pct) > 0.0)


def load_bundle(path: Path, *, allow_missing_values: bool = False) -> dict[str, Any]:
    bundle_path = path.expanduser().resolve()
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    bundle = normalize_bundle(payload)
    if not allow_missing_values and bundle_allows_missing_values(bundle):
        selection = cast(dict[str, Any], bundle["selection"])
        raise RuntimeError(
            "benchmark bundle permits missing-valued inputs while allow_missing_values=False: "
            f"path={bundle_path}, max_missing_pct={selection['max_missing_pct']}"
        )
    return bundle


def bundle_summary(bundle: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    task_ids = [int(task_id) for task_id in cast(list[Any], bundle["task_ids"])]
    selection_raw = bundle.get("selection")
    selection = (
        cast(dict[str, Any], json.loads(json.dumps(selection_raw, sort_keys=True)))
        if isinstance(selection_raw, Mapping)
        else None
    )
    allow_missing_values = None if not isinstance(selection_raw, Mapping) else bundle_allows_missing_values(bundle)
    return {
        "name": str(bundle["name"]),
        "version": int(bundle["version"]),
        "source_path": str(source_path.expanduser().resolve()),
        "task_count": int(len(task_ids)),
        "task_ids": task_ids,
        "selection": selection,
        "allow_missing_values": allow_missing_values,
        "all_tasks_no_missing": None if allow_missing_values is None else (not allow_missing_values),
    }


def coerce_finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be numeric") from exc
    if not math.isfinite(numeric):
        raise RuntimeError(f"{context} must be finite")
    return numeric


def lookup_task_listing_value(row: Mapping[str, Any], *names: str) -> Any:
    normalized_keys = {str(key).casefold(): key for key in row}
    for name in names:
        key = normalized_keys.get(str(name).casefold())
        if key is not None:
            return row[key]
    raise KeyError(names[0])


def task_listing_records(task_listing: Any) -> list[Mapping[str, Any]]:
    if hasattr(task_listing, "to_dict"):
        records = task_listing.to_dict(orient="records")
    elif isinstance(task_listing, dict):
        records = list(task_listing.values())
    else:
        raise RuntimeError("OpenML task listing must be a dataframe or dict of rows")
    if not isinstance(records, list):
        raise RuntimeError("OpenML task listing must resolve into a list of rows")
    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"OpenML task listing row {index} must be a mapping")
        normalized.append(record)
    return normalized


def _task_listing_rows_for_config(
    config: OpenMLBundleConfig,
    *,
    list_tasks_fn: Any,
) -> list[Mapping[str, Any]]:
    expected_task_type = (
        TaskType.SUPERVISED_CLASSIFICATION
        if config.task_type == _CLASSIFICATION_TASK_TYPE
        else TaskType.SUPERVISED_REGRESSION
    )
    listing_filters: dict[str, Any] = {}
    if int(config.min_instances) > 1:
        listing_filters["number_instances"] = (
            f"{int(config.min_instances)}..{_OPENML_TASK_LISTING_MAX_INSTANCES}"
        )
    listing_filters["number_features"] = f"0..{int(config.max_features)}"
    if config.task_type == _CLASSIFICATION_TASK_TYPE and config.max_classes is not None:
        listing_filters["number_classes"] = int(config.max_classes)
    if float(config.max_missing_pct) <= 0.0:
        listing_filters["number_missing_values"] = 0
    try:
        task_listing = list_tasks_fn(
            task_type=expected_task_type,
            output_format="dataframe",
            **listing_filters,
        )
    except Exception:
        task_listing = list_tasks_fn(task_type=expected_task_type, output_format="dataframe")
    return task_listing_records(task_listing)


def candidate_matches_listing_filters(
    candidate: OpenMLTaskCandidate,
    config: OpenMLBundleConfig,
) -> tuple[bool, str]:
    if candidate.number_of_instances is not None and candidate.number_of_instances < float(config.min_instances):
        return False, (
            f"number_of_instances={candidate.number_of_instances:g} below min_instances={config.min_instances}"
        )
    if candidate.number_of_features > float(config.max_features):
        return False, f"number_of_features={candidate.number_of_features:g} exceeds max_features={config.max_features}"
    if candidate.missing_pct > float(config.max_missing_pct):
        return False, f"missing_pct={candidate.missing_pct:g} exceeds max_missing_pct={config.max_missing_pct:g}"
    if config.task_type == _CLASSIFICATION_TASK_TYPE:
        if candidate.number_of_classes is None:
            return False, "number_of_classes missing from task listing"
        if config.max_classes is not None and candidate.number_of_classes > float(config.max_classes):
            return False, (
                f"number_of_classes={candidate.number_of_classes:g} exceeds max_classes={config.max_classes}"
            )
        if candidate.minority_class_pct is None:
            return False, "minority_class_pct missing from task listing"
        if candidate.minority_class_pct < float(config.min_minority_class_pct):
            return False, (
                "minority_class_pct="
                f"{candidate.minority_class_pct:g} below min_minority_class_pct={config.min_minority_class_pct:g}"
            )
    return True, "listing filters matched"


def candidate_from_task_listing_row(
    row: Mapping[str, Any],
    *,
    config: OpenMLBundleConfig,
) -> OpenMLTaskCandidate:
    task_id = int(coerce_finite_float(lookup_task_listing_value(row, "tid", "task_id"), context="task listing tid"))
    dataset_id = int(coerce_finite_float(lookup_task_listing_value(row, "did", "data_id"), context="task listing did"))
    dataset_name = str(lookup_task_listing_value(row, "name")).strip()
    if not dataset_name:
        raise RuntimeError("task listing dataset name must be non-empty")
    number_of_instances = coerce_finite_float(
        lookup_task_listing_value(row, "NumberOfInstances"),
        context=f"task listing NumberOfInstances for task {task_id}",
    )
    number_of_features = coerce_finite_float(
        lookup_task_listing_value(row, "NumberOfFeatures"),
        context=f"task listing NumberOfFeatures for task {task_id}",
    )
    missing_instances = coerce_finite_float(
        lookup_task_listing_value(row, "NumberOfInstancesWithMissingValues"),
        context=f"task listing NumberOfInstancesWithMissingValues for task {task_id}",
    )
    missing_pct = 0.0 if number_of_instances <= 0.0 else (100.0 * missing_instances / number_of_instances)
    number_of_classes = None
    minority_class_pct = None
    if config.task_type == _CLASSIFICATION_TASK_TYPE:
        number_of_classes = coerce_finite_float(
            lookup_task_listing_value(row, "NumberOfClasses"),
            context=f"task listing NumberOfClasses for task {task_id}",
        )
        minority_class_size = coerce_finite_float(
            lookup_task_listing_value(row, "MinorityClassSize"),
            context=f"task listing MinorityClassSize for task {task_id}",
        )
        minority_class_pct = (
            0.0 if number_of_instances <= 0.0 else (100.0 * minority_class_size / number_of_instances)
        )
    estimation_procedure_raw = row.get("estimation_procedure")
    estimation_procedure = None if estimation_procedure_raw is None else str(estimation_procedure_raw).strip()
    return OpenMLTaskCandidate(
        task_id=task_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        estimation_procedure=estimation_procedure,
        number_of_instances=number_of_instances,
        number_of_features=number_of_features,
        number_of_classes=number_of_classes,
        missing_pct=missing_pct,
        minority_class_pct=minority_class_pct,
    )


def is_preferred_ten_fold_cv(candidate: OpenMLTaskCandidate) -> bool:
    estimation_procedure = candidate.estimation_procedure
    if estimation_procedure is None:
        return False
    normalized = estimation_procedure.strip().casefold()
    return "10-fold" in normalized and "crossvalidation" in normalized.replace(" ", "")


def dedupe_discovered_candidates(
    candidates: list[OpenMLTaskCandidate],
) -> tuple[list[OpenMLTaskCandidate], list[OpenMLCandidateReportEntry]]:
    grouped: dict[int, list[OpenMLTaskCandidate]] = {}
    for candidate in candidates:
        if candidate.dataset_id is None:
            raise RuntimeError(f"discovered task {candidate.task_id} is missing a dataset_id")
        grouped.setdefault(int(candidate.dataset_id), []).append(candidate)
    selected: list[OpenMLTaskCandidate] = []
    report_entries: list[OpenMLCandidateReportEntry] = []
    for dataset_id, grouped_candidates in grouped.items():
        preferred = min(
            grouped_candidates,
            key=lambda candidate: (0 if is_preferred_ten_fold_cv(candidate) else 1, int(candidate.task_id)),
        )
        selected.append(preferred)
        for candidate in grouped_candidates:
            if candidate.task_id == preferred.task_id:
                continue
            report_entries.append(
                OpenMLCandidateReportEntry(
                    task_id=int(candidate.task_id),
                    dataset_id=candidate.dataset_id,
                    dataset_name=candidate.dataset_name,
                    estimation_procedure=candidate.estimation_procedure,
                    status="rejected",
                    reason=(
                        f"duplicate dataset_id={dataset_id}; preferred task_id={preferred.task_id} "
                        f"via estimation_procedure={preferred.estimation_procedure or '<missing>'}"
                    ),
                )
            )
    return selected, sorted(report_entries, key=lambda entry: int(entry.task_id))


def _collect_discovered_task_candidates(
    config: OpenMLBundleConfig,
    *,
    list_tasks_fn: Any,
) -> tuple[list[OpenMLTaskCandidate], list[OpenMLCandidateReportEntry]]:
    eligible_candidates: list[OpenMLTaskCandidate] = []
    report_entries: list[OpenMLCandidateReportEntry] = []
    for row in _task_listing_rows_for_config(config, list_tasks_fn=list_tasks_fn):
        try:
            candidate = candidate_from_task_listing_row(row, config=config)
            keep_candidate, reason = candidate_matches_listing_filters(candidate, config)
        except (KeyError, RuntimeError) as exc:
            raw_task_id = row.get("tid", row.get("task_id", -1))
            report_entries.append(
                OpenMLCandidateReportEntry(
                    task_id=int(raw_task_id),
                    dataset_id=None,
                    dataset_name=None if row.get("name") is None else str(row.get("name")),
                    estimation_procedure=(
                        None if row.get("estimation_procedure") is None else str(row.get("estimation_procedure"))
                    ),
                    status="rejected",
                    reason=str(exc),
                )
            )
            continue
        if keep_candidate:
            eligible_candidates.append(candidate)
            continue
        report_entries.append(
            OpenMLCandidateReportEntry(
                task_id=int(candidate.task_id),
                dataset_id=candidate.dataset_id,
                dataset_name=candidate.dataset_name,
                estimation_procedure=candidate.estimation_procedure,
                status="rejected",
                reason=reason,
            )
        )
    deduped_candidates, dedupe_report_entries = dedupe_discovered_candidates(eligible_candidates)
    report_entries.extend(dedupe_report_entries)
    return deduped_candidates, report_entries


def validate_prepared_task(
    prepared: PreparedOpenMLTask,
    *,
    config: OpenMLBundleConfig,
) -> None:
    if int(prepared.observed_task["n_rows"]) != int(config.new_instances):
        raise RuntimeError(
            "observed row count mismatch after subsampling: "
            f"expected={config.new_instances}, actual={prepared.observed_task['n_rows']}"
        )
    number_of_features = float(prepared.qualities["NumberOfFeatures"])
    if number_of_features > float(config.max_features):
        raise RuntimeError(f"number_of_features={number_of_features:g} exceeds max_features={config.max_features}")
    missing_pct = float(prepared.qualities["PercentageOfInstancesWithMissingValues"])
    if missing_pct > float(config.max_missing_pct):
        raise RuntimeError(f"missing_pct={missing_pct:g} exceeds max_missing_pct={config.max_missing_pct:g}")
    if config.task_type == _CLASSIFICATION_TASK_TYPE:
        number_of_classes = float(prepared.qualities["NumberOfClasses"])
        if config.max_classes is not None and number_of_classes > float(config.max_classes):
            raise RuntimeError(f"number_of_classes={number_of_classes:g} exceeds max_classes={config.max_classes}")
        minority_class_pct = float(prepared.qualities["MinorityClassPercentage"])
        if minority_class_pct < float(config.min_minority_class_pct):
            raise RuntimeError(
                "minority_class_pct="
                f"{minority_class_pct:g} below min_minority_class_pct={config.min_minority_class_pct:g}"
            )


def _collect_task_candidates(
    config: OpenMLBundleConfig,
    *,
    get_task_fn: Any,
) -> list[OpenMLTaskCandidate]:
    resolved_task_ids = (
        tuple(int(task_id) for task_id in config.task_ids)
        if config.task_ids is not None
        else tuple(int(task_id) for task_id in task_ids_for_source(config.task_source))
    )
    resolved_candidates: list[OpenMLTaskCandidate] = []
    for task_id in resolved_task_ids:
        task = get_task_fn(int(task_id), download_splits=False)
        expected_task_type = (
            TaskType.SUPERVISED_CLASSIFICATION
            if config.task_type == _CLASSIFICATION_TASK_TYPE
            else TaskType.SUPERVISED_REGRESSION
        )
        if task_type_value(task.task_type_id) != task_type_value(expected_task_type):
            continue
        task_any: Any = task
        dataset = task_any.get_dataset(download_data=False)
        dataset_any: Any = dataset
        raw_qualities = dataset_any.qualities
        candidate = OpenMLTaskCandidate(
            task_id=int(task_id),
            number_of_features=read_required_quality(
                raw_qualities,
                task_id=int(task_id),
                quality_name="NumberOfFeatures",
            ),
            number_of_classes=(
                None
                if config.task_type != _CLASSIFICATION_TASK_TYPE
                else read_required_quality(
                    raw_qualities,
                    task_id=int(task_id),
                    quality_name="NumberOfClasses",
                )
            ),
            missing_pct=read_required_quality(
                raw_qualities,
                task_id=int(task_id),
                quality_name="PercentageOfInstancesWithMissingValues",
            ),
            minority_class_pct=(
                None
                if config.task_type != _CLASSIFICATION_TASK_TYPE
                else read_required_quality(
                    raw_qualities,
                    task_id=int(task_id),
                    quality_name="MinorityClassPercentage",
                )
            ),
        )
        keep_candidate = (
            candidate.number_of_features <= float(config.max_features)
            and candidate.missing_pct <= float(config.max_missing_pct)
        )
        if config.task_type == _CLASSIFICATION_TASK_TYPE:
            keep_candidate = (
                keep_candidate
                and candidate.minority_class_pct is not None
                and candidate.minority_class_pct >= float(config.min_minority_class_pct)
            )
            if config.max_classes is not None:
                keep_candidate = (
                    keep_candidate
                    and candidate.number_of_classes is not None
                    and candidate.number_of_classes <= float(config.max_classes)
                )
        if keep_candidate:
            resolved_candidates.append(candidate)
    return resolved_candidates


def _resolve_selected_tasks(
    config: OpenMLBundleConfig,
    *,
    prepare_task_fn: _PreparedTaskProvider,
    get_task_fn: Any,
    list_tasks_fn: Any,
) -> tuple[list[PreparedOpenMLTask], int, tuple[OpenMLCandidateReportEntry, ...]]:
    report_entries: list[OpenMLCandidateReportEntry] = []
    if config.discover_from_openml:
        eligible_candidates, discovery_report_entries = _collect_discovered_task_candidates(
            config,
            list_tasks_fn=list_tasks_fn,
        )
        report_entries.extend(discovery_report_entries)
    else:
        eligible_candidates = _collect_task_candidates(config, get_task_fn=get_task_fn)
    if not eligible_candidates:
        raise RuntimeError("OpenML benchmark bundle produced no eligible tasks")

    effective_max_classes = (
        0
        if config.task_type != _CLASSIFICATION_TASK_TYPE
        else (
            max(
                int(candidate.number_of_classes)
                for candidate in eligible_candidates
                if candidate.number_of_classes is not None
            )
            if config.max_classes is None
            else int(config.max_classes)
        )
    )
    selected_candidates = (
        eligible_candidates
        if config.task_type != _CLASSIFICATION_TASK_TYPE
        else [
            candidate
            for candidate in eligible_candidates
            if candidate.number_of_classes is not None
            and int(candidate.number_of_classes) <= effective_max_classes
        ]
    )
    if not selected_candidates:
        raise RuntimeError("OpenML benchmark bundle produced no tasks after task-type filtering")
    selected_tasks: list[PreparedOpenMLTask] = []
    for candidate in selected_candidates:
        try:
            prepared = prepare_task_fn(
                int(candidate.task_id),
                new_instances=int(config.new_instances),
                task_type=str(config.task_type),
            )
            validate_prepared_task(prepared, config=config)
        except RuntimeError as exc:
            if config.discover_from_openml:
                report_entries.append(
                    OpenMLCandidateReportEntry(
                        task_id=int(candidate.task_id),
                        dataset_id=candidate.dataset_id,
                        dataset_name=candidate.dataset_name,
                        estimation_procedure=candidate.estimation_procedure,
                        status="rejected",
                        reason=str(exc),
                    )
                )
                continue
            raise
        if config.discover_from_openml:
            report_entries.append(
                OpenMLCandidateReportEntry(
                    task_id=int(prepared.task_id),
                    dataset_id=candidate.dataset_id,
                    dataset_name=str(prepared.dataset_name),
                    estimation_procedure=candidate.estimation_procedure,
                    status="accepted",
                    reason="validated via prepare_task",
                )
            )
        selected_tasks.append(prepared)
    if config.discover_from_openml and len(selected_tasks) < int(config.min_task_count):
        raise RuntimeError(
            "OpenML benchmark bundle validated task count is below min_task_count: "
            f"validated={len(selected_tasks)}, min_task_count={config.min_task_count}"
        )
    dataset_name_counts = Counter(str(prepared.dataset_name) for prepared in selected_tasks)
    duplicate_dataset_names = sorted(name for name, count in dataset_name_counts.items() if count > 1)
    if duplicate_dataset_names:
        raise RuntimeError(
            "OpenML benchmark bundle produced duplicate dataset names after validation: "
            f"{duplicate_dataset_names}"
        )
    return (
        sorted(selected_tasks, key=lambda prepared: int(prepared.task_id)),
        int(effective_max_classes),
        tuple(sorted(report_entries, key=lambda entry: (entry.status, int(entry.task_id)))),
    )


def bundle_selection_payload(config: OpenMLBundleConfig, *, max_classes: int) -> dict[str, Any]:
    payload = {
        "new_instances": int(config.new_instances),
        "task_type": str(config.task_type),
        "max_features": int(config.max_features),
        "max_missing_pct": float(config.max_missing_pct),
    }
    if config.task_type == _CLASSIFICATION_TASK_TYPE:
        payload["max_classes"] = int(max_classes)
        payload["min_minority_class_pct"] = float(config.min_minority_class_pct)
    return payload


def build_bundle_result(
    config: OpenMLBundleConfig,
    *,
    prepare_task_fn: _PreparedTaskProvider | None = None,
    get_task_fn: Any | None = None,
    list_tasks_fn: Any | None = None,
) -> OpenMLBundleBuildResult:
    selected_tasks, effective_max_classes, report_entries = _resolve_selected_tasks(
        config,
        prepare_task_fn=prepare_task if prepare_task_fn is None else prepare_task_fn,
        get_task_fn=openml.tasks.get_task if get_task_fn is None else get_task_fn,
        list_tasks_fn=openml.tasks.list_tasks if list_tasks_fn is None else list_tasks_fn,
    )
    payload = {
        "name": str(config.bundle_name),
        "version": int(config.version),
        "selection": bundle_selection_payload(config, max_classes=effective_max_classes),
        "task_ids": [int(prepared.task_id) for prepared in selected_tasks],
        "tasks": [dict(prepared.observed_task) for prepared in selected_tasks],
    }
    return OpenMLBundleBuildResult(
        bundle=normalize_bundle(payload),
        report_entries=report_entries,
    )


def build_bundle(config: OpenMLBundleConfig) -> dict[str, Any]:
    return build_bundle_result(config).bundle


def render_candidate_report(entries: Sequence[OpenMLCandidateReportEntry]) -> str:
    if not entries:
        return ""
    accepted = [entry for entry in entries if entry.status == "accepted"]
    rejected = [entry for entry in entries if entry.status == "rejected"]
    lines = [
        "OpenML discovery candidate report:",
        f"- accepted={len(accepted)}",
        f"- rejected={len(rejected)}",
    ]
    if accepted:
        lines.append("Accepted:")
        for entry in accepted:
            lines.append(
                "- "
                f"task_id={entry.task_id} dataset_id={entry.dataset_id} "
                f"dataset_name={entry.dataset_name!r} reason={entry.reason}"
            )
    if rejected:
        lines.append("Rejected:")
        for entry in rejected:
            lines.append(
                "- "
                f"task_id={entry.task_id} dataset_id={entry.dataset_id} "
                f"dataset_name={entry.dataset_name!r} reason={entry.reason}"
            )
    return "\n".join(lines)


def write_bundle(
    path: Path,
    config: OpenMLBundleConfig,
    *,
    bundle: Mapping[str, Any] | None = None,
) -> Path:
    payload = build_bundle(config) if bundle is None else normalize_bundle(dict(bundle))
    return write_json(path.expanduser().resolve(), payload)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "dataset"


def _build_split_table(rows: list[tuple[int, np.ndarray, np.ndarray]]) -> pa.Table:
    dataset_indices: list[int] = []
    row_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for dataset_index, x, y in rows:
        for row_index in range(int(x.shape[0])):
            dataset_indices.append(int(dataset_index))
            row_indices.append(int(row_index))
            x_rows.append(np.asarray(x[row_index], dtype=np.float32).tolist())
            y_rows.append(int(y[row_index]))
    return pa.table(
        {
            "dataset_index": pa.array(dataset_indices, type=pa.int64()),
            "row_index": pa.array(row_indices, type=pa.int64()),
            "x": pa.array(x_rows, type=pa.list_(pa.float32())),
            "y": pa.array(y_rows, type=pa.int64()),
        }
    )


def _write_packed_shard(
    shard_dir: Path,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(_build_split_table([(0, x_train, y_train)]), shard_dir / "train.parquet")
    pq.write_table(_build_split_table([(0, x_test, y_test)]), shard_dir / "test.parquet")
    payload = {
        "dataset_index": 0,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]),
        "feature_types": ["floating"] * int(x_train.shape[1]),
        "metadata": metadata,
    }
    with (shard_dir / "metadata.ndjson").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _split_prepared_task(
    prepared: PreparedOpenMLTask,
    *,
    split_seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    split_kwargs: dict[str, Any] = {
        "test_size": float(test_size),
        "random_state": int(split_seed),
    }
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            prepared.x,
            prepared.y,
            stratify=prepared.y,
            **split_kwargs,
        )
        return x_train, x_test, y_train, y_test, "stratified"
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            prepared.x,
            prepared.y,
            stratify=None,
            **split_kwargs,
        )
        return x_train, x_test, y_train, y_test, "unstratified_fallback"


def materialize_bundle(
    bundle_path: Path,
    out_root: Path,
    *,
    force: bool = False,
    split_seed: int = DEFAULT_COMPARATOR_SPLIT_SEED,
    test_size: float = DEFAULT_COMPARATOR_TEST_SIZE,
    prepare_task_fn: _PreparedTaskProvider | None = None,
) -> OpenMLMaterializationResult:
    """Materialize one OpenML bundle into packed shards and a manifest."""

    resolved_bundle_path = bundle_path.expanduser().resolve()
    resolved_out_root = out_root.expanduser().resolve()
    if resolved_out_root.exists():
        if not force:
            raise RuntimeError(f"output root already exists, rerun with force=True: {resolved_out_root}")
        shutil.rmtree(resolved_out_root)
    data_root = resolved_out_root / "packed_shards"
    manifest_path = resolved_out_root / "manifest.parquet"

    bundle = load_bundle(resolved_bundle_path, allow_missing_values=True)
    selection = cast(dict[str, Any], bundle["selection"])
    allow_missing_values = bundle_allows_missing_values(bundle)
    task_summaries: list[dict[str, Any]] = []
    provider = prepare_task if prepare_task_fn is None else prepare_task_fn

    data_root.mkdir(parents=True, exist_ok=True)
    for task_order, task_id in enumerate(cast(list[int], bundle["task_ids"]), start=1):
        prepared = provider(
            int(task_id),
            new_instances=int(selection["new_instances"]),
            task_type=str(selection["task_type"]),
        )
        x_train, x_test, y_train, y_test, split_mode = _split_prepared_task(
            prepared,
            split_seed=split_seed,
            test_size=test_size,
        )
        metadata = {
            "config": {
                "dataset": {
                    "task": (
                        "classification"
                        if str(selection["task_type"]) == _CLASSIFICATION_TASK_TYPE
                        else "regression"
                    )
                }
            },
            "filter": {"mode": "deferred", "status": "not_run"},
            "seed": int(split_seed),
            "n_features": int(prepared.x.shape[1]),
            "n_classes": (
                int(np.unique(prepared.y).size)
                if str(selection["task_type"]) == _CLASSIFICATION_TASK_TYPE
                else None
            ),
            "source_platform": "openml",
            "benchmark_bundle": {
                "name": str(bundle["name"]),
                "source_path": str(resolved_bundle_path),
                "task_id": int(task_id),
                "allow_missing_values": bool(allow_missing_values),
            },
            "openml": {
                "task_id": int(task_id),
                "dataset_name": str(prepared.dataset_name),
            },
            "split_policy": {
                "name": "deterministic_holdout",
                "test_size": float(test_size),
                "seed": int(split_seed),
                "mode": split_mode,
            },
        }
        if metadata["n_classes"] is None:
            del metadata["n_classes"]
        shard_dir = data_root / f"shard_{task_order:05d}_{_slugify(prepared.dataset_name)}"
        _write_packed_shard(
            shard_dir,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            metadata=metadata,
        )
        task_summary = {
            "task_id": int(task_id),
            "dataset_name": str(prepared.dataset_name),
            "n_rows": int(prepared.x.shape[0]),
            "n_features": int(prepared.x.shape[1]),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "split_mode": split_mode,
            "shard_dir": str(shard_dir),
        }
        if str(selection["task_type"]) == _CLASSIFICATION_TASK_TYPE:
            task_summary["n_classes"] = int(np.unique(prepared.y).size)
        task_summaries.append(task_summary)

    build_manifest([data_root], manifest_path)
    return OpenMLMaterializationResult(
        bundle_summary=bundle_summary(bundle, source_path=resolved_bundle_path),
        task_summaries=tuple(task_summaries),
        allow_missing_values=bool(allow_missing_values),
        data_root=data_root,
        manifest_path=manifest_path,
    )
