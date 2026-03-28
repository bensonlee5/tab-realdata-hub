"""Shared numeric validation helpers for dataset inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

MISSING_VALUE_STATUS_CLEAN = "clean"
MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF = "contains_nan_or_inf"
SUPPORTED_MISSING_VALUE_POLICIES = ("allow_any", "forbid_any")


def _numeric_array(value: Any, *, context: str) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        return array
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be numeric to validate missing-value policy") from exc


def contains_non_finite_values(value: Any, *, context: str) -> bool:
    """Return whether a numeric tensor/array/list contains NaN or Inf."""

    array = _numeric_array(value, context=context)
    if array.size <= 0:
        return False
    return bool(np.any(~np.isfinite(array)))


def missing_value_status(named_arrays: Mapping[str, Any], *, context: str) -> str:
    """Summarize whether any named arrays contain NaN or Inf."""

    for name, value in named_arrays.items():
        if contains_non_finite_values(value, context=f"{context}.{name}"):
            return MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF
    return MISSING_VALUE_STATUS_CLEAN


def assert_no_non_finite_values(named_arrays: Mapping[str, Any], *, context: str) -> None:
    """Raise when any named arrays contain NaN or Inf."""

    offenders = [
        name
        for name, value in named_arrays.items()
        if contains_non_finite_values(value, context=f"{context}.{name}")
    ]
    if offenders:
        joined = ", ".join(offenders)
        raise RuntimeError(
            f"{context} contains NaN or Inf in {joined} while allow_missing_values=False"
        )
