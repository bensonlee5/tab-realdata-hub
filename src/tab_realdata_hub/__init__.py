"""Manifest-backed real-data ingestion helpers."""

from .manifest import (
    DATASET_CATALOG_FILENAME,
    build_manifest,
    inspect_manifest,
    load_dataset_catalog_records,
    write_dataset_catalog,
)
from .openml import OpenMLBundleConfig, build_bundle, materialize_bundle, prepare_task, write_bundle

__all__ = [
    "DATASET_CATALOG_FILENAME",
    "OpenMLBundleConfig",
    "build_bundle",
    "build_manifest",
    "inspect_manifest",
    "load_dataset_catalog_records",
    "materialize_bundle",
    "prepare_task",
    "write_dataset_catalog",
    "write_bundle",
]
