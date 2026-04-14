# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2026-04-14

### Changed

- User-facing note: manifest builds with `missing_value_policy=allow_any` now
  avoid full value-column scans, mark missing-value status as `not_checked`,
  and emit progress telemetry during large parquet index construction while
  preserving the existing manifest contract.

## [0.1.6] - 2026-04-14

### Changed

- User-facing note: dagzoo handoff loading now accepts schema version 5
  manifests and persists v5 intervention provenance summaries alongside the
  existing target-derivation provenance fields.

## [0.1.5] - 2026-04-02

### Changed

- User-facing note: checked-in TF-RD-010 multiclass validation bundle
  artifacts dropped the legacy prefix. The checked-in filenames are now
  `src/tab_realdata_hub/bench/openml_classification_medium_v1.json` and
  `src/tab_realdata_hub/bench/openml_classification_large_v1.json`, and the
  bundle `name` values are now `openml_classification_medium` and
  `openml_classification_large`.

## [0.1.4] - 2026-04-01

### Changed

- User-facing note: dagzoo handoff loading now accepts schema version 4
  manifests, and manifest builds treat roots named `curated` as accepted
  curated shards even when legacy filter metadata is missing from the shard
  payload.

## [0.1.3] - 2026-03-31

### Changed

- User-facing note: dagzoo handoff loading now accepts schema version 3
  manifests, persists provenance target-derivation metadata into the manifest
  summary, and validates numeric provenance range bounds when present.

## [0.1.2] - 2026-03-31

### Changed

- User-facing note: the manifest contract now writes v2 parquet indexes that
  point at shard-level `dataset_catalog.ndjson` records via `catalog_*`
  locator columns, optionally expose `teacher_conditionals.parquet`, and
  continue reading older v1 manifests that still use `metadata.ndjson`.
- User-facing note: dagzoo handoff loading now accepts schema versions 1 and 2;
  v2 handoffs can describe an optional curated root and summarize teacher
  conditionals alongside the generated corpus identity.
- User-facing note: manifest builds now resolve dataset identity and task
  metadata from either the new catalog layout or legacy metadata payloads, and
  curated dagzoo shards default to accepted filter status when legacy filter
  annotations are absent.

## [0.1.1] - 2026-03-29

### Changed

- User-facing note: `tab-realdata-hub bundle build-openml` now accepts
  additive `--min-classes` filtering for classification bundles, and bundle
  selection payloads now persist additive `min_classes` metadata while
  remaining backward-compatible with older bundle JSON that omitted it.
- User-facing note: added checked-in TF-RD-010 multiclass validation bundle
  definitions at
  `src/tab_realdata_hub/bench/openml_classification_medium_v1.json`
  and
  `src/tab_realdata_hub/bench/openml_classification_large_v1.json`.
  The medium bundle is the clean no-missing rung and the large bundle is the
  allow-missing rung consumed downstream under `data/manifests/bench/`.
- User-facing note: OpenML discovery now deterministically dedupes duplicate
  dataset names after dataset-id dedupe and skips candidate-level prepare
  failures during discovery by recording them as rejected rows instead of
  aborting the whole build.

## [0.1.0] - 2026-03-28

### Added

- Initial public package surface for manifest-backed real-data ingestion and
  OpenML bundle materialization.
- Manifest contract ownership in `tab-realdata-hub`, including manifest build,
  inspect, and load helpers with a minimal parquet index and richer
  `metadata.ndjson` payload.
- Upstream-only dependency boundary checks so downstream consumers do not need
  local manifest compatibility shims.
