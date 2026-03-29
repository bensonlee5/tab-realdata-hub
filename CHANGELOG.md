# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-03-29

### Changed

- User-facing note: `tab-realdata-hub bundle build-openml` now accepts
  additive `--min-classes` filtering for classification bundles, and bundle
  selection payloads now persist additive `min_classes` metadata while
  remaining backward-compatible with older bundle JSON that omitted it.
- User-facing note: added checked-in TF-RD-010 multiclass validation bundle
  definitions at
  `src/tab_realdata_hub/bench/nanotabpfn_openml_classification_medium_v1.json`
  and
  `src/tab_realdata_hub/bench/nanotabpfn_openml_classification_large_v1.json`.
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
