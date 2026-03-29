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
- User-facing note: official checked-in benchmark definitions now use plain
  OpenML naming under `src/tab_realdata_hub/bench/`, including aggregate
  `openml_classification_medium_v1.json` and
  `openml_classification_large_v1.json` plus curated `top10`, `top25`, and
  `top50` real-only subset bundles for each surface.
- User-facing note: OpenML task preparation and materialization no longer
  downsample rows via `new_instances`; bundle loading remains backward-
  compatible with older JSON that still carry `selection.new_instances`, but
  that field is ignored when materializing datasets.
- User-facing note: `tab-realdata-hub` now excludes synthetic OpenML datasets
  such as `BNG(...)`, `SEA(...)`, `RandomRBF`, `LED-display`, and
  `monks-problems-*` from official OpenML retrieval flows and checked-in
  benchmark bundles.
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
