# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-28

### Added

- Initial public package surface for manifest-backed real-data ingestion and
  OpenML bundle materialization.
- Manifest contract ownership in `tab-realdata-hub`, including manifest build,
  inspect, and load helpers with a minimal parquet index and richer
  `metadata.ndjson` payload.
- Upstream-only dependency boundary checks so downstream consumers do not need
  local manifest compatibility shims.
