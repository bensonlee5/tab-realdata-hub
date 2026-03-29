# Development Patterns

## Environment and Entry Points

- Use `.venv/` for commands and tests in this repo.
- Use `uv sync` as the fast path for repo-local bootstrap.
- Discover the packaged CLI via `.venv/bin/tab-realdata-hub --help`, `.venv/bin/tab-realdata-hub <group> --help`, and `.venv/bin/tab-realdata-hub <group> <command> --help`.

## Inspection and Verification

- Prefer the narrow public surfaces before broad codebase sweeps: `tab-realdata-hub manifest inspect --manifest ...`, `tab-realdata-hub bundle build-openml --help`, and `uv run pytest -q`.
- Only fall back to broader greps or whole-repo inspection after those surfaces do not answer the question.
- Prior to declaring a branch ready for review, compare branch to main and verify that all intended changes are included and no unintended changes are included.
- When you commit and push, watch for failing CI and address it and push a fix if needed.

## Architecture and Implementation

- `tab-realdata-hub` is the sole owner of the manifest contract under `src/tab_realdata_hub/`; do not introduce downstream-facing compatibility shims or parallel manifest implementations.
- Keep the manifest contract logical rather than column-shaped: parquet is the stable index layer and `metadata.ndjson` carries richer evolving metadata.
- Keep dependency flow one-way: `tab-realdata-hub` must not depend on `tab-foundry`.
- Prefer shared utility packages over hand-rolled helpers to keep invariants centralized.
- We don't probe data "YOLO-style"; we validate boundaries or rely on typed SDKs.
- We optimize for iteration speed: internal Python APIs and internal config structure may change without backward-compat guarantees.

## User-Facing Changes and Release Hygiene

- If CLI flags, persisted metadata schema, or dataset artifact contract changes, treat it as a user-facing break and call it out explicitly.
- For behavior/schema changes under `src/tab_realdata_hub`, bump version in `pyproject.toml` just before merging into main so that the version reflects the latest changes (patch by default; minor for intentionally broad user-facing breaks). Docs/tests-only changes do not require a bump.
- On every version bump, update `CHANGELOG.md` in the same PR.
