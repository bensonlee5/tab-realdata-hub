# tab-realdata-hub

`tab-realdata-hub` materializes external tabular data sources into the
manifest-backed packed-shard contract consumed by `tab-foundry`.

`tab-realdata-hub` is the sole owner of that manifest contract. The parquet
manifest is the stable index layer, and richer evolving dataset/provenance
fields live in `metadata.ndjson`. Downstream consumers are expected to read
through this package rather than reimplementing compatibility shims.

Install from the upstream git tag with:

```bash
python -m pip install "tab-realdata-hub @ git+https://github.com/bensonlee5/tab-realdata-hub.git@v0.1.0"
```

For repo-local development:

```bash
uv sync
```

The v1 surface is OpenML-first:

- build pinned OpenML bundle JSON from known task pools or live discovery
- materialize bundle tasks into packed shards plus manifest parquet
- inspect manifest-backed datasets through a stable library and CLI surface

Example:

```bash
uv sync

tab-realdata-hub bundle build-openml \
  --out-path bundles/many_class_v1.json \
  --bundle-name many_class_v1 \
  --version 1 \
  --task-source tabarena_v0_1 \
  --max-features 10 \
  --max-classes 10 \
  --max-missing-pct 10.0

tab-realdata-hub materialize openml-bundle \
  --bundle-path bundles/many_class_v1.json \
  --out-root outputs/openml/many_class_v1

tab-realdata-hub manifest inspect \
  --manifest outputs/openml/many_class_v1/manifest.parquet
```
