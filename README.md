# tab-realdata-hub

`tab-realdata-hub` materializes external tabular data sources into the
manifest-backed packed-shard contract consumed by `tab-foundry`.

`tab-realdata-hub` is the sole owner of that manifest contract. The parquet
manifest is the stable index layer, and richer evolving dataset/provenance
fields live in `metadata.ndjson`. Downstream consumers are expected to read
through this package rather than reimplementing compatibility shims.

Install from the upstream git tag with:

```bash
python -m pip install "tab-realdata-hub @ git+https://github.com/bensonlee5/tab-realdata-hub.git@v0.1.1"
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

.venv/bin/tab-realdata-hub bundle build-openml \
  --out-path bundles/many_class_v1.json \
  --bundle-name many_class_v1 \
  --version 1 \
  --task-source tabarena_v0_1 \
  --min-classes 2 \
  --max-features 10 \
  --max-classes 10 \
  --max-missing-pct 10.0

.venv/bin/tab-realdata-hub materialize openml-bundle \
  --bundle-path bundles/many_class_v1.json \
  --out-root outputs/openml/many_class_v1

.venv/bin/tab-realdata-hub manifest inspect \
  --manifest outputs/openml/many_class_v1/manifest.parquet
```

The repo now tracks two hub-owned classification validation bundles for
`tab-foundry` under `src/tab_realdata_hub/bench/`:

- `nanotabpfn_openml_classification_medium_v1.json`
- `nanotabpfn_openml_classification_large_v1.json`

The current TF-RD-010 contract is:

- `medium`: no-missing multiclass validation with `max_features=10`,
  `min_classes=3`, `max_classes=10`, and `min_minority_class_pct=2.5`
- `large`: allow-missing multiclass validation with `max_features=20`,
  `max_missing_pct=5.0`, `min_classes=3`, `max_classes=10`, and
  `min_minority_class_pct=2.5`

Refresh the checked-in bundle definitions from the pinned `tabarena_v0_1`
source with:

```bash
.venv/bin/tab-realdata-hub bundle build-openml \
  --out-path src/tab_realdata_hub/bench/nanotabpfn_openml_classification_medium_v1.json \
  --bundle-name nanotabpfn_openml_classification_medium \
  --version 1 \
  --task-source tabarena_v0_1 \
  --new-instances 200 \
  --max-features 10 \
  --min-classes 3 \
  --max-classes 10 \
  --max-missing-pct 0.0 \
  --min-minority-class-pct 2.5

.venv/bin/tab-realdata-hub bundle build-openml \
  --out-path src/tab_realdata_hub/bench/nanotabpfn_openml_classification_large_v1.json \
  --bundle-name nanotabpfn_openml_classification_large \
  --version 1 \
  --task-source tabarena_v0_1 \
  --new-instances 200 \
  --max-features 20 \
  --min-classes 3 \
  --max-classes 10 \
  --max-missing-pct 5.0 \
  --min-minority-class-pct 2.5
```

Materialize the checked-in bundle definitions into the manifest paths consumed
downstream by `tab-foundry` with:

```bash
.venv/bin/tab-realdata-hub materialize openml-bundle \
  --bundle-path src/tab_realdata_hub/bench/nanotabpfn_openml_classification_medium_v1.json \
  --out-root data/manifests/bench/nanotabpfn_openml_classification_medium_v1

.venv/bin/tab-realdata-hub materialize openml-bundle \
  --bundle-path src/tab_realdata_hub/bench/nanotabpfn_openml_classification_large_v1.json \
  --out-root data/manifests/bench/nanotabpfn_openml_classification_large_v1
```

Inspect the resulting manifests with:

```bash
.venv/bin/tab-realdata-hub manifest inspect \
  --manifest data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet

.venv/bin/tab-realdata-hub manifest inspect \
  --manifest data/manifests/bench/nanotabpfn_openml_classification_large_v1/manifest.parquet
```
