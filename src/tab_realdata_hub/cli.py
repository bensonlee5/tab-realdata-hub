"""CLI entrypoint for tab-realdata-hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

from .manifest import build_manifest, inspect_manifest
from .openml import (
    OpenMLBundleConfig,
    build_bundle_result,
    materialize_bundle,
    parse_max_classes_arg,
    render_candidate_report,
    task_source_names,
    write_bundle,
)


def _print_manifest_summary(summary: object) -> None:
    data = cast(Any, summary)
    print(
        "Manifest built:",
        f"path={data.out_path}",
        f"filter_policy={data.filter_policy}",
        f"missing_value_policy={data.missing_value_policy}",
        f"discovered={data.discovered_records}",
        f"excluded={data.excluded_records}",
        f"excluded_for_missing_values={data.excluded_for_missing_values}",
        f"total={data.total_records}",
        f"train={data.train_records}",
        f"val={data.val_records}",
        f"test={data.test_records}",
    )


def _run_manifest_build(args: argparse.Namespace) -> int:
    summary = build_manifest(
        data_roots=[Path(path).expanduser() for path in args.data_root],
        out_path=Path(str(args.out_manifest)),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        filter_policy=str(args.filter_policy),
        missing_value_policy=str(args.missing_value_policy),
    )
    _print_manifest_summary(summary)
    return 0


def _run_manifest_inspect(args: argparse.Namespace) -> int:
    payload = inspect_manifest(Path(str(args.manifest)))
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_bundle_build_openml(args: argparse.Namespace) -> int:
    config = OpenMLBundleConfig(
        bundle_name=str(args.bundle_name),
        version=int(args.version),
        task_source=str(args.task_source),
        task_type=str(args.task_type),
        new_instances=int(args.new_instances),
        max_features=int(args.max_features),
        max_classes=parse_max_classes_arg(str(args.max_classes)),
        max_missing_pct=float(args.max_missing_pct),
        min_minority_class_pct=float(args.min_minority_class_pct),
        discover_from_openml=bool(args.discover_from_openml),
        min_instances=int(args.min_instances),
        min_task_count=int(args.min_task_count),
    )
    if config.min_instances <= 0:
        raise ValueError("min_instances must be a positive int")
    if config.min_task_count <= 0:
        raise ValueError("min_task_count must be a positive int")
    if config.discover_from_openml:
        build_result = build_bundle_result(config)
        report = render_candidate_report(build_result.report_entries)
        if report:
            print(report)
        out_path = write_bundle(Path(str(args.out_path)), config, bundle=build_result.bundle)
    else:
        out_path = write_bundle(Path(str(args.out_path)), config)
    print(f"wrote benchmark bundle: {out_path}")
    return 0


def _run_materialize_openml_bundle(args: argparse.Namespace) -> int:
    result = materialize_bundle(
        Path(str(args.bundle_path)),
        Path(str(args.out_root)),
        force=bool(args.force),
        split_seed=int(args.split_seed),
        test_size=float(args.test_size),
    )
    payload = {
        "bundle_summary": result.bundle_summary,
        "task_summaries": list(result.task_summaries),
        "allow_missing_values": result.allow_missing_values,
        "data_root": str(result.data_root),
        "manifest_path": str(result.manifest_path),
    }
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"Materialized OpenML bundle: {result.manifest_path}")
    print(f"Packed shards: {result.data_root}")
    print(f"Tasks: {len(result.task_summaries)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest-backed real-data ingestion")
    root = parser.add_subparsers(dest="group", required=True)

    manifest_parser = root.add_parser("manifest", help="Manifest build and inspect commands")
    manifest_nested = manifest_parser.add_subparsers(dest="manifest_command", required=True)

    manifest_build = manifest_nested.add_parser("build", help="Build one manifest from packed shard roots")
    manifest_build.add_argument("--data-root", action="append", required=True, help="Packed shard root")
    manifest_build.add_argument("--out-manifest", required=True, help="Output manifest parquet path")
    manifest_build.add_argument("--train-ratio", type=float, default=0.90)
    manifest_build.add_argument("--val-ratio", type=float, default=0.05)
    manifest_build.add_argument(
        "--filter-policy",
        choices=("include_all", "accepted_only"),
        default="include_all",
    )
    manifest_build.add_argument(
        "--missing-value-policy",
        choices=("allow_any", "forbid_any"),
        default="allow_any",
    )
    manifest_build.set_defaults(func=_run_manifest_build)

    manifest_inspect = manifest_nested.add_parser("inspect", help="Inspect one manifest parquet")
    manifest_inspect.add_argument("--manifest", required=True, help="Manifest parquet path")
    manifest_inspect.add_argument("--json", action="store_true")
    manifest_inspect.set_defaults(func=_run_manifest_inspect)

    bundle_parser = root.add_parser("bundle", help="Bundle build commands")
    bundle_nested = bundle_parser.add_subparsers(dest="bundle_command", required=True)
    bundle_build_openml = bundle_nested.add_parser("build-openml", help="Build a pinned OpenML bundle")
    bundle_build_openml.add_argument("--out-path", required=True)
    bundle_build_openml.add_argument("--bundle-name", required=True)
    bundle_build_openml.add_argument("--version", type=int, required=True)
    bundle_build_openml.add_argument(
        "--task-source",
        default="tabarena_v0_1",
        choices=task_source_names(),
    )
    bundle_build_openml.add_argument("--discover-from-openml", action="store_true")
    bundle_build_openml.add_argument("--new-instances", type=int, default=200)
    bundle_build_openml.add_argument("--min-instances", type=int, default=1)
    bundle_build_openml.add_argument("--min-task-count", type=int, default=1)
    bundle_build_openml.add_argument(
        "--task-type",
        default="supervised_classification",
        choices=("supervised_classification", "supervised_regression"),
    )
    bundle_build_openml.add_argument("--max-features", type=int, default=10)
    bundle_build_openml.add_argument("--max-classes", default="2")
    bundle_build_openml.add_argument("--max-missing-pct", type=float, default=0.0)
    bundle_build_openml.add_argument("--min-minority-class-pct", type=float, default=2.5)
    bundle_build_openml.set_defaults(func=_run_bundle_build_openml)

    materialize_parser = root.add_parser("materialize", help="Materialization commands")
    materialize_nested = materialize_parser.add_subparsers(dest="materialize_command", required=True)
    materialize_openml = materialize_nested.add_parser(
        "openml-bundle",
        help="Materialize one OpenML bundle into packed shards and a manifest",
    )
    materialize_openml.add_argument("--bundle-path", required=True)
    materialize_openml.add_argument("--out-root", required=True)
    materialize_openml.add_argument("--force", action="store_true")
    materialize_openml.add_argument("--split-seed", type=int, default=0)
    materialize_openml.add_argument("--test-size", type=float, default=0.20)
    materialize_openml.add_argument("--json", action="store_true")
    materialize_openml.set_defaults(func=_run_materialize_openml_bundle)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
