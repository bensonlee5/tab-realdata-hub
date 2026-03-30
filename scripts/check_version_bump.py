from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


def parse_semver(value: str) -> tuple[int, int, int]:
    parts = value.strip().split(".")
    if len(parts) != 3:
        raise ValueError(f"version must be SemVer major.minor.patch: {value!r}")
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"version must contain only integer parts: {value!r}") from exc
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError(f"version parts must be non-negative: {value!r}")
    return major, minor, patch


def classify_version_step(
    base_version: tuple[int, int, int],
    head_version: tuple[int, int, int],
) -> str | None:
    if head_version == (base_version[0], base_version[1], base_version[2] + 1):
        return "patch"
    if head_version == (base_version[0], base_version[1] + 1, 0):
        return "minor"
    if head_version == (base_version[0] + 1, 0, 0):
        return "major"
    return None


def requires_version_bump(changed_files: list[str]) -> bool:
    return any(path.startswith("src/tab_realdata_hub/") for path in changed_files)


def load_version_from_pyproject(payload: bytes) -> str:
    parsed = tomllib.loads(payload.decode("utf-8"))
    version = parsed["project"]["version"]
    if not isinstance(version, str):
        raise ValueError("project.version must be a string")
    return version


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def changed_files_against_base(base_ref: str) -> list[str]:
    output = git_output("diff", "--name-only", f"{base_ref}...HEAD")
    return [line for line in output.splitlines() if line]


def load_base_version(base_ref: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{base_ref}:pyproject.toml"],
        check=True,
        capture_output=True,
    )
    return load_version_from_pyproject(completed.stdout)


def load_head_version(pyproject_path: Path) -> str:
    return load_version_from_pyproject(pyproject_path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one-step SemVer bumps for src changes.")
    parser.add_argument("--base-ref", required=True, help="Base git ref to compare against, e.g. origin/main")
    parser.add_argument(
        "--pyproject-path",
        default="pyproject.toml",
        help="Path to the current branch pyproject.toml",
    )
    args = parser.parse_args(argv)

    changed_files = changed_files_against_base(str(args.base_ref))
    if not requires_version_bump(changed_files):
        print("version-policy: no src/tab_realdata_hub changes detected; skipping version bump check")
        return 0

    base_version_raw = load_base_version(str(args.base_ref))
    head_version_raw = load_head_version(Path(str(args.pyproject_path)))
    base_version = parse_semver(base_version_raw)
    head_version = parse_semver(head_version_raw)
    step_kind = classify_version_step(base_version, head_version)
    if step_kind is None:
        print(
            "version-policy: expected exactly one SemVer step for src/tab_realdata_hub changes; "
            f"base={base_version_raw}, head={head_version_raw}",
            file=sys.stderr,
        )
        return 1

    print(
        "version-policy: validated "
        f"{step_kind} bump for src/tab_realdata_hub changes; base={base_version_raw}, head={head_version_raw}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
