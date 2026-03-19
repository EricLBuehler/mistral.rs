#!/usr/bin/env python3
"""
Upload script for mistralrs Python wheels to PyPI.

Supports both PyPI and TestPyPI, with automatic package name detection.

Usage:
    python scripts/upload_wheels.py ./wheels --dry-run       # Verify
    python scripts/upload_wheels.py ./wheels --test          # TestPyPI
    python scripts/upload_wheels.py ./wheels --token $TOKEN  # PyPI
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

PYPI_URL = "https://upload.pypi.org/legacy/"
TESTPYPI_URL = "https://test.pypi.org/legacy/"

# Valid package name prefixes (wheel names use underscores)
VALID_PACKAGE_PREFIXES = {
    "mistralrs",
    "mistralrs_cuda",
    "mistralrs_metal",
    "mistralrs_accelerate",
    "mistralrs_mkl",
}

# Mapping from wheel name prefix to PyPI package name
WHEEL_TO_PYPI = {
    "mistralrs": "mistralrs",
    "mistralrs_cuda": "mistralrs-cuda",
    "mistralrs_metal": "mistralrs-metal",
    "mistralrs_accelerate": "mistralrs-accelerate",
    "mistralrs_mkl": "mistralrs-mkl",
}


@dataclass
class WheelInfo:
    path: Path
    package_name: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str


# ============================================================================
# Wheel Parsing
# ============================================================================


def parse_wheel_filename(wheel_path: Path) -> WheelInfo:
    """Parse wheel filename to extract metadata.

    Wheel format: {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
    """
    name = wheel_path.stem
    parts = name.split("-")

    if len(parts) < 5:
        raise ValueError(f"Invalid wheel filename: {wheel_path.name}")

    # Handle optional build tag
    if len(parts) == 6:
        dist, version, _build, python_tag, abi_tag, platform_tag = parts
    else:
        dist, version, python_tag, abi_tag, platform_tag = parts

    if dist not in VALID_PACKAGE_PREFIXES:
        raise ValueError(f"Unknown package in wheel: {dist}")

    return WheelInfo(
        path=wheel_path,
        package_name=WHEEL_TO_PYPI[dist],
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
    )


def find_wheels(directory: Path) -> list[WheelInfo]:
    """Find all wheel files in directory (recursively)."""
    wheels = []

    for wheel_path in directory.rglob("*.whl"):
        try:
            info = parse_wheel_filename(wheel_path)
            wheels.append(info)
        except ValueError as e:
            print(f"Warning: Skipping {wheel_path.name}: {e}", file=sys.stderr)

    return wheels


# ============================================================================
# Upload Functions
# ============================================================================


def upload_wheels(
    wheels: list[WheelInfo],
    use_testpypi: bool = False,
    token: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Upload wheels to PyPI or TestPyPI."""
    if not wheels:
        print("No wheels to upload")
        return True

    # Group by package
    by_package: dict[str, list[WheelInfo]] = {}
    for w in wheels:
        by_package.setdefault(w.package_name, []).append(w)

    repository_url = TESTPYPI_URL if use_testpypi else PYPI_URL
    repo_name = "TestPyPI" if use_testpypi else "PyPI"

    # Get token
    if token is None:
        token = os.environ.get("PYPI_TOKEN") or os.environ.get("TWINE_PASSWORD")

    if not token and not dry_run:
        print(
            "Error: No PyPI token provided. Use --token or set PYPI_TOKEN env var",
            file=sys.stderr,
        )
        return False

    success = True

    for package_name, package_wheels in sorted(by_package.items()):
        print(f"\n{'=' * 60}")
        print(f"Uploading {package_name} to {repo_name}")
        print(f"{'=' * 60}")

        wheel_paths = [str(w.path) for w in package_wheels]

        print(f"Wheels to upload ({len(wheel_paths)}):")
        for w in package_wheels:
            print(f"  - {w.path.name}")
            print(
                f"    Version: {w.version}, Platform: {w.platform_tag}, ABI: {w.abi_tag}"
            )

        if dry_run:
            print("  [DRY RUN - skipping actual upload]")
            continue

        cmd = [
            "twine",
            "upload",
            "--repository-url",
            repository_url,
            "--username",
            "__token__",
            "--password",
            token,
            "--skip-existing",  # Don't fail on already-uploaded wheels
        ]
        cmd.extend(wheel_paths)

        try:
            subprocess.run(cmd, check=True)
            print(f"  Successfully uploaded {package_name}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed to upload {package_name}: {e}", file=sys.stderr)
            success = False

    return success


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload mistralrs wheels to PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all wheels in ./wheels directory
  python scripts/upload_wheels.py ./wheels

  # Upload to TestPyPI first
  python scripts/upload_wheels.py ./wheels --test

  # Dry run to see what would be uploaded
  python scripts/upload_wheels.py ./wheels --dry-run

  # Provide token explicitly
  python scripts/upload_wheels.py ./wheels --token pypi-xxx

  # Upload specific packages only
  python scripts/upload_wheels.py ./wheels -p mistralrs-cuda
        """,
    )

    parser.add_argument("directory", type=Path, help="Directory containing wheel files")
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Upload to TestPyPI instead of PyPI",
    )
    parser.add_argument(
        "--token",
        help="PyPI API token (or set PYPI_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--packages",
        "-p",
        nargs="+",
        help="Only upload specific packages",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory does not exist: {args.directory}", file=sys.stderr)
        return 1

    # Find wheels
    wheels = find_wheels(args.directory)

    if not wheels:
        print(f"No wheels found in {args.directory}")
        return 1

    print(f"Found {len(wheels)} wheel(s)")

    # Filter by package if specified
    if args.packages:
        # Normalize package names (allow both hyphen and underscore)
        allowed = set()
        for p in args.packages:
            allowed.add(p)
            allowed.add(p.replace("-", "_"))
            allowed.add(p.replace("_", "-"))

        wheels = [
            w
            for w in wheels
            if w.package_name in allowed or w.package_name.replace("-", "_") in allowed
        ]

        if not wheels:
            print(f"No wheels match specified packages: {args.packages}")
            return 1

    # Summary
    by_package: dict[str, list[WheelInfo]] = {}
    for w in wheels:
        by_package.setdefault(w.package_name, []).append(w)

    print("\nSummary:")
    for pkg, pkg_wheels in sorted(by_package.items()):
        print(f"  {pkg}: {len(pkg_wheels)} wheel(s)")
        for w in pkg_wheels:
            print(f"    - {w.platform_tag}")

    # Upload
    target = "TestPyPI" if args.test else "PyPI"
    print(f"\nTarget: {target}")

    if args.dry_run:
        print("[DRY RUN MODE]")

    success = upload_wheels(
        wheels,
        use_testpypi=args.test,
        token=args.token,
        dry_run=args.dry_run,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
