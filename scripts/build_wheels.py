#!/usr/bin/env python3
"""
Build script for mistralrs Python wheels.

Auto-detects platform, architecture, and available accelerators.
Builds appropriate wheels based on the detected environment.

Usage:
    python scripts/build_wheels.py --list                    # Show buildable packages
    python scripts/build_wheels.py --all                     # Build all supported
    python scripts/build_wheels.py -p mistralrs mistralrs-cuda
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# ============================================================================
# Constants and Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
PYPROJECT_PATH = REPO_ROOT / "mistralrs-pyo3" / "pyproject.toml"
CARGO_MANIFEST = REPO_ROOT / "mistralrs-pyo3" / "Cargo.toml"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile.manylinux"

PACKAGE_NAMES = [
    "mistralrs",
    "mistralrs-cuda",
    "mistralrs-metal",
    "mistralrs-accelerate",
    "mistralrs-mkl",
]


class OS(Enum):
    LINUX = "linux"
    DARWIN = "darwin"
    WINDOWS = "windows"


class Arch(Enum):
    X86_64 = "x86_64"
    AARCH64 = "aarch64"


@dataclass
class Platform:
    os: OS
    arch: Arch
    has_cuda: bool
    has_metal: bool


@dataclass
class PackageConfig:
    name: str
    features: list[str]
    supported_os: list[OS]
    supported_arch: list[Arch]
    requires_accelerator: Optional[str]  # "cuda", "metal", or None


# ============================================================================
# Platform Detection
# ============================================================================


def detect_platform() -> Platform:
    """Auto-detect OS, architecture, and available accelerators."""
    # Detect OS
    system = platform.system().lower()
    if system == "linux":
        os_type = OS.LINUX
    elif system == "darwin":
        os_type = OS.DARWIN
    elif system == "windows":
        os_type = OS.WINDOWS
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    # Detect architecture
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = Arch.X86_64
    elif machine in ("aarch64", "arm64"):
        arch = Arch.AARCH64
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    # Detect CUDA
    has_cuda = _detect_cuda()

    # Detect Metal (macOS aarch64 only)
    has_metal = os_type == OS.DARWIN and arch == Arch.AARCH64

    return Platform(os=os_type, arch=arch, has_cuda=has_cuda, has_metal=has_metal)


def _detect_cuda() -> bool:
    """Check if CUDA is available."""
    # Check for nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and len(result.stdout.strip()) > 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Check for CUDA library paths
    cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        os.environ.get("CUDA_HOME", ""),
        os.environ.get("CUDA_PATH", ""),
    ]
    return any(Path(p).exists() for p in cuda_paths if p)


# ============================================================================
# Package Configuration
# ============================================================================


def get_package_configs() -> dict[str, PackageConfig]:
    """Define the build configuration for each package."""
    return {
        "mistralrs": PackageConfig(
            name="mistralrs",
            features=[],  # Features determined by platform
            supported_os=[OS.LINUX, OS.DARWIN, OS.WINDOWS],
            supported_arch=[Arch.X86_64, Arch.AARCH64],
            requires_accelerator=None,
        ),
        "mistralrs-cuda": PackageConfig(
            name="mistralrs-cuda",
            features=["cuda"],
            supported_os=[OS.LINUX, OS.WINDOWS],
            supported_arch=[Arch.X86_64, Arch.AARCH64],
            requires_accelerator="cuda",
        ),
        "mistralrs-metal": PackageConfig(
            name="mistralrs-metal",
            features=["metal"],
            supported_os=[OS.DARWIN],
            supported_arch=[Arch.AARCH64],
            requires_accelerator="metal",
        ),
        "mistralrs-accelerate": PackageConfig(
            name="mistralrs-accelerate",
            features=["accelerate"],
            supported_os=[OS.DARWIN],
            supported_arch=[Arch.AARCH64],
            requires_accelerator=None,  # Accelerate is always available on macOS
        ),
        "mistralrs-mkl": PackageConfig(
            name="mistralrs-mkl",
            features=["mkl"],
            supported_os=[OS.LINUX, OS.WINDOWS],
            supported_arch=[Arch.X86_64],
            requires_accelerator=None,
        ),
    }


def get_features_for_base_package(plat: Platform) -> list[str]:
    """Get features for the 'mistralrs' base package based on platform."""
    if plat.os == OS.DARWIN and plat.arch == Arch.AARCH64:
        return ["metal"]  # macOS aarch64: Metal
    elif plat.arch == Arch.X86_64:
        return ["mkl"]  # x86_64: MKL
    else:
        return []  # aarch64 Linux: CPU-only


def get_buildable_packages(
    configs: dict[str, PackageConfig], plat: Platform
) -> list[str]:
    """Get list of packages that can be built on the current platform."""
    buildable = []

    for name, cfg in configs.items():
        # Check OS and arch support
        if plat.os not in cfg.supported_os:
            continue
        if plat.arch not in cfg.supported_arch:
            continue

        # Check accelerator requirements
        if cfg.requires_accelerator == "cuda" and not plat.has_cuda:
            continue
        if cfg.requires_accelerator == "metal" and not plat.has_metal:
            continue

        buildable.append(name)

    return buildable


# ============================================================================
# PyProject.toml Modification
# ============================================================================


def modify_pyproject_name(name: str) -> None:
    """Modify project.name in pyproject.toml."""
    content = PYPROJECT_PATH.read_text()

    # Use regex to replace the name field
    new_content = re.sub(
        r'^name\s*=\s*"[^"]*"',
        f'name = "{name}"',
        content,
        flags=re.MULTILINE,
    )

    PYPROJECT_PATH.write_text(new_content)
    print(f"  Set project.name to '{name}'")


def restore_pyproject_name() -> None:
    """Restore project.name to default 'mistralrs'."""
    modify_pyproject_name("mistralrs")


# ============================================================================
# Build Functions
# ============================================================================


def build_wheel(
    package_config: PackageConfig,
    plat: Platform,
    output_dir: Path,
) -> Path:
    """Build a wheel for the given package configuration."""
    # Determine features
    if package_config.name == "mistralrs":
        features = get_features_for_base_package(plat)
    else:
        features = package_config.features

    # Create output directory
    package_output = output_dir / package_config.name
    package_output.mkdir(parents=True, exist_ok=True)

    # Modify pyproject.toml
    modify_pyproject_name(package_config.name)

    try:
        # Use Docker manylinux ONLY for CPU-only builds on Linux (no features)
        # CUDA, MKL, and other accelerator builds use native maturin
        if plat.os == OS.LINUX and not features:
            _build_with_docker(features, package_output, plat)
        else:
            _build_with_maturin(features, package_output, plat)
    finally:
        # Always restore pyproject.toml
        restore_pyproject_name()

    return package_output


def _build_with_maturin(features: list[str], output_dir: Path, plat: Platform) -> None:
    """Build using native maturin."""
    cmd = [
        "maturin",
        "build",
        "--release",
        "--strip",
        "-o",
        str(output_dir),
        "-m",
        str(CARGO_MANIFEST),
        "--interpreter",
        "python3.10",
    ]

    if features:
        cmd.extend(["--features", ",".join(features)])

    # Skip auditwheel for CUDA builds - don't bundle CUDA shared libraries
    # Users are expected to have CUDA installed on their system
    # Note: MKL is statically linked (mkl-static-lp64-iomp) so doesn't need this
    if "cuda" in features:
        cmd.extend(["--auditwheel", "skip"])

    env = os.environ.copy()

    # macOS-specific settings for Metal builds
    if plat.os == OS.DARWIN and "metal" in features:
        env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
        print(f"  Setting MACOSX_DEPLOYMENT_TARGET=15.0 for Metal build")

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)


def _build_with_docker(features: list[str], output_dir: Path, plat: Platform) -> None:
    """Build using Docker manylinux container."""
    # Build docker image if needed
    print("  Building Docker image...")
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            "mistralrs-wheelmaker:latest",
            "-f",
            "Dockerfile.manylinux",
            ".",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    # Construct maturin command for inside container
    maturin_args = [
        "build",
        "--release",
        "--strip",
        "-o",
        f"/io/wheels/{output_dir.name}",
        "-m",
        "mistralrs-pyo3/Cargo.toml",
        "--interpreter",
        "python3.10",
    ]

    if features:
        maturin_args.extend(["--features", ",".join(features)])

    # Docker command
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT}:/io",
        "-e",
        "RUSTFLAGS=-C target-cpu=generic",
    ]

    docker_cmd.extend(["mistralrs-wheelmaker:latest"] + maturin_args)

    print(f"  Running Docker build with RUSTFLAGS=-C target-cpu=generic")
    print(f"  Maturin args: {' '.join(maturin_args)}")
    subprocess.run(docker_cmd, check=True)

    # Fix ownership of target/ directory (Docker creates files as root)
    import getpass

    user = getpass.getuser()
    print(f"  Fixing ownership of target/ directory...")
    subprocess.run(
        ["sudo", "chown", "-R", f"{user}:{user}", "target/"], cwd=REPO_ROOT, check=False
    )

    # Move wheels from repo wheels/ to output_dir
    docker_wheels_dir = REPO_ROOT / "wheels" / output_dir.name
    if docker_wheels_dir.exists() and docker_wheels_dir != output_dir:
        for whl in docker_wheels_dir.glob("*.whl"):
            dest = output_dir / whl.name
            shutil.move(str(whl), str(dest))
            print(f"  Moved {whl.name} to {output_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build mistralrs Python wheels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build all packages supported on current platform
  python scripts/build_wheels.py --all

  # Build specific packages
  python scripts/build_wheels.py --packages mistralrs mistralrs-cuda

  # Specify output directory
  python scripts/build_wheels.py --all -o ./dist

  # List what can be built on this platform
  python scripts/build_wheels.py --list
        """,
    )

    parser.add_argument(
        "--packages",
        "-p",
        nargs="+",
        choices=PACKAGE_NAMES,
        help="Packages to build",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Build all packages supported on current platform",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("wheels"),
        help="Output directory for wheels (default: ./wheels)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List packages that can be built on current platform",
    )

    args = parser.parse_args()

    # Detect platform
    plat = detect_platform()
    print(f"Detected platform: {plat.os.value}/{plat.arch.value}")
    print(f"  CUDA available: {plat.has_cuda}")
    print(f"  Metal available: {plat.has_metal}")

    # Get package configs
    configs = get_package_configs()

    # Filter to packages buildable on this platform
    buildable = get_buildable_packages(configs, plat)

    if args.list:
        print("\nPackages buildable on this platform:")
        for name in buildable:
            cfg = configs[name]
            features = (
                cfg.features
                if cfg.name != "mistralrs"
                else get_features_for_base_package(plat)
            )
            print(f"  - {name} (features: {features or 'none'})")
        return 0

    # Determine which packages to build
    if args.packages:
        to_build = args.packages
    elif args.all:
        to_build = buildable
    else:
        parser.error(
            "Specify --packages or --all, or use --list to see available packages"
        )
        return 1

    # Validate packages
    for pkg in to_build:
        if pkg not in buildable:
            print(f"Error: {pkg} cannot be built on this platform", file=sys.stderr)
            print(f"Buildable packages: {buildable}", file=sys.stderr)
            return 1

    # Build each package
    args.output.mkdir(parents=True, exist_ok=True)

    for pkg_name in to_build:
        print(f"\n{'=' * 60}")
        print(f"Building {pkg_name}")
        print(f"{'=' * 60}")

        build_wheel(configs[pkg_name], plat, args.output)

    print(f"\n{'=' * 60}")
    print("All wheels built successfully!")
    print(f"Output directory: {args.output.absolute()}")

    # List built wheels
    print("\nBuilt wheels:")
    for whl in args.output.rglob("*.whl"):
        print(f"  {whl.relative_to(args.output)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
