#!/bin/bash
# Wheel generation commands for each target machine.
# Uses scripts/build_wheels.py which auto-detects platform and builds appropriate wheels.
#
# Build method:
# - Docker manylinux: Linux CPU wheels
# - Native maturin: Windows CPU and macOS Metal wheels
# - CUDA wheels are built by .github/workflows/release.yml because they vary by CUDA toolkit and SM.

###############################################################################
# BOX 1: Linux aarch64
###############################################################################

# mistralrs: CPU-only, uses Docker manylinux with RUSTFLAGS="-C target-cpu=generic"
python scripts/build_wheels.py -p mistralrs

###############################################################################
# BOX 2: Linux x86_64
###############################################################################

# mistralrs: CPU wheel; CUDA wheels are release assets
python scripts/build_wheels.py -p mistralrs

###############################################################################
# BOX 3: Windows x86_64
###############################################################################

# All use native maturin (no Docker on Windows)
python scripts/build_wheels.py -p mistralrs

###############################################################################
# BOX 4: macOS aarch64 + Metal
###############################################################################

# All use native maturin with MACOSX_DEPLOYMENT_TARGET=15.0 for Metal
python scripts/build_wheels.py --all

###############################################################################
# UPLOADING
###############################################################################

# Collect all wheels from all boxes to a single directory, then:

# Dry run to verify:
# python scripts/upload_wheels.py ./all_wheels --dry-run

# Upload to TestPyPI first:
# python scripts/upload_wheels.py ./all_wheels --test --token $TESTPYPI_TOKEN

# Upload to PyPI:
# python scripts/upload_wheels.py ./all_wheels --token $PYPI_TOKEN

###############################################################################
# PACKAGE SUMMARY
###############################################################################
#
# Package              | Features    | Platforms                      | Build Method
# ---------------------|-------------|--------------------------------|------------------
# mistralrs            | (none)      | Linux/Windows                  | Docker/native maturin
# mistralrs            | metal       | macOS aarch64                  | Native maturin
# mistralrs            | cuda        | Linux release assets           | Release workflow
#
# Python version: 3.10 only (abi3 provides forward compatibility to 3.11+)
