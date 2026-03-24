#!/bin/bash
# Wheel generation commands for each target machine.
# Uses scripts/build_wheels.py which auto-detects platform and builds appropriate wheels.
#
# Build method:
# - Docker manylinux: ONLY for CPU-only builds on Linux (no features)
# - Native maturin: For CUDA, MKL, Metal, Accelerate builds

###############################################################################
# BOX 1: Linux aarch64 + CUDA
###############################################################################

# mistralrs: CPU-only, uses Docker manylinux with RUSTFLAGS="-C target-cpu=generic"
# mistralrs-cuda: uses native maturin (not Docker)
python scripts/build_wheels.py -p mistralrs mistralrs-cuda

###############################################################################
# BOX 2: Linux x86_64 + CUDA + MKL
###############################################################################

# mistralrs: has MKL, uses native maturin (not Docker, because MKL feature)
# mistralrs-cuda: uses native maturin
# mistralrs-mkl: uses native maturin
python scripts/build_wheels.py -p mistralrs mistralrs-cuda mistralrs-mkl

###############################################################################
# BOX 2: Windows x86_64 + CUDA + MKL
###############################################################################

# All use native maturin (no Docker on Windows)
python scripts/build_wheels.py -p mistralrs mistralrs-cuda mistralrs-mkl

###############################################################################
# BOX 3: macOS aarch64 + Metal
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
# mistralrs            | (none)      | Linux aarch64                  | Docker manylinux
# mistralrs            | mkl         | Linux/Windows x86_64           | Native maturin
# mistralrs            | metal       | macOS aarch64                  | Native maturin
# mistralrs-cuda       | cuda        | Linux + Windows (x86_64/arm64) | Native maturin
# mistralrs-metal      | metal       | macOS aarch64                  | Native maturin
# mistralrs-accelerate | accelerate  | macOS aarch64                  | Native maturin
# mistralrs-mkl        | mkl         | Linux + Windows x86_64         | Native maturin
#
# Python version: 3.10 only (abi3 provides forward compatibility to 3.11+)
