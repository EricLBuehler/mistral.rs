#!/bin/bash
# Wheel generation commands for each target machine.
# Uses scripts/build_wheels.py which auto-detects platform and builds appropriate wheels.

###############################################################################
# BOX 1: Linux aarch64 + CUDA
###############################################################################

# Builds: mistralrs (CPU-only), mistralrs-cuda
# Uses Docker manylinux with RUSTFLAGS="-C target-cpu=generic"
python scripts/build_wheels.py -p mistralrs mistralrs-cuda

###############################################################################
# BOX 2: Linux x86_64 + CUDA + MKL
###############################################################################

# Builds: mistralrs (with MKL), mistralrs-cuda, mistralrs-mkl
# Uses Docker manylinux with RUSTFLAGS="-C target-cpu=generic"
python scripts/build_wheels.py -p mistralrs mistralrs-cuda mistralrs-mkl

###############################################################################
# BOX 2: Windows x86_64 + CUDA + MKL
###############################################################################

# Builds: mistralrs (with MKL), mistralrs-cuda, mistralrs-mkl
# Uses native maturin (no Docker)
python scripts/build_wheels.py -p mistralrs mistralrs-cuda mistralrs-mkl

###############################################################################
# BOX 3: macOS aarch64 + Metal
###############################################################################

# Builds: mistralrs (with Metal), mistralrs-metal, mistralrs-accelerate
# Uses native maturin with MACOSX_DEPLOYMENT_TARGET=15.0 for Metal
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
# Package              | Features    | Platforms
# ---------------------|-------------|------------------------------------------
# mistralrs            | Metal (mac) | All (Metal on macOS, MKL on x86_64,
#                      | MKL (x86)   | CPU-only on aarch64 Linux)
# mistralrs-cuda       | cuda        | Linux + Windows, x86_64 + aarch64
# mistralrs-metal      | metal       | macOS aarch64 only
# mistralrs-accelerate | accelerate  | macOS aarch64 only
# mistralrs-mkl        | mkl         | Linux + Windows, x86_64 only
#
# Python version: 3.10 only (abi3 provides forward compatibility to 3.11+)
