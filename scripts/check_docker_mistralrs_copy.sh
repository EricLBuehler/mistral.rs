#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

status=0

require_text() {
    local file=$1
    local text=$2
    local description=$3

    if grep -Fq -- "$text" "$file"; then
        printf 'ok: %s\n' "$description"
    else
        printf 'error: %s (%s)\n' "$description" "$file" >&2
        status=1
    fi
}

# CPU release binaries and their runtime image intentionally share Ubuntu 24.04's
# GLIBC baseline. The build-time --help invocation catches both a missing binary
# and a loader failure before an image can be published.
require_text .github/workflows/release.yml \
    '{ runner: ubuntu-24.04, triple: x86_64-unknown-linux-gnu }' \
    'the x86_64 CPU binary is built on Ubuntu 24.04'
require_text .github/workflows/release.yml \
    '{ runner: ubuntu-24.04-arm, triple: aarch64-unknown-linux-gnu,' \
    'the aarch64 CPU binary is built on Ubuntu 24.04'
require_text Dockerfile \
    'FROM ubuntu:24.04 AS runtime' \
    'the CPU image uses the same GLIBC baseline as its release binaries'
require_text Dockerfile \
    'COPY --chmod=755 dist/${TARGETARCH}/mistralrs /usr/local/bin/mistralrs' \
    'the CPU image copies the mistralrs binary'
require_text Dockerfile \
    'RUN mistralrs --help >/dev/null' \
    'the CPU image smoke-checks the copied binary'

# The source-built UBI image already uses matching UBI 9 builder/runtime stages.
require_text Dockerfile.cuda-13.0-ubi9 \
    'FROM nvidia/cuda:13.0.2-cudnn-devel-ubi9 AS builder' \
    'the UBI CUDA builder uses UBI 9'
require_text Dockerfile.cuda-13.0-ubi9 \
    'FROM nvidia/cuda:13.0.2-cudnn-runtime-ubi9' \
    'the UBI CUDA runtime uses UBI 9'
require_text Dockerfile.cuda-13.0-ubi9 \
    'COPY --chmod=755 --from=builder /mistralrs/target/release/mistralrs /usr/local/bin/mistralrs' \
    'the UBI CUDA image copies the mistralrs binary'

# Prebuilt CUDA packages are built and run on Ubuntu 22.04 CUDA images. The
# package directory contains mistralrs plus its bundled runtime libraries.
require_text .github/workflows/release.yml \
    'image: docker.io/nvidia/cuda:${{ matrix.cuda }}-cudnn-devel-ubuntu22.04' \
    'prebuilt CUDA binaries are built on Ubuntu 22.04'
require_text Dockerfile.cuda-all \
    'FROM nvidia/cuda:${BASE_TAG}-runtime-ubuntu22.04 AS runtime' \
    'prebuilt CUDA images run on Ubuntu 22.04'
require_text Dockerfile.cuda-all \
    'COPY dist/${TARGETARCH}/ /opt/mistralrs/' \
    'the CUDA image copies the prebuilt package'
require_text Dockerfile.cuda-all \
    'ln -sf /opt/mistralrs/mistralrs /usr/local/bin/mistralrs' \
    'the CUDA image exposes the mistralrs binary'

exit "$status"
