FROM lukemathwalker/cargo-chef:latest-rust-1.75-bookworm AS chef
WORKDIR /usr/src

ENV SCCACHE=0.5.4
ENV RUSTC_WRAPPER=/usr/local/bin/sccache

# Donwload and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

FROM chef AS planner

COPY mistralrs mistralrs
COPY mistralrs-core mistralrs-core
COPY mistralrs-lora mistralrs-lora
COPY mistralrs-pyo3 mistralrs-pyo3
COPY mistralrs-server mistralrs-server
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder

COPY mistralrs mistralrs
COPY mistralrs-core mistralrs-core
COPY mistralrs-lora mistralrs-lora
COPY mistralrs-pyo3 mistralrs-pyo3
COPY mistralrs-server mistralrs-server
COPY Cargo.toml ./
COPY Cargo.lock ./

ARG GIT_SHA
ARG DOCKER_LABEL

# sccache specific variables
ARG ACTIONS_CACHE_URL
ARG ACTIONS_RUNTIME_TOKEN
ARG SCCACHE_GHA_ENABLED

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=planner /usr/src/recipe.json recipe.json

RUN cargo chef cook --release --recipe-path recipe.json && sccache -s

RUN cargo build --release --bin mistralrs-server && sccache -s

FROM debian:bookworm-slim as base

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80 \
    MKL_ENABLE_INSTRUCTIONS=AVX512_E4 \
    RAYON_NUM_THREADS=8 \
    LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libomp-dev \
    ca-certificates \
    libssl-dev \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

FROM base

COPY --from=builder /usr/src/target/release/mistralrs-server /usr/local/bin/mistralrs-server
RUN chmod +x /usr/local/bin/mistralrs-server
ENTRYPOINT ["mistralrs-server"]