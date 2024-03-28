FROM rust:latest as builder

WORKDIR /mistralrs

COPY mistralrs mistralrs
COPY mistralrs-core mistralrs-core
COPY mistralrs-lora mistralrs-lora
COPY mistralrs-pyo3 mistralrs-pyo3
COPY mistralrs-server mistralrs-server
COPY Cargo.toml ./

RUN cargo build --release

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

COPY --from=builder /mistralrs/target/release/mistralrs-server /usr/local/bin/mistralrs-server
RUN chmod +x /usr/local/bin/mistralrs-server
ENTRYPOINT ["mistralrs-server", "--port", "80", "--token-source", "env:HUGGING_FACE_HUB_TOKEN"]