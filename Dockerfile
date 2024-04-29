FROM rust:latest as builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mistralrs

COPY . .

RUN cargo build --release --workspace --exclude mistralrs-pyo3

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
    python3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base

COPY --from=builder /mistralrs/target/release/mistralrs-bench /usr/local/bin/mistralrs-bench
RUN chmod +x /usr/local/bin/mistralrs-bench
COPY --from=builder /mistralrs/target/release/mistralrs-server /usr/local/bin/mistralrs-server
RUN chmod +x /usr/local/bin/mistralrs-server
ENTRYPOINT ["mistralrs-server", "--port", "80", "--token-source", "env:HUGGING_FACE_HUB_TOKEN"]