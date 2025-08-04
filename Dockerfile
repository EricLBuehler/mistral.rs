# syntax=docker/dockerfile:1

# Stage 1: Build environment
FROM rust:latest AS builder

# Set working directory and copy files
WORKDIR /mistralrs
COPY . .

# Build the project in release mode, excluding the specified workspace
RUN cargo build --release --workspace --exclude mistralrs-pyo3


# Stage 2: Minimal runtime environment
FROM debian:bookworm-slim AS runtime
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Install only essential runtime dependencies and clean up
ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get update
    apt-get install -y --no-install-recommends \
        libomp-dev \
        ca-certificates \
        libssl-dev \
        curl

    rm -rf /var/lib/apt/lists/*
HEREDOC

# Copy the built binaries from the builder stage
COPY --chmod=755 --from=builder /mistralrs/target/release/mistralrs-bench /usr/local/bin/
COPY --chmod=755 --from=builder /mistralrs/target/release/mistralrs-server /usr/local/bin/
COPY --chmod=755 --from=builder /mistralrs/target/release/mistralrs-web-chat /usr/local/bin/
# Copy chat templates for users running models which may not include them
COPY --from=builder /mistralrs/chat_templates /chat_templates

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80
