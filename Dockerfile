# Stage 1: Build environment
FROM rust:latest AS builder

# Update and install dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and copy files
WORKDIR /mistralrs
COPY . .

# Build the project in release mode, excluding the specified workspace
RUN cargo build --release --workspace --exclude mistralrs-pyo3

# Stage 2: Minimal runtime environment
FROM debian:bookworm-slim AS runtime

# Set environment variables
ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80 \
    MKL_ENABLE_INSTRUCTIONS=AVX512_E4 \
    RAYON_NUM_THREADS=8 \
    LD_LIBRARY_PATH=/usr/local/lib

# Install only essential runtime dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binaries from the builder stage
COPY --from=builder /mistralrs/target/release/mistralrs-bench /usr/local/bin/
COPY --from=builder /mistralrs/target/release/mistralrs-server /usr/local/bin/

# Make the binaries executable
RUN chmod +x /usr/local/bin/mistralrs-bench /usr/local/bin/mistralrs-server

# Set the entrypoint
ENTRYPOINT ["mistralrs-server", "--port", "80", "--token-source", "env:HUGGING_FACE_HUB_TOKEN"]
