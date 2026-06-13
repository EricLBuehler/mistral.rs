# syntax=docker/dockerfile:1

FROM rust:latest AS builder

WORKDIR /mistralrs
COPY . .

RUN cargo build --release -p mistralrs-cli

FROM debian:bookworm-slim AS runtime
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

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

COPY --chmod=755 --from=builder /mistralrs/target/release/mistralrs /usr/local/bin/mistralrs
# Chat templates for models that ship without one
COPY --from=builder /mistralrs/chat_templates /chat_templates

# hf-hub reads HF_HOME; mount a volume at /data to persist downloaded models
ENV HF_HOME=/data

# Default port of `mistralrs serve`
EXPOSE 1234

ENTRYPOINT ["mistralrs"]
CMD ["--help"]
