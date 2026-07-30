# syntax=docker/dockerfile:1
#
# Thin CPU runtime image. The prebuilt `mistralrs` binary is staged into `dist/` by the
# release workflow (extracted from the published tarball) and copied in - the image is not
# compiled here. Build via .github/workflows/release.yml.

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

# TARGETARCH (amd64/arm64) is set by buildx; the release workflow stages dist/<arch>/.
ARG TARGETARCH
COPY --chmod=755 dist/${TARGETARCH}/mistralrs /usr/local/bin/mistralrs
# Chat templates for models that ship without one
COPY chat_templates /chat_templates

# hf-hub reads HF_HOME; mount a volume at /data to persist downloaded models
ENV HF_HOME=/data

# Default port of `mistralrs serve`
EXPOSE 1234

ENTRYPOINT ["mistralrs"]
CMD ["--help"]
