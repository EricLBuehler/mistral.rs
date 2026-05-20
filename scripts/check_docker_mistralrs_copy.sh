#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

status=0
checked=0

for dockerfile in Dockerfile*; do
    if ! grep -q "cargo build --release --workspace" "$dockerfile"; then
        continue
    fi

    checked=$((checked + 1))

    if awk '
        /^FROM[[:space:]]/ {
            copies_mistralrs = 0
        }

        /COPY[[:space:]].*--from=builder.*\/mistralrs\/target\/release\/mistralrs([[:space:]]|$)/ {
            copies_mistralrs = 1
        }

        END {
            exit copies_mistralrs ? 0 : 1
        }
    ' "$dockerfile"; then
        printf "ok: %s copies /mistralrs/target/release/mistralrs into the final stage\n" "$dockerfile"
    else
        printf "error: %s builds the release workspace but does not copy /mistralrs/target/release/mistralrs into the final stage\n" "$dockerfile" >&2
        status=1
    fi
done

if [ "$checked" -eq 0 ]; then
    printf "error: no Dockerfiles with cargo build --release --workspace were found\n" >&2
    exit 1
fi

exit "$status"
