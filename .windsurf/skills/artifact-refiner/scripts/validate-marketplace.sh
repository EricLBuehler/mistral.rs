#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

required_files=(
  ".claude-plugin/plugin.json"
  ".claude-plugin/marketplace.json"
  "skills/artifact-refiner/SKILL.md"
  "SKILL.md"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 1
  fi
done

jq empty .claude-plugin/plugin.json
jq empty .claude-plugin/marketplace.json

if command -v claude >/dev/null 2>&1; then
  claude plugin validate .
else
  echo "Warning: 'claude' CLI not found; skipped 'claude plugin validate .'" >&2
fi

echo "Marketplace and plugin validation passed."
