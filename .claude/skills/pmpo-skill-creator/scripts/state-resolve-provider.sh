#!/usr/bin/env bash
set -euo pipefail

# Resolve state provider using 6-tier waterfall:
# 1. Environment variable (CREATOR_PROVIDER_CONFIG)
# 2. Project-local config (.creator/provider.json)
# 3. Global config (~/.config/pmpo-skill-creator/provider.json)
# 4. MCP state server (if connected)
# 5. Memory tool (if available)
# 6. Filesystem fallback (always available)

if [[ -n "${CREATOR_PROVIDER_CONFIG:-}" ]] && [[ -f "$CREATOR_PROVIDER_CONFIG" ]]; then
  cat "$CREATOR_PROVIDER_CONFIG"
  exit 0
fi

if [[ -f ".creator/provider.json" ]]; then
  cat ".creator/provider.json"
  exit 0
fi

GLOBAL_CONFIG="${HOME}/.config/pmpo-skill-creator/provider.json"
if [[ -f "$GLOBAL_CONFIG" ]]; then
  cat "$GLOBAL_CONFIG"
  exit 0
fi

# Fallback to filesystem provider
echo '{"provider_type": "filesystem", "config": {"state_directory": ".creator", "scope": "project"}}'
