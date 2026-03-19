#!/usr/bin/env bash
# state-resolve-provider.sh — Resolves the active state provider using 6-tier waterfall
# Output: JSON provider configuration to stdout
# Exit 0 = resolved, Exit 1 = error

set -euo pipefail

resolve_provider() {
  # Tier 1: Environment variable
  if [ -n "${REFINER_PROVIDER_CONFIG:-}" ] && [ -f "$REFINER_PROVIDER_CONFIG" ]; then
    cat "$REFINER_PROVIDER_CONFIG"
    return 0
  fi

  # Tier 2: Project-local config
  if [ -f ".refiner-provider.json" ]; then
    cat ".refiner-provider.json"
    return 0
  fi

  # Tier 3: Global config
  if [ -f "$HOME/.refiner/provider.json" ]; then
    cat "$HOME/.refiner/provider.json"
    return 0
  fi

  # Tier 4: MCP state tool probe
  if command -v mcp 2>/dev/null | grep -q "refiner_state" 2>/dev/null; then
    echo '{"provider_type": "mcp", "config": {"server_name": "refiner_state"}}'
    return 0
  fi

  # Tier 5: Agent memory probe (check for surreal_memory or similar)
  if command -v mcp 2>/dev/null | grep -q "memory" 2>/dev/null; then
    echo '{"provider_type": "memory", "config": {"tool_name": "memory", "entity_prefix": "refiner:"}}'
    return 0
  fi

  # Tier 6: Filesystem fallback (always available)
  echo '{"provider_type": "filesystem", "config": {"state_directory": ".refiner", "scope": "project"}}'
  return 0
}

resolve_provider
