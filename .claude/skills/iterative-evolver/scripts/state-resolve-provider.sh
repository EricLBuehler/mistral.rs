#!/usr/bin/env bash
# state-resolve-provider.sh — Resolves the active state provider using 6-tier waterfall
# Output: JSON provider configuration to stdout
# Exit 0 = resolved, Exit 1 = error

set -euo pipefail

resolve_provider() {
  # Tier 1: Environment variable
  if [ -n "${EVOLVER_PROVIDER_CONFIG:-}" ] && [ -f "$EVOLVER_PROVIDER_CONFIG" ]; then
    echo "$(cat "$EVOLVER_PROVIDER_CONFIG")"
    return 0
  fi

  # Tier 2: Project-local config
  if [ -f ".evolver-provider.json" ]; then
    echo "$(cat ".evolver-provider.json")"
    return 0
  fi

  # Tier 3: Global config
  local global_config="${HOME}/.evolver/provider.json"
  if [ -f "$global_config" ]; then
    echo "$(cat "$global_config")"
    return 0
  fi

  # Tier 4: MCP state tool probe
  # Check if an evolution-state MCP server is configured
  if [ -f ".mcp.json" ]; then
    local has_state_server
    has_state_server=$(python3 -c "
import json, sys
try:
    cfg = json.load(open('.mcp.json'))
    servers = cfg.get('mcpServers', {})
    for name in servers:
        if 'state' in name.lower() or 'evolution' in name.lower():
            print(json.dumps({'provider_type': 'mcp_tool', 'config': {'server_name': name}}))
            sys.exit(0)
except: pass
sys.exit(1)
" 2>/dev/null) && {
      echo "$has_state_server"
      return 0
    }
  fi

  # Tier 5: Agent memory probe
  # Check if a memory MCP server is available
  if [ -f ".mcp.json" ]; then
    local has_memory
    has_memory=$(python3 -c "
import json, sys
try:
    cfg = json.load(open('.mcp.json'))
    servers = cfg.get('mcpServers', {})
    for name in servers:
        if 'memory' in name.lower():
            print(json.dumps({'provider_type': 'agent_memory', 'config': {'memory_server': name, 'entity_prefix': 'evolution'}}))
            sys.exit(0)
except: pass
sys.exit(1)
" 2>/dev/null) && {
      echo "$has_memory"
      return 0
    }
  fi

  # Tier 6: Filesystem fallback (always available)
  echo '{"provider_type": "filesystem", "config": {"state_directory": ".evolver", "scope": "project"}}'
  return 0
}

resolve_provider
