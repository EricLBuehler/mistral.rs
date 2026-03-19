#!/usr/bin/env bash
# finalize-session.sh — Stop hook for session cleanup
# Ensures all state files are consistent before session ends
set -euo pipefail

echo "🏁 Finalizing refinement session..."

# Check for state file consistency
has_issues=false

if [ -f "artifact_manifest.json" ]; then
  if ! python3 -c "import json; json.load(open('artifact_manifest.json'))" 2>/dev/null; then
    echo "⚠️  artifact_manifest.json is corrupted" >&2
    has_issues=true
  fi
fi

if [ -f "constraints.json" ]; then
  if ! python3 -c "import json; json.load(open('constraints.json'))" 2>/dev/null; then
    echo "⚠️  constraints.json is corrupted" >&2
    has_issues=true
  fi
fi

if [ "$has_issues" = true ]; then
  echo "⚠️  Session ended with state file issues — review before next session" >&2
else
  echo "✅ Session finalized — all state files consistent"
fi

exit 0
