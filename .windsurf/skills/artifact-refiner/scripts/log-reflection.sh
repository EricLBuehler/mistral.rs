#!/usr/bin/env bash
# log-reflection.sh — SubagentStop hook for pmpo-reflector
# Appends reflection summary to refinement log
set -euo pipefail

LOG="refinement_log.md"
DECISIONS="decisions.md"

echo "📝 Logging reflection results..."

# Verify reflection state files exist
if [ ! -f "$LOG" ]; then
  echo "⚠️  refinement_log.md not found — reflection may not have logged properly" >&2
fi

if [ ! -f "$DECISIONS" ]; then
  echo "⚠️  decisions.md not found — convergence decision may not be recorded" >&2
fi

echo "✅ Reflection logging check complete"
exit 0
