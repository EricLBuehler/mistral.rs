#!/usr/bin/env bash
# post-execute-check.sh — SubagentStop hook for pmpo-executor
# Verifies that expected output files were created during execution
set -euo pipefail

MANIFEST="artifact_manifest.json"
LOG="refinement_log.md"

echo "🔍 Post-execution check..."

# Check manifest was updated
if [ ! -f "$MANIFEST" ]; then
  echo "⚠️  artifact_manifest.json not found after execution" >&2
fi

# Check dist/ has content
if [ ! -d "dist" ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
  echo "⚠️  dist/ directory is empty after execution" >&2
fi

# Check log was updated
if [ ! -f "$LOG" ]; then
  echo "⚠️  refinement_log.md not found" >&2
fi

if [ -f "$MANIFEST" ]; then
  python3 - <<'PY'
import json
import os
import sys

manifest_path = "artifact_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

artifact_type = manifest.get("artifact_type")
preview_obj = manifest.get("preview", {})
preview_required = preview_obj.get("required")
if preview_required is None:
    preview_required = artifact_type in {"ui", "a2ui"}

if not preview_required:
    print("ℹ️  Preview evidence not required for this artifact type")
    sys.exit(0)

runs = preview_obj.get("runs", [])
if not runs:
    print("⚠️  Preview required but manifest.preview.runs is empty", file=sys.stderr)
    sys.exit(0)

missing_refs = []
for run in runs:
    for key in ("html", "screenshot", "report"):
        ref = run.get(key)
        if not ref:
            missing_refs.append(f"{run.get('artifact_id', 'unknown')}:{key}=<missing>")
            continue
        if not os.path.exists(ref):
            missing_refs.append(ref)

if missing_refs:
    print(f"⚠️  Missing preview outputs: {missing_refs}", file=sys.stderr)
else:
    print("✅ Preview outputs verified for UI/A2UI execution")
PY
fi

echo "✅ Post-execution check complete"
exit 0
