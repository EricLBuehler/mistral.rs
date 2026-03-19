#!/usr/bin/env bash
# validate-manifest.sh — PostToolUse hook for artifact manifest validation
# Runs after Write/Edit operations to ensure manifest stays valid
set -euo pipefail

MANIFEST="artifact_manifest.json"
SCHEMA="references/schemas/artifact-manifest.schema.json"

# Only run if manifest exists (skip during initial setup)
if [ ! -f "$MANIFEST" ]; then
  echo "ℹ️  No manifest yet — skipping validation"
  exit 0
fi

# Check valid JSON
if ! python3 -c "import json; json.load(open('$MANIFEST'))" 2>/dev/null; then
  echo "❌ artifact_manifest.json is not valid JSON" >&2
  exit 2
fi

# Check required fields if schema exists
if [ -f "$SCHEMA" ]; then
  python3 -c "
import json, sys
with open('$MANIFEST') as f:
    manifest = json.load(f)
with open('$SCHEMA') as f:
    schema = json.load(f)
required = schema.get('required', [])
missing = [k for k in required if k not in manifest]
if missing:
    print(f'❌ Missing required fields: {missing}', file=sys.stderr)
    sys.exit(2)
print('✅ Manifest structure valid')
" 2>&1
fi

# Check file references and preview integrity
python3 -c "
import json, os, sys

with open('$MANIFEST') as f:
    manifest = json.load(f)

errors = []
warnings = []
refs = []

for variant in manifest.get('variants', []):
    name = variant.get('name', '<unnamed>')
    if isinstance(variant.get('file'), str):
        refs.append((name, variant['file']))
    if isinstance(variant.get('files'), list):
        for ref in variant['files']:
            if isinstance(ref, str):
                refs.append((name, ref))

preview = manifest.get('preview', {}) if isinstance(manifest.get('preview', {}), dict) else {}
runs = preview.get('runs', [])
if not isinstance(runs, list):
    errors.append('preview.runs must be an array when preview exists')
    runs = []

for run in runs:
    aid = run.get('artifact_id', 'unknown')
    for key in ('html', 'screenshot', 'report'):
        ref = run.get(key)
        if ref:
            refs.append((f'preview:{aid}', ref))

missing = []
for owner, ref in refs:
    if not os.path.exists(ref):
        missing.append(f'{owner}:{ref}')
if missing:
    errors.append(f'Missing referenced files: {missing}')

artifact_type = manifest.get('artifact_type')
preview_required = preview.get('required')
if preview_required is None:
    preview_required = artifact_type in {'ui', 'a2ui'}

if preview_required:
    if not runs:
        errors.append('Preview is required but preview.runs is empty')
    for run in runs:
        aid = run.get('artifact_id', 'unknown')
        for key in ('html', 'screenshot', 'report'):
            if not run.get(key):
                errors.append(f'Preview run {aid} missing field: {key}')

if warnings:
    print(f'⚠️  {warnings}', file=sys.stderr)
if errors:
    print(f'❌ {errors}', file=sys.stderr)
    sys.exit(2)
print('✅ Manifest file references and preview metadata checked')
" 2>&1

exit 0
