#!/usr/bin/env bash
# validate-constraints.sh — Validates constraints.json against schema
set -euo pipefail

CONSTRAINTS="constraints.json"
SCHEMA="references/schemas/constraints.schema.json"

if [ ! -f "$CONSTRAINTS" ]; then
  echo "ℹ️  No constraints file yet — skipping validation"
  exit 0
fi

if ! python3 -c "import json; json.load(open('$CONSTRAINTS'))" 2>/dev/null; then
  echo "❌ constraints.json is not valid JSON" >&2
  exit 2
fi

if [ -f "$SCHEMA" ]; then
  python3 -c "
import json, sys
with open('$CONSTRAINTS') as f:
    constraints = json.load(f)
with open('$SCHEMA') as f:
    schema = json.load(f)
required = schema.get('required', [])
missing = [k for k in required if k not in constraints]
if missing:
    print(f'❌ Missing required fields: {missing}', file=sys.stderr)
    sys.exit(2)
# Check blocking constraints have validation defined
items = constraints.get('constraints', constraints.get('items', []))
for c in items:
    if c.get('severity') == 'blocking' and not c.get('validation'):
        print(f\"⚠️  Blocking constraint '{c.get('id', 'unknown')}' has no validation hook\", file=sys.stderr)
print('✅ Constraints structure valid')
" 2>&1
fi

exit 0
