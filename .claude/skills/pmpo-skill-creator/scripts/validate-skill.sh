#!/usr/bin/env bash
set -euo pipefail

# Validate a generated skill against agentskills.io spec and quality standards
# Usage: validate-skill.sh <skill_directory>

SKILL_DIR="${1:?Usage: validate-skill.sh <skill_directory>}"

if [[ ! -d "$SKILL_DIR" ]]; then
  echo "ERROR: Directory '${SKILL_DIR}' does not exist" >&2
  exit 1
fi

PASS=0
FAIL=0
WARN=0

check_pass() { echo "  ✅ $1"; ((PASS++)); }
check_fail() { echo "  ❌ $1"; ((FAIL++)); }
check_warn() { echo "  ⚠️  $1"; ((WARN++)); }

echo "=== Skill Validation: ${SKILL_DIR} ==="
echo ""

# 1. SKILL.md exists
echo "--- SKILL.md ---"
SKILL_MD="${SKILL_DIR}/SKILL.md"
if [[ -f "$SKILL_MD" ]]; then
  check_pass "SKILL.md exists"
else
  check_fail "SKILL.md missing"
  echo ""
  echo "RESULT: FAIL (${FAIL} failures, ${WARN} warnings, ${PASS} passes)"
  exit 1
fi

# 2. Frontmatter validation
if head -1 "$SKILL_MD" | grep -q "^---"; then
  check_pass "Frontmatter delimiters present"
else
  check_fail "Missing frontmatter (no --- delimiter)"
fi

# Extract frontmatter
FRONTMATTER=$(sed -n '2,/^---$/p' "$SKILL_MD" | head -n -1)

# Check name field
if echo "$FRONTMATTER" | grep -q "^name:"; then
  NAME=$(echo "$FRONTMATTER" | grep "^name:" | head -1 | sed 's/^name: *//')
  if [[ ${#NAME} -le 64 ]]; then
    check_pass "name field valid (${NAME})"
  else
    check_fail "name field too long (${#NAME} > 64)"
  fi
else
  check_fail "name field missing"
fi

# Check description field
if echo "$FRONTMATTER" | grep -q "^description:"; then
  check_pass "description field present"
else
  check_fail "description field missing"
fi

# Line count
LINES=$(wc -l < "$SKILL_MD")
if [[ $LINES -le 500 ]]; then
  check_pass "SKILL.md line count OK (${LINES} ≤ 500)"
else
  check_warn "SKILL.md exceeds 500 lines (${LINES})"
fi

echo ""

# 3. JSON schema validation
echo "--- JSON Schemas ---"
SCHEMA_COUNT=0
SCHEMA_VALID=0
for f in $(find "$SKILL_DIR" -name "*.schema.json" -o -name "*.json" | grep -v node_modules); do
  ((SCHEMA_COUNT++))
  if python3 -c "import json; json.load(open('$f'))" 2>/dev/null; then
    check_pass "$(basename $f)"
    ((SCHEMA_VALID++))
  else
    check_fail "$(basename $f) — invalid JSON"
  fi
done
if [[ $SCHEMA_COUNT -eq 0 ]]; then
  echo "  (no JSON files found)"
fi

echo ""

# 4. Script validation
echo "--- Scripts ---"
for f in $(find "$SKILL_DIR/scripts" -name "*.sh" 2>/dev/null); do
  SCRIPT_NAME=$(basename "$f")
  
  # Check executable
  if [[ -x "$f" ]]; then
    check_pass "${SCRIPT_NAME} executable"
  else
    check_fail "${SCRIPT_NAME} not executable"
  fi
  
  # Check shebang
  if head -1 "$f" | grep -q "^#!/"; then
    check_pass "${SCRIPT_NAME} has shebang"
  else
    check_fail "${SCRIPT_NAME} missing shebang"
  fi
  
  # Check syntax
  if bash -n "$f" 2>/dev/null; then
    check_pass "${SCRIPT_NAME} syntax OK"
  else
    check_fail "${SCRIPT_NAME} syntax errors"
  fi
done
if [[ ! -d "$SKILL_DIR/scripts" ]]; then
  echo "  (no scripts directory)"
fi

echo ""

# 5. Cross-reference integrity
echo "--- Cross-References ---"
REF_TOTAL=0
REF_VALID=0
for ref in $(grep -roh 'references/[a-zA-Z0-9/_.-]*' "$SKILL_DIR/prompts/" "$SKILL_DIR/SKILL.md" 2>/dev/null | sort -u); do
  ((REF_TOTAL++))
  if [[ -e "$SKILL_DIR/$ref" ]]; then
    check_pass "$ref"
    ((REF_VALID++))
  else
    check_fail "$ref — not found"
  fi
done
if [[ $REF_TOTAL -eq 0 ]]; then
  echo "  (no cross-references found)"
fi

echo ""

# 6. Plugin manifest (if present)
echo "--- Plugin ---"
PLUGIN_JSON="$SKILL_DIR/.claude-plugin/plugin.json"
if [[ -f "$PLUGIN_JSON" ]]; then
  if python3 -c "import json; d=json.load(open('$PLUGIN_JSON')); assert 'name' in d; assert 'description' in d" 2>/dev/null; then
    check_pass "plugin.json valid"
  else
    check_fail "plugin.json missing required fields"
  fi
else
  echo "  (no plugin manifest)"
fi

echo ""

# 7. Sub-skills
echo "--- Sub-Skills ---"
for skill_dir in $(find "$SKILL_DIR/skills" -name "SKILL.md" 2>/dev/null); do
  SUBSKILL=$(dirname "$skill_dir" | xargs basename)
  if head -1 "$skill_dir" | grep -q "^---"; then
    check_pass "skills/${SUBSKILL}/SKILL.md"
  else
    check_fail "skills/${SUBSKILL}/SKILL.md — missing frontmatter"
  fi
done
if [[ ! -d "$SKILL_DIR/skills" ]]; then
  echo "  (no sub-skills directory)"
fi

echo ""

# Summary
TOTAL=$((PASS + FAIL + WARN))
echo "=== RESULT ==="
echo "  Passes:   ${PASS}"
echo "  Failures: ${FAIL}"
echo "  Warnings: ${WARN}"
echo "  Total:    ${TOTAL}"
echo ""

if [[ $FAIL -eq 0 ]]; then
  echo "  ✅ SKILL VALID"
  exit 0
else
  echo "  ❌ SKILL INVALID (${FAIL} failures)"
  exit 1
fi
