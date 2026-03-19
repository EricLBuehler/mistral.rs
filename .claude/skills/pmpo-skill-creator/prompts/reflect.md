# Reflect Phase — PMPO Skill Creator

You are the Reflect phase controller. Validate the generated skill against the agentskills.io spec and quality standards.

## Objective

Evaluate the generated skill for completeness, correctness, and spec compliance. Produce a validation report that determines whether to loop or terminate.

## Inputs

- `skill_spec` from Specify phase
- `skill_plan` from Plan phase
- `execution_result` from Execute phase

## Procedure

### Step 1: agentskills.io Spec Validation

Validate `SKILL.md` against the spec:

| Check | Requirement | Severity |
|-------|-------------|----------|
| Frontmatter present | `---` delimiters with YAML | **FAIL** |
| `name` field | ≤64 chars, lowercase + hyphens, no leading/trailing hyphen | **FAIL** |
| `description` field | ≤1024 chars, non-empty | **FAIL** |
| Body content | Non-empty markdown below frontmatter | **FAIL** |
| Line count | ≤500 lines recommended | **WARN** |
| Progressive disclosure | References use relative paths | **WARN** |

### Step 2: JSON Schema Validation

For every `.schema.json` file:

```bash
python3 -c "import json; json.load(open('$f'))"
```

Check:
- Valid JSON syntax
- Has `$schema` field
- Has `type` field
- Required properties are defined
- No circular references

### Step 3: Cross-Reference Integrity

Extract all file references from prompts and SKILL.md:

```bash
grep -roh 'references/[a-zA-Z0-9/_.-]*' prompts/ SKILL.md | sort -u
```

Verify each resolves to an existing file. **FAIL** on any dangling reference.

### Step 4: Script Validation

For every script in `scripts/`:

| Check | Requirement | Severity |
|-------|-------------|----------|
| Shebang line | Starts with `#!/usr/bin/env bash` | **FAIL** |
| Strict mode | Contains `set -euo pipefail` | **WARN** |
| Executable | `[ -x "$f" ]` | **FAIL** |
| No syntax errors | `bash -n "$f"` | **FAIL** |

### Step 5: Hooks Validation (if present)

Validate `hooks/hooks.json`:
- Valid JSON
- Each hook has `event` field
- Script paths use `${CLAUDE_PLUGIN_ROOT}` or relative
- No references to nonexistent scripts

### Step 6: Plugin Manifest Validation (if claude-code)

Validate `.claude-plugin/plugin.json`:
- Has `name` field
- Has `description` field
- Has `version` field (semver)

### Step 7: Completeness Check

Compare generated files against `skill_plan.file_map`:

```yaml
completeness:
  planned: integer      # Files in plan
  generated: integer    # Files on disk
  missing: string[]     # Planned but not generated
  extra: string[]       # Generated but not planned
  coverage: float       # generated / planned
```

**FAIL** if coverage < 100%.

### Step 8: PMPO Loop Integrity (standard/full tier)

Check that the PMPO loop is complete:
- All planned phases have controllers in `prompts/`
- Meta-controller references all phases in correct order
- Each phase controller has: objective, procedure, output contract, rules
- Agent files exist for all referenced agents

### Step 9: State Management Check (full tier)

Verify state lifecycle:
- `state-resolve-provider.sh` — provider resolution
- `state-init.sh` — creates initial state with UUID
- `state-checkpoint.sh` — accepts skill name and phase
- `state-finalize.sh` — archives and updates registry
- State directory structure documented in meta-controller

### Step 10: Quality Score

Aggregate checks into a quality score:

| Category | Weight | Score |
|----------|--------|-------|
| Spec compliance | 30% | pass/fail count |
| Schema validity | 20% | valid/total |
| Cross-references | 15% | resolved/total |
| Script quality | 15% | pass/fail |
| Completeness | 20% | coverage % |

**Pass threshold**: ≥95% weighted score with zero FAILs.

## Validation Script

Run the automated validation suite:

```bash
bash scripts/validate-skill.sh dist/<skill_name>/
```

This script performs Steps 1–9 automatically and outputs a JSON report.

## Output Contract

```yaml
reflection:
  overall_status: pass | fail | warn
  quality_score: float   # 0.0 - 1.0
  checks:
    - category: string
      check: string
      status: pass | fail | warn
      message: string
  missing_files: string[]
  failing_checks: integer
  warning_checks: integer
  recommendation: terminate | loop_execute | loop_plan
  fix_instructions: string[]  # What to fix if looping
```

## Loop Decision

| Condition | Recommendation | Return To |
|-----------|---------------|-----------|
| Zero FAILs, score ≥ 0.95 | `terminate` | Finalize |
| FAILs in generated files only | `loop_execute` | Execute (with fix list) |
| FAILs in architecture/structure | `loop_plan` | Plan (with constraints) |
| 3+ iterations with no progress | `terminate` | Output with warnings |

## Rules

1. NEVER mark a skill as passing if any check is FAIL
2. Include fix_instructions for EVERY failing check
3. Distinguish between fixable (execute-loop) and structural (plan-loop) issues
4. Run `validate-skill.sh` before reporting — manual checks supplement, not replace
