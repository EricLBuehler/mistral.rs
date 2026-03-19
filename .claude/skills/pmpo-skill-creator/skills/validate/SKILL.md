---
name: validate-skill
description: Validate any Agent Skill against the agentskills.io specification and quality standards. Reports compliance issues and suggestions.
---

# Validate Skill

Run the full validation suite against any skill directory. Checks agentskills.io spec compliance, JSON schema validity, cross-reference integrity, script quality, and completeness.

## Usage

```
/validate-skill
```

## What You'll Be Asked

1. **Skill path** — Directory containing the skill to validate

## What Gets Checked

| Category | Checks |
|----------|--------|
| **Spec compliance** | Frontmatter, name/description fields, line count |
| **JSON schemas** | Parse validity, required fields, types |
| **Cross-references** | All referenced files exist |
| **Scripts** | Executable, shebang, strict mode, syntax |
| **Hooks** | Valid JSON, correct event types, script paths resolve |
| **Plugin** | Valid manifest with required fields |
| **Sub-skills** | Each has valid SKILL.md with frontmatter |

## Output

A validation report with:
- Per-check pass/fail/warn status
- Overall quality score (0.0 - 1.0)
- Specific fix instructions for any failures
- Recommendation: valid, needs fixes, or needs restructuring

## Example

```
/validate-skill
> Path: dist/my-new-skill/
```

Or validate an existing installed skill:
```
/validate-skill
> Path: .agent/skills/iterative-evolver/
```

## Automated Script

The validation can also be run directly:
```bash
bash scripts/validate-skill.sh <skill-directory>
```
