# Agent Skills Specification Reference

Condensed specification from [agentskills.io](https://agentskills.io/specification). Use this as the ground truth when generating and validating skills.

## Directory Structure

Minimum viable skill:
```
skill-name/
└── SKILL.md          # Required
```

Full skill with optional directories:
```
skill-name/
├── SKILL.md          # Required — metadata + instructions
├── scripts/          # Optional — executable scripts
├── references/       # Optional — reference materials
└── assets/           # Optional — templates, images, data
```

## SKILL.md Format

### Frontmatter (Required)

```yaml
---
name: skill-name
description: A description of what this skill does and when to use it.
---
```

### Optional Frontmatter Fields

| Field | Constraints | Purpose |
|-------|------------|---------|
| `name` | ≤64 chars, lowercase letters + numbers + hyphens, no leading/trailing hyphen | Unique skill identifier |
| `description` | ≤1024 chars, non-empty | What the skill does and when to use it |
| `license` | License name or reference | Licensing info |
| `compatibility` | ≤500 chars | Environment requirements |
| `metadata` | Key-value mapping | Arbitrary metadata |
| `allowed-tools` | Space-delimited list | Pre-approved tools (experimental) |

### Body Content

The markdown body below the frontmatter contains instructions. No specific structure required, but best practices:

- Keep under **500 lines** for efficient context use
- Use progressive disclosure (keep SKILL.md lean, put details in references/)
- Reference external files with relative paths from skill root

## Progressive Disclosure

Structure skills for efficient context use:

1. **Metadata (~100 tokens)**: `name` and `description` loaded at startup for all skills
2. **Instructions (<5000 tokens)**: Full SKILL.md body loaded when skill is activated
3. **Resources (as needed)**: Files in `scripts/`, `references/`, `assets/` loaded on demand

## File References

Use relative paths from skill root:
```markdown
See the [reference guide](references/guide.md) for details.
Run the extraction script: `scripts/extract.py`
```

## Validation Rules

A valid skill MUST:
1. Have a `SKILL.md` file at the root
2. Include YAML frontmatter with `---` delimiters
3. Have a non-empty `name` field matching constraints
4. Have a non-empty `description` field ≤1024 chars
5. Have non-empty body content below frontmatter

## Sub-Skills

Skills can be nested as sub-skills within a parent `skills/` directory:
```
parent-skill/
├── SKILL.md
└── skills/
    ├── sub-command-1/
    │   └── SKILL.md
    └── sub-command-2/
        └── SKILL.md
```

Each sub-skill is a standalone skill with its own `SKILL.md` and optional directories.
