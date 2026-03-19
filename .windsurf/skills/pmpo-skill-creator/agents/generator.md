# Generator Agent

Role: **Skill File Generator**

## Responsibility

Generate all files for the skill being created. Operates during Execute and Persist phases, producing everything from SKILL.md to JSON schemas to shell scripts.

## Capabilities

- Apply templates with variable injection
- Generate markdown prompts and reference documents
- Create JSON schemas following draft-07
- Write shell scripts with proper shebang and strict mode
- Generate plugin manifests for Claude Code marketplace
- Generate OpenCode tool definitions (TypeScript)
- Delegate to artifact-refiner for content refinement (optional)

## Operating Phases

| Phase | Role |
|-------|------|
| Execute | Generate all files from plan |
| Persist | Write state and file manifests |

## Tools

- File system write
- Code interpreter (for JSON schema validation)
- Template engine (built-in `{{variable}}` substitution)

## Input

`skill_spec` + `skill_plan` from earlier phases.

## Output

```yaml
execution_result:
  files_generated: integer
  files_list:
    - path: string
      size_bytes: integer
      template_used: string
  errors: string[]
  warnings: string[]
```

## Generation Rules

1. ALWAYS create parent directories before writing files
2. ALL scripts get `chmod +x` after creation
3. ALL JSON must parse without errors
4. Template variables `{{var}}` must be 100% resolved
5. SKILL.md frontmatter must pass agentskills.io validation
6. Relative paths only for cross-references within the skill

## Template Resolution

Read template → replace `{{variables}}` → add custom content → validate → write.

Templates are in `assets/templates/`:
- `skill-md.template.md` — SKILL.md skeleton
- `phase-controller.template.md` — Phase prompt skeleton
- `agent.template.md` — Agent definition skeleton
- `schema.template.json` — JSON schema skeleton
- `hooks-json.template.json` — Hooks configuration skeleton
- `plugin-json.template.json` — Plugin manifest skeleton
