# Plan Phase — PMPO Skill Creator

You are the Plan phase controller. Design the complete file architecture for the skill being created.

## Objective

Produce a detailed file map and component design that the Execute phase can follow to generate every file.

## Inputs

Receive `skill_spec` from Specify phase.

## Procedure

### Step 1: Map Core Architecture

Based on complexity tier and mode, determine the full file tree:

#### Simple Tier
```
<skill_name>/
├── SKILL.md
├── CLAUDE.md
└── AGENTS.md
```

#### Standard Tier
```
<skill_name>/
├── SKILL.md
├── CLAUDE.md
├── AGENTS.md
├── prompts/
│   ├── meta-controller.md
│   ├── specify.md
│   ├── plan.md
│   ├── execute.md
│   ├── reflect.md
│   └── persist.md
├── agents/
│   └── <role>.md (per planned agent)
├── references/
│   ├── <domain>.md (per domain adapter)
│   └── schemas/
│       └── <state>.schema.json
├── scripts/
│   └── validate.sh
└── skills/
    └── <command>/SKILL.md (per sub-skill)
```

#### Full Tier
```
<skill_name>/
├── SKILL.md
├── CLAUDE.md
├── AGENTS.md
├── prompts/
│   ├── meta-controller.md
│   ├── specify.md
│   ├── plan.md
│   ├── execute.md
│   ├── reflect.md
│   └── persist.md
├── agents/
│   └── <role>.md (per planned agent)
├── references/
│   ├── <domain>.md (per domain adapter)
│   ├── pmpo-theory.md
│   ├── state-management.md
│   └── schemas/
│       ├── <state>.schema.json
│       └── <output>.schema.json
├── scripts/
│   ├── state-resolve-provider.sh
│   ├── state-init.sh
│   ├── state-checkpoint.sh
│   ├── state-finalize.sh
│   ├── workflow-dispatch.sh
│   └── validate.sh
├── hooks/
│   └── hooks.json
├── assets/
│   └── templates/ (if generating sub-artifacts)
├── skills/
│   └── <command>/SKILL.md (per sub-skill)
├── .claude-plugin/    (if claude-code platform)
│   └── plugin.json
└── <tools_directory>/ (if opencode platform)
    └── <tool>.ts
```

### Step 2: Design Domain Adapters

For each planned domain, define:
- Purpose and scope
- Domain-specific evaluation criteria
- Key constraints and quality measures
- Example inputs/outputs

### Step 3: Design Agent Roles

For each planned agent, define:
- Role name and responsibility
- Which phases it operates in
- What tools it needs access to
- Input/output contracts

### Step 4: Design Schemas

Define JSON schemas for:
- **State schema** — Creation/runtime state manifest
- **Output schema** — Skill output contract

Reference patterns from exemplar skills:
- `references/schemas/creation-state.schema.json` (this skill's state)
- Exemplar: `evolution-state.schema.json` (evolver) or `refinement-state.schema.json` (refiner)

### Step 5: Plan Hooks Configuration

If hooks are required (standard/full tier):
- Per-phase `SubagentStop` hooks for checkpoint + dispatch
- `Stop` hook for finalization + completion dispatch
- Map each hook to the correct script path

### Step 6: Plan Platform Outputs

For each target platform:

| Platform | Files to Generate |
|----------|-------------------|
| agentskills-io | `SKILL.md` (compliant frontmatter) |
| claude-code | `.claude-plugin/plugin.json`, `skills/`, `agents/`, `hooks/` |
| opencode | `<tools_directory>/` with tool definitions |

### Step 7: Clone/Extend Adaptation (if applicable)

For `clone`:
- Map source files → target files (1:1 with name substitution)
- Identify domain-specific content sections to replace
- List files that need structural changes vs. name-only changes

For `extend`:
- List new files to add
- List existing files that need modification
- Verify non-destructive additions (no renames, no deletions)

## Output Contract

```yaml
skill_plan:
  file_map:
    - path: string         # Relative path within skill directory
      purpose: string      # What this file does
      template: string     # Which template to use (or "custom")
      source_file: string  # For clone: source file to adapt from
  agents:
    - name: string
      role: string
      phases: string[]
  domains:
    - name: string
      scope: string
      criteria: string[]
  schemas:
    - name: string
      purpose: string
      key_fields: string[]
  hooks:
    - event: string
      phase: string
      action: string
  platform_outputs:
    - platform: string
      files: string[]
```

## Rules

1. Every file in file_map must have a clear purpose
2. Template references must match actual template files in `assets/templates/`
3. Schema designs must follow JSON Schema draft-07
4. Hook configurations must use `${CLAUDE_PLUGIN_ROOT}` for script paths
5. Clone file maps must account for 100% of source files
