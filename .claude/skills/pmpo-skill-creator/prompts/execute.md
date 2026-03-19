# Execute Phase — PMPO Skill Creator

You are the Execute phase controller. Generate all files specified in the skill plan.

## Objective

Produce every file in the `file_map` from the Plan phase, using templates where available and AI generation for custom content.

## Inputs

- `skill_spec` from Specify phase
- `skill_plan` from Plan phase

## Procedure

### Step 1: Prepare Output Directory

Create the skill directory structure:

```bash
mkdir -p dist/<skill_name>/{prompts,agents,references/schemas,scripts,hooks,skills,assets/templates}
```

Add platform directories as needed:
```bash
# Claude Code plugin
mkdir -p dist/<skill_name>/.claude-plugin

# OpenCode tools
mkdir -p dist/<skill_name>/<tools_directory>
```

### Step 2: Generate Core Files

#### SKILL.md
- Use template: `assets/templates/skill-md.template.md`
- Inject: `{{skill_name}}`, `{{skill_description}}`, `{{allowed_tools}}`
- Manually complete: inputs/outputs, execution model, quality standards
- Validate: frontmatter against agentskills.io spec

#### CLAUDE.md
- Generate development guide specific to the new skill
- Include: architecture overview, key files, development guidelines

#### AGENTS.md
- Generate contributor guide with commit conventions

### Step 3: Generate Phase Controllers

For each phase in `skill_plan.file_map` matching `prompts/*.md`:

- Use template: `assets/templates/phase-controller.template.md`
- Inject: `{{phase_name}}`, `{{phase_objective}}`, `{{phase_procedure}}`
- Customize procedure steps based on the skill's domain and purpose
- Include output contracts in YAML

#### Meta-Controller Special Case
- Include startup protocol (provider resolution, state init)
- Include mode routing logic
- Include phase loop with hooks
- Include loop control and finalization

### Step 4: Generate Agent Definitions

For each agent in `skill_plan.agents`:

- Use template: `assets/templates/agent.template.md`
- Inject: `{{agent_name}}`, `{{agent_role}}`, `{{agent_phases}}`, `{{agent_tools}}`
- Define clear responsibility and output expectations

### Step 5: Generate Reference Materials

For each domain adapter:
- Create `references/<domain>.md` with domain-specific evaluation criteria
- Include: purpose, quality criteria, example inputs/outputs

For theory/architecture docs:
- Generate `references/pmpo-theory.md` if full tier (use this skill's `references/pmpo-theory.md` as a starting point)
- Include state management concepts within pmpo-theory.md if state management is required

### Step 6: Generate JSON Schemas

For each schema in `skill_plan.schemas`:

- Use template: `assets/templates/schema.template.json`
- Define properties, required fields, types
- Validate: `python3 -c "import json; json.load(open('file'))"`

### Step 7: Generate Scripts

For each script:
- Set shebang: `#!/usr/bin/env bash`
- Set strict mode: `set -euo pipefail`
- Adapt from exemplar skill scripts (evolver/refiner)
- Make executable: `chmod +x scripts/*.sh`

State lifecycle scripts (full tier):
- `state-resolve-provider.sh` — 6-tier provider resolution
- `state-init.sh` — Init/resume named state
- `state-checkpoint.sh` — Mid-phase snapshot
- `state-finalize.sh` — Archive completed state
- `workflow-dispatch.sh` — Event-driven triggers

Validation script (all tiers):
- `validate.sh` — Skill-specific validation suite

### Step 8: Generate Hooks Configuration

Use template: `assets/templates/hooks-json.template.json`

Generate per-phase hooks:
```json
{
  "hooks": [
    {
      "event": "SubagentStop",
      "match_agent": "<phase>_agent",
      "steps": [
        { "type": "command", "command": "bash ${CLAUDE_PLUGIN_ROOT}/scripts/state-checkpoint.sh ..." },
        { "type": "command", "command": "bash ${CLAUDE_PLUGIN_ROOT}/scripts/workflow-dispatch.sh ..." }
      ]
    }
  ]
}
```

### Step 9: Generate Sub-Skills

For each sub-skill in `skill_plan.file_map` matching `skills/*/SKILL.md`:
- Create SKILL.md with proper frontmatter
- Include: name, description, instructions, examples

### Step 10: Generate Plugin Manifest (claude-code platform)

Use template: `assets/templates/plugin-json.template.json`

```json
{
  "name": "{{skill_name}}",
  "description": "{{skill_description}}",
  "version": "1.0.0"
}
```

### Step 11: Generate OpenCode Tools (opencode platform)

If `tools_directory` is specified, generate tool definitions:

```typescript
// <tools_directory>/<tool_name>.ts
import { tool } from "@opencode/tool";
import { z } from "zod";

export default tool({
  name: "{{tool_name}}",
  description: "{{tool_description}}",
  parameters: z.object({ /* ... */ }),
  execute: async (args) => { /* ... */ }
});
```

### Step 12: Artifact-Refiner Delegation (Optional)

If the artifact-refiner skill is available AND the user wants refined prompt content:

1. Identify markdown/prompt files that would benefit from iterative refinement
2. Pass each to artifact-refiner with:
   - `content_type: meta:agent-prompt`
   - `artifact_name: <skill_name>-<file_basename>`
3. Use the refined output as the final file content
4. If artifact-refiner is not available, use the directly generated content

## Clone Mode Execution

For `clone` mode, the procedure differs:

1. Copy all source files to `dist/<skill_name>/`
2. For each file marked in the plan:
   - Replace domain-specific terms (e.g., "evolve" → "analyze")
   - Update names, descriptions, and references
   - Preserve structural patterns (PMPO loop, state management)
3. Validate cross-references after all replacements

## Extend Mode Execution

For `extend` mode:

1. Copy existing skill to `dist/<skill_name>/` (or edit in-place)
2. Add new files (domains, schemas, sub-skills) without modifying existing
3. Update SKILL.md to reference new components
4. Update hooks.json to include new phase hooks
5. Validate that existing functionality is preserved

## Output Contract

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

## Rules

1. NEVER generate a file without creating its parent directory first
2. ALL scripts must be made executable with `chmod +x`
3. ALL JSON files must parse without errors
4. SKILL.md frontmatter must include required `name` and `description`
5. Cross-references must use relative paths from skill root
6. Template variables (`{{var}}`) must be fully resolved — no unresolved placeholders in output
