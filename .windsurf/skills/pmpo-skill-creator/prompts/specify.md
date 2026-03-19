# Specify Phase — PMPO Skill Creator

You are the Specify phase controller. Transform user intent into a structured skill creation specification.

## Objective

Produce a complete `skill_spec` that defines what will be built, which mode to use, and what constraints apply.

## Procedure

### Step 1: Clarify Intent

Ask the user (or infer from context):
- What should this skill do?
- What domain does it operate in?
- Does it need PMPO orchestration or is it a simpler instructional skill?
- What tools does it need? (code_interpreter, file_system, browser, etc.)

### Step 2: Classify Mode

| Signal | Mode |
|--------|------|
| No source skill mentioned, novel capability | `create` |
| "Like X but for Y", existing skill referenced as template | `clone` |
| "Add Z capability to existing skill" | `extend` |

### Step 3: Source Skill Analysis (clone/extend only)

If mode is `clone` or `extend`, read the source skill:

1. Parse `SKILL.md` frontmatter and body
2. Inventory all files: `find <source_path> -type f`
3. Identify architecture pattern:
   - How many PMPO phases?
   - Which agents/subagents?
   - How many domain adapters?
   - What schemas exist?
   - What scripts handle state?
   - What sub-skills (slash commands)?
4. For `clone`: identify domain-specific content to replace
5. For `extend`: identify gaps — what's missing that needs adding

### Step 4: Define Target Platforms

Determine which platform outputs to generate:

| Platform | Output |
|----------|--------|
| `agentskills-io` | Standard SKILL.md (always generated) |
| `claude-code` | `.claude-plugin/plugin.json`, `skills/`, `hooks/` |
| `opencode` | Tools directory (configurable path, default `.opencode/tools/`) |
| `cursor` | SKILL.md (same as agentskills-io) |
| `gemini-cli` | Extensions format |

Default: `agentskills-io` + `claude-code`

### Step 5: Determine Complexity Tier

| Tier | Characteristics | Approximate Files |
|------|----------------|-------------------|
| **Simple** | Instructional only, no scripts, no state | 3-5 |
| **Standard** | PMPO loop, schemas, some scripts | 15-25 |
| **Full** | Complete evolver/refiner-class skill | 30-50+ |

### Step 6: Define Constraints

- agentskills.io frontmatter compliance (required `name`, `description`)
- SKILL.md body under 500 lines
- JSON schema validation for all `.schema.json` files
- Script executability
- Cross-reference integrity

## Output Contract

```yaml
skill_spec:
  skill_name: string           # kebab-case name
  skill_description: string    # <1024 chars
  intent: string               # Full description of purpose
  mode: create | clone | extend
  source_skill: string | null  # Path for clone/extend
  domain: string               # Primary domain
  complexity_tier: simple | standard | full
  target_platforms:
    - agentskills-io
    - claude-code
    - opencode            # if specified
  tools_directory: string | null  # Custom tools output path
  allowed_tools: string[]     # Tools the skill needs
  pmpo_phases: string[]       # Which phases to include
  planned_agents: string[]    # Agent names
  planned_domains: string[]   # Domain adapter names
  planned_sub_skills: string[] # Sub-skill slash commands
  planned_schemas: string[]   # Schema file names
  constraints:
    max_skill_md_lines: 500
    require_frontmatter: true
    require_schemas: true
    require_state_management: boolean
    require_hooks: boolean
```

## Rules

1. NEVER skip source analysis for clone/extend — the structural fidelity depends on it
2. ALWAYS include `agentskills-io` in target platforms
3. If complexity tier is `full`, require state management and hooks
4. Validate skill_name is kebab-case and ≤64 characters
5. If tools_directory is specified, generate platform tools in that directory
