# Exemplar Skills Analysis

Structural analysis of the two production skills that serve as templates for the creator.

## iterative-evolver (48 files)

**Location**: `universal-agent-runtime/.agent/skills/iterative-evolver/`  
**Purpose**: Domain-agnostic evolution engine — iteratively improve anything from software to business processes.

### File Inventory

| Category | Files | Names |
|----------|-------|-------|
| Core | 4 | `SKILL.md`, `CLAUDE.md`, `AGENTS.md`, `README.md` |
| Prompts | 7 | `meta-controller`, `assess`, `analyze`, `plan`, `execute`, `reflect`, `persist` |
| Agents | 5 | `assessor`, `analyst`, `planner`, `executor`, `reflector` |
| References | 3 | `pmpo-theory.md`, `state-management.md`, `workflow-integration.md` |
| Domains | 8 | `software`, `business`, `compliance`, `content`, `generic`, `operations`, `product`, `research` |
| Schemas | 5 | `assessment`, `evolution-state`, `hook-event`, `state-provider`, `workflow-trigger` |
| Scripts | 8 | `state-resolve-provider`, `state-init`, `state-checkpoint`, `state-finalize`, `workflow-dispatch`, `finalize-session`, `log-reflection`, `validate-state` |
| Sub-skills | 7 | `evolve`, `evolve-analyze`, `evolve-assess`, `evolve-execute`, `evolve-plan`, `evolve-report`, `evolve-status` |
| Hooks | 1 | `hooks.json` |

### Architecture Pattern

- **7-phase PMPO**: Assess → Analyze → Plan → Execute → Reflect → Persist + Report
- **Named evolutions**: Cross-session state with `evolution_name`
- **8 domain adapters**: Software, business, compliance, content, generic, ops, product, research
- **State provider**: 6-tier resolution (env → project → global → MCP → memory → filesystem)
- **Workflow triggers**: Event-driven dispatch on phase/cycle completion

---

## artifact-refiner (79 files)

**Location**: `GQAdonis/artifact-refiner-skill` (standalone repo)  
**Purpose**: Artifact-centric refinement — iteratively improve logos, UIs, code, content, prompts.

### File Inventory

| Category | Files | Names |
|----------|-------|-------|
| Core | 5 | `SKILL.md`, `CLAUDE.md`, `AGENTS.md`, `README.md`, `LICENSE` |
| Prompts | 6 | `meta-controller`, `specify`, `plan`, `execute`, `reflect`, `persist` |
| Agents | 4 | `pmpo-specifier`, `pmpo-planner`, `pmpo-executor`, `pmpo-reflector`, `artifact-validator` |
| References | 2 | `pmpo-theory.md`, `content-types.md` |
| Domains | 7 | `a2ui`, `code`, `content`, `image`, `logo`, `meta-prompt`, `ui` |
| Schemas | 7 | `artifact-manifest`, `constraints`, `content-type`, `hook-event`, `refinement-state`, `state-provider`, `workflow-trigger` |
| Scripts | 5 | `state-resolve-provider`, `state-init`, `state-checkpoint`, `state-finalize`, `workflow-dispatch` |
| Sub-skills | 7 | `refine-a2ui`, `refine-ui`, `refine-image`, `refine-validate`, `refine-status`, `refine-logo`, `refine-content` |
| Hooks | 1 | `hooks.json` |
| Assets | 6 | Templates (a2ui preview, content report, logo showcase, react shadcn, vendor) |
| Examples | 3 | `content-refinement`, `logo-refinement`, `ui-preview-refinement` |
| Docs | 5 | Articles, audits, releases, SDK research |

### Architecture Pattern

- **6-phase PMPO**: Specify → Plan → Execute → Reflect → Persist + Meta-controller
- **Named artifacts**: Cross-session state with `artifact_name`
- **Content types**: `direct:*` (output IS the artifact) vs `meta:*` (output is a prompt)
- **7 domain adapters**: A2UI, code, content, image, logo, meta-prompt, UI
- **State provider**: 6-tier resolution (same as evolver)
- **Workflow triggers**: Same dispatch pattern

---

## Common Patterns (Template Extraction)

Both skills share these patterns that should be reproduced in generated skills:

### 1. SKILL.md Structure
```
frontmatter → overview → inputs/outputs → execution model → quality standards → quick start
```

### 2. Phase Controller Structure
```
objective → procedure (numbered steps) → output contract (YAML) → rules
```

### 3. Agent Structure
```
role → responsibility → operating phases → tools → input/output → decision criteria
```

### 4. State Scripts
```
state-resolve-provider.sh → state-init.sh → state-checkpoint.sh → state-finalize.sh
```

### 5. Hooks Pattern
```json
{
  "hooks": [
    { "event": "SubagentStop", "match_agent": "...", "steps": [checkpoint, dispatch] },
    { "event": "Stop", "steps": [finalize, dispatch_complete] }
  ]
}
```

### 6. Sub-skill Pattern
```
skills/<command>/SKILL.md with frontmatter pointing to parent phase
```

### 7. Domain Adapter Pattern
```
purpose → scope → evaluation criteria → quality measures → constraints → examples
```

## Feature Matrix

| Feature | Evolver | Refiner | Required in Generated? |
|---------|---------|---------|----------------------|
| SKILL.md (agentskills.io) | ✅ | ✅ | Always |
| PMPO loop | ✅ (7-phase) | ✅ (6-phase) | Standard/Full |
| Named state | ✅ | ✅ | Full |
| State provider | ✅ (6-tier) | ✅ (6-tier) | Full |
| Domain adapters | ✅ (8) | ✅ (7) | Standard/Full |
| JSON schemas | ✅ (5) | ✅ (7) | Standard/Full |
| Lifecycle hooks | ✅ | ✅ | Standard/Full |
| Workflow triggers | ✅ | ✅ | Full |
| Sub-skills | ✅ (7) | ✅ (7) | Standard/Full |
| Examples | ❌ | ✅ (3) | Optional |
| Content types | ❌ | ✅ | Domain-specific |
| Templates/assets | ❌ | ✅ (6) | Domain-specific |
