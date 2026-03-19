---
name: {{skill_name}}
description: >
  {{skill_description}}
{{#if allowed_tools}}allowed-tools: {{allowed_tools}}{{/if}}
---

# {{skill_name_title}}

{{skill_overview}}

## Inputs

```yaml
# Define inputs for this skill
{{inputs_yaml}}
```

## Outputs

```yaml
# Define outputs for this skill
{{outputs_yaml}}
```

## Execution Model

{{#if pmpo_phases}}
### PMPO Loop

{{phase_descriptions}}

### Phase Hooks

After each phase: checkpoint + dispatch.
{{/if}}

{{#if state_management}}
## Persistent State Files

- `.{{state_dir}}/{{state_subdir}}/<name>/state.json` — Named state
- `.{{state_dir}}/registry.json` — State registry
{{/if}}

## Quality Standards

{{quality_standards}}

## Quick Start

{{quick_start_commands}}
