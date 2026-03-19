---
name: clone-skill
description: Clone an existing Agent Skill and adapt it for a new domain. Preserves architecture while replacing domain-specific content.
---

# Clone Skill

Copy the structure of an existing skill (like iterative-evolver or artifact-refiner) and adapt it for a new domain. Preserves PMPO architecture, state management, and hooks while replacing domain-specific content.

## Usage

```
/clone-skill
```

## What You'll Be Asked

1. **Source skill** — Path to the skill to clone
2. **New skill name** — Name for the cloned skill
3. **New domain** — What domain the clone operates in
4. **Adaptations** — What domain-specific content to replace

## How It Works

1. **Analyze** the source skill structure (all files, patterns, references)
2. **Map** source files → target files with name substitution
3. **Identify** domain-specific content sections to replace
4. **Generate** adapted files preserving architectural patterns
5. **Validate** the clone against the same spec as the original

## Example

```
/clone-skill
> Source: .agent/skills/iterative-evolver
> Name: compliance-auditor
> Domain: compliance and regulatory review
```

This produces a skill with the evolver's PMPO architecture but all domain adapters, prompts, and references tailored for compliance auditing.
