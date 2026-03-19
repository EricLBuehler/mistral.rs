# Prometheus Meta-Prompting Orchestration (PMPO) — Iterative Evolution Edition

## Overview

Prometheus Meta-Prompting Orchestration (PMPO) is a structured cognitive orchestration methodology for designing, refining, and executing complex goals using AI systems.

PMPO is not a prompt template.
It is not a single agent pattern.
It is not a chain-of-thought wrapper.

It is a meta-level control architecture that governs how thinking, planning, execution, and reflection occur across iterative refinement cycles.

Within this skill, PMPO is adapted for **iterative evolution** — the process of continuously improving any subject against defined goals in a changing landscape.

---

## Why PMPO Exists

Traditional AI usage suffers from systemic limitations:
- Context window overflow
- Loss of refinement history
- Non-deterministic iteration
- Hallucinated outputs
- Lack of reproducibility
- Fragile conversational state

When evolving complex subjects (codebases, strategies, products, research programs), purely conversational workflows fail at the exact moment precision is required.

PMPO solves this by introducing:
- Structured iteration
- Persistent state
- Explicit goals and constraints
- Domain-adaptive tooling
- Convergence rules with delta tracking

---

## Core Principles

### 1. Goal-Centric State

Goals are the source of truth. All evolution is measured against them.

State must be written to disk:
- Evolution manifests
- Assessment records
- Analysis reports
- Plans and execution logs

Conversation is not state. Artifacts are state.

### 2. Separation of Cognition and Computation

PMPO explicitly separates:
- AI reasoning (assessment, analysis, planning, reflection)
- Deterministic execution (builds, tests, searches, measurements)

AI thinks. Tools transform. PMPO orchestrates.

### 3. Iterative Evolution Loop

All evolution follows the PMPO meta-loop:
1. **Assess** — Where are we now?
2. **Analyze** — What's happening around us?
3. **Plan** — What should we improve?
4. **Execute** — Make the improvements
5. **Reflect** — Did it work? Continue or stop?
6. **Persist** — Save all state
7. **Loop or Terminate**

### 4. Constraint-Driven Convergence

Goals and constraints are structured objects, not informal guidelines. They:
- Define blocking vs. non-blocking criteria
- Enable quantitative delta tracking
- Prevent scope drift
- Provide measurable convergence thresholds

### 5. Domain Agnosticism

PMPO is domain-agnostic because:
- The meta-loop is invariant
- Goals and constraints are structured universally
- Execution is modular and tool-adaptive

Only domain adapters change. This makes PMPO applicable to:
- Software development
- Business strategy
- Product design
- Research
- Content creation
- Operations
- Compliance
- Any structured improvement effort

---

## The Evolution Loop vs. The Refinement Loop

The original PMPO loop (as implemented in the Artifact Refiner) follows:
```
Specify → Plan → Execute → Reflect → Persist → Loop
```

The Evolution adaptation extends this with landscape awareness:
```
Assess → Analyze → Plan → Execute → Reflect → Persist → Loop
```

Key differences:
- **Assess** replaces **Specify** — we measure current reality instead of specifying desired output
- **Analyze** is new — we scan the external world for context
- **Reflect** includes delta tracking — comparing before/after quantitatively
- Goals persist across iterations instead of being redefined each cycle

---

## Preventing Context Collapse

A core motivation for PMPO is preventing catastrophic context loss in long evolution sessions.

By externalizing state to files and logs, evolution survives:
- Token limits
- Session resets
- Model swaps
- Multi-agent handoffs

PMPO turns fragile chat into resilient stateful orchestration.

---

## Summary

PMPO for Iterative Evolution is a structured cognitive orchestration architecture designed to:
- Assess current state against defined goals
- Analyze external landscape for context
- Plan evidence-based improvements
- Execute changes with domain-appropriate tools
- Reflect on results with quantitative delta tracking
- Persist all state for cross-session continuity
- Iterate until convergence

PMPO is not about prompting better. It is about orchestrating evolution itself.
