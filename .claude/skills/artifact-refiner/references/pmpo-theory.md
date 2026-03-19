Prometheus Meta‑Prompting Orchestration (PMPO)

Overview

Prometheus Meta‑Prompting Orchestration (PMPO) is a structured cognitive orchestration methodology for designing, refining, and executing complex goals using AI systems.

PMPO is not a prompt template.
It is not a single agent pattern.
It is not a chain-of-thought wrapper.

It is a meta‑level control architecture that governs how thinking, planning, execution, and reflection occur across iterative refinement cycles.

Within this repository, PMPO is operationalized through the Artifact Refiner Skill, which transforms conversational refinement into reproducible infrastructure.

⸻

Why PMPO Exists

Traditional AI usage suffers from systemic limitations:
	•	Context window overflow
	•	Loss of refinement history
	•	Non‑deterministic iteration
	•	Hallucinated outputs
	•	Lack of reproducibility
	•	Fragile conversational state

When creating complex artifacts (logos, UI systems, specifications, content, image assets), purely conversational workflows fail at the exact moment precision is required.

PMPO was designed to solve this by introducing:
	•	Structured iteration
	•	Persistent state
	•	Explicit constraints
	•	Deterministic tool integration
	•	Convergence rules

It converts ephemeral reasoning into structured refinement infrastructure.

⸻

Core Principles

1. Artifact‑Centric State

The artifact is the source of truth.

State must be written to disk:
	•	Manifests
	•	Constraint files
	•	Logs
	•	Generated outputs

Conversation is not state.
Artifacts are state.

⸻

2. Separation of Cognition and Computation

PMPO explicitly separates:
	•	AI reasoning (analysis, planning, abstraction)
	•	Deterministic execution (code, rendering, validation)

This prevents hallucination and enforces reproducibility.

AI thinks.
Tools transform.
PMPO orchestrates.

⸻

3. Recursive Structured Iteration

All refinement follows the PMPO meta‑loop:
	1.	Specify – Clarify intent, constraints, and target state
	2.	Plan – Decompose into minimal executable stages
	3.	Execute – Perform transformations (AI + tools)
	4.	Reflect – Evaluate against constraints and convergence criteria
	5.	Persist – Update artifact state
	6.	Loop or Terminate

This loop continues until convergence conditions are satisfied.

⸻

4. Constraint‑Driven Convergence

Constraints are structured objects, not informal guidelines.

They:
	•	Define blocking vs non‑blocking rules
	•	Enable deterministic validation
	•	Prevent aesthetic drift
	•	Provide measurable thresholds

Reflection evaluates constraints explicitly and determines whether refinement continues.

⸻

5. Deterministic Validation

If a requirement can be measured, it must be validated.

Examples:
	•	WCAG contrast compliance
	•	JSON schema validity
	•	File existence
	•	Image dimensions
	•	Build success

Deterministic validation is executed via tool integration (e.g., code interpreter).

⸻

PMPO as a Meta‑Architecture

PMPO operates at the meta level.

It does not solve a domain problem directly.
It orchestrates how domain problems are solved.

Within this repository:
	•	Domain modules (logo, UI, A2UI, image, content) implement domain logic.
	•	PMPO governs how those modules execute, refine, and converge.

PMPO is therefore a cognitive orchestration protocol, not a task prompt.

⸻

PMPO in the Context of the Artifact Refiner

The Artifact Refiner Skill is a concrete instantiation of PMPO.

It demonstrates:
	•	State persistence
	•	Manifest contracts
	•	Tool‑augmented execution
	•	Constraint‑driven reflection
	•	Domain abstraction

PMPO provides the theory.
The Artifact Refiner implements the machinery.

⸻

Theoretical Foundations

PMPO draws from multiple conceptual influences:
	•	Systems architecture
	•	Compiler design (spec → plan → execute → validate)
	•	Control theory (feedback loops)
	•	Iterative design refinement
	•	Declarative constraint systems

At a high level, PMPO functions like a compiler:
	•	Intent is parsed
	•	Constraints are structured
	•	Execution plan is generated
	•	Deterministic code runs
	•	Output is validated
	•	Refinement loops until stable

This is refinement as infrastructure.

⸻

Methodology

Step 1: Formalize Intent

Convert ambiguous goals into structured specifications.

Ambiguity must be reduced before execution.

⸻

Step 2: Constrain the Solution Space

Define explicit constraints.

Constraints limit drift and define convergence boundaries.

⸻

Step 3: Decompose Into Minimal Stages

Break transformation into small, deterministic steps.

Avoid monolithic execution.

⸻

Step 4: Execute With Tool Separation

Use AI for reasoning.
Use deterministic tools for transformation.

Never rely on hallucinated outputs.

⸻

Step 5: Reflect Against Structured Criteria

Evaluate:
	•	Constraint compliance
	•	Structural integrity
	•	Deterministic outputs
	•	Regression detection

Reflection decides continuation or termination.

⸻

Step 6: Persist State

All outputs must be persisted.

This enables:
	•	Re‑entry
	•	Versioning
	•	CI integration
	•	Multi‑agent collaboration

⸻

Multi‑Domain Adaptability

PMPO is domain‑agnostic because:
	•	The meta‑loop is invariant
	•	Constraints are structured
	•	Execution is modular

Only domain adapters change.

This makes PMPO applicable to:
	•	Branding
	•	UI systems
	•	Agent specifications
	•	Infrastructure design
	•	Documentation generation

⸻

Preventing Context Collapse

A core motivation for PMPO is preventing catastrophic context loss in long refinement sessions.

By externalizing state to artifacts and logs, refinement survives:
	•	Token limits
	•	Session resets
	•	Model swaps
	•	Multi‑agent handoffs

PMPO turns fragile chat into resilient stateful orchestration.

⸻

Future Evolution of PMPO

Potential expansions include:
	•	Distributed multi‑agent PMPO graphs
	•	Visual diffing between refinement cycles
	•	Regression detection automation
	•	Versioned artifact lineage tracking
	•	CI/CD blocking constraints

PMPO is intended to evolve into a general orchestration framework for agentic systems.

⸻

Summary

Prometheus Meta‑Prompting Orchestration (PMPO) is a structured cognitive orchestration architecture designed to:
	•	Transform ambiguous goals into structured specifications
	•	Execute deterministic transformations
	•	Persist artifact state
	•	Iterate until convergence
	•	Prevent context‑based failure

Within this repository, PMPO powers the Artifact Refiner Skill — converting conversational iteration into reproducible, production‑grade artifact infrastructure.

PMPO is not about prompting better.
It is about orchestrating thinking itself.