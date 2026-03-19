A2UI Refiner Module

Domain

A2UI (Agent-to-UI) Specification Artifacts

Purpose

Refine A2UI specifications into valid, normalized, schema-compliant artifacts ready for rendering, distribution, or execution within compatible runtimes.

This module ensures that A2UI artifacts are structurally correct, deterministic, and aligned with defined UI and interaction constraints.

⸻

Inputs
	•	specification (from Specify phase)
	•	plan (from Plan phase)
	•	existing A2UI JSON/YAML specification (optional)

⸻

Responsibilities
	1.	Validate structural integrity of the A2UI spec
	2.	Enforce schema compliance
	3.	Normalize structure and field ordering
	4.	Detect deprecated or invalid properties
	5.	Ensure required fields are present
	6.	Generate preview artifact if required
	7.	Render browser preview with HTMX runtime policy controls
	8.	Capture screenshot and diagnostics report
	9.	Persist normalized spec and preview metadata to disk

⸻

Deterministic Execution Triggers

Invoke code_interpreter when:
	•	Validating against JSON schema
	•	Reformatting JSON/YAML
	•	Generating diff reports
	•	Producing rendered preview HTML
	•	Resolving HTMX runtime source (local/network policy)
	•	Rendering previews and capturing screenshots
	•	Running structural validation scripts

⸻

Common Constraints
	•	Valid JSON or YAML syntax
	•	Schema compliance
	•	Required properties present
	•	Proper component hierarchy
	•	No circular references
	•	No undefined bindings
	•	Naming convention consistency

⸻

Validation Checklist

During Execute and Reflect phases, validate:
	•	Spec parses without error
	•	Schema validation passes
	•	Required fields exist
	•	No deprecated keys used
	•	Preview renders successfully (if applicable)
	•	Preview screenshot exists (if applicable)
	•	Preview report includes runtime source + diagnostics

⸻

Expected Outputs
	•	Normalized A2UI spec file
	•	Validation report
	•	Optional preview artifact (HTML)
	•	Optional preview screenshot (PNG)
	•	Optional preview diagnostics report (JSON)
	•	Updated artifact_manifest.json

⸻

Reflection Focus
	•	Structural correctness
	•	Schema alignment
	•	Forward compatibility
	•	Backward compatibility (if required)
	•	Completeness of specification

⸻

Termination Conditions

Refinement ends when:
	•	Spec validates successfully against schema
	•	All constraints satisfied
	•	No structural violations remain
	•	Preview renders correctly (if required)
	•	Preview evidence artifacts are persisted (if required)
