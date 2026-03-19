UI Refiner Module

Domain

React / HTML / UI Concept Artifacts

Purpose

Refine UI artifacts into structurally sound, accessible, production-ready interface outputs using PMPO orchestration.

This module handles both conceptual UI artifacts and executable UI codebases.

⸻

Inputs
	•	specification (from Specify phase)
	•	plan (from Plan phase)
	•	existing UI files (optional)

⸻

Responsibilities
	1.	Refine component hierarchy
	2.	Normalize layout to grid system
	3.	Enforce design token usage
	4.	Validate semantic HTML structure
	5.	Validate accessibility compliance
	6.	Compile TSX inputs for browser preview when required
	7.	Render browser previews and capture diagnostics
	8.	Persist preview artifacts and screenshot evidence

⸻

Deterministic Execution Triggers

Invoke code_interpreter when:
	•	Writing or modifying React/HTML files
	•	Running build processes (npm build, vite build, etc.)
	•	Compiling TSX preview bundles
	•	Generating static previews
	•	Rendering previews with browser automation fallback
	•	Capturing preview screenshots and reports
	•	Performing accessibility checks
	•	Running linters or formatters
	•	Validating CSS token usage

⸻

Common Constraints
	•	WCAG AA contrast compliance
	•	Proper semantic tags
	•	ARIA attribute correctness
	•	Design token consistency
	•	Spacing scale normalization
	•	No inline style violations (unless intentional)

⸻

Validation Checklist

During Execute and Reflect phases, validate:
	•	File structure correctness
	•	Successful build (if applicable)
	•	Preview HTML exists in `dist/previews/`
	•	Screenshot exists in `dist/previews/`
	•	Preview report JSON exists and is parseable
	•	No console errors
	•	No missing imports
	•	Accessibility thresholds met

⸻

Expected Outputs
	•	Refined UI files
	•	Build output (if applicable)
	•	Preview HTML artifact
	•	Preview screenshot (PNG)
	•	Preview diagnostics report (JSON)
	•	Validation logs
	•	Updated artifact_manifest.json

⸻

Reflection Focus
	•	Visual hierarchy clarity
	•	Structural soundness
	•	Accessibility compliance
	•	Token adherence
	•	Build stability

⸻

Termination Conditions

Refinement ends when:
	•	All constraints satisfied
	•	Build succeeds without errors
	•	Preview artifacts and diagnostics exist (if required)
	•	Accessibility checks pass
	•	No structural regressions remain
