Image Refiner Module

Domain

Generated, Edited, or Derived Image Artifacts

Purpose

Refine image-based artifacts into production-ready, constraint-compliant outputs using PMPO orchestration.

This module supports:
	•	Cover images
	•	Social media graphics
	•	UI illustrations
	•	Brand assets
	•	Marketing visuals
	•	Derived raster exports from vector sources

⸻

Inputs
	•	specification (from Specify phase)
	•	plan (from Plan phase)
	•	existing image file(s) (optional)

⸻

Responsibilities
	1.	Generate image prompts (if required)
	2.	Refine composition instructions
	3.	Enforce brand color constraints
	4.	Resize images to required dimensions
	5.	Convert file formats (PNG, JPG, WebP, SVG)
	6.	Validate resolution and aspect ratios
	7.	Validate contrast and accessibility when applicable
	8.	Persist final image variants

⸻

Deterministic Execution Triggers

Invoke code_interpreter when:
	•	Batch resizing images
	•	Converting formats
	•	Stripping metadata
	•	Calculating contrast ratios
	•	Checking file size limits
	•	Verifying dimensions
	•	Applying overlays or compositing
	•	Generating image manifests

⸻

Common Constraints
	•	Required output dimensions (e.g., 1200x630, 1080x1080)
	•	File size thresholds (e.g., < 500KB)
	•	Brand color palette adherence
	•	Background transparency preservation
	•	WCAG contrast compliance (if text present)
	•	Aspect ratio enforcement

⸻

Validation Checklist

During Execute and Reflect phases, validate:
	•	All required output sizes exist
	•	Dimensions match specification
	•	File sizes within threshold
	•	Format matches expected extension
	•	No corrupted or zero-byte files

⸻

Expected Outputs
	•	Final image files in required formats
	•	Resized variant set
	•	Optional composited versions
	•	artifact_manifest.json updated
	•	Validation report entries

⸻

Reflection Focus
	•	Visual composition quality
	•	Brand alignment
	•	Technical correctness
	•	Output consistency across variants

⸻

Termination Conditions

Refinement ends when:
	•	All required image variants generated
	•	Deterministic validations pass
	•	Constraints satisfied
	•	No structural or quality regressions remain