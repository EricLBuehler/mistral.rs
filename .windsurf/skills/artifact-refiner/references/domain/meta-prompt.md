# Meta-Prompt Refiner Module

## Domain

Meta-prompts — text outputs designed to drive other AI systems, image generators, video generators, or orchestration processes.

## Purpose

Refine prompt text into precise, reproducible instructions that consistently produce desired outputs when consumed by a downstream platform.

---

## Content Types Handled

- `meta:image-prompt` — Prompts for DALL-E, Midjourney, Flux, Stable Diffusion
- `meta:video-prompt` — Prompts for Sora, Runway, Kling, Pika
- `meta:agent-prompt` — System + user prompt pairs for AI agents
- `meta:workflow` — Orchestration instructions for multi-step processes

---

## Inputs

- `specification` (from Specify phase)
- `plan` (from Plan phase)
- `target_platform` — Which system will consume the prompt
- Existing prompt text (optional — for iterative refinement)

---

## Refinement Focus Areas

### Prompt Structure
- System vs user prompt separation (for `meta:agent-prompt`)
- Variable injection points (marked with `{{variable}}`)
- Section organization (context → task → constraints → output format)

### Specificity
- Replace vague terms with precise descriptions
- Add negative constraints ("do NOT include...")
- Specify exact output format expectations

### Platform Awareness
- Token/character limits for target platform
- Supported features (e.g., Midjourney parameters, DALL-E style keywords)
- Platform-specific prompt patterns and best practices

### Reproducibility
- Same prompt → consistent output quality
- Explicit seed/style parameters where supported
- Disambiguation of ambiguous instructions

---

## Deterministic Execution Use Cases

Use `code_interpreter` when:
- Counting tokens/characters against platform limits
- Validating JSON structure in structured prompts
- Running prompt linting rules
- Generating test outputs (when `test_generation: true`)

---

## Common Constraints

- Prompt length within platform limits
- Clear output format specification
- No conflicting instructions
- Appropriate detail level for target platform
- Variable injection points properly formatted
- Negative constraints present (what to avoid)
- Platform-specific keywords/parameters used correctly

---

## Evaluation Criteria

### Prompt Quality Assessment
1. **Clarity** — Is each instruction unambiguous?
2. **Completeness** — Are all aspects of the desired output specified?
3. **Consistency** — Do instructions conflict with each other?
4. **Specificity** — Are descriptions precise enough for consistent output?
5. **Platform Fit** — Does the prompt use the target platform's conventions?

### Test Generation Assessment (when enabled)
1. Generate a test artifact using the refined prompt
2. Evaluate the test artifact against the original constraints
3. If test artifact fails constraints, refine the prompt further

---

## Expected Outputs

- Refined prompt text file (`.md` or `.txt`)
- Prompt metadata (target platform, token count, variable map)
- Test artifacts (when `test_generation: true`)
- Updated `artifact_manifest.json`

---

## Reflection Focus
- Prompt precision and clarity
- Platform constraint compliance
- Reproducibility confidence
- Test output quality (when applicable)
