# Content Type Architecture

The Artifact Refiner classifies every artifact by **content type** — a two-part tag that determines how the artifact is produced, evaluated, and persisted.

---

## Taxonomy

Content types follow the pattern `{mode}:{kind}`:

### Direct Types — Output IS the Artifact

| Type | Output | Domain Adapter | Evaluation |
|------|--------|----------------|------------|
| `direct:react` | `.tsx` / `.jsx` components | `domain/ui.md` | Render → screenshot → visual inspection |
| `direct:html` | HTML/HTMX markup | `domain/ui.md` | Render → screenshot → visual inspection |
| `direct:content` | Reports, specs, docs | `domain/content.md` | Structure, tone, completeness |
| `direct:image` | SVG/PNG/WebP files | `domain/image.md` or `domain/logo.md` | Visual quality, dimensions, format |
| `direct:code` | Source files (any lang) | `domain/code.md` | Syntax, tests, lint, conventions |

### Meta Types — Output is a Prompt That DRIVES Another Process

| Type | Output | Domain Adapter | Evaluation |
|------|--------|----------------|------------|
| `meta:image-prompt` | Text prompt for image gen | `domain/meta-prompt.md` | Prompt clarity, specificity, platform fit |
| `meta:video-prompt` | Text prompt for video gen | `domain/meta-prompt.md` | Prompt clarity, temporal coherence |
| `meta:agent-prompt` | System + user prompt pair | `domain/meta-prompt.md` | Instruction clarity, guardrails, scope |
| `meta:workflow` | Orchestration instructions | `domain/meta-prompt.md` | Completeness, step ordering, error handling |
| `meta:composite` | Mixed bundle | Multiple adapters | Per-component evaluation |

---

## Evaluation Strategies

### `output_inspection` (all `direct:*` types)

The Reflect phase evaluates the **generated artifact** directly:
- Does the React component render without errors?
- Does the HTML pass accessibility checks?
- Is the content structured with proper heading hierarchy?
- Does the image meet the specified dimensions and format?

### `prompt_quality` (most `meta:*` types)

The Reflect phase evaluates the **prompt itself**, not what it would produce:
- Is the prompt specific enough to consistently produce the desired output?
- Does it include negative constraints (what to avoid)?
- Does it respect platform-specific limits (token count, supported features)?
- Is the instruction structure clear (system vs user, variable injection points)?

### `test_execution` (`meta:*` types with `test_generation: true`)

The Reflect phase **actually runs** the meta-prompt to generate a test artifact:
- Generate a test output from the prompt
- Evaluate the test output against the original constraints
- Use the test result to refine the prompt further

This is expensive but produces higher-quality meta-prompts.

---

## Content Type Detection

The Specify phase classifies content type using these heuristics:

1. **Explicit**: User says "create a prompt for..." → `meta:*`
2. **File extension**: `.tsx` → `direct:react`, `.html` → `direct:html`
3. **Artifact type mapping**: `logo` → `direct:image`, `ui` → `direct:react` or `direct:html`
4. **Intent keywords**: "generate", "produce", "create" → `direct:*`; "prompt for", "instructions to", "how to make" → `meta:*`
5. **Default**: If ambiguous, ask the user or default to `direct:*`

---

## Meta-Prompt Refinement Flow

When `content_type` starts with `meta:`, the PMPO loop shifts focus:

```
Specify:  Define what the prompt should produce (target output description)
Plan:     Decompose prompt structure (system/user split, variables, constraints)
Execute:  Write the prompt text + optionally generate test output
Reflect:  Evaluate prompt quality (+ test output if test_generation is true)
Persist:  Save prompt text + evaluation results + test artifacts (if any)
```

The key difference: **Execute produces prompt text, not final artifacts.** The prompt IS the artifact.

---

## Composite Content Types

`meta:composite` bundles multiple content types into a single refinement:

```yaml
content_type: "meta:composite"
components:
  - content_type: "meta:image-prompt"
    target_platform: "dall-e-3"
  - content_type: "direct:html"
    output_format: "html"
  - content_type: "meta:agent-prompt"
    target_platform: "claude"
```

Each component is evaluated independently using its own strategy. The Reflect phase produces a per-component assessment and an overall convergence decision.
