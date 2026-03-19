# Logo Refinement Example

This example demonstrates a complete PMPO refinement loop for a logo artifact.

## Scenario

**User request**: "Create a modern, minimal logo for 'NexaFlow' — a developer API platform. Use deep blue and electric cyan. Needs to work on both light and dark backgrounds."

## Step 1: Specify

The specification phase extracts structured requirements:

```json
{
  "artifact_type": "logo",
  "intent": "Create modern minimal logo for NexaFlow API platform",
  "constraints": [
    {
      "id": "c1",
      "description": "Primary color must be deep blue (#0a1628)",
      "severity": "blocking",
      "type": "brand"
    },
    {
      "id": "c2",
      "description": "Accent color must be electric cyan (#00d4ff)",
      "severity": "blocking",
      "type": "brand"
    },
    {
      "id": "c3",
      "description": "Logo must be legible at 16x16 favicon size",
      "severity": "blocking",
      "type": "visual"
    },
    {
      "id": "c4",
      "description": "Must work on both #ffffff and #0a1628 backgrounds",
      "severity": "blocking",
      "type": "visual"
    },
    {
      "id": "c5",
      "description": "Style should evoke developer tooling and data flow",
      "severity": "high",
      "type": "aesthetic"
    }
  ],
  "target_state": {
    "description": "Complete logo system with light/dark variants",
    "required_outputs": ["svg", "png-set", "showcase-html"]
  }
}
```

## Step 2: Plan

The plan phase creates an execution strategy (see [input-spec.json](./input-spec.json)):

1. Generate SVG logo with NexaFlow wordmark → `dist/nexaflow-logo.svg`
2. Generate SVG icon-only variant → `dist/nexaflow-icon.svg`
3. Rasterize to PNG set (16–512px) → `dist/nexaflow-*.png`
4. Generate dark background variant → `dist/nexaflow-logo-dark.svg`
5. Populate showcase template → `dist/showcase.html`
6. Update manifest → `artifact_manifest.json`

## Step 3: Execute

The executor runs each stage, using code interpreter for PNG rasterization and the image generation tool for SVG creation.

## Step 4: Reflect

The reflector evaluates outputs:
- ✅ c1: Deep blue primary confirmed
- ✅ c2: Electric cyan accent confirmed
- ✅ c3: 16x16 favicon legibility checked
- ✅ c4: Light and dark variants both exist
- ⚠️ c5: Style could be more "flow-like" — suggest iteration

**Decision**: Continue (1 high constraint partially unsatisfied)

## Step 5: Persist

State files updated, iteration logged.

## Iteration 2

Executor refines the icon to incorporate a subtle flow motif. Reflector confirms all constraints satisfied. **Decision**: Terminate.

## Final Output

See [artifact_manifest.json](./artifact_manifest.json) for the complete manifest.
