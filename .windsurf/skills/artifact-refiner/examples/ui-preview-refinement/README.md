# UI Preview Refinement Example

This example demonstrates browser preview evidence generated for a UI artifact refinement run.

## Scenario

**User request**: "Refine a React card component and provide browser-render proof with screenshot evidence."

## Preview Outputs

- `dist/previews/card/preview.html` — Renderable preview document
- `dist/previews/card/screenshot.png` — Captured browser screenshot
- `dist/previews/card/preview-report.json` — Render diagnostics + runtime metadata

## Manifest Integration

See [artifact_manifest.json](./artifact_manifest.json) for how preview evidence is persisted in `preview.runs` and `variants`.
