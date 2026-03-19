## Why

UI and A2UI refinement currently describes browser preview validation, but the workflow does not consistently render artifacts in a real browser, especially for React TSX that must be compiled and HTMX views that depend on runtime scripts. Adding deterministic browser rendering and screenshot evidence now will make refinement outputs verifiable, reproducible, and ready for CI-style validation.

## What Changes

- Add a first-class browser preview capability for UI and A2UI artifacts so generated outputs are rendered and validated before convergence.
- Add React TSX preview compilation support so TSX artifacts can be bundled and loaded in browser-renderable HTML.
- Add HTMX preview runtime support with explicit handling of script loading/network-dependent behavior during preview generation.
- Add screenshot capture outputs (PNG and metadata) from rendered artifacts to provide durable visual evidence in `dist/`.
- Add validation/reporting of browser execution signals (render success, console errors, and preview artifacts) into refinement logs and manifest outputs.
- Add optional tooling extension points (MCP/tool configuration) for environments that require dedicated browser automation or sandbox execution.

## Capabilities

### New Capabilities
- `browser-preview-rendering`: Render generated UI/A2UI artifacts in a browser context and record render outcomes.
- `react-tsx-preview-compilation`: Compile React TSX artifact outputs into browser-loadable preview bundles.
- `htmx-preview-runtime-support`: Support HTMX preview rendering with required client-side runtime/script handling.
- `artifact-screenshot-capture`: Capture browser screenshots of refined artifacts and persist image outputs for review.
- `browser-preview-validation-reporting`: Persist browser preview validation signals in logs and artifact manifest metadata.

### Modified Capabilities
- None.

## Impact

- Affected areas: `prompts/execute.md`, `prompts/plan.md`, `references/domain/ui.md`, `references/domain/a2ui.md`, executor/validator agent instructions, hook scripts, and manifest/schema expectations.
- New dependencies likely required: browser automation tooling (e.g., Playwright), TSX compilation tooling (e.g., esbuild or equivalent), and optional runtime/script fetch strategy for HTMX previews.
- MCP/tooling impact: may require extending `.mcp.json` and/or providing bundled local defaults for browser rendering and screenshot capture paths.
