## 1. Planning and Prompt Updates

- [x] 1.1 Update `prompts/plan.md` to add a `render_preview` stage for `ui` and `a2ui` artifacts.
- [x] 1.2 Update `prompts/execute.md` with deterministic browser preview execution flow and fallback order.
- [x] 1.3 Update `references/domain/ui.md` and `references/domain/a2ui.md` to require preview outputs and diagnostics.

## 2. Browser Preview Execution Scripts

- [x] 2.1 Add `scripts/render-preview.mjs` to open generated preview HTML, capture console diagnostics, and write screenshot/report outputs.
- [x] 2.2 Add `scripts/compile-tsx-preview.mjs` to compile TSX preview inputs into browser-loadable assets.
- [x] 2.3 Add shared script helpers for output path normalization, timeout handling, and JSON report writing.

## 3. HTMX Runtime Support

- [x] 3.1 Add offline-first HTMX runtime resolution logic with local runtime source preference.
- [x] 3.2 Add optional network-enabled runtime mode controlled by explicit preview constraints.
- [x] 3.3 Record HTMX runtime source details in preview report metadata.

## 4. Executor and Validator Integration

- [x] 4.1 Update `agents/pmpo-executor.md` to include browser preview rendering and screenshot capture responsibilities.
- [x] 4.2 Update `agents/artifact-validator.md` to validate preview artifact presence and preview report integrity.
- [x] 4.3 Update `scripts/post-execute-check.sh` to verify required preview outputs for `ui` and `a2ui` runs.

## 5. Manifest and Schema Changes

- [x] 5.1 Extend `references/schemas/artifact-manifest.schema.json` with optional preview artifact references and validation metadata.
- [x] 5.2 Update `scripts/validate-manifest.sh` to validate preview references when preview artifacts are present or required.
- [x] 5.3 Ensure preview artifacts are written to `dist/previews/<artifact-id>/` and included in manifest variants/metadata.

## 6. Tooling and MCP Configuration

- [x] 6.1 Add/declare local Playwright and TSX compilation dependencies required by preview scripts.
- [x] 6.2 Update `.mcp.json` documentation/config guidance for optional renderer-related MCP integrations.
- [x] 6.3 Implement graceful fallback behavior when `browser_renderer` tool or browser binaries are unavailable.

## 7. Documentation and Example Validation

- [x] 7.1 Update `SKILL.md` and `README.md` with browser preview, TSX compile, and HTMX runtime behavior.
- [x] 7.2 Add or update an example refinement to include generated preview HTML, screenshot, and preview report outputs.
- [x] 7.3 Run `bash scripts/validate-marketplace.sh` and confirm change artifacts/hook expectations remain valid.
