## Context

The current Artifact Refiner workflow declares browser preview as an optional capability, but execution and validation are not wired to consistently prove browser render success for `ui` and `a2ui` artifacts. This gap is most visible for React TSX outputs (which require compilation before browser execution) and HTMX outputs (which may require runtime script availability). We need deterministic, repeatable preview generation that produces machine-checkable outcomes and durable artifacts (screenshots + metadata) in `dist/`.

Constraints:
- Must preserve PMPO ordering and artifact-centric state persistence.
- Must degrade gracefully when optional browser tools are unavailable.
- Must avoid requiring live network access for core success paths when possible.
- Must integrate with existing hooks, manifest validation, and executor/validator roles.

## Goals / Non-Goals

**Goals:**
- Add deterministic browser preview rendering for `ui` and `a2ui` refinement outputs.
- Support React TSX preview by compiling TSX to browser-loadable JS before render.
- Support HTMX preview rendering with explicit runtime loading policy.
- Capture PNG screenshots and preview diagnostics (console errors, render status, timings).
- Persist preview outputs in `dist/` and include them in manifest/log state.

**Non-Goals:**
- Building a full dev server or hot-reload system for interactive development.
- Replacing existing image/logo/content refinement execution paths.
- Introducing a hard requirement on a single external MCP server for all environments.
- Adding visual regression diffing in this change (future enhancement).

## Decisions

### 1. Introduce a dedicated browser preview execution stage for `ui` and `a2ui`
Decision: Add a `render_preview` deterministic stage in planning/execution for applicable artifact types.

Rationale: This makes preview generation explicit in PMPO plans and easy to validate in Reflect/Persist. It also aligns with termination criteria based on measurable outputs.

Alternatives considered:
- Keep preview implicit in domain docs only: rejected because it is not enforceable.
- Run preview only in hooks: rejected because hooks should validate, not own core generation.

### 2. Implement dual-path renderer selection with graceful fallback
Decision: Use this order of execution:
1. `browser_renderer` tool (if available in runtime)
2. Local Playwright script fallback (`scripts/render-preview.mjs`)
3. Soft-fail with explicit log and unsatisfied constraint if preview is required

Rationale: The skill runs across different host environments with varying tool availability. A dual path preserves portability and deterministic behavior.

Alternatives considered:
- Require only `browser_renderer`: rejected due environment variability.
- Require only local Playwright: rejected because some hosts already provide better-integrated browser tooling.

### 3. Add TSX compilation pipeline before browser rendering
Decision: Compile React TSX artifacts into `dist/previews/react/` bundle output before render using `esbuild` (or equivalent deterministic bundler), then load a generated preview HTML entry.

Rationale: TSX cannot be directly loaded in browser execution. A compile step makes preview deterministic and reproducible.

Alternatives considered:
- Runtime transpilation in browser (Babel standalone): rejected due nondeterministic network/runtime complexity.
- Full framework build (Vite/Next): rejected as too heavy for artifact-level preview.

### 4. Add explicit HTMX runtime policy with offline-first default
Decision: For HTMX previews, default to local runtime resolution (`assets/vendor/htmx.min.js` or copied runtime into `dist/previews/vendor/`). Allow network fetch only when a constraint/flag explicitly permits it.

Rationale: Network-free rendering improves reproducibility and CI reliability while still allowing fallback behavior when explicitly requested.

Alternatives considered:
- Always fetch from CDN: rejected due flaky/offline execution.
- Never allow network: rejected because some artifacts intentionally reference remote dependencies.

### 5. Standardize preview evidence outputs and manifest extensions
Decision: Add structured preview outputs:
- `dist/previews/<artifact-id>/preview.html`
- `dist/previews/<artifact-id>/screenshot.png`
- `dist/previews/<artifact-id>/preview-report.json` (render status, console entries, errors, timing, viewport)

Extend manifest schema to include preview artifacts and validation metadata, and update validator checks to confirm required preview assets exist for `ui|a2ui` when preview constraints are enabled.

Rationale: Persistent evidence is required for convergence and post-run auditability.

Alternatives considered:
- Log-only preview evidence: rejected because image/report files are needed for downstream review.
- Screenshot without metadata: rejected because failures become hard to diagnose.

### 6. Keep browser automation as optional MCP extension, with bundled defaults
Decision: Keep `.mcp.json` extensible for additional renderer-related servers, but ship default local implementation scripts so the feature works without extra MCP setup.

Rationale: Minimizes setup friction while preserving advanced integration options.

Alternatives considered:
- MCP-only implementation: rejected due setup burden.
- Script-only implementation: rejected because it ignores existing host-provided render tools.

## Risks / Trade-offs

- [Playwright browser binary availability may be missing in some environments] → Mitigation: detect and surface actionable error; document installation path; allow `browser_renderer` path.
- [TSX compilation may fail for framework-specific imports] → Mitigation: constrain preview entry contract, provide alias/stub strategy, and log unresolved imports in report.
- [Offline HTMX mode may diverge from production CDN behavior] → Mitigation: record runtime source in preview report and allow explicit network-enabled mode.
- [Manifest schema change could break existing validations] → Mitigation: introduce backward-compatible optional fields first, then enforce preview-required constraints only for relevant artifact types.
- [Screenshot generation adds execution time] → Mitigation: single viewport default, optional multi-viewport mode, and bounded timeout settings.

## Migration Plan

1. Update planning/execution prompts and domain references to add explicit `render_preview` stage and outputs.
2. Add renderer scripts (`scripts/render-preview.mjs`, `scripts/compile-tsx-preview.mjs`) and supporting helper utilities.
3. Update executor/validator agent guidance to include preview generation and checks.
4. Extend manifest schema + `scripts/validate-manifest.sh` and post-execute checks for preview artifacts.
5. Add sample outputs in an example refinement (UI or A2UI) to validate end-to-end behavior.
6. Document setup/fallback behavior in `README.md` and `SKILL.md`.
7. Rollback strategy: disable preview-required constraint and revert to existing non-preview flow while retaining generated artifacts.

## Open Questions

- Should preview rendering be mandatory by default for all `ui` and `a2ui` runs, or gated by a constraint flag?
- What baseline viewport(s) should be standardized for screenshots (single desktop vs desktop+mobile)?
- Do we want to vendor HTMX runtime in-repo now, or generate/cache it during first run?
- Should preview report include accessibility checks now (axe) or defer to a separate capability?
- Which additional MCP servers (if any) should be officially supported in `.mcp.json` beyond e2b?
