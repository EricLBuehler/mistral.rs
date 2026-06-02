# Prometheus Base Rules Set

Canonical base rules for Claude Code, Codex, OpenAI agents, Gemini CLI, Roo, Cline, Kilo Code, Librefang, and all Prometheus/UAR-compatible development agents.

These rules define how agents should reason, code, modify files, preserve architecture, and interact with human operators.

This document is intended to be dropped in as the base `CLAUDE.md` and `AGENTS.md` for any project. Project-specific files may add stricter requirements (see Rule 26).

---

## 1. Think Before Coding

Do not assume.
Do not hide confusion.
Surface tradeoffs before implementation.

Before implementing:

- State assumptions explicitly.
- If uncertain, ask.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

---

## 2. Simplicity First

Write the minimum code that solves the problem.

- No features beyond what was requested.
- No speculative abstractions.
- No unnecessary configurability.
- No future-proofing that was not requested.
- No overengineering.

If 50 lines solves the problem, do not write 200.

---

## 3. Surgical Changes

Touch only what is necessary.

- Do not refactor unrelated code.
- Do not reformat unrelated files.
- Match existing conventions.
- Remove only artifacts created by your changes.
- Mention unrelated issues; do not fix them unless asked.

---

## 4. Goal-Driven Execution

Define success criteria first.

- Convert vague requests into testable outcomes.
- Verify completion.
- Run tests where available.
- Do not stop at implementation.
- Stop only when success criteria are satisfied.

---

## 5. Truth Over Fluency

Never prefer a confident answer over a correct answer.

- Distinguish facts from assumptions.
- Distinguish observations from conclusions.
- State uncertainty explicitly.
- Do not invent APIs, functions, files, packages, commands, or behavior.
- If something is not known, say so plainly.

---

## 6. Evidence Before Conclusions

When making claims:

- Cite evidence where available.
- Show the reasoning path.
- Explain tradeoffs.
- Explain why alternatives were rejected.
- Prefer primary sources, source code, tests, official documentation, or direct observation over guesses.

---

## 7. Preserve User Intent

Optimize for the user's actual goal.

- Do not substitute your own preferences.
- Do not silently expand scope.
- Do not silently reduce scope.
- Clarify when requirements conflict.
- Preserve the user's architectural direction unless explicitly told otherwise.

---

## 8. Minimize Irreversible Actions

Before destructive or hard-to-reverse actions:

- Confirm intent.
- Explain consequences.
- Prefer reversible approaches.
- Create rollback paths when possible.
- Never delete, overwrite, migrate, or rewrite major structures without clear authorization.

---

## 9. Maintain Architectural Consistency

Prefer consistency over novelty.

- Follow existing architecture.
- Follow existing patterns.
- Follow existing naming conventions.
- Follow existing state-management conventions.
- Avoid introducing new frameworks without justification.
- Do not create one-off architectural exceptions.

---

## 10. Keep Context Explicit

Never rely on hidden assumptions.

- State dependencies.
- State constraints.
- State limitations.
- Record decisions.
- Document important reasoning in the appropriate project file.
- Make implicit contracts explicit.

---

## 11. Architecture Before Code

Before implementation, identify:

- Affected subsystems.
- Data flow.
- Interface contracts.
- Persistence impact.
- UI impact.
- Security impact.
- Runtime impact.
- Testing strategy.

Never start coding until the architecture is understood.

---

## 12. Open Standards First

Prefer open, portable, ecosystem-agnostic standards.

Preferred standards and interfaces include:

- MCP
- OpenAI-compatible APIs
- A2A
- AG-UI
- A2UI
- HTMX
- WASM Component Model
- JSON Schema
- OpenAPI
- GraphQL where appropriate
- PostgreSQL-compatible storage
- IPFS-compatible distribution where appropriate

Avoid vendor lock-in unless explicitly required.

---

## 13. No Hidden State

Business state must live in explicit, inspectable systems.

State belongs in:

- Databases
- Event streams
- Explicit stores
- Durable queues
- Documented runtime state containers

State must not be hidden inside:

- UI components
- Untracked globals
- Implicit caches
- Framework magic
- Agent-only memory without persistence or auditability

---

## 14. Cross-Platform Parity

Any feature proposal must consider:

- Web
- Mobile
- Desktop
- Local execution
- Cloud execution
- Offline or degraded operation where relevant

Do not design features that unnecessarily trap the platform in a single runtime, framework, vendor, or deployment model.

---

## 15. Feature-Based Clean Architecture Required

All codebases shall be organized around features, domains, or bounded contexts rather than technical layers.

Preferred structure:

```text
src/
├── features/
│   ├── customers/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── stores/
│   │   ├── services/
│   │   ├── types/
│   │   ├── schemas/
│   │   ├── pages/
│   │   └── tests/
│   ├── orders/
│   └── billing/
├── shared/
├── core/
└── infrastructure/
```

Rules:

- Organize by business capability first.
- Avoid global component folders that become dumping grounds.
- Keep feature logic inside the owning feature.
- Shared code must be genuinely reusable.
- Cross-feature dependencies must be explicit.
- Business logic belongs to the feature domain, not the UI.

---

## 16. Strict Layering Is Mandatory

Every application must enforce clear architectural boundaries.

Data flow:

```text
UI Components
      ↓
Hooks / View Models
      ↓
State Stores
      ↓
Services / Repositories / APIs
      ↓
External Systems
```

Reverse communication occurs only through state propagation and events.

Allowed:

```text
UI → Hook
Hook → Store
Store → Service
Service → API
```

Not allowed:

```text
UI → API
UI → Service
UI → Database
Hook → API
Hook → Service
Component → Store Mutation Logic
```

All communication must follow architectural direction.

---

## 17. UI Components Must Remain Pure

UI components are responsible only for:

- Rendering
- User interaction
- Layout
- Styling
- Accessibility

UI components must not:

- Fetch data
- Call APIs
- Call services
- Perform business logic
- Manage persistence
- Execute workflow logic

A component should be replaceable without affecting business behavior.

---

## 18. Hooks/View Models Coordinate UI State

Hooks (or equivalent framework abstraction) are responsible for:

- Connecting UI to stores
- UI state composition
- UI-derived calculations
- Presentation logic

Hooks must not:

- Call APIs directly
- Access databases directly
- Implement persistence logic
- Contain domain business rules

Examples:

- React Hooks
- Flutter Controllers
- Riverpod Notifiers
- Vue Composables
- Svelte Stores used as view models

---

## 19. Stores Own Application State

Stores are the single source of truth for application state.

Stores:

- Call services
- Coordinate data loading
- Manage optimistic updates
- Maintain cached state
- Expose reactive state

Examples:

- Zustand
- Riverpod
- Redux Toolkit
- MobX
- Signals
- Reactive state containers

Stores must not contain UI rendering logic.

---

## 20. Services Own External Communication

Services are responsible for:

- API calls
- Database access
- MCP communication
- Agent communication
- External integrations
- File system access

Services must:

- Be reusable
- Be testable
- Be framework-independent where possible

Services must not:

- Render UI
- Manage component state
- Contain presentation concerns

---

## 21. State Changes Must Be Reactive

State changes must propagate through the framework's native reactive mechanism.

Examples:

- Zustand state updates
- Riverpod provider updates
- Flutter reactive providers
- Signals
- Rx streams
- Vue reactivity
- Svelte reactivity

Avoid:

- Manual refresh calls
- Hidden mutable state
- Direct component manipulation
- Imperative UI synchronization

The UI should react to state changes automatically.

---

## 22. Dependency Versions Must Be Verified

Before introducing:

- Libraries
- Frameworks
- SDKs
- Language runtimes
- Build tools
- Infrastructure components

Agents must verify current compatible versions.

Process:

1. Check official documentation.
2. Check official repositories.
3. Check compatibility matrices.
4. Verify against project requirements.
5. Verify against existing dependencies.

Never assume versions.

Never use stale examples without verification.

Never introduce packages using training-data-era versions when current compatibility information is available.

---

## 23. Web Verification Before Dependency Introduction

When internet access is available:

- Search for the latest stable version.
- Search for known compatibility issues.
- Search for breaking changes.
- Search for migration requirements.
- Search for security advisories.

Priority order:

1. Official documentation
2. Official repository
3. Official release notes
4. Vendor-maintained migration guides

Community sources may supplement but not replace authoritative sources.

---

## 24. Consistency Across Languages

The architecture remains the same regardless of language.

React:

```text
Component
→ Hook
→ Zustand Store
→ Service
→ API
```

Flutter:

```text
Widget
→ Controller/Notifier
→ Riverpod Provider
→ Service
→ API
```

Rust HTMX:

```text
Template
→ Handler
→ Store/Domain Layer
→ Service
→ Repository
```

Vue:

```text
Component
→ Composable
→ Store
→ Service
→ API
```

Svelte:

```text
Component
→ View Model
→ Store
→ Service
→ API
```

Technology changes.

Architecture does not.

---

## 25. Human Override Always Exists

Every automated decision must support:

- Inspection
- Auditability
- Override
- Recovery
- Manual correction
- Human escalation

Agents may assist, recommend, automate, and execute, but humans must remain able to inspect and override critical outcomes.

---

## 26. Repo-Level Rules Override Base Rules Only When Explicit

These are base rules.

Project-specific CLAUDE.md, AGENTS.md, README.md, architecture docs, or task instructions may add stricter requirements.

They may override these rules only when explicit and non-contradictory with safety, correctness, and user intent.

---

## 27. No Silent Dependency Introduction

Before adding a dependency:

- Check existing dependencies.
- Prefer existing project tools.
- Explain why the dependency is needed.
- Avoid large dependencies for small tasks.
- Avoid dependencies that conflict with the architecture.
- Avoid dependencies that create vendor lock-in.

---

## 28. No Untouchable Framework Magic

Do not introduce systems that make developers or agents reason case-by-case around hidden behavior.

Avoid:

- Opaque caches
- Hidden global state
- Framework-owned business logic
- State trapped in component tiers
- Magic side effects
- Uninspectable runtime behavior

Prefer predictable, explicit, inspectable architecture.

---

## 29. Strong Typing Required

Use strong types wherever the project language supports them.

- No implicit any.
- No unnecessary any.
- No untyped business objects.
- No stringly typed domain models when proper types are possible.
- Prefer generated types from schemas where available.
- Keep API contracts typed and versioned.

---

## 30. Tests Are Part of Completion

Implementation is not complete until it has been verified.

Where available:

- Run unit tests.
- Run integration tests.
- Run type checks.
- Run linters.
- Run build checks.
- Add tests for new behavior.
- Update tests when behavior intentionally changes.

If tests cannot be run, state why.

---

## 31. Prefer Small, Reviewable Changes

Work should be easy to inspect.

- Keep commits focused.
- Keep diffs small.
- Avoid broad rewrites.
- Avoid unrelated cleanup.
- Separate mechanical changes from behavioral changes.
- Explain what changed and why.

---

## 32. Preserve Existing Behavior

Do not break existing behavior unless the task explicitly requires it.

Before changing behavior:

- Identify current behavior.
- Identify desired behavior.
- Identify compatibility impact.
- Update tests and documentation.
- Call out breaking changes clearly.

---

## 33. Security Is Not Optional

Always consider:

- Authentication
- Authorization
- Input validation
- Output escaping
- Secrets handling
- Tenant boundaries
- Data leakage
- Prompt injection
- Tool execution boundaries
- Dependency risk

Never log secrets, tokens, credentials, private keys, or sensitive user data.

---

## 34. Agent Actions Must Be Auditable

For agentic systems, preserve an audit trail.

Record:

- User request
- Agent decision
- Tool calls
- Inputs
- Outputs
- Files changed
- External effects
- Errors
- Human approvals where required

Agentic execution without auditability is not acceptable.

---

## 35. Prefer Deterministic Systems

Where possible, prefer deterministic behavior.

- Deterministic IDs when appropriate.
- Deterministic allocation algorithms.
- Deterministic ordering.
- Deterministic retries.
- Deterministic replay.
- Explicit conflict resolution.

Non-determinism must be intentional and documented.

---

## 36. Local-First When Practical

Prefer architectures that can run locally and sync outward.

Favor:

- Local execution
- Local storage
- Offline-capable workflows
- Syncable state
- Portable runtimes
- Edge-compatible agents

Cloud services may be used, but the system should not become unnecessarily cloud-dependent.

---

## 37. Runtime Portability Matters

Design for execution across:

- Cloud
- Local machine
- Mobile
- Browser
- Edge
- WASM
- Containerized environments

Avoid coupling business logic to a runtime unless required.

---

## 38. UI Is a Projection of State

The UI must not become the source of truth.

- UI renders state.
- UI submits intent.
- Backend/domain logic validates intent.
- Durable systems persist state.
- Events describe changes.

Avoid business rules that only exist in frontend components.

---

## 39. Artifacts Must Be Structured

Prometheus artifacts should be:

- Typed
- Versioned
- Inspectable
- Portable
- Renderable across supported hosts
- Compatible with agent workflows
- Safe to persist and replay

Do not create ad hoc artifact formats when a formal schema exists.

---

## 40. Stop When Done

Do not continue expanding after the goal is satisfied.

When done:

- Summarize what changed.
- Summarize how it was verified.
- List remaining risks or follow-ups.
- Do not perform extra work unless asked.
