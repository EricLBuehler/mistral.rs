---
description: Generate BDD integration tests with Cucumber.js, Playwright browser validation, and video recording. Creates .feature files, step definitions, and hooks following the project's CustomWorld pattern.
---

# Generate BDD Integration Tests

Read `.agent/skills/bdd-testing/SKILL.md` for the full skill instructions. This command provides the same BDD test generation capability adapted for Claude Code.

## Quick Reference

### What This Command Does

1. Analyzes the feature or page you want to test
2. Generates a Gherkin `.feature` file in `tests/features/{layer}/`
3. Generates TypeScript step definitions in `tests/steps/`
4. Optionally adds new hooks to `tests/support/hooks.ts`

### Usage

Provide context about what to test:

```
/project:bdd-test Create UI tests for the login page with video recording
/project:bdd-test Add API tests for the /api/users endpoint
/project:bdd-test Test the agent chat workflow for the support bot
```

### Test Layers

- `@api` — REST endpoint testing via Playwright `request` API
- `@ui` — Browser E2E testing via Playwright `page` API
- `@agent` — AI agent orchestration testing
- `@video` — Enables Playwright video recording (combine with `@ui`)

### Project Conventions

- Feature files: `tests/features/{api,ui,agents,system}/*.feature`
- Step definitions: `tests/steps/*.steps.ts`
- World object: `tests/support/world.ts` (`CustomWorld` class)
- Hooks: `tests/support/hooks.ts`
- Config: `cucumber.js` (profiles: `default`, `api`, `ui`, `agents`, `video`)
- Videos: `tests/reports/videos/` (WebM format)

### Running Tests

```bash
pnpm test:bdd           # All tests
pnpm test:bdd:api       # API tests only
pnpm test:bdd:ui        # UI tests only
pnpm test:bdd:agents    # Agent tests only
pnpm test:bdd:video     # UI tests with video
```

### Key Rules

- Use `async function (this: CustomWorld)` in step definitions (no arrow functions)
- Use `data-testid` selectors for UI elements
- Use Cucumber Expressions (`{string}`, `{int}`) over regex
- Use Playwright `expect` from `@playwright/test`
- One feature per file, one behavior per scenario
- Declarative steps: `When I log in as "admin"` not `When I click username and type...`
