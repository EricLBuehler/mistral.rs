# Code Refiner Module

## Domain

Source code artifacts — Python, Rust, TypeScript, Shell, and any other programming language (distinct from UI components, which use `domain/ui.md`).

## Purpose

Refine code artifacts through iterative analysis, restructuring, and validation using deterministic execution and test-driven evaluation.

---

## Inputs

- `specification` (from Specify phase)
- `plan` (from Plan phase)
- Existing source files (optional)
- Test files (optional)

---

## Refinement Focus Areas

### Correctness
- Logic errors and edge cases
- Type safety and null handling
- Error propagation and recovery

### Structure
- Module organization and separation of concerns
- Function/method length and complexity
- Naming conventions and readability

### Quality
- Documentation completeness (docstrings, comments)
- Test coverage for critical paths
- Style/lint compliance for target language

### Performance
- Algorithm efficiency (when specified in constraints)
- Resource usage patterns
- Concurrency correctness (when applicable)

---

## Deterministic Execution Use Cases

Use `code_interpreter` or e2b sandbox for:
- Running linters (`ruff`, `eslint`, `clippy`, `shellcheck`)
- Executing unit tests
- Measuring code complexity metrics
- Checking type annotations (`mypy`, `tsc`)
- Validating build success
- Formatting checks (`prettier`, `rustfmt`, `black`)

---

## Language-Specific Patterns

| Language | Lint | Format | Test | Type Check |
|----------|------|--------|------|------------|
| Python | `ruff check` | `ruff format` | `pytest` | `mypy` |
| TypeScript | `eslint` | `prettier` | `jest` / `vitest` | `tsc --noEmit` |
| Rust | `cargo clippy` | `cargo fmt` | `cargo test` | (built-in) |
| Shell | `shellcheck` | `shfmt` | `bats` | N/A |

---

## Common Constraints

- All tests pass
- No lint warnings at configured severity
- Type checking passes
- Documentation present for public APIs
- Maximum function complexity threshold
- Consistent code style
- No security anti-patterns (hardcoded secrets, SQL injection, etc.)

---

## Expected Outputs

- Refined source files in `dist/`
- Test results summary
- Lint/format report
- Updated `artifact_manifest.json`

---

## Reflection Focus
- Test pass rate and coverage delta
- Lint violation count and severity
- Structural improvement metrics
- Documentation completeness
