# Domain Adapter: Software

Assessment and evolution knowledge for software development projects.

## Assessment Criteria

### Build Health
- **Build status**: Does the project compile/build without errors?
- **Warning count**: How many compiler/linter warnings exist?
- **Dependency freshness**: Are dependencies up to date? Any security advisories?
- **Build time**: Is compilation fast enough for the development workflow?

### Test Health
- **Test count**: How many tests exist?
- **Pass rate**: What percentage of tests pass?
- **Coverage**: What percentage of code is covered by tests?
- **Test types**: Unit, integration, end-to-end — are all levels represented?

### Code Quality
- **Lint violations**: How many linter errors/warnings?
- **Code smells**: Dead code, unused imports, complex functions
- **Documentation**: Are public APIs documented? README current?
- **Architecture**: Is the module structure clean? Circular dependencies?

### Specification Compliance
- **Spec coverage**: What percentage of specified features are implemented?
- **Protocol compliance**: Does the implementation match protocol specifications?
- **API surface**: Are all documented endpoints implemented and tested?

### Deployment Readiness
- **CI/CD**: Is there an automated build/test/deploy pipeline?
- **Containerization**: Can it be deployed as a container?
- **Configuration**: Are configs externalized? Secrets managed?
- **Monitoring**: Are health checks and logging in place?

## Analysis Criteria

### Competitive Landscape
- Search for: `"{project_type} frameworks {language} {year}"`
- Search for: `"{protocol_name} implementations comparison"`
- Search for: `"{technology} alternatives comparison"`

### Ecosystem Changes
- Dependency ecosystem updates (new major versions, deprecations)
- Language/runtime updates (new features, breaking changes)
- Protocol/standard revisions

### Benchmarks
- Compare against similar projects by: stars, contributors, release cadence, feature set
- Compare architecture choices: monolith vs. microservice, sync vs. async, etc.

## Planning Patterns

### Quick Wins
- Fix compiler warnings
- Update documentation
- Add missing test cases
- Fix linter violations

### Strategic Improvements
- Architectural refactors
- New protocol support
- Performance optimization
- CI/CD pipeline setup

## Execution Tools

| Action | Tool |
|---|---|
| Build check | `bash` → `cargo check`, `npm run build`, `go build`, etc. |
| Run tests | `bash` → `cargo test`, `npm test`, `pytest`, etc. |
| Lint check | `bash` → `cargo clippy`, `eslint`, `golangci-lint`, etc. |
| Code changes | File editing tools |
| Dependency update | `bash` → package manager commands |
| Documentation | File creation/editing |

## Health Indicator Thresholds

| Indicator | Healthy | Warning | Critical |
|---|---|---|---|
| Build errors | 0 | 1-5 | >5 |
| Test pass rate | >95% | 80-95% | <80% |
| Lint violations | 0 | 1-10 | >10 |
| Doc coverage | >80% | 50-80% | <50% |
| Dependency age | <6 months | 6-12 months | >12 months |
