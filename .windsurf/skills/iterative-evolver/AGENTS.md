# Contributing to Iterative Evolver

Guidelines for contributors working on this repository.

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new domain adapter for healthcare
fix: correct convergence logic in reflect phase
docs: update README with operations domain example
refactor: extract common assessment patterns
chore: update marketplace.json version
```

## Branch Strategy

- `main` — Stable, release-ready
- `feat/*` — Feature branches
- `fix/*` — Bug fix branches

## Pull Request Process

1. Create a feature branch from `main`
2. Make changes following the development guidelines in `CLAUDE.md`
3. Verify all SKILL.md files have YAML frontmatter
4. Ensure no cross-reference links are broken
5. Submit PR with clear description of changes

## Code Review Checklist

- [ ] YAML frontmatter present on all `.md` files in `skills/` and `agents/`
- [ ] No domain-specific logic in core prompts (must be in domain adapters)
- [ ] All `references/` paths resolve to real files
- [ ] New domains added to meta-controller routing table
- [ ] Hook scripts are executable (`chmod +x`)
- [ ] `plugin.json` updated if new skills/agents added
- [ ] README updated with new domain examples if applicable
- [ ] JSON schemas validate with draft-07

## Architecture Overview

See `CLAUDE.md` for the technical architecture and development guidelines.
See `README.md` for project overview and quickstart.
See `SKILL.md` for the canonical skill definition.
