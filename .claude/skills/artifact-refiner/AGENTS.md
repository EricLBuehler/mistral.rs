# Contributing to Artifact Refiner

Guidelines for contributors working on this repository.

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new domain adapter for diagrams
fix: remove hardcoded brand references from logo.md
docs: update README quickstart section
refactor: consolidate phase controller examples
chore: update marketplace.json version
```

## Branch Strategy

- `main` — Stable, release-ready
- `feat/*` — Feature branches
- `fix/*` — Bug fix branches

## Pull Request Process

1. Create a feature branch from `main`
2. Make changes following the development guidelines in `CLAUDE.md`
3. Run validation: `bash scripts/validate-marketplace.sh`
4. Verify all SKILL.md files have YAML frontmatter
5. Ensure no cross-reference links are broken
6. Submit PR with clear description of changes

## Code Review Checklist

- [ ] YAML frontmatter present on all `.md` files in `skills/` and `agents/`
- [ ] No duplicated content across docs (README, CLAUDE.md, SKILL.md)
- [ ] All `references/` and `assets/` paths resolve to real files
- [ ] New domains added to meta-controller routing table
- [ ] Hook scripts are executable (`chmod +x`)
- [ ] Examples updated if behavior changed
- [ ] `plugin.json` updated if new skills/agents added

## Architecture Overview

See `CLAUDE.md` for the technical architecture and development guidelines.
See `README.md` for project overview and quickstart.
See `SKILL.md` for the canonical skill definition.
