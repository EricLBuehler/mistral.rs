# Contributor Guide

## Commit Conventions

Use conventional commits:
- `feat:` — New templates, platform support, modes
- `fix:` — Template fixes, validation corrections
- `docs:` — Reference updates, SKILL.md changes
- `refactor:` — Internal restructuring

## Branch Strategy

- `main` — Stable, tested
- `feat/*` — Feature branches

## Pull Request Process

1. Create feature branch from `main`
2. Make changes following `CLAUDE.md` guidelines
3. Run validation: `bash scripts/validate-skill.sh dist/test-skill/`
4. Submit PR with description of changes

## Code Review Checklist

- [ ] Templates use `{{variable}}` syntax consistently
- [ ] New files referenced in `SKILL.md` and `prompts/meta-controller.md`
- [ ] JSON schemas validate with `python3 -c "import json; json.load(open('file'))"`
- [ ] Scripts have `#!/usr/bin/env bash` shebang and `set -euo pipefail`
- [ ] Cross-references resolve (no dangling paths)
- [ ] agentskills.io spec compliance maintained

## Architecture References

- `SKILL.md` — Skill functionality and behavior
- `CLAUDE.md` — Development guidelines
- `references/agentskills-spec.md` — Spec compliance reference
- `references/exemplar-skills.md` — Template extraction guidance
