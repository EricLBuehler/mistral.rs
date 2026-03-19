#!/usr/bin/env bash
# Cursor / Cursor Agent installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Cursor"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Cursor / Cursor Agent.

Cursor uses .mdc rule files in .cursor/rules/. This script generates a rule
file that references the cloned repo via an @-include, and creates a symlink
for the full skill directory.

Options:
  --project     Install to ./.cursor/rules/ (default, project-level)
  --global      Install to ~/.cursor/rules/ (user-level)
  --uninstall   Remove installed files
  --help, -h    Show this help message
EOF
}

main() {
  INSTALL_SCOPE="project"  # Cursor rules are project-level by default
  UNINSTALL=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --global)  INSTALL_SCOPE="global"; shift ;;
      --project) INSTALL_SCOPE="project"; shift ;;
      --uninstall) UNINSTALL=true; shift ;;
      --help|-h) show_help; exit 0 ;;
      *) error "Unknown option: $1"; show_help; exit 1 ;;
    esac
  done

  print_banner "$ENV_NAME"

  local repo_root
  repo_root="$(resolve_repo_root)"

  if [ "$INSTALL_SCOPE" = "global" ]; then
    local rules_dir="$HOME/.cursor/rules"
    local skills_dir="$HOME/.cursor/skills"
  else
    local rules_dir="./.cursor/rules"
    local skills_dir="./.cursor/skills"
  fi

  local rule_file="${rules_dir}/artifact-refiner.mdc"
  local skill_link="${skills_dir}/${SKILL_NAME}"

  if [ "$UNINSTALL" = true ]; then
    info "Uninstalling from ${rules_dir}..."
    [ -f "$rule_file" ] && rm "$rule_file" && success "Removed rule file: $rule_file"
    remove_symlink "$skill_link" "$repo_root"
    return
  fi

  info "Installing ${SKILL_NAME} for ${ENV_NAME} (${INSTALL_SCOPE})..."

  # Create the .mdc rule file
  mkdir -p "$rules_dir"
  cat > "$rule_file" <<MDC
---
description: PMPO Artifact Refiner — iterative artifact refinement for logos, UI, content, images, and A2UI specs
globs:
alwaysApply: false
---

# Artifact Refiner Skill

This project has the PMPO Artifact Refiner skill installed.

## Skill Location

The full skill is available at: ${repo_root}

## Quick Reference

- Read \`${repo_root}/SKILL.md\` for the canonical skill definition
- Read \`${repo_root}/prompts/meta-controller.md\` for the PMPO orchestration loop
- Domain adapters are in \`${repo_root}/references/domain/\`

## Usage

When the user asks to refine an artifact (logo, UI component, content, image, or A2UI spec),
follow the PMPO methodology: Specify → Plan → Execute → Reflect → Persist → Loop/Terminate.

See the full SKILL.md for detailed instructions.
MDC
  success "Created rule file: $rule_file"

  # Also create a symlink for direct skill access
  mkdir -p "$skills_dir"
  create_symlink "$repo_root" "$skill_link"

  echo ""
  info "Verify with:"
  echo "  cat ${rule_file}"
  echo "  ls -la ${skill_link}"
  echo ""
  info "The rule will appear in Cursor's rule picker (Agent Select mode)."
  echo ""
}

main "$@"
