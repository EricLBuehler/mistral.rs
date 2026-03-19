#!/usr/bin/env bash
# Windsurf installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Windsurf"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Windsurf (Codeium).

Windsurf uses .windsurf/rules/ for project rules. This script generates a
rules file that references the cloned skill repo and creates a symlink.

Options:
  --project     Install to ./.windsurf/rules/ (default, project-level)
  --global      Install to ~/.codeium/windsurf/rules/
  --uninstall   Remove installed files
  --help, -h    Show this help message
EOF
}

main() {
  INSTALL_SCOPE="project"
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
    local rules_dir="$HOME/.codeium/windsurf/rules"
    local skills_dir="$HOME/.codeium/windsurf/skills"
  else
    local rules_dir="./.windsurf/rules"
    local skills_dir="./.windsurf/skills"
  fi

  local rule_file="${rules_dir}/artifact-refiner.md"
  local skill_link="${skills_dir}/${SKILL_NAME}"

  if [ "$UNINSTALL" = true ]; then
    info "Uninstalling from ${rules_dir}..."
    [ -f "$rule_file" ] && rm "$rule_file" && success "Removed rule file: $rule_file"
    remove_symlink "$skill_link" "$repo_root"
    return
  fi

  info "Installing ${SKILL_NAME} for ${ENV_NAME} (${INSTALL_SCOPE})..."

  mkdir -p "$rules_dir"
  cat > "$rule_file" <<'RULE'
# Artifact Refiner Skill

This project has the PMPO Artifact Refiner skill installed.

## When to Use

When the user asks to refine an artifact (logo, UI component, content, image, or A2UI spec),
follow the PMPO methodology: Specify → Plan → Execute → Reflect → Persist → Loop/Terminate.

## Skill Location

RULE
  echo "The full skill is symlinked at: ${skill_link}" >> "$rule_file"
  echo "" >> "$rule_file"
  echo "Read the SKILL.md in that directory for the canonical skill definition." >> "$rule_file"
  echo "Read prompts/meta-controller.md for the PMPO orchestration loop entry point." >> "$rule_file"
  success "Created rule file: $rule_file"

  mkdir -p "$skills_dir"
  create_symlink "$repo_root" "$skill_link"

  echo ""
  info "Verify with:"
  echo "  cat ${rule_file}"
  echo "  ls -la ${skill_link}"
  echo ""
  info "Windsurf's Cascade agent will pick up the rules file automatically."
  echo ""
}

main "$@"
