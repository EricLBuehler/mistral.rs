#!/usr/bin/env bash
# Zed IDE installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Zed"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Zed IDE.

Zed uses .rules files at the worktree root and a Rules Library for global rules.
This script generates a .rules file or AGENTS.md for project-level instructions,
and creates a symlink for full skill access.

Options:
  --project     Install to current project directory (default)
  --global      Install symlink to ~/.config/zed/skills/
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
    local target_dir="$HOME/.config/zed/skills"
    local target="${target_dir}/${SKILL_NAME}"

    if [ "$UNINSTALL" = true ]; then
      info "Uninstalling from ${target}..."
      remove_symlink "$target" "$repo_root"
      return
    fi

    info "Installing ${SKILL_NAME} for ${ENV_NAME} (global)..."
    mkdir -p "$target_dir"
    create_symlink "$repo_root" "$target"

    echo ""
    info "Verify with:"
    echo "  ls -la ${target}"
    echo ""
    info "To add to Zed's Rules Library:"
    echo "  1. Open Agent Panel → '...' menu → 'Rules...'"
    echo "  2. Click '+' to create a new rule"
    echo "  3. Reference the skill at: ${target}/SKILL.md"
    echo "  4. Mark as 'Default' to include in all threads"
    echo ""
  else
    local skills_dir="./.zed/skills"
    local skill_link="${skills_dir}/${SKILL_NAME}"
    local rules_file="./.rules"

    if [ "$UNINSTALL" = true ]; then
      info "Uninstalling from project..."
      remove_symlink "$skill_link" "$repo_root"
      if [ -f "$rules_file" ] && grep -q "artifact-refiner" "$rules_file" 2>/dev/null; then
        warn ".rules file contains artifact-refiner references."
        warn "Please manually remove the artifact-refiner section from ${rules_file}"
      fi
      return
    fi

    info "Installing ${SKILL_NAME} for ${ENV_NAME} (project)..."

    # Create skill symlink
    mkdir -p "$skills_dir"
    create_symlink "$repo_root" "$skill_link"

    # Append to .rules if present, or create one
    local marker="<!-- artifact-refiner -->"
    if [ -f "$rules_file" ] && grep -q "$marker" "$rules_file" 2>/dev/null; then
      info ".rules already contains artifact-refiner section — skipping."
    else
      info "Adding artifact-refiner reference to ${rules_file}..."
      cat >> "$rules_file" <<EOF

${marker}
## Artifact Refiner Skill

When the user asks to refine an artifact (logo, UI, content, image, or A2UI spec),
read the full skill definition at \`.zed/skills/artifact-refiner/SKILL.md\` and
follow the PMPO methodology: Specify → Plan → Execute → Reflect → Persist.

The orchestration entry point is at \`.zed/skills/artifact-refiner/prompts/meta-controller.md\`.
EOF
      success "Updated ${rules_file} with artifact-refiner section."
    fi

    echo ""
    info "Verify with:"
    echo "  ls -la ${skill_link}"
    echo "  cat ${rules_file}"
    echo ""
    info "Zed reads .rules from the worktree root for all Agent Panel interactions."
    info "You can also use @rule in threads to include specific rules on demand."
    echo ""
  fi
}

main "$@"
