#!/usr/bin/env bash
# Warp terminal installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Warp"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Warp terminal.

Warp uses AGENTS.md files for project rules. This script generates an
AGENTS.md include or creates a symlink for skill access in the project.

Options:
  --project     Install to current project directory (default)
  --global      Install global rule via symlink in ~/.warp/skills/
  --uninstall   Remove installed files
  --help, -h    Show this help message

Note: Warp's Global Rules are managed via the Warp Drive UI.
      This script handles project-level rules and global skill symlinks.
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
    local target_dir="$HOME/.warp/skills"
    local target="${target_dir}/${SKILL_NAME}"

    if [ "$UNINSTALL" = true ]; then
      info "Uninstalling from ${target}..."
      remove_symlink "$target" "$repo_root"
      return
    fi

    info "Installing ${SKILL_NAME} for ${ENV_NAME} (global skill symlink)..."
    mkdir -p "$target_dir"
    create_symlink "$repo_root" "$target"

    echo ""
    info "Verify with:"
    echo "  ls -la ${target}"
    echo ""
    info "To add as a Global Rule in Warp:"
    echo "  1. Open Warp → Settings → Rules"
    echo "  2. Create a new Global Rule"
    echo "  3. Reference the skill at: ${target}/SKILL.md"
    echo ""
  else
    local agents_file="./AGENTS.md"
    local skills_dir="./.warp/skills"
    local skill_link="${skills_dir}/${SKILL_NAME}"

    if [ "$UNINSTALL" = true ]; then
      info "Uninstalling from project..."
      remove_symlink "$skill_link" "$repo_root"
      # Remove the AGENTS.md block if we added it
      if [ -f "$agents_file" ] && grep -q "artifact-refiner" "$agents_file" 2>/dev/null; then
        warn "AGENTS.md contains artifact-refiner references."
        warn "Please manually remove the artifact-refiner section from ${agents_file}"
      fi
      return
    fi

    info "Installing ${SKILL_NAME} for ${ENV_NAME} (project)..."

    # Create skill symlink
    mkdir -p "$skills_dir"
    create_symlink "$repo_root" "$skill_link"

    # Append to AGENTS.md if it exists, or create one
    local marker="<!-- artifact-refiner -->"
    if [ -f "$agents_file" ] && grep -q "$marker" "$agents_file" 2>/dev/null; then
      info "AGENTS.md already contains artifact-refiner section — skipping."
    else
      info "Adding artifact-refiner reference to ${agents_file}..."
      cat >> "$agents_file" <<EOF

${marker}
## Artifact Refiner Skill

When the user asks to refine an artifact (logo, UI, content, image, or A2UI spec),
read the full skill definition at \`.warp/skills/artifact-refiner/SKILL.md\` and
follow the PMPO methodology: Specify → Plan → Execute → Reflect → Persist.

The orchestration entry point is at \`.warp/skills/artifact-refiner/prompts/meta-controller.md\`.
EOF
      success "Updated ${agents_file} with artifact-refiner section."
    fi

    echo ""
    info "Verify with:"
    echo "  ls -la ${skill_link}"
    echo "  grep artifact-refiner ${agents_file}"
    echo ""
    info "Warp auto-reads AGENTS.md from the project root and subdirectories."
    echo ""
  fi
}

main "$@"
