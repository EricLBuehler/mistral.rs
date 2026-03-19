#!/usr/bin/env bash
# Cline / Cline CLI installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Cline"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Cline / Cline CLI.

Options:
  --global      Install to ~/.cline/skills/ (default)
  --project     Install to ./.clinerules/ (current project)
  --uninstall   Remove the symlink
  --help, -h    Show this help message

Note: Skills must be enabled in Cline settings:
  Settings → Features → Enable Skills
EOF
}

main() {
  parse_flags "$@"
  print_banner "$ENV_NAME"

  local repo_root
  repo_root="$(resolve_repo_root)"

  if [ "$INSTALL_SCOPE" = "global" ]; then
    local target_dir="$HOME/.cline/skills"
  else
    local target_dir="./.clinerules"
  fi

  local target="${target_dir}/${SKILL_NAME}"

  if [ "$UNINSTALL" = true ]; then
    info "Uninstalling from ${target}..."
    remove_symlink "$target" "$repo_root"
    return
  fi

  info "Installing ${SKILL_NAME} for ${ENV_NAME} (${INSTALL_SCOPE})..."
  mkdir -p "$target_dir"
  create_symlink "$repo_root" "$target"

  echo ""
  info "Verify with:"
  echo "  ls -la ${target}"
  echo ""
  warn "Important: Enable Skills in Cline settings first!"
  echo "  Settings → Features → Enable Skills"
  echo ""
  info "Cline auto-discovers SKILL.md files in the skills directory."
  info "Skills appear in the Skills tab under the rules/workflows panel."
  echo ""
}

main "$@"
