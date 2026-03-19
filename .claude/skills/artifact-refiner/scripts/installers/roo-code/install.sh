#!/usr/bin/env bash
# Roo Code installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Roo Code"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Roo Code (VS Code extension).

Options:
  --global      Install to ~/.roo/rules/ (default)
  --project     Install to ./.roo/rules/ (current project)
  --uninstall   Remove the symlink
  --help, -h    Show this help message
EOF
}

main() {
  parse_flags "$@"
  print_banner "$ENV_NAME"

  local repo_root
  repo_root="$(resolve_repo_root)"

  if [ "$INSTALL_SCOPE" = "global" ]; then
    local target_dir="$HOME/.roo/rules"
  else
    local target_dir="./.roo/rules"
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
  info "Roo Code reads .md files recursively from .roo/rules/."
  info "The skill's SKILL.md and prompts will be discovered automatically."
  info "For mode-specific rules, use .roo/rules-{mode-slug}/ directories."
  echo ""
}

main "$@"
