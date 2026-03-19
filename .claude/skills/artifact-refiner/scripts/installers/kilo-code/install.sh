#!/usr/bin/env bash
# Kilo Code installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Kilo Code"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Kilo Code (VS Code extension).

Options:
  --global      Install to ~/.kilocode/rules/ (default)
  --project     Install to ./.kilocode/rules/ (current project)
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
    local target_dir="$HOME/.kilocode/rules"
  else
    local target_dir="./.kilocode/rules"
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
  info "Kilo Code reads .md files from .kilocode/rules/ automatically."
  info "For mode-specific rules, use .kilocode/rules-\${mode}/ directories."
  echo ""
}

main "$@"
