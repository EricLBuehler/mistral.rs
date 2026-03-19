#!/usr/bin/env bash
# Claude Code / Claude Desktop installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Claude Code"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Claude Code / Claude Desktop.

Options:
  --global      Install to ~/.claude/skills/ (default)
  --project     Install to ./.claude/skills/ (current project)
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
    local target_dir="$HOME/.claude/skills"
  else
    local target_dir="./.claude/skills"
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
  info "Claude Code will auto-discover skills in this directory."
  info "You can also install via marketplace:"
  echo "  claude plugin install artifact-refiner"
  echo ""
}

main "$@"
