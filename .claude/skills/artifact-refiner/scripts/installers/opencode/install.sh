#!/usr/bin/env bash
# OpenCode installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="OpenCode"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for OpenCode.

Options:
  --global      Install to ~/.config/opencode/skills/ (default)
  --project     Install to ./.opencode/skills/ (current project)
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
    local target_dir="$HOME/.config/opencode/skills"
  else
    local target_dir="./.opencode/skills"
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
  info "OpenCode also supports these alternative locations:"
  echo "  .claude/skills/${SKILL_NAME}/  (Claude-compatible)"
  echo "  .agents/skills/${SKILL_NAME}/  (agents-compatible)"
  echo ""
  info "Skills are auto-discovered via the 'skill' tool."
  echo ""
}

main "$@"
