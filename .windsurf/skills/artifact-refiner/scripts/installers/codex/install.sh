#!/usr/bin/env bash
# Codex CLI installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Codex CLI"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for OpenAI Codex CLI.

Options:
  --global      Install to ~/.codex/skills/ (default)
  --project     Install to ./.codex/skills/ (current project)
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
    local target_dir="$HOME/.codex/skills"
  else
    local target_dir="./.codex/skills"
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
  info "Codex CLI uses the Agent Skills (SKILL.md) format."
  info "Skills are auto-discovered from ~/.codex/skills/."
  echo ""
}

main "$@"
