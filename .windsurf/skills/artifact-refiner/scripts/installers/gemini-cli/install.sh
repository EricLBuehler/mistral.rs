#!/usr/bin/env bash
# Gemini CLI installer for artifact-refiner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ENV_NAME="Gemini CLI"

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the Artifact Refiner skill for Gemini CLI.

Creates a symlink in ~/.gemini/extensions/ (global) or .gemini/skills/ (project).
If your version of Gemini CLI supports 'gemini extensions link', you can also run
that command manually after installing.

Options:
  --global      Install to ~/.gemini/extensions/ (default)
  --project     Install to ./.gemini/skills/ (current project)
  --uninstall   Remove the extension symlink
  --help, -h    Show this help message
EOF
}

main() {
  parse_flags "$@"
  print_banner "$ENV_NAME"

  local repo_root
  repo_root="$(resolve_repo_root)"

  if [ "$INSTALL_SCOPE" = "project" ]; then
    local target_dir="./.gemini/skills"
    local target="${target_dir}/${SKILL_NAME}"

    if [ "$UNINSTALL" = true ]; then
      info "Uninstalling from ${target}..."
      remove_symlink "$target" "$repo_root"
      return
    fi

    info "Installing ${SKILL_NAME} for ${ENV_NAME} (project)..."
    mkdir -p "$target_dir"
    create_symlink "$repo_root" "$target"
  else
    local target_dir="$HOME/.gemini/extensions"
    local target="${target_dir}/${SKILL_NAME}"

    if [ "$UNINSTALL" = true ]; then
      info "Uninstalling from ${target}..."
      remove_symlink "$target" "$repo_root"
      return
    fi

    info "Installing ${SKILL_NAME} for ${ENV_NAME} (global extension)..."
    mkdir -p "$target_dir"
    create_symlink "$repo_root" "$target"
  fi

  echo ""
  info "Verify with:"
  echo "  ls -la ${target}"
  echo ""
  info "If your Gemini CLI supports extensions, you can also run:"
  echo "  gemini extensions link ${repo_root}"
  echo ""
  info "Gemini CLI will auto-discover skills via the extension."
  info "You can also use GEMINI.md context files for project-level instructions."
  echo ""
}

main "$@"
