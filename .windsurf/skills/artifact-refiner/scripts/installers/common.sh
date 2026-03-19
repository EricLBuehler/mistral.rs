#!/usr/bin/env bash
# common.sh — Shared helper functions for artifact-refiner installer scripts
# Source this file from individual installer scripts.
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPO_URL="https://github.com/GQAdonis/artifact-refiner-skill"
SKILL_NAME="artifact-refiner"

info()    { echo -e "${BLUE}ℹ${NC}  $*"; }
success() { echo -e "${GREEN}✅${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}❌${NC} $*" >&2; }

# Resolve the absolute path of the artifact-refiner repo root.
# Works whether called from the repo directly or via a symlink.
resolve_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  # Walk up from scripts/installers/<env>/ to repo root
  echo "$(cd "${script_dir}/../../.." && pwd)"
}

# Create a symlink, handling existing links/directories gracefully.
# Usage: create_symlink <source> <target>
create_symlink() {
  local source="$1"
  local target="$2"

  if [ -L "$target" ]; then
    local existing
    existing="$(readlink "$target")"
    if [ "$existing" = "$source" ]; then
      info "Symlink already exists: $target → $source"
      return 0
    else
      warn "Updating existing symlink: $target"
      warn "  Old: $existing"
      warn "  New: $source"
      rm "$target"
    fi
  elif [ -e "$target" ]; then
    error "Target already exists and is not a symlink: $target"
    error "Please remove or rename it first, then re-run this script."
    exit 1
  fi

  ln -s "$source" "$target"
  success "Created symlink: $target → $source"
}

# Remove a symlink if it exists and points to our repo.
# Usage: remove_symlink <target> <expected_source>
remove_symlink() {
  local target="$1"
  local expected_source="$2"

  if [ -L "$target" ]; then
    local existing
    existing="$(readlink "$target")"
    if [ "$existing" = "$expected_source" ]; then
      rm "$target"
      success "Removed symlink: $target"
    else
      warn "Symlink exists but points elsewhere: $target → $existing"
      warn "Skipping removal (not ours)."
    fi
  elif [ -e "$target" ]; then
    warn "Target is not a symlink: $target — skipping."
  else
    info "Nothing to remove: $target does not exist."
  fi
}

# Parse common flags: --global, --project, --uninstall, --help
parse_flags() {
  INSTALL_SCOPE="global"
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
}

# Print the repo banner
print_banner() {
  local env_name="$1"
  echo ""
  echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║${NC}  Artifact Refiner — ${env_name} Installer           ${BLUE}║${NC}"
  echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
  echo ""
}
