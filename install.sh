#!/bin/sh
set -e

# mistral.rs Installation Script
# Cross-platform installer for Linux and macOS with automatic hardware detection

# Check if we can prompt the user (stdin is a tty or we have /dev/tty)
can_prompt() {
    [ -t 0 ] || [ -e /dev/tty ]
}

# Read user input, using /dev/tty if stdin is not a terminal (e.g., piped from curl)
read_input() {
    # MISTRALRS_INSTALL_YES=1 auto-confirms every prompt (non-interactive installs, `mistralrs update`).
    if [ "${MISTRALRS_INSTALL_YES:-}" = "1" ]; then
        REPLY="y"
        return
    fi
    if [ -t 0 ]; then
        read -r REPLY
    else
        read -r REPLY </dev/tty
    fi
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions (output to stderr so they don't get captured in command substitution)
info() { printf "${BLUE}info:${NC} %s\n" "$1" >&2; }
success() { printf "${GREEN}success:${NC} %s\n" "$1" >&2; }
warn() { printf "${YELLOW}warning:${NC} %s\n" "$1" >&2; }
error() { printf "${RED}error:${NC} %s\n" "$1" >&2; exit 1; }

# Banner
print_banner() {
    printf "${BOLD}"
    echo "  __  __ _     _             _              "
    echo " |  \\/  (_)___| |_ _ __ __ _| |  _ __ ___   "
    echo " | |\\/| | / __| __| '__/ _\` | | | '__/ __|  "
    echo " | |  | | \\__ \\ |_| | | (_| | |_| |  \\__ \\  "
    echo " |_|  |_|_|___/\\__|_|  \\__,_|_(_)_|  |___/  "
    echo ""
    printf "${NC}${BLUE}Fast, flexible LLM inference.${NC}\n"
    echo ""
}

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            echo "linux"
            ;;
        *)
            error "Unsupported operating system: $(uname -s)"
            ;;
    esac
}

# Minimum required Rust version
REQUIRED_RUST_VERSION="1.94"
MISTRALRS_REPO_URL="https://github.com/EricLBuehler/mistral.rs"
MISTRALRS_BRANCH="master"
MISTRALRS_CLI_PACKAGE="mistralrs-cli"

# Check if Rust is installed
check_rust() {
    command -v cargo >/dev/null 2>&1
}

# Get installed Rust version (major.minor)
get_rust_version() {
    rustc --version 2>/dev/null | sed -n 's/rustc \([0-9]*\.[0-9]*\).*/\1/p'
}

# Compare two version strings (returns 0 if $1 >= $2, 1 otherwise)
version_gte() {
    v1_major=$(echo "$1" | cut -d. -f1)
    v1_minor=$(echo "$1" | cut -d. -f2)
    v2_major=$(echo "$2" | cut -d. -f1)
    v2_minor=$(echo "$2" | cut -d. -f2)

    if [ "$v1_major" -gt "$v2_major" ] 2>/dev/null; then
        return 0
    elif [ "$v1_major" -eq "$v2_major" ] 2>/dev/null && [ "$v1_minor" -ge "$v2_minor" ] 2>/dev/null; then
        return 0
    fi
    return 1
}

# Install Rust via rustup
install_rust() {
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
    success "Rust installed successfully"
}

# Update Rust to latest version
update_rust() {
    info "Updating Rust to latest version..."
    rustup update stable
    success "Rust updated successfully"
}

# Detect CUDA compute capability
detect_cuda_compute_cap() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi

    # Try direct query
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

    if [ -n "$cc" ]; then
        echo "$cc"
    fi
}

# Detect CUDA toolkit major version from nvcc (e.g. 13). Empty if nvcc is unavailable.
# CUDA toolkit version as major*100+minor (e.g. 13.1 -> 1301), empty if nvcc absent.
detect_cuda_version_code() {
    if command -v nvcc >/dev/null 2>&1; then
        ver=$(nvcc --version 2>/dev/null | grep -oE "release [0-9]+\.[0-9]+" | head -1 | grep -oE "[0-9]+\.[0-9]+")
        if [ -n "$ver" ]; then
            echo $(( ${ver%%.*} * 100 + ${ver#*.} ))
        fi
    fi
}

# Detect the maximum CUDA runtime version supported by the installed NVIDIA driver.
detect_cuda_driver_version_code() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi

    ver=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version:[[:space:]]*[0-9]+\.[0-9]+" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    if [ -n "$ver" ]; then
        echo $(( ${ver%%.*} * 100 + ${ver#*.} ))
    fi
}

version_code_to_str() {
    code="$1"
    if [ -n "$code" ]; then
        printf "%s.%s" $(( code / 100 )) $(( code % 100 ))
    fi
}

# Check if MKL is installed
detect_mkl() {
    # Check MKLROOT environment variable
    if [ -n "$MKLROOT" ] && [ -d "$MKLROOT" ]; then
        return 0
    fi

    # Check common installation paths
    for path in /opt/intel/oneapi/mkl/latest /opt/intel/mkl /opt/intel/oneapi/mkl; do
        if [ -d "$path" ]; then
            return 0
        fi
    done

    return 1
}

# Check if CPU is Intel
is_intel_cpu() {
    if [ -f /proc/cpuinfo ]; then
        grep -qi "intel" /proc/cpuinfo && return 0
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "intel" && return 0
    fi
    return 1
}

# Check/install Xcode Command Line Tools (macOS)
check_xcode_cli_tools() {
    if ! xcrun --version >/dev/null 2>&1; then
        warn "Xcode Command Line Tools are not installed"
        echo ""
        printf "Would you like to install them now? [Y/n] "
        read_input
        case "$REPLY" in
            [Nn]*)
                error "Xcode Command Line Tools are required for Metal support"
                ;;
        esac
        info "Installing Xcode Command Line Tools..."
        xcode-select --install
        echo "Please complete the installation in the dialog, then press Enter to continue..."
        read_input
        sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    fi
}

# Check/install Metal Toolchain (macOS)
check_metal_toolchain() {
    if ! xcrun metal --version >/dev/null 2>&1; then
        warn "Metal Toolchain is not installed"
        echo ""
        printf "Would you like to install it now? [Y/n] "
        read_input
        case "$REPLY" in
            [Nn]*)
                error "Metal Toolchain is required for Metal support"
                ;;
        esac
        info "Installing Metal Toolchain..."
        xcodebuild -downloadComponent MetalToolchain
    fi
}

# Check if cuDNN is installed
detect_cudnn() {
    # Check common cuDNN library paths
    for path in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib64; do
        if [ -f "$path/libcudnn.so" ] || ls "$path"/libcudnn.so.* >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

# Check if NCCL is installed
detect_nccl() {
    for root in "$NCCL_ROOT" "$NCCL_HOME" "$CUDA_HOME" "$CUDA_PATH" /usr/local/cuda; do
        [ -n "$root" ] || continue
        for subdir in lib lib64 lib/x86_64-linux-gnu; do
            if ls "$root/$subdir"/libnccl.so* >/dev/null 2>&1; then
                return 0
            fi
        done
    done

    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libnccl\\.so"; then
        return 0
    fi

    for path in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/local/lib64 /usr/lib64; do
        if ls "$path"/libnccl.so* >/dev/null 2>&1; then
            return 0
        fi
    done

    return 1
}

# Build feature string based on detected hardware
build_features() {
    os="$1"
    features=""

    if [ "$os" = "macos" ]; then
        features="metal"
        info "macOS detected - enabling metal"
    else
        # Check for CUDA
        cuda_cc=$(detect_cuda_compute_cap)
        if [ -n "$cuda_cc" ]; then
            features="cuda"
            # Extract major.minor from compute cap (e.g., 89 -> 8.9)
            cc_major=$(echo "$cuda_cc" | cut -c1)
            cc_minor=$(echo "$cuda_cc" | cut -c2-)
            info "CUDA detected (compute capability: ${cc_major}.${cc_minor})"

            if [ "${MISTRALRS_INSTALL_NO_NCCL:-}" = "1" ]; then
                info "MISTRALRS_INSTALL_NO_NCCL=1 set - skipping nccl"
            elif detect_nccl; then
                features="$features nccl"
                info "NCCL detected - enabling nccl for CUDA multi-GPU tensor parallelism"
            elif [ "${MISTRALRS_INSTALL_NCCL:-}" = "1" ]; then
                features="$features nccl"
                warn "MISTRALRS_INSTALL_NCCL=1 set but NCCL was not detected; the build may fail unless libnccl is on the linker path"
            else
                warn "NCCL not found - skipping nccl. Install NCCL or set MISTRALRS_INSTALL_NCCL=1 to force it; NCCL is the preferred CUDA multi-GPU path."
            fi

            # Check for cuDNN
            if detect_cudnn; then
                features="$features cudnn"
                info "cuDNN detected - enabling cudnn"
            else
                info "cuDNN not found - skipping cudnn feature"
            fi

            # Add flash attention based on compute capability
            if [ "$cuda_cc" = "90" ]; then
                features="$features flash-attn-v3"
                info "Hopper GPU detected - enabling flash-attn-v3"
            elif [ "$cuda_cc" -ge 80 ] 2>/dev/null; then
                features="$features flash-attn"
                info "Ampere+ GPU detected - enabling flash-attn"
            fi
            
            # cuTile: optimized CUDA kernels. Needs CUDA >= 13.1 (its JIT tool tileiras ships with 13.1+);
            # runs on Ampere (80-89) or Blackwell+ (>=100), not Hopper (90-99).
            cuda_ver_code=$(detect_cuda_version_code)
            if [ -n "$cuda_ver_code" ] && [ "$cuda_ver_code" -ge 1301 ] 2>/dev/null; then
                if { [ "$cuda_cc" -ge 80 ] && [ "$cuda_cc" -lt 90 ]; } || [ "$cuda_cc" -ge 100 ] 2>/dev/null; then
                    features="$features cutile"
                    info "CUDA >= 13.1 and supported arch - enabling cutile (optimized kernels)"
                fi
            fi
        else
            info "No NVIDIA GPU detected"
        fi
    fi

    # Check for MKL on Intel
    if is_intel_cpu && detect_mkl; then
        features="$features mkl"
        info "Intel MKL detected - enabling mkl"
    fi

    # Trim leading/trailing whitespace
    echo "$features" | xargs
}

# Check if ffmpeg is installed
check_ffmpeg() {
    command -v ffmpeg >/dev/null 2>&1 && ffmpeg -version >/dev/null 2>&1
}

# Install ffmpeg using the system package manager
install_ffmpeg() {
    os="$1"
    if [ "$os" = "macos" ]; then
        if command -v brew >/dev/null 2>&1; then
            if brew list ffmpeg >/dev/null 2>&1; then
                info "Reinstalling FFmpeg via Homebrew..."
                brew reinstall ffmpeg
            else
                info "Installing FFmpeg via Homebrew..."
                brew install ffmpeg
            fi
        else
            warn "Homebrew not found. Install FFmpeg manually: https://ffmpeg.org/download.html"
            return 1
        fi
    else
        if command -v apt-get >/dev/null 2>&1; then
            info "Installing FFmpeg via apt..."
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v dnf >/dev/null 2>&1; then
            info "Installing FFmpeg via dnf..."
            sudo dnf install -y ffmpeg
        else
            warn "Could not detect package manager. Install FFmpeg manually: https://ffmpeg.org/download.html"
            return 1
        fi
    fi
}

# Install mistralrs-cli
install_mistralrs() {
    features="$1"

    # MISTRALRS_INSTALL_TAG pins a git tag; otherwise build the latest master.
    if [ -n "$MISTRALRS_INSTALL_TAG" ]; then
        git_ref="--tag $MISTRALRS_INSTALL_TAG"
        ref_desc="tag $MISTRALRS_INSTALL_TAG"
    else
        git_ref="--branch $MISTRALRS_BRANCH"
        ref_desc="branch $MISTRALRS_BRANCH"
    fi

    if [ -n "$features" ]; then
        info "Installing mistralrs-cli from GitHub $ref_desc with features: $features"
        cargo install --force --locked --git "$MISTRALRS_REPO_URL" $git_ref "$MISTRALRS_CLI_PACKAGE" --features "$features"
    else
        info "Installing mistralrs-cli from GitHub $ref_desc with default features"
        cargo install --force --locked --git "$MISTRALRS_REPO_URL" $git_ref "$MISTRALRS_CLI_PACKAGE"
    fi
}

# Prebuilt binaries: SMs we publish a CUDA build for, per arch (see .github/workflows/release.yml).
# aarch64 covers the Grace parts only (GH200/GB200/GB10).
PREBUILT_CUDA_SMS_X86="80 86 89 90 100 120"
PREBUILT_CUDA_SMS_AARCH64="90 100 121"
# Newest first. Format is asset-token:minimum-driver-cuda-code.
PREBUILT_CUDA_VARIANTS="131:1301 129:1209 128:1208"
# MISTRALRS_INSTALL_TAG pins a specific release (e.g. v0.8.9); default is the latest stable release.
if [ -n "$MISTRALRS_INSTALL_TAG" ]; then
    RELEASE_BASE="https://github.com/EricLBuehler/mistral.rs/releases/download/$MISTRALRS_INSTALL_TAG"
else
    RELEASE_BASE="https://github.com/EricLBuehler/mistral.rs/releases/latest/download"
fi
PREBUILT_DIR="$HOME/.mistralrs"
BIN_DIR="$HOME/.local/bin"
MISTRALRS_ENV="$PREBUILT_DIR/env"

cuda_sms_for_variant() {
    cuda_token="$1"
    arch="$2"

    case "$cuda_token:$arch" in
        128:x86_64) echo "80 86 89 90 100 120" ;;
        128:aarch64|128:arm64) echo "90 100" ;;
        129:aarch64|129:arm64) echo "121" ;;
        131:x86_64) echo "$PREBUILT_CUDA_SMS_X86" ;;
        131:aarch64|131:arm64) echo "$PREBUILT_CUDA_SMS_AARCH64" ;;
    esac
}

# Echo the prebuilt asset name for this platform, or empty if none is published for it.
detect_prebuilt_asset() {
    os="$1"
    arch=$(uname -m)
    if [ "$os" = "macos" ]; then
        # Only Apple Silicon has a prebuilt; Intel Macs build from source.
        [ "$arch" = "arm64" ] && echo "mistralrs-metal-aarch64-apple-darwin.tar.gz"
        return
    fi
    # Linux x86_64 and aarch64 have prebuilts; other arches build from source.
    case "$arch" in
        x86_64) triple="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) triple="aarch64-unknown-linux-gnu" ;;
        *) return ;;
    esac
    cc=$(detect_cuda_compute_cap)
    if [ -n "$cc" ]; then
        driver_cuda_code=$(detect_cuda_driver_version_code)
        if [ -z "$driver_cuda_code" ]; then
            warn "Could not detect the CUDA version supported by the NVIDIA driver; building from source."
            return
        fi

        for variant in $PREBUILT_CUDA_VARIANTS; do
            cuda_token=${variant%:*}
            cuda_min_code=${variant#*:}
            if [ "$driver_cuda_code" -ge "$cuda_min_code" ] 2>/dev/null; then
                cuda_sms=$(cuda_sms_for_variant "$cuda_token" "$arch")
                for sm in $cuda_sms; do
                    if [ "$cc" = "$sm" ]; then
                        echo "mistralrs-cuda${cuda_token}-sm${cc}-${triple}.tar.gz"
                        return
                    fi
                done
            fi
        done
        if [ "$driver_cuda_code" -lt 1208 ] 2>/dev/null; then
            warn "NVIDIA driver supports CUDA $(version_code_to_str "$driver_cuda_code"), but the oldest CUDA prebuilt requires CUDA 12.8; building from source."
        else
            warn "No CUDA prebuilt matches compute capability $cc and driver CUDA $(version_code_to_str "$driver_cuda_code"); building from source."
        fi
        return
    fi
    echo "mistralrs-cpu-${triple}.tar.gz"
}

detect_legacy_cuda_prebuilt_asset() {
    os="$1"
    arch=$(uname -m)
    [ "$os" = "linux" ] || return
    case "$arch" in
        x86_64) triple="x86_64-unknown-linux-gnu"; cuda_sms="$PREBUILT_CUDA_SMS_X86" ;;
        aarch64|arm64) triple="aarch64-unknown-linux-gnu"; cuda_sms="$PREBUILT_CUDA_SMS_AARCH64" ;;
        *) return ;;
    esac
    cc=$(detect_cuda_compute_cap)
    [ -n "$cc" ] || return
    driver_cuda_code=$(detect_cuda_driver_version_code)
    [ -n "$driver_cuda_code" ] && [ "$driver_cuda_code" -ge 1301 ] 2>/dev/null || return
    for sm in $cuda_sms; do
        if [ "$cc" = "$sm" ]; then
            echo "mistralrs-cuda-sm${cc}-${triple}.tar.gz"
            return
        fi
    done
}

# Download and install a prebuilt asset. Returns 0 on success, 1 on failure.
install_prebuilt() {
    asset="$1"
    tmp=$(mktemp -d)
    info "Downloading $asset"
    if ! curl --proto '=https' --tlsv1.2 -fSL --progress-bar "$RELEASE_BASE/$asset" -o "$tmp/$asset"; then
        rm -rf "$tmp"
        return 1
    fi
    rm -rf "$PREBUILT_DIR"
    mkdir -p "$PREBUILT_DIR"
    # CPU/Metal tarballs contain a bare `mistralrs`; CUDA tarballs add lib/ and bin/tileiras.
    if ! tar xzf "$tmp/$asset" -C "$PREBUILT_DIR"; then
        rm -rf "$tmp"
        return 1
    fi
    rm -rf "$tmp"
    chmod +x "$PREBUILT_DIR/mistralrs" 2>/dev/null || true
    [ -f "$PREBUILT_DIR/bin/tileiras" ] && chmod +x "$PREBUILT_DIR/bin/tileiras" 2>/dev/null || true
    mkdir -p "$BIN_DIR"
    # Symlink onto PATH; $ORIGIN/lib resolves through the symlink to the real lib dir, and a
    # tileiras symlink lets cutile's PATH probe find the bundled assembler.
    ln -sf "$PREBUILT_DIR/mistralrs" "$BIN_DIR/mistralrs"
    [ -f "$PREBUILT_DIR/bin/tileiras" ] && ln -sf "$PREBUILT_DIR/bin/tileiras" "$BIN_DIR/tileiras"
    if ! "$PREBUILT_DIR/mistralrs" --version >/dev/null 2>&1; then
        warn "Prebuilt binary did not run; falling back to source build."
        return 1
    fi
    return 0
}

# Build and install from source via cargo. Used when no prebuilt matches or when forced.
# Builds the latest `master` (bleeding edge), unlike the prebuilt path which is the stable release.
build_from_source() {
    os="$1"
    if [ -n "$MISTRALRS_INSTALL_TAG" ]; then
        info "Building from source: tag $MISTRALRS_INSTALL_TAG."
    else
        info "Building from source: latest $MISTRALRS_BRANCH (bleeding edge)."
    fi

    # Check for Rust
    if check_rust; then
        rust_version_full=$(rustc --version 2>/dev/null || echo "unknown")
        rust_version=$(get_rust_version)
        info "Rust is installed: $rust_version_full"

        if [ -n "$rust_version" ] && ! version_gte "$rust_version" "$REQUIRED_RUST_VERSION"; then
            warn "Rust $rust_version is below the required version $REQUIRED_RUST_VERSION"
            echo ""
            printf "Would you like to update Rust now? [Y/n] "
            read_input
            case "$REPLY" in
                [Nn]*)
                    error "Rust $REQUIRED_RUST_VERSION or newer is required to install mistral.rs"
                    ;;
            esac
            update_rust
            rust_version=$(get_rust_version)
            if ! version_gte "$rust_version" "$REQUIRED_RUST_VERSION"; then
                error "Failed to update Rust to required version $REQUIRED_RUST_VERSION"
            fi
        fi
    else
        warn "Rust is not installed"
        echo ""
        printf "Would you like to install Rust now? [Y/n] "
        read_input
        case "$REPLY" in
            [Nn]*)
                error "Rust is required to build mistral.rs from source"
                ;;
        esac
        install_rust
    fi

    # Run prereq installers outside any $() so xcodebuild stdout (asset paths with slashes) can't leak into the captured feature string.
    if [ "$os" = "macos" ]; then
        check_xcode_cli_tools
        check_metal_toolchain
    fi

    echo ""
    info "Detecting hardware capabilities..."
    features=$(build_features "$os")

    echo ""
    printf "${BOLD}Installation Summary${NC}\n"
    echo "===================="
    if [ -n "$features" ]; then
        printf "Features: ${GREEN}%s${NC}\n" "$features"
    else
        printf "Features: ${YELLOW}(none - CPU only)${NC}\n"
    fi
    echo ""
    printf "Proceed with installation? [Y/n] "
    read_input
    case "$REPLY" in
        [Nn]*)
            info "Installation cancelled"
            exit 0
            ;;
    esac

    echo ""
    install_mistralrs "$features"
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi
}

# Offer FFmpeg (optional, video input). Sets FFMPEG_SKIPPED. Shared by both install paths.
maybe_install_ffmpeg() {
    os="$1"
    FFMPEG_SKIPPED=""
    # MISTRALRS_INSTALL_IGNORE_FFMPEG=1 leaves ffmpeg untouched (CI, `mistralrs update`).
    if [ "${MISTRALRS_INSTALL_IGNORE_FFMPEG:-}" = "1" ]; then
        if check_ffmpeg; then
            info "FFmpeg is installed (enables video input support)"
        else
            FFMPEG_SKIPPED=1
        fi
        return
    fi
    if check_ffmpeg; then
        info "FFmpeg is installed (enables video input support)"
        return
    fi
    echo ""
    printf "${YELLOW}(Optional)${NC} FFmpeg is required for video input support.\n"
    printf "Would you like to install FFmpeg? [y/N] "
    read_input
    case "$REPLY" in
        [Yy]*)
            install_ffmpeg "$os"
            if check_ffmpeg; then
                success "FFmpeg installed successfully"
            else
                warn "FFmpeg installation failed - you can install it manually later"
                FFMPEG_SKIPPED=1
            fi
            ;;
        *)
            info "Skipping FFmpeg installation"
            FFMPEG_SKIPPED=1
            ;;
    esac
}

# Render a path with $HOME collapsed to ~ for readability.
tildify() {
    case "$1" in
        "$HOME"/*) printf '~%s' "${1#$HOME}" ;;
        *) printf '%s' "$1" ;;
    esac
}

write_env_script() {
    printf '%s\n' \
        '#!/bin/sh' \
        '# mistral.rs shell setup' \
        'case ":${PATH}:" in' \
        '    *:"$HOME/.local/bin":*) ;;' \
        '    *) export PATH="$HOME/.local/bin:$PATH" ;;' \
        'esac' \
        > "$MISTRALRS_ENV"
}

append_source_line() {
    rc="$1"
    [ -n "$rc" ] || return
    mkdir -p "$(dirname "$rc")"
    touch "$rc"
    source_line='. "$HOME/.mistralrs/env"'
    if ! grep -Fqx "$source_line" "$rc"; then
        printf '\n%s\n' "$source_line" >> "$rc"
        PATH_RCS="${PATH_RCS}${PATH_RCS:+, }$(tildify "$rc")"
    fi
}

setup_shell_path() {
    PATH_RCS=""
    write_env_script
    append_source_line "$HOME/.profile"
    for rc in "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.bashrc"; do
        [ -f "$rc" ] && append_source_line "$rc"
    done
    if command -v zsh >/dev/null 2>&1 || [ "${SHELL##*/}" = "zsh" ]; then
        append_source_line "${ZDOTDIR:-$HOME}/.zshenv"
    fi
    printf '%s' "$PATH_RCS"
}

warn_if_shadowed() {
    resolved=$(command -v mistralrs 2>/dev/null || true)
    [ -n "$resolved" ] || return
    case "$resolved" in
        "$BIN_DIR/mistralrs"|"$PREBUILT_DIR/mistralrs") ;;
        *)
            printf "${YELLOW}Note:${NC} another mistralrs appears earlier on PATH: %s\n" "$(tildify "$resolved")"
            printf "      The prebuilt install is available at: %s\n\n" "$(tildify "$PREBUILT_DIR/mistralrs")"
            ;;
    esac
}

# Shared success message + examples + PATH guidance, tailored to how the binary was installed.
print_success() {
    method="$1"
    if [ "$method" = "prebuilt" ]; then
        success_bin="$PREBUILT_DIR/mistralrs"
    else
        success_bin="${CARGO_HOME:-$HOME/.cargo}/bin/mistralrs"
    fi
    ver=$("$success_bin" --version 2>/dev/null | head -1)
    [ -n "$ver" ] || ver="mistral.rs"
    echo ""
    if [ "$method" = "prebuilt" ]; then
        success "$ver installed successfully (prebuilt binary)!"
    else
        success "$ver installed successfully (built from source)!"
    fi
    echo ""
    printf "${BOLD}Installed${NC}\n"
    echo "========="
    if [ "$method" = "prebuilt" ]; then
        printf "  binary      %s\n" "$(tildify "$PREBUILT_DIR/mistralrs")"
        printf "  on PATH     %s -> %s\n" "$(tildify "$BIN_DIR/mistralrs")" "$(tildify "$PREBUILT_DIR/mistralrs")"
        if [ -L "$BIN_DIR/tileiras" ]; then
            printf "  cutile JIT  %s -> %s\n" "$(tildify "$BIN_DIR/tileiras")" "$(tildify "$PREBUILT_DIR/bin/tileiras")"
        fi
    else
        printf "  binary      %s\n" "$(tildify "${CARGO_HOME:-$HOME/.cargo}/bin/mistralrs")"
    fi
    echo ""
    printf "${BOLD}Quick Start${NC}\n"
    echo "==========="
    echo ""
    echo "  # Chat in your terminal (downloads the model on first run)"
    echo "  mistralrs run -m Qwen/Qwen3-4B"
    echo ""
    echo "  # Serve an OpenAI-compatible + Anthropic-compatible API on port 1234"
    echo "  mistralrs serve -m Qwen/Qwen3-4B"
    echo ""
    echo "  # Run as a local agent (tools, web search, code execution)"
    echo "  mistralrs serve --agent -m google/gemma-4-E4B-it"
    echo ""
    echo "Docs:     https://ericlbuehler.github.io/mistral.rs/"
    echo "Source:   https://github.com/EricLBuehler/mistral.rs"
    echo ""
    if [ -n "$FFMPEG_SKIPPED" ]; then
        printf "${YELLOW}Note:${NC} FFmpeg was not installed; video input will be unavailable. Install it later to enable video.\n\n"
    fi
    if [ "$method" = "prebuilt" ]; then
        path_rcs=$(setup_shell_path)
        case ":$PATH:" in
            *":$BIN_DIR:"*)
                if [ -n "$path_rcs" ]; then
                    printf "${YELLOW}Note:${NC} configured future shells via %s.\n" "$(tildify "$MISTRALRS_ENV")"
                    printf "      Updated: %s\n\n" "$path_rcs"
                fi
                ;;
            *)
                printf "${YELLOW}Note:${NC} added %s to your PATH via %s.\n" "$(tildify "$BIN_DIR")" "$(tildify "$MISTRALRS_ENV")"
                if [ -n "$path_rcs" ]; then
                    printf "      Updated: %s\n" "$path_rcs"
                fi
                printf "      Restart your terminal or run: . \"%s\"\n\n" "$MISTRALRS_ENV"
                ;;
        esac
        warn_if_shadowed
    else
        printf "${YELLOW}Note:${NC} to use 'mistralrs' now, restart your terminal or run ${BOLD}. \"\$HOME/.cargo/env\"${NC}.\n\n"
    fi
}

# Main installation flow
main() {
    print_banner

    os=$(detect_os)
    info "Detected OS: $os"

    # Bifurcate only on where the binary comes from: a prebuilt download (no toolchain), or a
    # source build. Everything after (FFmpeg, examples, PATH guidance) is shared.
    # Set MISTRALRS_INSTALL_FROM_SOURCE=1 to force a source build.
    method=""
    if [ -z "$MISTRALRS_INSTALL_FROM_SOURCE" ]; then
        info "Checking for a prebuilt binary for your platform..."
        asset=$(detect_prebuilt_asset "$os")
        if [ -n "$asset" ]; then
            info "Prebuilt available: $asset"
            if install_prebuilt "$asset"; then
                method="prebuilt"
            else
                legacy_asset=$(detect_legacy_cuda_prebuilt_asset "$os")
                if [ -n "$legacy_asset" ] && [ "$legacy_asset" != "$asset" ]; then
                    warn "Versioned CUDA prebuilt was not found; trying legacy prebuilt name $legacy_asset."
                    if install_prebuilt "$legacy_asset"; then
                        method="prebuilt"
                    fi
                fi
                if [ "$method" != "prebuilt" ]; then
                    warn "Prebuilt install failed; building from source instead."
                fi
            fi
        else
            info "No prebuilt for this platform; building from source."
        fi
    fi

    if [ "$method" != "prebuilt" ]; then
        build_from_source "$os"
        method="source"
    fi

    maybe_install_ffmpeg "$os"
    print_success "$method"
}

main "$@"
