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
REQUIRED_RUST_VERSION="1.88"

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

# Build feature string based on detected hardware
build_features() {
    os="$1"
    features=""

    if [ "$os" = "macos" ]; then
        check_xcode_cli_tools
        check_metal_toolchain
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

# Install mistralrs-cli
install_mistralrs() {
    features="$1"

    if [ -n "$features" ]; then
        info "Installing mistralrs-cli with features: $features"
        cargo install mistralrs-cli@0.7.0 --features "$features"
    else
        info "Installing mistralrs-cli with default features"
        cargo install mistralrs-cli@0.7.0
    fi
}

# Main installation flow
main() {
    print_banner

    # Detect OS
    os=$(detect_os)
    info "Detected OS: $os"

    # Check for Rust
    if check_rust; then
        rust_version_full=$(rustc --version 2>/dev/null || echo "unknown")
        rust_version=$(get_rust_version)
        info "Rust is installed: $rust_version_full"

        # Check if version meets minimum requirement
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
            # Re-check version after update
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
                error "Rust is required to install mistral.rs"
                ;;
        esac
        install_rust
    fi

    echo ""
    info "Detecting hardware capabilities..."

    # Build features
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

    # Confirm installation
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

    # Ensure cargo bin is in PATH for this session
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi

    echo ""
    success "mistral.rs installed successfully!"
    echo ""
    printf "${BOLD}Quick Start${NC}\n"
    echo "==========="
    echo ""
    echo "  mistralrs run -m Qwen/Qwen3-4B"
    echo ""
    echo "  mistralrs serve --ui -m google/gemma-3-4b-it"
    echo ""
    echo "For more information, visit: https://github.com/EricLBuehler/mistral.rs"
    echo ""
    printf "${YELLOW}Note:${NC} To use 'mistralrs' now, run: ${BOLD}. \"\$HOME/.cargo/env\"${NC}\n"
    printf "      Or restart your terminal.\n"
}

main "$@"
