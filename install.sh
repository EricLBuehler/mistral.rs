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
    printf "${NC}${BLUE}Blazingly fast LLM inference.${NC}\n"
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

# Check if Rust is installed
check_rust() {
    command -v cargo >/dev/null 2>&1
}

# Install Rust via rustup
install_rust() {
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
    success "Rust installed successfully"
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
        features="metal accelerate"
        info "macOS detected - enabling metal, accelerate"
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
        cargo install mistralrs-cli --features "$features"
    else
        info "Installing mistralrs-cli with default features"
        cargo install mistralrs-cli
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
        rust_version=$(rustc --version 2>/dev/null || echo "unknown")
        info "Rust is installed: $rust_version"
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
    printf "${YELLOW}Note:${NC} Restart your terminal to use the 'mistralrs' command.\n"
}

main "$@"
