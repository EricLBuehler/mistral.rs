#!/bin/bash
set -e

# mistral.rs Installation Script
# Cross-platform installer for Linux and macOS with automatic hardware detection

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
info() { echo -e "${BLUE}info:${NC} $1"; }
success() { echo -e "${GREEN}success:${NC} $1"; }
warn() { echo -e "${YELLOW}warning:${NC} $1"; }
error() { echo -e "${RED}error:${NC} $1"; exit 1; }

# Banner
print_banner() {
    echo -e "${BOLD}"
    echo "  __  __ _     _             _              "
    echo " |  \/  (_)___| |_ _ __ __ _| |  _ __ ___   "
    echo " | |\/| | / __| __| '__/ _\` | | | '__/ __|  "
    echo " | |  | | \__ \ |_| | | (_| | |_| |  \__ \  "
    echo " |_|  |_|_|___/\__|_|  \__,_|_(_)_|  |___/  "
    echo ""
    echo -e "${NC}${BLUE}Blazing fast LLM inference engine${NC}"
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
    if command -v cargo &> /dev/null; then
        return 0
    fi
    return 1
}

# Install Rust via rustup
install_rust() {
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    success "Rust installed successfully"
}

# Detect CUDA compute capability
detect_cuda_compute_cap() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo ""
        return
    fi

    # Try direct query
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

    if [[ -n "$cc" ]]; then
        echo "$cc"
    fi
}

# Check if MKL is installed
detect_mkl() {
    # Check MKLROOT environment variable
    if [[ -n "$MKLROOT" ]] && [[ -d "$MKLROOT" ]]; then
        return 0
    fi

    # Check common installation paths
    local mkl_paths=(
        "/opt/intel/oneapi/mkl/latest"
        "/opt/intel/mkl"
        "/opt/intel/oneapi/mkl"
    )

    for path in "${mkl_paths[@]}"; do
        if [[ -d "$path" ]]; then
            return 0
        fi
    done

    return 1
}

# Check if CPU is Intel
is_intel_cpu() {
    if [[ -f /proc/cpuinfo ]]; then
        grep -qi "intel" /proc/cpuinfo && return 0
    elif command -v sysctl &> /dev/null; then
        sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "intel" && return 0
    fi
    return 1
}

# Build feature string based on detected hardware
build_features() {
    local os="$1"
    local features=""

    if [[ "$os" == "macos" ]]; then
        features="metal accelerate"
        info "macOS detected - enabling metal, accelerate"
    else
        # Check for CUDA
        local cuda_cc
        cuda_cc=$(detect_cuda_compute_cap)
        if [[ -n "$cuda_cc" ]]; then
            features="cuda cudnn nccl"
            info "CUDA detected (compute capability: ${cuda_cc:0:1}.${cuda_cc:1})"

            if [[ "$cuda_cc" == "90" ]]; then
                features="$features flash-attn-v3"
                info "Hopper GPU detected - enabling flash-attn-v3"
            elif [[ "$cuda_cc" -ge 80 ]]; then
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
    local features="$1"

    if [[ -n "$features" ]]; then
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
    local os
    os=$(detect_os)
    info "Detected OS: $os"

    # Check for Rust
    if check_rust; then
        local rust_version
        rust_version=$(rustc --version 2>/dev/null || echo "unknown")
        info "Rust is installed: $rust_version"
    else
        warn "Rust is not installed"
        echo ""
        read -p "Would you like to install Rust now? [Y/n] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            error "Rust is required to install mistral.rs"
        fi
        install_rust
    fi

    echo ""
    info "Detecting hardware capabilities..."

    # Build features
    local features
    features=$(build_features "$os")

    echo ""
    echo -e "${BOLD}Installation Summary${NC}"
    echo "===================="
    if [[ -n "$features" ]]; then
        echo -e "Features: ${GREEN}$features${NC}"
    else
        echo -e "Features: ${YELLOW}(none - CPU only)${NC}"
    fi
    echo ""

    # Confirm installation
    read -p "Proceed with installation? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Installation cancelled"
        exit 0
    fi

    echo ""
    install_mistralrs "$features"

    echo ""
    success "mistral.rs installed successfully!"
    echo ""
    echo -e "${BOLD}Quick Start${NC}"
    echo "==========="
    echo "  # Run a model interactively"
    echo "  mistralrs run --isq 4 -m google/gemma-3-4b-it"
    echo ""
    echo "  # Start the OpenAI-compatible server with the builtin UI"
    echo "  mistralrs serve --ui -m Qwen/Qwen3-4B"
    echo ""
    echo "For more information, visit: https://github.com/EricLBuehler/mistral.rs"
}

main "$@"
