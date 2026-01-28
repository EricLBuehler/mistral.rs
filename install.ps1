#Requires -Version 5.1
# mistral.rs Installation Script for Windows
# Automatic hardware detection and feature configuration

$ErrorActionPreference = "Stop"

# Color output functions
function Write-Info { Write-Host "info: $args" -ForegroundColor Blue }
function Write-Success { Write-Host "success: $args" -ForegroundColor Green }
function Write-Warn { Write-Host "warning: $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "error: $args" -ForegroundColor Red; exit 1 }

# Banner
function Show-Banner {
    Write-Host ""
    Write-Host "  __  __ _     _             _              " -ForegroundColor Cyan
    Write-Host " |  \/  (_)___| |_ _ __ __ _| |  _ __ ___   " -ForegroundColor Cyan
    Write-Host " | |\/| | / __| __| '__/ `` | | | '__/ __|  " -ForegroundColor Cyan
    Write-Host " | |  | | \__ \ |_| | | (_| | |_| |  \__ \  " -ForegroundColor Cyan
    Write-Host " |_|  |_|_|___/\__|_|  \__,_|_(_)_|  |___/  " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Fast, flexible LLM inference." -ForegroundColor Blue
    Write-Host ""
}

# Minimum required Rust version (from Cargo.toml rust-version)
$RequiredRustVersion = "1.88"

# Check if Rust is installed
function Test-Rust {
    try {
        $null = Get-Command cargo -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Get installed Rust version (major.minor)
function Get-RustVersion {
    try {
        $output = & rustc --version 2>$null
        if ($output -match 'rustc (\d+\.\d+)') {
            return $matches[1]
        }
    } catch {}
    return $null
}

# Compare two version strings (returns $true if $v1 >= $v2)
function Test-VersionGte {
    param([string]$v1, [string]$v2)

    $v1Parts = $v1 -split '\.'
    $v2Parts = $v2 -split '\.'

    $v1Major = [int]$v1Parts[0]
    $v1Minor = [int]$v1Parts[1]
    $v2Major = [int]$v2Parts[0]
    $v2Minor = [int]$v2Parts[1]

    if ($v1Major -gt $v2Major) {
        return $true
    } elseif ($v1Major -eq $v2Major -and $v1Minor -ge $v2Minor) {
        return $true
    }
    return $false
}

# Install Rust via rustup
function Install-Rust {
    Write-Info "Installing Rust via rustup..."
    $rustupInit = "$env:TEMP\rustup-init.exe"

    try {
        Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInit -UseBasicParsing
        & $rustupInit -y

        # Add cargo to PATH for current session
        $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

        Write-Success "Rust installed successfully"
    } catch {
        Write-Err "Failed to install Rust: $_"
    }
}

# Update Rust to latest version
function Update-Rust {
    Write-Info "Updating Rust to latest version..."
    try {
        & rustup update stable
        Write-Success "Rust updated successfully"
    } catch {
        Write-Err "Failed to update Rust: $_"
    }
}

# Get CUDA compute capability
function Get-CudaComputeCap {
    try {
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
        $output = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($output) {
            $cc = ($output -split "`n")[0].Trim() -replace '\.',''
            return $cc
        }
    } catch {}
    return $null
}

# Check if MKL is installed
function Test-MKL {
    if ($env:MKLROOT -and (Test-Path $env:MKLROOT)) {
        return $true
    }

    $mklPaths = @(
        "C:\Program Files (x86)\Intel\oneAPI\mkl\latest",
        "C:\Program Files\Intel\oneAPI\mkl\latest",
        "$env:USERPROFILE\intel\oneapi\mkl\latest"
    )

    foreach ($path in $mklPaths) {
        if (Test-Path $path) {
            return $true
        }
    }
    return $false
}

# Check if CPU is Intel
function Test-IntelCpu {
    try {
        $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
        return $cpu.Manufacturer -match "Intel" -or $cpu.Name -match "Intel"
    } catch {
        return $false
    }
}

# Check if cuDNN is installed
function Test-CuDNN {
    # Check common cuDNN library paths on Windows
    $cudnnPaths = @(
        "$env:CUDA_PATH\bin\cudnn*.dll",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\cudnn*.dll",
        "C:\Program Files\NVIDIA\CUDNN\*\bin\cudnn*.dll"
    )

    foreach ($pattern in $cudnnPaths) {
        if (Get-Item $pattern -ErrorAction SilentlyContinue) {
            return $true
        }
    }
    return $false
}

# Build feature string based on detected hardware
function Get-Features {
    $features = @()

    # Check for CUDA
    $cudaCC = Get-CudaComputeCap
    if ($cudaCC) {
        $features += "cuda"

        $ccMajor = $cudaCC.Substring(0, 1)
        $ccMinor = if ($cudaCC.Length -gt 1) { $cudaCC.Substring(1) } else { "0" }
        Write-Info "CUDA detected (compute capability: $ccMajor.$ccMinor)"

        # Check for cuDNN
        if (Test-CuDNN) {
            $features += "cudnn"
            Write-Info "cuDNN detected - enabling cudnn"
        } else {
            Write-Info "cuDNN not found - skipping cudnn feature"
        }

        # Add flash attention based on compute capability
        if ($cudaCC -eq "90") {
            $features += "flash-attn-v3"
            Write-Info "Hopper GPU detected - enabling flash-attn-v3"
        } elseif ([int]$cudaCC -ge 80) {
            $features += "flash-attn"
            Write-Info "Ampere+ GPU detected - enabling flash-attn"
        }
    } else {
        Write-Info "No NVIDIA GPU detected"
    }

    # Check for MKL on Intel
    if ((Test-IntelCpu) -and (Test-MKL)) {
        $features += "mkl"
        Write-Info "Intel MKL detected - enabling mkl"
    }

    return $features -join " "
}

# Install mistralrs-cli
function Install-MistralRS {
    param([string]$Features)

    if ($Features) {
        Write-Info "Installing mistralrs-cli with features: $Features"
        & cargo install mistralrs-cli@0.7.0 --features "$Features"
    } else {
        Write-Info "Installing mistralrs-cli with default features"
        & cargo install mistralrs-cli@0.7.0
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Err "Installation failed"
    }
}

# Main installation flow
function Main {
    Show-Banner

    Write-Info "Detected OS: Windows"

    # Check for Rust
    if (Test-Rust) {
        $rustVersionFull = & rustc --version 2>$null
        $rustVersion = Get-RustVersion
        Write-Info "Rust is installed: $rustVersionFull"

        # Check if version meets minimum requirement
        if ($rustVersion -and -not (Test-VersionGte $rustVersion $RequiredRustVersion)) {
            Write-Warn "Rust $rustVersion is below the required version $RequiredRustVersion"
            Write-Host ""
            $response = Read-Host "Would you like to update Rust now? [Y/n]"
            if ($response -match "^[Nn]") {
                Write-Err "Rust $RequiredRustVersion or newer is required to install mistral.rs"
            }
            Update-Rust
            # Re-check version after update
            $rustVersion = Get-RustVersion
            if (-not (Test-VersionGte $rustVersion $RequiredRustVersion)) {
                Write-Err "Failed to update Rust to required version $RequiredRustVersion"
            }
        }
    } else {
        Write-Warn "Rust is not installed"
        Write-Host ""
        $response = Read-Host "Would you like to install Rust now? [Y/n]"
        if ($response -match "^[Nn]") {
            Write-Err "Rust is required to install mistral.rs"
        }
        Install-Rust
    }

    Write-Host ""
    Write-Info "Detecting hardware capabilities..."

    # Build features
    $features = Get-Features

    Write-Host ""
    Write-Host "Installation Summary" -ForegroundColor White
    Write-Host "====================" -ForegroundColor White
    if ($features) {
        Write-Host "Features: " -NoNewline
        Write-Host $features -ForegroundColor Green
    } else {
        Write-Host "Features: " -NoNewline
        Write-Host "(none - CPU only)" -ForegroundColor Yellow
    }
    Write-Host ""

    # Confirm installation
    $response = Read-Host "Proceed with installation? [Y/n]"
    if ($response -match "^[Nn]") {
        Write-Info "Installation cancelled"
        exit 0
    }

    Write-Host ""
    Install-MistralRS -Features $features

    Write-Host ""
    Write-Success "mistral.rs installed successfully!"
    Write-Host ""
    Write-Host "Quick Start" -ForegroundColor White
    Write-Host "===========" -ForegroundColor White
    Write-Host ""
    Write-Host "  mistralrs run -m Qwen/Qwen3-4B"
    Write-Host ""
    Write-Host "  mistralrs serve --ui -m google/gemma-3-4b-it"
    Write-Host ""
    Write-Host "For more information, visit: https://github.com/EricLBuehler/mistral.rs"
    Write-Host ""
    Write-Host "Note: " -ForegroundColor Yellow -NoNewline
    Write-Host "Restart your terminal to use the 'mistralrs' command."
}

# Run main
Main
