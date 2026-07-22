#Requires -Version 5.1
# mistral.rs Installation Script for Windows
# Automatic hardware detection and feature configuration

$ErrorActionPreference = "Stop"
$RemoteSizeTimeoutSec = 5

# Color output functions
function Write-Info { Write-Host "info: $args" -ForegroundColor Blue }
function Write-Success { Write-Host "success: $args" -ForegroundColor Green }
function Write-Warn { Write-Host "warning: $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "error: $args" -ForegroundColor Red; exit 1 }
# Echo a dependency-install command (rustup, ...) before running it.
function Show-Cmd($cmd) { Write-Host "  > $cmd" -ForegroundColor DarkGray }

# Format a byte count for humans (e.g. 1234567 -> 1.2 MiB).
function Format-ByteSize([long]$Bytes) {
    $units = @("B", "KiB", "MiB", "GiB", "TiB")
    $value = [double]$Bytes
    $i = 0
    while ($value -ge 1024 -and $i -lt ($units.Count - 1)) { $value /= 1024; $i++ }
    if ($i -eq 0) { return "{0} {1}" -f [long]$value, $units[$i] }
    return "{0:N1} {1}" -f $value, $units[$i]
}

# Content-Length of a URL after redirects; $null if it cannot be determined.
function Get-RemoteDownloadSize($Url) {
    try {
        $head = Invoke-WebRequest -Uri $Url -Method Head -UseBasicParsing -TimeoutSec $RemoteSizeTimeoutSec -ErrorAction Stop
        $len = @($head.Headers['Content-Length']) | Select-Object -Last 1
        if ($len) { return [long]$len }
    } catch {}
    return $null
}

# MISTRALRS_INSTALL_YES=1 auto-confirms every prompt (non-interactive installs, `mistralrs update`).
function Read-Confirm($prompt) {
    if ($env:MISTRALRS_INSTALL_YES -eq "1") { return "y" }
    return Read-Host $prompt
}

function Add-UserPath($PathToAdd) {
    $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $Entries = @()
    if ($UserPath) {
        $Entries = $UserPath -split ';' | Where-Object { $_ }
    }
    if ($Entries -notcontains $PathToAdd) {
        $NewPath = @($PathToAdd) + $Entries
        [Environment]::SetEnvironmentVariable('Path', ($NewPath -join ';'), 'User')
    }
    if (($env:PATH -split ';') -notcontains $PathToAdd) {
        $env:PATH = "$PathToAdd;$env:PATH"
    }
}

function Warn-IfShadowed {
    param([string]$ExpectedBin, [string]$ExpectedInstall)

    $Command = Get-Command mistralrs -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $Command) { return }

    $Resolved = $Command.Source
    if (-not $Resolved) { $Resolved = $Command.Path }
    if (-not $Resolved) { return }

    if (($Resolved -ine $ExpectedBin) -and ($Resolved -ine $ExpectedInstall)) {
        Write-Warn "Another mistralrs appears earlier on PATH: $Resolved"
        Write-Host "      The managed install is available at: $ExpectedInstall"
    }
}

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
$RequiredRustVersion = "1.94"
$MistralRsRepoUrl = "https://github.com/EricLBuehler/mistral.rs"
$MistralRsBranch = "master"
$MistralRsCliPackage = "mistralrs-cli"

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

# CUDA toolkit version as major*100+minor (e.g. 13.1 -> 1301). $null if nvcc is unavailable.
function Get-CudaVersionCode {
    try {
        $null = Get-Command nvcc -ErrorAction Stop
        $output = & nvcc --version 2>$null | Out-String
        if ($output -match "release (\d+)\.(\d+)") {
            return [int]$Matches[1] * 100 + [int]$Matches[2]
        }
    } catch {}
    return $null
}

function Get-CudaDriverVersionCode {
    try {
        $null = Get-Command nvidia-smi -ErrorAction Stop
        $output = & nvidia-smi 2>$null | Out-String
        if ($output -match "CUDA Version:\s*(\d+)\.(\d+)") {
            return [int]$Matches[1] * 100 + [int]$Matches[2]
        }
    } catch {}
    return $null
}

function Format-CudaVersionCode($Code) {
    if ($null -eq $Code) { return "unknown" }
    return "{0}.{1}" -f [math]::Floor($Code / 100), ($Code % 100)
}

function Test-CuTileSupported {
    param([int]$CudaVersionCode, [int]$CudaComputeCap)

    if (($CudaComputeCap -ge 80) -and ($CudaComputeCap -lt 90) -and ($CudaVersionCode -ge 1302)) {
        return $true
    }
    if (($CudaComputeCap -eq 90) -and ($CudaVersionCode -ge 1303)) {
        return $true
    }
    if (($CudaComputeCap -ge 100) -and ($CudaVersionCode -ge 1302)) {
        return $true
    }
    return $false
}

function Test-CudaSourceBuildVersions {
    $cudaCC = Get-CudaComputeCap
    if (-not $cudaCC) { return }

    $cudaVer = Get-CudaVersionCode
    $driverCuda = Get-CudaDriverVersionCode
    if (($null -eq $cudaVer) -or ($null -eq $driverCuda)) { return }

    if ($cudaVer -gt $driverCuda) {
        if ($env:MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH -eq "1") {
            Write-Warn "Local nvcc CUDA $(Format-CudaVersionCode $cudaVer) is newer than the NVIDIA driver supports ($(Format-CudaVersionCode $driverCuda)); continuing because MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH=1."
        } else {
            Write-Err "Local nvcc CUDA $(Format-CudaVersionCode $cudaVer) is newer than the NVIDIA driver supports ($(Format-CudaVersionCode $driverCuda)). Source builds can fail with CUDA_ERROR_UNSUPPORTED_PTX_VERSION; upgrade the driver, install a matching CUDA toolkit, use a prebuilt, or set MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH=1 to override."
        }
    }
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
        "C:\Program Files\NVIDIA\CUDNN\*\bin\*\x64\cudnn*.dll"
    )

    foreach ($pattern in $cudnnPaths) {
        if (Get-Item $pattern -ErrorAction SilentlyContinue) {
            return $true
        }
    }
    return $false
}

# Check if NCCL is installed
function Test-NCCL {
    $ncclPaths = @(
        "$env:NCCL_ROOT\bin\nccl*.dll",
        "$env:NCCL_ROOT\lib\nccl*.lib",
        "$env:NCCL_HOME\bin\nccl*.dll",
        "$env:NCCL_HOME\lib\nccl*.lib",
        "$env:CUDA_PATH\bin\nccl*.dll",
        "$env:CUDA_PATH\lib\x64\nccl*.lib",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nccl*.dll",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\lib\x64\nccl*.lib"
    )

    foreach ($pattern in $ncclPaths) {
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

        if ($env:MISTRALRS_INSTALL_NO_NCCL -eq "1") {
            Write-Info "MISTRALRS_INSTALL_NO_NCCL=1 set - skipping nccl"
        } elseif (Test-NCCL) {
            $features += "nccl"
            Write-Info "NCCL detected - enabling nccl for CUDA multi-GPU tensor parallelism"
        } elseif ($env:MISTRALRS_INSTALL_NCCL -eq "1") {
            $features += "nccl"
            Write-Warn "MISTRALRS_INSTALL_NCCL=1 set but NCCL was not detected; the build may fail unless NCCL is on the linker path"
        } else {
            Write-Warn "NCCL not found - skipping nccl. Install NCCL or set MISTRALRS_INSTALL_NCCL=1 to force it; NCCL is the preferred CUDA multi-GPU path."
        }

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

        $cudaVer = Get-CudaVersionCode
        $ccNum = [int]$cudaCC
        if ($cudaVer -and (Test-CuTileSupported -CudaVersionCode $cudaVer -CudaComputeCap $ccNum)) {
            $features += "cutile"
            Write-Info "CUDA $(Format-CudaVersionCode $cudaVer) and supported arch - enabling cutile"
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
    $script:SourceInstallRoot = Join-Path $env:TEMP "mistralrs-cargo-install-$([guid]::NewGuid().ToString())"
    $script:SourceMistralRs = Join-Path $script:SourceInstallRoot "bin\mistralrs.exe"

    # MISTRALRS_INSTALL_TAG pins a git tag; otherwise build the latest master.
    if ($env:MISTRALRS_INSTALL_TAG) {
        $gitRef = @("--tag", $env:MISTRALRS_INSTALL_TAG)
        $refDesc = "tag $($env:MISTRALRS_INSTALL_TAG)"
    } else {
        $gitRef = @("--branch", $MistralRsBranch)
        $refDesc = "branch $MistralRsBranch"
    }

    if ($Features) {
        Write-Info "Installing mistralrs-cli from GitHub $refDesc with features: $Features"
        & cargo install --root $script:SourceInstallRoot --force --locked --git $MistralRsRepoUrl @gitRef $MistralRsCliPackage --features "$Features"
    } else {
        Write-Info "Installing mistralrs-cli from GitHub $refDesc with default features"
        & cargo install --root $script:SourceInstallRoot --force --locked --git $MistralRsRepoUrl @gitRef $MistralRsCliPackage
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Err "Installation failed"
    }
}

# MISTRALRS_INSTALL_TAG pins a specific release (e.g. v0.8.9); default is the latest stable release.
$ReleaseBase = if ($env:MISTRALRS_INSTALL_TAG) {
    "https://github.com/EricLBuehler/mistral.rs/releases/download/$($env:MISTRALRS_INSTALL_TAG)"
} else {
    "https://github.com/EricLBuehler/mistral.rs/releases/latest/download"
}
$PrebuiltDir = "$env:USERPROFILE\.mistralrs"
$BinDir = "$env:USERPROFILE\.local\bin"
$CargoBinDir = if ($env:CARGO_HOME) { Join-Path $env:CARGO_HOME "bin" } else { "$env:USERPROFILE\.cargo\bin" }
$CargoMistralRs = Join-Path $CargoBinDir "mistralrs.exe"
$ManagedBin = Join-Path $PrebuiltDir "mistralrs.exe"
$LauncherPath = Join-Path $BinDir "mistralrs.cmd"
$LegacyBinPath = Join-Path $BinDir "mistralrs.exe"
$script:SourceInstallRoot = $null
$script:SourceMistralRs = $null
$script:ReplaceDuplicateInstalls = $false
$script:DuplicateInstalls = @()

function Normalize-InstallPath {
    param([string]$Path)
    try {
        return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    } catch {
        return $Path
    }
}

function Test-SameInstallPath {
    param([string]$Left, [string]$Right)
    return [string]::Equals((Normalize-InstallPath $Left), (Normalize-InstallPath $Right), [StringComparison]::OrdinalIgnoreCase)
}

function Add-DuplicateInstall {
    param([string]$Path)
    if (-not $Path) { return }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    if ((Test-SameInstallPath $Path $ManagedBin) -or (Test-SameInstallPath $Path $LauncherPath)) { return }
    $normalized = Normalize-InstallPath $Path
    foreach ($existing in $script:DuplicateInstalls) {
        if (Test-SameInstallPath $existing $normalized) { return }
    }
    $script:DuplicateInstalls += $normalized
}

function Find-DuplicateInstalls {
    $script:DuplicateInstalls = @()
    Add-DuplicateInstall $CargoMistralRs
    Add-DuplicateInstall $LegacyBinPath
    foreach ($dir in ($env:PATH -split ';' | Where-Object { $_ })) {
        Add-DuplicateInstall (Join-Path $dir "mistralrs.exe")
        Add-DuplicateInstall (Join-Path $dir "mistralrs.cmd")
        Add-DuplicateInstall (Join-Path $dir "mistralrs.bat")
    }
    return $script:DuplicateInstalls
}

function Confirm-DuplicateReplacement {
    $duplicates = @(Find-DuplicateInstalls)
    if ($duplicates.Count -eq 0) { return }
    Write-Host ""
    Write-Warn "Found duplicate mistralrs installs:"
    foreach ($duplicate in $duplicates) {
        Write-Host "  $duplicate"
    }
    Write-Host ""
    $response = Read-Confirm "Replace duplicate installs? [Y/n]"
    if ($response -match "^[Nn]") {
        Write-Err "duplicate mistralrs installs must be resolved before installing"
    }
    $script:ReplaceDuplicateInstalls = $true
}

function Remove-DuplicateInstalls {
    if (-not $script:ReplaceDuplicateInstalls) { return }
    $duplicates = @(Find-DuplicateInstalls)
    foreach ($duplicate in $duplicates) {
        try {
            Remove-Item -LiteralPath $duplicate -Force
            Write-Info "Removed duplicate install: $duplicate"
        } catch {
            Write-Err "failed to remove duplicate install: $duplicate"
        }
    }
}

function Install-Launcher {
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $cmdPath = $ManagedBin -replace '%', '%%'
    Set-Content -LiteralPath $LauncherPath -Encoding ASCII -Value @(
        "@echo off",
        "`"$cmdPath`" %*"
    )
}

function Install-SourceFromStaging {
    if (-not (Test-Path -LiteralPath $script:SourceMistralRs -PathType Leaf)) {
        Write-Err "cargo install succeeded but $script:SourceMistralRs was not found"
    }
    if (Test-Path $PrebuiltDir) { Remove-Item -Recurse -Force $PrebuiltDir }
    New-Item -ItemType Directory -Force -Path $PrebuiltDir | Out-Null
    Copy-Item -Force $script:SourceMistralRs $ManagedBin
    Remove-Item -Recurse -Force $script:SourceInstallRoot
    Install-Launcher
    & $ManagedBin --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Err "source-built binary did not run after installation"
    }
}

function Write-InstallSuccess {
    param([string]$Method)
    $ver = (& $ManagedBin --version 2>$null | Select-Object -First 1)
    if (-not $ver) { $ver = "mistral.rs" }
    if ($Method -eq "prebuilt") {
        Write-Success "$ver installed successfully (prebuilt binary)!"
    } else {
        Write-Success "$ver installed successfully (built from source)!"
    }
    Write-Host ""
    Write-Host "Installed" -ForegroundColor White
    Write-Host "========="
    Write-Host "  binary   $ManagedBin"
    Write-Host "  on PATH  $LauncherPath"
    Write-Host ""
    if (($env:PATH -split ';') -notcontains $BinDir) {
        Add-UserPath $BinDir
        Write-Warn "Added $BinDir to your user PATH. Restart your terminal to use 'mistralrs'."
    }
    Warn-IfShadowed $LauncherPath $ManagedBin
    Write-Host ""
    Write-Host "  mistralrs run -m Qwen/Qwen3-4B"
    Write-Host ""
}

# Download and install the Windows CPU prebuilt. Returns $true on success.
function Install-Prebuilt {
    $asset = "mistralrs-cpu-x86_64-pc-windows-msvc.zip"
    $tmp = Join-Path $env:TEMP $asset
    $downloadSize = Get-RemoteDownloadSize "$ReleaseBase/$asset"
    if ($downloadSize) {
        Write-Info "Downloading $asset ($(Format-ByteSize $downloadSize))"
    } else {
        Write-Info "Downloading $asset"
    }
    try {
        # Start-BitsTransfer shows a native progress bar and is fast; fall back to IWR with its bar.
        try {
            Start-BitsTransfer -Source "$ReleaseBase/$asset" -Destination $tmp -ErrorAction Stop
        } catch {
            $ProgressPreference = 'Continue'
            Invoke-WebRequest -Uri "$ReleaseBase/$asset" -OutFile $tmp -UseBasicParsing
        }
    } catch {
        return $false
    }
    try {
        if (Test-Path $PrebuiltDir) { Remove-Item -Recurse -Force $PrebuiltDir }
        New-Item -ItemType Directory -Force -Path $PrebuiltDir | Out-Null
        Expand-Archive -Path $tmp -DestinationPath $PrebuiltDir -Force
        Remove-Item -Force $tmp
        Install-Launcher
        & $ManagedBin --version *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
    } catch {
        return $false
    }
    return $true
}

# Main installation flow
function Main {
    Show-Banner

    Write-Info "Detected OS: Windows"
    Confirm-DuplicateReplacement

    # Prefer the prebuilt CPU binary: no Rust toolchain, no compile. Set
    # MISTRALRS_INSTALL_FROM_SOURCE=1 to force a source build instead.
    if (-not $env:MISTRALRS_INSTALL_FROM_SOURCE) {
        Write-Info "Checking for a prebuilt binary..."
        if (Install-Prebuilt) {
            Remove-DuplicateInstalls
            Write-InstallSuccess "prebuilt"
            return
        }
        Write-Warn "Prebuilt install failed; building from source instead."
        Write-Host ""
    }

    if ($env:MISTRALRS_INSTALL_TAG) {
        Write-Info "Building from source: tag $($env:MISTRALRS_INSTALL_TAG)."
    } else {
        Write-Info "Building from source: latest $MistralRsBranch (bleeding edge)."
    }

    # Check for Rust
    if (Test-Rust) {
        $rustVersionFull = & rustc --version 2>$null
        $rustVersion = Get-RustVersion
        Write-Info "Rust is installed: $rustVersionFull"

        # Check if version meets minimum requirement
        if ($rustVersion -and -not (Test-VersionGte $rustVersion $RequiredRustVersion)) {
            Write-Warn "Rust $rustVersion is below the required version $RequiredRustVersion"
            Write-Host ""
            Show-Cmd "rustup update stable"
            $response = Read-Confirm "Would you like to update Rust now? [Y/n]"
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
        Show-Cmd "Invoke-WebRequest https://win.rustup.rs/x86_64 -OutFile rustup-init.exe; .\rustup-init.exe -y"
        $response = Read-Confirm "Would you like to install Rust now? [Y/n]"
        if ($response -match "^[Nn]") {
            Write-Err "Rust is required to install mistral.rs"
        }
        Install-Rust
    }

    Write-Host ""
    Write-Info "Detecting hardware capabilities..."
    Test-CudaSourceBuildVersions

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
    $response = Read-Confirm "Proceed with installation? [Y/n]"
    if ($response -match "^[Nn]") {
        Write-Info "Installation cancelled"
        exit 0
    }

    Write-Host ""
    Install-MistralRS -Features $features
    Install-SourceFromStaging
    Remove-DuplicateInstalls

    Write-Host ""
    Write-InstallSuccess "source"
}

# Run main
Main
