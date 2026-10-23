#Requires -Version 5.1
param(
    [string]$Release,
    [string]$Backend,
    [switch]$FromSource,
    [switch]$WithFFmpeg,
    [switch]$NoNccl,
    [switch]$AllowCudaMismatch,
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$Verbose,
    [switch]$Help
)

if ($args.Count) { throw "Unknown installer arguments: $($args -join ' ')" }

# A private scope keeps installer helpers and preferences out of the caller's session.
New-Module -ArgumentList @($PSBoundParameters) -ScriptBlock {
param([hashtable]$InstallerArguments)
$ErrorActionPreference = 'Stop'

function Initialize-Installer([hashtable]$Options = @{}) {
    $script:RequiredRustVersion = [version]'1.94.0'
    $script:Repository = 'https://github.com/EricLBuehler/mistral.rs'
    $script:PollMilliseconds = 100
    $script:OutputBufferSize = 8192
    $script:DownloadBufferSize = 65536
    $script:StageCount = 4
    $script:Utf8 = New-Object System.Text.UTF8Encoding($false)
    $profileDir = $env:USERPROFILE
    if (-not $profileDir) { $profileDir = [Environment]::GetFolderPath('UserProfile') }
    $script:ProfileDir = $profileDir
    $script:PrebuiltDir = Join-Path $profileDir '.mistralrs'
    $script:BinDir = Join-Path $profileDir '.local/bin'
    $script:ManagedBin = Join-Path $script:PrebuiltDir 'mistralrs.exe'
    $script:LauncherPath = Join-Path $script:BinDir 'mistralrs.cmd'
    $script:ConfigPath = Join-Path $script:PrebuiltDir 'install.conf'
    $script:Options = @{
        Release = 'latest'; Backend = 'auto'; Method = 'auto'
        Video = $false; NoNccl = $false; AllowCudaMismatch = $false
        Yes = $false; DryRun = $false; Verbose = $false; Help = $false
    }
    $script:TempPaths = New-Object 'System.Collections.Generic.List[string]'
    $script:LogPath = $null
    $script:ActiveProcess = $null
    $script:Activation = $null
    $script:ConsoleState = $null
    $script:Stage = ''
    $script:LastCommandText = ''
    $script:ProgressVisible = $false
    $script:Duplicates = @()
    $script:ReplaceDuplicates = $false
    $script:Installed = $false
    $script:BuildEnvironment = @{}
    $script:MsvcInstallPath = $null
    $script:Interactive = $false
    try { $script:Interactive = -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected } catch {}
    $script:UseColor = $script:Interactive -and $null -eq [Environment]::GetEnvironmentVariable('NO_COLOR')
    if (Test-Path -LiteralPath $script:ConfigPath -PathType Leaf) {
        foreach ($line in [IO.File]::ReadAllLines($script:ConfigPath)) {
            $parts = $line -split '=', 2
            if ($parts.Count -ne 2) { continue }
            switch ($parts[0]) {
                'release' { $script:Options.Release = $parts[1] }
                'backend' { $script:Options.Backend = $parts[1] }
                'method' { $script:Options.Method = $parts[1] }
                'no_nccl' { $script:Options.NoNccl = $parts[1] -eq '1' }
                'allow_cuda_mismatch' { $script:Options.AllowCudaMismatch = $parts[1] -eq '1' }
            }
        }
    }
    if ($env:MISTRALRS_INSTALL_FROM_SOURCE) {
        $script:Options.Method = 'source'
        $script:Options.Release = 'master'
    }
    if ($env:MISTRALRS_INSTALL_TAG) { $script:Options.Release = $env:MISTRALRS_INSTALL_TAG }
    if ($env:MISTRALRS_INSTALL_NO_NCCL) {
        $script:Options.NoNccl = $env:MISTRALRS_INSTALL_NO_NCCL -eq '1'
    }
    if ($env:MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH) {
        $script:Options.AllowCudaMismatch = $env:MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH -eq '1'
    }
    $script:Options.Yes = $env:MISTRALRS_INSTALL_YES -eq '1'
    foreach ($name in @('Release', 'Backend', 'NoNccl', 'AllowCudaMismatch', 'Yes', 'DryRun', 'Verbose', 'Help')) {
        if ($Options.ContainsKey($name)) { $script:Options[$name] = $Options[$name] }
    }
    if ($Options.FromSource) { $script:Options.Method = 'source' }
    if ($Options.WithFFmpeg) { $script:Options.Video = $true }
}

function Write-Line([string]$Text = '', [string]$Color = '') {
    if ($script:UseColor -and $Color) { Write-Host $Text -ForegroundColor $Color }
    else { Write-Host $Text }
}
function Write-Warn([string]$Text) { Write-Line "  Warning: $Text" Yellow }
function Write-Err([string]$Text) { throw $Text }
function Write-Log([string]$Text) {
    if ($script:LogPath) { [IO.File]::AppendAllText($script:LogPath, $Text + [Environment]::NewLine, $script:Utf8) }
}
function Show-Banner {
    Write-Line
    @(
        '  __  __ _     _             _              '
        ' |  \/  (_)___| |_ _ __ __ _| |  _ __ ___   '
        ' | |\/| | / __| __| ''__/ _` | | | ''__/ __|  '
        ' | |  | | \__ \ |_| | | (_| | |_| |  \__ \  '
        ' |_|  |_|_|___/\__|_|  \__,_|_(_)_|  |___/  '
    ) | ForEach-Object { Write-Line $_ }
    Write-Line
    Write-Line '  Install mistral.rs'
    Write-Line '  Fast, flexible LLM inference.' Blue
    Write-Line
}
function Show-Help {
    Write-Line @'
mistral.rs installer for Windows

  .\install.ps1 [-Release latest|master|TAG] [-Backend auto|cpu|cuda]
                [-FromSource] [-WithFFmpeg] [-NoNccl] [-AllowCudaMismatch]
                [-Yes] [-DryRun] [-Verbose] [-Help]

Without -Yes, review your installation and customize it with the arrow keys.
-DryRun previews the plan without downloading or changing anything.
-Verbose streams complete command output. Failed command logs are always shown.
Settings are saved in %USERPROFILE%\.mistralrs\install.conf.
'@
}
function Assert-Options {
    if ($script:Options.Release -notmatch '^[A-Za-z0-9][A-Za-z0-9._/-]*$') { throw 'Release must be latest, master, or a Git tag.' }
    if ($script:Options.Backend -notin @('auto', 'cpu', 'cuda')) { throw 'Backend must be auto, cpu, or cuda.' }
    if ($script:Options.Method -notin @('auto', 'source')) { throw 'Installation method must be auto or source.' }
}
function Get-Architecture {
    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    if ($arch -match '^(AMD64|x86_64)$') { return 'x86_64' }
    if ($arch -match '^(ARM64|aarch64)$') { return 'aarch64' }
    if (-not $arch) { return 'x86_64' }
    throw "Unsupported Windows architecture: $arch"
}
function Get-CudaComputeCap {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    try {
        $caps = @(& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null)
        if ($LASTEXITCODE -ne 0) { return $null }
        foreach ($cap in $caps) {
            if ("$cap".Trim() -match '^([0-9]+)\.([0-9]+)$') { return [int]($Matches[1] + $Matches[2]) }
        }
    } catch {}
    return $null
}
function Get-Plan {
    $backend = $script:Options.Backend
    if ($backend -eq 'auto') { $backend = if ($script:CudaCC) { 'cuda' } else { 'cpu' } }
    if ($backend -eq 'cuda' -and -not $script:CudaCC) { throw 'No supported NVIDIA GPU was detected. Choose CPU or install the NVIDIA driver first.' }
    $method = 'source'
    if ($script:Options.Method -ne 'source' -and $script:Options.Release -ne 'master' -and $backend -eq 'cpu' -and $script:Arch -eq 'x86_64') {
        $method = 'prebuilt'
    }
    $releaseLabel = switch ($script:Options.Release) {
        'latest' { 'Latest stable release' }
        'master' { 'Latest development version (master)' }
        default { $script:Options.Release }
    }
    [pscustomobject]@{
        Backend = $backend; Method = $method; ReleaseLabel = $releaseLabel
        Asset = 'mistralrs-cpu-x86_64-pc-windows-msvc.zip'
    }
}
function Save-Options {
    $lines = @(
        "release=$($script:Options.Release)", "backend=$($script:Options.Backend)"
        "method=$($script:Options.Method)", "no_nccl=$([int][bool]$script:Options.NoNccl)"
        "allow_cuda_mismatch=$([int][bool]$script:Options.AllowCudaMismatch)"
    )
    [IO.File]::WriteAllLines($script:ConfigPath, [string[]]$lines, $script:Utf8)
}
function Restore-Console {
    if ($script:ConsoleState) {
        try { [Console]::TreatControlCAsInput = $script:ConsoleState.ControlC } catch {}
        if ($null -ne $script:ConsoleState.CursorVisible) {
            try { [Console]::CursorVisible = $script:ConsoleState.CursorVisible } catch {}
        }
        $script:ConsoleState = $null
    }
}
function Read-MenuKey { return [Console]::ReadKey($true) }
function Select-Menu([string]$Title, [int]$Default, [string[]]$Choices) {
    if (-not $script:Interactive) {
        Write-Line $Title
        foreach ($choice in $Choices) { Write-Line ('  ' + $choice.Replace(':', '  ')) }
        if ([Console]::IsInputRedirected) { throw 'Interactive input is unavailable. Run this script in a terminal or pass -Yes.' }
        $answer = Read-Host 'Choose an option (Enter accepts the default)'
        if (-not $answer) { return $Default }
        for ($i = 0; $i -lt $Choices.Count; $i++) {
            if ($answer -eq ($Choices[$i] -split ':', 2)[0]) { return $i }
        }
        return -1
    }
    $cursorVisible = $null
    try { $cursorVisible = [Console]::CursorVisible } catch {}
    $script:ConsoleState = @{ ControlC = [Console]::TreatControlCAsInput; CursorVisible = $cursorVisible }
    $selected = $Default
    try {
        [Console]::TreatControlCAsInput = $true
        if ($null -ne $cursorVisible) { [Console]::CursorVisible = $false }
        while ($true) {
            Write-Line $Title
            Write-Line
            for ($i = 0; $i -lt $Choices.Count; $i++) {
                $parts = $Choices[$i] -split ':', 2
                $prefix = if ($i -eq $selected) { '  > ' } else { '    ' }
                $color = if ($i -eq $selected) { 'Cyan' } else { '' }
                Write-Line ($prefix + $parts[1] + ' [' + $parts[0] + ']') $color
            }
            Write-Line
            Write-Line '  Up/Down to move, Enter to select, or press a shortcut. Esc to go back.' DarkGray
            $key = Read-MenuKey
            $lineCount = $Choices.Count + 4
            $top = [Math]::Max(0, [Console]::CursorTop - $lineCount)
            $width = [Math]::Max(1, [Console]::WindowWidth - 1)
            [Console]::SetCursorPosition(0, $top)
            for ($line = 0; $line -lt $lineCount; $line++) { [Console]::WriteLine((' ' * $width)) }
            [Console]::SetCursorPosition(0, $top)
            if ([int]$key.KeyChar -eq 3) { throw [OperationCanceledException]::new('Installation cancelled.') }
            switch ($key.Key) {
                'UpArrow' { $selected = ($selected + $Choices.Count - 1) % $Choices.Count; continue }
                'DownArrow' { $selected = ($selected + 1) % $Choices.Count; continue }
                'Home' { $selected = 0; continue }
                'End' { $selected = $Choices.Count - 1; continue }
                'Enter' { return $selected }
                'Escape' { return -1 }
            }
            for ($i = 0; $i -lt $Choices.Count; $i++) {
                if ("$($key.KeyChar)" -ieq ($Choices[$i] -split ':', 2)[0]) { return $i }
            }
        }
    } finally { Restore-Console }
}
function Read-Confirm([string]$Prompt) {
    if ($script:Options.Yes) { return $true }
    return (Select-Menu $Prompt 1 @('y:Yes', 'n:No')) -eq 0
}
function Show-Plan {
    Write-Line ('  {0,-18} {1}' -f 'Platform', "Windows ($($script:Arch))")
    Write-Line ('  {0,-18} {1}' -f 'Version', $script:Plan.ReleaseLabel)
    $backend = if ($script:Plan.Backend -eq 'cuda') { 'NVIDIA CUDA' } else { 'CPU' }
    if ($script:Options.Backend -eq 'auto') { Write-Line ('  {0,-18} {1} (recommended)' -f 'Acceleration', $backend) Green }
    else { Write-Line ('  {0,-18} {1}' -f 'Acceleration', $backend) }
    $method = if ($script:Plan.Method -eq 'prebuilt') { 'Prebuilt binary' } else { 'Build from source' }
    Write-Line ('  {0,-18} {1}' -f 'Installation', $method)
    Write-Line ('  {0,-18} {1}' -f 'Command location', $script:BinDir)
    Write-Line ('  {0,-18} {1}' -f 'Managed files', $script:PrebuiltDir)
    $video = if ($script:Options.Video) { 'Install FFmpeg if needed' } else { 'Leave FFmpeg unchanged' }
    Write-Line ('  {0,-18} {1}' -f 'Video support', $video)
    if ($script:Plan.Method -eq 'source') {
        Write-Line
        Write-Line '  A source build needs Rust and platform build tools.' Yellow
        if ($script:Plan.Backend -eq 'cuda') { Write-Line '  Windows CUDA also requires the CUDA toolkit.' Yellow }
    }
    Write-Line
}
function Edit-Plan {
    while ($true) {
        $choice = Select-Menu '  Customize your installation' 4 @('r:Release', 'b:Hardware', 'm:Install method', 'v:Video support', 'd:Done')
        switch ($choice) {
            0 {
                $selected = switch ($script:Options.Release) { 'latest' { 0 }; 'master' { 1 }; default { 2 } }
                $release = Select-Menu '  Choose a release' $selected @('s:Latest stable', 'd:Development (master)', 't:Specific tag')
                if ($release -eq 0) { $script:Options.Release = 'latest' }
                elseif ($release -eq 1) { $script:Options.Release = 'master' }
                elseif ($release -eq 2) {
                    $prompt = if ($selected -eq 2) { "  Git tag [$($script:Options.Release)]" } else { '  Git tag' }
                    $tag = Read-Host $prompt
                    if (-not $tag -and $selected -eq 2) { $tag = $script:Options.Release }
                    if ($tag -match '^[A-Za-z0-9][A-Za-z0-9._/-]*$') { $script:Options.Release = $tag }
                    else { Write-Warn 'Enter a valid Git tag.' }
                }
            }
            1 {
                $choices = @('a:Automatic', 'c:CPU')
                if ($script:CudaCC) { $choices += 'n:NVIDIA CUDA' }
                $selected = [array]::IndexOf(@('auto', 'cpu', 'cuda'), $script:Options.Backend)
                $backend = Select-Menu '  Choose hardware' $selected $choices
                if ($backend -ge 0) { $script:Options.Backend = @('auto', 'cpu', 'cuda')[$backend] }
            }
            2 {
                $selected = [int]($script:Options.Method -eq 'source')
                $method = Select-Menu '  Choose an install method' $selected @('a:Automatic (prebuilt when available)', 's:Build from source')
                if ($method -ge 0) { $script:Options.Method = @('auto', 'source')[$method] }
            }
            3 {
                $video = Select-Menu '  Optional video support' ([int][bool]$script:Options.Video) @('s:Skip FFmpeg setup', 'i:Install FFmpeg if needed')
                if ($video -ge 0) { $script:Options.Video = $video -eq 1 }
            }
            default { return }
        }
    }
}
function Show-GuidedInstall {
    while ($true) {
        $script:Plan = Get-Plan
        Show-Plan
        if ($script:Options.Yes -or $script:Options.DryRun) { return $true }
        $choice = Select-Menu '  Ready to install?' 0 @('i:Install', 'c:Customize', 'q:Cancel')
        if ($choice -eq 0) { return $true }
        if ($choice -eq 1) { Edit-Plan; continue }
        return $false
    }
}
function New-TemporaryDirectory([switch]$AlongsideInstall) {
    $parent = if ($AlongsideInstall) { $script:ProfileDir } else { [IO.Path]::GetTempPath() }
    $path = Join-Path $parent ('.mistralrs-stage-' + [guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($path)
    $script:TempPaths.Add($path)
    return $path
}
function Start-Stage([int]$Number, [string]$Label) {
    $script:Stage = "[$Number/$script:StageCount] $Label"
    Write-Line
    Write-Line "  $script:Stage" Cyan
    Write-Log $script:Stage
}
function Format-ByteSize([long]$Bytes) {
    $units = @('B', 'KiB', 'MiB', 'GiB', 'TiB')
    $value = [double]$Bytes
    $i = 0
    while ($value -ge 1024 -and $i -lt $units.Count - 1) { $value /= 1024; $i++ }
    if ($i -eq 0) { return "$Bytes B" }
    return ('{0:N1} {1}' -f $value, $units[$i])
}
function ConvertTo-PlainLog([string]$Text) {
    $escape = [string][char]27
    return [regex]::Replace($Text, $escape + '\[[0-?]*[ -/]*[@-~]', '').Replace([string][char]13, [string][char]10)
}
function Get-ProgressDetail([string]$Fallback) {
    $lines = @((ConvertTo-PlainLog $script:LastCommandText) -split '\n' | Where-Object { $_.Trim() })
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        if ($lines[$i] -match '^\s*Building\s+\[[^\]]*\]\s+([0-9]+/[0-9]+)(.*)$') { return "Building $($Matches[1])$($Matches[2])" }
    }
    if ($lines.Count) { return $lines[-1].Trim() }
    return $Fallback
}
function Show-Activity([string]$Detail, [Diagnostics.Stopwatch]$Clock) {
    if (-not $script:Interactive -or $script:Options.Verbose) { return }
    $frames = @('|', '/', '-', '\')
    $frame = $frames[[int][Math]::Floor($Clock.Elapsed.TotalMilliseconds / $script:PollMilliseconds) % $frames.Count]
    $elapsed = '{0}m {1:00}s' -f [int][Math]::Floor($Clock.Elapsed.TotalMinutes), $Clock.Elapsed.Seconds
    $text = "    [$frame] $elapsed  $Detail"
    $width = [Math]::Max(1, [Console]::WindowWidth - 1)
    if ($text.Length -gt $width) {
        $text = if ($width -gt 3) { $text.Substring(0, $width - 3) + '...' } else { $text.Substring(0, $width) }
    }
    $color = [Console]::ForegroundColor
    try {
        if ($script:UseColor) { [Console]::ForegroundColor = [ConsoleColor]::Cyan }
        [Console]::Write(([string][char]13) + $text.PadRight($width))
    } finally { if ($script:UseColor) { [Console]::ForegroundColor = $color } }
    $script:ProgressVisible = $true
}
function Clear-Activity {
    if ($script:ProgressVisible) {
        try { [Console]::Write(([string][char]13) + (' ' * [Math]::Max(1, [Console]::WindowWidth - 1)) + [char]13) } catch {}
        $script:ProgressVisible = $false
    }
}
function ConvertTo-NativeArgument([AllowEmptyString()][string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}
function Stop-ChildProcess {
    $process = $script:ActiveProcess
    if (-not $process) { return }
    try {
        if (-not $process.HasExited) {
            if ($env:OS -eq 'Windows_NT') {
                $info = New-Object Diagnostics.ProcessStartInfo
                $info.FileName = Join-Path $env:SystemRoot 'System32/taskkill.exe'
                $info.Arguments = "/PID $($process.Id) /T /F"
                $info.UseShellExecute = $false
                $info.CreateNoWindow = $true
                $info.RedirectStandardOutput = $true
                $info.RedirectStandardError = $true
                $killer = [Diagnostics.Process]::Start($info)
                [void]$killer.StandardOutput.ReadToEnd()
                [void]$killer.StandardError.ReadToEnd()
                $killer.WaitForExit()
                $killer.Dispose()
            } else { $process.Kill() }
            $process.WaitForExit()
        }
    } catch { Write-Warn "Could not stop installer child process: $($_.Exception.Message)" }
}
function Invoke-LoggedCommand(
    [string]$Label, [string]$FilePath, [string[]]$Arguments = @(), [hashtable]$Environment = @{}
) {
    $command = Get-Command $FilePath -CommandType Application -ErrorAction Stop | Select-Object -First 1
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $command.Source
    $info.Arguments = (@($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = $script:Utf8
    $info.StandardErrorEncoding = $script:Utf8
    foreach ($key in $Environment.Keys) { $info.EnvironmentVariables[$key] = [string]$Environment[$key] }
    $part = $script:LogPath + '.part'
    $writer = New-Object IO.StreamWriter($part, $false, $script:Utf8)
    $writer.AutoFlush = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    $script:ActiveProcess = $process
    $script:LastCommandText = ''
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $completed = $false
    $status = -1
    Write-Log "> $FilePath $($info.Arguments)"
    if (-not $script:Interactive -or $script:Options.Verbose) { Write-Line "  $Label" }
    try {
        [void]$process.Start()
        $readers = @($process.StandardOutput, $process.StandardError)
        $buffers = @((New-Object char[] $script:OutputBufferSize), (New-Object char[] $script:OutputBufferSize))
        $pending = @($readers[0].ReadAsync($buffers[0], 0, $buffers[0].Length), $readers[1].ReadAsync($buffers[1], 0, $buffers[1].Length))
        $ended = @($false, $false)
        while (-not ($ended[0] -and $ended[1] -and $process.HasExited)) {
            for ($i = 0; $i -lt 2; $i++) {
                if (-not $ended[$i] -and $pending[$i].IsCompleted) {
                    $count = $pending[$i].GetAwaiter().GetResult()
                    if ($count -eq 0) { $ended[$i] = $true; continue }
                    $chunk = New-Object string($buffers[$i], 0, $count)
                    $writer.Write($chunk)
                    if ($script:Options.Verbose) { Write-Host -NoNewline $chunk }
                    $script:LastCommandText += $chunk
                    if ($script:LastCommandText.Length -gt $script:OutputBufferSize) {
                        $script:LastCommandText = $script:LastCommandText.Substring($script:LastCommandText.Length - $script:OutputBufferSize)
                    }
                    $pending[$i] = $readers[$i].ReadAsync($buffers[$i], 0, $buffers[$i].Length)
                }
            }
            Show-Activity (Get-ProgressDetail $Label) $clock
            Start-Sleep -Milliseconds $script:PollMilliseconds
        }
        $process.WaitForExit()
        $status = $process.ExitCode
        $completed = $true
    } finally {
        if (-not $completed) { Stop-ChildProcess }
        $writer.Dispose()
        Clear-Activity
        if ($script:Options.Verbose) { Write-Line }
        if (Test-Path -LiteralPath $part) {
            $target = [IO.File]::Open($script:LogPath, [IO.FileMode]::Append, [IO.FileAccess]::Write)
            $source = [IO.File]::OpenRead($part)
            try { $source.CopyTo($target) } finally { $source.Dispose(); $target.Dispose() }
            if (-not $completed) { Remove-Item -LiteralPath $part -Force }
        }
        $script:ActiveProcess = $null
        $process.Dispose()
    }
    if ($status -ne 0) {
        Write-Line "  [x] $Label" Red
        if (-not $script:Options.Verbose) {
            foreach ($line in ((ConvertTo-PlainLog ([IO.File]::ReadAllText($part, $script:Utf8))) -split '\n')) {
                if ($line -notmatch '^\s*Building\s+\[[^\]]*\]\s+[0-9]+/[0-9]+') { Write-Line $line }
            }
        }
        Remove-Item -LiteralPath $part -Force
        throw "$Label failed (exit code $status)."
    }
    Remove-Item -LiteralPath $part -Force
    Write-Line ("  [ok] {0}  {1:mm\:ss}" -f $Label, $clock.Elapsed) Green
}

function New-HttpClient {
    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $true
    $client = New-Object Net.Http.HttpClient($handler)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd('mistralrs-installer')
    return $client
}
function Download-File([string]$Url, [string]$Destination, [string]$Label) {
    $client = New-HttpClient
    $response = $null
    $source = $null
    $target = $null
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $detail = $Label
    Write-Log "Download: $Url"
    if (-not $script:Interactive -or $script:Options.Verbose) { Write-Line "  $Label" }
    try {
        $request = $client.GetAsync($Url, [Net.Http.HttpCompletionOption]::ResponseHeadersRead)
        while (-not $request.IsCompleted) {
            Show-Activity $detail $clock
            Start-Sleep -Milliseconds $script:PollMilliseconds
        }
        $response = $request.GetAwaiter().GetResult()
        [void]$response.EnsureSuccessStatusCode()
        $total = $response.Content.Headers.ContentLength
        $source = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $target = [IO.File]::Create($Destination)
        $buffer = New-Object byte[] $script:DownloadBufferSize
        [long]$received = 0
        while ($true) {
            $read = $source.ReadAsync($buffer, 0, $buffer.Length)
            while (-not $read.IsCompleted) {
                Show-Activity $detail $clock
                Start-Sleep -Milliseconds $script:PollMilliseconds
            }
            $count = $read.GetAwaiter().GetResult()
            if ($count -eq 0) { break }
            $target.Write($buffer, 0, $count)
            $received += $count
            $detail = if ($total -gt 0) {
                '{0}  {1}/{2} ({3:0}%)' -f $Label, (Format-ByteSize $received), (Format-ByteSize $total), ($received * 100.0 / $total)
            } else { "$Label  $(Format-ByteSize $received)" }
            Show-Activity $detail $clock
        }
        if ($total -gt 0 -and $received -ne $total) { throw "Incomplete download: expected $total bytes, received $received." }
    } catch {
        Write-Log $_.Exception.ToString()
        throw
    } finally {
        Clear-Activity
        if ($target) { $target.Dispose() }
        if ($source) { $source.Dispose() }
        if ($response) { $response.Dispose() }
        $client.Dispose()
    }
    Write-Line ("  [ok] {0}  {1}  {2:mm\:ss}" -f $Label, (Format-ByteSize $received), $clock.Elapsed) Green
}
function Resolve-Release {
    $script:SourceTag = $null
    if ($script:Options.Release -eq 'master') { return }
    $tag = $script:Options.Release
    if ($tag -eq 'latest') {
        $client = New-HttpClient
        $response = $null
        $clock = [Diagnostics.Stopwatch]::StartNew()
        try {
            $request = $client.GetAsync("$script:Repository/releases/latest", [Net.Http.HttpCompletionOption]::ResponseHeadersRead)
            while (-not $request.IsCompleted) {
                Show-Activity 'Find the latest stable release' $clock
                Start-Sleep -Milliseconds $script:PollMilliseconds
            }
            $response = $request.GetAwaiter().GetResult()
            [void]$response.EnsureSuccessStatusCode()
            $uri = $response.RequestMessage.RequestUri
            if ($uri.AbsolutePath -notmatch '/releases/tag/(.+)$') { throw 'Could not determine the latest stable release tag.' }
            $tag = [Uri]::UnescapeDataString($Matches[1])
        } finally {
            Clear-Activity
            if ($response) { $response.Dispose() }
            $client.Dispose()
        }
    }
    $script:SourceTag = $tag
    $script:ReleaseBase = "$script:Repository/releases/download/$([Uri]::EscapeDataString($tag))"
    Write-Line "  Release: $tag" DarkGray
}
function Get-RustVersion {
    if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) { return $null }
    try {
        $output = & rustc --version 2>$null
        if ($LASTEXITCODE -eq 0 -and "$output" -match 'rustc ([0-9]+\.[0-9]+\.[0-9]+)') { return [version]$Matches[1] }
    } catch {}
    return $null
}
function Get-CudaVersionCode {
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) { return $null }
    try {
        $output = (& nvcc --version 2>$null) -join ' '
        if ($LASTEXITCODE -eq 0 -and $output -match 'release ([0-9]+)\.([0-9]+)') {
            return ([int]$Matches[1] * 100 + [int]$Matches[2])
        }
    } catch {}
    return $null
}
function Format-CudaVersionCode([int]$Code) { return "$([int][Math]::Floor($Code / 100)).$($Code % 100)" }
function Test-CudaSourceBuildVersions {
    $toolkit = Get-CudaVersionCode
    if (-not $toolkit) { throw 'CUDA source builds require nvcc. Install the CUDA toolkit, then rerun this installer.' }
    try { $driverOutput = (& nvidia-smi 2>$null) -join ' ' } catch { return }
    if ($driverOutput -match 'CUDA Version:\s*([0-9]+)\.([0-9]+)') {
        $driver = [int]$Matches[1] * 100 + [int]$Matches[2]
        if ($toolkit -gt $driver) {
            $message = "The CUDA toolkit ($(Format-CudaVersionCode $toolkit)) is newer than the driver supports ($(Format-CudaVersionCode $driver)). Update the driver or install a matching toolkit."
            if ($script:Options.AllowCudaMismatch) { Write-Warn $message }
            else { throw "$message Use -AllowCudaMismatch to override." }
        }
    }
}
function Test-MsvcTools {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return $true }
    $programFiles = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
    if (-not $programFiles) { return $false }
    $vswhere = Join-Path $programFiles 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { return $false }
    try {
        $component = if ($script:Arch -eq 'aarch64') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
        $installation = & $vswhere -latest -products '*' -requires $component -property installationPath 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $installation) { return $false }
        $script:MsvcInstallPath = [string](@($installation)[0])
        return $true
    } catch { return $false }
}
function Initialize-MsvcEnvironment {
    if (-not $script:MsvcInstallPath) { return }
    $setup = Join-Path $script:MsvcInstallPath 'Common7/Tools/VsDevCmd.bat'
    if (-not (Test-Path -LiteralPath $setup)) { throw "Visual Studio environment setup was not found: $setup" }
    $architecture = if ($script:Arch -eq 'aarch64') { 'arm64' } else { 'x64' }
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = Join-Path $env:SystemRoot 'System32/cmd.exe'
    $info.Arguments = '/d /u /s /c ""' + $setup + '" -no_logo -arch=' + $architecture + ' -host_arch=' + $architecture + ' 1>&2 && set"'
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = [Text.Encoding]::Unicode
    $info.StandardErrorEncoding = [Text.Encoding]::Unicode
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    $script:ActiveProcess = $process
    $completed = $false
    $clock = [Diagnostics.Stopwatch]::StartNew()
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        while (-not ($process.HasExited -and $stdout.IsCompleted -and $stderr.IsCompleted)) {
            Show-Activity 'Prepare C++ build tools' $clock
            Start-Sleep -Milliseconds $script:PollMilliseconds
        }
        $process.WaitForExit()
        $output = $stdout.GetAwaiter().GetResult()
        $diagnostics = $stderr.GetAwaiter().GetResult()
        $completed = $true
        if ($process.ExitCode -ne 0) {
            Write-Line $diagnostics
            Write-Log $diagnostics
            throw 'Visual Studio environment setup failed. Repair the C++ build tools or run the installer from Developer PowerShell.'
        }
        foreach ($line in ($output -split '\r?\n')) {
            if ($line -match '^([^=]+)=(.*)$') { $script:BuildEnvironment[$Matches[1]] = $Matches[2] }
        }
        if (-not $script:BuildEnvironment.VCToolsInstallDir) { throw 'Visual Studio did not initialize the C++ build tools.' }
    } finally {
        if (-not $completed) { Stop-ChildProcess }
        Clear-Activity
        $script:ActiveProcess = $null
        $process.Dispose()
    }
}
function Prepare-SourceBuild {
    if ($script:Plan.Backend -eq 'cuda') { Test-CudaSourceBuildVersions }
    if (-not (Test-MsvcTools)) {
        Write-Line '  Source builds need Visual Studio Build Tools with "Desktop development with C++".'
        Write-Line '  https://visualstudio.microsoft.com/visual-cpp-build-tools/' Cyan
        if (-not $script:Options.Yes -and (Read-Confirm '  Open the Build Tools download page?')) {
            Start-Process 'https://visualstudio.microsoft.com/visual-cpp-build-tools/'
        }
        throw 'Install the C++ build tools, then rerun this installer.'
    }
    $rustVersion = Get-RustVersion
    if (-not $rustVersion) {
        if (-not (Read-Confirm '  Rust is needed to build mistral.rs. Install it now?')) {
            throw 'Rust installation cancelled.'
        }
        $target = if ($script:Arch -eq 'aarch64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
        $rustup = Join-Path (New-TemporaryDirectory) 'rustup-init.exe'
        Download-File "https://static.rust-lang.org/rustup/dist/$target/rustup-init.exe" $rustup 'Download Rust installer'
        Invoke-LoggedCommand 'Install Rust' $rustup @('-y', '--default-host', $target)
        $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $script:ProfileDir '.cargo' }
        $env:PATH = (Join-Path $cargoHome 'bin') + [IO.Path]::PathSeparator + $env:PATH
        $rustVersion = Get-RustVersion
    }
    if (-not $rustVersion) { throw 'Rust was not found after installation.' }
    if ($rustVersion -lt $script:RequiredRustVersion) {
        if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
            throw "Rust $script:RequiredRustVersion or newer is needed. Update Rust and rerun this installer."
        }
        if (-not (Read-Confirm "  Update Rust $rustVersion to the latest stable version?")) { throw 'Rust update cancelled.' }
        Invoke-LoggedCommand 'Update Rust' 'rustup' @('update', 'stable')
        $rustVersion = Get-RustVersion
        if (-not $rustVersion -or $rustVersion -lt $script:RequiredRustVersion) {
            throw "The active Rust toolchain is still older than $script:RequiredRustVersion. Select an updated toolchain and rerun."
        }
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'Source builds need Git. Install Git for Windows from https://git-scm.com/download/win, then rerun.'
    }
    Initialize-MsvcEnvironment
}
function Test-CuDNN {
    $paths = @(
        "$env:CUDA_PATH/bin/cudnn*.dll"
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin/cudnn*.dll'
        'C:/Program Files/NVIDIA/CUDNN/*/bin/cudnn*.dll'
        'C:/Program Files/NVIDIA/CUDNN/*/bin/*/x64/cudnn*.dll'
    )
    foreach ($path in $paths) { if (Get-Item $path -ErrorAction SilentlyContinue) { return $true } }
    return $false
}
function Test-NCCL {
    foreach ($root in @($env:NCCL_ROOT, $env:NCCL_HOME, $env:CUDA_PATH)) {
        if (-not $root) { continue }
        foreach ($suffix in @('bin/nccl*.dll', 'lib/nccl*.lib', 'lib/x64/nccl*.lib')) {
            if (Get-Item (Join-Path $root $suffix) -ErrorAction SilentlyContinue) { return $true }
        }
    }
    return $false
}
function Test-MKL {
    if ($env:MKLROOT -and (Test-Path -LiteralPath $env:MKLROOT)) { return $true }
    foreach ($path in @(
        'C:/Program Files (x86)/Intel/oneAPI/mkl/latest'
        'C:/Program Files/Intel/oneAPI/mkl/latest'
        (Join-Path $script:ProfileDir 'intel/oneapi/mkl/latest')
    )) {
        if (Test-Path -LiteralPath $path) { return $true }
    }
    return $false
}
function Test-IntelCpu {
    try {
        $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
        return $cpu.Manufacturer -match 'Intel' -or $cpu.Name -match 'Intel'
    } catch { return $false }
}
function Test-CuTileSupported([int]$CudaVersionCode, [int]$CudaComputeCap) {
    return ($CudaVersionCode -ge 1302 -and $CudaComputeCap -ge 80 -and $CudaComputeCap -lt 90) -or
        ($CudaVersionCode -ge 1303 -and $CudaComputeCap -eq 90) -or
        ($CudaVersionCode -ge 1302 -and $CudaComputeCap -ge 100)
}
function Get-Features {
    $features = @()
    if ($script:Plan.Backend -eq 'cuda') {
        $features += 'cuda'
        if (-not $script:Options.NoNccl) {
            if (Test-NCCL) { $features += 'nccl' }
            elseif ($env:MISTRALRS_INSTALL_NCCL -eq '1') {
                $features += 'nccl'
                Write-Warn 'NCCL was explicitly requested but not detected. The build requires its library on the linker path.'
            }
        }
        if (Test-CuDNN) { $features += 'cudnn' }
        if ([int]$script:CudaCC -eq 90) { $features += 'flash-attn-v3' }
        elseif ([int]$script:CudaCC -ge 80) { $features += 'flash-attn' }
        $cudaVersion = Get-CudaVersionCode
        if ($cudaVersion -and (Test-CuTileSupported $cudaVersion $script:CudaCC)) { $features += 'cutile' }
    }
    if ((Test-IntelCpu) -and (Test-MKL)) { $features += 'mkl' }
    return $features -join ' '
}
function Install-MistralRS {
    $root = New-TemporaryDirectory
    $arguments = @('install', '--root', $root, '--force', '--locked', '--git', $script:Repository)
    if ($script:Options.Release -eq 'master') { $arguments += @('--branch', 'master') }
    else {
        if (-not $script:SourceTag) { throw 'A stable release tag must be resolved before building.' }
        $arguments += @('--tag', $script:SourceTag)
    }
    $features = Get-Features
    if ($features) { $arguments += @('--features', $features) }
    $arguments += 'mistralrs-cli'
    $progress = if ($script:Interactive -and -not $script:Options.Verbose) { 'always' } else { 'never' }
    $environment = @{}
    foreach ($key in $script:BuildEnvironment.Keys) { $environment[$key] = $script:BuildEnvironment[$key] }
    $cargoEnvironment = @{
        CARGO_TERM_COLOR = 'never'; CARGO_TERM_PROGRESS_WHEN = $progress
        CARGO_TERM_PROGRESS_WIDTH = '80'; CARGO_TERM_PROGRESS_TERM_INTEGRATION = 'false'
        CARGO_TERM_UNICODE = 'false'; CARGO_TERM_HYPERLINKS = 'false'
    }
    foreach ($key in $cargoEnvironment.Keys) { $environment[$key] = $cargoEnvironment[$key] }
    Invoke-LoggedCommand 'Build mistral.rs' 'cargo' $arguments $environment
    $candidate = New-TemporaryDirectory -AlongsideInstall
    Copy-Item -LiteralPath (Join-Path $root 'bin/mistralrs.exe') -Destination (Join-Path $candidate 'mistralrs.exe')
    return $candidate
}
function Get-DuplicateInstalls {
    $paths = @()
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $script:ProfileDir '.cargo' }
    $paths += Join-Path $cargoHome 'bin/mistralrs.exe'
    $paths += Join-Path $script:BinDir 'mistralrs.exe'
    foreach ($entry in ($env:PATH -split [regex]::Escape([string][IO.Path]::PathSeparator))) {
        if (-not $entry) { continue }
        foreach ($name in @('mistralrs.exe', 'mistralrs.cmd', 'mistralrs.bat')) {
            $paths += Join-Path $entry.Trim('"') $name
        }
    }
    $managed = [IO.Path]::GetFullPath($script:ManagedBin)
    $launcher = [IO.Path]::GetFullPath($script:LauncherPath)
    return @($paths | ForEach-Object { [IO.Path]::GetFullPath($_) } | Where-Object {
        $_ -ine $managed -and $_ -ine $launcher -and (Test-Path -LiteralPath $_ -PathType Leaf)
    } | Sort-Object -Unique)
}
function Confirm-DuplicateInstalls {
    $script:Duplicates = @(Get-DuplicateInstalls)
    if (-not $script:Duplicates.Count) { return }
    Write-Line '  Other mistralrs installations were found:'
    foreach ($path in $script:Duplicates) { Write-Line "    $path" }
    $script:ReplaceDuplicates = Read-Confirm '  Remove these after the new installation is verified?'
}
function Verify-Candidate([string]$Candidate) {
    $binary = Join-Path $Candidate 'mistralrs.exe'
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) { throw "The installation does not contain mistralrs.exe: $Candidate" }
    Invoke-LoggedCommand 'Check mistral.rs' $binary @('--version')
}
function Install-Launcher {
    [void][IO.Directory]::CreateDirectory($script:BinDir)
    $lines = @('@echo off', 'setlocal DisableDelayedExpansion', '"%~dp0..\..\.mistralrs\mistralrs.exe" %*')
    [IO.File]::WriteAllLines($script:LauncherPath, [string[]]$lines, [Text.Encoding]::ASCII)
}
function Restore-Activation {
    $activation = $script:Activation
    if (-not $activation -or $activation.Committed) { return }
    if (Test-Path -LiteralPath $activation.Backup) {
        if (Test-Path -LiteralPath $script:PrebuiltDir) { Remove-Item -LiteralPath $script:PrebuiltDir -Recurse -Force }
        Move-Item -LiteralPath $activation.Backup -Destination $script:PrebuiltDir
    } elseif (-not $activation.HadInstall -and -not (Test-Path -LiteralPath $activation.Candidate)) {
        if (Test-Path -LiteralPath $script:PrebuiltDir) { Remove-Item -LiteralPath $script:PrebuiltDir -Recurse -Force }
    }
    if ($null -ne $activation.Launcher) { [IO.File]::WriteAllBytes($script:LauncherPath, $activation.Launcher) }
    elseif (Test-Path -LiteralPath $script:LauncherPath) { Remove-Item -LiteralPath $script:LauncherPath -Force }
    $script:Activation = $null
}
function Activate-Install([string]$Candidate) {
    $backup = Join-Path $script:ProfileDir ('.mistralrs-backup-' + [guid]::NewGuid().ToString('N'))
    $launcher = $null
    if (Test-Path -LiteralPath $script:LauncherPath -PathType Leaf) { $launcher = [IO.File]::ReadAllBytes($script:LauncherPath) }
    $script:Activation = @{
        Candidate = $Candidate; Backup = $backup; HadInstall = (Test-Path -LiteralPath $script:PrebuiltDir)
        Launcher = $launcher; Committed = $false
    }
    try {
        if ($script:Activation.HadInstall) { Move-Item -LiteralPath $script:PrebuiltDir -Destination $backup }
        Move-Item -LiteralPath $Candidate -Destination $script:PrebuiltDir
        Verify-Candidate $script:PrebuiltDir
        Install-Launcher
        Save-Options
        $script:Activation.Committed = $true
        if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Recurse -Force }
        $script:Activation = $null
    } catch {
        Restore-Activation
        throw
    }
}
function Add-UserPath([string]$PathToAdd) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { $_ })
    if ($entries -notcontains $PathToAdd) {
        [Environment]::SetEnvironmentVariable('Path', ((@($PathToAdd) + $entries) -join ';'), 'User')
    }
}
function Install-FFmpeg {
    if (-not $script:Options.Video -or $env:MISTRALRS_INSTALL_IGNORE_FFMPEG -or (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Warn 'FFmpeg setup needs winget. Install FFmpeg manually to enable video support.'
        return
    }
    try {
        Invoke-LoggedCommand 'Install optional FFmpeg' 'winget' @(
            'install', '--id', 'Gyan.FFmpeg', '--exact', '--source', 'winget'
            '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity'
        )
    } catch {
        Write-Warn "Optional FFmpeg setup failed. mistral.rs is installed. $($_.Exception.Message)"
        $script:KeepLog = $true
    }
}
function Show-Success {
    Write-Line
    Write-Line '  mistral.rs is installed.' Green
    Write-Line "  Command: $script:LauncherPath" DarkGray
    Write-Line
    Write-Line '  Try browser chat:'
    Write-Line
    Write-Line '    mistralrs serve -m Qwen/Qwen3-4B --quant 4' Cyan
    Write-Line
    Write-Line '  Then open http://localhost:1234/ui'
    Write-Line
}
function Clear-Installer {
    Stop-ChildProcess
    Clear-Activity
    Restore-Console
    try { Restore-Activation }
    catch { Write-Warn "Could not restore the previous install. Backup: $($script:Activation.Backup). $($_.Exception.Message)" }
    foreach ($path in $script:TempPaths) {
        if (Test-Path -LiteralPath $path) {
            try { Remove-Item -LiteralPath $path -Recurse -Force } catch { Write-Warn "Could not remove temporary files: $path" }
        }
    }
    if ($script:LogPath -and (Test-Path -LiteralPath ($script:LogPath + '.part'))) {
        Remove-Item -LiteralPath ($script:LogPath + '.part') -Force -ErrorAction SilentlyContinue
    }
}
function Main {
    Assert-Options
    $script:Arch = Get-Architecture
    $script:CudaCC = Get-CudaComputeCap
    Show-Banner
    if (-not (Show-GuidedInstall)) { throw [OperationCanceledException]::new('Installation cancelled.') }
    if ($script:Options.DryRun) { Write-Line '  Dry run complete. No files were changed.'; return }
    if ($env:OS -ne 'Windows_NT') { throw 'This installer requires Windows. Use install.sh on macOS or Linux.' }
    $script:LogPath = Join-Path ([IO.Path]::GetTempPath()) ('mistralrs-install-' + [guid]::NewGuid().ToString('N') + '.log')
    [IO.File]::WriteAllText($script:LogPath, '', $script:Utf8)
    Start-Stage 1 'Prepare installation'
    Confirm-DuplicateInstalls
    Resolve-Release
    if ($script:Plan.Method -eq 'source') { Prepare-SourceBuild }
    if ($script:Plan.Method -eq 'prebuilt') {
        Start-Stage 2 'Download binary'
        $archive = Join-Path (New-TemporaryDirectory) 'mistralrs.zip'
        try { Download-File "$script:ReleaseBase/$($script:Plan.Asset)" $archive 'Download mistral.rs' }
        catch {
            throw "The prebuilt download failed. Retry the installer, or use -FromSource to build it yourself. $($_.Exception.Message)"
        }
        Start-Stage 3 'Verify binary'
        $candidate = New-TemporaryDirectory -AlongsideInstall
        Expand-Archive -LiteralPath $archive -DestinationPath $candidate -Force
    } else {
        Start-Stage 2 'Build from source'
        $candidate = Install-MistralRS
        Start-Stage 3 'Verify binary'
    }
    Verify-Candidate $candidate
    Start-Stage 4 'Finish setup'
    Activate-Install $candidate
    foreach ($path in $script:Duplicates) {
        if ($script:ReplaceDuplicates) {
            try { Remove-Item -LiteralPath $path -Force }
            catch { Write-Warn "Could not remove an older install: $path" }
        }
    }
    Install-FFmpeg
    Add-UserPath $script:BinDir
    $script:Installed = $true
    Show-Success
}

Initialize-Installer $InstallerArguments
$originalPath = $env:PATH
$originalProtocol = [Net.ServicePointManager]::SecurityProtocol
$succeeded = $false
try {
    if ($script:Options.Help) { Show-Help }
    else {
        [Net.ServicePointManager]::SecurityProtocol = $originalProtocol -bor [Net.SecurityProtocolType]::Tls12
        Main
    }
    $succeeded = $true
} catch [OperationCanceledException] {
    Write-Line
    Write-Line '  Installation cancelled.'
} catch {
    Write-Line
    Write-Line "  Error: $($_.Exception.Message)" Red
    Write-Log $_.Exception.ToString()
    throw
} finally {
    Clear-Installer
    [Net.ServicePointManager]::SecurityProtocol = $originalProtocol
    $env:PATH = $originalPath
    if ($script:Installed -and ($env:PATH -split ';') -notcontains $script:BinDir) { $env:PATH = "$script:BinDir;$env:PATH" }
    if ($script:LogPath -and (Test-Path -LiteralPath $script:LogPath)) {
        if ($succeeded -and -not $script:KeepLog) { Remove-Item -LiteralPath $script:LogPath -Force }
        else { Write-Line "  Full log: $script:LogPath" DarkGray }
    }
}
} | Out-Null
