#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$InstallerPath = (Join-Path $PSScriptRoot '../../install.ps1'),
    [string]$PowerShellPath = (Get-Process -Id $PID).Path,
    [string]$Filter = '*'
)

$ErrorActionPreference = 'Stop'
$parseTokens = $null
$parseErrors = $null
$installerAst = [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $InstallerPath).Path, [ref]$parseTokens, [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw ($parseErrors | Out-String)
}
$functionNodes = $installerAst.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true)
$functionDefinitions = ($functionNodes | ForEach-Object { $_.Extent.Text }) -join "`n"
# Key-selection tests replace physical console calls because macOS cannot read CursorVisible.
$consoleFunctionDefinitions = ($functionNodes | ForEach-Object {
    if ($_.Name -in @('Select-Menu', 'Restore-Console')) {
        $_.Extent.Text.Replace('[Console]', '[InstallerTestConsole]')
    } else { $_.Extent.Text }
}) -join "`n"
Add-Type -TypeDefinition @'
public static class InstallerTestConsole {
    public static bool TreatControlCAsInput;
    public static bool CursorVisible = true;
    public static bool IsInputRedirected;
    public static int CursorTop = 100;
    public static int WindowWidth = 80;
    public static void SetCursorPosition(int left, int top) { CursorTop = top; }
    public static void WriteLine(string text) { CursorTop++; }
}
'@

$fixtureFunctions = {
    function Assert-True($Actual, [string]$Message = 'Expected a true value') {
        if (-not $Actual) { throw $Message }
    }

    function Assert-Equal($Actual, $Expected) {
        if ($Actual -cne $Expected) {
            throw "Expected <$Expected>, got <$Actual>"
        }
    }

    function Assert-Contains([string]$Actual, [string]$Expected) {
        if (-not $Actual.Contains($Expected)) {
            throw "Expected output to contain <$Expected>, got:`n$Actual"
        }
    }

    function Assert-NotContains([string]$Actual, [string]$Unexpected) {
        if ($Actual.Contains($Unexpected)) {
            throw "Unexpected <$Unexpected> in:`n$Actual"
        }
    }

    function Assert-Throws([scriptblock]$Action, [string]$Pattern = '.') {
        $caught = $null
        try { & $Action | Out-Null } catch { $caught = $_ }
        Assert-True ($null -ne $caught) 'Expected an exception'
        Assert-True ($caught.ToString() -match $Pattern) "Unexpected exception: $caught"
        $script:LastTestException = $caught
    }

    function Capture-Output([scriptblock]$Action) {
        (& $Action *>&1 | ForEach-Object { $_.ToString() }) -join "`n"
    }

    function Write-FixtureFile([string]$Path, [string]$Content) {
        $directory = Split-Path -Parent $Path
        if (-not (Test-Path -LiteralPath $directory)) {
            New-Item -ItemType Directory -Path $directory -Force | Out-Null
        }
        [IO.File]::WriteAllText($Path, $Content, (New-Object Text.UTF8Encoding $false))
    }

    function Assert-NoInstall {
        Assert-True (-not (Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.mistralrs'))) 'Managed installation was created'
        Assert-True (-not (Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.local/bin'))) 'Launcher directory was created'
    }

    function Invoke-InstallerProcess([string[]]$Arguments) {
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $output = (& $PowerShellPath -NoProfile -NonInteractive -File $InstallerPath @Arguments 2>&1 | Out-String)
            return [pscustomobject]@{ Status = $LASTEXITCODE; Output = $output }
        } finally {
            $ErrorActionPreference = $previousPreference
        }
    }

    function Set-FakeRelease([string]$Tag = 'v0.9.0') {
        Add-Type -AssemblyName System.Net.Http
        $response = [pscustomobject]@{
            RequestMessage = [pscustomobject]@{ RequestUri = [uri]('https://github.com/EricLBuehler/mistral.rs/releases/tag/' + $Tag) }
        }
        $response | Add-Member ScriptMethod EnsureSuccessStatusCode { return $this }
        $response | Add-Member ScriptMethod Dispose {}
        $script:ReleaseTask = [pscustomobject]@{ IsCompleted = $true; Response = $response }
        $script:ReleaseTask | Add-Member ScriptMethod GetAwaiter { return $this }
        $script:ReleaseTask | Add-Member ScriptMethod GetResult { return $this.Response }
        $script:ReleaseClient = [pscustomobject]@{}
        $script:ReleaseClient | Add-Member ScriptMethod GetAsync {
            param($Url, $Completion)
            $script:RequestedReleaseUrl = $Url
            return $script:ReleaseTask
        }
        $script:ReleaseClient | Add-Member ScriptMethod Dispose {}
        function script:New-HttpClient { return $script:ReleaseClient }
    }

    function Enable-FakeNativeCommands {
        $script:NativeCalls = New-Object 'System.Collections.Generic.List[object]'
        function script:Invoke-LoggedCommand($Label, $FilePath, [string[]]$Arguments, $Environment) {
            $script:NativeCalls.Add([pscustomobject]@{ File = $FilePath; Arguments = $Arguments; Environment = $Environment })
            if ($FilePath -eq 'cargo') {
                $rootIndex = [array]::IndexOf($Arguments, '--root') + 1
                Assert-True ($rootIndex -gt 0) 'Cargo root is missing'
                Write-FixtureFile (Join-Path $Arguments[$rootIndex] 'bin/mistralrs.exe') 'new binary'
            } elseif ($Arguments.Count -eq 1 -and $Arguments[0] -eq '--version') {
                Assert-Equal ([IO.File]::ReadAllText($FilePath)) 'new binary'
            } else { throw "Unexpected native command: $FilePath $Arguments" }
        }
    }

    function Enable-FakeSourcePrerequisites {
        function script:Test-MsvcTools { return $true }
        function script:Get-RustVersion { return [version]'1.94.0' }
        function script:git { throw 'Git should only be detected by this fixture' }
    }

    function Enable-FakeDownload {
        function script:Download-File($Url, $Destination, $Label) {
            $script:DownloadedUrl = $Url
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $archive = [IO.Compression.ZipFile]::Open($Destination, [IO.Compression.ZipArchiveMode]::Create)
            try {
                $entry = $archive.CreateEntry('mistralrs.exe')
                $writer = New-Object IO.StreamWriter($entry.Open())
                try { $writer.Write('new binary') } finally { $writer.Dispose() }
            } finally { $archive.Dispose() }
        }
    }

    function Assert-NoStages {
        $leftovers = @(Get-ChildItem -LiteralPath $env:USERPROFILE -Force | Where-Object {
            $_.Name -like '.mistralrs-stage-*' -or $_.Name -like '.mistralrs-backup-*'
        })
        Assert-Equal $leftovers.Count 0
    }

    function Assert-StageOrder([string]$Output, [string]$SecondStage) {
        $previous = -1
        foreach ($stage in @('[1/4] Prepare installation', ('[2/4] ' + $SecondStage), '[3/4] Verify binary', '[4/4] Finish setup')) {
            $position = $Output.IndexOf($stage)
            Assert-True ($position -gt $previous) "Stage missing or out of order: $stage"
            $previous = $position
        }
    }

    function Set-MenuKeys([ConsoleKeyInfo[]]$Keys) {
        $script:Interactive = $true
        [InstallerTestConsole]::TreatControlCAsInput = $false
        [InstallerTestConsole]::CursorVisible = $true
        $script:MenuKeys = New-Object 'System.Collections.Generic.Queue[System.ConsoleKeyInfo]'
        foreach ($key in $Keys) { $script:MenuKeys.Enqueue($key) }
        function script:Read-MenuKey {
            Assert-True ($script:MenuKeys.Count -gt 0) 'Menu requested an unexpected extra key'
            return $script:MenuKeys.Dequeue()
        }
    }

    function New-MenuKey([ConsoleKey]$Key, [char]$Character = [char]0) {
        return New-Object ConsoleKeyInfo($Character, $Key, $false, $false, $false)
    }

    function Assert-ConsoleRestored {
        Assert-Equal ([InstallerTestConsole]::TreatControlCAsInput) $false
        Assert-Equal ([InstallerTestConsole]::CursorVisible) $true
        Assert-True ($null -eq $script:ConsoleState)
        Assert-Equal $script:MenuKeys.Count 0
    }

    function Add-UserPath { throw 'Test attempted to change the user PATH' }
    function Download-File { throw 'Test attempted a real download' }
    function New-HttpClient { throw 'Test attempted a real HTTP request' }
    function Invoke-WebRequest { throw 'Test attempted a real network request' }
    function Invoke-RestMethod { throw 'Test attempted a real network request' }
    function Get-CudaComputeCap { return $null }
    function Get-CudaDriverVersionCode { return 0 }
    function Get-CudaVersionCode { return 0 }
    function Test-IntelCpu { return $false }
    function Test-MKL { return $false }
    function Test-CuDNN { return $false }
    function Test-NCCL { return $false }
}.ToString()

$childScript = @'
param([string]$Mode, [string]$Value, [string]$SecondValue)
[Console]::OutputEncoding = New-Object Text.UTF8Encoding $false
switch ($Mode) {
    'arguments' {
        [Console]::Out.WriteLine(('first=' + $Value))
        [Console]::Out.WriteLine(('second=' + $SecondValue))
        [Console]::Out.WriteLine(('env=' + $env:MISTRALRS_TEST_CHILD_VALUE))
    }
    'failure' {
        [Console]::Out.WriteLine('early diagnostic')
        1..40 | ForEach-Object { [Console]::Error.WriteLine(('context line ' + $_)) }
        [Console]::Error.WriteLine(('unicode diagnostic: caf' + [char]0x00e9 + ' ' + [char]0x65e5))
        [Console]::Error.WriteLine('last diagnostic')
        exit 23
    }
    'success' {
        [Console]::Out.WriteLine('child stdout')
        [Console]::Error.WriteLine('child stderr')
    }
    'handshake' {
        [Console]::Out.WriteLine('stream-ready')
        [Console]::Out.Flush()
        $deadline = [DateTime]::UtcNow.AddSeconds(5)
        while (-not (Test-Path -LiteralPath $Value)) {
            if ([DateTime]::UtcNow -gt $deadline) { exit 29 }
            Start-Sleep -Milliseconds 25
        }
        [Console]::Out.WriteLine('stream-finished')
    }
    'flood' {
        Add-Type -TypeDefinition 'public static class TestTimeout { public static System.Threading.Timer Timer; public static void Start() { Timer = new System.Threading.Timer(delegate(object state) { System.Environment.Exit(31); }, null, 10000, System.Threading.Timeout.Infinite); } }'
        [TestTimeout]::Start()
        $stdoutLine = 'out-' + ('o' * 1024)
        $stderrLine = 'err-' + ('e' * 1024)
        1..80 | ForEach-Object {
            [Console]::Out.WriteLine($stdoutLine)
            [Console]::Error.WriteLine($stderrLine)
        }
        [Console]::Out.WriteLine('OUT_END')
        [Console]::Error.WriteLine('ERR_END')
        [TestTimeout]::Timer.Dispose()
    }
    'wait' {
        [IO.File]::WriteAllText($Value, [string]$PID)
        [Console]::Out.WriteLine('child-running')
        [Console]::Out.Flush()
        Start-Sleep -Seconds 10
        [IO.File]::WriteAllText($SecondValue, 'child-survived')
    }
    default { throw ('Unknown child mode: ' + $Mode) }
}
'@

$script:Cases = @()
function Test-Case([string]$Name, [scriptblock]$Body, [switch]$ConsoleFacade) {
    $script:Cases += [pscustomobject]@{ Name = $Name; Body = $Body.ToString(); ConsoleFacade = $ConsoleFacade }
}

Test-Case 'help exits without installation' {
    $result = Invoke-InstallerProcess @('-Help')
    Assert-Equal $result.Status 0
    Assert-Contains $result.Output '-DryRun'
    Assert-Contains $result.Output '-Release'
    Assert-NoInstall
}

Test-Case 'unknown command-line arguments fail without installation' {
    $result = Invoke-InstallerProcess @('-UnknownInstallerOption')
    Assert-True ($result.Status -ne 0) 'Unknown option was accepted'
    Assert-Contains $result.Output 'UnknownInstallerOption'
    Assert-NoInstall
}

Test-Case 'invalid releases fail before filesystem changes' {
    Initialize-Installer @{ Release = '../unsafe' }
    Assert-Throws { Assert-Options } 'release|version|tag'
    Assert-NoInstall
}

Test-Case 'invalid backends fail before filesystem changes' {
    Initialize-Installer @{ Backend = 'unknown' }
    Assert-Throws { Assert-Options } 'backend|acceleration'
    Assert-NoInstall
}

Test-Case 'command-line dry run makes no installation changes' {
    $result = Invoke-InstallerProcess @('-DryRun', '-Yes', '-Backend', 'cpu', '-Release', 'v0.9.0')
    Assert-Equal $result.Status 0
    Assert-Contains $result.Output 'Dry run complete'
    Assert-Contains $result.Output 'v0.9.0'
    Assert-NoInstall
}

Test-Case 'CPU plan uses a prebuilt binary despite an available NVIDIA GPU' {
    Initialize-Installer @{ Backend = 'cpu' }
    $script:Arch = 'x86_64'
    $script:CudaCC = 89
    $plan = Get-Plan
    Assert-Equal $plan.Backend 'cpu'
    Assert-Equal $plan.Method 'prebuilt'
    Assert-Contains $plan.Asset 'cpu'
    Assert-NoInstall
}

Test-Case 'menu arrows and Enter select the highlighted item' {
    Initialize-Installer @{}
    Set-MenuKeys @((New-MenuKey DownArrow), (New-MenuKey Enter))
    Capture-Output { $script:Selection = Select-Menu 'Test menu' 0 @('i:Install', 'c:Customize', 'q:Cancel') } | Out-Null
    Assert-Equal $script:Selection 1
    Assert-ConsoleRestored
} -ConsoleFacade

Test-Case 'menu shortcuts act immediately and ignore unknown Unicode keys' {
    Initialize-Installer @{}
    Set-MenuKeys @((New-MenuKey A ([char]0x00e9)), (New-MenuKey C 'c'))
    Capture-Output { $script:Selection = Select-Menu 'Test menu' 0 @('i:Install', 'c:Customize', 'q:Cancel') } | Out-Null
    Assert-Equal $script:Selection 1
    Assert-ConsoleRestored
} -ConsoleFacade

Test-Case 'menu Escape returns without selecting and restores console state' {
    Initialize-Installer @{}
    Set-MenuKeys @((New-MenuKey Escape))
    Capture-Output { $script:Selection = Select-Menu 'Test menu' 0 @('i:Install', 'q:Cancel') } | Out-Null
    Assert-Equal $script:Selection -1
    Assert-ConsoleRestored
} -ConsoleFacade

Test-Case 'menu Ctrl-C cancels and restores console state' {
    Initialize-Installer @{}
    Set-MenuKeys @((New-MenuKey C ([char]3)))
    Capture-Output { Assert-Throws { Select-Menu 'Test menu' 0 @('i:Install', 'q:Cancel') } 'cancelled' } | Out-Null
    Assert-ConsoleRestored
} -ConsoleFacade

Test-Case 'latest source installs pin the resolved stable release' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; FromSource = $true; Yes = $true }
    Set-FakeRelease 'v0.9.0'
    Enable-FakeNativeCommands
    Enable-FakeSourcePrerequisites
    function Test-CudaSourceBuildVersions { throw 'CPU install checked CUDA prerequisites' }
    function Add-UserPath($PathToAdd) { $script:AddedPath = $PathToAdd }
    $output = Capture-Output { try { Main } finally { Clear-Installer } }
    $cargo = @($script:NativeCalls | Where-Object { $_.File -eq 'cargo' })
    Assert-Equal $cargo.Count 1
    $tagIndex = [array]::IndexOf($cargo[0].Arguments, '--tag')
    Assert-True ($tagIndex -ge 0) 'Cargo was not pinned to a release'
    Assert-Equal $cargo[0].Arguments[$tagIndex + 1] 'v0.9.0'
    Assert-True (-not ($cargo[0].Arguments -contains '--branch')) 'Stable build used a moving branch'
    Assert-True (-not ($cargo[0].Arguments -contains '--features')) 'CPU build selected CUDA features'
    Assert-Contains $script:RequestedReleaseUrl '/releases/latest'
    Assert-StageOrder $output 'Build from source'
    Assert-Equal ([IO.File]::ReadAllText($script:ManagedBin)) 'new binary'
    Assert-Contains ([IO.File]::ReadAllText($script:ConfigPath)) 'method=source'
    Assert-Equal $script:AddedPath $script:BinDir
    Assert-NoStages
}

Test-Case 'development source builds select master without release lookup' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; Release = 'master'; Yes = $true }
    Enable-FakeNativeCommands
    Enable-FakeSourcePrerequisites
    function Add-UserPath {}
    Capture-Output { try { Main } finally { Clear-Installer } } | Out-Null
    $cargo = @($script:NativeCalls | Where-Object { $_.File -eq 'cargo' })[0]
    $branchIndex = [array]::IndexOf($cargo.Arguments, '--branch')
    Assert-True ($branchIndex -ge 0) 'Development build omitted its branch'
    Assert-Equal $cargo.Arguments[$branchIndex + 1] 'master'
    Assert-True (-not ($cargo.Arguments -contains '--tag')) 'Development build selected a stable tag'
    Assert-NoStages
}

Test-Case 'prebuilt install activates the binary and persists selected settings' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; Release = 'v0.9.0'; Yes = $true; NoNccl = $true }
    Enable-FakeNativeCommands
    Enable-FakeDownload
    function Add-UserPath {}
    $output = Capture-Output { try { Main } finally { Clear-Installer } }
    Assert-StageOrder $output 'Download binary'
    Assert-Contains $script:DownloadedUrl '/releases/download/v0.9.0/mistralrs-cpu-x86_64-pc-windows-msvc.zip'
    Assert-Equal ([IO.File]::ReadAllText($script:ManagedBin)) 'new binary'
    Assert-Contains $output 'mistralrs serve -m Qwen/Qwen3-4B --quant 4'
    Assert-Contains $output 'http://localhost:1234/ui'
    foreach ($unwanted in @('Start now', 'Finish ]', 'downloads on first run')) { Assert-NotContains $output $unwanted }
    $launcher = [IO.File]::ReadAllText($script:LauncherPath)
    Assert-Contains $launcher 'DisableDelayedExpansion'
    Assert-Contains $launcher '"%~dp0..\..\.mistralrs\mistralrs.exe" %*'
    Initialize-Installer @{}
    Assert-Equal $script:Options.Release 'v0.9.0'
    Assert-Equal $script:Options.Backend 'cpu'
    Assert-True $script:Options.NoNccl
    Assert-NoStages
}

Test-Case 'download failure preserves the old installation without source fallback' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; Release = 'v0.9.0'; Yes = $true }
    Write-FixtureFile $script:ManagedBin 'old binary'
    function Download-File { throw 'Synthetic download failure' }
    function Prepare-SourceBuild { throw 'Unexpected source fallback' }
    function Install-MistralRS { throw 'Unexpected source fallback' }
    $output = Capture-Output { Assert-Throws { try { Main } finally { Clear-Installer } } 'Synthetic download failure' }
    Assert-Equal ([IO.File]::ReadAllText($script:ManagedBin)) 'old binary'
    Assert-True (Test-Path -LiteralPath $script:LogPath) 'Diagnostic log was removed'
    Assert-NotContains $output 'mistral.rs is installed.'
    Assert-NoStages
}

Test-Case 'source build failure keeps the old installation and diagnostic log' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; Release = 'v0.9.0'; FromSource = $true; Yes = $true }
    Write-FixtureFile $script:ManagedBin 'old binary'
    Enable-FakeSourcePrerequisites
    function Invoke-LoggedCommand {
        Write-Log 'Synthetic compiler diagnostic'
        throw 'Synthetic compiler failure (exit code 23)'
    }
    $output = Capture-Output { Assert-Throws { try { Main } finally { Clear-Installer } } 'Synthetic compiler failure' }
    Assert-Equal ([IO.File]::ReadAllText($script:ManagedBin)) 'old binary'
    Assert-Contains ([IO.File]::ReadAllText($script:LogPath)) 'Synthetic compiler diagnostic'
    Assert-NotContains $output '[4/4]'
    Assert-NotContains $output 'mistral.rs is installed.'
    Assert-NoStages
}

Test-Case 'a failed candidate is rejected before replacing the old installation' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; Release = 'v0.9.0'; Yes = $true }
    Write-FixtureFile $script:ManagedBin 'old binary'
    Enable-FakeDownload
    function Invoke-LoggedCommand { throw 'Candidate failed to execute' }
    Capture-Output { Assert-Throws { try { Main } finally { Clear-Installer } } 'Candidate failed to execute' } | Out-Null
    Assert-Equal ([IO.File]::ReadAllText($script:ManagedBin)) 'old binary'
    Assert-True (-not (Test-Path -LiteralPath $script:LauncherPath))
    Assert-NoStages
}

Test-Case 'activation verification failure restores the old binary and launcher' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; Release = 'v0.9.0'; Yes = $true }
    Write-FixtureFile $script:ManagedBin 'old binary'
    Write-FixtureFile $script:LauncherPath 'old launcher'
    Enable-FakeDownload
    function Invoke-LoggedCommand($Label, $FilePath, $Arguments, $Environment) {
        if ($FilePath -eq $script:ManagedBin) { throw 'Activated binary failed' }
    }
    Capture-Output { Assert-Throws { try { Main } finally { Clear-Installer } } 'Activated binary failed' } | Out-Null
    Assert-Equal ([IO.File]::ReadAllText($script:ManagedBin)) 'old binary'
    Assert-Equal ([IO.File]::ReadAllText($script:LauncherPath)) 'old launcher'
    Assert-NoStages
}

Test-Case 'cancellation after backup rename restores the old installation' {
    $env:OS = 'Windows_NT'
    Initialize-Installer @{ Backend = 'cpu'; Release = 'v0.9.0'; Yes = $true }
    Write-FixtureFile $script:ManagedBin 'old binary'
    Write-FixtureFile $script:LauncherPath 'old launcher'
    Enable-FakeDownload
    Enable-FakeNativeCommands
    function Move-Item([string]$LiteralPath, [string]$Destination) {
        Microsoft.PowerShell.Management\Move-Item -LiteralPath $LiteralPath -Destination $Destination
        if ($LiteralPath -eq $script:PrebuiltDir) { throw [OperationCanceledException]::new('Synthetic cancellation') }
    }
    $output = Capture-Output { Assert-Throws { try { Main } finally { Clear-Installer } } 'Synthetic cancellation' }
    Assert-Equal ([IO.File]::ReadAllText($script:ManagedBin)) 'old binary'
    Assert-Equal ([IO.File]::ReadAllText($script:LauncherPath)) 'old launcher'
    Assert-NotContains $output 'mistral.rs is installed.'
    Assert-NoStages
}

Test-Case 'saved settings are loaded' {
    $configPath = Join-Path $env:USERPROFILE '.mistralrs/install.conf'
    Write-FixtureFile $configPath "release=v0.7.0`nbackend=cpu`nmethod=source`nno_nccl=1`nallow_cuda_mismatch=1`n"
    Initialize-Installer @{}
    Assert-Equal $script:Options.Release 'v0.7.0'
    Assert-Equal $script:Options.Backend 'cpu'
    Assert-Equal $script:Options.Method 'source'
}

Test-Case 'explicit options override environment and saved settings' {
    $configPath = Join-Path $env:USERPROFILE '.mistralrs/install.conf'
    Write-FixtureFile $configPath "release=v0.7.0`nbackend=auto`nmethod=source`n"
    $env:MISTRALRS_INSTALL_TAG = 'v0.8.0'
    Initialize-Installer @{ Release = 'v0.9.0'; Backend = 'cpu' }
    Assert-Equal $script:Options.Release 'v0.9.0'
    Assert-Equal $script:Options.Backend 'cpu'
    Assert-Equal $script:Options.Method 'source'
}

Test-Case 'legacy release environment overrides saved release' {
    Write-FixtureFile (Join-Path $env:USERPROFILE '.mistralrs/install.conf') "release=v0.7.0`nbackend=cpu`nmethod=auto`n"
    $env:MISTRALRS_INSTALL_TAG = 'v0.8.0'
    Initialize-Installer @{}
    Assert-Equal $script:Options.Release 'v0.8.0'
}

Test-Case 'legacy updater flags retain source and automatic confirmation' {
    $env:MISTRALRS_INSTALL_FROM_SOURCE = '1'
    $env:MISTRALRS_INSTALL_YES = '1'
    Initialize-Installer @{ Release = 'v0.9.0' }
    Assert-Equal $script:Options.Method 'source'
    Assert-Equal $script:Options.Release 'v0.9.0'
    Assert-True $script:Options.Yes
    Assert-NoInstall
}

Test-Case 'empty legacy flags preserve saved settings and zero overrides them' {
    Write-FixtureFile (Join-Path $env:USERPROFILE '.mistralrs/install.conf') "release=v0.9.0`nbackend=cpu`nmethod=auto`nno_nccl=1`nallow_cuda_mismatch=1`n"
    $env:MISTRALRS_INSTALL_NO_NCCL = ''
    $env:MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH = ''
    Initialize-Installer @{}
    Assert-True $script:Options.NoNccl
    Assert-True $script:Options.AllowCudaMismatch
    $env:MISTRALRS_INSTALL_NO_NCCL = '0'
    $env:MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH = '0'
    Initialize-Installer @{}
    Assert-Equal $script:Options.NoNccl $false
    Assert-Equal $script:Options.AllowCudaMismatch $false
}

Test-Case 'command runner preserves arguments and child-only environment' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    $value = 'space "quoted" trailing\'
    $secondValue = 'semi; dollar$ amp& parens()'
    Invoke-LoggedCommand -Label 'Testing arguments' -FilePath $PowerShellPath -Arguments @(
        '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'arguments', $value, $secondValue
    ) -Environment @{ MISTRALRS_TEST_CHILD_VALUE = 'temporary child value' }
    $log = [IO.File]::ReadAllText($script:LogPath)
    Assert-Contains $log ('first=' + $value)
    Assert-Contains $log ('second=' + $secondValue)
    Assert-Contains $log 'env=temporary child value'
    Assert-True ([string]::IsNullOrEmpty($env:MISTRALRS_TEST_CHILD_VALUE)) 'Child environment leaked into installer'
}

Test-Case 'command failure prints every diagnostic and preserves the full log' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    $output = Capture-Output {
        Assert-Throws {
            Invoke-LoggedCommand -Label 'Testing failure' -FilePath $PowerShellPath -Arguments @(
                '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'failure'
            )
        } '23'
    }
    $log = [IO.File]::ReadAllText($script:LogPath)
    $expectedLines = @('early diagnostic', 'last diagnostic') + @(1..40 | ForEach-Object { 'context line ' + $_ })
    foreach ($line in $expectedLines) {
        Assert-Contains $output $line
        Assert-Contains $log $line
    }
    $unicode = 'unicode diagnostic: caf' + [char]0x00e9 + ' ' + [char]0x65e5
    Assert-Contains $output $unicode
    Assert-Contains $log $unicode
}

Test-Case 'verbose command success streams both output channels' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true; Verbose = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    $output = Capture-Output {
        Invoke-LoggedCommand -Label 'Testing verbose output' -FilePath $PowerShellPath -Arguments @(
            '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'success'
        )
    }
    Assert-Contains $output 'child stdout'
    Assert-Contains $output 'child stderr'
    Assert-Contains ([IO.File]::ReadAllText($script:LogPath)) 'child stderr'
}

Test-Case 'redirected commands do not emit cursor controls or ANSI colors' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    $output = Capture-Output {
        Invoke-LoggedCommand -Label 'Testing plain output' -FilePath $PowerShellPath -Arguments @(
            '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'success'
        )
    }
    Assert-NotContains $output ([string][char]27)
}

Test-Case 'command runner drains stdout and stderr beyond pipe capacity' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    Invoke-LoggedCommand -Label 'Testing simultaneous output' -FilePath $PowerShellPath -Arguments @(
        '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'flood'
    )
    $log = [IO.File]::ReadAllText($script:LogPath)
    Assert-Equal ([regex]::Matches($log, 'out-o').Count) 80
    Assert-Equal ([regex]::Matches($log, 'err-e').Count) 80
    Assert-Contains $log 'OUT_END'
    Assert-Contains $log 'ERR_END'
}

Test-Case 'command cancellation stops the active child and keeps captured output' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    $script:ChildPidPath = Join-Path $TestRoot 'child.pid'
    $survivedPath = Join-Path $TestRoot 'child-survived'
    function Show-Activity($Detail, $Clock) {
        if ((Test-Path -LiteralPath $script:ChildPidPath) -and $script:LastCommandText.Contains('child-running')) {
            throw [OperationCanceledException]::new('Synthetic command cancellation')
        }
    }
    Assert-Throws {
        Invoke-LoggedCommand -Label 'Testing cancellation' -FilePath $PowerShellPath -Arguments @(
            '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'wait', $script:ChildPidPath, $survivedPath
        )
    } 'Synthetic command cancellation'
    $childProcessId = [int][IO.File]::ReadAllText($script:ChildPidPath)
    $child = $null
    try { $child = [Diagnostics.Process]::GetProcessById($childProcessId) } catch [ArgumentException] {}
    try { Assert-True ($null -eq $child -or $child.HasExited) 'Cancelled child process is still running' }
    finally { if ($child) { $child.Dispose() } }
    Assert-True (-not (Test-Path -LiteralPath $survivedPath))
    Assert-Contains ([IO.File]::ReadAllText($script:LogPath)) 'child-running'
    Assert-True ($null -eq $script:ActiveProcess) 'Active child reference was not cleared'
    Assert-True (-not (Test-Path -LiteralPath ($script:LogPath + '.part'))) 'Interrupted command left a duplicate partial log'
}

Test-Case 'verbose output is delivered before the child finishes' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true; Verbose = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    $readyPath = Join-Path $TestRoot 'output received'
    $output = (& {
        Invoke-LoggedCommand -Label 'Testing live output' -FilePath $PowerShellPath -Arguments @(
            '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'handshake', $readyPath
        )
    } *>&1 | ForEach-Object {
        $line = $_.ToString()
        if ($line.Contains('stream-ready')) { Write-FixtureFile $readyPath 'received' }
        $line
    }) -join "`n"
    Assert-Contains $output 'stream-finished'
    Assert-True (Test-Path -LiteralPath $readyPath) 'Live output was not observed'
}

Test-Case 'verbose failures retain output without replaying it' {
    Initialize-Installer @{ Backend = 'cpu'; Yes = $true; Verbose = $true }
    $script:LogPath = Join-Path $TestRoot 'command.log'
    $output = Capture-Output {
        Assert-Throws {
            Invoke-LoggedCommand -Label 'Testing verbose failure' -FilePath $PowerShellPath -Arguments @(
                '-NoProfile', '-NonInteractive', '-File', $ChildPath, 'failure'
            )
        } '23'
    }
    Assert-Equal ([regex]::Matches($output, 'early diagnostic').Count) 1
    Assert-Equal ([regex]::Matches($output, 'last diagnostic').Count) 1
    Assert-Contains ([IO.File]::ReadAllText($script:LogPath)) 'context line 20'
}

$environmentKeys = @(
    'USERPROFILE', 'HOME', 'TEMP', 'TMP', 'TMPDIR', 'PATH', 'OS', 'CARGO_HOME', 'PROCESSOR_ARCHITECTURE',
    'PROCESSOR_ARCHITEW6432', 'MKLROOT', 'CUDA_PATH', 'NCCL_ROOT', 'NCCL_HOME',
    'MISTRALRS_INSTALL_VERSION', 'MISTRALRS_INSTALL_TAG', 'MISTRALRS_INSTALL_SOURCE',
    'MISTRALRS_INSTALL_FROM_SOURCE', 'MISTRALRS_INSTALL_NCCL',
    'MISTRALRS_INSTALL_BACKEND', 'MISTRALRS_INSTALL_WITH_FFMPEG', 'MISTRALRS_INSTALL_NO_NCCL',
    'MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH', 'MISTRALRS_INSTALL_YES', 'YES',
    'MISTRALRS_TEST_CHILD_VALUE'
)

$failures = @()
$passed = 0
foreach ($case in $script:Cases) {
    if ($case.Name -notlike $Filter) { continue }
    $testRoot = Join-Path ([IO.Path]::GetTempPath()) ('mistralrs-windows-test-' + [guid]::NewGuid().ToString('N'))
    $savedEnvironment = @{}
    foreach ($key in $environmentKeys) {
        $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        Remove-Item -LiteralPath ('Env:' + $key) -ErrorAction SilentlyContinue
    }
    try {
        $env:USERPROFILE = Join-Path $testRoot 'home'
        $env:HOME = $env:USERPROFILE
        $env:TEMP = Join-Path $testRoot 'temp'
        $env:TMP = $env:TEMP
        $env:TMPDIR = $env:TEMP
        $env:OS = $savedEnvironment['OS']
        $env:PROCESSOR_ARCHITECTURE = 'AMD64'
        New-Item -ItemType Directory -Path $env:USERPROFILE, $env:TEMP | Out-Null
        $childPath = Join-Path $testRoot 'child script.ps1'
        [IO.File]::WriteAllText($childPath, $childScript, (New-Object Text.UTF8Encoding $false))
        $definitions = if ($case.ConsoleFacade) { $consoleFunctionDefinitions } else { $functionDefinitions }
        $program = 'param($TestRoot, $ChildPath, $PowerShellPath, $InstallerPath)' + "`n" + $definitions + "`n" + $fixtureFunctions + "`n" + $case.Body
        New-Module -Name ('InstallerTest_' + [guid]::NewGuid().ToString('N')) -ScriptBlock ([scriptblock]::Create($program)) -ArgumentList $testRoot, $childPath, $PowerShellPath, (Resolve-Path $InstallerPath).Path | Out-Null
        $passed++
        Write-Host ('PASS ' + $case.Name)
    } catch {
        $failures += $case.Name
        Write-Host ('FAIL ' + $case.Name + ': ' + $_.ToString()) -ForegroundColor Red
        Write-Host $_.ScriptStackTrace
    } finally {
        foreach ($key in $environmentKeys) {
            if ($null -eq $savedEnvironment[$key]) { Remove-Item -LiteralPath ('Env:' + $key) -ErrorAction SilentlyContinue }
            else { [Environment]::SetEnvironmentVariable($key, $savedEnvironment[$key], 'Process') }
        }
        if (Test-Path -LiteralPath $testRoot) {
            Remove-Item -LiteralPath $testRoot -Recurse -Force
        }
    }
}

Write-Host ("`n{0} passed; {1} failed" -f $passed, $failures.Count)
if ($failures.Count -gt 0) { exit 1 }
if ($passed -eq 0) { throw 'No tests matched the filter' }
