[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $RootDir "output\runtime\dev-servers.json"

function Stop-ProcessTree {
    param([int]$ProcessId)

    $Children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue
    foreach ($Child in $Children) {
        Stop-ProcessTree -ProcessId $Child.ProcessId
    }

    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($Process) {
        Stop-Process -Id $ProcessId -Force
        Write-Host "Stopped process: $ProcessId"
    }
}

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "Dev server process file not found: $PidFile"
    exit 0
}

$ProcessInfo = Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json
$Pids = @($ProcessInfo.backendPid, $ProcessInfo.frontendPid) | Where-Object { $_ }

foreach ($ProcessId in $Pids) {
    Stop-ProcessTree -ProcessId $ProcessId
}

Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
Write-Host "Development services stopped."
