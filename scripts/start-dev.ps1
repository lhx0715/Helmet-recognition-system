[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [switch]$DryRun,
    [string]$PythonPath,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$RuntimeDir = Join-Path $RootDir "output\runtime"
$PidFile = Join-Path $RuntimeDir "dev-servers.json"

function Find-Python {
    param([string]$PreferredPath)

    if ($PreferredPath) {
        if (-not (Test-Path -LiteralPath $PreferredPath)) {
            throw "Python path does not exist: $PreferredPath"
        }
        return (Resolve-Path -LiteralPath $PreferredPath).Path
    }

    $candidates = @(
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files\Python310\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "Python was not found. Please install Python 3.10 or 3.11."
}

function Find-Npm {
    $command = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $command = Get-Command npm -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "npm was not found. Please install Node.js first."
}

function Find-Shell {
    $command = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $command = Get-Command powershell.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "PowerShell was not found."
}

$Python = Find-Python -PreferredPath $PythonPath
$Npm = Find-Npm
$Shell = Find-Shell
$PythonVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($PythonVersion -notin @("3.10", "3.11")) {
    Write-Warning "Current Python version is $PythonVersion. YOLO/Torch is recommended with Python 3.10 or 3.11."
}

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

if ($InstallDeps) {
    Write-Host "Installing backend dependencies..."
    & $Python -m pip install -r (Join-Path $BackendDir "requirements.txt")

    Write-Host "Installing frontend dependencies..."
    Push-Location $FrontendDir
    try {
        & $Npm install
    }
    finally {
        Pop-Location
    }
}

$BackendCommand = @"
Set-Location '$BackendDir'
Write-Host 'Backend API: http://localhost:$BackendPort'
`$env:PYTHONUTF8='1'
& '$Python' -m uvicorn main:app --host 0.0.0.0 --port $BackendPort
"@

$FrontendCommand = @"
Set-Location '$FrontendDir'
Write-Host 'Frontend UI: http://localhost:$FrontendPort'
& '$Npm' run dev -- --host localhost --port $FrontendPort
"@

if ($DryRun) {
    Write-Host "Python: $Python"
    Write-Host "npm: $Npm"
    Write-Host "Shell: $Shell"
    Write-Host "Backend command:"
    Write-Host $BackendCommand
    Write-Host "Frontend command:"
    Write-Host $FrontendCommand
    exit 0
}

$BackendProcess = Start-Process -FilePath $Shell -ArgumentList @("-NoExit", "-Command", $BackendCommand) -PassThru
$FrontendProcess = Start-Process -FilePath $Shell -ArgumentList @("-NoExit", "-Command", $FrontendCommand) -PassThru

$ProcessInfo = [ordered]@{
    backendPid = $BackendProcess.Id
    frontendPid = $FrontendProcess.Id
    backendUrl = "http://localhost:$BackendPort"
    frontendUrl = "http://localhost:$FrontendPort"
    startedAt = (Get-Date).ToString("s")
}

$ProcessInfo | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding UTF8

Write-Host ""
Write-Host "Development environment started."
Write-Host "Backend API: http://localhost:$BackendPort"
Write-Host "Frontend UI: http://localhost:$FrontendPort"
Write-Host "API docs: http://localhost:$BackendPort/docs"
Write-Host "Process info: $PidFile"
Write-Host ""
Write-Host "To stop: close the two opened terminal windows, or run scripts\stop-dev.ps1."
