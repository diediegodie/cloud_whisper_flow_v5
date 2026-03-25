param(
    [int]$Attempts = 10,
    [int]$DurationSeconds = 3,
    [switch]$SkipRecording
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "=== Windows Validation Runner ==="
Write-Host "Project: $ProjectRoot"

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

pytest -q tests\core tests\backend tests\project

$SmokeArgs = @(
    "scripts/windows_native_smoke.py",
    "--attempts", "$Attempts",
    "--duration-seconds", "$DurationSeconds"
)

if ($SkipRecording) {
    $SmokeArgs += "--skip-recording"
}

python @SmokeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Smoke validation failed with exit code $LASTEXITCODE"
}

Write-Host "Validation completed successfully."
