param([string]$PythonExe = '', [string[]]$Cases = @())
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$evidence = Join-Path (Split-Path -Parent $PSScriptRoot) 'evidence'
$outDir = Join-Path $evidence 'dev9_validation'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$log = Join-Path $outDir 'run.log'
Start-Transcript -Path $log -Append | Out-Null
try {
    & (Join-Path $PSScriptRoot 'Setup-Python.ps1') -PythonExe $PythonExe
    $localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    & $localPython -c 'import tpmslab; print("TPMS Lab", tpmslab.__version__)'
    & $localPython (Join-Path $evidence 'run_dev9_validation.py') @Cases
    Write-Host "Exit code: $LASTEXITCODE"
} finally {
    Stop-Transcript | Out-Null
}
