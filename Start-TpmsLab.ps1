param([int]$Port = 0, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$verifiedPython = Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp\tpms-quality-venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) {
    $taskPython = $localPython
} elseif (Test-Path -LiteralPath $verifiedPython) {
    $taskPython = $verifiedPython
} else {
    & py -3.11 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11, then run this launcher again.' }
    $taskPython = $localPython
    & $taskPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-quality-verified.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $taskPython -m pip install --no-deps -e $PSScriptRoot
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
}
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$argumentsForWeb = @('-m', 'tpmslab', 'web', '--output', (Join-Path $PSScriptRoot 'local_models'), '--port', "$Port")
if ($NoBrowser) { $argumentsForWeb += '--no-browser' }
& $taskPython @argumentsForWeb
exit $LASTEXITCODE
