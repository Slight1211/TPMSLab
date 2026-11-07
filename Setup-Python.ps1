param([string]$PythonExe = '', [switch]$WithTests)
$ErrorActionPreference = 'Stop'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($PythonExe -and -not (Test-Path -LiteralPath $PythonExe)) { throw "Python executable does not exist: $PythonExe" }
if (-not (Test-Path -LiteralPath $localPython)) {
    $candidates = @($PythonExe,
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'))
    $basePython = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    if (-not $basePython) { throw 'Provide a Python 3.11+ executable using -PythonExe.' }
    & $basePython -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
# Always retry installation: an existing python.exe does not prove that a
# previous pip installation completed. pip reuses already satisfied packages.
$package = if ($WithTests) { "$PSScriptRoot[test]" } else { $PSScriptRoot }
& $localPython -m pip install --disable-pip-version-check -e $package
if ($LASTEXITCODE -ne 0) { throw 'Dependency/package installation failed. Correct the connection and run this setup again.' }
& $localPython -c 'import tpmslab,numpy,scipy,trimesh; print("TPMS Lab",tpmslab.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Package import verification failed.' }
Write-Host "Python ready: $localPython"
