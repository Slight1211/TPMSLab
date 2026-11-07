param([int[]]$Resolutions = @(8, 12, 16, 24), [string[]]$Designs = @('Gyroid_central', 'Diamond_low', 'Primitive_Schwartz_central'), [switch]$SkipElasticity, [switch]$SkipFlow, [string]$ResumeDirectory = '', [int]$Timeout = 3600, [int]$MaxTetrahedra = 1000000, [string]$PythonExe = '', [string]$PreferencesDirectory = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& (Join-Path $PSScriptRoot 'Setup-Python.ps1') -PythonExe $PythonExe
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path (Split-Path -Parent $PSScriptRoot) "comsol_convergence_$stamp"
if ($ResumeDirectory) { $out = $ResumeDirectory }
$arguments = @('benchmarks\comsol_convergence.py', '--out', $out, '--resolutions') + ($Resolutions | ForEach-Object { "$_" }) + @('--designs') + $Designs
$arguments += @('--timeout', "$Timeout", '--max-tetrahedra', "$MaxTetrahedra")
if ($ResumeDirectory) { $arguments += '--resume' }
if ($PreferencesDirectory) { $arguments += @('--preferences-directory', $PreferencesDirectory) }
if ($SkipElasticity) { $arguments += '--skip-elasticity' }
if ($SkipFlow) { $arguments += '--skip-flow' }
Write-Host "Writing results to $out"
& $localPython @arguments 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot "comsol_convergence_$stamp.log")
exit $LASTEXITCODE
