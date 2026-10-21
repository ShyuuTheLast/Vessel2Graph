# Dot-source this file to activate in the current PowerShell session:
# . .\scripts\activate.ps1
$projectRoot = Split-Path -Parent $PSScriptRoot
$manager = Join-Path $projectRoot '.tools\miniconda3\Scripts\conda.exe'
$projectEnv = Join-Path $projectRoot '.tools\miniconda3\envs\VesselEMPy'
$legacyEnv = Join-Path $projectRoot '.tools\miniconda3\envs\vessel2graph'
if (-not (Test-Path -LiteralPath "$projectEnv\python.exe") -and (Test-Path -LiteralPath "$legacyEnv\python.exe")) {
    $projectEnv = $legacyEnv
}
if (-not (Test-Path -LiteralPath $manager)) { throw 'Miniconda is not installed in this workspace.' }
if (-not (Test-Path -LiteralPath "$projectEnv\python.exe")) { throw 'The VesselEMPy environment has not been created; see docs/LOCAL_SETUP.md.' }
(& $manager 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate $projectEnv
