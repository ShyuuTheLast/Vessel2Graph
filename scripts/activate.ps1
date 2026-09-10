# Dot-source this file to activate in the current PowerShell session:
# . .\scripts\activate.ps1
$projectRoot = Split-Path -Parent $PSScriptRoot
$manager = Join-Path $projectRoot '.tools\miniconda3\Scripts\conda.exe'
$projectEnv = Join-Path $projectRoot '.tools\miniconda3\envs\vessel2graph'
if (-not (Test-Path -LiteralPath $manager)) { throw 'Miniconda is not installed in this workspace.' }
if (-not (Test-Path -LiteralPath "$projectEnv\python.exe")) { throw 'The vessel2graph environment has not been created.' }
(& $manager 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate $projectEnv
