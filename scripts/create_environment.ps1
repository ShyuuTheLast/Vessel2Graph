# Run after installing Miniconda in .tools/miniconda3. Network access is required.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$manager = Join-Path $projectRoot '.tools\miniconda3\Scripts\conda.exe'
$projectEnv = Join-Path $projectRoot '.tools\miniconda3\envs\vessel2graph'
$env:CONDA_PKGS_DIRS = Join-Path $projectRoot '.runtime\conda-pkgs'
if (-not (Test-Path -LiteralPath $manager)) { throw 'Install Miniconda first; see docs/LOCAL_SETUP.md.' }
if (-not (Test-Path -LiteralPath "$projectEnv\python.exe")) {
    & $manager create --prefix $projectEnv --override-channels --channel conda-forge python=3.11 pip --yes --no-default-packages
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}
& "$projectEnv\python.exe" -m pip install --only-binary=:all: -r (Join-Path $projectRoot 'requirements-core.txt')
if ($LASTEXITCODE -ne 0) { throw 'Core package installation failed.' }
& "$projectEnv\python.exe" -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
