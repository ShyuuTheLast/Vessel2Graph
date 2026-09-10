# Local environment

Miniconda is installed at `.tools/miniconda3`. The dedicated environment is
`.tools/miniconda3/envs/vessel2graph` and uses Python 3.11.16. Original standalone interpreters
(3.10.6, 3.11.9, 3.14.3 under AppData/Local/Python) remain unchanged.

From the project directory in PowerShell:

```powershell
. .\scripts\activate.ps1
python --version
python -m pip check
python scripts/check_environment.py --input 'data\macaque_mpi_176-240nm_bv_gt_cc.h5'
```

Activation only changes the current shell. Alternatively invoke the environment's python.exe
by its full path; scientific DLL loading is most reliable through activation or `conda run`.
No global PATH or PowerShell profile changes were made. The installer used JustMe,
RegisterPython=0, AddToPath=0 and silent mode inside the workspace.

The verified installer is retained in work/installers. Its filename and checksum are in STATUS.
Miniconda's base version is independent of the isolated environment's Python version.

## Reproduce

Install Miniconda at the same relative path using the official installer, then run
`scripts/create_environment.ps1`. It creates Python 3.11 from conda-forge and installs the
core pins using binary wheels. For exact reconstruction on Windows x64, use the explicit
Conda file first and then the full pip snapshot:

```powershell
& .\.tools\miniconda3\Scripts\conda.exe create --prefix .\.tools\miniconda3\envs\vessel2graph --file locks/conda-win-64.txt --yes
& .\.tools\miniconda3\envs\vessel2graph\python.exe -m pip install --only-binary=:all: -r locks/requirements-win-py311.txt
```

Run reconstruction only into a new environment. The lock includes transitive packages from the
verified setup; requirements-core.txt records direct intended dependencies. Linux/Misha needs a
separate platform resolution and smoke test. Neither CuPy nor visualization packages are required
for this phase-1 CPU environment. No WSL installation was necessary.

## Compatibility finding

Kimimaro 4.1.2 installed successfully from a Windows wheel, but current cloud-volume 12.14.4
failed at runtime with an osteoid casting exception. The original repository's cloud-volume
10.4.0 pin fixes the tested case. Do not remove it without rerunning the synthetic smoke test.
