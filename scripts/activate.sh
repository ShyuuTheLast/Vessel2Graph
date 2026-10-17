# Source from Bash to activate the workspace environment on Misha:
# source scripts/activate.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Source this script: source scripts/activate.sh' >&2
    exit 1
fi

module load miniconda || return
conda activate "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/.tools/conda-envs/vessel2graph"
