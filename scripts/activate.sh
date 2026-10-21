# Source from Bash to activate the workspace environment on Misha:
# source scripts/activate.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Source this script: source scripts/activate.sh' >&2
    exit 1
fi

_vesselempy_activate() {
    local project_root project_env legacy_env
    module load miniconda || return
    project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return
    project_env="$project_root/.tools/conda-envs/VesselEMPy"
    legacy_env="$project_root/.tools/conda-envs/vessel2graph"
    if [[ ! -x "$project_env/bin/python" && -x "$legacy_env/bin/python" ]]; then
        project_env="$legacy_env"
    fi
    if [[ ! -x "$project_env/bin/python" ]]; then
        echo 'Create the VesselEMPy environment first; see docs/CLUSTER_HANDOFF.md.' >&2
        return 1
    fi
    conda activate "$project_env"
}

if _vesselempy_activate; then
    unset -f _vesselempy_activate
else
    unset -f _vesselempy_activate
    return 1
fi
