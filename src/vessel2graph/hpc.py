"""Slurm batch scripts for whole-volume stages, and the job's own memory allowance.

Tracing one connected vessel is a single in-memory Kimimaro call, so a stage is
one large-memory job, not an array. Submission is opt-in; writing the script is not.
"""
from pathlib import Path
import math
import os
import shlex
import subprocess
import sys


def job_memory_limit():
    """Bytes granted to this Slurm job, or None outside a memory-limited job."""
    node=os.environ.get('SLURM_MEM_PER_NODE')
    if node:
        return int(node)*2**20
    cpu=os.environ.get('SLURM_MEM_PER_CPU')
    if cpu:
        return int(cpu)*int(os.environ.get('SLURM_CPUS_ON_NODE',1))*2**20
    return None


def requested_memory_gib(cfg):
    # Default allocation is the process budget plus its reserve, which makes
    # check_memory's two conditions coincide inside the job.
    explicit=cfg['hpc']['mem_gib']
    budget=cfg['resources']['max_memory_gib']+cfg['resources']['reserve_memory_gib']
    if explicit is not None and explicit<budget:
        raise ValueError('hpc.mem_gib is below resources.max_memory_gib + reserve_memory_gib')
    return budget if explicit is None else explicit


def write_job(cfg, command, arguments):
    """Write <output>/slurm/<command>.sbatch running the given CLI arguments."""
    options=cfg['hpc']
    folder=Path(cfg['output']['directory'])/'slurm'
    folder.mkdir(parents=True,exist_ok=True)
    lines=['#!/bin/bash',f'#SBATCH --job-name=v2g-{command}',f'#SBATCH --partition={options["partition"]}',
        f'#SBATCH --cpus-per-task={options["cpus"]}',
        f'#SBATCH --mem={math.ceil(requested_memory_gib(cfg)*1024)}M',
        f'#SBATCH --time={options["time"]}',f'#SBATCH --output={folder}/{command}-%j.log']
    if options['account']:
        lines.append(f'#SBATCH --account={options["account"]}')
    # The environment's interpreter is addressed directly; setup is for extra modules.
    # GNU time reports the measured peak RSS and wall time at the end of the log.
    lines+=['set -euo pipefail',*options['setup'],
        '/usr/bin/time -v '+shlex.join([sys.executable,'-m','vessel2graph',command,*arguments])]
    script=folder/f'{command}.sbatch'
    script.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return script


def submit(script):
    return subprocess.run(['sbatch',str(script)],check=True,capture_output=True,text=True).stdout.strip()
