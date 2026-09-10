"""Development-only tracing with live memory limits in addition to preflight.

The parent terminates only its own worker if the configured RSS budget or system
reserve is crossed. Sampling cannot guarantee protection against allocation bursts.
Scientific execution and output provenance use the ordinary pipeline unchanged.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import psutil
from vessel2graph.config import load_config
from vessel2graph.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    cfg=load_config(args.config)
    if cfg['skeleton']['parallel']!=1:
        raise ValueError('This development monitor supports serial tracing only')
    if args.worker:
        import vessel2graph.pipeline as pipeline
        pipeline.run(cfg,target='graph')
        return
    output=Path(cfg['output']['directory'])
    output.mkdir(parents=True,exist_ok=True)
    peak=0;minimum=psutil.virtual_memory().available;reason=None
    started=time.perf_counter()
    with (output/'monitored-tracing.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
            '--config',args.config,'--worker'],stdout=log,stderr=subprocess.STDOUT)
        process=psutil.Process(worker.pid)
        try:
            while worker.poll() is None:
                try:
                    rss=process.memory_info().rss
                except psutil.NoSuchProcess:
                    break
                available=psutil.virtual_memory().available
                peak=max(peak,rss);minimum=min(minimum,available)
                if rss>cfg['resources']['max_memory_gib']*2**30:
                    reason='Process RSS budget crossed'
                elif available<cfg['resources']['reserve_memory_gib']*2**30:
                    reason='System free-memory reserve crossed'
                if reason:
                    worker.terminate()
                    break
                time.sleep(.1)
            code=worker.wait()
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait()
    report=dict(status='complete' if code==0 else 'failed',exit_code=code,
        termination_reason=reason,sampled_peak_rss_gib=peak/2**30,
        minimum_available_gib=minimum/2**30,elapsed_seconds=time.perf_counter()-started,
        resources=cfg['resources'],sampling_seconds=.1,
        limitation='Sampled RSS can miss allocation bursts; development opt-in, not a new general preflight model.')
    write_json(output/'memory-monitor.json',report)
    print(json.dumps(report,indent=2))
    sys.exit(0 if code==0 else 1)


if __name__=='__main__':
    main()
