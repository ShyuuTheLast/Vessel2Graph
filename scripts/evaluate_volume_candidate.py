"""Evaluate saved binary-only candidates; never feed reference data into estimation."""
from pathlib import Path
import argparse
import json
import h5py
import numpy as np
import yaml
from scipy.spatial import cKDTree
from VesselEMPy.io import load_arrays,write_json,digest_file
from VesselEMPy.segment import check_memory


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=root/'runs/macaque-volume-estimated-v2')
    args=parser.parse_args()
    base=root/'runs/macaque'
    frozen=json.loads((root/'docs/PHASE1_BASELINE.json').read_text())['artifacts']
    for name in ('graph.npz','branches.json','metrics.json','skeleton.npz'):
        if digest_file(base/name)!=frozen['runs/macaque/'+name]['sha256']:
            raise ValueError('Baseline changed: '+name)
    old=load_arrays(base/'graph.npz');new=load_arrays(args.candidate/'graph.npz')
    bm=json.loads((base/'metrics.json').read_text());cm=json.loads((args.candidate/'metrics.json').read_text())
    branches=json.loads((args.candidate/'branches.json').read_text())
    old_branches=json.loads((base/'branches.json').read_text())
    d=cKDTree(new['vertices']).query(old['vertices'])[0]
    reverse=cKDTree(old['vertices']).query(new['vertices'])[0]
    cfg=yaml.safe_load((args.candidate/'resolved-config.yaml').read_text())
    spacing=np.array(cfg['input']['voxel_size_nm'])
    ijk=np.rint(new['vertices']/spacing).astype(int)
    foreground=0
    with h5py.File(cfg['input']['path']) as f:
        ds=f[cfg['input']['dataset']]
        check_memory(cfg['resources'],ds.size*ds.dtype.itemsize)
        volume=ds[:]
        foreground=int(np.count_nonzero(volume[tuple(ijk.T)]))
        del volume
    q=[0,.5,.9,.95,.99,1]
    result=dict(metrics={k:dict(baseline=bm[k],candidate=cm[k]) for k in
        ('vertices','edges','components','endpoints','junctions','branches','total_length_nm')},
        quantiles=q,baseline_to_candidate_nm=np.quantile(d,q).tolist(),
        candidate_to_baseline_nm=np.quantile(reverse,q).tolist(),
        baseline_vertices_within_1um=float(np.mean(d<=1000)),
        short_branches={str(t):sum(b['length_nm']<1000*t for b in branches) for t in (2,5,10)},
        vertices_in_prepared_mask=foreground,vertices_outside_prepared_mask=len(ijk)-foreground,
        maximum_edge_length_nm=float(np.linalg.norm(new['vertices'][new['edges'][:,0]]-new['vertices'][new['edges'][:,1]],axis=1).max()),
        per_baseline_branch=[dict(id=b['id'],length_um=b['length_nm']/1000,
            p95_offset_nm=float(np.quantile(d[b['nodes']],.95)),max_offset_nm=float(d[b['nodes']].max())) for b in old_branches],
        skeleton_seconds=json.loads((args.candidate/'skeleton.stage.json').read_text())['elapsed_seconds'],
        candidate_graph_sha256=digest_file(args.candidate/'graph.npz'),
        caution='Reference is evaluation only, not ground truth. Vertex proximity and component counts are not proof of anatomical completeness.')
    write_json(args.candidate/'baseline-evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_baseline_branch'},indent=2))


if __name__=='__main__':
    main()
