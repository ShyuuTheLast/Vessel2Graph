"""Evaluate user-identified candidate artifacts; never used by parameter estimation."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from VesselEMPy.io import load_arrays,write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate',type=Path,required=True)
args=parser.parse_args()
old_dir=Path('runs/macaque-volume-estimated-v2')
old=load_arrays(old_dir/'graph.npz')
new=load_arrays(args.candidate/'graph.npz')
ob=json.loads((old_dir/'branches.json').read_text())
nb=json.loads((args.candidate/'branches.json').read_text())
degree=np.bincount(new['edges'].ravel(),minlength=len(new['vertices']))
junction=np.array([73280,173824,143360])
result={'evaluation_reference':str(old_dir),'old_spur_id':12,
    'old_junction_zyx_nm':junction.tolist(),
    'nearest_new_junction_nm':float(np.linalg.norm(new['vertices'][degree>=3]-junction,axis=1).min()),
    'new_local_endpoints_zyx_nm':new['vertices'][(degree==1)&(np.linalg.norm(new['vertices']-junction,axis=1)<15000)].tolist(),
    'old_branch_matches':[]}
for bid in (8,12,13):
    b=next(b for b in ob if b['id']==bid)
    points=old['vertices'][b['nodes']]
    scores=[]
    for candidate in nb:
        d=cKDTree(new['vertices'][candidate['nodes']]).query(points)[0]
        scores.append(dict(id=candidate['id'],length_um=candidate['length_nm']/1000,
            fraction_within_1um=float(np.mean(d<=1000)),p95_nm=float(np.quantile(d,.95))))
    scores.sort(key=lambda s:s['p95_nm'])
    result['old_branch_matches'].append(dict(old_id=bid,old_length_um=b['length_nm']/1000,best_new_branches=scores[:3]))
write_json(args.candidate/'closing-evaluation.json',result)
print(json.dumps(result,indent=2))
