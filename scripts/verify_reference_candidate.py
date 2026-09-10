"""Compare saved candidate geometry/radii and branch metrics to the accepted reference."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from vessel2graph.io import digest_file,load_arrays,write_json


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=root/'runs/macaque-reference-calibrated-v1')
    args=parser.parse_args()
    base=root/'runs/macaque'
    frozen=json.loads((root/'docs/PHASE1_BASELINE.json').read_text())['artifacts']
    for name in ('graph.npz','skeleton.npz','branches.json','metrics.json'):
        if digest_file(base/name)!=frozen['runs/macaque/'+name]['sha256']:
            raise ValueError('Frozen baseline mismatch: '+name)
    old=load_arrays(base/'graph.npz')
    new=load_arrays(args.candidate/'graph.npz')
    bm=json.loads((base/'metrics.json').read_text())
    cm=json.loads((args.candidate/'metrics.json').read_text())
    distances,mapping=cKDTree(new['vertices']).query(old['vertices'])
    reverse=cKDTree(old['vertices']).query(new['vertices'])[0]
    edges={tuple(sorted(e)) for e in new['edges'].tolist()}
    preserved=sum(distances[a]==0 and distances[b]==0 and
        tuple(sorted((int(mapping[a]),int(mapping[b])))) in edges for a,b in old['edges'])
    exact=(len(old['vertices'])==len(new['vertices']) and len(np.unique(mapping))==len(mapping)
        and np.all(distances==0) and preserved==len(old['edges'])==len(edges))
    keys=('vertices','edges','components','endpoints','junctions','branches','total_length_nm')
    report=dict(metrics={k:dict(baseline=bm[k],candidate=cm[k]) for k in keys},
        exact_geometry_match=bool(exact), baseline_edges_preserved=int(preserved),
        baseline_vertex_max_offset_nm=float(distances.max()),candidate_to_baseline_max_nm=float(reverse.max()),
        exact_radius_match_at_mapped_vertices=bool(np.array_equal(old['radius'],new['radius'][mapping])) if exact else None,
        short_branches_below_5um=sum(b['length_nm']<5000 for b in json.loads((args.candidate/'branches.json').read_text())),
        candidate_graph_sha256=digest_file(args.candidate/'graph.npz'),baseline_graph_sha256=digest_file(base/'graph.npz'),
        skeleton_seconds=json.loads((args.candidate/'skeleton.stage.json').read_text())['elapsed_seconds'],
        caveat='Reference-calibrated training-set reproduction, not independent accuracy or transfer validation.')
    write_json(args.candidate/'reference-comparison.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
