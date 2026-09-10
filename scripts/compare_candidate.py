"""Structural comparison against phase-1; these metrics do not establish accuracy."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from vessel2graph.io import load_arrays, write_json

root = Path(__file__).resolve().parents[1]
baseline = root / 'runs/macaque'
candidate = root / 'runs/macaque-estimated-v1'
old = load_arrays(baseline / 'skeleton.npz')
new = load_arrays(candidate / 'skeleton.npz')
old_metrics = json.loads((baseline / 'metrics.json').read_text())
new_metrics = json.loads((candidate / 'metrics.json').read_text())
keys = ['vertices','edges','components','endpoints','junctions','branches','total_length_nm']
comparison = {key: {'baseline':old_metrics[key], 'candidate':new_metrics[key]} for key in keys}
for name, source, target in [('baseline_to_candidate',old,new), ('candidate_to_baseline',new,old)]:
    distances = cKDTree(target['vertices']).query(source['vertices'])[0]
    comparison[name + '_nearest_vertex_nm'] = dict(zip(['median','p95','maximum'],
                                                      np.quantile(distances,[.5,.95,1]).tolist()))
sites = json.loads((root / 'docs/PHASE1_BASELINE.json').read_text())['inspection']['candidate_gaps']
tree = cKDTree(new['vertices'])
comparison['gap_sites'] = []
for site in sites:
    dist, node_ids = tree.query([p['zyx_nm'] for p in site['endpoints']])
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(len(new['vertices'])))
    g.add_edges_from(new['edges'])
    comparison['gap_sites'].append({'baseline_gap_nm': site['distance_nm'],
        'nearest_candidate_nodes': node_ids.tolist(), 'offset_nm': dist.tolist(),
        'connected_in_candidate': nx.has_path(g, int(node_ids[0]), int(node_ids[1]))})
comparison['caution'] = 'Nearest-vertex distances and connectivity are diagnostic, not proof of correct tracing or joins.'
comparison['candidate_skeleton_seconds'] = json.loads((candidate / 'skeleton.stage.json').read_text())['elapsed_seconds']
write_json(candidate / 'baseline-comparison.json', comparison)
print(json.dumps(comparison,indent=2))
