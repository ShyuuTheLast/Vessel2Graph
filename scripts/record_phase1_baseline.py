"""Record existing outputs and candidate inspection sites without rerunning science."""
from pathlib import Path
import json
import argparse
import itertools
import networkx as nx
import numpy as np
from vessel2graph.io import digest_file, load_arrays, write_json

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True,help='Original macaque HDF5 input')
source_input=parser.parse_args().input
data = load_arrays(root / 'runs/macaque/graph.npz')
graph = nx.Graph()
graph.add_nodes_from(range(len(data['vertices'])))
graph.add_edges_from(data['edges'])
components = sorted(nx.connected_components(graph), key=lambda c: min(c))
spacing = np.array([320, 256, 256])

def point(node):
    return {'node': int(node), 'zyx_nm': data['vertices'][node].tolist(),
            'zyx_voxel': (data['vertices'][node] / spacing).tolist()}

gaps = []
for a, b in itertools.combinations(components, 2):
    endpoints_a = [n for n in a if graph.degree(n) == 1]
    endpoints_b = [n for n in b if graph.degree(n) == 1]
    if not endpoints_a or not endpoints_b:
        continue
    distance, left, right = min((float(np.linalg.norm(data['vertices'][u]-data['vertices'][v])), u, v)
                                for u in endpoints_a for v in endpoints_b)
    gaps.append({'distance_nm': distance, 'endpoints': [point(left), point(right)],
                 'interpretation': 'Nearest inter-component endpoints: candidate inspection, not proof of missing connection'})
branches = json.loads((root / 'runs/macaque/branches.json').read_text())
smallest = min(branches, key=lambda b: (b['median_radius_nm'], b['id']))
artifacts = {}
for run in ('macaque', 'macaque-caliber', 'macaque-radius'):
    for name in ('skeleton.npz', 'graph.npz', 'branches.json', 'metrics.json', 'segmentation.h5',
                 'segmentation.json', 'report.json', 'resolved-config.yaml', 'provenance.json'):
        file = root / 'runs' / run / name
        artifacts[str(file.relative_to(root)).replace('\\','/')] = {'sha256': digest_file(file), 'bytes': file.stat().st_size}
write_json(root / 'docs/PHASE1_BASELINE.json', {
    'recorded_date': '2026-09-10', 'status': 'User accepted phase-1 execution and visualization; not ground-truth accuracy',
    'input': {'path': source_input.name,
              'sha256': digest_file(source_input),
              'dataset': 'main', 'labels': [1], 'voxel_size_nm_zyx': spacing.tolist()},
    'artifacts': artifacts,
    'inspection': {'candidate_gaps': gaps,
        'junctions': [point(n) for n in graph if graph.degree(n) > 2],
        'smallest_median_branch': {'branch_id': smallest['id'], 'median_radius_nm': smallest['median_radius_nm'],
                                  'midpoint': point(smallest['nodes'][len(smallest['nodes'])//2])}},
    'notes': ['Node and branch IDs apply only to this baseline; compare future runs in physical coordinates.',
              'Do not regenerate this manifest after changing outputs unless intentionally establishing a new baseline.']})
print('Recorded docs/PHASE1_BASELINE.json')
