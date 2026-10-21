"""Diagnose added tracing using saved skeletons; never prune or rerun science."""
from pathlib import Path
import csv
import json
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from VesselEMPy.io import load_arrays, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/macaque-estimated-v1'
old = load_arrays(ROOT / 'runs/macaque/skeleton.npz')
new = load_arrays(OUT / 'skeleton.npz')
branches = json.loads((OUT / 'branches.json').read_text())
old_branches = json.loads((ROOT / 'runs/macaque/branches.json').read_text())

def graph(data):
    g = nx.Graph()
    g.add_nodes_from(range(len(data['vertices'])))
    for a, b in data['edges']:
        g.add_edge(int(a), int(b), weight=float(np.linalg.norm(data['vertices'][a]-data['vertices'][b])))
    return g

def quantiles(values):
    return dict(zip(['min', 'p25', 'median', 'p75', 'p90', 'max'],
                    np.quantile(values, [0, .25, .5, .75, .9, 1]).tolist()))

og, ng = graph(old), graph(new)
dist, mapping = cKDTree(new['vertices']).query(old['vertices'])
assert np.all(dist == 0), 'Requires exact preservation of baseline vertex positions'
assert len(np.unique(mapping)) == len(mapping), 'Ambiguous coordinate mapping'
old_edges = {tuple(sorted((int(mapping[a]), int(mapping[b])))) for a,b in og.edges}
assert old_edges.issubset({tuple(sorted(e)) for e in ng.edges}), 'Baseline edges were changed'
baseline_nodes = set(mapping.tolist())
endpoints = [n for n in og if og.degree(n) == 1]
endpoint_xyz = old['vertices'][endpoints]
endpoint_tree = cKDTree(endpoint_xyz)
geo = nx.multi_source_dijkstra_path_length(og, endpoints, weight='weight')
new_to_old = {int(v): i for i,v in enumerate(mapping)}
added = nx.Graph()
for a,b,d in ng.edges(data=True):
    if tuple(sorted((a,b))) not in old_edges:
        added.add_edge(a,b,**d)

clusters = []
for nodes in nx.connected_components(added):
    sub = added.subgraph(nodes)
    anchors = sorted(nodes & baseline_nodes)
    clusters.append(dict(nodes=len(nodes), edges=sub.number_of_edges(),
        length_um=sum(d['weight'] for _,_,d in sub.edges(data=True))/1000,
        anchor_count=len(anchors), anchor_nodes=anchors,
        nearest_baseline_endpoint_um=min(endpoint_tree.query(new['vertices'][anchors])[0])/1000,
        baseline_geodesic_to_endpoint_um=min(geo[new_to_old[a]] for a in anchors)/1000,
        nearest_endpoint_id=int(endpoints[endpoint_tree.query(new['vertices'][anchors[0]])[1]])))

rows=[]
for branch in branches:
    p=branch['nodes']
    xyz=new['vertices'][p]
    lengths=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
    center=np.average((xyz[:-1]+xyz[1:])/2,axis=0,weights=lengths)
    added_length=sum(l for a,b,l in zip(p[:-1],p[1:],lengths)
                     if tuple(sorted((a,b))) not in old_edges)
    rows.append(dict(id=branch['id'], length_um=branch['length_nm']/1000,
        radius_um=branch['median_radius_nm']/1000,
        length_over_diameter=branch['length_nm']/(2*branch['median_radius_nm']),
        terminal=ng.degree(p[0])==1 or ng.degree(p[-1])==1,
        added_length_fraction=added_length/branch['length_nm'],
        endpoint_distance_um=float(endpoint_tree.query(center)[0])/1000,
        z_nm=center[0], y_nm=center[1], x_nm=center[2]))
with (OUT/'branch-diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

summary=dict(baseline_edges_preserved=len(old_edges), added_edges=added.number_of_edges(),
    baseline_length_um=sum(b['length_nm'] for b in old_branches)/1000,
    added_length_um=sum(c['length_um'] for c in clusters),
    baseline_branch_length_um=quantiles([b['length_nm']/1000 for b in old_branches]),
    candidate_branch_length_um=quantiles([r['length_um'] for r in rows]),
    added_clusters=len(clusters), cluster_length_um=quantiles([c['length_um'] for c in clusters]),
    cluster_anchor_counts={str(k):sum(c['anchor_count']==k for c in clusters)
                           for k in sorted({c['anchor_count'] for c in clusters})},
    cluster_geodesic_distance_to_endpoint_um=quantiles([c['baseline_geodesic_to_endpoint_um'] for c in clusters]),
    short_branch_thresholds=[], endpoint_neighborhoods=[])
for cutoff in [2,5,10,20]:
    selected=[r for r in rows if r['length_um']<cutoff]
    centers=np.array([[r['z_nm'],r['y_nm'],r['x_nm']] for r in selected])
    nn=cKDTree(centers).query(centers,k=2)[0][:,1]/1000 if len(centers)>1 else []
    summary['short_branch_thresholds'].append(dict(length_below_um=cutoff,count=len(selected),
        terminal_count=sum(r['terminal'] for r in selected),
        entirely_added_count=sum(bool(r['added_length_fraction']>.999999) for r in selected),
        total_length_um=sum(r['length_um'] for r in selected),
        center_nearest_neighbor_um=quantiles(nn) if len(nn) else None))
for cutoff in [5,10,20,30]:
    near=[c for c in clusters if c['baseline_geodesic_to_endpoint_um']<=cutoff]
    # Exact length of baseline edges within geodesic endpoint neighborhoods (forest).
    covered=sum(min(d['weight'],max(0,cutoff*1000-geo[a])+max(0,cutoff*1000-geo[b]))
                for a,b,d in og.edges(data=True))/1000
    summary['endpoint_neighborhoods'].append(dict(geodesic_radius_um=cutoff,
        clusters=len(near), added_length_um=sum(c['length_um'] for c in near),
        baseline_length_fraction=covered/summary['baseline_length_um']))
summary['endpoint_cluster_counts']={str(e):sum(c['nearest_endpoint_id']==e for c in clusters) for e in endpoints}
summary['limitations']='Baseline is a reference, not truth. Thresholds are descriptive, not pruning rules. Cluster distance uses baseline attachment geodesic; branch proximity uses length-weighted centers.'
write_json(OUT/'terminal-branch-analysis.json',dict(summary=summary,clusters=clusters))
print(json.dumps(summary,indent=2))
