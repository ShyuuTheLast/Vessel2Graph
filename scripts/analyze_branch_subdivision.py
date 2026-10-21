"""Exact branch lineage from saved graphs; no skeletonization or graph mutation.

Requires unique coordinates, preserved baseline edges, and complete branch tables.
Terminal paths are oriented from a junction toward the original endpoint; paths
with two endpoints use distance to the nearer endpoint for terminal exposure.
"""
from pathlib import Path
import argparse
import csv
import json
import networkx as nx
import numpy as np
from VesselEMPy.io import digest_file, load_arrays, write_json


def edge(a, b):
    return tuple(sorted((int(a), int(b))))


def make_graph(data):
    g = nx.Graph()
    g.add_nodes_from(range(len(data['vertices'])))
    g.add_edges_from(data['edges'].tolist())
    return g


def owners(branches, graph):
    result = {}
    for branch in branches:
        for a, b in zip(branch['nodes'][:-1], branch['nodes'][1:]):
            key = edge(a, b)
            if key in result or not graph.has_edge(a, b):
                raise ValueError('Branch table repeats or invents edges')
            result[key] = branch['id']
    if set(result) != {edge(a, b) for a, b in graph.edges}:
        raise ValueError('Requires edge-complete branch tables (no isolated cycles)')
    return result


def analyze(old, new, old_branches, new_branches):
    og, ng = make_graph(old), make_graph(new)
    coordinates = [tuple(p) for p in new['vertices']]
    if len(set(coordinates)) != len(coordinates) or len(set(map(tuple, old['vertices']))) != len(old['vertices']):
        raise ValueError('Requires unique vertex coordinates')
    lookup = {p: i for i, p in enumerate(coordinates)}
    try:
        mapping = [lookup[tuple(p)] for p in old['vertices']]
    except KeyError as exc:
        raise ValueError('Baseline vertex not preserved exactly') from exc
    mapped_edges = {edge(mapping[a], mapping[b]) for a, b in og.edges}
    if not mapped_edges <= {edge(a, b) for a, b in ng.edges}:
        raise ValueError('Baseline edge not preserved')
    owners(old_branches, og)
    candidate_owner = owners(new_branches, ng)
    rows, junctions, segments = [], [], []
    for branch in old_branches:
        p = list(branch['nodes'])
        terminal_count = sum(og.degree(n) == 1 for n in (p[0], p[-1]))
        kind = {0: 'junction_to_junction', 1: 'junction_to_endpoint', 2: 'endpoint_to_endpoint'}[terminal_count]
        if terminal_count == 1 and og.degree(p[0]) == 1:
            p.reverse()
        lengths = np.linalg.norm(np.diff(old['vertices'][p], axis=0), axis=1)
        s = np.r_[0., np.cumsum(lengths)] / 1000
        total = float(s[-1])
        if total <= 0:
            raise ValueError('Zero-length reference branch')
        mapped = [mapping[n] for n in p]
        split = [i for i in range(1, len(p)-1) if ng.degree(mapped[i]) > 2]
        boundaries = [0, *split, len(p)-1]
        ids = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            child_ids = {candidate_owner[edge(mapped[i], mapped[i+1])] for i in range(a, b)}
            if len(child_ids) != 1:
                raise ValueError('Candidate ownership does not match junction subdivision')
            child = child_ids.pop()
            ids.append(child)
            segments.append(dict(baseline_branch_id=branch['id'], candidate_branch_id=child,
                start_um=float(s[a]), end_um=float(s[b]), length_um=float(s[b]-s[a])))
        for i in split:
            distance = (total-s[i]) if terminal_count == 1 else min(s[i], total-s[i]) if terminal_count == 2 else None
            junctions.append(dict(baseline_branch_id=branch['id'], kind=kind,
                baseline_node=p[i], candidate_node=mapped[i], s_um=float(s[i]),
                normalized_arclength=float(s[i]/total),
                terminal_distance_um=None if distance is None else float(distance),
                terminal_distance_fraction=None if distance is None else float(distance/total),
                degree_before=og.degree(p[i]), degree_after=ng.degree(mapped[i]),
                zyx_nm=old['vertices'][p[i]].tolist()))
        rows.append(dict(baseline_branch_id=branch['id'], kind=kind, length_um=total,
            new_interior_junctions=len(split), descendant_count=len(ids), candidate_branch_ids=ids,
            split_positions_fraction=[float(s[i]/total) for i in split],
            terminal_piece_um=float(s[-1]-s[boundaries[-2]]) if terminal_count == 1 else None))
    lineage_ids = {s['candidate_branch_id'] for s in segments}
    mixed = []
    for branch in new_branches:
        flags = [edge(a, b) in mapped_edges for a, b in zip(branch['nodes'][:-1], branch['nodes'][1:])]
        if any(flags) and not all(flags):
            mixed.append(branch['id'])
    all_new_junctions = {n for n in ng if ng.degree(n)>2} - {mapping[n] for n in og if og.degree(n)>2}
    backbone = set(mapping)
    bins = []
    for kind in ('junction_to_endpoint', 'endpoint_to_endpoint', 'junction_to_junction'):
        group = [r for r in rows if r['kind'] == kind]
        length = sum(r['length_um'] for r in group)
        for lo, hi in zip(np.arange(0., 1., .2), np.arange(.2, 1.01, .2)):
            count = int(sum(j['kind']==kind and lo <= j['normalized_arclength'] < hi for j in junctions))
            exposure = length * (hi-lo)
            bins.append(dict(kind=kind, start_fraction=float(lo), end_fraction=float(hi),
                junctions=count, exposure_um=exposure,
                junctions_per_100um=100*count/exposure if exposure else None))
    neighborhoods = []
    for mode, cutoffs in [('fraction', [.1, .2, .25, .5]), ('um', [5., 10., 20., 30.])]:
        for cutoff in cutoffs:
            terminal_rows = [r for r in rows if r['kind'] != 'junction_to_junction']
            exposure = sum(min(r['length_um'], (2 if r['kind']=='endpoint_to_endpoint' else 1)*
                (cutoff*r['length_um'] if mode=='fraction' else cutoff)) for r in terminal_rows)
            total = sum(r['length_um'] for r in terminal_rows)
            js = [j for j in junctions if j['terminal_distance_um'] is not None]
            count = sum(j['terminal_distance_'+mode] <= cutoff for j in js)
            neighborhoods.append(dict(mode=mode, cutoff=cutoff, junctions=count,
                terminal_branch_junctions=len(js), exposure_um=exposure,
                exposure_fraction=exposure/total if total else None,
                inside_per_100um=100*count/exposure if exposure else None,
                outside_per_100um=100*(len(js)-count)/(total-exposure) if total>exposure else None))
    return dict(summary=dict(baseline_branches=len(rows), candidate_branches=len(new_branches),
        split_baseline_branches=sum(r['new_interior_junctions']>0 for r in rows),
        interior_splitting_junctions=len(junctions), backbone_descendants=len(lineage_ids),
        entirely_added_branches=len(new_branches)-len(lineage_ids),
        mixed_baseline_added_branch_ids=mixed,
        new_junctions=len(all_new_junctions), new_junctions_off_backbone=len(all_new_junctions-backbone),
        baseline_endpoints_changed=[dict(baseline_node=n, candidate_node=mapping[n], degree_after=ng.degree(mapping[n]))
            for n in og if og.degree(n)==1 and ng.degree(mapping[n])!=1]),
        branches=rows, junctions=junctions, segments=segments, normalized_bins=bins,
        terminal_neighborhoods=neighborhoods,
        limitations='Exact preserved-edge lineage only. Baseline is not ground truth. Terminal means original degree-1 endpoint, not anatomical termination. Thresholds descriptive; small junction counts do not establish enrichment.')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=root/'runs/macaque')
    parser.add_argument('--candidate', type=Path, default=root/'runs/macaque-estimated-v1')
    args = parser.parse_args()
    paths = [folder/name for folder in (args.baseline, args.candidate) for name in ('graph.npz', 'branches.json')]
    hashes = {str(p): digest_file(p) for p in paths}
    frozen = json.loads((root/'docs/PHASE1_BASELINE.json').read_text())['artifacts']
    for p in paths[:2]:
        key = p.relative_to(root).as_posix() if p.is_relative_to(root) else ''
        if key in frozen and hashes[str(p)] != frozen[key]['sha256']:
            raise ValueError('Frozen baseline hash mismatch: '+key)
    result = analyze(load_arrays(paths[0]), load_arrays(paths[2]),
        json.loads(paths[1].read_text()), json.loads(paths[3].read_text()))
    result['source_sha256'] = hashes
    if hashes != {str(p): digest_file(p) for p in paths}:
        raise ValueError('Source artifacts changed during analysis')
    write_json(args.candidate/'branch-subdivision-analysis.json', result)
    for key in ('branches', 'junctions', 'segments'):
        rows = result[key]
        if rows:
            with (args.candidate/f'subdivision-{key}.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ('summary', 'branches', 'terminal_neighborhoods')}, indent=2))


if __name__ == '__main__':
    main()
