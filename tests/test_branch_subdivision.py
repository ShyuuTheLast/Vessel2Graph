"""Lineage must detect splits even when every reference edge is preserved."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest
from vessel2graph.graph import branch_paths

spec = importlib.util.spec_from_file_location('subdivision', Path(__file__).resolve().parents[1]/'scripts/analyze_branch_subdivision.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def table(data):
    paths, loops = branch_paths(module.make_graph(data))
    assert not loops
    return [dict(id=i+1, nodes=p) for i, p in enumerate(paths)]


def fixture():
    # A junction with a long terminal arm, split 80% along its length.
    old = dict(vertices=np.array([[0,0,0], [8000,0,0], [10000,0,0],
        [0,1000,0], [0,-1000,0]], float), edges=np.array([[0,1],[1,2],[0,3],[0,4]]))
    new = dict(vertices=np.vstack([old['vertices'], [8000,1000,0], [8000,2000,0], [9000,1000,0]]),
        edges=np.vstack([old['edges'], [1,5], [5,6], [5,7]]))
    return old, new


def test_terminal_split_and_offshoot_junction_are_distinct():
    old, new = fixture()
    result = module.analyze(old, new, table(old), table(new))
    json.dumps(result, allow_nan=False)
    assert result['summary']['interior_splitting_junctions'] == 1
    assert result['summary']['new_junctions_off_backbone'] == 1
    assert result['summary']['backbone_descendants'] == 4
    assert result['summary']['entirely_added_branches'] == 3
    j = result['junctions'][0]
    assert j['normalized_arclength'] == pytest.approx(.8)
    assert j['terminal_distance_um'] == pytest.approx(2)
    assert sum(s['length_um'] for s in result['segments']) == pytest.approx(12)
    # Stored branch orientation cannot change terminal-facing statistics.
    reversed_table = [dict(b, nodes=b['nodes'][::-1]) for b in table(old)]
    again = module.analyze(old, new, reversed_table, table(new))
    assert again['junctions'] == result['junctions']


def test_missing_edge_rejected_even_with_all_vertices_preserved():
    old, new = fixture()
    new['edges'] = new['edges'][1:]
    with pytest.raises(ValueError, match='Baseline edge'):
        module.analyze(old, new, table(old), table(new))


def test_two_endpoint_neighborhoods_do_not_double_count_length():
    old = dict(vertices=np.array([[0,0,0], [1000,0,0], [2000,0,0]], float),
        edges=np.array([[0,1],[1,2]]))
    new = dict(vertices=np.vstack([old['vertices'], [1000,1000,0]]),
        edges=np.vstack([old['edges'], [1,3]]))
    result = module.analyze(old, new, table(old), table(new))
    neighborhood = next(r for r in result['terminal_neighborhoods'] if r['mode']=='fraction' and r['cutoff']==.5)
    assert neighborhood['exposure_um'] == 2
    assert neighborhood['junctions'] == 1
    assert neighborhood['outside_per_100um'] is None
