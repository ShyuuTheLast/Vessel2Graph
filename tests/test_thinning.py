import networkx as nx
import numpy as np
import pytest
import yaml
from VesselEMPy.config import load_config
from VesselEMPy.thinning import thin_skeleton

# Synthetic shapes are tens of voxels across, far below any anatomical loop size.
OPTIONS = dict(closing_radius_nm=0, fill_holes=True, prune_factor=1.5, loop_factor=3.0, min_loop_perimeter_nm=0)


def tube(shape, points, radius):
    """Union of balls along straight segments between consecutive points."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), -1).astype(float)
    mask = np.zeros(shape, bool)
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        along = np.clip(((grid - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        mask |= np.linalg.norm(grid - (a + along[..., None] * (b - a)), axis=-1) <= radius
    return mask.astype(np.uint8)


def topology(data):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data['vertices'])))
    graph.add_edges_from(data['edges'].tolist())
    degree = np.array([graph.degree(n) for n in graph])
    return dict(components=nx.number_connected_components(graph), endpoints=int((degree == 1).sum()),
                junctions=int((degree > 2).sum()),
                loops=graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph))


def test_branching_tube_keeps_face_tips_and_drops_bump():
    # Trunk through both z faces, a side branch to a y face, and a surface bump that must not survive.
    mask = tube((48, 48, 32), [(0, 16, 16), (47, 16, 16)], 4) | tube((48, 48, 32), [(24, 16, 16), (24, 47, 16)], 3)
    mask |= tube((48, 48, 32), [(10, 16, 16), (10, 16, 22)], 2)
    data, report = thin_skeleton(mask, [1, 1, 1], OPTIONS)
    assert topology(data) == dict(components=1, endpoints=3, junctions=1, loops=0)
    assert report['face_tips'] == 3 and report['protected_tips'] == 0 and report['stubs_pruned'] >= 1
    assert np.all(data['radius'] > 0) and data['vertices'].shape[1] == 3


def test_real_loop_survives_and_pore_is_opened():
    ring = [(12, 8, 8), (12, 40, 8), (12, 40, 40), (12, 8, 40), (12, 8, 8)]
    data, report = thin_skeleton(tube((24, 48, 48), ring, 3), [1, 1, 1], OPTIONS)
    assert topology(data)['loops'] == 1 and report['loops_opened'] == 0
    # A thick vessel pierced by a one-voxel-wide tunnel: the ring around the tunnel is not a vessel loop.
    pierced = tube((64, 32, 32), [(0, 16, 16), (63, 16, 16)], 8)
    pierced[32, :, 16] = 0
    data, report = thin_skeleton(pierced, [1, 1, 1], OPTIONS)
    assert report['loops_before'] >= 1 and topology(data)['loops'] == 0
    assert topology(data)['components'] == 1


def test_minimum_loop_perimeter_overrides_geometry():
    # A ring with a side vessel to a face, so the ring has a junction to be judged from.
    shape = (24, 48, 48)
    mask = tube(shape, [(12, 8, 8), (12, 40, 8), (12, 40, 40), (12, 8, 40), (12, 8, 8)], 3) | tube(shape, [(12, 24, 8), (12, 24, 0)], 3)
    kept, _ = thin_skeleton(mask, [1, 1, 1], OPTIONS)
    assert topology(kept)['loops'] == 1
    # About 128 nm around at unit spacing: kept by the geometric rule, opened by a larger anatomical floor.
    data, report = thin_skeleton(mask, [1, 1, 1], dict(OPTIONS, min_loop_perimeter_nm=500))
    assert report['loops_opened'] >= 1 and topology(data)['loops'] == 0 and topology(data)['components'] == 1


def test_offset_places_crop_in_full_volume_and_anisotropy_scales():
    mask = tube((40, 24, 24), [(0, 12, 12), (39, 12, 12)], 4)
    data, report = thin_skeleton(mask, [5, 1, 1], OPTIONS, offset=(10, 0, 0), full_shape=(60, 24, 24))
    # Inside a larger volume neither end touches a face.
    assert report['protected_tips'] == 0
    assert data['vertices'][:, 0].min() >= 50 and data['vertices'][:, 0].max() <= 49 * 5
    assert data['radius'].max() <= 5


def test_thinning_config_validation(tmp_path):
    base = dict(input=dict(path='x.h5', voxel_size_nm=[1, 1, 1]), output=dict(directory='run'))
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(base))
    cfg = load_config(path)
    assert cfg['skeleton']['method'] == 'thinning' and cfg['skeleton']['thinning']['prune_factor'] == 1.5
    for bad in (dict(method='teasar'), dict(thinning=dict(prune_factor=-1)), dict(thinning=dict(fill_holes=1)),
                dict(thinning=dict(typo=1))):
        path.write_text(yaml.safe_dump(dict(base, skeleton=bad)))
        with pytest.raises(ValueError):
            load_config(path)
