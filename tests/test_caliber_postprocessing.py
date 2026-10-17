import networkx as nx
import numpy as np
import pytest
import yaml
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from vessel2graph.config import load_config
from vessel2graph.graph import caliber_seed_labels, ellipsoid_seed_nodes, sandwich_labels
from vessel2graph.segment import remove_dust, surface_seeds

RESOURCES = dict(max_memory_gib=4, reserve_memory_gib=0.01)


def graph_of(paths):
    g = nx.Graph()
    for p in paths:
        g.add_edges_from(zip(p, p[1:]))
    return g


def test_sandwich_is_snapshot_and_needs_two_junction_ends():
    # Large 0-1-2 | small bridge 2..5 | large 5-6-7, plus a dead-end small branch at 2 and a
    # small branch 5..9..10 whose far end touches only small branches.
    paths = [[0, 1, 2], [2, 3, 4, 5], [5, 6, 7], [2, 8], [5, 9, 10], [10, 11], [10, 12], [0, 13], [0, 14],
             [7, 15], [7, 16]]
    labels = np.array([2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2], np.uint32)
    result, changes = sandwich_labels(graph_of(paths), paths, labels)
    assert result[1] == 2  # both ends touch the large class
    assert result[3] == 1  # dead end
    assert result[4] == 1  # far end touches only small branches; no cascade from branch 1
    assert [c['branch'] for c in changes] == [2]
    assert labels[1] == 1  # input untouched


def test_sandwich_takes_smaller_of_the_two_larger_classes():
    paths = [[0, 1, 2], [0, 3], [0, 4], [2, 5], [2, 6]]
    labels = np.array([1, 3, 3, 2, 2], np.uint32)
    result, _ = sandwich_labels(graph_of(paths), paths, labels)
    assert result[0] == 2


def test_junction_takes_largest_class_not_largest_radius_neighbour():
    paths = [[0, 1, 2], [0, 3, 4], [0, 5, 6]]
    g = graph_of(paths)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0], [0, 1, 0], [0, 5, 0], [0, 0, 1], [0, 0, 5]], float)
    radius = np.ones(7)
    radius[1] = 9  # the class-1 branch has the thickest neighbour of the junction
    labels = np.array([1, 2, 1], np.uint32)
    original = np.array([0, 1, 1, 2, 2, 1, 1], np.uint32)
    result, _ = caliber_seed_labels(g, paths, labels, dict(vertices=vertices, radius=radius), original, False)
    assert result[0] == 2


def test_ellipsoids_only_on_larger_class_near_mixed_junctions():
    # Junction 0: large branch along x, small side branch along y. Junction 20: all small.
    paths = [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7], [3, 20], [20, 21], [20, 22]]
    vertices = np.zeros((23, 3))
    vertices[[1, 2, 3]] = [[1, 0, 0], [2, 0, 0], [6, 0, 0]]
    vertices[[4, 5, 6]] = [[-1, 0, 0], [-4, 0, 0], [-8, 0, 0]]
    vertices[[7, 20, 21, 22]] = [[0, 3, 0], [9, 0, 0], [9, 2, 0], [9, -2, 0]]
    radius = np.ones(23)
    radius[0] = 2.0
    labels = np.array([2, 2, 1, 1, 1, 1], np.uint32)
    flag = ellipsoid_seed_nodes(graph_of(paths), paths, labels, dict(vertices=vertices, radius=radius), 1.5)
    assert set(np.flatnonzero(flag)) == {0, 1, 2, 4}  # within 3 nm of junction 0, large branches only
    assert not ellipsoid_seed_nodes(graph_of(paths), paths, labels, dict(vertices=vertices, radius=radius), 0)[[1, 2, 4]].any()


def test_points_and_ellipsoids_with_ellipsoid_winning_collisions():
    data = dict(vertices=np.array([[5, 5, 5], [5, 5, 8], [5, 5, 2]], float), radius=np.array([3.0, 0, 0]),
                segmentation_label=np.array([2, 1, 1], np.uint32), ellipsoid_seed=np.array([True, False, False]))
    coords, values = surface_seeds(data, (11, 11, 11), np.ones(3), 40, 'label', RESOURCES)
    at = {tuple(c): v for c, v in zip(coords.astype(int), values)}
    assert at[(5, 5, 8)] == 2 and at[(5, 5, 2)] == 2  # node 1 and 2 lie on node 0's surface
    assert (5, 5, 5) not in at  # an ellipsoid node seeds its surface, not its centre
    data['ellipsoid_seed'][:] = False
    coords, values = surface_seeds(data, (11, 11, 11), np.ones(3), 40, 'label', RESOURCES)
    assert sorted(map(tuple, coords.astype(int))) == [(5, 5, 2), (5, 5, 5), (5, 5, 8)]


def test_tree_matches_euclidean_distance_transform_with_anisotropy():
    rng = np.random.default_rng(0)
    shape, spacing = (12, 20, 20), np.array([5.0, 1.0, 1.0])
    seeds = rng.integers(0, shape, size=(25, 3))
    binary = np.ones(shape, bool)
    binary[tuple(seeds.T)] = False
    expected = distance_transform_edt(binary, sampling=spacing)
    query = np.indices(shape).reshape(3, -1).T
    distance, _ = cKDTree(seeds * spacing).query(query * spacing, workers=2)
    np.testing.assert_allclose(distance.reshape(shape), expected)


def test_dust_is_top_down_and_relative_to_each_class():
    volume = np.zeros((10, 40, 40), np.uint8)
    volume[:, 0:20, 0:20] = 3      # class 3: one big piece ...
    volume[0, 30, 30] = 3          # ... isolated dust: demoted to 2, still dust there, so 1
    volume[:, 25:40, 0:10] = 2     # class 2: one big piece ...
    volume[0, 30, 10] = 3          # ... with class-3 dust touching it: demoted to 2 and joins it
    volume[5, 30:32, 20:22] = 2    # ... and class-2 dust, which becomes class 1
    report = remove_dust(volume, 0.01, RESOURCES)
    assert volume[0, 30, 30] == 1
    assert volume[0, 30, 10] == 2
    assert (volume[5, 30:32, 20:22] == 1).all()
    assert (volume[:, 0:20, 0:20] == 3).all() and (volume[:, 25:40, 0:10] == 2).all()
    assert report['3']['relabelled_pieces'] == 2 and report['2']['relabelled_pieces'] == 2


def test_new_settings_are_validated(tmp_path):
    def load(**sections):
        raw = dict(input=dict(path='in.h5', voxel_size_nm=[1, 1, 1]), output=dict(directory='run'), **sections)
        path = tmp_path / 'c.yaml'
        path.write_text(yaml.safe_dump(raw))
        return load_config(path)
    cfg = load()
    assert cfg['graph']['sandwich'] is True and cfg['graph']['ellipsoid_reach'] == 1.5
    assert cfg['segment']['method'] == 'surface_euclidean' and cfg['segment']['dust_ratio'] == 0.01
    for bad in (dict(graph=dict(sandwich=1)), dict(graph=dict(ellipsoid_reach=-1)),
                dict(segment=dict(dust_ratio=1)), dict(segment=dict(workers=0))):
        with pytest.raises(ValueError):
            load(**bad)
