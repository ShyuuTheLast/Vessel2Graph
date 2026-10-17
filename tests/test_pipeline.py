from copy import deepcopy
from pathlib import Path
import json
import h5py
import networkx as nx
import numpy as np
import pytest
import yaml
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import cKDTree
from vessel2graph.config import load_config
from vessel2graph.graph import branch_paths, build_graph, caliber_seed_labels
from vessel2graph.io import save_arrays
from vessel2graph.pipeline import run
from vessel2graph.segment import surface_seeds


def test_branch_edges_and_isolated_loop():
    g = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (10, 11), (11, 12), (12, 10)])
    paths, loops = branch_paths(g)
    assert len(paths) == 5
    assert [1, 3] in paths
    assert len(loops) == 1 and loops[0][0] == loops[0][-1]
    assert sum(len(p) - 1 for p in paths) == 5
    assert branch_paths(nx.path_graph(4))[0] == [[0, 1, 2, 3]]


def test_cycle_attached_to_junction():
    paths, loops = branch_paths(nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)]))
    assert not loops
    assert sorted(len(p) for p in paths) == [2, 4]


@pytest.fixture
def config(tmp_path):
    mask = np.zeros((16, 16, 16), np.uint8)
    mask[2:14, 6:10, 6:10] = 1
    mask[0, 0, 0] = 2
    with h5py.File(tmp_path / 'input.h5', 'w') as f:
        f['main'] = mask
    data = dict(vertices=np.array([[3, 8, 8], [8, 8, 8], [12, 8, 8]], float),
                edges=np.array([[0, 1], [1, 2]]), radius=np.ones(3), object_id=np.ones(3, int))
    save_arrays(tmp_path / 'source.npz', **data)
    raw = dict(input=dict(path='input.h5', voxel_size_nm=[1, 1, 1]), output=dict(directory='run'),
               skeleton=dict(source='source.npz'), segment=dict(surface_samples=8),
               resources=dict(max_memory_gib=4, reserve_memory_gib=0.01))
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_pipeline_resume_and_selection(config):
    assert set(run(config).values()) == {'complete'}
    assert set(run(config).values()) == {'cached'}
    from pathlib import Path
    output = Path(config['output']['directory'])
    report = json.loads((output / 'report.json').read_text())
    assert report['selected_voxels'] == 12 * 4 * 4
    assert report['assigned_nonzero_voxels'] == report['selected_voxels']
    assert report['nonzero_outside_selection'] == 0
    config['segment']['attribute'] = 'label'
    result = run(config)
    assert result['skeleton'] == result['graph'] == 'cached'
    assert result['segment'] == result['report'] == 'complete'
    (output / 'graph.npz').write_bytes(b'corrupted')
    assert run(config)['graph'] == 'complete'


def test_generated_skeleton(config):
    config['skeleton']['source'] = None
    assert config['skeleton']['method'] == 'thinning'
    # Odd width: thinning erases a free-standing, perfectly even-width bar.
    with h5py.File(config['input']['path'], 'r+') as f:
        f['main'][2:14, 6:11, 6:11] = 1
    assert run(config, target='graph')['graph'] == 'complete'
    report = json.loads((Path(config['output']['directory']) / 'skeleton-thinning.json').read_text())
    assert report['skeleton_voxels'] > 0 and report['loops_after'] == 0


def test_generated_kimimaro_skeleton(config):
    config['skeleton']['source'] = None
    config['skeleton']['method'] = 'kimimaro'
    config['skeleton']['teasar_params']['const'] = 1
    assert run(config, target='graph')['graph'] == 'complete'


def test_viewer_changes_do_not_invalidate_science(config, tmp_path):
    from vessel2graph.pipeline import scientific_code_digest, SCIENTIFIC_MODULES
    for name in SCIENTIFIC_MODULES:
        (tmp_path / name).write_text('science')
    initial = scientific_code_digest(tmp_path)
    for name in ('viewer.py', 'cli.py', 'config.py'):
        (tmp_path / name).write_text('changed presentation')
    assert scientific_code_digest(tmp_path) == initial
    (tmp_path / 'graph.py').write_text('changed algorithm')
    assert scientific_code_digest(tmp_path) != initial
    run(config)
    config['visualization']['layers'] = ['input_mask']
    config['visualization']['open_browser'] = False
    assert set(run(config).values()) == {'cached'}
    config['graph']['radius_clusters'] = 3
    result = run(config)
    assert result['skeleton'] == 'cached' and result['graph'] == 'complete'


def test_strict_config(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('skeleton:\n  typo: 1\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_config(path)


def test_chessboard_nearest_distance_and_last_seed(config):
    data = dict(vertices=np.array([[2, 2, 2], [6, 6, 6], [2, 2, 2]], float),
                radius=np.zeros(3), branch=np.array([1, 2, 300], np.uint32))
    coords, values = surface_seeds(data, (9, 9, 9), np.ones(3), 8, 'branch', config['resources'])
    assert set(values) == {2, 300}
    binary = np.ones((9, 9, 9), bool)
    binary[tuple(coords.astype(int).T)] = False
    expected = distance_transform_cdt(binary)
    query = np.indices(binary.shape).reshape(3, -1).T
    distance, _ = cKDTree(coords).query(query, p=np.inf)
    np.testing.assert_array_equal(distance.reshape(binary.shape), expected)


def test_multiclass_end_correction_is_local_and_snapshot_based():
    # Class 1 joins class 2 at junction 0; class 3 is elsewhere in the graph.
    # Class 2's own end is promoted at junction 6, but must not cascade to junction 0.
    paths = [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8], [6, 9, 10], [6, 11, 12]]
    g = nx.Graph()
    for p in paths:
        g.add_edges_from(zip(p, p[1:]))
    vertices = np.array([[0,0,0], [1,0,0], [3,0,0], [1,0,0],
                         [0,1,0], [0,2,0], [0,3,0], [-1,0,0], [-3,0,0],
                         [0,4,0], [0,5,0], [0,3,1], [0,3,3]], float)
    radii = np.ones(13)
    radii[[0,6]] = 2
    radii[4] = 5
    labels = np.array([1,2,1,3,1], np.uint32)
    original = np.array([0,1,1,1,2,2,0,1,1,3,3,1,1], np.uint32)
    data = dict(vertices=vertices, radius=radii)
    corrected, audit = caliber_seed_labels(g, paths, labels, data, original, True)
    assert corrected[1] == 2  # Works below the globally largest class.
    assert corrected[4] == corrected[5] == 3
    assert corrected[0] == 2  # Immediate largest-radius neighbor's ORIGINAL class.
    assert corrected[2] == corrected[3] == 1  # Stop at first outside; no reentry.
    assert original[1] == 1 and original[4] == 2
    assert audit


def test_junction_has_no_branch_seed_and_stats_do_not_change(config):
    data = dict(vertices=np.array([[5,5,5],[6,5,5],[9,5,5],[5,6,5],[5,9,5],[5,5,6],[5,5,9]], float),
                edges=np.array([[0,1],[1,2],[0,3],[3,4],[0,5],[5,6]]),
                radius=np.array([2,1,1,3,3,5,5], float), object_id=np.ones(7, int))
    cfg = dict(radius_clusters=3, random_seed=42, correct_branch_ends=True)
    corrected, branches, metrics = build_graph(data, cfg)
    cfg['correct_branch_ends'] = False
    baseline, original_branches, _ = build_graph(data, cfg)
    assert branches == original_branches
    np.testing.assert_array_equal(corrected['branch'], baseline['branch'])
    np.testing.assert_array_equal(corrected['label'], baseline['label'])
    assert corrected['branch'][0] == corrected['label'][0] == 0
    assert corrected['segmentation_label'][0] == 3
    assert corrected['segmentation_label'][1] == 3
    assert corrected['segmentation_label'][2] == corrected['label'][2] == 1
    coords, values = surface_seeds(corrected, (16,16,16), np.ones(3), 8, 'branch', config['resources'])
    assert 0 not in values
    # A zero-radius unowned junction colliding with a branch seed must not erase it.
    seeds = dict(vertices=np.array([[1,1,1],[1,1,1]], float), radius=np.zeros(2), branch=np.array([7,0]))
    _, ids = surface_seeds(seeds, (3,3,3), np.ones(3), 8, 'branch', config['resources'])
    assert ids.tolist() == [7]


def test_viewer_reads_saved_artifacts_and_checks_volume_type(config):
    ng = pytest.importorskip('neuroglancer')
    from contextlib import ExitStack
    from vessel2graph.viewer import make_viewer, SelectedMask
    run(config)
    config['visualization']['layers'] = ['input_mask', 'branch_segmentation', 'skeleton_by_branch', 'junctions', 'corrected_nodes']
    config['visualization']['hidden_layers'] = ['junctions']
    try:
        with ExitStack() as stack:
            viewer = make_viewer(config, stack)
            state = viewer.state.to_json()
            assert [x['name'] for x in state['layers']] == config['visualization']['layers']
            assert state['layers'][3]['visible'] is False
            assert state['dimensions']['z'] == [1e-9, 'm']
            with h5py.File(config['input']['path'], 'r') as f:
                mask = SelectedMask(f['main'], [1])
                assert mask[0:1, 0:1, 0:1].item() == 0
                assert mask[2:3, 6:7, 6:7].item() == 1
        config['visualization']['layers'] = ['caliber_segmentation']
        config['visualization']['hidden_layers'] = []
        with ExitStack() as stack, pytest.raises(ValueError, match='expected label segmentation'):
            make_viewer(config, stack)
    finally:
        ng.stop()


def test_viewer_config_validation_and_relative_paths(tmp_path):
    raw = dict(input=dict(path='input.h5', voxel_size_nm=[320,256,256]), output=dict(directory='run'),
               visualization=dict(layers=['caliber_segmentation'], caliber_volume='caliber.h5'))
    path = tmp_path / 'viewer.yaml'
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    assert cfg['visualization']['caliber_volume'] == str(tmp_path / 'caliber.h5')
    raw['visualization']['layers'] = ['not_a_layer']
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='supported layers'):
        load_config(path)


def test_actual_neuroglancer_mesh_from_hdf5(tmp_path):
    ng = pytest.importorskip('neuroglancer')
    from vessel2graph.viewer import VolumeArray, SelectedMask, local_volume
    raw = np.zeros((16,20,24), np.uint32)
    raw[3:10,4:12,5:15] = 11
    raw[12:14,15:17,18:20] = 2
    dimensions = ng.CoordinateSpace(names=['z','y','x'], units='nm', scales=[320,256,256])
    with h5py.File(tmp_path / 'mesh.h5', 'w') as f:
        ds = f.create_dataset('main', data=raw)
        adapter = VolumeArray(ds)
        np.testing.assert_array_equal(adapter.transpose(), raw.transpose())
        selected = SelectedMask(ds, [11])
        np.testing.assert_array_equal(selected.transpose(), (raw == 11).astype(np.uint8).transpose())
        for array, identifier in [(adapter,11), (selected,1)]:
            volume = local_volume(array, dimensions)
            mesh = volume.get_object_mesh(identifier)
            assert len(mesh) > 4 and int.from_bytes(mesh[:4], 'little') > 0
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(3) as pool:
                assert all(result == mesh for result in pool.map(volume.get_object_mesh, [identifier]*3))
            # Mesh serving must not change the 2D source values.
            raw_bytes, _ = volume.get_encoded_subvolume('raw', np.array([0,0,0]), np.array(raw.shape), '1,1,1')
            assert len(raw_bytes) == raw.size * array.dtype.itemsize


def test_radius_volume_and_image_layer(config):
    ng = pytest.importorskip('neuroglancer')
    from pathlib import Path
    from contextlib import ExitStack
    from vessel2graph.viewer import make_viewer
    config['segment']['attribute'] = 'radius'
    run(config)
    with h5py.File(Path(config['output']['directory']) / 'segmentation.h5', 'r') as f:
        ds = f['main']
        assert ds.dtype == np.dtype('float32')
        assert ds[0,0,0] == 0 and ds[8,8,8] == 1
        assert ds.attrs['attribute'] == 'radius'
    config['visualization']['layers'] = ['radius_volume']
    try:
        with ExitStack() as stack:
            viewer = make_viewer(config, stack)
            layer = viewer.state.to_json()['layers'][0]
            assert layer['type'] == 'image'
            assert 'toRaw(getDataValue())' in layer['shader']
            assert 'emitTransparent' in layer['shader']
            volume = next(x for x in viewer.volume_manager.volumes.values() if isinstance(x, ng.LocalVolume))
            assert volume.info()['downsamplingLayout'] is None
            assert volume.info()['maxDownsampling'] == 1
            with pytest.raises(ValueError, match='Invalid downsampling factor'):
                volume.get_encoded_subvolume('raw', np.array([0,0,0]), np.array([1,1,1]), '2,2,2')
    finally:
        ng.stop()


def test_viewer_interrupt_stops_busy_worker(monkeypatch, capsys):
    import subprocess
    import sys
    import time
    from vessel2graph.viewer import serve
    real_popen = subprocess.Popen
    real_sleep = time.sleep
    workers = []
    def start_busy_worker(args, **kwargs):
        # A real process with long-lived executor work, independent of NG availability.
        process = real_popen([sys.executable, '-c',
            'import sys,time; from concurrent.futures import ThreadPoolExecutor; '
            'sys.stdin.read(); pool=ThreadPoolExecutor(1); pool.submit(time.sleep,60); time.sleep(60)'], **kwargs)
        workers.append(process)
        return process
    def interrupt(delay):
        # Simulate one Ctrl+C; POSIX subprocess.wait also sleeps during cleanup.
        monkeypatch.setattr(time, 'sleep', real_sleep)
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess, 'Popen', start_busy_worker)
    monkeypatch.setattr(time, 'sleep', interrupt)
    start = time.monotonic()
    serve({})
    assert time.monotonic() - start < 4
    assert workers[0].poll() is not None
    assert 'Viewer stopped.' in capsys.readouterr().out


def test_slurm_script_and_job_memory_limit(config, tmp_path, monkeypatch, capsys):
    from vessel2graph.cli import main
    from vessel2graph.segment import check_memory
    path = tmp_path / 'config.yaml'
    assert main(['estimate', '--config', str(path), '--fill-holes', '--closing-radius-nm', '1024', '--slurm']) == 0
    script = tmp_path / 'run' / 'slurm' / 'estimate.sbatch'
    text = script.read_text()
    assert '--mem=4107M' in text and '--partition=bigmem' in text
    assert f'estimate --config {path} --fill-holes --closing-radius-nm 1024.0' in text
    assert str(script) in capsys.readouterr().out
    # The job allowance, not host-wide free memory, bounds the preflight.
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '1024')
    resources = dict(max_memory_gib=1024, reserve_memory_gib=0.25)
    with pytest.raises(MemoryError, match='reserve'):
        check_memory(resources, 2**30)
    monkeypatch.delenv('SLURM_MEM_PER_NODE')
    monkeypatch.delenv('SLURM_MEM_PER_CPU', raising=False)
    check_memory(resources, 2**20)
    raw = yaml.safe_load(path.read_text())
    raw['hpc'] = {'time': 36000}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='hpc.time'):
        load_config(path)
