"""Local, read-only Neuroglancer layers over saved artifacts."""
from contextlib import ExitStack
from pathlib import Path
import colorsys
import json
import threading
import webbrowser
import h5py
import numpy as np
from .io import load_arrays


def local_volume(array, dimensions):
    """Avoid OpenMesh simplification and concurrent native mesh work."""
    import neuroglancer as ng

    class SerializedVolume(ng.LocalVolume):
        def get_object_mesh(self, object_id):
            with _mesh_lock:
                try:
                    return super().get_object_mesh(object_id)
                except Exception:
                    # Upstream otherwise leaves a failed constructor marked pending.
                    with self._mesh_generator_lock:
                        self._mesh_generator_pending = None
                        self._mesh_generator_lock.notify_all()
                    raise

    return SerializedVolume(array, dimensions=dimensions, volume_type='segmentation',
                            mesh_options={'max_quadrics_error': -1})


_mesh_lock = threading.Lock()


class VolumeArray:
    """Slice-lazy HDF5 adapter with explicit materialization for NG's mesh API."""
    def __init__(self, dataset, resources=None):
        self.dataset = dataset
        self.shape, self.ndim, self.dtype = dataset.shape, 3, dataset.dtype
        self.resources = resources

    def __getitem__(self, key):
        return self.dataset[key]

    def transpose(self, *axes):
        from .segment import check_memory
        if self.resources:
            check_memory(self.resources, int(np.prod(self.shape)) * self.dtype.itemsize * 2)
        # Avoid a second full-sized temporary, especially for label selection.
        array = np.empty(self.shape, dtype=self.dtype)
        for z in range(0, self.shape[0], 8):
            array[z:z+8] = self[z:z+8]
        return array.transpose(*axes)


class SelectedMask(VolumeArray):
    """Apply input selection on requested slices, never load the full mask eagerly."""
    def __init__(self, dataset, labels, resources=None):
        super().__init__(dataset, resources)
        self.dataset, self.labels = dataset, labels
        self.shape, self.ndim, self.dtype = dataset.shape, 3, np.dtype('uint8')

    def __getitem__(self, key):
        return np.isin(self.dataset[key], self.labels).astype(np.uint8)


def color(identifier):
    if identifier == 0:
        return '#aaaaaa'
    rgb = colorsys.hsv_to_rgb((int(identifier) * 0.61803398875) % 1, .72, .95)
    return '#' + ''.join(f'{round(c * 255):02x}' for c in rgb)


def make_viewer(cfg, stack):
    import neuroglancer as ng
    from neuroglancer.skeleton import SkeletonSource, Skeleton, VertexAttributeInfo
    settings = cfg['visualization']
    root = Path(cfg['output']['directory'])
    layers = settings['layers']
    spacing = cfg['input']['voxel_size_nm']
    dimensions = ng.CoordinateSpace(names=['z', 'y', 'x'], units='nm', scales=spacing)
    physical = ng.CoordinateSpace(names=['z', 'y', 'x'], units='nm', scales=[1, 1, 1])
    source = stack.enter_context(h5py.File(cfg['input']['path'], 'r'))[cfg['input']['dataset']]
    if source.ndim != 3:
        raise ValueError('Viewer input must be 3D')
    need_graph = any(n.startswith('skeleton_') or n in ('junctions', 'corrected_nodes', 'branch_segmentation', 'caliber_segmentation', 'radius_volume') for n in layers)
    data = load_arrays(root / 'graph.npz') if need_graph else None
    branches = json.loads((root / 'branches.json').read_text()) if need_graph else []
    metrics = json.loads((root / 'metrics.json').read_text()) if need_graph else {}
    by_id = {b['id']: b for b in branches}
    branch_colors = {str(i): color(i) for i in by_id}
    class_ids = sorted({b['caliber_class'] for b in branches})
    class_colors = {str(i): color(i) for i in class_ids}

    class GraphSource(SkeletonSource):
        def __init__(self, mode, graph=None, branch_records=None):
            super().__init__(physical)
            self.mode = mode
            self.data = data if graph is None else graph
            self.by_id = by_id if branch_records is None else {b['id']:b for b in branch_records}
            self.vertex_attributes['node_color'] = VertexAttributeInfo(np.float32, 3)

        def get_skeleton(self, object_id):
            data,by_id=self.data,self.by_id
            if object_id not in by_id:
                return None
            branch = by_id[object_id]
            nodes = branch['nodes']
            if self.mode == 'skeleton_by_radius':
                r = data['radius'][nodes] / max(float(data['radius'].max()), 1)
                colors = np.column_stack((r, .2 + .6 * (1-r), 1-r))
            else:
                ids = (data['segmentation_label'][nodes] if self.mode == 'skeleton_by_corrected_caliber'
                       else np.full(len(nodes), branch['caliber_class'] if self.mode == 'skeleton_by_caliber' else object_id))
                colors = np.array([[int(color(i)[j:j+2], 16)/255 for j in (1,3,5)] for i in ids])
            edges = np.column_stack((np.arange(len(nodes)-1), np.arange(1,len(nodes))))
            return Skeleton(data['vertices'][nodes], edges, {'node_color': colors.astype(np.float32)})

    built = []
    for name in layers:
        if name == 'input_mask':
            layer = ng.SegmentationLayer(source=local_volume(SelectedMask(source, cfg['input']['labels'], cfg['resources']),
                dimensions), segments=[1], segment_colors={'1':'#b0b0b0'}, selected_alpha=.5)
        elif name in ('branch_segmentation', 'caliber_segmentation'):
            attr = 'branch' if name == 'branch_segmentation' else 'label'
            field = 'branch_volume' if attr == 'branch' else 'caliber_volume'
            path = settings[field] or str(root / 'segmentation.h5')
            if not Path(path).is_file():
                raise ValueError(f'{name}: missing {path}. Generate it separately and set visualization.{field}.')
            ds = stack.enter_context(h5py.File(path, 'r'))['main']
            if ds.shape != source.shape or ds.dtype.kind not in 'iu' or ds.attrs.get('attribute') != attr:
                raise ValueError(f'{name}: expected {attr} segmentation with input shape in {path}')
            if not np.array_equal(ds.attrs.get('voxel_size_nm'), spacing) or ds.attrs.get('axes') != 'ZYX':
                raise ValueError(f'{name}: axes/spacing mismatch in {path}')
            layer = ng.SegmentationLayer(source=local_volume(VolumeArray(ds, cfg['resources']), dimensions),
                segments=list(by_id) if attr == 'branch' else class_ids,
                segment_colors=branch_colors if attr == 'branch' else class_colors, selected_alpha=.5)
        elif name == 'radius_volume':
            path = settings['radius_volume'] or str(root / 'segmentation.h5')
            ds = stack.enter_context(h5py.File(path, 'r'))['main']
            if ds.shape != source.shape or ds.dtype.kind != 'f' or ds.attrs.get('attribute') != 'radius':
                raise ValueError(f'radius_volume: expected float radius volume with input shape in {path}')
            if not np.array_equal(ds.attrs.get('voxel_size_nm'), spacing) or ds.attrs.get('axes') != 'ZYX':
                raise ValueError(f'radius_volume: axes/spacing mismatch in {path}')
            maximum = max(float(data['radius'].max()), 1)
            shader = f'''#uicontrol float maxRadius slider(min=1, max={maximum * 2:.6f}, default={maximum:.6f})
void main() {{
  float radius = toRaw(getDataValue());
  if (radius <= 0.0) {{ emitTransparent(); return; }}
  float r = clamp(radius / maxRadius, 0.0, 1.0);
  emitRGB(vec3(r, 0.2 + 0.6 * (1.0-r), 1.0-r));
}}'''
            layer = ng.ImageLayer(source=ng.LocalVolume(VolumeArray(ds), dimensions=dimensions,
                                  volume_type='image', downsampling=None, max_downsampling=1), shader=shader)
        elif name.startswith('skeleton_'):
            layer = ng.SegmentationLayer(source=GraphSource(name), segments=list(by_id),
                skeleton_shader='void main() { emitRGB(node_color); }')
            layer.skeleton_rendering.mode2d = 'lines_and_points'
            layer.skeleton_rendering.mode3d = 'lines_and_points'
            layer.skeleton_rendering.line_width2d = 2
        else:
            records = metrics.get('branch_end_corrections', []) if name == 'corrected_nodes' else metrics.get('junction_caliber_labels', [])
            annotations = []
            for record in records:
                node = record['node']
                description = (f"node {node}: class {record['original_class']} -> {record['segmentation_class']}"
                               if name == 'corrected_nodes' else f"junction {node}: no branch ID; caliber {record['segmentation_class']}")
                annotations.append(ng.PointAnnotation(id=str(node), point=data['vertices'][node], description=description))
            layer = ng.LocalAnnotationLayer(dimensions=physical, annotations=annotations,
                annotation_color='#ff00ff' if name == 'corrected_nodes' else '#ffffff')
        built.append((name, layer))
    class OverlaySource(SkeletonSource):
        def __init__(self,path):
            super().__init__(physical)
            self.graph=load_arrays(path)

        def get_skeleton(self,object_id):
            if object_id!=1:
                return None
            return Skeleton(self.graph['vertices'],self.graph['edges'])

    overlay_layers=[]
    for overlay in settings.get('skeleton_overlays',[]):
        if overlay.get('mode')=='branch':
            graph_path=Path(overlay['path'])
            records=json.loads(graph_path.with_name('branches.json').read_text())
            branch_source=GraphSource('skeleton_by_branch',load_arrays(graph_path),records)
            layer=ng.SegmentationLayer(source=branch_source,segments=list(branch_source.by_id),
                segment_colors={str(i):color(i) for i in branch_source.by_id},
                skeleton_shader='void main() { emitRGB(node_color); }')
        else:
            rgb=[int(overlay['color'][j:j+2],16)/255 for j in (1,3,5)]
            shader='void main() { emitRGB(vec3('+','.join(f'{v:.8f}' for v in rgb)+')); }'
            layer=ng.SegmentationLayer(source=OverlaySource(overlay['path']),segments=[1],
                segment_colors={'1':overlay['color']},skeleton_shader=shader)
        layer.skeleton_rendering.mode2d='lines_and_points'
        layer.skeleton_rendering.mode3d='lines_and_points'
        layer.skeleton_rendering.line_width2d=2
        overlay_layers.append((overlay,layer))
    viewer = ng.Viewer()
    with viewer.txn() as state:
        state.dimensions = dimensions
        state.display_dimensions = ['x', 'y', 'z']
        state.position = ((data['vertices'].min(axis=0) + data['vertices'].max(axis=0)) / (2*np.asarray(spacing))
                          if data is not None else np.asarray(source.shape)/2)
        state.layout = '4panel'
        state.cross_section_scale = 2
        state.projection_scale = max(source.shape) * 1.4
        for name, layer in built:
            state.layers.append(name=name, layer=layer, visible=name not in settings['hidden_layers'])
        for overlay,layer in overlay_layers:
            state.layers.append(name=overlay['name'],layer=layer,visible=overlay.get('visible',True))
    return viewer


def _stop_worker(process):
    """Only terminate the dedicated read-only viewer process, never the pipeline."""
    import subprocess
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def serve(cfg):
    # Keep native meshing and executor shutdown off the terminal's main process.
    # A finite wait also avoids Windows' uninterruptible indefinite Event.wait().
    import os
    import signal
    import subprocess
    import sys
    import time
    previous = None
    if os.name == 'nt':
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGBREAK, interrupt)
    process = None
    try:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'VesselEMPy.viewer'],
            stdin=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            start_new_session=os.name != 'nt')
        process.stdin.write(json.dumps(cfg))
        process.stdin.close()
        while process.poll() is None:
            time.sleep(.1)
        if process.returncode:
            raise OSError(f'Viewer worker exited with code {process.returncode}')
    except KeyboardInterrupt:
        if process is not None:
            _stop_worker(process)
        print('\nViewer stopped.', flush=True)
    finally:
        if process is not None:
            _stop_worker(process)
        if previous is not None:
            signal.signal(signal.SIGBREAK, previous)


def _serve_worker(cfg):
    import faulthandler
    faulthandler.enable()
    try:
        import neuroglancer as ng
    except ImportError as exc:
        raise ImportError('Install the viewer extra: python -m pip install -e ".[view]" -c requirements-core.txt') from exc
    ng.set_server_bind_address('127.0.0.1', cfg['visualization']['port'])
    with ExitStack() as stack:
        try:
            viewer = make_viewer(cfg, stack)
            print(f'Neuroglancer: {viewer}\nKeep this terminal running. Ctrl+C stops the viewer.', flush=True)
            if cfg['visualization']['open_browser']:
                webbrowser.open(str(viewer))
            threading.Event().wait()
        except KeyboardInterrupt:
            print('\nViewer stopped.', flush=True)
        finally:
            ng.stop()


if __name__ == '__main__':
    import sys
    _serve_worker(json.load(sys.stdin))
