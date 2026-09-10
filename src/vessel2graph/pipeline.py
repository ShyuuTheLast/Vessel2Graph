"""Config-driven stages with content-checked caching and provenance."""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import platform
import time
import h5py
import numpy as np
import yaml
from .io import digest_file, inspect_input, load_mask, load_arrays, save_arrays, skeleton_arrays, validate_skeleton, write_json
from .graph import build_graph
from .segment import segment, check_memory

STAGES = ('skeleton', 'graph', 'segment', 'report')
ARTIFACTS = {'skeleton': ['skeleton.npz'], 'graph': ['graph.npz', 'branches.json', 'metrics.json'],
             'segment': ['segmentation.h5', 'segmentation.json'], 'report': ['report.json']}


SCIENTIFIC_MODULES = ('pipeline.py', 'io.py', 'graph.py', 'segment.py')


def scientific_code_digest(root=None):
    # Effective configuration values are fingerprinted separately. Viewer/CLI/config
    # presentation changes must not invalidate scientific artifacts.
    root = Path(root) if root is not None else Path(__file__).parent
    h = hashlib.sha256()
    for name in SCIENTIFIC_MODULES:
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()


def provenance():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {p: version(p) for p in ('numpy', 'scipy', 'h5py', 'networkx', 'scikit-learn', 'kimimaro', 'cloud-volume')},
            'scientific_modules': list(SCIENTIFIC_MODULES),
            'code_sha256': scientific_code_digest()}


def run(cfg, target='report', force=False, status=False):
    output = Path(cfg['output']['directory'])
    meta = inspect_input(cfg['input'])
    prov = provenance()
    source = cfg['skeleton']['source']
    signature = {'input': cfg['input'], 'input_sha256': digest_file(cfg['input']['path']),
                 'source_sha256': digest_file(source) if source else None, 'implementation': prov}
    # Prevent accidental overwriting of input artifacts by output files.
    protected = {Path(cfg['input']['path']).resolve()}
    if source:
        protected.add(Path(source).resolve())
    if any((output / name).resolve() in protected for names in ARTIFACTS.values() for name in names):
        raise ValueError('Output directory would overwrite an input artifact')
    if not status:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'resolved-config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        write_json(output / 'provenance.json', prov)
    results = {}
    for stage in STAGES[:STAGES.index(target) + 1]:
        signature = {'upstream': signature, 'settings': cfg.get(stage, {}), 'stage': stage}
        fingerprint = hashlib.sha256(yaml.safe_dump(signature, sort_keys=True).encode()).hexdigest()
        marker = output / (stage + '.stage.json')
        old = json.loads(marker.read_text()) if marker.exists() else {}
        valid = old.get('fingerprint') == fingerprint and old.get('status') == 'complete'
        valid = valid and all((output / name).exists() and digest_file(output / name) == old.get('artifacts', {}).get(name)
                              for name in ARTIFACTS[stage])
        if status:
            results[stage] = 'cached' if valid else 'missing or stale'
        elif valid and not force:
            print(f'{stage}: cached', flush=True)
            results[stage] = 'cached'
        else:
            print(f'{stage}: running', flush=True)
            start = time.perf_counter()
            write_json(marker, dict(status='running', fingerprint=fingerprint))
            try:
                execute(stage, cfg, output, meta)
                write_json(marker, dict(status='complete', fingerprint=fingerprint,
                    elapsed_seconds=time.perf_counter() - start,
                    artifacts={name: digest_file(output / name) for name in ARTIFACTS[stage]}))
            except Exception as exc:
                write_json(marker, dict(status='failed', fingerprint=fingerprint, error=str(exc)))
                raise
            print(f'{stage}: complete ({time.perf_counter() - start:.2f}s)', flush=True)
            results[stage] = 'complete'
        # Incorporate actual upstream bytes, including manually edited artifacts.
        signature = {'fingerprint': fingerprint,
                     'artifacts': {n: digest_file(output / n) if (output / n).exists() else None for n in ARTIFACTS[stage]}}
    return results


def crop_empty_background(mask):
    """Keep a zero halo and original touched faces; return voxel-coordinate offset."""
    bounds=[]
    for axis in range(3):
        occupied=np.flatnonzero(np.any(mask,axis=tuple(i for i in range(3) if i!=axis)))
        if not len(occupied):
            raise ValueError('Cannot crop an empty mask')
        bounds.append((max(0,int(occupied[0])-1),min(mask.shape[axis],int(occupied[-1])+2)))
    offset=np.array([a for a,b in bounds])
    return mask[tuple(slice(a,b) for a,b in bounds)].copy(),offset


def execute(stage, cfg, output, meta):
    if stage == 'skeleton':
        options = cfg['skeleton']
        if options['source']:
            if options['format'] == 'legacy_npz':
                # Explicit opt-in: only use locally trusted legacy pickle archives.
                with np.load(options['source'], allow_pickle=True) as archive:
                    data = skeleton_arrays(archive['skeletons'].item())
            else:
                data = load_arrays(options['source'])
        else:
            import kimimaro
            crop=options.get('crop_empty_border',False)
            check_memory(cfg['resources'], int(np.prod(meta['shape'])) * (2 if crop else 24))
            mask = load_mask(cfg['input'])
            offset=np.zeros(3,dtype=int)
            if crop:
                mask,offset=crop_empty_background(mask)
                write_json(output/'skeleton-crop.json',dict(original_shape=list(meta['shape']),
                    cropped_shape=list(mask.shape),offset_zyx_voxels=offset.tolist(),background_halo=1))
                check_memory(cfg['resources'],int(mask.size)*24)
            skels = kimimaro.skeletonize(mask, teasar_params=options['teasar_params'],
                anisotropy=cfg['input']['voxel_size_nm'], dust_threshold=options['dust_threshold'],
                parallel=options['parallel'], fix_branching=options['fix_branching'],
                fix_borders=options['fix_borders'], progress=False)
            data = skeleton_arrays(skels)
            if crop:
                data['vertices']+=offset*np.asarray(cfg['input']['voxel_size_nm'])
        save_arrays(output / 'skeleton.npz', **validate_skeleton(data))
    elif stage == 'graph':
        data, branches, metrics = build_graph(load_arrays(output / 'skeleton.npz'), cfg['graph'])
        save_arrays(output / 'graph.npz', **data)
        write_json(output / 'branches.json', branches)
        write_json(output / 'metrics.json', metrics)
    elif stage == 'segment':
        data = load_arrays(output / 'graph.npz')
        metrics = json.loads((output / 'metrics.json').read_text())
        if metrics['isolated_loops'] or metrics['isolated_vertices']:
            raise ValueError('Isolated loops/vertices have no branch definition. Inspect metrics.json; dense assignment is paused.')
        result = segment(data, cfg, output / 'segmentation.h5')
        write_json(output / 'segmentation.json', result)
    else:
        selected = assigned = outside = 0
        with h5py.File(cfg['input']['path'], 'r') as f, h5py.File(output / 'segmentation.h5', 'r') as g:
            ds, seg = f[cfg['input']['dataset']], g['main']
            for z in range(0, ds.shape[0], 8):
                mask = np.isin(ds[z:z+8], cfg['input']['labels'])
                nonzero = seg[z:z+8] != 0
                selected += int(mask.sum())
                assigned += int((mask & nonzero).sum())
                outside += int((~mask & nonzero).sum())
        write_json(output / 'report.json', dict(input=meta, selected_voxels=selected,
            assigned_nonzero_voxels=assigned, nonzero_outside_selection=outside,
            graph=json.loads((output / 'metrics.json').read_text()),
            segmentation=json.loads((output / 'segmentation.json').read_text())))
