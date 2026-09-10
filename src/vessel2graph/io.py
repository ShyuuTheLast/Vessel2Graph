"""HDF5 and versioned array artifacts, without GUI dependencies."""
from pathlib import Path
import hashlib
import json
import os
import h5py
import numpy as np


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def inspect_input(cfg):
    with h5py.File(cfg['path'], 'r') as f:
        ds = f[cfg['dataset']]
        if ds.ndim != 3 or ds.dtype.kind not in 'biu':
            raise ValueError('Input must be a 3D integer or Boolean segmentation')
        return {'shape': list(ds.shape), 'dtype': str(ds.dtype), 'nbytes': int(ds.size * ds.dtype.itemsize)}


def load_mask(cfg):
    meta = inspect_input(cfg)
    mask = np.empty(meta['shape'], dtype=np.uint8)
    with h5py.File(cfg['path'], 'r') as f:
        ds = f[cfg['dataset']]
        for z in range(0, ds.shape[0], 8):
            mask[z:z+8] = np.isin(ds[z:z+8], cfg['labels'])
    if not mask.any():
        raise ValueError('Selected labels produce an empty mask')
    return mask


def save_arrays(path, **arrays):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, schema_version=np.int32(1), **arrays)
    os.replace(temp, path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        if int(data['schema_version']) != 1:
            raise ValueError('Unsupported artifact version')
        return {k: data[k] for k in data.files if k != 'schema_version'}


def skeleton_arrays(skels):
    vertices, edges, radius, object_ids = [], [], [], []
    offset = 0
    for key, skel in sorted(skels.items()):
        v = np.asarray(skel.vertices, dtype=np.float64)
        vertices.append(v)
        edges.append(np.asarray(skel.edges, dtype=np.int64).reshape(-1,2) + offset)
        radius.append(np.asarray(skel.radius, dtype=np.float64))
        object_ids.append(np.full(len(v), int(key), dtype=np.int64))
        offset += len(v)
    if not offset:
        raise ValueError('Skeletonization produced no vertices')
    return dict(vertices=np.concatenate(vertices), edges=np.concatenate(edges),
                radius=np.concatenate(radius), object_id=np.concatenate(object_ids))


def validate_skeleton(data):
    v, e, r = data['vertices'], data['edges'], data['radius']
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or r.shape != (len(v),):
        raise ValueError('Invalid skeleton array shapes')
    if not np.isfinite(v).all() or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError('Skeleton coordinates/radii must be finite with nonnegative radii')
    if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in 'iu':
        raise ValueError('Invalid edge array')
    if len(e) and ((e < 0).any() or (e >= len(v)).any()):
        raise ValueError('Edge endpoint out of range')
    if 'object_id' not in data or data['object_id'].shape != (len(v),):
        raise ValueError('object_id must identify each vertex')
    return data
