"""Sparse seeds with bounded foreground queries; no volume-sized index map."""
from pathlib import Path
import os
import cc3d
import h5py
import numpy as np
import psutil
from scipy.spatial import cKDTree
from .hpc import job_memory_limit


def check_memory(resources, additional=0):
    rss = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    # Inside a Slurm job the allowance is the job's, not the host's free memory.
    limit = job_memory_limit()
    if limit is not None:
        available = min(available, limit - rss)
    if rss + additional > resources['max_memory_gib'] * 1024**3:
        raise MemoryError('Estimated process memory exceeds resources.max_memory_gib')
    if available - additional < resources['reserve_memory_gib'] * 1024**3:
        raise MemoryError('Insufficient available memory after configured reserve')


def surface_seeds(data, shape, spacing, samples, attribute, resources):
    """Every node is a point seed; nodes flagged ellipsoid_seed (all nodes when the flag is
    absent) are seeded on their ellipsoid surface instead. Later nodes win at collisions."""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, samples), np.linspace(0, 2 * np.pi, samples))
    unit = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1).reshape(-1, 3)
    attributes = data['segmentation_label'] if attribute == 'label' else data[attribute]
    eligible = np.ones(len(data['vertices']), dtype=bool)
    if attribute in ('branch', 'dist_from_largest'):
        eligible = data['branch'] != 0  # An unowned junction must not erase existing seeds at collisions.
    ellipsoid = data['ellipsoid_seed'] if 'ellipsoid_seed' in data else np.ones(len(eligible), dtype=bool)
    base = (data['vertices'] / spacing).astype(np.int64)
    indices, owners = [], []
    points = np.flatnonzero(eligible & ~ellipsoid)
    for nodes, offsets in ((points, np.zeros((1, 3))), (np.flatnonzero(eligible & ellipsoid), unit)):
        for start in range(0, len(nodes), 2000):
            batch = nodes[start:start + 2000]
            coords = (base[batch, None, :] + offsets[None] * (data['radius'][batch, None, None] / spacing)).astype(np.int64)
            owner = np.repeat(batch, len(offsets))
            coords = coords.reshape(-1, 3)
            inside = ((coords >= 0) & (coords < shape)).all(axis=1)
            indices.append(np.ravel_multi_index(coords[inside].T, shape))
            owners.append(owner[inside])
            check_memory(resources, sum(a.nbytes for a in indices) * 3)
    flat, owner = np.concatenate(indices), np.concatenate(owners)
    if not len(flat):
        raise ValueError('Skeleton has no surface seeds within the input volume')
    # Points first, then ellipsoids, each in node order: keep the last at collisions.
    order = np.lexsort((owner, ellipsoid[owner]))
    flat, owner = flat[order], owner[order]
    unique, reverse = np.unique(flat[::-1], return_index=True)
    owner = owner[len(owner) - 1 - reverse]
    coords = np.column_stack(np.unravel_index(unique, shape)).astype(np.float64)
    return coords, attributes[owner]


def remove_dust(volume, ratio, resources):
    """Top-down over classes: 26-connected pieces of class k smaller than ratio x its largest
    piece are relabelled k - 1 (legacy remove_small_components_cc3d, for every class)."""
    report = {}
    for k in range(int(volume.max()), 1, -1):
        check_memory(resources, volume.size * 5)
        pieces = cc3d.connected_components(volume == k, connectivity=26)
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        small = np.flatnonzero((sizes > 0) & (sizes < ratio * sizes.max()))
        moved = np.isin(pieces, small)
        volume[moved] = k - 1
        report[str(k)] = dict(pieces=int((sizes > 0).sum()), relabelled_pieces=len(small),
                              relabelled_voxels=int(moved.sum()), threshold_voxels=float(ratio * sizes.max()))
    return report


def segment(data, cfg, destination):
    settings, resources = cfg['segment'], cfg['resources']
    spacing = np.asarray(cfg['input']['voxel_size_nm'])
    workers = settings['workers'] or (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
    with h5py.File(cfg['input']['path'], 'r') as source:
        ds = source[cfg['input']['dataset']]
        shape = ds.shape
        seeds, values = surface_seeds(data, shape, spacing, settings['surface_samples'], settings['attribute'], resources)
        check_memory(resources, seeds.nbytes * 3)
        physical = settings['method'] == 'surface_euclidean'
        tree = cKDTree(seeds * spacing if physical else seeds)
        path = Path(destination)
        temporary = path.with_suffix('.tmp.h5')
        dtype = np.float32 if settings['attribute'] == 'radius' else np.uint32
        count, distribution = 0, {}
        with h5py.File(temporary, 'w') as out:
            result = out.create_dataset('main', shape=shape, dtype=dtype, compression=settings['compression'], chunks=True)
            result.attrs['voxel_size_nm'] = spacing
            result.attrs['axes'] = 'ZYX'
            result.attrs['attribute'] = settings['attribute']
            result.attrs['method'] = settings['method']
            for z in range(0, shape[0], settings['read_slices']):
                check_memory(resources)
                mask = np.isin(ds[z:z + settings['read_slices']], cfg['input']['labels'])
                slab = np.zeros(mask.shape, dtype=dtype)
                coords = np.column_stack(np.nonzero(mask))
                count += len(coords)
                for start in range(0, len(coords), settings['query_batch_size']):
                    local = coords[start:start + settings['query_batch_size']]
                    query = local + [z, 0, 0]
                    _, nearest = tree.query(query * spacing if physical else query, p=2 if physical else np.inf, workers=workers)
                    slab[tuple(local.T)] = values[nearest]
                result[z:z + len(slab)] = slab
            dust = {}
            if settings['attribute'] == 'label' and settings['dust_ratio']:
                small_type = np.uint8 if values.max() < 256 else np.uint32
                check_memory(resources, int(np.prod(shape)) * (np.dtype(small_type).itemsize + 5))
                volume = np.empty(shape, small_type)
                for z in range(0, shape[0], settings['read_slices']):
                    volume[z:z + settings['read_slices']] = result[z:z + settings['read_slices']]
                dust = remove_dust(volume, settings['dust_ratio'], resources)
                result[:] = volume
                del volume
            if settings['attribute'] != 'radius':
                for z in range(0, shape[0], settings['read_slices']):
                    ids, counts = np.unique(result[z:z + settings['read_slices']], return_counts=True)
                    for key, n in zip(ids, counts):
                        if key:
                            distribution[str(int(key))] = distribution.get(str(int(key)), 0) + int(n)
        if not count:
            raise ValueError('Selected labels produce an empty mask')
        os.replace(temporary, path)
    return dict(selected_voxels=count, surface_seeds=len(seeds), voxel_counts=distribution,
                ellipsoid_seed_nodes=int(data['ellipsoid_seed'].sum()) if 'ellipsoid_seed' in data else None,
                dust_ratio=settings['dust_ratio'] if settings['attribute'] == 'label' else None, dust=dust,
                method=settings['method'], attribute=settings['attribute'], dtype=np.dtype(dtype).name,
                workers=workers, tie_policy='cKDTree nearest seed; equal-distance ties may differ from a dense distance transform')
