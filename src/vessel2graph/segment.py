"""Sparse surface seeds with bounded foreground queries; no volume-sized index map."""
from pathlib import Path
import os
import h5py
import numpy as np
import psutil
from scipy.spatial import cKDTree


def check_memory(resources, additional=0):
    rss = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    if rss + additional > resources['max_memory_gib'] * 1024**3:
        raise MemoryError('Estimated process memory exceeds resources.max_memory_gib')
    if available - additional < resources['reserve_memory_gib'] * 1024**3:
        raise MemoryError('Insufficient available memory after configured reserve')


def surface_seeds(data, shape, spacing, samples, attribute, resources):
    theta, phi = np.meshgrid(np.linspace(0, np.pi, samples), np.linspace(0, 2 * np.pi, samples))
    unit = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1).reshape(-1, 3)
    indices, owners = [], []
    attributes = data['segmentation_label'] if attribute == 'label' else data[attribute]
    for i, (vertex, radius) in enumerate(zip(data['vertices'], data['radius'])):
        if attribute in ('branch', 'dist_from_largest') and data['branch'][i] == 0:
            continue  # An unowned junction must not erase existing seeds at collisions.
        coords = (unit * radius / spacing + (vertex / spacing).astype(np.int64)).astype(np.int64)
        coords = coords[((coords >= 0) & (coords < shape)).all(axis=1)]
        flat = np.unique(np.ravel_multi_index(coords.T, shape))
        indices.append(flat)
        owners.append(np.full(len(flat), i, dtype=np.int64))
        if i % 100 == 0:
            check_memory(resources, sum(a.nbytes for a in indices) * 5)
    if not indices:
        raise ValueError('No eligible segmentation seed nodes')
    flat, owner = np.concatenate(indices), np.concatenate(owners)
    # Keep last node at collisions, as in the original category volume.
    unique, reverse = np.unique(flat[::-1], return_index=True)
    owner = owner[len(owner) - 1 - reverse]
    values = attributes[owner]
    if not len(unique):
        raise ValueError('Skeleton has no surface seeds within the input volume')
    coords = np.column_stack(np.unravel_index(unique, shape)).astype(np.float64)
    return coords, values


def segment(data, cfg, destination):
    settings, resources = cfg['segment'], cfg['resources']
    spacing = np.asarray(cfg['input']['voxel_size_nm'])
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
                    _, nearest = tree.query(query * spacing if physical else query, p=2 if physical else np.inf)
                    slab[tuple(local.T)] = values[nearest]
                result[z:z + len(slab)] = slab
                if settings['attribute'] != 'radius':
                    ids, counts = np.unique(slab[mask], return_counts=True)
                    for key, n in zip(ids, counts):
                        distribution[str(int(key))] = distribution.get(str(int(key)), 0) + int(n)
        if not count:
            raise ValueError('Selected labels produce an empty mask')
        os.replace(temporary, path)
    return dict(selected_voxels=count, surface_seeds=len(seeds), voxel_counts=distribution,
                method=settings['method'], attribute=settings['attribute'], dtype=np.dtype(dtype).name,
                tie_policy='cKDTree nearest seed; equal-distance ties may differ from legacy CDT')
