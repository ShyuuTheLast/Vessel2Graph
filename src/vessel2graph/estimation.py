"""Experimental exclusion-parameter proposal from EDT and sampled cross-sections.

No optimization or accuracy guarantee. Does not run skeletonization or alter the input.
"""
from copy import deepcopy
from pathlib import Path
from itertools import product
import time
import threading
import importlib.metadata
import numpy as np
import edt
import cc3d
import psutil
import yaml
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.signal import find_peaks
from .io import load_mask, inspect_input, digest_file, write_json, save_arrays
from .segment import check_memory


def cross_sections(mask, distance, spacing, count, seed):
    """Sample volume-weighted interior seeds, ascend EDT, reject non-tubular Hessians."""
    rng = np.random.default_rng(seed)
    fg = np.flatnonzero(distance > 2 * min(spacing))
    if not len(fg):
        return np.empty((0, 6))
    chosen = rng.choice(fg, size=min(count, len(fg)), replace=False)
    del fg
    points = np.column_stack(np.unravel_index(chosen, mask.shape))
    offsets = np.array(list(product((-1,0,1), repeat=3)))
    for _ in range(32):
        neighbors = np.clip(points[:,None,:] + offsets, 0, np.asarray(mask.shape)-1)
        values = distance[tuple(neighbors.transpose(2,0,1))]
        best = values.argmax(axis=1)
        next_points = neighbors[np.arange(len(points)), best]
        improve = values[np.arange(len(points)), best] > distance[tuple(points.T)]
        if not improve.any():
            break
        points[improve] = next_points[improve]
    points = np.unique(points, axis=0)
    rows = []
    step = min(spacing)/2
    for p in points:
        r = float(distance[tuple(p)])
        if np.any(p < 2) or np.any(p >= np.asarray(mask.shape)-2):
            continue
        H = np.zeros((3,3))
        unit = np.eye(3, dtype=int)
        for i in range(3):
            H[i,i] = (float(distance[tuple(p+unit[i])]) - 2*r + float(distance[tuple(p-unit[i])])) / spacing[i]**2
            for j in range(i):
                H[i,j] = H[j,i] = sum(a*b*float(distance[tuple(p+a*unit[i]+b*unit[j])])
                                      for a,b in product((-1,1), repeat=2)) / (4*spacing[i]*spacing[j])
        eigenvalues, vectors = np.linalg.eigh(H)
        if eigenvalues[1] >= -1e-12 or abs(eigenvalues[2]) > .5*abs(eigenvalues[1]):
            continue
        theta = np.linspace(0,2*np.pi,16,endpoint=False)
        directions = np.cos(theta)[:,None]*vectors[:,0] + np.sin(theta)[:,None]*vectors[:,1]
        radii = np.arange(step,4*r+step,step)
        coordinates = p + directions[:,None,:]*radii[None,:,None]/spacing
        inside = map_coordinates(mask, coordinates.reshape(-1,3).T, order=0, mode='constant', cval=0).reshape(16,-1)
        if np.any(inside.all(axis=1)):
            continue
        first_exit = (inside == 0).argmax(axis=1)
        exits = coordinates[np.arange(16),first_exit]
        if np.any(exits < 0) or np.any(exits >= np.asarray(mask.shape)-1):
            continue  # A crop boundary is not a measured vessel wall.
        outer = float(np.max(radii[first_exit] - step/2))
        rows.append([*p, r, outer, outer/r])
    return np.asarray(rows, dtype=float).reshape(-1,6)


def propose_exclusion(samples, spacing, quantile):
    if len(samples) < 16:
        raise ValueError('Fewer than 16 accepted cross-sections: no parameter proposal produced')
    # Heuristic: upper-quantile transverse extent/inscribed-radius ratio, plus
    # half a voxel diagonal for discretization uncertainty. Constants uncalibrated.
    return {'scale': max(1.0, float(np.quantile(samples[:,5], quantile))),
            'const': float(np.linalg.norm(spacing)/2)}


def estimate(cfg):
    folder = Path(cfg['output']['directory']) / 'estimation'
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stopped = threading.Event()
    peak = [psutil.Process().memory_info().rss]
    def monitor():
        while not stopped.wait(.1):
            peak[0] = max(peak[0], psutil.Process().memory_info().rss)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        meta = inspect_input(cfg['input'])
        check_memory(cfg['resources'], int(np.prod(meta['shape']))*12)
        mask = load_mask(cfg['input'])
        spacing = np.asarray(cfg['input']['voxel_size_nm'], float)
        print('estimate: physical EDT', flush=True)
        distance = edt.edt(np.ascontiguousarray(mask), anisotropy=spacing, black_border=False, parallel=1)
        vals = np.sort(distance[mask != 0])
        maximum = float(vals[-1])
        if not np.isfinite(maximum):
            raise ValueError('EDT is not finite; volume requires observable background')
        grid = np.linspace(0, maximum, cfg['estimation']['erosion_steps'])
        curve = []
        for i, r in enumerate(grid):
            check_memory(cfg['resources'], mask.nbytes*5)
            eroded = distance > r
            components, count = cc3d.connected_components(eroded, connectivity=26, return_N=True)
            curve.append({'radius_nm': float(r), 'components': int(count),
                          'surviving_voxels': int(len(vals)-np.searchsorted(vals,r,side='right'))})
            del components, eroded
            print(f'estimate: erosion {i+1}/{len(grid)}', flush=True)
        smoothed = gaussian_filter1d(np.log1p([x['components'] for x in curve]), .75)
        peaks, _ = find_peaks(smoothed, prominence=max(.05,float(np.ptp(smoothed))*.15))
        samples = cross_sections(mask, distance, spacing, cfg['estimation']['sample_count'], cfg['estimation']['random_seed'])
        save_arrays(folder / 'cross-sections.npz', samples=samples)
        proposal = propose_exclusion(samples, spacing, cfg['estimation']['coverage_quantile'])
        candidate = deepcopy(cfg)
        base = Path(cfg['output']['directory'])
        candidate['output']['directory'] = str(base.parent / (base.name + '-estimated-v1'))
        candidate['skeleton']['source'] = None
        candidate['skeleton']['teasar_params'].update(proposal)
        # Avoid pointing the experiment viewer at old segmentation outputs.
        candidate['visualization'].update(layers=['input_mask','skeleton_by_branch','junctions'],
            hidden_layers=['input_mask','junctions'], branch_volume=None, caliber_volume=None, radius_volume=None)
        candidate_file = folder / 'candidate.yaml'
        candidate_file.write_text(yaml.safe_dump(candidate,sort_keys=False),encoding='utf-8')
        result = {'status':'experimental_unvalidated', 'input':cfg['input'],
            'input_sha256':digest_file(cfg['input']['path']), 'settings':cfg['estimation'],
            'edt_max_nm':maximum, 'foreground_voxels':len(vals), 'erosion_curve':curve,
            'erosion_peak_radii_nm':grid[peaks].tolist(), 'accepted_cross_sections':len(samples),
            'proposal':proposal, 'candidate_config':str(candidate_file),
            'elapsed_seconds':time.perf_counter()-t0, 'sampled_peak_rss_bytes':peak[0],
            'versions':{p:importlib.metadata.version(p) for p in ('edt','connected-components-3d','numpy','scipy')},
            'limitations':['Volume-weighted samples bias toward thick vessels; not a vessel population histogram.',
                'Coarse erosion peaks are diagnostic and may miss narrow modes; no class count is inferred.',
                'Hessian tube filter and 16-ray extent measurement are uncalibrated heuristics.',
                'Only scale and const change; centering/soma/path-cap settings remain as configured.',
                'Half voxel diagonal and coverage quantile are assumptions, not derived optimal constants.']}
        write_json(folder / 'measurements.json', result)
        return result
    finally:
        stopped.set()
        watcher.join()
