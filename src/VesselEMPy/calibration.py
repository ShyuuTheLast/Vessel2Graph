"""Experimental reference-calibrated exclusion, without a production reference graph.

Training fits a dimensionless additive margin from an accepted reference. Application
measures only the selected input mask. Neither operation runs skeletonization.
"""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
import edt
import yaml
from scipy.spatial import cKDTree
from .io import digest_file, inspect_input, load_mask, load_arrays, write_json
from .segment import check_memory


def exclusion_from_model(radius_max_nm, spacing, model):
    if model.get('schema_version') != 1 or model.get('method') != 'reference_coverage_v1':
        raise ValueError('Unsupported exclusion calibration model')
    values = [radius_max_nm, *spacing, model['scale'], model['const_radius_multiplier'],
              model['voxel_diagonal_multiplier']]
    if not np.isfinite(values).all() or radius_max_nm <= 0 or np.any(np.asarray(spacing)<=0):
        raise ValueError('Calibration measurements must be finite and positive')
    if model['scale'] <= 0 or model['const_radius_multiplier'] < 0 or model['voxel_diagonal_multiplier'] < 0:
        raise ValueError('Invalid calibration coefficients')
    return dict(scale=float(model['scale']), const=float(model['const_radius_multiplier']*radius_max_nm +
        model['voxel_diagonal_multiplier']*np.linalg.norm(spacing)))


def measure(cfg):
    meta = inspect_input(cfg['input'])
    check_memory(cfg['resources'], int(np.prod(meta['shape']))*6)
    mask = load_mask(cfg['input'])
    distance = edt.edt(mask, anisotropy=cfg['input']['voxel_size_nm'], parallel=1)
    maximum = float(distance.max())
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Input requires a finite positive physical EDT maximum')
    del distance
    return mask, maximum


def calibrate(cfg, reference, destination):
    """Reference is a completed run directory with graph and resolved configuration."""
    started = time.perf_counter()
    reference = Path(reference)
    path = reference/'graph.npz'
    settings = yaml.safe_load((reference/'resolved-config.yaml').read_text())
    ref_input = settings['input']
    if any(ref_input[k] != cfg['input'][k] for k in ('dataset','labels','voxel_size_nm')):
        raise ValueError('Calibration input selection/spacing differs from reference')
    input_hash = digest_file(cfg['input']['path'])
    if digest_file(ref_input['path']) != input_hash:
        raise ValueError('Calibration input differs from reference input')
    graph = load_arrays(path)
    reference_hash = digest_file(path)
    mask, maximum = measure(cfg)
    spacing = np.asarray(cfg['input']['voxel_size_nm'])
    tree = cKDTree(graph['vertices'])
    scale = settings['skeleton']['teasar_params']['scale']
    required = 0.
    foreground = 0
    for z in range(0, len(mask), 8):
        points = np.argwhere(mask[z:z+8])
        points[:,0] += z
        distances, nodes = tree.query(points*spacing)
        foreground += len(points)
        if len(points):
            required = max(required,float(np.max(distances-scale*graph['radius'][nodes])))
    context = {k:v for k,v in settings['skeleton']['teasar_params'].items() if k not in ('scale','const')}
    if digest_file(path) != reference_hash:
        raise ValueError('Reference changed during calibration')
    model = dict(schema_version=1,method='reference_coverage_v1',
        status='experimental_single_dataset_calibration', scale=float(scale),
        const_radius_multiplier=required/maximum,voxel_diagonal_multiplier=.5,
        training_radius_max_nm=maximum,training_required_const_nm=required,
        training_input_sha256=input_hash,training_reference_sha256=reference_hash,
        training_input=cfg['input'], training_reference=str(path.resolve()),
        fixed_teasar_context=context,
        fixed_tracing_context={k:settings['skeleton'][k] for k in ('fix_branching','fix_borders','dust_threshold')},
        selected_voxels=foreground,elapsed_seconds=time.perf_counter()-started,
        limitations=['Single-dataset calibration; transfer is unvalidated.',
            'Euclidean nearest-reference coverage is not an invalidation simulation.',
            'Scale and centering are inherited from the accepted reference, not newly inferred.',
            'EDT maximum and the reference coverage maximum can be sensitive to local geometry.'])
    # JSON cannot encode infinity: fixed settings are recorded with YAML's .inf spelling.
    model['fixed_teasar_context'] = {k:('.inf' if v==float('inf') else v) for k,v in context.items()}
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,model)
    return model


def estimate_calibrated(cfg, calibration_file):
    started = time.perf_counter()
    model = json.loads(Path(calibration_file).read_text())
    exclusion_from_model(1,cfg['input']['voxel_size_nm'],model)  # Validate before allocating.
    for key,expected in model.get('fixed_teasar_context',{}).items():
        if expected == '.inf':
            expected = float('inf')
        if cfg['skeleton']['teasar_params'][key] != expected:
            raise ValueError(f'Calibration requires reference TEASAR context: {key}={expected}')
    for key,expected in model.get('fixed_tracing_context',{}).items():
        if cfg['skeleton'][key] != expected:
            raise ValueError(f'Calibration requires reference tracing context: {key}={expected}')
    mask,maximum = measure(cfg)
    foreground = int(np.count_nonzero(mask))
    del mask
    proposal = exclusion_from_model(maximum,cfg['input']['voxel_size_nm'],model)
    candidate = deepcopy(cfg)
    base = Path(cfg['output']['directory'])
    candidate['output']['directory'] = str(base.parent/(base.name+'-reference-calibrated-v1'))
    candidate['skeleton']['source'] = None
    candidate['skeleton']['teasar_params'].update(proposal)
    candidate['visualization'].update(layers=['input_mask','skeleton_by_branch','junctions'],
        hidden_layers=['input_mask','junctions'],branch_volume=None,caliber_volume=None,radius_volume=None)
    folder = base/'estimation-reference'
    folder.mkdir(parents=True,exist_ok=True)
    candidate_file = folder/'candidate.yaml'
    candidate_file.write_text(yaml.safe_dump(candidate,sort_keys=False),encoding='utf-8')
    report = dict(status='experimental_unvalidated',method=model['method'],proposal=proposal,
        radius_max_nm=maximum,selected_voxels=foreground,
        input_sha256=digest_file(cfg['input']['path']),calibration_sha256=digest_file(calibration_file),
        calibration_file=str(Path(calibration_file).resolve()),candidate_config=str(candidate_file),
        elapsed_seconds=time.perf_counter()-started,
        evaluation_role='training_dataset' if digest_file(cfg['input']['path'])==model['training_input_sha256'] else 'transfer_unvalidated',
        limitations=model['limitations'])
    write_json(folder/'measurements.json',report)
    return report
