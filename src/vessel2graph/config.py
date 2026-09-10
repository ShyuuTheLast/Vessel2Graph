"""Strict YAML configuration; relative paths are relative to the declaring file."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    'schema_version': 1,
    'input': {'path': None, 'dataset': 'main', 'labels': [1], 'voxel_size_nm': None},
    'output': {'directory': None},
    'skeleton': {
        'source': None, 'format': 'arrays',
        'teasar_params': {'scale': 1.5, 'const': 300, 'pdrf_scale': 100000, 'pdrf_exponent': 4,
                          'soma_detection_threshold': float('inf'),
                          'soma_acceptance_threshold': float('inf'),
                          'soma_invalidation_scale': 2, 'soma_invalidation_const': 300,
                          'max_paths': None},
        'dust_threshold': 0, 'parallel': 1, 'fix_branching': True, 'fix_borders': True,
        'crop_empty_border': False,
    },
    'graph': {'radius_clusters': 2, 'random_seed': 42, 'correct_branch_ends': True},
    'segment': {'attribute': 'branch', 'method': 'surface_chessboard', 'surface_samples': 100,
                'query_batch_size': 100000, 'read_slices': 8, 'compression': 'gzip'},
    'resources': {'max_memory_gib': 10, 'reserve_memory_gib': 2},
    'estimation': {'erosion_steps': 16, 'sample_count': 1024, 'random_seed': 42,
                   'coverage_quantile': 0.9},
    'visualization': {'layers': ['input_mask', 'branch_segmentation', 'skeleton_by_branch'],
                      'hidden_layers': [], 'skeleton_overlays': [], 'branch_volume': None, 'caliber_volume': None, 'radius_volume': None,
                      'port': 0, 'open_browser': True},
}


def _merge(defaults, raw, path=''):
    if not isinstance(raw, dict):
        raise ValueError(f'{path or "configuration"} must be a mapping')
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown settings in {path or "root"}: {sorted(unknown)}')
    out = deepcopy(defaults)
    for key, value in raw.items():
        out[key] = _merge(defaults[key], value, path + key + '.') if isinstance(defaults[key], dict) else value
    return out


def _positive(value, name, zero=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < 0 or (not zero and value == 0) or (integer and not isinstance(value, int)):
        raise ValueError(f'Invalid {name}: {value}')


def load_config(path):
    file = Path(path).resolve()
    raw = yaml.safe_load(file.read_text(encoding='utf-8'))
    # Magneton-like root switchboard: main points to one dataset configuration.
    if isinstance(raw, dict) and set(raw) == {'main'}:
        target = (file.parent / raw['main']).resolve()
        if target == file:
            raise ValueError('Configuration points to itself')
        nested = yaml.safe_load(target.read_text(encoding='utf-8'))
        if isinstance(nested, dict) and set(nested) == {'main'}:
            raise ValueError('Only one level of config indirection is supported')
        file, raw = target, nested
    cfg = _merge(DEFAULTS, raw)
    if cfg['schema_version'] != 1:
        raise ValueError('Unsupported config schema_version')
    for section, key in [('input', 'path'), ('output', 'directory'), ('skeleton', 'source')]:
        value = cfg[section][key]
        if value is None and section == 'skeleton':
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{section}.{key} is required')
        cfg[section][key] = str((file.parent / value).resolve())
    spacing = cfg['input']['voxel_size_nm']
    if not isinstance(spacing, list) or len(spacing) != 3:
        raise ValueError('input.voxel_size_nm must contain three ZYX spacings in nm')
    for value in spacing:
        _positive(value, 'voxel_size_nm')
    labels = cfg['input']['labels']
    if not isinstance(labels, list) or not labels or any(type(v) is not int or v <= 0 for v in labels):
        raise ValueError('input.labels must be nonzero positive integer labels')
    if cfg['skeleton']['format'] not in ('arrays', 'legacy_npz'):
        raise ValueError('skeleton.format must be arrays or legacy_npz')
    for section, keys in [('skeleton', ['fix_branching', 'fix_borders', 'crop_empty_border']), ('graph', ['correct_branch_ends'])]:
        for key in keys:
            if type(cfg[section][key]) is not bool:
                raise ValueError(f'{section}.{key} must be Boolean')
    for section, key, zero in [('skeleton','dust_threshold',True), ('skeleton','parallel',False),
                                ('graph','radius_clusters',False), ('graph','random_seed',True),
                                ('segment','query_batch_size',False), ('segment','read_slices',False),
                                ('segment','surface_samples',False)]:
        _positive(cfg[section][key], f'{section}.{key}', zero=zero, integer=True)
    if cfg['segment']['surface_samples'] < 4:
        raise ValueError('surface_samples must be at least 4')
    for key, value in cfg['resources'].items():
        _positive(value, key)
    params = cfg['skeleton']['teasar_params']
    for key, value in params.items():
        if key == 'max_paths' and value is None:
            continue
        if key in ('soma_detection_threshold','soma_acceptance_threshold') and value == float('inf'):
            continue
        _positive(value, key, zero=key in ('const','soma_invalidation_const','soma_invalidation_scale'),
                  integer=key == 'max_paths')
    if cfg['segment']['attribute'] not in ('branch','label','radius','dist_from_largest'):
        raise ValueError('Unknown segmentation attribute')
    if cfg['segment']['method'] not in ('surface_chessboard','surface_euclidean'):
        raise ValueError('Unknown segmentation method')
    if cfg['segment']['compression'] not in ('gzip','lzf',None):
        raise ValueError('compression must be gzip, lzf or null')
    view = cfg['visualization']
    overlays=view.setdefault('skeleton_overlays',[])
    if not isinstance(overlays,list):
        raise ValueError('visualization.skeleton_overlays must be a list')
    overlay_names=set(view['layers'])
    for overlay in overlays:
        if not isinstance(overlay,dict) or not all(isinstance(overlay.get(k),str) and overlay[k].strip() for k in ('name','path')):
            raise ValueError('Each skeleton overlay requires name and path')
        if overlay.get('mode','solid') not in ('solid','branch'):
            raise ValueError('Skeleton overlay mode must be solid or branch')
        if overlay['name'] in overlay_names:
            raise ValueError('Duplicate skeleton overlay name')
        overlay_names.add(overlay['name'])
        shade=overlay.get('color','#ffffff')
        if not isinstance(shade,str) or len(shade)!=7 or shade[0]!='#' or any(c not in '0123456789abcdefABCDEF' for c in shade[1:]):
            raise ValueError('Skeleton overlay color must be #RRGGBB')
        overlay['color']=shade
        if type(overlay.get('visible',True)) is not bool:
            raise ValueError('Skeleton overlay visible must be Boolean')
        overlay['path']=str((file.parent/overlay['path']).resolve())
    for key in ('erosion_steps', 'sample_count', 'random_seed'):
        _positive(cfg['estimation'][key], 'estimation.' + key, integer=True, zero=key == 'random_seed')
    if cfg['estimation']['erosion_steps'] < 4:
        raise ValueError('estimation.erosion_steps must be at least 4')
    _positive(cfg['estimation']['coverage_quantile'], 'coverage_quantile')
    if cfg['estimation']['coverage_quantile'] > 1:
        raise ValueError('coverage_quantile must be <= 1')
    supported = {'input_mask', 'branch_segmentation', 'caliber_segmentation', 'radius_volume', 'skeleton_by_branch',
                 'skeleton_by_caliber', 'skeleton_by_corrected_caliber', 'skeleton_by_radius',
                 'junctions', 'corrected_nodes'}
    if not isinstance(view['layers'], list) or not view['layers'] or any(not isinstance(x, str) or x not in supported for x in view['layers']):
        raise ValueError(f'visualization.layers must list supported layers: {sorted(supported)}')
    if len(set(view['layers'])) != len(view['layers']):
        raise ValueError('Duplicate visualization layers')
    if not isinstance(view['hidden_layers'], list) or any(x not in view['layers'] for x in view['hidden_layers']):
        raise ValueError('hidden_layers must be a subset of layers')
    if type(view['open_browser']) is not bool or type(view['port']) is not int or not 0 <= view['port'] <= 65535:
        raise ValueError('Invalid viewer open_browser or port')
    for key in ('branch_volume', 'caliber_volume', 'radius_volume'):
        if view[key] is not None:
            if not isinstance(view[key], str) or not view[key].strip():
                raise ValueError(f'visualization.{key} must be a path or null')
            view[key] = str((file.parent / view[key]).resolve())
    return cfg
