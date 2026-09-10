"""Import checks, tiny synthetic skeleton, and read-only HDF5 metadata inspection."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}
    packages = {
        'numpy': 'numpy', 'scipy': 'scipy', 'h5py': 'h5py', 'networkx': 'networkx',
        'scikit-learn': 'sklearn', 'PyYAML': 'yaml', 'kimimaro': 'kimimaro',
        'connected-components-3d': 'cc3d', 'edt': 'edt',
        'fill-voids': 'fill_voids', 'psutil': 'psutil',
    }
    report['packages'] = {}
    for distribution, module in packages.items():
        importlib.import_module(module)
        report['packages'][distribution] = importlib.metadata.version(distribution)

    import numpy as np
    import kimimaro
    import h5py

    z, y, x = np.indices((40, 20, 20))
    mask = (((y - 10)**2 + (x - 10)**2 <= 3**2) & (z >= 3) & (z < 37)).astype(np.uint8)
    skels = kimimaro.skeletonize(
        mask, anisotropy=(320, 256, 256), dust_threshold=0, progress=False,
        parallel=1, fix_branching=True, fix_borders=False,
        teasar_params={'scale': 1.5, 'const': 256, 'pdrf_scale': 100000,
                       'pdrf_exponent': 4, 'soma_detection_threshold': float('inf'),
                       'soma_acceptance_threshold': float('inf'), 'max_paths': None},
    )
    if len(skels) != 1:
        raise AssertionError(f'Expected one synthetic skeleton, got {len(skels)}')
    skel = next(iter(skels.values()))
    if len(skel.vertices) < 2 or len(skel.edges) < 1 or not np.isfinite(skel.vertices).all():
        raise AssertionError('Invalid synthetic skeleton')
    report['synthetic_tube'] = {'vertices': len(skel.vertices), 'edges': len(skel.edges)}

    if args.input:
        report['input'] = {'path': str(args.input.resolve()), 'datasets': {}}
        with h5py.File(args.input, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    report['input']['datasets'][name] = {
                        'shape': list(obj.shape), 'dtype': str(obj.dtype),
                        'uncompressed_bytes': int(obj.size * obj.dtype.itemsize),
                        'compression': obj.compression, 'chunks': obj.chunks,
                    }
            f.visititems(visit)
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + '\n', encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
