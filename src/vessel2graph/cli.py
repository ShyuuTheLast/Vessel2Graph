"""Headless command line interface."""
import argparse
import json
from .config import load_config
from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(description='Vessel2Graph configuration-driven pipeline')
    parser.add_argument('command', choices=['run', 'skeleton', 'graph', 'segment', 'report', 'status', 'view', 'estimate', 'calibrate'])
    parser.add_argument('--config', default='config.yaml', help='Dataset YAML or main switchboard YAML')
    parser.add_argument('--force', action='store_true', help='Recompute requested stage and all prerequisites')
    parser.add_argument('--reference', help='Accepted run directory, required for calibrate')
    parser.add_argument('--calibration', help='Calibration JSON: output for calibrate, input for estimate')
    parser.add_argument('--method', choices=['volume','cross-section'], help='Estimator method (default: volume)')
    parser.add_argument('--fill-holes', action='store_true', help='Explicitly fill enclosed holes for volume estimation and its candidate')
    parser.add_argument('--closing-radius-nm', type=float, default=0, help='Explicit physical-ball closing radius before optional filling (default: 0)')
    args = parser.parse_args(argv)
    if (args.method or args.fill_holes or args.closing_radius_nm) and args.command != 'estimate':
        parser.error('--method/--fill-holes are only supported for estimate')
    if args.calibration and (args.method or args.fill_holes or args.closing_radius_nm):
        parser.error('Reference calibration cannot be combined with binary-volume estimator options')
    if (args.fill_holes or args.closing_radius_nm) and args.method == 'cross-section':
        parser.error('--fill-holes requires the volume method')
    if args.command == 'calibrate' and (not args.reference or not args.calibration):
        parser.error('calibrate requires --reference and --calibration')
    if args.reference and args.command != 'calibrate':
        parser.error('--reference is only supported for calibrate')
    if args.calibration and args.command not in ('calibrate','estimate'):
        parser.error('--calibration is only supported for calibrate or estimate')
    try:
        if args.command == 'calibrate':
            from .calibration import calibrate
            print(json.dumps(calibrate(load_config(args.config),args.reference,args.calibration),indent=2))
            return 0
        if args.command == 'estimate':
            if args.calibration:
                from .calibration import estimate_calibrated
                print(json.dumps(estimate_calibrated(load_config(args.config),args.calibration),indent=2))
                return 0
            if args.method == 'cross-section':
                from .estimation import estimate
                result=estimate(load_config(args.config))
            else:
                from .volume_estimation import estimate_volume
                result=estimate_volume(load_config(args.config),fill_holes=args.fill_holes,closing_radius_nm=args.closing_radius_nm)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == 'view':
            from .viewer import serve
            serve(load_config(args.config))
            return 0
        result = run(load_config(args.config), target='report' if args.command in ('run', 'status') else args.command,
                     force=args.force, status=args.command == 'status')
    except (ValueError, OSError, KeyError, MemoryError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0
