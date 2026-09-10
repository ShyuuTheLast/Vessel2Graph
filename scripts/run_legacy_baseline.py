"""Run the original main with explicit local settings and disabled GUI-only imports.

Scientific functions in reference/Vessel2Graph/src are executed unchanged. This is
a temporary reproducibility harness, not the phase-1 package implementation.
"""
import argparse
import ast
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clusters', type=int, default=2)
    parser.add_argument('--reuse-skeleton', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'reference/Vessel2Graph/src'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_path = args.input.resolve()
    reuse = str(args.reuse_skeleton.resolve()) if args.reuse_skeleton else None
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(root / 'work/matplotlib')
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0, str(source))
    os.chdir(output)
    import psutil
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import graph_segmentation as gs
    import data_saver as ds

    start = time.monotonic()
    report = {'upstream_commit': 'fbaa80e1437aaeb6b5e30d99a5ef13f6aad5c9b9',
              'status': 'running', 'stages': [], 'input': str(input_path),
              'selected_labels': [1], 'spacing_zyx_nm': [320, 256, 256],
              'clusters': args.clusters, 'peak_rss_bytes': 0,
              'packages': {p: importlib.metadata.version(p) for p in
                           ['kimimaro', 'numpy', 'scipy', 'cloud-volume', 'networkx', 'matplotlib']},
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source.glob('*.py')}}
    def write_report():
        (output / 'run-report.json').write_text(json.dumps(report, indent=2) + '\n')

    process = psutil.Process()
    stop = threading.Event()
    def monitor():
        while not stop.wait(1):
            rss = process.memory_info().rss
            report['peak_rss_bytes'] = max(report['peak_rss_bytes'], rss)
            if rss > 13 * 1024**3 or psutil.virtual_memory().available < 1.5 * 1024**3:
                report['status'] = 'stopped_memory_guard'
                report['elapsed_seconds'] = time.monotonic() - start
                write_report()
                print('Stopped baseline to preserve memory headroom.', flush=True)
                os._exit(70)
    threading.Thread(target=monitor, daemon=True).start()

    def timed(module, name):
        original = getattr(module, name)
        @functools.wraps(original)
        def run(*a, **kw):
            print(f'START {name}', flush=True)
            report['active_stage'] = name
            write_report()
            t = time.monotonic()
            before = nx.number_connected_components(a[0]) if name == 'merge_graph_components' else None
            result = original(*a, **kw)
            stage = {'name': name, 'seconds': time.monotonic() - t}
            if name == 'skeletonize_volume':
                report['teasar_params'] = a[1]
                stage.update(skeletons=len(result), vertices=sum(len(s.vertices) for s in result.values()),
                             edges=sum(len(s.edges) for s in result.values()))
            elif name == 'full_graph_generation':
                stage.update(nodes=result.number_of_nodes(), edges=result.number_of_edges(),
                             components=nx.number_connected_components(result))
            elif name == 'merge_graph_components':
                stage.update(components_before=before, components_after=nx.number_connected_components(result[0]))
            elif name == 'simplified_graph_generation':
                stage['branches'] = len(result[1])
            elif name == 'relabel_graph_with_branches':
                stage['total_length_nm'] = float(result[2])
            report['stages'].append(stage)
            write_report()
            print(f'END {name}: {stage}', flush=True)
            return result
        setattr(module, name, run)

    for name in ['skeletonize_volume', 'full_graph_generation', 'merge_graph_components',
                 'simplified_graph_generation', 'plot_elbow_curve', 'cluster_radius',
                 'relabel_graph_with_branches', 'calculate_branching_angles',
                 'relabel_small_branches_near_big_ones', 'label_branch_points',
                 'skeleton_in_array', 'segment_volume']:
        timed(gs, name)
    for name in ['save_skeleton_to_file', 'save_segmented_volume', 'save_stats', 'save_paths_as_pickle']:
        timed(ds, name)

    def save_plot():
        plt.savefig(output / 'elbow-curve.png', dpi=150)
        plt.close('all')
    plt.show = save_plot

    settings = {
        'input_file': str(input_path), 'generate_new_skeleton': not bool(reuse),
        'existing_skeletons_path': reuse, 'do_segment': True,
    }
    tree = ast.parse((source / 'main.py').read_text(encoding='utf-8-sig'))
    # Only imports of optional Mayavi/Napari viewer functions are omitted.
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'visualization')]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    for node in fn.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in settings:
                node.value = ast.Constant(settings[key])
            elif key == 'optimal_clusters':
                node.value = ast.Constant(args.clusters)
    ast.fix_missing_locations(tree)
    (output / 'adapted-main.py').write_text(ast.unparse(tree) + '\n', encoding='utf-8')
    report['launcher_adaptations'] = {'settings': settings, 'clusters': args.clusters,
                                     'viewer_imports_omitted': True, 'elbow_plot_saved_headlessly': True}
    write_report()
    try:
        ns = {'__name__': 'legacy_baseline', '__file__': str(source / 'main.py')}
        exec(compile(tree, str(source / 'main.py'), 'exec'), ns)
        ns['main']()
        report['status'] = 'completed'
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        stop.set()
        report['elapsed_seconds'] = time.monotonic() - start
        report['files'] = {p.name: p.stat().st_size for p in output.iterdir() if p.is_file()}
        write_report()


if __name__ == '__main__':
    main()
