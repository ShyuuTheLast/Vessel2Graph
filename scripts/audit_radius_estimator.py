"""Read-only audit of saved caliber experiment and installed Kimimaro mechanisms.

One EDT of the saved mouse chunk, no erosion sweep or large-volume skeletonization.
Original scripts are inspected through AST, never imported/executed wholesale.
"""
from pathlib import Path
import ast
import argparse
import json
import math
import time
import h5py
import numpy as np
import edt
import psutil
from scipy.optimize import brentq
from VesselEMPy.io import digest_file, write_json
from kimimaro.trace import compute_pdrf
import kimimaro.skeletontricks as tricks


def crossing(mu1, sigma1, w1, mu2, sigma2, w2):
    """Weighted-density crossing between ordered means; refuse absent crossings."""
    def delta(x):
        return math.log(w1/sigma1)-.5*((x-mu1)/sigma1)**2 - math.log(w2/sigma2)+.5*((x-mu2)/sigma2)**2
    return brentq(delta, mu1, mu2)


def main():
    t0 = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Directory containing the original radius scripts and saved experiment')
    source=parser.parse_args().source.resolve()
    out = root/'outputs/radius-estimator-audit'
    out.mkdir(parents=True, exist_ok=True)
    stats_file = source/'mouse_chunk_2_radius_stats_gmm.npz'
    with np.load(stats_file, allow_pickle=False) as f:
        stats = {k: f[k].tolist() for k in f.files}
    script = source/'estimate_radii.py'
    tree = ast.parse(script.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='gaussian_intersection')
    scope = {'math': math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(script), 'exec'), scope)
    original = scope['gaussian_intersection']
    regress = dict(original=original(0,1,.8,4,1,.2), corrected=crossing(0,1,.8,4,1,.2),
        expected=2+math.log(4)/4)
    assert abs(regress['corrected']-regress['expected'])<1e-12
    splits = [crossing(stats['peak_nm'][i], stats['sigma_nm'][i], stats['weight_frac'][i],
        stats['peak_nm'][i+1], stats['sigma_nm'][i+1], stats['weight_frac'][i+1]) for i in (1,2)]
    specs = [('chunk2_dist_seg.h5','main'), ('chunk2_label_seg.h5','main'),
        ('mouse_chunk_2_radius_classes_gmm.h5','radius_class_gmm'),
        ('mouse_chunk_2_radius_classes_dustclean.h5','main'),
        ('mouse_chunk_2_radius_classes_full.h5','main')]
    if psutil.virtual_memory().available < 3*1024**3:
        raise MemoryError('Audit needs 3 GiB available before chunk EDT')
    with h5py.File(source/specs[0][0],'r') as f:
        mask = np.empty(f['main'].shape, dtype=np.uint8)
        for z in range(0, len(mask), 8):
            mask[z:z+8] = f['main'][z:z+8] > 0
    distance = edt.edt(mask, anisotropy=stats['vox_size_nm'], parallel=1)
    counts = [dict() for _ in specs]
    transitions = [np.zeros((4,4), dtype=np.int64) for _ in range(2)]
    mismatches = dict(dist_vs_label_foreground=0, gmm_vs_input_foreground=0,
        dust_vs_gmm_foreground=0, full_vs_gmm_foreground=0,
        gmm_vs_saved_threshold_reconstruction=0, gmm_vs_corrected_thresholds=0)
    files = [h5py.File(source/name,'r') for name,_ in specs]
    try:
        for z in range(0,len(mask),8):
            arrays = [f[key][z:z+8] for f,(_,key) in zip(files,specs)]
            a,b,g,d,full = arrays
            for values, counter in zip(arrays,counts):
                u,n=np.unique(values, return_counts=True)
                for label,count in zip(u,n): counter[int(label)] = counter.get(int(label),0)+int(count)
            for left,right,matrix in ((g,d,transitions[0]),(d,full,transitions[1])):
                matrix += np.bincount((left.astype(np.int64)*4+right).ravel(),minlength=16).reshape(4,4)
            for key,left,right in [('dist_vs_label_foreground',a,b),('gmm_vs_input_foreground',g,a),
                ('dust_vs_gmm_foreground',d,g),('full_vs_gmm_foreground',full,g)]:
                mismatches[key] += int(np.count_nonzero((left>0)!=(right>0)))
            for key,thresholds in [('gmm_vs_saved_threshold_reconstruction',stats['split_nm']),
                ('gmm_vs_corrected_thresholds',splits)]:
                labels = np.where(mask[z:z+8],np.digitize(distance[z:z+8],thresholds)+1,0)
                mismatches[key] += int(np.count_nonzero(labels!=g))
    finally:
        for f in files: f.close()
    penalties=[]
    maximum=stats['r_max_nm']
    for exponent in (4,8,16,32,64):
        for r in stats['peak_nm'][1:]:
            dbf=np.array([r,.8*r],dtype=np.float32).reshape(2,1,1)
            p=compute_pdrf(maximum,100000,exponent,dbf,np.zeros_like(dbf)).ravel()
            penalties.append(dict(exponent=exponent, radius_nm=r, center=float(p[0]),
                offcenter=float(p[1]), contrast=float(p[1]/p[0]), difference=float(p[1]-p[0])))
    # Tiny call into the installed extension: does its invalidation scale accept a field?
    labels=np.ones((3,3,3),dtype=np.uint8,order='F')
    dbf=np.ones_like(labels,dtype=np.float32,order='F')
    try:
        tricks.roll_invalidation_ball_inside_component(labels,dbf,np.ones_like(dbf),0,(1,1,1),[(1,1,1)])
        field_support='accepted array in this probe'
    except (TypeError,ValueError) as e:
        field_support=type(e).__name__+': '+str(e)
    result=dict(stats=stats, corrected_split_nm=splits, gaussian_regression=regress,
        source_hashes={name:digest_file(source/name) for name,_ in specs},
        stats_sha256=digest_file(stats_file), chunk_shape=list(mask.shape),
        chunk_edt_max_nm=float(distance.max()), label_counts={name:c for (name,_),c in zip(specs,counts)},
        mismatches=mismatches, gmm_to_dust_transition=transitions[0].tolist(),
        dust_to_full_transition=transitions[1].tolist(), scalar_pdrf_survey=penalties,
        invalidation_array_scale_probe=field_support,
        limitations=['Chunk parent crop coordinates unknown.',
            'PDRF survey uses saved global maximum and representative radii, not measured local profiles.',
            'Single CPU EDT reproduces thresholds only; original GPU 512-pass curve not rerun.',
            'Source code and saved artifacts may differ; mask identity does not establish full lineage.'],
        elapsed_seconds=time.perf_counter()-t0)
    write_json(out/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
