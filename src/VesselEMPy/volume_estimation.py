"""Experimental TEASAR initialization from a binary mask and physical spacing only.

No reference skeleton or learned calibration coefficient is read. Explicit geometric
and penalty tolerances are hypotheses; the report records them for evaluation.
"""
from copy import deepcopy
from pathlib import Path
import time
import importlib.metadata
import h5py
import numpy as np
import edt
import cc3d
import fill_voids
import yaml
from scipy.ndimage import map_coordinates, gaussian_filter1d, binary_closing
from scipy.optimize import brentq
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from .io import load_mask, inspect_input, digest_file, write_json, save_arrays
from .segment import check_memory
from .estimation import cross_sections


def close_physical(mask, spacing, radius_nm):
    """Explicit physical-ball closing, padded to preserve foreground at crop edges."""
    spacing=np.asarray(spacing,dtype=float)
    if not np.isfinite(radius_nm) or radius_nm<0:
        raise ValueError('Closing radius must be finite and nonnegative')
    if spacing.shape!=(3,) or not np.all(np.isfinite(spacing)&(spacing>0)):
        raise ValueError('Closing requires three positive finite spacings')
    width=np.floor(radius_nm/spacing).astype(int)
    axes=np.ogrid[tuple(slice(-w,w+1) for w in width)]
    footprint=sum((a*s)**2 for a,s in zip(axes,spacing))<=radius_nm**2
    padded=np.pad(mask,[(int(w),int(w)) for w in width],mode='constant')
    closed=binary_closing(padded,structure=footprint)
    result=closed[tuple(slice(int(w),int(w)+n) for w,n in zip(width,mask.shape))].astype(np.uint8)
    if np.any((mask!=0)&(result==0)):
        raise ValueError('Closing unexpectedly removed foreground')
    return result,dict(radius_nm=float(radius_nm),footprint_shape=list(footprint.shape),
        footprint_voxels=int(footprint.sum()),added_voxels=int(result.sum()-np.count_nonzero(mask)),
        removed_voxels=0,boundary='zero padded before closing; cropped afterward')


def prepare(mask, fill_holes):
    """Fill enclosed background only; do not use dilation/closing to bridge gaps."""
    if not fill_holes:
        return mask.copy(),0
    result,n=fill_voids.fill(mask,in_place=False,return_fill_count=True)
    return np.asarray(result,dtype=np.uint8),int(n)


def profile_sections(mask, distance, spacing, count, seed):
    # EDT ridge candidates from the first estimator; PCA supplies a second axis
    # estimate insensitive to individual finite-difference Hessian directions.
    seeds=cross_sections(mask,distance,spacing,count,seed)
    rows=[]
    for row in seeds:
        p=row[:3].astype(int)
        r=row[3]
        width=np.ceil(3*r/spacing).astype(int)
        lo=np.maximum(0,p-width); hi=np.minimum(mask.shape,p+width+1)
        xyz=(np.argwhere(mask[tuple(slice(a,b) for a,b in zip(lo,hi))])+lo-p)*spacing
        xyz=xyz[np.linalg.norm(xyz,axis=1)<=3*r]
        if len(xyz)<16:
            continue
        eig,v=np.linalg.eigh(np.cov(xyz.T))
        if eig[-1]<1.5*eig[-2]:
            continue
        theta=np.arange(32)*np.pi/16
        directions=np.cos(theta)[:,None]*v[:,0]+np.sin(theta)[:,None]*v[:,1]
        step=min(spacing)/2
        lengths=np.arange(step,4*r,step)
        points=p+directions[:,None,:]*lengths[None,:,None]/spacing
        inside=map_coordinates(mask,points.reshape(-1,3).T,order=0,mode='constant',cval=0).reshape(32,-1)
        if inside.all(axis=1).any():
            continue
        exits=(inside==0).argmax(axis=1)
        xyz_exit=points[np.arange(32),exits]
        if np.any(xyz_exit<0) or np.any(xyz_exit>=np.asarray(mask.shape)-1):
            continue
        extent=float(np.max(lengths[exits]-step/2))
        displaced=p+directions*(.5*r)/spacing
        off=map_coordinates(distance,displaced.T,order=1,mode='nearest')
        # Median over directions limits contamination from the longitudinal plane
        # and a neighboring arm; reject samples that do not center their profiles.
        off=float(np.median(off))
        if not 0<=off<r:
            continue
        rows.append([*p,r,extent,extent/r,off,float(eig[-1]/eig[-2])])
    return np.asarray(rows,dtype=float).reshape(-1,8)


def derive_parameters(samples, maximum, spacing, quantile=.9):
    if len(samples)<16:
        raise ValueError('Fewer than 16 reliable profiles; no automatic parameters produced')
    # One transverse radius is a tip-clearance floor: unlike scale*DBF it does not
    # collapse at a boundary target. This is an explicit geometric heuristic.
    scale=max(1.,float(np.quantile(samples[:,5],quantile)))
    tip_radius=float(np.quantile(samples[:,4],quantile))
    const=tip_radius+float(np.linalg.norm(spacing)/2)
    # Balanced radius strata keep the most numerous caliber from defining the
    # entire centering criterion. These quantiles are not inferred vessel classes.
    boundaries=np.quantile(samples[:,3],[0,1/3,2/3,1])
    groups=np.clip(np.searchsorted(boundaries[1:-1],samples[:,3],side='right'),0,2)
    M=maximum**1.01
    center=1-samples[:,3]/M; off=1-samples[:,6]/M
    selected=None; survey=[]
    for exponent in (1,2,4,8,16,32,64):
        ratio=(off/center)**exponent
        delta=off**exponent-center**exponent
        contrasts=[float(np.quantile(ratio[groups==g],.1)) for g in range(3) if np.any(groups==g)]
        differences=[float(np.quantile(delta[groups==g],.1)) for g in range(3) if np.any(groups==g)]
        weakest=min(differences)
        amplitude=float(2**np.ceil(np.log2(32/weakest))) if weakest>0 else None
        survey.append(dict(exponent=exponent,lower_decile_contrast_by_radius_stratum=contrasts,
            required_amplitude=amplitude))
        # Target a fourfold penalty ratio at half-radius displacement, plus
        # a penalty difference 32 times the normalized DAF range. Neither is a
        # path-optimality guarantee. Avoid an unlimited precision-destroying A.
        if min(contrasts)>=4 and amplitude is not None and amplitude<=2**24:
            selected=dict(pdrf_exponent=exponent,pdrf_scale=amplitude)
            break
    if selected is None:
        raise ValueError('No numerically bounded global centering proposal satisfies sampled profiles')
    return dict(scale=scale,const=const,**selected),dict(tip_radius_nm=tip_radius,
        radius_stratum_edges_nm=boundaries.tolist(),centering_survey=survey,
        half_radius_penalty_ratio_target=4,absolute_penalty_difference_target=32,
        maximum_amplitude=2**24,coverage_quantile=quantile)


def erosion_features(distance, steps):
    maximum=float(distance.max())
    # Reuse the original erosion/GMM recipe, but retain its results as features,
    # not independent physical radius populations. Full arrays are saved.
    grid=np.linspace(0,maximum,steps)
    curve=[]
    for radius in grid:
        labels,n=cc3d.connected_components(distance>radius,connectivity=26,return_N=True)
        curve.append(dict(radius_nm=float(radius),components=int(n),surviving_voxels=int(np.count_nonzero(labels))))
        del labels
    counts=np.array([r['components'] for r in curve])
    smooth=gaussian_filter1d(np.log1p(counts),.75)
    peaks,_=find_peaks(smooth,prominence=max(.05,float(np.ptp(smooth))*.15))
    valid=counts>0
    w=np.log1p(counts[valid])
    sample=np.repeat(grid[valid],np.maximum(1,np.round(5*(w-w.min()+1)).astype(int)))
    fit=GaussianMixture(n_components=min(3,len(np.unique(sample))),random_state=0).fit(sample[:,None])
    order=np.argsort(fit.means_.ravel())
    means=fit.means_.ravel()[order];sigmas=np.sqrt(fit.covariances_.ravel()[order]);weights=fit.weights_[order]
    splits=[]
    for i in range(len(means)-1):
        def difference(x):
            return np.log(weights[i]/sigmas[i])-.5*((x-means[i])/sigmas[i])**2 - np.log(weights[i+1]/sigmas[i+1])+.5*((x-means[i+1])/sigmas[i+1])**2
        splits.append(float(brentq(difference,means[i],means[i+1])) if difference(means[i])*difference(means[i+1])<0 else None)
    return dict(curve=curve,peak_radii_nm=grid[peaks].tolist(),gmm_means_nm=means.tolist(),
        gmm_sigma_nm=sigmas.tolist(),gmm_weights=weights.tolist(),corrected_crossings_nm=splits)


def estimate_volume(cfg, fill_holes=False, closing_radius_nm=0):
    started=time.perf_counter()
    implementation_hash=digest_file(__file__)
    meta=inspect_input(cfg['input'])
    check_memory(cfg['resources'],int(np.prod(meta['shape']))*12)
    base=Path(cfg['output']['directory'])
    folder=base/'estimation-volume-v2'
    folder.mkdir(parents=True,exist_ok=True)
    original=load_mask(cfg['input'])
    components,original_components=cc3d.connected_components(original,connectivity=26,return_N=True)
    del components
    spacing=np.asarray(cfg['input']['voxel_size_nm'],dtype=float)
    closing=None
    if closing_radius_nm!=0:
        closed,closing=close_physical(original,spacing,closing_radius_nm)
        mask,added=prepare(closed,fill_holes)
        del closed
        print(f'estimate: closing added {closing["added_voxels"]} voxels',flush=True)
    else:
        mask,added=prepare(original,fill_holes)
    raw_count=int(np.count_nonzero(original)); del original
    print(f'estimate: enclosed-hole filling added {added} voxels',flush=True)
    spacing=np.asarray(cfg['input']['voxel_size_nm'],dtype=float)
    distance=edt.edt(mask,anisotropy=spacing,parallel=1)
    maximum=float(distance.max())
    if not np.isfinite(maximum) or maximum<=0:
        raise ValueError('Input requires a finite positive EDT maximum')
    print('estimate: erosion/caliber features',flush=True)
    features=erosion_features(distance,cfg['estimation']['erosion_steps'])
    if features['curve'][0]['components']<original_components:
        raise ValueError('Preparation would merge input components; inspect preparation before proceeding')
    print('estimate: physical transverse profiles',flush=True)
    samples=profile_sections(mask,distance,spacing,cfg['estimation']['sample_count'],cfg['estimation']['random_seed'])
    save_arrays(folder/'profiles.npz',samples=samples)
    proposal,rationale=derive_parameters(samples,maximum,spacing,cfg['estimation']['coverage_quantile'])
    del distance
    prepared=folder/'prepared-mask.h5'
    with h5py.File(prepared,'w') as f:
        ds=f.create_dataset('main',data=mask,compression='gzip')
        ds.attrs['voxel_size_nm_zyx']=spacing
    candidate=deepcopy(cfg)
    candidate['input'].update(path=str(prepared.resolve()),dataset='main',labels=[1])
    candidate['output']['directory']=str(base.parent/(base.name+'-volume-estimated-v2'))
    candidate['skeleton']['source']=None
    candidate['skeleton']['teasar_params'].update(proposal,soma_detection_threshold=float('inf'),
        soma_acceptance_threshold=float('inf'),max_paths=None)
    candidate['skeleton']['dust_threshold']=0
    candidate['visualization'].update(layers=['input_mask','skeleton_by_branch','junctions'],
        hidden_layers=['input_mask','junctions'],branch_volume=None,caliber_volume=None,radius_volume=None)
    candidate_file=folder/'candidate.yaml'
    candidate_file.write_text(yaml.safe_dump(candidate,sort_keys=False),encoding='utf-8')
    report=dict(status='experimental_binary_only',method='volume_profiles_v2',
        source_input=cfg['input'],input_sha256=digest_file(cfg['input']['path']),
        prepared_mask_sha256=digest_file(prepared),implementation_sha256=implementation_hash,
        fill_enclosed_holes=fill_holes,closing=closing,original_foreground=raw_count,filled_voxels=added,
        total_added_voxels=int(np.count_nonzero(mask))-raw_count,
        original_components=int(original_components),prepared_components=features['curve'][0]['components'],
        prepared_foreground=int(np.count_nonzero(mask)),edt_max_nm=maximum,
        accepted_profiles=len(samples),profile_columns=['z','y','x','center_radius_nm','extent_nm','extent_ratio','offcenter_edt_nm','axis_eigenvalue_ratio'],
        proposal=proposal,rationale=rationale,erosion_features=features,
        candidate_config=str(candidate_file),elapsed_seconds=time.perf_counter()-started,
        versions={p:importlib.metadata.version(p) for p in ('edt','fill-voids','connected-components-3d','numpy','scipy','scikit-learn')},
        limitations=['No reference skeleton or calibration file was used.',
            'Tip-clearance floor, quantiles and penalty tolerances are unvalidated geometric heuristics.',
            'Samples favor larger volumes and exclude very thin/poorly resolved profiles.',
            'GMM curve features are diagnostic, not fitted physical vessel populations.',
            'Centering uses the global EDT maximum; multi-component normalization requires further validation.',
            'Explicit prepared mask is used for both estimation and tracing; soma handling is disabled.'])
    write_json(folder/'measurements.json',report)
    return report
