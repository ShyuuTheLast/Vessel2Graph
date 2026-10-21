"""Small known-geometry checks of the original erosion/GMM recipe; CPU only."""
from pathlib import Path
import time
import numpy as np
import edt
import cc3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from VesselEMPy.io import write_json
from VesselEMPy.estimation import cross_sections, propose_exclusion


def main():
    started=time.perf_counter()
    root=Path(__file__).resolve().parents[1]
    rows=[]
    for name, circles in [('single_round_tube',[(24,24,6)]),
                          ('three_round_tubes',[(16,16,3),(16,48,6),(48,32,9)])]:
        shape=(48,72,72)
        z,y,x=np.indices(shape)
        mask=np.zeros(shape,dtype=np.uint8)
        for cy,cx,r in circles:
            mask |= (((y-cy)**2+(x-cx)**2<=r*r)&(z>=4)&(z<44)).astype(np.uint8)
        spacing=np.array([256.,256.,256.])
        distance=edt.edt(mask,anisotropy=spacing,parallel=1)
        grid=np.linspace(distance[mask>0].min(),distance.max(),512,dtype=np.float32)
        counts=[]
        for radius in grid:
            _,n=cc3d.connected_components(distance>radius,connectivity=26,return_N=True)
            counts.append(n)
        counts=np.asarray(counts)
        valid=counts>0
        weights=np.log1p(counts[valid].astype(float))
        repeated=np.repeat(grid[valid],np.maximum(1,np.round((weights-weights.min()+1)*5).astype(int)))
        fit=GaussianMixture(n_components=3,random_state=0).fit(repeated[:,None])
        smooth=gaussian_filter1d(np.log1p(counts),2)
        peaks,_=find_peaks(smooth,prominence=.15*np.ptp(smooth),distance=int(.05*512))
        samples=cross_sections(mask,distance,spacing,1024,42)
        rows.append(dict(name=name,nominal_radius_nm=[r*256 for _,_,r in circles],
            edt_max_nm=float(distance.max()), erosion_peak_radii_nm=grid[peaks].tolist(),
            forced_three_gmm_means_nm=sorted(fit.means_.ravel().tolist()),
            initial_component_count=int(counts[0]),
            accepted_cross_sections=len(samples),
            exclusion_proposal=propose_exclusion(samples,spacing,.9) if len(samples)>=16 else None,
            extent_ratio_quantiles=np.quantile(samples[:,5],[0,.5,.9,1]).tolist() if len(samples) else [],
            radius_grid_nm=grid.tolist(),component_counts=counts.tolist()))
    out=root/'outputs/radius-estimator-audit'
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'synthetic-caliber-probe.json',dict(cases=rows,elapsed_seconds=time.perf_counter()-started,
        caveat='Ideal digital tubes with flat ends, isotropic sampling. Tests interpretation of the estimator, not biological applicability. No skeletonization.'))
    for row in rows:
        print({k:v for k,v in row.items() if k not in ('radius_grid_nm','component_counts')})


if __name__=='__main__':
    main()
