import numpy as np
import pytest
import h5py
import yaml
from copy import deepcopy
from vessel2graph.config import DEFAULTS
from vessel2graph.volume_estimation import estimate_volume
from vessel2graph.volume_estimation import prepare,derive_parameters
from vessel2graph.volume_estimation import close_physical


def test_closing_seals_exterior_channel_and_preserves_crop_edges():
    mask=np.ones((11,11,11),np.uint8)
    mask[:6,5,5]=0
    before=mask.copy()
    filled,n=prepare(mask,True)
    assert n==0
    closed,audit=close_physical(mask,[320,256,256],320)
    assert closed[4,5,5]==1
    assert np.all(closed[mask!=0])
    assert np.array_equal(mask,before)
    assert audit['footprint_voxels']==7
    assert audit['removed_voxels']==0
    unchanged,_=close_physical(mask,[1280,256,256],200)
    assert np.array_equal(unchanged,mask)


def test_estimator_rejects_closing_that_merges_components(tmp_path):
    mask=np.zeros((15,15,15),np.uint8)
    mask[2:7,2:13,2:13]=1
    mask[8:13,2:13,2:13]=1
    path=tmp_path/'split.h5'
    with h5py.File(path,'w') as f:
        f['main']=mask
    cfg=deepcopy(DEFAULTS)
    cfg['input'].update(path=str(path),voxel_size_nm=[1,1,1])
    cfg['output']['directory']=str(tmp_path/'run')
    with pytest.raises(ValueError,match='merge input components'):
        estimate_volume(cfg,closing_radius_nm=1)


def samples():
    radius=np.linspace(1200,4000,60)
    return np.column_stack([np.zeros((60,3)),radius,1.5*radius,
        np.full(60,1.5),.5*radius,np.full(60,3.)])


def test_fill_is_enclosed_only_and_does_not_mutate_input():
    mask=np.zeros((11,11,11),np.uint8)
    mask[2:9,2:9,2:9]=1
    mask[4:7,4:7,4:7]=0
    before=mask.copy()
    result,n=prepare(mask,True)
    assert n==27
    assert result[5,5,5]==1
    assert result[0,0,0]==0
    assert np.array_equal(mask,before)
    mask[2:5,5,5]=0  # Cavity connected to exterior must stay open.
    result,n=prepare(mask,True)
    assert n==0


def test_derived_parameters_have_geometry_and_penalty_constraints():
    s=samples()
    proposal,details=derive_parameters(s,4500,np.array([320,256,256]))
    assert proposal['scale']==pytest.approx(1.5)
    assert proposal['const']==pytest.approx(np.quantile(s[:,4],.9)+np.linalg.norm([320,256,256])/2)
    final=details['centering_survey'][-1]
    assert min(final['lower_decile_contrast_by_radius_stratum'])>=4
    assert proposal['pdrf_scale']==final['required_amplitude']
    assert proposal['pdrf_scale']<=2**24


def test_no_silent_fallback_for_missing_or_uncentered_profiles():
    with pytest.raises(ValueError,match='Fewer than 16'):
        derive_parameters(samples()[:10],4500,[320,256,256])
    s=samples();s[:,6]=s[:,3]
    with pytest.raises(ValueError,match='No numerically bounded'):
        derive_parameters(s,4500,[320,256,256])


def test_end_to_end_uses_binary_input_not_existing_parameters_or_skeleton(tmp_path,monkeypatch):
    z,y,x=np.indices((48,25,25))
    mask=(((y-12)**2+(x-12)**2<=25)&(z>=3)&(z<45)).astype(np.uint8)
    mask[0,0,0]=2
    path=tmp_path/'mask.h5'
    with h5py.File(path,'w') as f:
        f['main']=mask
    cfg=deepcopy(DEFAULTS)
    cfg['input'].update(path=str(path),voxel_size_nm=[320,256,256])
    cfg['output']['directory']=str(tmp_path/'run')
    cfg['skeleton']['source']=str(tmp_path/'does-not-exist.npz')
    for key in ('scale','const','pdrf_scale','pdrf_exponent'):
        cfg['skeleton']['teasar_params'][key]=999999
    def forbidden(*args,**kwargs):
        raise AssertionError('No skeleton artifact may be loaded by estimator')
    monkeypatch.setattr('vessel2graph.io.load_arrays',forbidden)
    result=estimate_volume(cfg,fill_holes=True)
    assert result['accepted_profiles']>=16
    assert all(v!=999999 for v in result['proposal'].values())
    candidate=yaml.safe_load(open(result['candidate_config']))
    assert candidate['skeleton']['source'] is None
    assert candidate['skeleton']['teasar_params']['max_paths'] is None
    assert candidate['skeleton']['teasar_params']['soma_detection_threshold']==float('inf')
    with h5py.File(candidate['input']['path']) as f:
        assert f['main'][0,0,0]==0
