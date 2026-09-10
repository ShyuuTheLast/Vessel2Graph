from copy import deepcopy
import json
import numpy as np
import h5py
import pytest
import yaml
from vessel2graph.calibration import calibrate, estimate_calibrated, exclusion_from_model
from vessel2graph.config import DEFAULTS
from vessel2graph.io import save_arrays


def setup_case(tmp_path):
    cfg = deepcopy(DEFAULTS)
    volume = tmp_path/'input.h5'
    with h5py.File(volume,'w') as f:
        a=np.zeros((9,9,9),np.uint8)
        a[2:7,3:6,3:6]=1
        a[0,0,0]=2  # Explicit label exclusion remains in force.
        f['main']=a
    cfg['input'].update(path=str(volume),voxel_size_nm=[2,1,1])
    cfg['output']['directory']=str(tmp_path/'run')
    ref=tmp_path/'reference'
    ref.mkdir()
    (ref/'resolved-config.yaml').write_text(yaml.safe_dump(cfg))
    save_arrays(ref/'graph.npz',vertices=np.array([[4.,4,4],[12.,4,4]]),
        radius=np.array([2.,2.]),edges=np.array([[0,1]]),object_id=np.ones(2,int))
    return cfg,ref


def test_model_scales_physical_lengths():
    model=dict(schema_version=1,method='reference_coverage_v1',scale=1.5,
        const_radius_multiplier=2.4,voxel_diagonal_multiplier=.5)
    a=exclusion_from_model(4,[2,1,1],model)
    b=exclusion_from_model(4000,[2000,1000,1000],model)
    assert b['scale']==a['scale']
    assert b['const']==pytest.approx(1000*a['const'])
    with pytest.raises(ValueError):
        exclusion_from_model(float('nan'),[1,1,1],model)


def test_calibration_and_application_without_reference(tmp_path,monkeypatch):
    cfg,ref=setup_case(tmp_path)
    target=tmp_path/'model.json'
    model=calibrate(cfg,ref,target)
    assert model['selected_voxels']==45
    # The intermediate z=4 center voxel is 4 units from either reference node,
    # so coverage needs at least 4 - 1.5*2 = 1 additive unit.
    assert model['training_required_const_nm']>=1
    assert model['fixed_teasar_context']['soma_detection_threshold']=='.inf'
    def forbidden(*args,**kwargs):
        raise AssertionError('Production estimation must not load a reference graph')
    monkeypatch.setattr('vessel2graph.calibration.load_arrays',forbidden)
    (ref/'graph.npz').unlink()
    result=estimate_calibrated(cfg,target)
    assert result['selected_voxels']==45
    assert result['evaluation_role']=='training_dataset'
    candidate=yaml.safe_load(open(result['candidate_config']))
    assert candidate['skeleton']['source'] is None
    assert candidate['skeleton']['teasar_params']['const']==pytest.approx(
        model['training_required_const_nm']+.5*np.sqrt(6))
    assert candidate['input']['labels']==[1]
    assert candidate['skeleton']['teasar_params']['pdrf_exponent']==cfg['skeleton']['teasar_params']['pdrf_exponent']


def test_context_mismatch_rejected_before_measurement(tmp_path,monkeypatch):
    cfg,ref=setup_case(tmp_path)
    target=tmp_path/'model.json'
    calibrate(cfg,ref,target)
    cfg['skeleton']['teasar_params']['pdrf_exponent']=8
    def forbidden(*args,**kwargs):
        raise AssertionError('Should reject before EDT')
    monkeypatch.setattr('vessel2graph.calibration.measure',forbidden)
    with pytest.raises(ValueError,match='pdrf_exponent'):
        estimate_calibrated(cfg,target)
