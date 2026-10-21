import numpy as np
import pytest
import edt
from VesselEMPy.estimation import cross_sections, propose_exclusion


def test_round_tube_cross_sections_and_units():
    z,y,x = np.indices((72,25,25))
    mask = (((y-12)**2+(x-12)**2 <= 25) & (z>1) & (z<70)).astype(np.uint8)
    spacing = np.array([2.,1.,1.])
    distance = edt.edt(mask, anisotropy=spacing)
    samples = cross_sections(mask,distance,spacing,256,42)
    assert len(samples) >= 16
    assert np.all(samples[:,3] > 0)
    assert .7 < np.median(samples[:,5]) < 1.4
    proposal = propose_exclusion(samples,spacing,.9)
    scaled = samples.copy()
    scaled[:,3:5] *= 1000
    other = propose_exclusion(scaled,spacing*1000,.9)
    assert other['scale'] == proposal['scale']
    assert other['const'] == pytest.approx(proposal['const']*1000)


def test_no_unsupported_fallback_proposal():
    with pytest.raises(ValueError,match='Fewer than 16'):
        propose_exclusion(np.empty((0,6)),[1,1,1],.9)


def test_edt_physical_axis_order():
    mask = np.zeros((9,11,13),np.uint8)
    mask[2:7,2:9,2:11] = 1
    result = edt.edt(mask,anisotropy=(4,2,1))
    assert result[4,5,6] == pytest.approx(5)
