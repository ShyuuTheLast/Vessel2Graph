import numpy as np
import pytest
import kimimaro
from VesselEMPy.pipeline import crop_empty_background


@pytest.mark.parametrize('touch_face',[False,True])
def test_crop_preserves_kimimaro_geometry_and_boundary_targets(touch_face):
    mask=np.zeros((45,40,35),np.uint8)
    mask[0 if touch_face else 5:40,17:24,12:19]=1
    mask[20:27,17:34,12:19]=1
    cropped,offset=crop_empty_background(mask)
    assert cropped.size<mask.size
    spacing=np.array([320,256,256])
    opts=dict(anisotropy=spacing,dust_threshold=0,parallel=1,progress=False,
        fix_branching=True,fix_borders=True,
        teasar_params=dict(scale=2,const=1000,pdrf_exponent=8,pdrf_scale=16384,
            soma_detection_threshold=float('inf'),soma_acceptance_threshold=float('inf')))
    full=kimimaro.skeletonize(mask,**opts)[1]
    small=kimimaro.skeletonize(cropped,**opts)[1]
    np.testing.assert_array_equal(full.vertices,small.vertices+offset*spacing)
    np.testing.assert_array_equal(full.edges,small.edges)
    np.testing.assert_array_equal(full.radii,small.radii)
