# note - this is not a real test as it requires a specific filesystem
# pyright: reportMissingImports=false

import sys
import os
import pytest
import numpy as np
import nibabel as nib
sys.path.append("../brainsss")
sys.path.append("../scripts")
from fly_builder import copy_nifti_file # noqa


@pytest.fixture
def data():
    return(np.random.randint(0, 255, (4, 4, 4, 3)))


@pytest.fixture
def pixdims():
    return np.array([.05, .05, .03, 1])


@pytest.fixture
def qform(pixdims):
    qform = np.eye(4)
    qform[np.diag_indices_from(qform)] = pixdims
    return(qform)


def test_copy_nifti_file(qform, pixdims, data):
    target='/data/brainsss/processed/fly_001/anat_0/imaging/anatomy_channel_1.nii'
    if os.path.exists(target):
        os.remove(target)
    source = '/data/brainsss/imports/20220329/fly_2/anat_0/TSeries-12172018-1322-009/TSeries-12172018-1322-009_channel_1.nii'

    copy_nifti_file(source, target)
    assert os.path.exists(target)
    targetimg = nib.load(target)
    sourceimg = nib.load(source)
    print('affines')
    print(sourceimg.affine)
    print(targetimg.affine)
    print('qforms')
    print(sourceimg.header.get_qform())
    print(targetimg.header.get_qform())
    desired_qform = np.array(
        [[6.52800314e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 6.52786519e-04, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000005e-03, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    )
    assert np.allclose(desired_qform, targetimg.header.get_qform())
