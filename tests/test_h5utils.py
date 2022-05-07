# pyright: reportMissingImports=false

import os
import pytest
import numpy as np
import nibabel as nib
import h5py
from brainsss2.hdf5_utils import make_empty_h5, h5_to_nii


data_shape = (4, 4, 4, 3)

@pytest.fixture
def affine():
    affine = np.eye(len(data_shape))
    affine[0, 0] = -1 # test non-identity affine
    return affine


@pytest.fixture
def empty_h5_dataset():
    h5_file_empty = "/tmp/test_h5_creation_empty.h5"
    if os.path.exists(h5_file_empty):
        os.remove(h5_file_empty)
    _ = make_empty_h5(h5_file_empty, data_shape)
    return(h5_file_empty)


@pytest.fixture
def filled_h5_dataset(affine):
    h5_file = "/tmp/test_h5_creation.h5"
    if os.path.exists(h5_file):
        os.remove(h5_file)
    # create a dataset
    with h5py.File(h5_file, "w") as f:
        f.create_dataset('data', data=np.random.randint(0, 255, data_shape))
        f.create_dataset('affine', data=affine)
    return(h5_file)


def test_empty_h5_creation(empty_h5_dataset):
    with h5py.File(empty_h5_dataset, "r") as f:
        assert 'data' in f
        assert 'qform' in f
        assert f['data'].shape == data_shape
        assert f['qform'].shape == (len(data_shape), len(data_shape))


def test_filled_h5_creation(filled_h5_dataset):
    with h5py.File(filled_h5_dataset, "r") as f:
        assert 'data' in f
        assert 'affine' in f
        assert f['data'].shape == data_shape
        assert f['affine'].shape == (len(data_shape), len(data_shape))


def test_h5_to_nii(filled_h5_dataset, affine):
    h5_to_nii(filled_h5_dataset)
    assert os.path.exists(filled_h5_dataset.replace('.h5', '.nii'))
    img = nib.load(filled_h5_dataset.replace('.h5', '.nii'))
    assert img.shape == data_shape
    assert np.allclose(img.affine, affine)

