# pyright: reportMissingImports=false

import sys
import os
import pytest
import numpy as np
import nibabel as nib
import h5py
from brainsss2.imgmath import imgmath  # noqa
from brainsss2.hdf5_utils import make_empty_h5, h5_to_nii  # noqa


data_shape = (4, 4, 4, 3)


@pytest.fixture
def qform():
    affine = np.eye(len(data_shape))
    affine[0, 0] = -1  # test non-identity affine
    return affine


@pytest.fixture
def zooms(qform):
    return(np.abs(np.diag(qform)))


@pytest.fixture
def dataset():
    np.random.seed(1)
    return(np.random.randint(0, 255, data_shape))


@pytest.fixture
def filled_h5_dataset(dataset, qform, zooms):
    h5_file = "/tmp/test_imgmean.h5"
    if os.path.exists(h5_file):
        os.remove(h5_file)
    # create a dataset
    with h5py.File(h5_file, "w") as f:
        f.create_dataset('data', data=dataset)
        f.create_dataset('qform', data=qform)
        f.create_dataset('zooms', data=zooms)
        f.create_dataset('xyzt_units', data=('mm', 'sec'))
    return(h5_file)


def test_h5file_smoke(filled_h5_dataset):
    assert filled_h5_dataset is not None


@pytest.fixture
def nii_file(dataset, qform, zooms):
    nii_file = "/tmp/test_imgmean.nii"
    if os.path.exists(nii_file):
        os.remove(nii_file)
    img = nib.Nifti1Image(dataset, qform)
    img.header.set_zooms(zooms)
    img.header.set_xyzt_units(xyz='mm', t='sec')
    nib.save(img, nii_file)
    return(nii_file)


def test_niifile_smoke(nii_file):
    assert nii_file is not None


def test_imgmean_nii_smoke(nii_file):
    imgmath(nii_file, 'mean', verbose=True)


def test_imgstd_nii_smoke(nii_file):
    imgmath(nii_file, 'std', verbose=True)


def test_tsnr_nii_smoke(nii_file):
    imgmath(nii_file, 'std', verbose=True)


def test_imgmean_h5_smoke(filled_h5_dataset):
    imgmath(filled_h5_dataset, 'mean', verbose=True)


def test_imgstd_h5_smoke(filled_h5_dataset):
    imgmath(filled_h5_dataset, 'std', verbose=True)


def test_imgmean_h5_to_nii(filled_h5_dataset, dataset):
    meanfile = imgmath(filled_h5_dataset, 'mean',
                       outfile_type='nii', verbose=True)
    assert os.path.exists(meanfile)
    img = nib.load(meanfile)
    assert img.shape == data_shape[:3]
    assert np.allclose(img.get_fdata(), np.mean(dataset, axis=-1))


def test_imgstd_h5_to_nii(filled_h5_dataset, dataset):
    stdfile = imgmath(filled_h5_dataset, 'std',
                      outfile_type='nii', verbose=True)
    assert os.path.exists(stdfile)
    img = nib.load(stdfile)
    assert img.shape == data_shape[:3]
    assert np.allclose(img.get_fdata(), np.std(dataset, axis=-1))


def test_tsnr_h5_to_nii(filled_h5_dataset, dataset):
    tsnrfile = imgmath(filled_h5_dataset, 'tsnr',
                       outfile_type='nii', verbose=True)
    assert os.path.exists(tsnrfile)
    img = nib.load(tsnrfile)
    assert img.shape == data_shape[:3]
    assert np.allclose(
        img.get_fdata(),
        np.mean(dataset, axis=-1) / np.std(dataset, axis=-1)
    )


def test_imgmean_nii_to_nii(nii_file, dataset):
    meanfile = imgmath(nii_file, 'mean',
        outfile_type='nii', verbose=True)
    assert os.path.exists(meanfile)
    img = nib.load(meanfile)
    assert img.shape == data_shape[:3]
    assert np.allclose(img.get_fdata(), np.mean(dataset, axis=-1))


def test_imgstd_nii_to_nii(nii_file, dataset):
    stdfile = imgmath(nii_file, 'std',
        outfile_type='nii', verbose=True)
    assert os.path.exists(stdfile)
    img = nib.load(stdfile)
    assert img.shape == data_shape[:3]
    assert np.allclose(img.get_fdata(), np.std(dataset, axis=-1))


def test_imgmean_h5_to_h5(filled_h5_dataset, dataset, qform):
    meanfile = imgmath(filled_h5_dataset, 'mean',
                       outfile_type='h5', verbose=True)
    assert os.path.exists(meanfile)
    with h5py.File(meanfile, "r") as f:
        assert 'data' in f
        assert 'qform' in f
        assert 'zooms' in f
        assert 'xyzt_units' in f
        assert f['data'].shape == data_shape[:3]
        assert np.allclose(f['data'][...], np.mean(dataset, axis=-1))
        assert np.allclose(f['qform'][...], qform)


def test_imgmean_nii_to_h5(nii_file, dataset, qform):
    meanfile = imgmath(nii_file, 'mean',
                       outfile_type='h5', verbose=True)
    assert os.path.exists(meanfile)
    with h5py.File(meanfile, "r") as f:
        assert 'data' in f
        assert 'qform' in f
        assert 'zooms' in f
        assert 'xyzt_units' in f
        assert f['data'].shape == data_shape[:3]
        assert np.allclose(f['data'][...], np.mean(dataset, axis=-1))
        assert np.allclose(f['qform'][...], qform)


if __name__ == "__main__":
    # create h5 dataset for testing
    qform = np.eye(len(data_shape))
    qform[0, 0] = -1 # test non-identity affine
    dataset = np.random.randint(0, 255, data_shape)
    zooms = np.abs(np.diag(qform))
    xyzt_units = ('mm', 'sec')
    h5_file = "/tmp/test_imgmean.h5"
    if os.path.exists(h5_file):
        os.remove(h5_file)
    # create a dataset
    with h5py.File(h5_file, "w") as f:
        f.create_dataset('data', data=dataset)
        f.create_dataset('qform', data=qform)
        f.create_dataset('zooms', data=zooms)
        f.create_dataset('xyzt_units', data=xyzt_units)

    filled_h5_dataset = h5_file
    meanfile = imgmath(filled_h5_dataset, 'mean', outfile_type='nii', verbose=True)
    assert os.path.exists(meanfile)
    img = nib.load(meanfile)
    assert img.shape == data_shape[:3]
    assert np.allclose(img.affine, qform)
    assert np.allclose(img.get_fdata(), np.mean(dataset, axis=-1))
