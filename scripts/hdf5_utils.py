# hdf5 utils
# pyright: reportMissingImports=false

import os
import h5py
import nibabel as nib
import numpy as np


def make_empty_h5(file, brain_dims, qform=None,
                  xyzt_units=None, zooms=None, stepsize=None):
    """for optimal speed, using chunk dim4 equal to dimension 4 of data chunks"""
    if stepsize is None:
        chunks = True
    else:
        chunks = (brain_dims[0], brain_dims[1], brain_dims[2], stepsize)

    if qform is None:
        qform = np.eye(4)
    else:
        assert qform.shape == (4, 4), 'affine must be 4 x 4'

    if zooms is None:
        zooms = np.ones(len(brain_dims))

    if xyzt_units is None:
        xyzt_units = ['mm', 'sec']

    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.mkdir(directory)

    with h5py.File(file, "w") as f:
        _ = f.create_dataset("data", brain_dims, dtype="float32", chunks=chunks)
        _ = f.create_dataset("qform", data=qform)
        _ = f.create_dataset("zooms", data=qform)
        _ = f.create_dataset("xyzt_units", data=qform)

    return directory, os.path.basename(file)


def get_chunk_boundaries(stepsize, n_timepoints):
    """get chunk boundaries"""
    if stepsize is None:
        stepsize = 100  # reasonable default
    chunk_starts = list(range(0, n_timepoints, stepsize))
    chunk_ends = list(range(stepsize, n_timepoints + stepsize, stepsize))
    chunk_ends = [x if x < n_timepoints else n_timepoints for x in chunk_ends]
    return list(zip(chunk_starts, chunk_ends))


def h5_to_nii(file):
    assert '.h5' in file, 'file type must be .h5'
    nii_savefile = file.replace('.h5', '.nii')
    assert nii_savefile != file, f'nii_savefile should be different from file: {nii_savefile}'
    print(f'loading data from h5 file: {file}')
    with h5py.File(file, 'r+') as h5_file:
        image_array = h5_file.get("data")[:].astype('float32')
        if 'affine' in h5_file:
            affine = h5_file['affine'][:]
            print('using affine from h5 file:', affine)
        else:
            affine = np.eye(len(image_array.shape))

    print(f'saving to nii file {nii_savefile}')
    nib.Nifti1Image(image_array, affine=affine).to_filename(nii_savefile)
