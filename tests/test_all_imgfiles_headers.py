# open all nii files within a directory tree
# and make sure that they have non-identity affine matrices

import os
import numpy as np
import nibabel as nib
import h5py


def test_nii_files():
    basedir = '/data/brainsss/processed/fly_001'
    for root, dirs, files in os.walk(basedir):
        for f in files:
            if f.endswith('.nii'):
                nii = os.path.join(root, f)
                print(f'checking: {nii}')
                img = nib.load(nii)
                print(img.affine)
                assert not np.allclose(img.affine, np.eye(4))


def test_h5_files():
    basedir = '/data/brainsss/processed/fly_001'
    for root, dirs, files in os.walk(basedir):
        for f in files:
            if f.endswith('.h5') and 'functional_channel' in f or 'anatomy_channel' in f:
                h5file = os.path.join(root, f)
                print(f'checking: {h5file}')
                try:
                    with h5py.File(h5file, 'r') as h5:
                        assert 'data' in h5
                        assert 'qform' in h5
                        assert 'zooms' in h5
                        assert 'xyzt_units' in h5
                except IOError:
                    print('problem loading file:', h5file)