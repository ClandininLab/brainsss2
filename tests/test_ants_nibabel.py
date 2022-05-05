# testing movement between ants and nibabel
# to make sure the coordinates are the same


import os
import nibabel as nib
import ants
import numpy as np


if __name__ == "__main__":
    data = np.random.rand(10, 10, 10, 10)
    resolution = [-3, -3, -5, 1]
    affine = np.diag(resolution)
    img = nib.Nifti1Image(data, affine=affine)
    img.header.set_zooms(np.abs(resolution))
    img.to_filename('/tmp/test.nii')

    antsimg = ants.image_read('/tmp/test.nii')
    antsimg.to_filename('/tmp/test_ants.nii')

    antsimg2 = ants.from_nibabel(img)
    antsimg.to_filename('/tmp/test_ants_from_nib.nii')
