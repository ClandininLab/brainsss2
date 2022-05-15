# test why motion correction is failing

import os
import nibabel as nib
import numpy as np
import ants

basedir = '/data/brainsss/processed/fly_001/anat_0'

files = {'channel_1': 
    os.path.join(basedir, 'imaging/anatomy_channel_1.nii')}

def resample_img(file, res, newfile=None):
    """resample image to new isotropic resolution"""
    orig_img = nib.load(file)
    if newfile is None:
        newfile = file.replace('.nii', f'_res-{res:.1f}mu.nii')
    assert newfile != file, 'newfile is same as old file'

    new_res = [res for i in range(3)]

    new_img = None

    for i in range(orig_img.shape[-1]):
        print(i)
        tp_ants = ants.from_numpy(orig_img.dataobj[..., i].astype('float32'),
            spacing = orig_img.header.get_zooms()[:3])
        tp_resampled = ants.resample_image(tp_ants, new_res, interp_type=3)
        if new_img is None:
            newshape = list(tp_resampled.shape) + [orig_img.shape[-1]]
            new_affine = np.diag(new_res + [1])
            new_img = nib.Nifti1Image(np.zeros(newshape),
                affine = new_affine)
        new_img.dataobj[..., i] = tp_resampled.numpy()

    new_img.header.set_zooms(new_res + [orig_img.header.get_zooms()[-1]])
    new_img.header.set_qform(new_affine)
    new_img.header.set_sform(new_affine)
    new_img.header.set_xyzt_units(xyz='mm', t='sec')

    new_img.to_filename(newfile)
    return(newfile)

f = resample_img(files['channel_1'], 2.0)