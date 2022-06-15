
import nibabel
from pathlib import Path
import numpy as np
import os

atlasdir = '/Users/poldrack/data_unsynced/brainsss/flydata/atlas'

atlasfiles = ['20220301_luke_2_jfrc_affine_zflip_2umiso.nii']

for file in atlasfiles:
    file = os.path.join(atlasdir, file)
    print(file)
    outfile = file.replace('.nii', '_fixed.nii')
    assert file != outfile, f'outfile should be different from file: {outfile}'
    img = nibabel.load(file)
    resolution = [2, 2, 2, 1]
    img.header.set_zooms(resolution[:3])
    img.header.set_xyzt_units(xyz='mm')
    img.header.set_qform(np.diag(np.array(resolution)))
    img.header.set_sform(np.diag(np.array(resolution)))
    newimg = nibabel.Nifti1Image(img.get_fdata(),
        img.header.get_sform(), img.header)
    newimg.to_filename(outfile)
