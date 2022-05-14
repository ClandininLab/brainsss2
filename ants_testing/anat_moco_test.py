# test why motion correction is failing

import os
from brainsss2.motion_correction import get_mean_brain
import nibabel as nib
import numpy as np
import ants


basedir = '/data/brainsss/processed/fly_001/anat_0'

files = {'channel_1': 
    os.path.join(basedir, 'imaging/anatomy_channel_1.nii')}

ch1_meanbrain = get_mean_brain(files['channel_1'])

ch1_img = nib.load(files['channel_1'])
spacing = list(ch1_img.header.get_zooms())
print('initial spacing: ', spacing)

if len(spacing) > 4:
    spacing = spacing[:4]

direction = np.diag([-1., -1., 1., 1.])

chunk_start = 0
chunk_end = 2

chunkdata = ch1_img.dataobj[..., chunk_start:chunk_end].astype('float32')
print(spacing)
print(direction)
print(chunkdata.shape)

chunkdata_ants = ants.from_numpy(
    chunkdata,
    spacing=spacing,
    direction=direction)

print(chunkdata_ants.shape)
print(chunkdata_ants.spacing)
print(chunkdata_ants.direction)
print(ch1_meanbrain.shape)
print(ch1_meanbrain.spacing)
print(ch1_meanbrain.direction)
print(ch1_meanbrain.mean())

#mytx = ants.motion_correction(image=chunkdata_ants, fixed=ch1_meanbrain,
#    verbose=True, type_of_transform='SyN')

tpdata = ants.slice_image()
ants.registration()