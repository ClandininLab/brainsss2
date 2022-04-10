# testing built-in ants method

import ants
import os 
import nibabel as nib
import numpy as np

basedir = '/data/brainsss/processed/fly_001/func_0/imaging'

print('reading full img')
fullimg = nib.load(os.path.join(basedir, 'functional_channel_1.nii'))

fullimg_data = fullimg.get_fdata(dtype='float32')

start = 0
end = 100

partimg_data = fullimg_data[:, :, :, start:end]

meanimg = ants.image_read(os.path.join(basedir, 'functional_channel_1_mean.nii'))

partimg_ants = ants.from_numpy(partimg_data)

print('running moco')
transform_type = 'Rigid' # 'SyN'
mytx = ants.motion_correction(image=partimg_ants, fixed=meanimg,
    verbose=True, type_of_transform=transform_type, total_sigma=5, flow_sigma=5)

mytx['motion_corrected'].to_filename(os.path.join(basedir, 'motion_corrected_ants_part1.nii'))

print('creating warped images')
warped_img = np.zeros(partimg_data.shape)

for i in range(partimg_data.shape[3]):
    tmpimg = ants.from_numpy(partimg_data[:, :, :, i])
    warped = ants.apply_transforms(fixed=meanimg, moving=tmpimg, 
        transformlist=mytx['motion_parameters'][i])
    warped_img[:, :, :, i] = warped[:, :, :]

ants.from_numpy(warped_img).to_filename(os.path.join(basedir, 'moco_manual_part1.nii'))