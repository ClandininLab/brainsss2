# testing built-in ants method

import ants
import os 
import nibabel as nib
import numpy as np
from brainsss.utils import get_resolution
import pickle
from ants_utils import get_dataset_resolution, get_motion_parameters_from_transforms


if __name__ == "__main__":
    basedir = '/data/brainsss/processed/fly_001/func_0/imaging'

    print('reading full img')
    fullimg = nib.load(os.path.join(basedir, 'functional_channel_1.nii'))

    fullimg_data = fullimg.get_fdata(dtype='float32')

    start = 0
    end = 500

    partimg_data = fullimg_data[:, :, :, start:end]

    meanimg = ants.from_numpy(np.mean(fullimg_data, axis=3))

    partimg_ants = ants.from_numpy(partimg_data)

    print('running moco')
    transform_type =  'SyN' # 'Rigid' #

    use_ants_moco = True
    total_sigma=0
    flow_sigma=3

    warped_img = np.zeros(partimg_data.shape)

    mk_warped = True 

    if use_ants_moco:

        mytx = ants.motion_correction(image=partimg_ants, fixed=meanimg,
            verbose=True, type_of_transform=transform_type, 
            total_sigma=total_sigma, flow_sigma=flow_sigma)
        analysis_string = f'transform-{transform_type}_totalsigma-{total_sigma}_flowsigma-{flow_sigma}_start-{start}_end-{end}'
        mytx['motion_corrected'].to_filename(os.path.join(basedir, f'moco_{analysis_string}.nii'))

        transform_parameters = get_motion_parameters_from_transforms(mytx['motion_parameters'],
            get_dataset_resolution(basedir))

        
        if mk_warped:
            print('creating warped images')

            for i in range(partimg_data.shape[3]):
                tmpimg = ants.from_numpy(partimg_data[:, :, :, i])
                warped = ants.apply_transforms(fixed=meanimg, moving=tmpimg, 
                    transformlist=mytx['motion_parameters'][i])
                warped_img[:, :, :, i] = warped[:, :, :]

    else:
        # use standard ants registration
        print('registering and creating warped images')

        for i in range(partimg_data.shape[3]):
            tmpimg = ants.from_numpy(partimg_data[:, :, :, i])
            mytx = ants.registration(fixed=meanimg, moving=tmpimg,
                type_of_transform=transform_type, 
                total_sigma=total_sigma, flow_sigma=flow_sigma)
            mytx['warpedmovout'].to_filename(os.path.join(basedir, f'moco_manual_{transform_type}_idx{i}.nii'))
            warped_img[:, :, :, i] = mytx['warpedmovout'].numpy()
        
    ants.from_numpy(warped_img).to_filename(os.path.join(basedir, 'moco_manual_part1.nii'))

