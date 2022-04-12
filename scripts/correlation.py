import os
import sys
import numpy as np
import argparse
import subprocess
import json
import h5py
import time
from scipy.ndimage import gaussian_filter1d
import nibabel as nib
import brainsss
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from logging_utils import setup_logging
import logging

def parse_args(input):
    parser = argparse.ArgumentParser(description='temporally highpass filter an hdf5 file')
    parser.add_argument('-d', '--dir', type=str,
        help='func directory to be analyzed', required=True)
    parser.add_argument('-f', '--file', type=str, help='file to process',
        default='moco/functional_channel_2_moco_zscore_highpass.h5')
    parser.add_argument('-b', '--behavior', default=2, type=int, help='behavior to analyze',
        choices=['dRotLabY', 'dRotLabZ'])
    parser.add_argument('-l', '--logdir', type=str, help='directory to save log file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('--fps', type=float, default=100, help='frame rate of fictrac camera')
    parser.add_argument('--resolution', type=float, default=10, help='resolution of fictrac data')

    args = parser.parse_args(input)
    return(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    setattr(args, 'dir', os.path.dirname(args.file))

    setup_logging(args, logtype='correlation')

    ### load brain timestamps ###
    timestamps = brainsss.load_timestamps(os.path.join(args.dir, 'imaging'))

    ### Load fictrac ###
    fictrac_raw = brainsss.load_fictrac(os.path.join(args.dir, 'fictrac'))
    expt_len = fictrac_raw.shape[0] / (fps * 1000) 
    
    behavior = args.behavior.replace('dRotLab', '')
    assert behavior in ['Y', 'Z'], 'behavior must be either Y or Z'

    ### Load brain ###
    full_load_path = os.path.join(args.dir, args.file)
    with h5py.File(full_load_path, 'r') as hf:
        brain = hf['data'][:] 
    
    ### Correlate ###
    logging.info("Performing Correlation on {}; behavior: {}".format(args.file, behavior))

    corr_brain = np.zeros((brain.dims[:3]))

    for z in range(brain.dims[2]):
        
        ### interpolate fictrac to match the timestamps of this slice
        printlog(F"{z}")
        fictrac_interp = brainsss.smooth_and_interp_fictrac(
            fictrac_raw, args.fps, args.resolution, expt_len, behavior, timestamps=timestamps, z=z)

        for i in range(256):
            for j in range(128):
                # nan to num should be taken care of in zscore, but checking here for some already processed brains
                if np.any(np.isnan(brain[i,j,z,:])):
                    printlog(F'warning found nan at x = {i}; y = {j}; z = {z}')
                    corr_brain[i,j,z] = 0
                elif len(np.unique(brain[i,j,z,:])) == 1:
                    printlog(F'warning found constant value at x = {i}; y = {j}; z = {z}')
                    corr_brain[i,j,z] = 0
                else:
                    corr_brain[i,j,z] = scipy.stats.pearsonr(fictrac_interp, brain[i,j,z,:])[0]

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    save_file = os.path.join(save_directory, 'corr_{}.nii'.format(behavior))
    nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
    printlog("Saved {}".format(save_file))
    save_maxproj_img(save_file)

def save_maxproj_img(file):
    brain = np.asarray(nib.load(file).get_data().squeeze(), dtype='float32')

    plt.figure(figsize=(10,4))
    plt.imshow(np.max(brain,axis=-1).T,cmap='gray')
    plt.axis('off')
    plt.colorbar()
    
    save_file = file[:-3] + 'png'
    plt.savefig(save_file, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main(json.loads(sys.argv[1]))