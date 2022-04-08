import os
import sys
import json
from time import sleep
import datetime
import brainsss
import numpy as np
import nibabel as nib
import h5py
import argparse
from pathlib import Path

def parse_args(input):
    parser = argparse.ArgumentParser(description='run bleaching qc')
    parser.add_argument('-d', '--dir', type=str, 
        help='directory containing func or anat data', required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    args = parser.parse_args(input)
    return(args)


def make_mean_brain(args):
    files = [f.as_posix() for f in Path(args.dir).glob('*_channel*.nii')]
    if args.verbose:
        print('found files:')
        print(files)

    for file in files:
        try:
            ### make mean ###
            full_path = os.path.join(args.dir, file)
            if full_path.endswith('.nii'):
                brain = np.asarray(nib.load(full_path).get_fdata(), dtype='uint16')
            elif full_path.endswith('.h5'):
                with h5py.File(full_path, 'r') as hf:
                    brain = np.asarray(hf['data'][:], dtype='uint16')
            meanbrain = np.mean(brain, axis=-1)

            ### Save ###
            save_file = os.path.join(args.dir, file[:-4] + '_mean.nii')
            if args.verbose:
                print(f'Saving to {save_file}')
            aff = np.eye(4)
            img = nib.Nifti1Image(meanbrain, aff)
            img.to_filename(save_file)

            # assumes specific file naming...
            fly_func_str = ('|').join(args.dir.split('/')[-3:-1])
            fly_print = args.dir.split('/')[-3]
            func_print = args.dir.split('/')[-2]
            #printlog(f"COMPLETE | {fly_func_str} | {file} | {brain.shape} --> {meanbrain.shape}")
            print(F"meanbrn | COMPLETED | {fly_print} | {func_print} | {file} | {brain.shape} ===> {meanbrain.shape}")
            print(brain.shape[-1]) ### IMPORTANT: for communication to main
        except FileNotFoundError:
            print(F"Not found (skipping){file:.>{width-20}}")
            #printlog(f'{file} not found.')

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    #TODO: Fix logging
    # logfile = args['logfile']
    width = 120

    make_mean_brain(args)
