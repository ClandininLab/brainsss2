import os
import sys
import numpy as np
import pandas as pd
import argparse
import nibabel as nib
import h5py
import json
import ants
import pyfiglet
import matplotlib.pyplot as plt
from time import time
from time import strftime
from pathlib import Path
import logging
import brainsss
import shutil
from make_mean_brain import make_mean_brain
import datetime
import git
from ants_utils import get_motion_parameters_from_transforms, get_dataset_resolution
from hdf5_utils import make_empty_h5


def parse_args(input):
    parser = argparse.ArgumentParser(description='run motion correction')
    parser.add_argument('-d', '--dir', type=str, 
        help='directory containing func or anat data', required=True)
    parser.add_argument('-l', '--logdir', type=str, help='directory to save log file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-t', '--type_of_transform', type=str, default='SyN', 
        help='type of transform to use')
    parser.add_argument('-i', '--interpolation_method', type=str, default='linear')
    parser.add_argument('--output_format', type=str, choices=['h5', 'nii'], 
        default='h5', help='output format for registered image data')
    parser.add_argument('--flow_sigma', type=int, default=3, 
        help='flow sigma for registration - higher sigma focuses on coarser features')
    parser.add_argument('--total_sigma', type=int, default=0, 
        help='total sigma for registration - higher values will restrict the amount of deformation allowed')
    parser.add_argument('--meanbrain_n_frames', type=int, default=None, 
        help='number of frames to average over when computing mean/fixed brain')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files')
    parser.add_argument('--save_nii', action='store_true', help='save nifti files')

    args = parser.parse_args(input)
    return(args)


def load_data(args):
    """determine directory type and load data"""
    files = [f.as_posix() for f in Path(args.dir).glob('*_channel*.nii') if 'mean' not in f.stem]
    files.sort()
    if args.verbose:
        logging.info(f'Data files: {files}')
    assert len(files) in [1, 2], 'Must have exactly one or two data files in directory'
    assert 'channel_1' in files[0], 'data for first channel must be named channel_1'
    if len(files) == 1:
        logging.info('Only one channel found, no mirror will be used')
    if len(files) == 2:
        assert 'channel_2' in files[1], 'data for second channel must be named channel_2'

    # NOTE: should probably be using "scantype" thbroughout instead of "datatype" 
    # since the latter is quite confusing as each scan includes both func and anat data
    if 'functional' in files[0]:
        scantype = 'func'
    elif 'anatomy' in files[0]:
        scantype = 'anat'
    else:
        raise ValueError('Could not determine scan type')
    if args.verbose:
        logging.info(f'Scan type: {scantype}')

    setattr(args, 'scantype', scantype) 

    files_dict = {}
    files_dict['channel_1'] = files[0]
    files_dict['channel_2'] = files[1] if len(files) == 2 else None
    return(files_dict, args)


def get_current_git_hash(return_length=8):
    script = os.path.realpath(__file__)
    repo = git.Repo(path=script, search_parent_directories=True)
    return(repo.head.object.hexsha[:return_length])


# TODO: this is modified from preprocess.py - should be refactored to create a common function
def setup_logging(args, logtype='moco'):
    if args.logdir is None:  # this shouldn't happen, but check just in case
        args.logdir = os.path.join(args.dir, 'logs')
    args.logdir = os.path.realpath(args.logdir)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    #  RP: use os.path.join rather than combining strings
    setattr(args, 'logfile', os.path.join(args.logdir, strftime(f"{logtype}_%Y%m%d-%H%M%S.txt")))

    #  RP: replace custom code with logging.basicConfig
    logging_handlers = [logging.FileHandler(args.logfile)]
    if args.verbose:
        #  use logging.StreamHandler to echo log messages to stdout
        logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(handlers=logging_handlers, level=logging.INFO,
        format='%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    title = pyfiglet.figlet_format("Brainsss", font="doom")
    title_shifted = ('\n').join([' ' * 42 + line for line in title.split('\n')][:-2])
    logging.info(title_shifted)
    logging.info(f'jobs started: {datetime.datetime.now()}')
    setattr(args, 'git_hash', get_current_git_hash())
    logging.info(f'git commit: {args.git_hash}')
    if args.verbose:
        logging.info(f'logging enabled: {args.logfile}')

    logging.info('\n\nArguments:')
    args_dict = vars(args)
    for key, value in args_dict.items():
        logging.info(f'{key}: {value}')
    logging.info('\n')
    return(args)


def set_stepsize(args, scantype_stepsize_dict=None):
    if scantype_stepsize_dict is None:
        scantype_stepsize_dict = {'func': 100, 'anat': 5}
    setattr(args, 'stepsize', scantype_stepsize_dict[args.scantype])
    return(args)


def get_mean_brain(args, file):
    """get mean brain for channel 1"""
    if args.verbose:
        print('Getting mean brain')
    assert 'channel_1' in file, 'file must be channel_1'
    meanbrain_file = file.replace('.nii', '_mean.nii')
    if not os.path.exists(meanbrain_file):
        if args.verbose:
            print(f'making mean brain for {meanbrain_file}')
        make_mean_brain(args)
    img = nib.load(meanbrain_file)
    meanbrain = img.get_fdata(dtype='float32')
    meanbrain_ants = ants.from_numpy(meanbrain)
    return(meanbrain_ants)


def setup_h5_datasets(args, files):
    """Make Empty MOCO files that will be filled chunk by chunk"""

    h5_file_names = {
        'channel_1': os.path.basename(files['channel_1']).replace(
            '.nii', '_moco.h5'),
        'channel_2': None}
    if args.verbose:
        logging.info(f'Creating channel_1 h5 file: {h5_file_names["channel_1"]}')

    # full paths to file names
    h5_files = {
        'channel_1': None,
        'channel_2': None}

    brain_dims = nib.load(files['channel_1']).shape
    moco_dir, h5_files['channel_1'] = make_empty_h5(
        args.dir, h5_file_names['channel_1'], brain_dims, stepsize=args.stepsize)
    logging.info(f"Created empty hdf5 file: {h5_file_names['channel_1']}")

    if 'channel_2' in files:
        h5_file_names['channel_2'] = os.path.basename(files['channel_2']).replace(
            '.nii', '_moco.h5')
        moco_dir, h5_files['channel_2'] = make_empty_h5(
            args.dir, h5_file_names['channel_2'], brain_dims, stepsize=args.stepsize)
        logging.info(f"Created empty hdf5 file: {h5_file_names['channel_2']}")
    return h5_files


def create_moco_output_dir(args):
    setattr(args, 'moco_output_dir', os.path.join(args.dir, 'moco'))

    if os.path.exists(args.moco_output_dir) and args.overwrite:
        if args.verbose:
            print('removing existing moco output directory')
        shutil.rmtree(args.moco_output_dir)
    elif os.path.exists(args.moco_output_dir) and not args.overwrite:
        raise ValueError(f'{args.moco_output_dir} already exists, use --overwrite to overwrite')

    if not os.path.exists(args.moco_output_dir):
        os.mkdir(args.moco_output_dir)
    if args.verbose:
        logging.info(f'Moco output directory: {args.moco_output_dir}')
    return(args)


def get_chunk_boundaries(args, n_timepoints):
    """get chunk boundaries"""
    chunk_starts = list(range(0, n_timepoints, args.stepsize))
    chunk_ends = list(range(
        args.stepsize, n_timepoints + args.stepsize, args.stepsize))
    chunk_ends = [x if x < n_timepoints else n_timepoints for x in chunk_ends]
    return(list(zip(chunk_starts, chunk_ends)))


def apply_moco_parameters_to_channel_2(args, files,
                                       h5_files, transform_files):
    """Apply moco parameters to channel 2"""
    if args.verbose:
        logging.info('Applying moco parameters to channel 2')
    assert 'channel_2' in files, 'files must include channel_2'

    # load ch1 image to get dimensions for chunking
    ch2_img = nib.load(files['channel_2'])
    n_timepoints = ch2_img.shape[-1]
    # assert n_timepoints == len(transform_files), 'number of transform files must match number of timepoints'

    ch1_meanbrain = get_mean_brain(args, files['channel_1'])

    # load full data
    ch2_data = ch2_img.get_fdata(dtype='float32')
    print('ch1_data.shape:', ch2_data.shape)

    # overwrite existing data in place to prevent need for additional memory
    for timepoint in range(n_timepoints):
        # load transform
        try:
            transform = transform_files[timepoint]
        except IndexError:
            logging.warning(f'No transform file for timepoint {timepoint}')
            continue
        # apply transform
        result = ants.apply_transforms(
            fixed=ch1_meanbrain,
            moving=ants.from_numpy(ch2_data[..., timepoint]), 
            transformlist=transform,
            interpolator=args.interpolation_method)
        ch2_data[..., timepoint] = result.numpy()
    # save data
    if args.verbose:
        logging.info('Saving channel 2 data')

    # setup chunking into smaller parts (for memory)
    chunk_boundaries = get_chunk_boundaries(args, n_timepoints)
    for i, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        with h5py.File(h5_files['channel_2'], 'a') as f:
            f['data'][..., chunk_start:chunk_end] = ch2_data[..., chunk_start:chunk_end]


def run_motion_correction(args, files, h5_files):
    """Run motion correction on tdTomato channel (1)"""

    if args.verbose:
        logging.info('Running motion correction')
    
    # load ch1 image to get dimensions for chunking
    ch1_img = nib.load(files['channel_1'])
    n_timepoints = ch1_img.shape[-1]

    # setup chunking into smaller parts (for memory)
    chunk_boundaries = get_chunk_boundaries(args, n_timepoints)

    # load full data
    ch1_data = ch1_img.get_fdata(dtype='float32')
    print('ch1_data.shape:', ch1_data.shape)

    # NB: need to make sure that the data is in the correct orientation
    # (i.e. direction of mean brain and chunkdata must be identical)
    ch1_meanbrain = get_mean_brain(args, files['channel_1'])
    print('made meanbrain')

    motion_parameters = None
    transform_files = []
    # loop through chunks
    for i, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        if args.verbose:
            logging.info('processing chunk {} of {}'.format(i + 1, len(chunk_boundaries)))
        # get chunk data
        chunkdata = ch1_data[..., chunk_start:chunk_end]
        chunkdata_ants = ants.from_numpy(chunkdata)

        # run moco on chunk
        mytx = ants.motion_correction(image=chunkdata_ants, fixed=ch1_meanbrain,
            verbose=args.verbose, type_of_transform=args.type_of_transform, 
            total_sigma=args.total_sigma, flow_sigma=args.flow_sigma)
        transform_files = transform_files + mytx['motion_parameters']

        # extract rigid body transform parameters (translation/rotation)
        if motion_parameters is None:
            motion_parameters = get_motion_parameters_from_transforms(
                mytx['motion_parameters'],
                get_dataset_resolution(args.dir))[1]
        else:
            motion_parameters = np.vstack((motion_parameters,
                get_motion_parameters_from_transforms(
                    mytx['motion_parameters'],
                    get_dataset_resolution(args.dir))[1]))

        # save results from chunk
        if args.verbose:
            logging.info('saving chunk {} of {}'.format(i + 1, len(chunk_boundaries)))
        with h5py.File(h5_files['channel_1'], 'a') as f:
            f['data'][..., chunk_start:chunk_end] = mytx['motion_corrected'].numpy()

    return(transform_files, motion_parameters)


def save_motion_parameters(args, motion_parameters):
    moco_dir = os.path.join(args.dir, 'moco')
    assert os.path.exists(moco_dir), 'something went terribly wrong, moco dir does not exist'
    motion_df = pd.DataFrame(motion_parameters, columns=['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
    motion_file = os.path.join(moco_dir, 'motion_parameters.csv')
    motion_df.to_csv(motion_file, index=False)
    return(motion_file)


def save_motcorr_settings_to_json(args, files, h5_files, nii_files=None):
    moco_dir = os.path.join(args.dir, 'moco')
    assert os.path.exists(moco_dir), 'something went terribly wrong, moco dir does not exist'
    args_dict = vars(args)
    args_dict['files'] = files
    args_dict['h5_files'] = h5_files
    if args.save_nii:
        args_dict['nii_files'] = nii_files
    with open(os.path.join(moco_dir, 'moco_settings.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)


def moco_plot(args, motion_file):
    """Make motion correction plot"""

    motion_parameters = pd.read_csv(motion_file)

    # Get voxel resolution for figure
    x_res, y_res, z_res = get_dataset_resolution(args.dir)

    moco_dir = os.path.join(args.dir, 'moco')
    assert os.path.exists(moco_dir), 'something went terribly wrong, moco dir does not exist'

    # Save figure of motion over time
    save_file = os.path.join(moco_dir, 'motion_correction.png')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(motion_parameters.iloc[:, :3])
    plt.legend(labels=list(motion_parameters.columns[:3]), loc='upper right')
    plt.title(f"{'/'.join(args.dir.split('/')[-3:])}: Translation")
    plt.xlabel('Timepoint')
    plt.ylabel('Translation (microns)')
    plt.subplot(1, 2, 2)
    plt.plot(motion_parameters.iloc[:, 3:])
    plt.legend(labels=list(motion_parameters.columns[ 3:]), loc='upper right')
    plt.title(f"{'/'.join(args.dir.split('/')[-3:])}: Rotation")
    plt.xlabel('Timepoint')
    plt.ylabel('Rotation (degrees)')

    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    return(None)


def h5_to_nii(h5_path):
    nii_savefile = h5_path.replace('h5', 'nii')
    with h5py.File(h5_path, 'r+') as h5_file:
        image_array = h5_file.get("data")[:].astype('uint16')

    nifti1_limit = (2**16 / 2)
    if np.any(np.array(image_array.shape) >= nifti1_limit):  # Need to save as nifti2
        nib.save(nib.Nifti2Image(image_array, np.eye(4)), nii_savefile)
    else:  # Nifti1 is OK
        nib.save(nib.Nifti1Image(image_array, np.eye(4)), nii_savefile)

    return nii_savefile


def save_nii(args, h5_files):
    """save moco data to nifti
    - reuse header info from existing nifti files"""
    for channel, h5_file in h5_files.items():
        if args.verbose:
            logging.info(f'converting {h5_file} to nifti')
        _ = h5_to_nii(h5_file)
    return(None)


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    args = setup_logging(args)

    files, args = load_data(args)

    if args.verbose:
        logging.info(files)

    args = create_moco_output_dir(args)

    args = set_stepsize(args)

    h5_files = setup_h5_datasets(args, files)

    transform_files, motion_parameters = run_motion_correction(args, files, h5_files)

    if 'channel_2' in files:
        apply_moco_parameters_to_channel_2(args, files, h5_files, transform_files)
    
    save_motcorr_settings_to_json(args)

    motion_file = save_motion_parameters(args, motion_parameters)

    make_moco_plot(args, motion_file)

    if args.save_nii:
        save_nii(args, h5_files)

    logging.info(f'Motion correction complete: {datetime.datetime.now()}')
