# pyright: reportMissingImports=false

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import h5py
import json
import ants
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import datetime
import nilearn.image
from ants_utils import get_motion_parameters_from_transforms, get_dataset_resolution
from hdf5_utils import make_empty_h5, get_chunk_boundaries
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, os.path.realpath("../brainsss"))
sys.path.insert(0, os.path.realpath("../brainsss/scripts"))
from argparse_utils import get_base_parser, add_moco_arguments # noqa
from logging_utils import setup_logging # noqa


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('moco')

    parser = add_moco_arguments(parser)

    # need to add this manually to procesing steps in order to make required
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        help='func directory',
        required=True)

    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()

    return args


def load_data(args):
    """determine directory type and load data"""
    if args.dir.split('/')[-1] != 'imaging':
        datadir = os.path.join(args.dir, 'imaging')
    logging.info(f'Loading data from {datadir}')
    files = [f.as_posix() for f in Path(datadir).glob('*_channel*.nii') if 'mean' not in f.stem]
    files.sort()
    logging.info(f'Data files: {files}')
    assert len(files) in {1, 2}, 'Must have exactly one or two data files in directory'

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
    logging.info(f'Scan type: {scantype}')

    setattr(args, 'scantype', scantype)

    files_dict = {
        'channel_1': files[0],
        'channel_2': files[1] if len(files) == 2 else None
    }

    return(files_dict, args)


def set_stepsize(args, scantype_stepsize_dict=None):
    if scantype_stepsize_dict is None:
        scantype_stepsize_dict = {'func': 50, 'anat': 5}
    setattr(args, 'stepsize', scantype_stepsize_dict[args.scantype])
    return(args)


def get_mean_brain(args, file):
    """get mean brain for channel 1"""
    assert 'channel_1' in file, 'file must be channel_1'
    meanbrain_file = file.replace('.nii', '_mean.nii')
    if not os.path.exists(meanbrain_file):
        logging.info(f'making mean brain for {meanbrain_file}')
        nilearn.image.mean_img(file).to_filename(meanbrain_file)
    img = nib.load(meanbrain_file)
    meanbrain = img.get_fdata(dtype='float32')
    return ants.from_numpy(meanbrain)


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
        args.moco_output_dir, h5_file_names['channel_1'], brain_dims, stepsize=args.stepsize)
    logging.info(f"Created empty hdf5 file: {h5_file_names['channel_1']}")

    if 'channel_2' in files:
        h5_file_names['channel_2'] = os.path.basename(files['channel_2']).replace(
            '.nii', '_moco.h5')
        moco_dir, h5_files['channel_2'] = make_empty_h5(
            args.moco_output_dir, h5_file_names['channel_2'], brain_dims, stepsize=args.stepsize)
        logging.info(f"Created empty hdf5 file: {h5_file_names['channel_2']}")
    return h5_files


def create_moco_output_dir(args):
    setattr(args, 'moco_output_dir', os.path.join(args.dir, 'preproc'))

    # NOTE: this might be a good idea to enable in the future, not sure...
    # if os.path.exists(args.moco_output_dir) and args.overwrite:
    #     if args.verbose:
    #         print('removing existing moco output directory')
    #     shutil.rmtree(args.moco_output_dir)
    # elif os.path.exists(args.moco_output_dir) and not args.overwrite:
    #     raise ValueError(f'{args.moco_output_dir} already exists, use --overwrite to overwrite')

    if not os.path.exists(args.moco_output_dir):
        os.mkdir(args.moco_output_dir)
    logging.info(f'Moco output directory: {args.moco_output_dir}')
    return(args)


def apply_moco_parameters_to_channel_2(args, files,
                                       h5_files, transform_files):
    """Apply moco parameters to channel 2"""
    logging.info('Applying moco parameters to channel 2')
    assert 'channel_2' in files, 'files must include channel_2'

    # load ch1 image to get dimensions for chunking
    ch2_img = nib.load(files['channel_2'])
    n_timepoints = ch2_img.shape[-1]
    # assert n_timepoints == len(transform_files), 'number of transform files must match number of timepoints'

    ch1_meanbrain = get_mean_brain(args, files['channel_1'])

    # load full data
    ch2_data = ch2_img.get_fdata(dtype='float32')

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
    logging.info('Saving channel 2 data')

    # setup chunking into smaller parts (for memory)
    chunk_boundaries = get_chunk_boundaries(args, n_timepoints)
    for (chunk_start, chunk_end) in chunk_boundaries:
        with h5py.File(h5_files['channel_2'], 'a') as f:
            f['data'][..., chunk_start:chunk_end] = ch2_data[..., chunk_start:chunk_end]


def run_motion_correction(args, files, h5_files):
    """Run motion correction on tdTomato channel (1)"""

    logging.info('Running motion correction')

    # NB: need to make sure that the data is in the correct orientation
    # (i.e. direction of mean brain and chunkdata must be identical)
    logging.info('loading mean brain')
    ch1_meanbrain = get_mean_brain(args, files['channel_1'])

    # load ch1 image to get dimensions for chunking
    logging.info(f'opening file {files["channel_1"]}')
    ch1_img = nib.load(files['channel_1'])
    n_timepoints = ch1_img.shape[-1]

    # setup chunking into smaller parts (for memory)
    chunk_boundaries = get_chunk_boundaries(args, n_timepoints)

    # load full data
    logging.info('loading full data from channel 1')
    ch1_data = ch1_img.get_fdata(dtype='float32')

    motion_parameters = None
    transform_files = []
    # loop through chunks
    for i, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        logging.info(f'processing chunk {i + 1} of {len(chunk_boundaries)}')
        # get chunk data
        chunkdata = ch1_data[..., chunk_start:chunk_end]
        chunkdata_ants = ants.from_numpy(chunkdata)

        # run moco on chunk
        logging.info(f'moco on chunk {i + 1}')
        mytx = ants.motion_correction(image=chunkdata_ants, fixed=ch1_meanbrain,
            verbose=args.verbose, type_of_transform=args.type_of_transform,
            total_sigma=args.total_sigma, flow_sigma=args.flow_sigma)
        transform_files = transform_files + mytx['motion_parameters']

        logging.info(f'get motion parameters on chunk {i + 1}')
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
        logging.info(f'saving chunk {i + 1} of {len(chunk_boundaries)}')
        with h5py.File(h5_files['channel_1'], 'a') as f:
            f['data'][..., chunk_start:chunk_end] = mytx['motion_corrected'].numpy()

    return(transform_files, motion_parameters)


def save_motion_parameters(args, motion_parameters):
    assert os.path.exists(args.moco_output_dir), 'something went terribly wrong, moco dir does not exist'
    motion_df = pd.DataFrame(motion_parameters, columns=['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
    motion_file = os.path.join(args.moco_output_dir, 'motion_parameters.csv')
    motion_df.to_csv(motion_file, index=False)
    return(motion_file)


def save_motcorr_settings_to_json(args, files, h5_files, nii_files=None):
    assert os.path.exists(args.moco_output_dir), 'something went terribly wrong, moco dir does not exist'
    args_dict = vars(args)
    args_dict['files'] = files
    args_dict['h5_files'] = h5_files
    if args.save_nii:
        args_dict['nii_files'] = nii_files
    with open(os.path.join(args.moco_output_dir, 'moco_settings.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)


def moco_plot(args, motion_file):
    """Make motion correction plot"""

    motion_parameters = pd.read_csv(motion_file)

    # Get voxel resolution for figure
    x_res, y_res, z_res = get_dataset_resolution(args.dir)

    assert os.path.exists(args.moco_output_dir), 'something went terribly wrong, moco dir does not exist'

    # Save figure of motion over time - separate columns for translation and rotation
    save_file = os.path.join(args.moco_output_dir, 'motion_correction.png')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(motion_parameters.iloc[:, :3])
    plt.legend(labels=list(motion_parameters.columns[:3]), loc='upper right')
    plt.title(f"{'/'.join(args.dir.split('/')[-3:])}: Translation")
    plt.xlabel('Timepoint')
    plt.ylabel('Translation (microns)')
    plt.subplot(1, 2, 2)
    plt.plot(motion_parameters.iloc[:, 3:])
    plt.legend(labels=list(motion_parameters.columns[3:]), loc='upper right')
    plt.title(f"{'/'.join(args.dir.split('/')[-3:])}: Rotation")
    plt.xlabel('Timepoint')
    plt.ylabel('Rotation (degrees)')

    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    return(None)


def h5_to_nii(h5_path):
    nii_savefile = h5_path.replace('h5', 'nii')
    assert nii_savefile != h5_path, 'h5_to_nii: nii_savefile is the same as h5_path'
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
        logging.info(f'converting {h5_file} to nifti')
        _ = h5_to_nii(h5_file)
    return(None)


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype='moco')

    files, args = load_data(args)

    logging.info(f'files: {files}')

    args = create_moco_output_dir(args)

    args = set_stepsize(args)

    logging.info('set up h5 datsets')
    h5_files = setup_h5_datasets(args, files)

    logging.info('running motion correction')
    transform_files, motion_parameters = run_motion_correction(args, files, h5_files)

    if 'channel_2' in files:
        logging.info('applying motion correction for channel 2')
        apply_moco_parameters_to_channel_2(args, files, h5_files, transform_files)

    logging.info('saving to json')
    save_motcorr_settings_to_json(args, files, h5_files)

    logging.info('saving motion parameters')
    motion_file = save_motion_parameters(args, motion_parameters)

    logging.info('plotting motion')
    moco_plot(args, motion_file)

    if args.save_nii:
        logging.info('saving nifti')
        save_nii(args, h5_files)

    logging.info(f'Motion correction complete: {datetime.datetime.now()}')
