# pyright: reportMissingImports=false

import os
import numpy as np
import pandas as pd
import nibabel as nib
import h5py
import json
import ants
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from brainsss2.argparse_utils import get_base_parser, add_moco_arguments # noqa
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.h5_to_nii import h5_to_nii
from brainsss2.ants_utils import get_motion_parameters_from_transforms, get_dataset_resolution
from brainsss2.hdf5_utils import make_empty_h5, get_chunk_boundaries
from brainsss2.imgmath import imgmath


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
    
    if args.dirtype == 'func':
        assert 'functional' in files[0], 'dirtype is func but no functional data'
    elif args.dirtype == 'anat':
        assert 'anatomy' in files[0], 'dirtype is anat but no anatomical data'
    logging.info(f'Scan type: {args.dirtype}')

    files_dict = {
        'channel_1': files[0],
        'channel_2': files[1] if len(files) == 2 else None
    }
    if args.dirtype == 'anat':
        files_dict['channel_2'] = None

    return(files_dict, args)


def set_stepsize(args, scantype_stepsize_dict=None):
    if scantype_stepsize_dict is None:
        scantype_stepsize_dict = {'func': 40, 'anat': 5}
    setattr(args, 'stepsize', scantype_stepsize_dict[args.scantype])
    return(args)


def get_mean_brain(file):
    """get mean brain for channel 1 - return ants image object"""
    assert 'channel_1' in file, 'file must be channel_1'
    assert '.nii' in file, 'file must be nifti'
    meanbrain_file = file.replace('.nii', '_mean.nii')
    if os.path.exists(meanbrain_file):
        return ants.image_read(meanbrain_file)
    meanbrain_file = imgmath(file, 'mean')
    return(ants.image_read(meanbrain_file))


def setup_h5_datasets(args, files):
    """Make Empty MOCO files that will be filled chunk by chunk"""

    h5_file_names = {
        'channel_1': os.path.basename(files['channel_1']).replace(
            '.nii', '_moco.h5'),
        'channel_2': None}
    if args.verbose:
        logging.info(f'Creating channel_1 h5 file: {h5_file_names["channel_1"]}')

    # assume same affine for both channels
    img = nib.load(files['channel_1'])
    qform = img.header.get_qform()
    zooms = img.header.get_zooms()
    xyzt_units = img.header.get_xyzt_units()

    brain_dims = nib.load(files['channel_1']).shape
    moco_dir, filename = make_empty_h5(
        os.path.join(args.moco_output_dir, h5_file_names['channel_1']),
        brain_dims,
        qform=qform,
        zooms=zooms,
        xyzt_units=xyzt_units,
        stepsize=args.stepsize)
    h5_files = {'channel_2': None, 'channel_1': os.path.join(moco_dir, filename)}
    logging.info(f"Created empty hdf5 file: {h5_files['channel_1']}")

    if 'channel_2' in files and files['channel_2'] is not None:
        h5_file_names['channel_2'] = os.path.basename(files['channel_2']).replace(
            '.nii', '_moco.h5')
        moco_dir, filename = make_empty_h5(
            os.path.join(args.moco_output_dir, h5_file_names['channel_2']),
            brain_dims,
            qform=qform,
            zooms=zooms,
            xyzt_units=xyzt_units,
            stepsize=args.stepsize)
        h5_files['channel_2'] = os.path.join(moco_dir, filename)
        logging.info(f"Created empty hdf5 file: {h5_files['channel_2']}")
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
    ch2_img = ants.image_read(files['channel_2'], reorient=False)
    n_timepoints = ch2_img.shape[-1]
    spacing, direction = ch2_img.spacing, ch2_img.direction
    del ch2_img

    ch2_img = nib.load(files['channel_2'])
    # assert n_timepoints == len(transform_files), 'number of transform files must match number of timepoints'

    ch1_meanbrain = get_mean_brain(files['channel_1'])

    corrected_data = np.zeros(ch2_img.shape, dtype='float32')

    # overwrite existing data in place to prevent need for additional memory
    for timepoint in range(n_timepoints):
        # load transform
        try:
            transform = transform_files[timepoint]
        except IndexError:
            logging.warning(f'No transform file for timepoint {timepoint}')
            continue
        # apply transform
        moving_img = ants.from_numpy(ch2_img.dataobj[..., timepoint].astype('float32'))
        moving_img.set_spacing(spacing[:3])
        # for some reason ants.apply_transforms doesn't work with ch2_img.direction directly
        direction_vec = np.eye(3)
        direction_vec[np.diag_indices_from(direction_vec)] = np.diag(direction[:3])
        moving_img.set_direction(direction_vec)
        result = ants.apply_transforms(
            fixed=ch1_meanbrain,
            moving=moving_img,
            transformlist=transform,
            interpolator=args.interpolation_method)
        corrected_data[..., timepoint] = result.numpy()
    # save data
    logging.info('Saving channel 2 data')

    # setup chunking into smaller parts (for memory)
    chunk_boundaries = get_chunk_boundaries(args.stepsize, n_timepoints)
    for (chunk_start, chunk_end) in chunk_boundaries:
        with h5py.File(h5_files['channel_2'], 'a') as f:
            f['data'][..., chunk_start:chunk_end] = corrected_data[..., chunk_start:chunk_end]


def run_motion_correction(args, files, h5_files):
    """Run motion correction on tdTomato channel (1)"""
    if args.stepsize == 1:
        logging.info('Running motion correction using ants.registration')
    else:
        logging.info('Running motion correction using ants.motion_correction')

    # NB: need to make sure that the data is in the correct orientation
    # (i.e. direction of mean brain and chunkdata must be identical)
    logging.info('loading mean brain')
    ch1_meanbrain = get_mean_brain(files['channel_1'])
    logging.info(f'ch1_meanbrain shape: {ch1_meanbrain.shape}')
    # load ch1 image to get dimensions for chunking
    logging.info(f'opening file {files["channel_1"]}')

    ch1_img = nib.load(files['channel_1'])
    spacing = list(ch1_img.header.get_zooms())

    logging.info('image loaded successfully')
    n_timepoints = ch1_img.shape[-1]

    # setup chunking into smaller parts (for memory)
    chunk_boundaries = get_chunk_boundaries(args.stepsize, n_timepoints)

    motion_parameters = None
    transform_files = []

    # loop through chunks
    for i, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        logging.info(f'processing chunk {i + 1} of {len(chunk_boundaries)}')
        # get chunk data
        chunkdata = ch1_img.dataobj[..., chunk_start:chunk_end].astype('float32').squeeze()

        # create ants data obhect
        if len(chunkdata.shape) == 3:
            spacing = spacing[:3]
            direction = np.diag([-1., -1., 1.])
        else:
            spacing = spacing[:4]
            direction = np.diag([-1., -1., 1., 1.])
        chunkdata_ants = ants.from_numpy(
            chunkdata,
            spacing=spacing,
            direction=direction)

        # run moco on chunk
        if args.stepsize > 1:
            mytx = ants.motion_correction(image=chunkdata_ants, fixed=ch1_meanbrain,
                verbose=args.verbose, type_of_transform=args.type_of_transform,
                total_sigma=args.total_sigma, flow_sigma=args.flow_sigma)

            assert mytx is not None, 'ants.motioncorrection failed'
            step_transforms = mytx['motion_parameters']
            corrected_img = mytx['motion_corrected'].numpy()
        else:
            # this addresses a failure of ants.motion_correction
            # for high-resolution data - set stepsize to 1 to use
            # ants.registration instead
            myreg = ants.registration(moving=chunkdata_ants, fixed=ch1_meanbrain,
                verbose=args.verbose, type_of_transform=args.type_of_transform,
                total_sigma=args.total_sigma, flow_sigma=args.flow_sigma)

            assert myreg is not None, 'ants.registration failed'
            step_transforms = myreg['fwdtransforms']
            # extend to 4d for insertion below into h5 file
            corrected_img = myreg['warpedmovout'][:, :, :, np.newaxis]

        transform_files = transform_files + step_transforms
        logging.info(f'get motion parameters on chunk {i + 1}')
        # extract rigid body transform parameters (translation/rotation)
        if motion_parameters is None:
            # TODO: get_dataset_resolution is failing...
            motion_parameters = get_motion_parameters_from_transforms(
                step_transforms)[1]
        else:
            motion_parameters = np.vstack((motion_parameters,
                get_motion_parameters_from_transforms(
                    step_transforms)[1]))

        # save results from chunk
        logging.info(f'saving chunk {i + 1} of {len(chunk_boundaries)}')
        with h5py.File(h5_files['channel_1'], 'a') as f:
            f['data'][..., chunk_start:chunk_end] = corrected_img
    return(transform_files, motion_parameters)


def save_motion_parameters(args, motion_parameters):
    assert os.path.exists(args.moco_output_dir), 'something went terribly wrong, moco dir does not exist'
    motion_df = pd.DataFrame(motion_parameters, columns=['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
    motion_file = os.path.join(args.moco_output_dir, 'motion_parameters.csv')
    motion_df.to_csv(motion_file, index=False)
    FD_df = pd.DataFrame(get_framewise_displacement(motion_parameters), columns=['FD'])
    FD_file = os.path.join(args.moco_output_dir, 'framewise_displacement.csv')
    FD_df.to_csv(FD_file, index=False)

    return(motion_file)


def save_motcorr_settings_to_json(args, files, h5_files, nii_files=None):
    assert os.path.exists(args.moco_output_dir), 'something went terribly wrong, moco dir does not exist'
    args_dict = vars(args)
    args_dict['files'] = files
    args_dict['h5_files'] = h5_files
    if args.save_nii:
        args_dict['nii_files'] = nii_files
    if 'file_handler' in args_dict:
        del args_dict['file_handler']
    with open(os.path.join(args.moco_output_dir, 'moco_settings.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)


def moco_plot(args, motion_file):
    """Make motion correction plot"""

    motion_parameters = pd.read_csv(motion_file)

    # Get voxel resolution for figure
    x_res, y_res, z_res = get_dataset_resolution(
        os.path.join(args.dir, 'imaging')
    )

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


def save_nii(args, h5_files):
    """save moco data to nifti
    - reuse header info from existing nifti files"""
    for channel, h5_file in h5_files.items():
        logging.info(f'converting {h5_file} to nifti')
        _ = h5_to_nii(h5_file)
    return(None)


def get_framewise_displacement(motion_parameters, radius=1):
    """Calculate framewise displacement
    - radius of 1 is a guess value for fly brain
    based on pilot examination"""

    fd = np.zeros(motion_parameters.shape[0])
    diff = np.diff(motion_parameters, axis=0)
    diff[:, 3:] *= radius
    fd[1:] = diff.sum(axis=1)
    return fd
