# pyright: reportMissingImports=false

import os
import sys
import numpy as np
import argparse
import h5py
import nibabel as nib
import scipy
import datetime
import ants
from columnwise_corrcoef_perf import AlmightyCorrcoefEinsumOptimized
import logging
from nilearn.plotting import plot_stat_map
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, os.path.realpath("../brainsss"))
sys.path.insert(0, os.path.realpath("../brainsss/scripts"))
from argparse_utils import get_base_parser # noqa
from logging_utils import setup_logging # noqa
from fictrac import load_fictrac, smooth_and_interp_fictrac
from imgmean import imgmean
from utils import load_timestamps


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('correlation between activity and behavior')

    parser.add_argument(
        "-b", "--basedir",
        type=str,
        help="base directory for fly data",
        required=True)
    parser = argparse.ArgumentParser(description='compute correlation between neural data and behavior')
    parser.add_argument('-d', '--dir', type=str,
        help='func directory to be analyzed', required=True)
    parser.add_argument('-f', '--file', type=str, help='file to process',
        default='preproc/functional_channel_2_moco_hpf.h5')
    parser.add_argument('--bg_img', type=str, help='background image for plotting')
    parser.add_argument('-b', '--behavior', type=str,
        help='behavior(s) to analyze (add + or - as suffix to limit values',
        required=True, nargs='+')
    # TODO: also allow mask image or threshold value
    parser.add_argument('-m', '--maskpct', default=50, type=float,
        help='percentage (1-100) of image to include in mask')
    parser.add_argument('-l', '--logfile', type=str, help='directory to save log file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('--fps', type=float, default=100, help='frame rate of fictrac camera')
    parser.add_argument('--resolution', type=float, default=10, help='resolution of fictrac data')
    parser.add_argument('-o', '--outdir', type=str, help='directory to save output')
    parser.add_argument('--corrthresh', type=float, default=0.1, help='correlation threshold for plotting')
    parser.add_argument('--outstem', type=str, help='output stem')

    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()
    return args


def setup_mask(args, brain, meanbrainfile,
               maskfile, cleanup=4):
    if args.maskpct is not None:
        meanimg = nib.load(meanbrainfile)
        meanbrain = meanimg.get_fdata()
        maskthresh = scipy.stats.scoreatpercentile(meanbrain, args.maskpct)
        meanimg_ants = ants.from_nibabel(meanimg)
        mask_ants = ants.get_mask(meanimg_ants,
            low_thresh=maskthresh, high_thresh=np.inf, 
            cleanup=cleanup)
        logging.info(f'Mask threshold for {args.maskpct} percent:: {maskthresh}')
        maskimg = ants.to_nibabel(mask_ants)
        maskimg.to_filename(maskfile)
        mask = mask_ants[:, :, :]
    else:
        mask = np.ones(brain.shape[:3], dtype=bool)
    return(mask)


def load_fictrac_data(args):
    fictrac_raw = load_fictrac(os.path.join(args.dir, 'fictrac'))
    expt_len = (fictrac_raw.shape[0] / args.fps) * 1000
    return(fictrac_raw, expt_len)


def load_brain(args):
    full_load_path = os.path.join(args.dir, args.file)
    logging.info(f'loading brain file: {full_load_path}')
    with h5py.File(full_load_path, 'r') as hf:
        brain = hf['data'][:]
        qform = hf['qform'][:]
        zooms = hf['zooms'][:]
        xyzt_units = [i.decode('utf-8') for i in hf['xyzt_units'][:]]
    logging.info(f'brain shape: {brain.shape}')
    return(brain, qform, zooms, xyzt_units)


def get_transformed_data_slice(args, brain, mask, z):
    zdata = brain[:, :, z, :]
    zmask = mask[:, :, z].reshape(np.prod(brain.shape[:2])).astype('bool')
    if zmask.sum() == 0:
        return(None, zmask)
    return(zdata.transpose(2, 0, 1).reshape(brain.shape[3], -1), zmask)


def save_corrdata(args, corr_brain, behavior, qform=None, zooms=None, xyzt_units=None):
    save_file = os.path.join(args.outdir, f'{args.outstem}_corr-{behavior}.nii')
    img = nib.Nifti1Image(corr_brain, None)
    if qform is not None:
        img.header.set_qform(qform)
        img.header.set_sform(qform)
    else:
        img.header.set_qform(np.eye(4))
        img.header.set_sform(np.eye(4))

    if zooms is not None:
        img.header.set_zooms(zooms[:3])
    if xyzt_units is not None:
        img.header.set_xyzt_units(xyz=xyzt_units[0], t=xyzt_units[1])
    img.to_filename(save_file)
    logging.info(f"Saved {save_file}")
    return(save_file)


def transform_behavior(behavior, transform):
    assert transform in ['+', '-'], 'transform must be + or -'
    if transform == '+':
        behavior[behavior < 0] = 0
    else:
        behavior[behavior > 0] = 0
    return(behavior)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if 'dir' not in args:
        setattr(args, 'dir', os.path.dirname(args.file))

    if args.outdir is None:
        args.outdir = os.path.join(args.dir, 'corr')
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if args.outstem is None:
        setattr(args, 'outstem', os.path.basename(args.file).replace('.h5', ''))

    setup_logging(args, logtype='correlation',
        logfile=args.logfile)

    if args.bg_img is None:
        args.bg_img = os.path.join(args.dir, 'preproc/functional_channel_1_moco_mean.nii')
    if not os.path.exists(args.bg_img):
        baseimg = os.path.join(args.dir, 'preproc/functional_channel_1_moco.h5')
        logging.warning(f'Background image {args.bg_img} does not exist - trying to create mean from {baseimg}')
        assert os.path.exists(baseimg), 'base image for mean anat does not exist'
        imgmean(baseimg, outfile_type='nii')
        assert os.path.exists(args.bg_img), 'mean image still does not exist'

    timestamps = load_timestamps(os.path.join(args.dir, 'imaging'))

    fictrac_raw, expt_len = load_fictrac_data(args)

    logging.info('loading data from h5 file')
    brain, qform, zooms, xyzt_units = load_brain(args)

    meanbrainfile = imgmean(os.path.join(args.dir, args.file),
        outfile_type='nii')

    maskfile = meanbrainfile.replace('mean.', 'mask.')
    if os.path.exists(maskfile):
        logging.info('loading existing mask')
        mask = nib.load(maskfile).get_fdata()
    else:
        logging.info('creating mask')
        mask = setup_mask(args, brain,
            meanbrainfile, maskfile)

    for behavior in args.behavior:

        if behavior[-1] in ['+', '-']:
            behavior_name = behavior[:-1]  # name without sign, for use in loading fictrac data
            behavior_transform = behavior[-1]
            logging.info(f'Transforming behavior {behavior_name}: {behavior_transform} values only')
        else:
            behavior_name = behavior
            behavior_transform = None

        if behavior_name not in fictrac_raw.columns:
            logging.warning(f'behavior {behavior} not found in fictrac data')
            continue

        logging.info(f"Performing Correlation on {args.file}; behavior: {behavior}")

        corr_brain = np.zeros((brain.shape[:3]))

        # loop over slices to save memory
        for z in range(brain.shape[2]):

            zdata_trans, zmask = get_transformed_data_slice(args, brain, mask, z)
            if zdata_trans is None:
                logging.info(f'Skipping slice {z} because it has no in-mask voxels')
                continue

            logging.info(F"Processing slice {z}: {np.sum(zmask)} voxels")
            # interpolate fictrac to match the timestamps of this slice
            fictrac_interp = smooth_and_interp_fictrac(
                fictrac_raw, args.fps, args.resolution, expt_len,
                behavior_name, timestamps=timestamps, z=z)[:, np.newaxis]

            if behavior_transform is not None:
                fictract_interp = transform_behavior(fictrac_interp, behavior_transform)

            # compute correlation using optimized method for large matrices
            cc = AlmightyCorrcoefEinsumOptimized(
                zdata_trans[:, zmask], fictrac_interp)[0, :]
            cc_full = np.zeros(zdata_trans.shape[1])
            cc_full[zmask == True] = cc  # noqa: E712
            corr_brain[:, :, z] = cc_full.reshape(brain.shape[0], brain.shape[1])

        save_file = save_corrdata(
            args, corr_brain, behavior,
            qform, zooms, xyzt_units)
        logging.info(f'job completed: {datetime.datetime.now()}')

        plot_stat_map(save_file, os.path.join(args.dir, args.bg_img),
            display_mode='z', threshold=args.corrthresh, draw_cross=False,
            cut_coords=np.arange(8, 49, 8), title=f'Correlation: {behavior}',
            output_file=os.path.join(args.outdir, f'{args.outstem}_corr-{behavior}.png'))
