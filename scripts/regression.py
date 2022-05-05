# use linear regression to model relation
# between activity and behavior
# use cosine bases to perform high-pass filtering

# pyright: reportMissingImports=false

import os
import sys
import numpy as np
import pandas as pd
import argparse
import h5py
import nibabel as nib
import scipy
import datetime
import logging
import shutil
from nilearn.plotting import plot_stat_map
from sklearn.linear_model import LinearRegression
from statsmodels.api import add_constant
from sklearn.metrics import r2_score
import scipy.stats
from statsmodels.stats.multitest import fdrcorrection

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
    parser.add_argument('--label', type=str, help='model label', required=True)
    parser.add_argument('-f', '--file', type=str, help='file to process',
        default='preproc/functional_channel_2_moco_hpf.h5')
    parser.add_argument('--bg_img', type=str, help='background image for plotting')
    parser.add_argument('-b', '--behavior', type=str,
        help='behavior(s) to include in model',
        required=True, nargs='+')
    # TODO: also allow mask image or threshold value
    parser.add_argument('-m', '--maskpct', default=10, type=float,
        help='percentage (1-100) of image to include in mask')
    parser.add_argument('-l', '--logfile', type=str, help='directory to save log file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('--fps', type=float, default=100, help='frame rate of fictrac camera')
    parser.add_argument('--resolution', type=float, default=10, help='resolution of fictrac data')
    parser.add_argument('-o', '--outdir', type=str, help='directory to save output')
    parser.add_argument('--corrthresh', type=float, default=0.1, help='correlation threshold for plotting')
    parser.add_argument('--cores', type=int, default=4, help='number of cores to use')
    parser.add_argument('--dct_bases', type=int, default=8, help='number of dct bases to use')
    parser.add_argument('--confound_files', type=str, nargs='+', help='confound files')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing output')
    parser.add_argument('--save-residuals', action='store_true',
        help='save model residuals - NOT YET IMPLEMENTED')
    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()
    return args


def setup_mask(args, brain):
    if args.maskpct is not None:
        meanbrain = np.mean(brain, axis=3)
        maskthresh = scipy.stats.scoreatpercentile(meanbrain, args.maskpct)
        mask = meanbrain > maskthresh
        logging.info(f'Mask threshold for {args.maskpct} percent:: {maskthresh}')
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


def get_transformed_data_slice(zdata, zmask):
    if zmask.sum() == 0:
        return(None, zmask)
    zmask_vec = zmask.reshape(np.prod(zmask.shape))
    zdata_mat = zdata.reshape((np.prod(zmask.shape), zdata.shape[-1]))
    return(zdata_mat[zmask_vec], zmask_vec)


def save_desmtx(args, X, confound_names=None):
    colnames = args.behavior
    if confound_names is not None:
        colnames += confound_names
    df = pd.DataFrame(X, columns=colnames)
    outfile = os.path.join(args.outdir, f'{args.label}_desmtx.csv')
    df.to_csv(outfile, index=False)
    logging.info(f'Saved design matrix to {outfile}')


def get_dct_mtx(X, ndims):
    # based loosely on spm_dctmtx.m
    N = X.shape[0]
    dct_mtx = np.zeros((X.shape[0], ndims))
    for i in range(ndims):
        dct_mtx[:, i] = np.sqrt(2 / N) * np.cos(
            np.pi * (2 * np.arange(N) + 1) * (i + 1) / (2 * N))
    return(dct_mtx)


def save_regressiondata(
        args, results, behavior,
        qform, zooms, xyzt_units):
    save_files = {}
    for k, result in results.items():
        if len(result.shape) > 3:
            n_results = result.shape[3]
        else:
            n_results = 1  # rsquared
        k = k.replace('pvalue', '1-p')
        for i in range(n_results):
            if k == 'rsquared':
                result_data = result
                save_file = os.path.join(args.outdir, f'{k}.nii')
            else:
                result_data = result[:, :, :, i]
                save_file = os.path.join(args.outdir, f'{k}_{args.behavior[i]}.nii')
            if '1-p' in k:
                logging.info(f'using 1-value for {k}')
                result_data = 1 - result_data
            print(f'Saving {k} slot {i}, shape {result_data.shape}')
            img = nib.Nifti1Image(np.squeeze(result_data), None)
            img.header.set_qform(qform)
            img.header.set_sform(qform)

            img.header.set_zooms(zooms[:3])
            img.header.set_xyzt_units(xyz=xyzt_units[0], t=xyzt_units[1])
            img.to_filename(save_file)
            save_files[(k, i)] = save_file

            logging.info(f"Saved {k} to {save_file}")
    return(save_files)


def transform_behavior(behavior, transform):
    assert transform in ['+', '-'], 'transform must be + or -'
    if transform == '+':
        behavior[behavior < 0] = 0
    else:
        behavior[behavior > 0] = 0
    return(behavior)


def setup_confounds(args, brain):
    ntimepoints = brain.shape[-1]

    confound_mtx = None
    confound_names = []

    if args.confound_files is not None:
        for confound_file in args.confound_files:
            confound_df = pd.read_csv(os.path.join(args.dir, confound_file))
            confound_names += list(confound_df.columns)
            if confound_mtx is None:
                confound_mtx = confound_df.values
            else:
                confound_mtx = np.hstack((confound_mtx, confound_df.values))

    if args.dct_bases is not None:
        dct_mtx = get_dct_mtx(np.zeros((ntimepoints, 1)), args.dct_bases)
        if confound_mtx is None:
            confound_mtx = dct_mtx
        else:
            confound_mtx = np.hstack((
                confound_mtx,
                dct_mtx
            ))
        confound_names += ['dct_basis_%d' % i for i in range(args.dct_bases)]
    else:
        confound_mtx = None

    return confound_mtx, confound_names


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if 'dir' not in args:
        setattr(args, 'dir', os.path.dirname(args.file))

    if args.outdir is None:
        args.outdir = os.path.join(args.dir, 'regression', args.label)

    args = setup_logging(args, logtype='regression',
        logdir=args.outdir)

    logging.info(f'saving output to {args.outdir}')

    if os.path.exists(args.outdir) and not args.overwrite:
        logging.info(f'output directory {args.outdir} already exists and overwrite is False')
        sys.exit(0)
    elif os.path.exists(args.outdir) and args.overwrite:
        shutil.rmtree(args.outdir)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

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

    mask = setup_mask(args, brain)

    logging.info(f"Performing regression on {args.file}")
    logging.info(f'behaviors: {args.behavior}')

    results = {
        'beta': np.zeros(list(brain.shape[:-1]) + [len(args.behavior)]),
        'tstat': np.zeros(list(brain.shape[:-1]) + [len(args.behavior)]),
        'rsquared': np.zeros(brain.shape[:-1])

    }

    confound_regressors, confound_names = setup_confounds(args, brain)
    if len(confound_names) > 0:
        logging.info(f'confound regressors: {confound_names}')
    # loop over slices
    for z in range(brain.shape[2]):

        # setup model for each slice
        logging.info(f'setting up model for slice {z}')
        regressors = {}
        for behavior in args.behavior:
            if behavior[-1] in ['+', '-']:
                behavior_name = behavior[:-1]  # name without sign, for use in loading fictrac data
                behavior_transform = behavior[-1]
                logging.info(f'Transforming behavior {behavior_name}: {behavior_transform} values only')
            else:
                behavior_name = behavior
                behavior_transform = None

            fictrac_interp = smooth_and_interp_fictrac(
                fictrac_raw, args.fps, args.resolution, expt_len,
                behavior_name, timestamps=timestamps, z=z)[:, np.newaxis]

            if behavior_transform is not None:
                fictract_interp = transform_behavior(fictrac_interp, behavior_transform)

            regressors[behavior] = fictrac_interp

        X = np.concatenate([regressors[behavior] for behavior in args.behavior], axis=1)

        zdata_trans, zmask_vec = get_transformed_data_slice(brain[:, :, z, :], mask[:, :, z])
        if zdata_trans is None:
            logging.info(f'Skipping slice {z} because it has no brain')
            continue
        else:
            logging.info(F"Processing slice {z}: {np.sum(zmask_vec)} voxels")

        lm = LinearRegression(n_jobs=args.cores)
        y = zdata_trans.T
        # add confound regressors
        if confound_regressors is not None:
            X = np.hstack((X, confound_regressors))

        lm.fit(X, y)
        predictions = lm.predict(X)
        X_w_const = add_constant(X)
        squared_resids = (y - predictions)**2
        df = X_w_const.shape[0] - X_w_const.shape[1]
        MSE = squared_resids.sum(axis=0) / (df)
        XtX = np.dot(X_w_const.T, X_w_const)

        for i in range(len(args.behavior)):
            slice_coefs = np.zeros(brain.shape[:2])
            slice_coefs[mask[:, :, z]] = lm.coef_[:, i]
            results['beta'][:, :, z, i] = slice_coefs
            slice_tstat = np.zeros(brain.shape[:2])
            slice_tstat[mask[:, :, z]] = lm.coef_[:, i] / np.sqrt(MSE[i] * np.diag(XtX)[i])
            results['tstat'][:, :, z, i] = slice_tstat

        slice_rsquared = np.zeros(brain.shape[:2])
        slice_rsquared[mask[:, :, z]] = r2_score(y, predictions, multioutput='raw_values')
        results['rsquared'][:, :, z] = slice_rsquared

    results['pvalue'] = (1 - scipy.stats.t.cdf(x=np.abs(results['tstat']), df=df)) * 2
    results['fdr_pvalue'] = fdrcorrection(
        results['pvalue'].reshape(np.prod(results['pvalue'].shape)))[1].reshape(results['pvalue'].shape)

    logging.info('saving results')
    save_desmtx(args, X, confound_names)

    save_files = save_regressiondata(
        args, results, behavior,
        qform, zooms, xyzt_units)

    logging.info(f'job completed: {datetime.datetime.now()}')

    # for k, file in save_files:
    #     if k[0]
    plot_stat_map(save_files, os.path.join(args.dir, args.bg_img),
        display_mode='z', threshold=.05, draw_cross=False,
        cut_coords=np.arange(8, 49, 8), title=f'Correlation: {behavior}',
        output_file=os.path.join(args.outdir, f'corr_{behavior}.png'))
