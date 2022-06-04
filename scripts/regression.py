# use linear regression to model relation
# between activity and behavior
# use cosine bases to perform high-pass filtering

# pyright: reportMissingImports=false

import os
import sys
import numpy as np
import h5py
import nibabel as nib
import scipy
import datetime
import logging
from sklearn.linear_model import LinearRegression
from statsmodels.api import add_constant
from sklearn.metrics import r2_score
import scipy.stats
from statsmodels.stats.multitest import fdrcorrection
from brainsss2.regression_utils import get_transformed_data_slice
from brainsss2.argparse_utils import get_base_parser # noqa
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.fictrac import smooth_and_interp_fictrac
from brainsss2.imgmath import imgmath
from brainsss2.utils import load_timestamps
from brainsss2.preprocess_utils import check_for_existing_files
from brainsss2.regression import (
    load_brain,
    load_fictrac_data,
    setup_mask,
    setup_confounds,
    transform_behavior,
    save_desmtx,
    save_regressiondata
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('regression with movement')

    parser.add_argument('-d', '--dir', type=str,
        help='func directory to be analyzed', required=True)
    parser.add_argument('--label', type=str, help='model label', required=True)
    parser.add_argument('-f', '--file', type=str, help='file to process',
        default='preproc/functional_channel_2_moco_smooth-2.0mu.h5')
    parser.add_argument('--bg_img', type=str, help='background image for plotting')
    parser.add_argument('-b', '--behavior', type=str,
        help='behavior(s) to include in model',
        nargs='+', default=[])
    # TODO: also allow mask image or threshold value
    parser.add_argument('-m', '--maskpct', default=10, type=float,
        help='percentage (1-100) of image to include in mask')
    parser.add_argument('--fps', type=float, default=100, help='frame rate of fictrac camera')
    parser.add_argument('--resolution', type=float, default=10, help='resolution of fictrac data')
    parser.add_argument('--outdir', type=str, help='directory to save output')
    parser.add_argument('--pthresh', type=float, default=0.05, help='p value cutoff for plotting')
    parser.add_argument('--cores', type=int, default=4, help='number of cores to use')
    parser.add_argument('--dct_bases', type=int, default=12, help='number of dct bases to use')
    parser.add_argument('--confound_files', type=str, nargs='+', help='confound files')
    parser.add_argument('--baseline_r2', type=str, help='baseline r2 image to compute delta r2')
    parser.add_argument('--save_residuals', action='store_true',
        help='save model residuals')
    parser.add_argument('--residfile', type=str,
        help='filename/path for residuals (defaults to residuals.nii in regression folder')
    parser.add_argument('--std_betas', action='store_true', help='normalize regressors')
    if allow_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if 'dir' not in args:
        setattr(args, 'dir', os.path.dirname(args.file))

    if args.outdir is None:
        args.outdir = os.path.join(args.dir, 'regression', args.label)

    args = setup_logging(args, logtype=f'regression_{args.label}')

    check_for_existing_files(args, args.outdir, ['rsquared.nii'])
    args.logger.info(f'saving output to {args.outdir}')

    if args.bg_img is None:
        args.bg_img = os.path.join(args.dir, 'preproc/functional_channel_1_moco_mean.nii')
    if not os.path.exists(args.bg_img):
        baseimg = os.path.join(args.dir, 'preproc/functional_channel_1_moco.h5')
        args.logger.warning(f'Background image {args.bg_img} does not exist - trying to create mean from {baseimg}')
        assert os.path.exists(baseimg), 'base image for mean anat does not exist'
        imgmath(baseimg, 'mean', outfile_type='nii')
        assert os.path.exists(args.bg_img), 'mean image still does not exist'

    timestamps = load_timestamps(os.path.join(args.dir, 'imaging'))

    try:
        fictrac_raw, expt_len = load_fictrac_data(args)
    except FileNotFoundError:
        args.logger.warning('Fictrac data not found - cannot run regression, exiting')
        sys.exit(0)

    args.logger.info('loading data from h5 file')
    brain, qform, zooms, xyzt_units = load_brain(args)

    if args.save_residuals:
        args.logger.info('creating/loading functional mean')
        funcmeanfile = os.path.join(args.dir, args.file.replace('.h5', '_mean.nii'))
        if not os.path.exists(funcmeanfile):
            imgmath(os.path.join(args.dir, args.file), 'mean', outfile_type='nii')
        funcmean_img = nib.load(funcmeanfile)

    maskfile = args.bg_img.replace('mean.', 'mask.')
    assert maskfile != args.bg_img, 'maskfile should not be the same as the bg_img'
    if os.path.exists(maskfile):
        args.logger.info('loading existing mask')
        mask = nib.load(maskfile).get_fdata().astype('int')
    else:
        args.logger.info('creating mask')
        mask = setup_mask(args, brain,
            args.bg_img, maskfile)

    args.logger.info(f"Performing regression on {args.file}")
    if args.behavior is not None:
        # change variable to use an empty list for confound only
        behaviors = args.behavior
        args.logger.info(f'behaviors: {behaviors}')
    else:
        behaviors = []
        args.logger.info('confound modeling only')

    results = {'rsquared': np.zeros(brain.shape[:-1])}
    if args.behavior is not None:
        results['beta'] = np.zeros(list(brain.shape[:-1]) + [len(args.behavior)])
        results['tstat'] = np.zeros(list(brain.shape[:-1]) + [len(args.behavior)])

    confound_regressors, confound_names = setup_confounds(args, brain.shape[-1])
    if len(confound_names) > 0:
        args.logger.info(f'confound regressors: {confound_names}')

    if args.save_residuals:
        residual_img = nib.Nifti1Image(
            np.zeros(brain.shape, dtype='float32'),
            affine=qform)
        residual_img.header.set_qform(qform)
        residual_img.header.set_sform(qform)
        residual_img.header.set_zooms(zooms)
        residual_img.header.set_xyzt_units(xyz=xyzt_units[0], t=xyzt_units[1])

    # loop over slices
    for z in range(brain.shape[2]):

        # setup model for each slice
        regressors = {}
        for behavior in behaviors:
            if behavior[-1] in ['+', '-']:
                behavior_name = behavior[:-1]  # name without sign, for use in loading fictrac data
                behavior_transform = behavior[-1]
            else:
                behavior_name = behavior
                behavior_transform = None

            fictrac_interp = smooth_and_interp_fictrac(
                fictrac_raw, args.fps, args.resolution, expt_len,
                behavior_name, timestamps=timestamps, z=z)[:, np.newaxis]

            if behavior_transform is not None:
                fictract_interp = transform_behavior(fictrac_interp, behavior_transform)

            regressors[behavior] = fictrac_interp

        if len(regressors) > 0:
            X = np.concatenate(
                [regressors[behavior] for behavior in args.behavior], axis=1)
            X = X - np.mean(X, axis=0)  # demean regressors
        else:
            X = None

        zdata_trans, zmask_vec = get_transformed_data_slice(brain[:, :, z, :], mask[:, :, z])
        if zdata_trans is None:
            args.logger.info(f'Skipping slice {z} because it has no brain')
            continue
        else:
            args.logger.info(F"Processing slice {z}: {np.sum(zmask_vec)} voxels")

        lm = LinearRegression(n_jobs=args.cores, fit_intercept=False)
        y = zdata_trans.T
        # add confound regressors
        if confound_regressors is not None:
            if X is not None:
                X = np.hstack((X, confound_regressors))
            else:
                X = confound_regressors
        if args.std_betas:
            args.logger.debug('Standardizing regressors')
            X = X - np.mean(X, axis=0)
            X = X / np.std(X, axis=0)
        X = add_constant(X, prepend=False)
        lm.fit(X, y)
        predictions = lm.predict(X)
        residuals = y - predictions
        if args.save_residuals:
            # need to map mask residuals back into full space
            residuals_full = np.zeros((residuals.shape[0], zmask_vec.shape[0]))
            residuals_full[:, zmask_vec == 1] = residuals
            residual_img.dataobj[:, :, z, :] = residuals_full.T.reshape(
                brain.shape[0], brain.shape[1], brain.shape[3])
            # add mean back onto residuals - this is slow but not clear how else to do it
            for timepoint in range(residual_img.shape[-1]):
                residual_img.dataobj[:, :, z, timepoint] += funcmean_img.dataobj[:, :, z]
        squared_resids = (residuals)**2
        df = X.shape[0] - X.shape[1]
        MSE = squared_resids.sum(axis=0) / (df)
        XtX = np.dot(X.T, X)

        for i in range(len(behaviors)):
            slice_coefs = np.zeros(brain.shape[:2])
            slice_coefs[mask[:, :, z] == 1] = lm.coef_[:, i]
            results['beta'][:, :, z, i] = slice_coefs
            slice_tstat = np.zeros(brain.shape[:2])
            slice_tstat[mask[:, :, z] == 1] = lm.coef_[:, i] / np.sqrt(MSE[i] / np.diag(XtX)[i])
            args.logger.debug(f'max tstat: {np.max(slice_tstat[mask[:, :, z] == 1])}')
            results['tstat'][:, :, z, i] = slice_tstat

        slice_rsquared = np.zeros(brain.shape[:2])
        slice_rsquared[mask[:, :, z] == 1] = r2_score(y, predictions, multioutput='raw_values')
        results['rsquared'][:, :, z] = slice_rsquared

    if 'tstat' in results:
        results['pvalue'] = (1 - scipy.stats.t.cdf(x=np.abs(results['tstat']), df=df)) * 2
        results['fdr_pvalue'] = fdrcorrection(
            results['pvalue'].reshape(np.prod(results['pvalue'].shape)))[1].reshape(results['pvalue'].shape)

    args.logger.info('saving results')
    save_desmtx(args, X, confound_names)

    save_files = save_regressiondata(
        args, results,
        qform, zooms, xyzt_units)
    if args.save_residuals:
        del results
        del brain

        if args.residfile is None:
            residfile = os.path.join(args.outdir, 'residuals.h5')
        else:
            residfile = os.path.join(args.dir, args.residfile)

        with h5py.File(residfile, 'w') as f:
            f.create_dataset('data', data=residual_img.dataobj)
            f.create_dataset('qform', data=residual_img.header.get_qform())
            f.create_dataset('zooms', data=residual_img.header.get_zooms())
            f.create_dataset('xyzt_units', data=residual_img.header.get_xyzt_units())

    args.logger.info(f'job completed: {datetime.datetime.now()}')
