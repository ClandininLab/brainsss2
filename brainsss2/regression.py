import os
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy
import logging
import ants
import warnings
from nilearn.plotting import plot_stat_map
import scipy.stats
from brainsss2.argparse_utils import get_base_parser # noqa
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.fictrac import load_fictrac


def setup_mask(args, brain, meanbrainfile,
               maskfile, cleanup=4):
    if args.maskpct is None:
        return np.ones(brain.shape[:3], dtype=bool)
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
    return mask_ants[:, :, :].astype('int')


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


def save_desmtx(args, X, confound_names=None):
    colnames = args.behavior if args.behavior is not None else []
    if confound_names is not None:
        colnames += confound_names
    colnames += ['constant']
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
        args, results,
        qform, zooms, xyzt_units,
        use_fdr=True):

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
            cut_coords = np.arange(8, 49, 8) * zooms[2]
            with warnings.catch_warnings():
                # filter matplotlib warnings
                warnings.filterwarnings("ignore", module="matplotlib\..*")  # noqa
                if '1-p' in k:
                    plot_stat_map(save_file, os.path.join(args.dir, args.bg_img),
                        display_mode='z', threshold=1 - args.pthresh, draw_cross=False,
                        cut_coords=cut_coords, vmax=1, cmap='bwr',
                        title=f'Regression fdr p: {args.behavior[i]}',
                        output_file=os.path.join(
                            args.outdir, f'{k}_{args.behavior[i]}.png'))
                elif k == 'rsquared':
                    save_files[k] = save_file
                    plot_stat_map(save_file, os.path.join(args.dir, args.bg_img),
                        display_mode='z', threshold=.05, draw_cross=False,
                        cut_coords=cut_coords,
                        title='Regression r-squared',
                        output_file=os.path.join(
                            args.outdir, 'rsquared.png'))
                    if args.baseline_r2 is not None:
                        baseline_r2_img = nib.load(
                            os.path.join(args.dir, args.baseline_r2))
                        delta_r2 = result_data - baseline_r2_img.get_fdata()
                        delta_r2 = np.clip(delta_r2, 0, 1)
                        delta_r2_img = nib.Nifti1Image(
                            delta_r2, img.affine, img.header
                        )
                        delta_r2_file = os.path.join(args.outdir, 'delta_r2.nii')
                        delta_r2_img.to_filename(delta_r2_file)
                        plot_stat_map(delta_r2_file, os.path.join(args.dir, args.bg_img),
                            display_mode='z', threshold=0, draw_cross=False,
                            cut_coords=cut_coords,
                            title='Delta r-squared',
                            output_file=os.path.join(
                                args.outdir, 'delta_rsquared.png'))
                else:
                    save_files[(k, args.behavior[i])] = save_file

            logging.info(f"Saved {k} to {save_file}")
    return(save_files)


def transform_behavior(behavior, transform):
    assert transform in ['+', '-'], 'transform must be + or -'
    if transform == '+':
        behavior[behavior < 0] = 0
    else:
        behavior[behavior > 0] = 0
    return(behavior)


def setup_confounds(args, ntimepoints):

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

    if args.dct_bases is not None and args.dct_bases > 0:
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
