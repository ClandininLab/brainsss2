import os
import sys
import nibabel as nib
from nilearn.plotting import plot_stat_map
import numpy as np
import pandas as pd
from brainsss2.regression_utils import get_transformed_data_slice
import h5py
import scipy.stats
from brainsss2.argparse_utils import get_base_parser, add_dr_args
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from brainsss2.logging_utils import setup_logging  # noqa
from brainsss2.preprocess_utils import check_for_existing_files, dump_args_to_json


def parse_args(args, allow_unknown=True):
    parser = get_base_parser('dimensionality reduction')

    parser = add_dr_args(parser)

    parser.add_argument('--dir', type=str, help='fly dir')
    if allow_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


def plot_comps(comps, varexp, compts, args):

    if args.label is not None:
        labelstring = f'_{args.label}'
    else:
        labelstring = ''

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    mask_img = nib.load(args.maskfile)

    for compnum in range(args.ncomps):
        comp_img = nib.Nifti1Image(np.zeros(mask_img.shape), mask_img.affine, mask_img.header)
        ctr = 0
        for slice in range(mask_img.shape[2]):
            full_slice = np.zeros(mask_img.shape[:2])
            nslicevox = np.sum(mask_img.dataobj[:, :, slice] > 0)
            slicemask = np.where(mask_img.dataobj[:, :, slice] > 0)
            full_slice[slicemask] = pca.components_[compnum, ctr:ctr + nslicevox]
            comp_img.dataobj[:, :, slice] = full_slice
            ctr += nslicevox
        thresh = scipy.stats.scoreatpercentile(comp_img.dataobj, args.threshpct)
        plot_stat_map(comp_img, args.meanfile, threshold=thresh,
            display_mode='z', cut_coords=args.ncuts,
            title=f'PCA{labelstring} Component {compnum} ({100*varexp[compnum]:.03} % var)',
            output_file=os.path.join(args.outdir, f'PCA{labelstring}_comp_{compnum:03}.png'))
        plt.figure(figsize=(12, 3))
        plt.plot(comp_timeseries[:, compnum])
        plt.title(f'PCA{labelstring} Component {compnum}')
        plt.savefig(os.path.join(args.outdir, f'PCA{labelstring}_timeseries_comp_{compnum:03}.png'))

    compts_df = pd.DataFrame(compts,
        columns=[f'pc{compnum:03}' for compnum in range(args.ncomps)])
    compts_df.to_csv(os.path.join(args.outdir, f'PCA{labelstring}_components.csv'))


def load_masked_data(datafile, maskfile):
    alldata = None
    if 'h5' in datafile:
        f = h5py.File(datafile, 'r')
        dataobj = f['data']
    elif 'nii' in datafile:
        f = nib.load(datafile)
        dataobj = f.dataobj
    else:
        raise ValueError('Unknown data file type')

    for slice in range(dataobj.shape[2]):
        data_slice, mask_slice = get_transformed_data_slice(
            dataobj[:, :, slice, :],
            mask_img.dataobj[:, :, slice]
        )
        if data_slice is None:
            continue
        if alldata is None:
            alldata = data_slice.T
        else:
            alldata = np.concatenate((alldata, data_slice.T), axis=1)
    return(alldata)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype="PCA")
    print(args)
    if args.outdir is None:
        args.outdir = os.path.join(
            os.path.dirname(args.dir),
            f'report/images/{args.funcdir}/PCA')

    if args.label is not None:
        labelstring = f'_{args.label}'
    else:
        labelstring = ''

    required_files = [f"PCA{labelstring}.json", f'PCA{labelstring}_components.csv']
    check_for_existing_files(args, args.outdir, required_files)

    args.logger.info(f'outdir: {args.outdir}')

    setattr(args, 'datafile',
        os.path.join(args.dir, args.datafile))

    setattr(args, 'meanfile',
        os.path.join(args.dir, args.meanfile))

    setattr(args, 'maskfile',
        os.path.join(args.dir, args.maskfile))

    mask_img = nib.load(args.maskfile)

    # load data
    args.logger.info('Loading data...')
    alldata = load_masked_data(args.datafile, args.maskfile)

    args.logger.info('Performing PCA...')
    alldata = alldata - np.mean(alldata, axis=0)
    alldata = alldata / np.std(alldata, axis=0)
    pca = PCA(n_components=args.ncomps, svd_solver='randomized')
    pca.fit(alldata)
    scale_comps = scale(pca.components_, axis=1)
    comp_timeseries = alldata.dot(scale_comps.T) / alldata.shape[1]

    args.logger.info('Plotting components...')
    plot_comps(pca.components_, pca.explained_variance_ratio_, comp_timeseries,
        args)

    jsonfile = os.path.join(args.outdir, f'PCA{labelstring}.json')
    args.logger.info(f'Saving PCA results to {jsonfile}')
    dump_args_to_json(args, jsonfile)
