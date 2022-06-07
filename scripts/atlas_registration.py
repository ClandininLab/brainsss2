# compute registration between anatomical mean and atlas
# pyright: reportMissingImports=false


import os
import sys
import ants
import logging
from dipy.viz import regtools
import nibabel as nib
import scipy.stats
import numpy as np
from sklearn.preprocessing import quantile_transform
from nilearn.plotting import plot_roi, plot_anat

from brainsss2.argparse_utils import get_base_parser, add_moco_arguments # noqa
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.imgmath import imgmath # noqa
from brainsss2.preprocess_utils import check_for_existing_files, dump_args_to_json
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('anatomical_registration')

    # need to add this manually to procesing steps in order to make required
    parser.add_argument(
        '-d', '--dir',
        type=str,
        help='fly dir to be processed',
        required=True)
    parser.add_argument('--funcfile',
        type=str,
        default='func_0/preproc/functional_channel_1_moco_mean.nii',
        help='channel 1 mean img for functional data (after moco)')
    parser.add_argument(
        '--transformdir',
        type=str,
        help='directory to save transforms')

    atlas = parser.add_argument_group('atlas options')
    atlas.add_argument('--anatfile',
        type=str,
        default='anat_0/preproc/anatomy_channel_1_res-2.0mu_moco_mean.nii',
        help='channel 1 mean image for anat data (after moco)')
    atlas.add_argument(
        '--atlasfile',
        type=str,
        help='atlas file',
        default='20220301_luke_2_jfrc_affine_zflip_2umiso.nii')
    atlas.add_argument('--atlasname',
        type=str,
        help='identifier for atlas space',
        default='jfrc')

    group = parser.add_argument_group('ANTs registration options')
    group.add_argument('--type_of_transform', type=str, default='SyN',
        help='type of transform to use')
    group.add_argument('--interpolation_method', type=str, default='lanczosWindowedSinc')
    group.add_argument('--flow_sigma', type=float, default=3,
        help='flow sigma for registration - higher sigma focuses on coarser features')
    group.add_argument('--grad_step', type=float, default=.1,
        help='gradient step size')
    group.add_argument('--total_sigma', type=float, default=1,
        help='total sigma for registration - higher values will restrict the amount of deformation allowed')
    group.add_argument('--syn_sampling', type=int, default=64,
        help='he nbins or radius parameter for the syn metric')
    group.add_argument(
        '--maskthresh',
        type=float,
        default=.1,
        help='density threshold for masking'
    )

    if allow_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args


def make_clean_anat(anatfile, sigma=5, thresh_pct=40, normalize=False):
    anat_img = nib.load(anatfile)
    anat_ants = ants.image_read(anatfile)
    low_thresh = scipy.stats.scoreatpercentile(anat_ants.numpy(), thresh_pct)
    anat_ants_masked = ants.get_mask(anat_ants,
        low_thresh=low_thresh,
        cleanup=4)
    maskfile = anatfile.replace('.nii', '_mask.nii')
    assert maskfile != anatfile, 'maskfile should not be the same as anatfile'
    anat_ants_masked.to_filename(maskfile)

    brain = anat_img.get_fdata(dtype='float32') * anat_ants_masked[...]

    # ### Blur brain and mask small values ###
    # brain_copy = brain.copy()
    # brain_copy = gaussian_filter(brain_copy, sigma=sigma)
    # threshold = triangle(brain_copy)
    # brain_copy[np.where(brain_copy < threshold/2)] = 0

    # ### Remove blobs outside contiguous brain ###
    # labels, label_nb = scipy.ndimage.label(brain_copy)
    # brain_label = np.bincount(labels.flatten())[1:].argmax()+1
    # brain_copy = brain.copy().astype('float32')
    # brain_copy[np.where(labels != brain_label)] = np.nan

    ### Perform quantile normalization ###
    if normalize:
        brain_out = quantile_transform(brain.flatten().reshape(-1, 1), n_quantiles=500, random_state=0, copy=True)
        brain_out = brain_out.reshape(brain.shape)
        brain = np.nan_to_num(brain_out, copy=False)

    clean_anatfile = anatfile.replace('.nii', '_clean.nii')
    assert clean_anatfile != anatfile, 'clean_anatfile is same as anatfile'
    nib.save(nib.Nifti1Image(brain, anat_img.affine, anat_img.header), clean_anatfile)
    return clean_anatfile, maskfile


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.dir.split('/')[-1] == 'func_0':
        print('funcdir provided - using parent fly dir')
        args.dir = os.path.dirname(args.dir)

    assert os.path.exists(args.dir), f"fly dir {args.dir} does not exist"
    assert os.path.exists(args.atlasfile), f"atlas file {args.atlasfile} does not exist"

    if 'basedir' not in args or args.basedir is None:
        args.basedir = os.path.dirname(args.funcfile)

    args = setup_logging(args, logtype="atlasreg")

    registration_dir = os.path.join(args.dir, "registration")
    setattr(args, "registration_dir", registration_dir)

    required_files = ['registration.json']
    check_for_existing_files(args, args.registration_dir, required_files)

    if not os.path.exists(registration_dir):
        os.mkdir(registration_dir)

    if args.transformdir is None:
        args.transformdir = os.path.join(registration_dir, 'transforms')
    if not os.path.exists(args.transformdir):
        os.mkdir(args.transformdir)
    setattr(args, "transformdir", args.transformdir)

    filestem = os.path.basename(args.funcfile).split('.')[0]

    atlasimg = ants.image_read(args.atlasfile)
    print(atlasimg)

    # first register anat channel 1 mean to atlas

    clean_anat_file, maskfile = make_clean_anat(os.path.join(args.dir, args.anatfile))
    mask_plot_file = maskfile.replace('.nii', '_plot.png')
    assert mask_plot_file != maskfile, 'mask_plot_file should not be the same as maskfile'
    plot_roi(maskfile, bg_img=clean_anat_file,
        output_file=mask_plot_file, display_mode='z', cut_coords=np.arange(10, 100, 10))

    meananatimg = ants.image_read(clean_anat_file)

    ## register anat channel 1 mean to atlas
    args.logger.info('registering anat channel 1 mean to atlas')
    anat_to_atlas = ants.registration(
        fixed=atlasimg,
        moving=meananatimg,
        verbose=args.verbose,
        outprefix=f'{args.transformdir}/anat_to_atlas_',
        type_of_transform=args.type_of_transform,
        syn_sampling=args.syn_sampling,
        total_sigma=args.total_sigma,
        flow_sigma=args.flow_sigma,
        grad_step=args.grad_step,
        reg_iterations=[100, 100, 20]
    )

    ## save transformed mean
    mean_reg_to_atlas_file = os.path.join(
        registration_dir,
        os.path.basename(args.anatfile).replace('.nii', f'_space-{args.atlasname}.nii')
    )
    setattr(args, "mean_reg_to_atlas_file", mean_reg_to_atlas_file)
    anat_to_atlas['warpedmovout'].to_filename(mean_reg_to_atlas_file)

    # save transformed atlas to mean space
    atlas_to_mean_reg_file = os.path.join(
        registration_dir,
        os.path.basename(args.atlasfile).replace('.nii', '_space-anat.nii')
    )
    setattr(args, "atlas_to_mean_reg_file", atlas_to_mean_reg_file)
    anat_to_atlas['warpedfixout'].to_filename(atlas_to_mean_reg_file)

    regtools.overlay_slices(
        atlasimg.numpy(),
        anat_to_atlas['warpedmovout'].numpy(),
        ltitle='atlas',
        rtitle='warped anat',
        fname=os.path.join(registration_dir, 'anat_to_atlas_overlay.png'))

    p = plot_anat(anat_to_atlas['warpedmovout'].to_nibabel(),
        display_mode='z', cut_coords=np.arange(10, 100, 10))
    p.add_contours(atlasimg.to_nibabel(), linewidths=1)
    p.savefig(os.path.join(registration_dir, 'anat_to_atlas_contours.png'))

    # then register functional channel 1 mean to anat channel 1 mean
    meanfuncimg = ants.image_read(os.path.join(args.dir, args.funcfile))

    ## register functional channel 1 mean to anat channel 1 mean
    args.logger.info('registering functional channel 1 mean to anat channel 1 mean')
    func_to_anat = ants.registration(
        fixed=meananatimg,
        moving=meanfuncimg,
        verbose=args.verbose,
        outprefix=f'{args.transformdir}/func_to_anat_',
        type_of_transform=args.type_of_transform,
        total_sigma=args.total_sigma,
        flow_sigma=args.flow_sigma)

    ## save transformed mean
    func_reg_to_anat_file = os.path.join(
        registration_dir,
        os.path.basename(args.funcfile).replace('.nii', '_space-anat.nii')
    )
    setattr(args, "func_reg_to_anat_file", func_reg_to_anat_file)
    func_to_anat['warpedmovout'].to_filename(func_reg_to_anat_file)
    regtools.overlay_slices(
        meananatimg.numpy(),
        func_to_anat['warpedmovout'].numpy(),
        ltitle='mean anat',
        rtitle='warped func',
        fname=os.path.join(registration_dir, 'func_to_anat_overlay.png'))

    p = plot_anat(func_to_anat['warpedmovout'].to_nibabel(),
        display_mode='z', cut_coords=np.arange(10, 100, 10))
    p.add_contours(meananatimg.to_nibabel(), linewidths=1)
    p.savefig(os.path.join(registration_dir, 'func_to_anat_contours.png'))

    # save inverse
    anat_reg_to_func_file = os.path.join(
        registration_dir,
        os.path.basename(args.anatfile).replace('.nii', '_space-func.nii')
    )
    setattr(args, "anat_reg_to_func_file", anat_reg_to_func_file)
    func_to_anat['warpedfixout'].to_filename(anat_reg_to_func_file)

    # warp atlas to func space
    args.logger.info('warping atlas to functional space')
    atlas_to_func = ants.apply_transforms(moving=atlasimg,
        fixed=meanfuncimg,
        transformlist=anat_to_atlas['invtransforms'] + func_to_anat['invtransforms'],
        whichtoinvert=[True, False, True, False])
    atlas_to_func_file = os.path.join(
        registration_dir,
        os.path.basename(args.atlasfile).replace('.nii', '_space-func.nii')
    )
    setattr(args, "atlas_to_func_file", atlas_to_func_file)
    atlas_to_func.to_filename(atlas_to_func_file)
    regtools.overlay_slices(
        meanfuncimg.numpy(),
        atlas_to_func.numpy(),
        ltitle='mean func',
        rtitle='warped atlas',
        fname=os.path.join(registration_dir, 'atlas_to_func_overlay.png'))

    p = plot_anat(meanfuncimg.to_nibabel(),
        display_mode='z', cut_coords=np.arange(10, 100, 10))
    p.add_contours(atlas_to_func.to_nibabel(), linewidths=1)
    p.savefig(os.path.join(registration_dir, 'atlas_to_func_contours.png'))

    setattr(args, 'completed', True)

    dump_args_to_json(args,
        os.path.join(registration_dir, 'registration.json'))
    args.logger.info("Completed atlas registration")
