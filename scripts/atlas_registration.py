# compute registration between anatomical mean and atlas
# pyright: reportMissingImports=false


import os
import sys
import ants
import logging
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
    group.add_argument('--total_sigma', type=float, default=1,
        help='total sigma for registration - higher values will restrict the amount of deformation allowed')
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

    meananatimg = ants.image_read(os.path.join(args.dir, args.anatfile))

    ## register anat channel 1 mean to atlas
    args.logger.info('registering anat channel 1 mean to atlas')
    anat_to_atlas = ants.registration(
        fixed=atlasimg,
        moving=meananatimg,
        verbose=args.verbose,
        outprefix=f'{args.transformdir}/anat_to_atlas_',
        type_of_transform=args.type_of_transform,
        total_sigma=args.total_sigma,
        flow_sigma=args.flow_sigma)

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
        transformlist=anat_to_atlas['invtransforms'] + func_to_anat['invtransforms'])
    atlas_to_func_file = os.path.join(
        registration_dir,
        os.path.basename(args.atlasfile).replace('.nii', '_space-func.nii')
    )
    setattr(args, "atlas_to_func_file", atlas_to_func_file)
    atlas_to_func.to_filename(atlas_to_func_file)

    setattr(args, 'completed', True)

    dump_args_to_json(args,
        os.path.join(registration_dir, 'registration.json'))
    args.logger.info("Completed atlas registration")
