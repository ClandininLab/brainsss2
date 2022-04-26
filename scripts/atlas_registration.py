# compute registration between anatomical mean and atlas
# pyright: reportMissingImports=false


import os
import sys
import ants
from argparse_utils import add_moco_arguments
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, os.path.realpath("../brainsss"))
sys.path.insert(0, os.path.realpath("../brainsss/scripts"))
from argparse_utils import get_base_parser, add_moco_arguments # noqa
from logging_utils import setup_logging # noqa
from imgmean import imgmean # noqa


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('anatomical_registration')
    parser = add_moco_arguments(parser)

    # need to add this manually to procesing steps in order to make required
    parser.add_argument(
        '--funcfile',
        type=str,
        help='func file to be registered (usually the mean)',
        required=True)
    parser.add_argument(
        '--atlasfile',
        type=str,
        help='atlas file',
        required=True)
    parser.add_argument(
        '--transformdir',
        type=str,
        help='directory to save transforms')
    parser.add_argument(
        '--maskthresh',
        type=float,
        default=.01,
        help='threshold for masking'
    )
    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    assert os.path.exists(args.funcfile), f"funcfile {args.funcfile} does not exist"
    assert os.path.exists(args.atlasfile), f"atlas file {args.atlasfile} does not exist"

    setattr(args, 'dir', '/'.join(args.funcfile.split('/')[:-2]))
    if 'basedir' not in args or args.basedir is None:
        args.basedir = os.path.dirname(args.funcfile)

    args = setup_logging(args, logtype="atlasreg")

    if args.transformdir is None:
        args.transformdir = os.path.join(args.dir, "transforms")

    if not os.path.exists(args.transformdir):
        os.mkdir(args.transformdir)

    filestem = os.path.basename(args.funcfile).split('.')[0]

    atlasimg = ants.image_read(args.atlasfile)
    print(atlasimg)

    funcimg = ants.image_read(args.funcfile)
    print(funcimg)

    registration = ants.registration(
        fixed=atlasimg,
        moving=funcimg,
        verbose=args.verbose,
        outprefix=f'{args.transformdir}/atlas_to_{filestem}_',
        type_of_transform=args.type_of_transform,
        total_sigma=args.total_sigma,
        flow_sigma=args.flow_sigma)

    outdir = os.path.join(args.dir, 'atlas')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    ants.image_write(
        registration['warpedfixout'],
        os.path.join(outdir, 'atlas_warped_to_functional_space.nii.gz'))

    # save the mask image
    maskimg = registration['warpedfixout'].clone().threshold_image(
        low_thresh=args.maskthresh, binary=True)

    maskimg.to_filename(os.path.join(outdir, 'atlasmask_warped_to_functional_space.nii.gz'))
