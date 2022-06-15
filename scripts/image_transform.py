# transform between spaces using registration outputs

import sys
import os
import numpy as np
import nibabel as nib
import json
import ants
from pathlib import Path
from brainsss2.argparse_utils import get_base_parser


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('image transformation')

    # need to add this manually to procesing steps in order to make required
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='file to be transformed',
        required=True)
    parser.add_argument(
        '--regdir',
        type=str,
        required=True,
        help='directory containing registration outputs')
    parser.add_argument(
        '--inspace',
        type=str, choices=['func', 'anat', 'atlas'],
        help='space for input image (will try to determine if unspecified')
    parser.add_argument(
        '--outspace',
        required=True,
        type=str, choices=['func', 'anat', 'atlas'],
        help='space for output image')
    parser.add_argument(
        '--outfile',
        type=str,
        help='output file name (defaults to infile_space-outspace.nii'
    )

    group = parser.add_argument_group('ANTs registration options')
    group.add_argument('--interpolation_method', type=str,
        default='lanczosWindowedSinc')

    if allow_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args


def find_reg_files(args):
    """
    find the registration files
    assumes specific naming of files from atlas registration
    """
    regdir = Path(args.regdir)
    regfiles = {'func': None, 'anat': None, args.reg_params['atlasname']: None}
    for k in regfiles:
        potential_match = [i.as_posix() for i in regdir.glob(f'*_space-{k}.nii')]
        if len(potential_match) > 0:
            regfiles[k] = potential_match[0]
    setattr(args, 'regfiles', regfiles)

    # get transforms
    transformdir = Path(os.path.join(args.regdir, 'transforms'))
    transforms = {'func_to_anat': {},
                  f'anat_to_{args.reg_params["atlasname"]}': {}}
    for k in transforms:
        for potential_match in [i.as_posix() for i in transformdir.glob(f'{k}*')]:
            for transformtype in ['0GenericAffine', '1Warp', '1InverseWarp']:
                if transformtype in potential_match:
                    transforms[k][transformtype] = potential_match
        setattr(args, 'transforms', transforms)
    return(args)


def guess_space(args):
    """
    guess the space of the file
    """
    img = nib.load(args.file)
    # try to load images from registration dir
    space_shape = {}
    for k, filename in args.regfiles.items():
        if filename is not None:
            space_shape[k] = nib.load(filename).shape
            if np.linalg.norm(np.array(space_shape[k]) - np.array(img.shape)) == 0:
                setattr(args, 'inspace', k)
                return(args)

    # raise exception if nothing matches
    raise ValueError('input image does not match shape of any space')


def transform_image(args):
    """
    transform image between spaces

    fix_atlas_spacing: if True, set spacing to 2 mm iso
    """

    infile_ants = ants.image_read(args.file)
    fixed_ants = ants.image_read(args.regfiles[args.outspace])

    # determine transforms
    if args.inspace == 'func' and args.outspace == 'anat':
        transforms = [
            args.transforms['func_to_anat']['1Warp'],
            args.transforms['func_to_anat']['0GenericAffine'],
        ]
        whichtoinvert = [False, False]
    elif args.inspace == 'anat' and args.outspace == 'func':
        transforms = [
            args.transforms['func_to_anat']['0GenericAffine'],
            args.transforms['func_to_anat']['1InverseWarp'],
        ]
        whichtoinvert = [True, False]
    elif args.inspace == 'anat' and args.outspace == args.reg_params["atlasname"]:
        transforms = [
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['1Warp'],
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['0GenericAffine']
        ]
        whichtoinvert = [False, False]
    elif args.inspace == args.reg_params["atlasname"] and args.outspace == 'anat':
        transforms = [
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['0GenericAffine'],
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['1InverseWarp']
        ]
        whichtoinvert = [True, False]
    elif  args.inspace == args.reg_params["atlasname"] and args.outspace == 'func':
        #     func_to_atlas_transforms = anat_to_atlas['invtransforms'] + func_to_anat['invtransforms']  # noqa
        transforms = [
            args.transforms['func_to_anat']['0GenericAffine'],
            args.transforms['func_to_anat']['1InverseWarp'],
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['0GenericAffine'],
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['1InverseWarp']
        ]
        whichtoinvert = [True, False, True, False]
    elif  args.inspace == 'func' and args.outspace == args.reg_params["atlasname"]:
        #     func_to_atlas_transforms = anat_to_atlas['invtransforms'] + func_to_anat['invtransforms']  # noqa
        transforms = [
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['0GenericAffine'],
            args.transforms[f'anat_to_{args.reg_params["atlasname"]}']['1InverseWarp'],
            args.transforms['func_to_anat']['0GenericAffine'],
            args.transforms['func_to_anat']['1InverseWarp'],
        ]
        whichtoinvert = [False, False, False, False]
  
    else:
        raise Exception('not implemented')
    
    transformed_ants = ants.apply_transforms(
        fixed_ants, infile_ants, transforms,
        whichtoinvert=whichtoinvert,
        interpolator=args.interpolation_method)

    if args.outfile is None:
        args.outfile = args.file.replace('.nii', f'_space-{args.outspace}.nii')
    assert args.outfile != args.file, 'output file name should be different'
    # ants.image_write(transformed_ants, outfile)
    fixed_img = nib.load(args.regfiles[args.outspace])
    out_img = nib.Nifti1Image(transformed_ants.numpy(),
        fixed_img.affine, fixed_img.header)
    out_img.to_filename(args.outfile)
    print(f'transformed image saved to {args.outfile}')


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if 'nii' not in args.file:
        raise ValueError('only nii files are currently supported')

    if not os.path.exists(os.path.join(args.regdir, 'registration.json')):
        raise ValueError('registration.json not found')
    with open(os.path.join(args.regdir, 'registration.json')) as f:
        setattr(args, 'reg_params', json.load(f))

    args = find_reg_files(args)
    if args.inspace is None:
        args = guess_space(args)
    print(f'input space: {args.inspace}')
    if args.outspace == 'atlas':
        args.outspace = args.reg_params['atlasname']
    print(f'output space: {args.outspace}')
    if args.inspace == args.outspace:
        print('input and output spaces are the same, nothing to do')
        sys.exit(0)

    transform_image(args)