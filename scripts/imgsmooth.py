# create and save mean image to/from nii/h5
# pyright: reportMissingImports=false

import sys
from brainsss2.imgmath import imgmath, parse_args  # noqa


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f'spatially smoothing {args.file}')
    meanimg = imgmath(args.file, 'smooth',
        args.verbose, args.outfile_type, args.stepsize, args.fwhm)
