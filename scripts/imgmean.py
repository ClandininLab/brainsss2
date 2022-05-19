# create and save mean image to/from nii/h5
# pyright: reportMissingImports=false

import sys
import os
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, os.path.realpath("../brainsss"))
from imgmath import imgmath, parse_args  # noqa


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f'making mean brain for {args.file}')
    meanimg = imgmath(args.file, 'mean',
        args.verbose, args.outfile_type, args.stepsize)
