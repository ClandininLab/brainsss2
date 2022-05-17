# functions for performing image math operations

# pyright: reportMissingImports=false

import sys
from brainsss2.imgmath import imgmath, parse_args  # noqa
from brainsss2.logging_utils import setup_logging # noqa
import logging


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype='imgmath')

    logging.info(f'applying {args.operation} to {args.file}')
    _ = imgmath(args.file, args.operation,
        args.verbose, args.outfile_type, args.stepsize,
        fwhm=args.fwhm)
