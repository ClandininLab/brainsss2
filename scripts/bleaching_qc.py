# pyright: reportMissingImports=false

import sys
import os
import logging
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.bleaching_qc import (
    parse_args,
    load_data,
    get_bleaching_curve
)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype="bleaching_qc")

    logging.info(f"loading data from {args.dir}")
    data_mean = load_data(args)

    logging.info("getting bleaching curve")
    outdir = os.path.join(args.dir, "QC")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    logging.info(f'saving results to {outdir}')
    get_bleaching_curve(data_mean, outdir)
