# pyright: reportMissingImports=false

import sys
import os
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.bleaching_qc import (
    parse_args,
    load_data,
    get_bleaching_curve
)
from brainsss2.preprocess_utils import check_for_existing_files


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype="bleaching_qc")
    outdir = os.path.join(args.dir, "QC")
    required_files = [
        "bleaching.png"]
    check_for_existing_files(args, outdir, required_files)

    args.logger.info(f"loading data from {args.dir}")
    data_mean = load_data(args)
    if data_mean is None or len(data_mean) == 0:
        raise ValueError('No data found')

    args.logger.info("getting bleaching curve")

    args.logger.info(f'saving results to {outdir}')
    save_file = get_bleaching_curve(data_mean, outdir)
