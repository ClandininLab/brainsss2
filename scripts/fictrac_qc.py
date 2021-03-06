import numpy as np
import sys
import os
from brainsss2.logging_utils import setup_logging
from brainsss2.fictrac_qc import (
    parse_args,
    make_2d_hist,
    make_velocity_trace
)
from brainsss2.fictrac import load_fictrac, smooth_and_interp_fictrac
from brainsss2.preprocess_utils import check_for_existing_files


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype="fictrac_qc")
    print('logger handlers', args.logger.handlers)
    print(args)

    fictrac_dir = os.path.join(args.dir, "fictrac")
    outdir = os.path.join(args.dir, 'QC')

    # check for necessary files
    required_files = [
        "fictrac_2d_hist_fixed.png",
        "fictrac_velocity_trace.png"]
    check_for_existing_files(args, outdir, required_files)

    args.logger.info(f'running fictrac_qc.py on {fictrac_dir}')
    if not os.path.exists(fictrac_dir):
        args.logger.info(f"fictrac directory {fictrac_dir} not found, skipping fictrac_qc")
        sys.exit(0)

    try:
        fictrac_raw = load_fictrac(fictrac_dir)
    except FileNotFoundError:
        args.logger.error(f"fictrac directory {fictrac_dir} not found, skipping fictrac_qc")
        sys.exit(0)

    # fly = os.path.split(os.path.split(directory)[0])[1]
    # expt = os.path.split(directory)[1]
    # TODO: This is making some assumptions about the naming convention...
    full_id = ", ".join(args.dir.split("/")[-3:-1])

    expt_len = (fictrac_raw.shape[0] / args.fps) * 1000
    behaviors = ["dRotLabY", "dRotLabZ"]
    fictrac = {}
    for behavior in behaviors:
        if behavior == "dRotLabY":
            short = "Y"
        elif behavior == "dRotLabZ":
            short = "Z"
        fictrac[short] = smooth_and_interp_fictrac(
            fictrac_raw, args.fps, args.resolution, expt_len, behavior
        )
    xnew = np.arange(0, expt_len, args.resolution)

    make_2d_hist(fictrac, fictrac_dir, full_id, outdir, save=True,
                 fixed_crop=True)
    make_2d_hist(fictrac, fictrac_dir, full_id, outdir, save=True,
                 fixed_crop=False)
    make_velocity_trace(fictrac, fictrac_dir, full_id, xnew,
                        outdir, save=True)
