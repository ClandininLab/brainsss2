import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from brainsss import smooth_and_interp_fictrac, load_fictrac
import argparse
from logging_utils import setup_logging


def parse_args(input):
    parser = argparse.ArgumentParser(description="process fictrac qc")
    parser.add_argument(
        "-d", "--dir", type=str, help="directory containing fictrac data", required=True
    )
    parser.add_argument(
        "--fps", type=float, default=100, help="frame rate of fictrac camera"
    )
    # TODO: What is this? not clear from smooth_and_interp_fictrac
    parser.add_argument("--resolution", type=float, help="resolution of fictrac data")
    parser.add_argument('-v', "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args(input)
    return args


def make_2d_hist(fictrac, fictrac_folder, full_id, save=True, fixed_crop=True):
    plt.figure(figsize=(10, 10))
    norm = mpl.colors.LogNorm()
    plt.hist2d(fictrac["Y"], fictrac["Z"], bins=100, cmap="Blues", norm=norm)
    plt.ylabel("Rotation, deg/sec")
    plt.xlabel("Forward, mm/sec")
    plt.title("Behavior 2D hist {}".format(full_id))
    plt.colorbar()
    name = "fictrac_2d_hist.png"
    if fixed_crop:
        plt.ylim(-400, 400)
        plt.xlim(-10, 15)
        name = "fictrac_2d_hist_fixed.png"
    if save:
        fname = os.path.join(fictrac_folder, name)
        plt.savefig(fname, dpi=100, bbox_inches="tight")


def make_velocity_trace(fictrac, fictrac_folder, full_id, xnew, save=True):
    plt.figure(figsize=(10, 10))
    plt.plot(xnew / 1000, fictrac["Y"], color="xkcd:dusk")
    plt.ylabel("forward velocity mm/sec")
    plt.xlabel("time, sec")
    plt.title(full_id)
    if save:
        fname = os.path.join(fictrac_folder, "velocity_trace.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    setup_logging(args, logtype="fictrac_qc")

    fictrac_raw = load_fictrac(args.dir)

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

    make_2d_hist(fictrac, args.dir, full_id, save=True, fixed_crop=True)
    make_2d_hist(fictrac, args.dir, full_id, save=True, fixed_crop=False)
    make_velocity_trace(fictrac, args.dir, full_id, xnew, save=True)
