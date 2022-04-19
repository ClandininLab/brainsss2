import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from brainsss import smooth_and_interp_fictrac
# THIS A HACK FOR DEVELOPMENT
sys.path.append("../brainsss")
sys.path.append("../brainsss/scripts")
from argparse_utils import (
    get_base_parser,
    add_fictrac_qc_arguments,
)
from logging_utils import setup_logging
import logging



def load_fictrac(directory, file="fictrac.dat"):
    """ Loads fictrac data from .dat file that fictrac outputs.

    To-do: change units based on diameter of ball etc.
    For speed sanity check, instead remove bad frames so we don't have to throw out whole trial.

    Parameters
    ----------
    directory: string of full path to file
    file: string of file name

    Returns
    -------
    fictrac_data: pandas dataframe of all parameters saved by fictrac """

    for item in os.listdir(directory):
        if ".dat" in item:
            file = item

    fictrac_file = os.path.join(directory, file)
    if not os.path.exists(fictrac_file):
        raise FileNotFoundError(
            f"Fictrac data file not found: {fictrac_file}. ")
    with open(fictrac_file, "r") as f:
        df = pd.DataFrame(line.rstrip().split() for line in f)

        # Name columns
        df = df.rename(
            index=str,
            columns={
                0: "frameCounter",
                1: "dRotCamX",
                2: "dRotCamY",
                3: "dRotCamZ",
                4: "dRotScore",
                5: "dRotLabX",
                6: "dRotLabY",
                7: "dRotLabZ",
                8: "AbsRotCamX",
                9: "AbsRotCamY",
                10: "AbsRotCamZ",
                11: "AbsRotLabX",
                12: "AbsRotLabY",
                13: "AbsRotLabZ",
                14: "positionX",
                15: "positionY",
                16: "heading",
                17: "runningDir",
                18: "speed",
                19: "integratedX",
                20: "integratedY",
                21: "timeStamp",
                22: "sequence",
            },
        )

        # Remove commas
        for column in df.columns.values[:-1]:
            df[column] = [float(x[:-1]) for x in df[column]]

        fictrac_data = df

    # sanity check for extremely high speed (fictrac failure)
    speed = np.asarray(fictrac_data["speed"])
    max_speed = np.max(speed)
    if max_speed > 10:
        raise ValidationError(
            "Fictrac ball tracking failed (reporting impossibly high speed)."
        )
    return fictrac_data


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('fictrac_qc')

    parser = add_fictrac_qc_arguments(parser)

    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()
    return args


def make_2d_hist(fictrac, fictrac_folder, full_id, save=True, fixed_crop=True):
    plt.figure(figsize=(10, 10))
    norm = mpl.colors.LogNorm()
    plt.hist2d(fictrac["Y"], fictrac["Z"], bins=100, cmap="Blues", norm=norm)
    plt.ylabel("Rotation, deg/sec")
    plt.xlabel("Forward, mm/sec")
    plt.title(f"Behavior 2D hist {full_id}")
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

    print(args)

    fictrac_dir = os.path.join(args.dir, "fictrac")
    logging.info(f'running fictrac_qc.py on {fictrac_dir}')
    if not os.path.exists(fictrac_dir):
        logging.info(f"fictrac directory {fictrac_dir} not found, skipping fictrac_qc")
        sys.exit(0)

    fictrac_raw = load_fictrac(fictrac_dir)

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

    make_2d_hist(fictrac, fictrac_dir, full_id, save=True, fixed_crop=True)
    make_2d_hist(fictrac, fictrac_dir, full_id, save=True, fixed_crop=False)
    make_velocity_trace(fictrac, fictrac_dir, full_id, xnew, save=True)
