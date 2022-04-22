# pyright: reportMissingImports=false

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import logging
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, os.path.realpath("../brainsss"))
sys.path.insert(0, os.path.realpath("../brainsss/scripts"))
from argparse_utils import get_base_parser # noqa
from logging_utils import setup_logging # noqa


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('bleaching_qc')

    # need to add this manually to procesing steps in order to make required
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        help='func directory',
        required=True)

    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()

    return args


def load_data(args):
    """determine directory type and load data"""
    files = [f.as_posix() for f in Path(args.dir).glob("*_channel*.nii")]
    data_mean = {}
    for file in files:
        if args.verbose:
            print(f"processing {file}")
        if os.path.exists(file):
            brain = np.asarray(nib.load(file).get_fdata(), dtype="uint16")
            data_mean[file] = np.mean(brain, axis=(0, 1, 2))
            del brain
        else:
            print(f"Not found (skipping){file}")
    return data_mean


def get_bleaching_curve(data_mean, outdir):
    """get bleaching curve"""

    plt.rcParams.update({"font.size": 24})
    _ = plt.figure(figsize=(10, 10))
    signal_loss = {}
    for file in data_mean:
        xs = np.arange(len(data_mean[file]))
        color = "k"
        if file[-1] == "1":
            color = "red"
        if file[-1] == "2":
            color = "green"
        plt.plot(data_mean[file], color=color, label=file)
        linear_fit = np.polyfit(xs, data_mean[file], 1)
        plt.plot(np.poly1d(linear_fit)(xs), color="k", linewidth=3, linestyle="--")
        signal_loss[file] = linear_fit[0] * len(data_mean[file]) / linear_fit[1] * -100
    plt.xlabel("Frame Num")
    plt.ylabel("Avg signal")
    loss_string = ""
    for file in data_mean:
        loss_string = loss_string + file + " lost" + f"{int(signal_loss[file])}" + "%\n"
    plt.title(loss_string, ha="center", va="bottom")

    save_file = os.path.join(outdir, "bleaching.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")


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
