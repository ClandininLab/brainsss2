# pyright: reportMissingImports=false

import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from brainsss2.argparse_utils import get_base_parser # noqa
from brainsss2.logging_utils import setup_logging # noqa


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
    else:
        args = parser.parse_args()

    return args


def load_data(args):
    """determine directory type and load data"""
    imaging_dir = os.path.join(args.dir, 'imaging')
    files = [f.as_posix() for f in Path(imaging_dir).glob("*_channel*.nii") if 'mean' not in f.as_posix()]
    assert len(files) > 0, "No imaging files found"
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

    plt.rcParams.update({"font.size": 12})
    _ = plt.figure(figsize=(10, 10))
    signal_loss = {}
    for file, meandata in data_mean.items():
        xs = np.arange(len(meandata))
        color = "k"
        if 'channel_1' in file:
            color = "red"
            label = 'ch1'
        else:
            color = "green"
            label = 'ch2'
        plt.plot(data_mean[file], color=color, label=label)
        linear_fit = np.polyfit(xs, meandata, 1)
        plt.plot(np.poly1d(linear_fit)(xs), color="k", linewidth=3, linestyle="--")
        signal_loss[file] = linear_fit[0] * len(meandata) / linear_fit[1] * -100
    plt.xlabel("Frame Num")
    plt.ylabel("Avg signal")
    loss_string = ""
    for file in data_mean:
        loss_string = loss_string + file + " lost" + f"{int(signal_loss[file])}" + "%\n"
    plt.title(loss_string, ha="center", va="bottom")
    plt.legend(loc=2)
    plt.tight_layout()

    save_file = os.path.join(outdir, "bleaching.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    return save_file
