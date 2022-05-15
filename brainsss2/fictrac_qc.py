import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from brainsss2.argparse_utils import (
    get_base_parser,
    add_fictrac_qc_arguments,
)


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('fictrac_qc')

    parser = add_fictrac_qc_arguments(parser)

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


def make_2d_hist(fictrac, fictrac_folder, full_id,
                 outdir, save=True, fixed_crop=True):
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
        fname = os.path.join(outdir, name)
        plt.savefig(fname, dpi=100, bbox_inches="tight")


def make_velocity_trace(fictrac, fictrac_folder,
                        full_id, xnew, outdir, save=True,
                        filename="fictrac_velocity_trace.png"):
    plt.figure(figsize=(10, 10))
    plt.plot(xnew / 1000, fictrac["Y"], color="xkcd:dusk")
    plt.ylabel("forward velocity mm/sec")
    plt.xlabel("time, sec")
    plt.title(full_id)
    if save:
        fname = os.path.join(outdir, filename)
        plt.savefig(fname, dpi=100, bbox_inches="tight")
