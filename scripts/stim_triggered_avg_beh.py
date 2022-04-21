import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# THIS A HACK FOR DEVELOPMENT
sys.path.append("../brainsss")
sys.path.append("../brainsss/scripts")
from logging_utils import setup_logging
import logging
from visual import (
    load_photodiode,
    get_stimulus_metadata,
    extract_stim_times_from_pd,
)
from argparse_utils import (
    get_base_parser,
    add_fictrac_qc_arguments,
)
from fictrac import smooth_and_interp_fictrac, load_fictrac


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


def plot_avg_trace(fictrac, starts_angle_0, starts_angle_180,
                   outdir):
    """
    Plot the average fictrac trace for the stimulus-triggered average
    """
    pre_window = 200  # in units of 10ms
    post_window = 300

    traces = []
    for i in range(len(starts_angle_0)):
        trace = fictrac["Z"][
            starts_angle_0[i] - pre_window:starts_angle_0[i] + post_window
        ]
        if (
            len(trace) == pre_window + post_window
        ):  # this handles fictrac that crashed or was aborted or some bullshit
            traces.append(trace)
    mean_trace_0 = np.mean(np.asarray(traces), axis=0)

    traces = []
    for i in range(len(starts_angle_180)):
        trace = fictrac["Z"][
            starts_angle_180[i] - pre_window:starts_angle_180[i] + post_window
        ]
        if (
            len(trace) == pre_window + post_window
        ):  # this handles fictrac that crashed or was aborted or some bullshit
            traces.append(trace)
    mean_trace_180 = np.mean(np.asarray(traces), axis=0)

    plt.figure(figsize=(10, 10))
    xs = np.arange(-pre_window, post_window) * 10
    plt.plot(xs, mean_trace_0, color="r", linewidth=5)
    plt.plot(xs, mean_trace_180, color="green", linewidth=5)
    plt.axvline(0, color="grey", lw=3, linestyle="--")  # stim appears
    plt.axvline(1000, color="k", lw=3, linestyle="--")  # stim moves
    plt.axvline(1500, color="grey", lw=3, linestyle="--")  # grey
    plt.ylim(-50, 50)
    plt.legend(['mean_trace_0', 'mean_trace_180'])
    plt.xlabel("Time, ms")
    plt.ylabel("Angular Velocity")

    fname = os.path.join(outdir, 'stim_triggered_turning.png')
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    logging.info(f"saved average trace to {fname}")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype='stim_triggered_avg_beh')
    logging.info(f'Running stim_triggered_avg_beh on {args.dir}')
    vision_path = os.path.join(args.dir, "visual")
    logging.info(f'loading vision data from {vision_path}')

    assert os.path.exists(vision_path), f"vision_path {vision_path} does not exist"

    t, ft_triggers, pd1, pd2 = load_photodiode(vision_path)
    stimulus_start_times = extract_stim_times_from_pd(pd2, t)

    stim_ids, angles = get_stimulus_metadata(vision_path)
    logging.info(F"Found {len(stim_ids)} presented stimuli.")

    # *100 puts in units of 10ms, which will match fictrac
    starts_angle_0 = [
        int(stimulus_start_times[i] * 100)
        for i in range(len(stimulus_start_times))
        if angles[i] == 0
    ]
    starts_angle_180 = [
        int(stimulus_start_times[i] * 100)
        for i in range(len(stimulus_start_times))
        if angles[i] == 180
    ]
    logging.info(F"starts_angle_0: {len(starts_angle_0)}. starts_angle_180: {len(starts_angle_180)}")

    # PREP FICTRAC
    fictrac_path = os.path.join(args.dir, "fictrac")
    fictrac_raw = load_fictrac(fictrac_path)

    expt_len = fictrac_raw.shape[0] / args.fps * 1000
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

    outdir = os.path.join(args.dir, "QC")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plot_avg_trace(fictrac, starts_angle_0, starts_angle_180,
                   outdir)
