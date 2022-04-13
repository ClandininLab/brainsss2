import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from time import time
import h5py
from scipy.interpolate import interp1d
import argparse
from logging_utils import setup_logging
import logging
from brainsss.visual import (
    load_photodiode,
    get_stimulus_metadata,
    extract_stim_times_from_pd,
)
from brainsss.utils import load_timestamps


def parse_args(input):
    parser = argparse.ArgumentParser(description="process stimulus triggered neural activity")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="base directory for imaging session",
        required=True,
    )
    parser.add_argument(
        "--funcfile",
        type=str,
        help="functional file",
        default="functional_channel_2_moco_zscore_highpass.h5",
    )
    parser.add_argument(
        "--fps", type=float, default=100, help="frame rate of fictrac camera"
    )
    # TODO: What is this? not clear from smooth_and_interp_fictrac
    parser.add_argument(
        "--resolution", default=10, type=float, help="resolution of fictrac data"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument(
        "--outfile",
        default="stim_triggered_turning.png",
        type=str,
        help="output file name",
    )
    return parser.parse_args(input)


def load_slice(brain_path, slice_num):
    with h5py.File(brain_path, "r") as hf:
        single_slice = hf["data"][:, :, slice_num, :]
    return single_slice


def interpolation(slice_num, timestamps, single_slice):
    x = timestamps[:, slice_num]
    f = interp1d(x, single_slice, fill_value="extrapolate")
    xnew = np.arange(0, timestamps[-1, slice_num], 100)
    return f(xnew)


def make_new_stim_timestamps(list_in_ms0):
    return [elem // 100 for elem in list_in_ms0]


def make_chunk_edges(new_stim_timestamps, bin_start=20, bin_end=55):
    #     bin_start = 20 # 2 seconds before the stimulus is presented
    #     bin_end = 55 #5.5 seconds after the stimulus is presented
    chunk_edges = []
    for elem in new_stim_timestamps:
        start = elem - bin_start
        end = elem + bin_end
        chunk = (start, end)
        chunk_edges.append(chunk)
    return chunk_edges


def make_stas(ynew, new_stim_timestamps, chunk_edges, bin_start=20, bin_end=55):
    #     bin_start = 20 # 2 seconds before the stimulus is presented
    #     bin_end = 55 #5.5 seconds after the stimulus is presented
    step_size = bin_end + bin_start + 1
    dims = [np.shape(ynew)[0], np.shape(ynew)[1], step_size]
    running_sum = np.zeros(dims)
    for elem in chunk_edges:
        section = ynew[:, :, elem[0]:elem[1] + 1]
        if np.shape(section)[2] == step_size:  # to catch any not long enough
            running_sum += section
    return running_sum / np.size(new_stim_timestamps)


def save_maxproj_img(file):

    data = np.load(file)
    data = np.nan_to_num(data)

    plt.figure(figsize=(10, 4))
    plt.imshow(np.max(np.max(data, axis=0), axis=-1).T)

    # TODO: use a better method here
    save_file = f'{file[:-3]}png'
    plt.savefig(save_file, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    setup_logging(args, logtype="stim_triggered_avg_neu")

    vision_path = os.path.join(args.dir, "visual")
    assert os.path.exists(vision_path), f"visual path {vision_path} not found"

    t, ft_triggers, pd1, pd2 = load_photodiode(vision_path)
    stimulus_start_times = extract_stim_times_from_pd(pd2, t)

    stim_ids, angles = get_stimulus_metadata(vision_path)
    logging.info(f"Found {len(stim_ids)} presented stimuli.")

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
    logging.info(
        f"starts_angle_0: {len(starts_angle_0)}. starts_angle_180: {len(starts_angle_180)}"
    )
    list_in_ms0 = {
        "0": [i * 10 for i in starts_angle_0],
        "180": [i * 10 for i in starts_angle_180],
    }

    # prep brain
    brain_path = os.path.join(args.dir, "imaging", args.funcfile)
    timestamps = load_timestamps(
        os.path.join(args.dir, "imaging"), file="functional.xml"
    )

    with h5py.File(brain_path, "r") as hf:
        dims = np.shape(hf["data"])

    for angle in list_in_ms0:

        stas = []
        for slice_num in range(dims[2]):
            t0 = time()
            single_slice = load_slice(brain_path, slice_num)
            ynew = interpolation(slice_num, timestamps, single_slice)
            new_stim_timestamps = make_new_stim_timestamps(list_in_ms0[angle])
            chunk_edges = make_chunk_edges(new_stim_timestamps)
            sta = make_stas(ynew, new_stim_timestamps, chunk_edges)
            stas.append(sta)
            logging.info(f"Slice: {slice_num}. Duration: {time()-t0}")
        stas_array = np.asarray(stas)

        # SAVE STA
        savedir = os.path.join(args.dir, "STA")
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savefile = os.path.join(savedir, f"sta_{angle}.npy")
        np.save(savefile, stas_array)

        # SAVE PNG
        save_maxproj_img(savefile)
