import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from time import time
import h5py
import logging
import shutil
from scipy.interpolate import interp1d
from brainsss2.argparse_utils import get_base_parser
from brainsss2.logging_utils import setup_logging
from brainsss2.visual import load_photodiode, extract_stim_times_from_pd, get_stimulus_metadata
from brainsss2.utils import load_timestamps


def parse_args(args, allow_unknown=True):
    parser = get_base_parser('Stimulus-triggered averaging')
    parser.add_argument('-d', '--dir', type=str, default=None,
        help='Path to functional data', required=True)
    parser.add_argument('-f', '--filename', type=str,
        default='preproc/functional_channel_2_moco_smooth-2.0mu.h5')
    parser.add_argument('--outdir', type=str, default=None,
        help='Path to output directory')
    parser.add_argument('--overwrite', action='store_true', default=False,
        help='Overwrite existing files')
    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()
    return args


def prep_visual_stimuli(args):
    vision_path = os.path.join(args.dir, 'visual')
    assert os.path.exists(vision_path), f'{vision_path} does not exist'

    ### Load Photodiode ###
    t, ft_triggers, pd1, pd2 = load_photodiode(vision_path)
    stimulus_start_times = extract_stim_times_from_pd(pd2, t)

    ### Get Metadata ###
    stim_ids, angles = get_stimulus_metadata(vision_path)
    logging.info(F"Found {len(stim_ids)} presented stimuli.")

    # *100 puts in units of 10ms, which will match fictrac
    starts_angle_0 = [int(stimulus_start_times[i] * 100)
        for i in range(len(stimulus_start_times)) if angles[i] == 0]
    starts_angle_180 = [int(stimulus_start_times[i] * 100)
        for i in range(len(stimulus_start_times)) if angles[i] == 180]
    logging.info(F"starts_angle_0: {len(starts_angle_0)}. starts_angle_180: {len(starts_angle_180)}")
    return({'0': [i * 10 for i in starts_angle_0],
            '180': [i * 10 for i in starts_angle_180]})


def load_slice(brain_path, slice_num):
    with h5py.File(brain_path, 'r') as hf:
        single_slice = hf['data'][:, :, slice_num, :]
    return single_slice


def interpolation(slice_num, timestamps, single_slice):
    x = timestamps[:, slice_num]
    f = interp1d(x, single_slice, fill_value="extrapolate")
    xnew = np.arange(0, timestamps[-1, slice_num], 100)
    ynew = f(xnew)
    return ynew


def make_new_stim_timestamps(list_in_ms0):
    new_stim_timestamps = []
    for elem in list_in_ms0:
        new_elem = (elem // 100)
        new_stim_timestamps.append(new_elem)
    return new_stim_timestamps


def make_chunk_edges(new_stim_timestamps,
                     bin_start=20, bin_end=55):
    #     bin_start = 20 # 2 seconds before the stimulus is presented
    #     bin_end = 55 #5.5 seconds after the stimulus is presented
    chunk_edges = []
    for elem in new_stim_timestamps:
        start = elem - bin_start
        end = elem + bin_end
        chunk = (start, end)
        chunk_edges.append(chunk)
    return chunk_edges


def make_stas(ynew, new_stim_timestamps,
              chunk_edges, bin_start=20, bin_end=55):
    #     bin_start = 20 # 2 seconds before the stimulus is presented
    #     bin_end = 55 #5.5 seconds after the stimulus is presented
    step_size = bin_end + bin_start + 1
    dims = [np.shape(ynew)[0], np.shape(ynew)[1], step_size]
    running_sum = np.zeros(dims)
    for elem in chunk_edges:
        section = ynew[:, :, elem[0]:elem[1] + 1]
        if np.shape(section)[2] == step_size:  # to catch any not long enough
            running_sum += section
    sta = running_sum / np.size(new_stim_timestamps)
    return sta


def save_maxproj_img(file):

    data = np.load(file)
    data = np.nan_to_num(data)

    plt.figure(figsize=(10, 4))
    plt.imshow(np.max(np.max(data, axis=0), axis=-1).T)

    save_file = file[:-3] + 'png'
    plt.savefig(save_file, bbox_inches='tight', dpi=300)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    if args.outdir is None:
        args.outdir = os.path.join(args.dir, 'STA')

    args = setup_logging(args, logtype='STA',
        logdir=args.outdir)

    logging.info(f'saving output to {args.outdir}')

    if os.path.exists(args.outdir) and not args.overwrite:
        logging.info(f'output directory {args.outdir} already exists and overwrite is False')
        sys.exit(0)
    elif os.path.exists(args.outdir) and args.overwrite:
        shutil.rmtree(args.outdir)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    list_in_ms0 = prep_visual_stimuli(args)

    brain_path = os.path.join(args.dir,
        args.filename)
    assert os.path.exists(brain_path)
    timestamps = load_timestamps(
        os.path.join(args.dir, 'imaging'),
        file='functional.xml')

    with h5py.File(brain_path, 'r') as hf:
        dims = np.shape(hf['data'])

    for angle in list_in_ms0.keys():
        stas = []
        for slice_num in range(dims[2]):
            t0 = time()
            single_slice = load_slice(brain_path, slice_num)
            ynew = interpolation(slice_num, timestamps, single_slice)
            new_stim_timestamps = make_new_stim_timestamps(list_in_ms0[angle])
            chunk_edges = make_chunk_edges(new_stim_timestamps)
            sta = make_stas(ynew, new_stim_timestamps, chunk_edges)
            stas.append(sta)
            logging.info(F"Slice: {slice_num}. Duration: {time()-t0}")
        stas_array = np.asarray(stas)

        ### SAVE STA ###
        if not os.path.exists(args.outdir):
            os.mkdir(args.outdir)
        savefile = os.path.join(args.outdir, F'sta_{angle}.npy')
        np.save(savefile, stas_array)

        ### SAVE PNG ###
        save_maxproj_img(savefile)
