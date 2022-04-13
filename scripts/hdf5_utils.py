# hdf5 utils

import os
import h5py


def make_empty_h5(directory, file, brain_dims, save_type="curr_dir", stepsize=None):
    """for optimal speed, using chunk dim4 equal to dimension 4 of data chunks"""
    if stepsize is None:
        chunks = True
    else:
        chunks = (brain_dims[0], brain_dims[1], brain_dims[2], stepsize)

    assert save_type in [
        "curr_dir",
        "parent_dir",
    ], "save_type must be either curr_dir or parent_dir"
    if save_type == "curr_dir":
        moco_dir = os.path.join(directory, "moco")
    elif save_type == "parent_dir":
        directory = os.path.dirname(directory)  # go back one directory
        moco_dir = os.path.join(directory, "moco")

    if not os.path.exists(moco_dir):
        os.mkdir(moco_dir)

    savefile = os.path.join(moco_dir, file)
    with h5py.File(savefile, "w") as f:
        _ = f.create_dataset("data", brain_dims, dtype="float32", chunks=chunks)
    return moco_dir, savefile


def get_chunk_boundaries(args, n_timepoints):
    """get chunk boundaries"""
    chunk_starts = list(range(0, n_timepoints, args.stepsize))
    chunk_ends = list(range(args.stepsize, n_timepoints + args.stepsize, args.stepsize))
    chunk_ends = [x if x < n_timepoints else n_timepoints for x in chunk_ends]
    return list(zip(chunk_starts, chunk_ends))
