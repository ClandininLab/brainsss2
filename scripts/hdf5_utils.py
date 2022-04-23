# hdf5 utils
# pyright: reportMissingImports=false

import os
import h5py


def make_empty_h5(directory, file, brain_dims, stepsize=None):
    """for optimal speed, using chunk dim4 equal to dimension 4 of data chunks"""
    if stepsize is None:
        chunks = True
    else:
        chunks = (brain_dims[0], brain_dims[1], brain_dims[2], stepsize)

    if not os.path.exists(directory):
        os.mkdir(directory)

    savefile = os.path.join(directory, file)
    with h5py.File(savefile, "w") as f:
        _ = f.create_dataset("data", brain_dims, dtype="float32", chunks=chunks)
    return directory, savefile


def get_chunk_boundaries(args, n_timepoints):
    """get chunk boundaries"""
    chunk_starts = list(range(0, n_timepoints, args.stepsize))
    chunk_ends = list(range(args.stepsize, n_timepoints + args.stepsize, args.stepsize))
    chunk_ends = [x if x < n_timepoints else n_timepoints for x in chunk_ends]
    return list(zip(chunk_starts, chunk_ends))
