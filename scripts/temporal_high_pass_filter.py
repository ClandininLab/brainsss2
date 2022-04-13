import os
import sys
import numpy as np
import argparse
import h5py
from scipy.ndimage import gaussian_filter1d
from hdf5_utils import get_chunk_boundaries
from logging_utils import setup_logging
import logging


def parse_args(input):
    parser = argparse.ArgumentParser(
        description="temporally highpass filter an hdf5 file"
    )
    parser.add_argument(
        "-f", "--file", type=str, help="hdf5 file to zscore", required=True
    )
    parser.add_argument(
        "--sigma", default=200, type=float, help="sigma for gaussian filter"
    )
    parser.add_argument(
        "-s", "--stepsize", default=2, type=int, help="stepsize for zscoring"
    )
    parser.add_argument("-l", "--logdir", type=str, help="directory to save log file")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    return parser.parse_args(input)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    setattr(args, "dir", os.path.dirname(args.file))

    assert os.path.exists(args.file), "file does not exist"
    file_extension = args.file.split(".")[-1]
    assert file_extension in ["h5", "hdf5"], "file must be hdf5"

    setup_logging(args, logtype="highpass")

    save_file = args.file.replace(f".{file_extension}", f"_highpass.{file_extension}")
    assert save_file != args.file, "save file cannot be the same as the input file"

    with h5py.File(args.file, "r") as hf:
        data = hf["data"]  # this doesn't actually LOAD the data - it is just a proxy
        dims = np.shape(data)
        chunk_boundaries = get_chunk_boundaries(args, dims[-2])
        logging.info(f"Data shape is {dims}")

        with h5py.File(save_file, "w") as f:
            dset = f.create_dataset("data", dims, dtype="float32", chunks=True)

            for chunk_num, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
                logging.info(f"Processing chunk {chunk_num} of {len(chunk_boundaries)}")
                chunk = data[:, :, chunk_start:chunk_end, :]
                chunk_mean = np.mean(chunk, axis=-1)

                smoothed_chunk = gaussian_filter1d(
                    chunk, sigma=args.sigma, axis=-1, truncate=1
                )

                # Apply Smooth Correction
                chunk_high_pass = (
                    chunk - smoothed_chunk + chunk_mean[:, :, :, None]
                )  # need to add back in mean to preserve offset

                f["data"][:, :, chunk_start:chunk_end, :] = chunk_high_pass

    logging.info("high pass done")
