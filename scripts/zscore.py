import os
import sys
import numpy as np
import argparse
import h5py
from time import time
from hdf5_utils import get_chunk_boundaries
from logging_utils import setup_logging
import logging


def parse_args(input):
    parser = argparse.ArgumentParser(description="zscore an hdf5 file")
    parser.add_argument(
        "-f", "--file", type=str, help="hdf5 file to zscore", required=True
    )
    parser.add_argument(
        "-s", "--stepsize", default=100, type=int, help="stepsize for zscoring"
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

    setup_logging(args, logtype="zscore")

    save_file = args.file.replace(f".{file_extension}", f"_zscore.{file_extension}")
    assert save_file != args.file, "save file cannot be the same as the input file"

    with h5py.File(args.file, "r") as hf:
        data = hf["data"]  # this doesn't actually LOAD the data - it is just a proxy
        dims = np.shape(data)
        chunk_boundaries = get_chunk_boundaries(args, dims[-1])

        logging.info(f"Data shape is {dims}")

        running_sum = np.zeros(dims[:3])
        running_sumofsq = np.zeros(dims[:3])

        logging.info("Calculating meanbrain")
        for chunk_start, chunk_end in chunk_boundaries:
            t0 = time()
            chunk = data[:, :, :, chunk_start:chunk_end]
            running_sum += np.sum(chunk, axis=3)
            logging.info(f"vol: {chunk_start} to {chunk_end} time: {time()-t0}")
        meanbrain = running_sum / dims[-1]

        logging.info("Calculating std")
        for chunk_start, chunk_end in chunk_boundaries:
            t0 = time()
            chunk = data[:, :, :, chunk_start:chunk_end]
            running_sumofsq += np.sum((chunk - meanbrain[..., None]) ** 2, axis=3)
            logging.info(f"vol: {chunk_start} to {chunk_end} time: {time()-t0}")
        final_std = np.sqrt(running_sumofsq / dims[-1])

        # Calculate zscore and save
        # optimize the step size
        if args.stepsize is None:
            chunks = True
        else:
            chunks = (dims[0], dims[1], dims[2], args.stepsize)

        with h5py.File(save_file, "w") as f:
            dset = f.create_dataset("data", dims, dtype="float32", chunks=chunks)

            for chunk_start, chunk_end in chunk_boundaries:
                t0 = time()
                chunk = data[:, :, :, chunk_start:chunk_end]
                running_sumofsq += np.sum((chunk - meanbrain[..., None]) ** 2, axis=3)
                zscored = (chunk - meanbrain[..., None]) / final_std[..., None]
                f["data"][:, :, :, chunk_start:chunk_end] = np.nan_to_num(
                    zscored
                )  # Added nan to num because if a pixel is a constant value (over saturated) will divide by 0
                logging.info(f"vol: {chunk_start} to {chunk_end} time: {time()-t0}")

    logging.info("zscore completed successfully")
