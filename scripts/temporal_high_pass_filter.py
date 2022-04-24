# pyright: reportMissingImports=false

import os
import sys
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
from hdf5_utils import get_chunk_boundaries
import logging
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, os.path.realpath("../brainsss"))
sys.path.insert(0, os.path.realpath("../brainsss/scripts"))
from argparse_utils import get_base_parser, add_highpassfilter_arguments # noqa
from logging_utils import setup_logging # noqa


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('highpassfilter')
    parser = add_highpassfilter_arguments(parser)
    # need to add this manually to procesing steps in order to make required
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        help='func directory',
        required=True)
    # add this here since it's already part of moco args
    parser.add_argument(
        "-s", "--stepsize", default=2, type=int, help="stepsize for chunking"
    )

    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    setattr(args, 'file', os.path.join(args.dir, args.hpf_filename))
    assert os.path.exists(args.file), f"file {args.file} does not exist"
    file_extension = args.file.split(".")[-1]
    assert file_extension in ["h5", "hdf5"], "file must be hdf5"

    args = setup_logging(args, logtype="highpass")

    save_file = args.file.replace(f".{file_extension}", f"_hpf.{file_extension}")
    assert save_file != args.file, "save file cannot be the same as the input file"

    with h5py.File(args.file, "r") as hf:
        data = hf["data"]  # this doesn't actually LOAD the data - it is just a proxy
        dims = np.shape(data)
        chunk_boundaries = get_chunk_boundaries(args.stepsize, dims[-2])
        logging.info(f"Data shape is {dims}")

        with h5py.File(save_file, "w") as f:
            dset = f.create_dataset("data", dims, dtype="float32", chunks=True)

            for chunk_num, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
                logging.info(f"Processing chunk {chunk_num} of {len(chunk_boundaries)}")
                # chunk across slices to maintain timeseries continuity
                chunk = data[:, :, chunk_start:chunk_end, :]
                chunk_mean = np.mean(chunk, axis=-1)

                # use subtractive strategy - smooth the timeseries and then subtract
                # the smoothed timeseries from the original timeseries
                smoothed_chunk = gaussian_filter1d(
                    chunk, sigma=args.sigma, axis=-1, truncate=1
                )

                chunk_high_pass = (
                    chunk - smoothed_chunk + chunk_mean[:, :, :, None]
                )  # need to add back in mean to preserve offset

                f["data"][:, :, chunk_start:chunk_end, :] = chunk_high_pass

    logging.info("high pass done")
