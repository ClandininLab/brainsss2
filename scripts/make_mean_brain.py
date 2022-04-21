import os
import sys
import numpy as np
import nibabel as nib
import h5py
import argparse
from pathlib import Path
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, os.path.realpath("../brainsss"))
sys.path.insert(0, os.path.realpath("../brainsss/scripts"))
from logging_utils import setup_logging
import logging

def parse_args(input, allow_unknown=True):
    parser = argparse.ArgumentParser(
        description="make mean brain for all functional channels in dir"
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="imaging directory containing func or anat data",
        required=True,
    )
    parser.add_argument(
        "--regexp",
        type=str,
        help="regexp to match files to process",
        default="*_channel_[1,2].nii",
    )
    parser.add_argument('--dirtype', type=str, default="func", 
                        help="func or anat", choices=['func', 'anat'])
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()
    return args


def make_mean_brain(args, file):
    logging.info(f"loading {file}")
    full_path = os.path.join(args.dir, file)
    if full_path.endswith(".nii"):
        brain = nib.load(full_path).get_fdata()
    elif full_path.endswith(".h5"):
        with h5py.File(full_path, "r") as hf:
            brain = np.asarray(hf["data"][:], dtype="uint16")
    else:
        raise ValueError(f"Unknown file type: {full_path}")

    return (np.mean(brain, axis=-1), brain.shape)


def save_mean_brain(args, file):
    file_extension = os.path.splitext(file)[-1]
    assert file_extension in [
        ".nii",
        ".h5",
    ], f"Unknown file extension: {file_extension}"
    save_file = file.replace(file_extension, "_mean.nii")
    logging.info(f"Saving mean brain to: {save_file}")

    # NOTE: This seems dangerous!  could result in differences in qform/sform across files
    aff = np.eye(4)
    img = nib.Nifti1Image(meanbrain, aff)
    img.to_filename(save_file)

    # assumes specific file naming...
    fly_print = args.dir.split("/")[-3]
    func_print = args.dir.split("/")[-2]
    logging.info(
        f"meanbrn | COMPLETED | {fly_print} | {func_print} | {file} | {brainshape} ===> {meanbrain.shape}"
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype="make_mean_brain")

    imaging_dir = os.path.join(args.dir, 'imaging')
    assert os.path.exists(imaging_dir), f"{imaging_dir} does not exist"

    files = [f.as_posix() for f in Path(imaging_dir).glob(args.regexp)]

    if len(files) == 0:
        raise FileNotFoundError(f"No files matching {args.regexp} found in {args.dir}")

    logging.info(f"found files: {files}")

    for file in files:

        meanbrain, brainshape = make_mean_brain(args, file)
        logging.info(
            f"generated mean brain from {file} with original shape {brainshape}"
        )
        save_mean_brain(args, file)
