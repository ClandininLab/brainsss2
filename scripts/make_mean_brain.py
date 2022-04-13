import os
import sys
import numpy as np
import nibabel as nib
import h5py
import argparse
from pathlib import Path
from logging_utils import setup_logging
import logging


def parse_args(input):
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
        default="functional_channel_[1,2].nii",
        help="regexp to match files to process",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    return parser.parse_args(input)


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

    setup_logging(args, logtype="meanbrain")

    files = [f.as_posix() for f in Path(args.dir).glob("*_channel*.nii")]

    logging.info(f"found files: {files}")

    for file in files:

        meanbrain, brainshape = make_mean_brain(args, file)
        logging.info(
            f"generated mean brain from {file} with original shape {brainshape}"
        )
        save_mean_brain(args, file)
