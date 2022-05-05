# create and save skewness image to/from nii/h5
# pyright: reportMissingImports=false

import logging
import h5py
import numpy as np
import argparse
import sys
import nibabel as nib
from scipy.stats import skew


def parse_args(input, allow_unknown=True):
    parser = argparse.ArgumentParser(
        description="compute skewness of image")

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="h5 file",
        required=True,
    )
    parser.add_argument('--absolute', action="store_true", help="store absolute value")

    parser.add_argument('--stepsize', type=int, default=50,
                        help="stepsize for chunking")
    parser.add_argument('--outfile_type', type=str, choices=['h5', 'nii'], default=None)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    return parser.parse_args()


def imgskewness(file, absolute=True, verbose=False, outfile_type=None, stepsize=50):
    """
    create and save temporal skewness image to/from nii/h5

    Parameters
    ----------
    file : str
        h5 or nii file
    stepsize : int
        stepsize for chunking
    verbose : bool
        verbose output
    outfile_type : str
        output file type (nii or h5)
        defaults to same as input file
    """
    if 'h5' in file:
        infile_type = 'h5'
    elif 'nii' in file:
        infile_type = 'nii'
    else:
        raise ValueError(f"Unknown file type: {file}")

    if outfile_type is not None:
        assert outfile_type in ['nii', 'h5'], "outfile_type must be either 'nii' or 'h5'"
    else:
        outfile_type = infile_type

    if absolute:
        skewfile = file.replace(f".{infile_type}", f"_absoluteskew.{outfile_type}")
    else:
        skewfile = file.replace(f".{infile_type}", f"_skew.{outfile_type}")

    assert skewfile != file, f"skewfile should be different from file: {skewfile}"
    if infile_type == 'h5':
        print('computing mean of h5 file')
        with h5py.File(file, 'r') as f:

            # convert immediately to nibabel image
            img = nib.Nifti1Image(f['data'], affine=f['qform'][:])
            skewness = nib.Nifti1Image(np.zeros(img.shape[:3]), affine=f['qform'][:])
            # iterate over slices to save memory
            for slice in range(img.shape[-2]):
                skewness.dataobj[:, :, slice] += skew(img.dataobj[:, :, slice, :], axis=-1)

            if 'qform' in f:
                skewness.header.set_qform(f['qform'][:])
                # need to set sform as well as qform
                skewness.header.set_sform(f['qform'][:])
            else:
                print('no qform found in h5 file')

            if 'zooms' in f:
                skewness.header.set_zooms(f['zooms'][:3])
            else:
                print('no zooms found in h5 file')

            if 'xyzt_units' in f:
                # hdf saves to byte strings
                try:
                    xyz_units = f['xyzt_units'][0].decode('utf-8')
                except AttributeError:
                    xyz_units = f['xyzt_units'][0]
                skewness.header.set_xyzt_units(xyz=xyz_units)
            else:
                print('no xyzt_units found in h5 file')

    else:
        print('computing skewness of nii file')
        img = nib.load(file, mmap='r')
        # print('original image header:', img.header)
        # try using dataobj to get the data without loading the whole image
        skewness = nib.Nifti1Image(np.zeros(img.shape[:3]), affine=img.affine)
        # iterate over slices to save memory
        for slice in range(img.shape[-2]):
            skewness.dataobj[:, :, slice] += skew(img.dataobj[:, :, slice, :], axis=-1)
        skewness.header.set_qform(img.header.get_qform())
        skewness.header.set_sform(img.header.get_sform())
        skewness.header.set_xyzt_units(xyz=img.header.get_xyzt_units()[0])
        skewness.header.set_zooms(img.header.get_zooms()[:3])

    logging.info(f'image mean: {np.mean(skewness.get_fdata())}')

    if verbose:
        print(f'saving mean of {file} to file {skewfile}')

    if absolute:
        skewness = nib.Nifti1Image(np.abs(skewness.dataobj),
            affine=skewness.affine, header=skewness.header)

    if outfile_type == 'h5':
        print("saving to h5")
        with h5py.File(skewfile, 'w') as f:
            f.create_dataset('data', data=skewness.get_fdata(), dtype="float32", chunks=True)
            f.create_dataset('qform', data=skewness.header.get_qform())
            f.create_dataset('zooms', data=skewness.header.get_zooms())
            f.create_dataset('xyzt_units', data=skewness.header.get_xyzt_units())

    else:
        print('saving to nii')
        skewness.to_filename(skewfile)
    return(skewfile)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f'making skewness image for {args.file}')
    skewimg = imgskewness(args.file, args.absolute, args.verbose, args.outfile_type)
