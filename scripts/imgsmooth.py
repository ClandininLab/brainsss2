# spatially smooth nii/h5
# pyright: reportMissingImports=false

import logging
import h5py
import numpy as np
import argparse
import sys
import nibabel as nib
import nilearn.image
from hdf5_utils import get_chunk_boundaries
from nibabel.processing import smooth_image

def parse_args(input, allow_unknown=True):
    parser = argparse.ArgumentParser(
        description="make mean brain over time for a single h5 or nii file"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="input file",
        required=True,
    )
    parser.add_argument('--fwhm', type=float, 
        help='fwhm for smoothing', required=True)
    parser.add_argument('--stepsize', type=int, default=50,
                        help="stepsize for chunking")
    parser.add_argument('--outfile_type', type=str, choices=['h5', 'nii'], default=None)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    return parser.parse_args()


def imgsmooth(file, fwhm, verbose=False, outfile_type=None, stepsize=50):
    """
    create and save temporal mean image to/from nii/h5

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

    assert fwhm > 0, "fwhm must be greater than 0"

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

    smoothedfile = file.replace(f".{infile_type}", f"_smooth-{args.fwhm}.{outfile_type}")

    assert smoothedfile != file, f"smoothed file should be different from file: {smoothedfile}"
    if infile_type == 'h5':
        print('computing smoothed version of h5 file')
        with h5py.File(file, 'r') as f:
            datashape = f['data'].shape
            # convert immediately to nibabel image
            smoothed_img = nib.Nifti1Image(np.zeros(datashape, dtype='float32'),
                affine=f['qform'][:])
            print(f'stepping through chunks of {stepsize}')
            chunk_boundaries = get_chunk_boundaries(stepsize, datashape[-1])

            nchunks = len(chunk_boundaries)
            for chunk_num, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
                print(f'chunk {chunk_num+1}/{nchunks}')
                tmp_img = nib.Nifti1Image(f['data'][:, :, :, chunk_start:chunk_end],
                    f['qform'][:])
                smoothed_img.dataobj[:, :, :, chunk_start:chunk_end] += smooth_image(
                    tmp_img, fwhm=fwhm
                ).get_fdata(dtype='float32')

            if 'qform' in f:
                smoothed_img.header.set_qform(f['qform'][:])
                # need to set sform as well as qform
                smoothed_img.header.set_sform(f['qform'][:])
            else:
                print('no qform found in h5 file')

            if 'zooms' in f:
                smoothed_img.header.set_zooms(f['zooms'])
            else:
                print('no zooms found in h5 file')

            if 'xyzt_units' in f:
                # hdf saves to byte strings
                try:
                    xyz_units = f['xyzt_units'][0].decode('utf-8')
                except AttributeError:
                    xyz_units = f['xyzt_units'][0]
                smoothed_img.header.set_xyzt_units(xyz=xyz_units)
            else:
                print('no xyzt_units found in h5 file')

    else:
        print('smoothing nii file')
        img = nib.load(file, mmap='r')
        # print('original image header:', img.header)
        # try using dataobj to get the data without loading the whole image
        smoothed_img = nib.Nifti1Image(np.zeros(img.shape), img.affine)
        chunk_boundaries = get_chunk_boundaries(stepsize, img.shape[-1])

        nchunks = len(chunk_boundaries)
        for chunk_num, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            tmp_img = nib.Nifti1Image(img.dataobj[:, :, :, chunk_start:chunk_end], img.affine)
            smoothed_img.dataobj[:, :, :, chunk_start:chunk_end] += smooth_image(
                tmp_img, fwhm=fwhm,
            )
        # meanimg = nilearn.image.mean_img(file)
        smoothed_img.header.set_qform(img.header.get_qform())
        smoothed_img.header.set_sform(img.header.get_sform())
        smoothed_img.header.set_xyzt_units(xyz=img.header.get_xyzt_units()[0])
        smoothed_img.header.set_zooms(img.header.get_zooms())

    if verbose:
        print(f'saving mean of {file} to file {smoothedfile}')

    if outfile_type == 'h5':
        print("saving to h5")
        with h5py.File(smoothedfile, 'w') as f:
            f.create_dataset('data', data=smoothed_img.dataobj, dtype="float32", chunks=True)
            f.create_dataset('qform', data=smoothed_img.header.get_qform())
            f.create_dataset('zooms', data=smoothed_img.header.get_zooms())
            f.create_dataset('xyzt_units', data=smoothed_img.header.get_xyzt_units())

    else:
        print('saving to nii')
        smoothed_img.to_filename(smoothedfile)
    return(smoothedfile)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f'making mean brain for {args.file}')
    smoothed_img = imgsmooth(args.file, args.fwhm, 
        args.verbose, args.outfile_type, args.stepsize)
