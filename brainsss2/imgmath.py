# functions for performing image math operations

# pyright: reportMissingImports=false

import h5py
import numpy as np
import argparse
import nibabel as nib
from brainsss2.hdf5_utils import get_chunk_boundaries
import sys
from nibabel.processing import smooth_image
from brainsss2.argparse_utils import get_base_parser, add_imgmath_arguments


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('imgmath')

    parser = add_imgmath_arguments(parser)

    # need to add this manually to procesing steps in order to make required
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="input file",
        required=True,
    )
    parser.add_argument('--stepsize', type=int, default=50,
                        help="stepsize for chunking")
    parser.add_argument('-o', '--operation', type=str,
            help='operation to perform', required=True)
    parser.add_argument('--outfile_type', type=str,
        choices=['h5', 'nii'], default=None)
    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()

    return args


def parse_args_old(input, allow_unknown=True, add_op=False):
    """add_op allows to enable -o just for this script"""
    parser = argparse.ArgumentParser(
        description="make mean brain over time for a single h5 or nii file"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="h5 file",
        required=True,
    )
    parser.add_argument('--stepsize', type=int, default=50,
                        help="stepsize for chunking")
    if add_op:
        parser.add_argument('-o', '--operation', type=str,
            help='operation to perform', required=True)
    parser.add_argument('--outfile_type', type=str, choices=['h5', 'nii'], default=None)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--fwhm', type=float, help='fwhm for smoothing')
    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f'skipping unknown arguments:{unknown}')
    else:
        args = parser.parse_args()

    return(args)


def get_infile_type(file):
    if 'h5' in file:
        return('h5')
    elif 'nii' in file:
        return('nii')
    raise ValueError(f"Unknown file type: {file}")


def img_from_h5(f):
    assert 'data' in f, 'no data found in h5 file'
    assert 'qform' in f and 'zooms' in f and 'xyzt_units' in f, 'h5 must contain nii header fields'
    img = nib.Nifti1Image(f['data'], affine=f['qform'][:])
    img.header.set_qform(f['qform'][:])
    img.header.set_sform(f['qform'][:])
    img.header.set_zooms(f['zooms'][:len(f['data'].shape)])
    # deal with byte strings in h5 data
    try:
        xyz_units = f['xyzt_units'][0].decode('utf-8')
    except AttributeError:
        xyz_units = f['xyzt_units'][0]
    img.header.set_xyzt_units(xyz=xyz_units)
    return img


def nii_to_h5(outimg, outfile):
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('data', data=outimg.dataobj, dtype="float32", chunks=True)
        f.create_dataset('qform', data=outimg.header.get_qform())
        f.create_dataset('zooms', data=outimg.header.get_zooms())
        f.create_dataset('xyzt_units', data=outimg.header.get_xyzt_units())


def fix_outimg_header(outimg, img):
    outimg.header.set_qform(img.header.get_qform())
    outimg.header.set_sform(img.header.get_sform())
    outimg.header.set_xyzt_units(xyz=img.header.get_xyzt_units()[0])
    outimg.header.set_zooms(img.header.get_zooms()[:len(outimg.header.get_zooms())])
    return(outimg)


def img_mean(img, stepsize):
    outimg = nib.Nifti1Image(np.zeros(img.shape[:3]), img.affine)
    chunk_boundaries = get_chunk_boundaries(stepsize, img.shape[-1])

    nchunks = len(chunk_boundaries)
    for (chunk_start, chunk_end) in chunk_boundaries:
        outimg.dataobj[:, :, :] += np.mean(
            img.dataobj[:, :, :, chunk_start:chunk_end], axis=-1) / nchunks
    return outimg


def img_smooth(img, fwhm, stepsize=50):
    outimg = nib.Nifti1Image(np.zeros(img.shape), img.affine)
    chunk_boundaries = get_chunk_boundaries(stepsize, img.shape[-1])

    nchunks = len(chunk_boundaries)
    for chunk_num, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        print(f'chunk {chunk_num+1}/{nchunks}')
        tmp_img = nib.Nifti1Image(img.dataobj[:, :, :, chunk_start:chunk_end],
            img.affine, img.header)
        outimg.dataobj[:, :, :, chunk_start:chunk_end] = smooth_image(
            tmp_img, fwhm=fwhm
        ).get_fdata(dtype='float32')
    return outimg


def img_std(img):
    """ compute std deviation chunking by slice"""
    outimg = nib.Nifti1Image(np.zeros(img.shape[:3]), img.affine)

    for i in range(outimg.shape[2]):
        outimg.dataobj[:, :, i] = np.std(
            img.dataobj[:, :, i, :], axis=-1)
    return outimg


def imgmath(file, operation,
            verbose=False, outfile_type=None,
            stepsize=50, fwhm=None):
    """
    perform image math

    Parameters
    ----------
    file : str
        h5 or nii file
    operation: str
        name of operation to be performed (mean, std)
    stepsize : int
        stepsize for chunking
    verbose : bool
        verbose output
    outfile_type : str
        output file type (nii or h5)
        defaults to same as input file
    """
    infile_type = get_infile_type(file)

    if outfile_type is None:
        outfile_type = infile_type

    if operation == 'smooth':
        operation = f'smooth-{fwhm:.1f}'

    assert outfile_type in ['nii', 'h5'], "outfile_type must be either 'nii' or 'h5'"

    outfile = file.replace(f".{infile_type}", f"_{operation}.{outfile_type}")
    assert outfile != file, f"outfile should be different from file: {outfile}"

    if infile_type == 'h5':
        print(f'computing {operation} of h5 file')
        f = h5py.File(file, 'r')
        # convert immediately to nibabel image and set header fields
        img = img_from_h5(f)

    else:
        print(f'computing {operation} of nii file')
        img = nib.load(file, mmap='r')

    if operation == 'mean':
        outimg = img_mean(img, stepsize)
    elif operation == 'std':
        outimg = img_std(img)
    elif operation == 'tsnr':
        meanimg = img_mean(img, stepsize)
        sdimg = img_std(img)
        outimg = nib.Nifti1Image(meanimg.dataobj / sdimg.dataobj,
            meanimg.affine, meanimg.header)
    elif 'smooth' in operation:
        assert fwhm is not None, 'must specify fwhm for smoothing'
        outimg = img_smooth(img, fwhm, stepsize)
    else:
        raise ValueError(f'invalid operation {operation}')

    outimg = fix_outimg_header(outimg, img)

    if verbose:
        print(f'image grand mean: {np.mean(outimg.get_fdata())}')

    print(f'saving {operation} of {file} to file {outfile}')

    if outfile_type == 'h5':
        nii_to_h5(outimg, outfile)
    else:
        outimg.to_filename(outfile)
    return(outfile)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f'applying {args.operation} to {args.file}')
    _ = imgmath(args.file, args.operation,
        args.verbose, args.outfile_type, args.stepsize,
        fwhm=args.fwhm)
