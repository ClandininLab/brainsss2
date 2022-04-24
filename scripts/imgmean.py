# create and save mean image to/from nii/h5
# pyright: reportMissingImports=false

import h5py
import numpy as np
import argparse
import sys
import nibabel as nib
import nilearn.image

def parse_args(input, allow_unknown=True):
    parser = argparse.ArgumentParser(
        description="make mean brain for a single h5 file"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="h5 file",
        required=True,
    )
    parser.add_argument('--stepsize', type=int, default=None,
                        help="stepsize for chunking")

    return parser.parse_args()


def imgmean(file, stepsize=None, verbose=False, outfile_type=None):

    if 'h5' in file:
        infile_type = 'h5'
    elif 'nii' in file:
        infile_type = 'nii'
    else:
        raise ValueError(f"Unknown file type: {file}")

    if outfile_type is not None:
        assert outfile_type in ['nii', 'h5'], f"outfile_type must be either 'nii' or 'h5'"
    else:
        outfile_type = infile_type

    meanfile = file.replace(f".{infile_type}", f"_mean.{outfile_type}")

    assert meanfile != file, f"meanfile should be different from file: {meanfile}"
    if verbose:
        print(f'saving mean of {file} to file {meanfile}')

    if infile_type == 'h5':
        print('compute mean of h5 file')
        with h5py.File(file, 'r') as f:
            data = f['data']
            print('data shape: ', data.shape)
            meandata = np.zeros(data.shape[:3])

            for slice in range(data.shape[-2]):
                if verbose:
                    print(f'processing slice {slice + 1} of {data.shape[-2]}')
                    meandata[:, :, slice] = np.mean(data[:, :, slice, :], axis=-1)
            if 'affine' in f:
                affine = f['affine'][:]
            else:
                print('no affine found in h5 file')
                affine = None
    else:
        meanimg = nilearn.image.mean_img(file)
        meandata = meanimg.get_fdata()
        affine = meanimg.affine

    brain_dims = meandata.shape
    if stepsize is None:
        chunks = True
    else:
        chunks = (brain_dims[0], brain_dims[1], brain_dims[2], stepsize)

    if outfile_type == 'h5':
        with h5py.File(meanfile, 'w') as f:
            f.create_dataset('data', data=meandata, dtype="float32", chunks=chunks)
            if affine is not None:
                f.create_dataset('affine', data=affine)

    else:
        nib.Nifti1Image(meandata, affine=affine).to_filename(meanfile)
    return(meanfile)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f'making mean brain for {args.file}')
    make_mean_from_h5(args.file, args.stepsize, args.verbose)
