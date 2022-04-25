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
    parser.add_argument('--outfile_type', type=str, choices=['h5', 'nii'], default=None)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    return parser.parse_args()


def imgmean(file, stepsize=None, verbose=False, outfile_type=None):
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

            if 'qform' in f:
                qform = f['qform'][:]
            else:
                print('no qform found in h5 file')
                qform = None

            if 'zooms' in f:
                zooms = f['zooms'][:]
            else:
                print('no zooms found in h5 file')
                zooms = None

            if 'xyzt_units' in f:
                # hdf saves to byte strings
                xyzt_units = [i.decode('utf-8') for i in f['xyzt_units'][:]]
            else:
                print('no xyzt_units found in h5 file')
                xyzt_units = None
            if verbose:
                print('zooms', zooms)
                print('xyzt_units', xyzt_units)
                print('qform', qform)
    else:
        img = nib.load(file)
        qform = img.header.get_qform()
        zooms = img.header.get_zooms()
        xyzt_units = img.header.get_xyzt_units()
        meanimg = nilearn.image.mean_img(file)
        meandata = meanimg.get_fdata()

    print(f'image mean: {np.mean(meandata)}')

    brain_dims = meandata.shape
    if stepsize is None:
        chunks = True
    else:
        chunks = (brain_dims[0], brain_dims[1], brain_dims[2], stepsize)

    if outfile_type == 'h5':
        with h5py.File(meanfile, 'w') as f:
            f.create_dataset('data', data=meandata, dtype="float32", chunks=chunks)
            if qform is not None:
                f.create_dataset('qform', data=qform)
            if zooms is not None:
                f.create_dataset('zooms', data=zooms)
            if xyzt_units is not None:
                f.create_dataset('xyzt_units', data=xyzt_units)

    else:
        meanimg = nib.Nifti1Image(meandata, affine=None)
        if qform is not None:
            meanimg.header.set_qform(qform)
        if zooms is not None:
            meanimg.header.set_zooms(zooms[:3])
        if xyzt_units is not None:
            meanimg.header.set_xyzt_units(xyz=xyzt_units[0], t=xyzt_units[1])
        meanimg.to_filename(meanfile)
    return(meanfile)

 
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f'making mean brain for {args.file}')
    imgmean(args.file, args.stepsize, args.verbose, args.outfile_type)
