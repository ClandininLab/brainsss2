# pyright: reportMissingImports=false
import sys
import numpy as np
import argparse
import nibabel as nib
import h5py


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert nii to h5')
    parser.add_argument('-f', '--file', type=str, help='file to process', required=True)
    parser.add_argument('-o', '--outfile', type=str, help='output file')

    args = parser.parse_args(sys.argv[1:])

    assert 'nii' in args.file, 'file must be nii'
    if args.outfile is None:
        args.outfile = args.file.replace('nii', 'h5')
    assert args.outfile != args.file, 'outfile should be different from file'

    print(f'converting {args.file} to {args.outfile}')

    img = nib.load(args.file)

    with h5py.File(args.outfile, 'w') as f:
        f.create_dataset('data', data=img.get_fdata(dtype='float32'), dtype="float32", chunks=True)
        f.create_dataset('qform', data=img.header.get_qform())
        f.create_dataset('zooms', data=img.header.get_zooms())
        f.create_dataset('xyzt_units', data=img.header.get_xyzt_units())
