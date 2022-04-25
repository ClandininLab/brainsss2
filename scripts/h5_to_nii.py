# pyright: reportMissingImports=false
import sys
import numpy as np
import argparse
import nibabel as nib
import h5py


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert h5 to nii')
    parser.add_argument('-f', '--file', type=str, help='file to process', required=True)
    parser.add_argument('-o', '--outfile', type=str, help='output file')

    args = parser.parse_args(sys.argv[1:])

    assert 'h5' in args.file, 'file must be h5'
    if args.outfile is None:
        args.outfile = args.file.replace('h5', 'nii')

    print(f'converting {args.file} to {args.outfile}')

    with h5py.File(args.file, 'r+') as f:
        image_array = f.get("data")[:].astype('float32')

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

    img = nib.Nifti1Image(image_array, np.eye(4))
    if qform is not None:
        img.header.set_qform(qform)
    if zooms is not None:
        img.header.set_zooms(zooms[:3])
    if xyzt_units is not None:
        img.header.set_xyzt_units(xyz=xyzt_units[0], t=xyzt_units[1])

    img.to_filename(args.outfile)
