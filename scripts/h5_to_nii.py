# pyright: reportMissingImports=false
import sys
import argparse
import nibabel as nib
import h5py


def h5_to_nii(file, outfile):
    with h5py.File(file, 'r') as f:
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

    img = nib.Nifti1Image(image_array, None)
    if qform is not None:
        img.header.set_qform(qform)
        img.header.set_sform(qform)
    if zooms is not None:
        if image_array.shape[-1] == 3:
            img.header.set_zooms(zooms[:3])
        else:
            img.header.set_zooms(zooms)
    if xyzt_units is not None:
        img.header.set_xyzt_units(xyz=xyzt_units[0], t=xyzt_units[1])

    img.to_filename(outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert h5 to nii')
    parser.add_argument('-f', '--file', type=str, help='file to process', required=True)
    parser.add_argument('-o', '--outfile', type=str, help='output file')

    args = parser.parse_args(sys.argv[1:])

    assert 'h5' in args.file, 'file must be h5'
    if args.outfile is None:
        args.outfile = args.file.replace('h5', 'nii')

    print(f'converting {args.file} to {args.outfile}')
    h5_to_nii(args.file, args.outfile)
