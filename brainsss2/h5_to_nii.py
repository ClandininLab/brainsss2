# pyright: reportMissingImports=false
import nibabel as nib
import h5py


def h5_to_nii(file, outfile=None):
    if outfile is None:
        outfile = file.replace('.h5', '.nii')
        assert outfile != file, 'Output file cannot be the same as input file'
    print(f'converting {file} to {outfile}')
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
