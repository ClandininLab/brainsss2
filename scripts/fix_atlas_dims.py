
import nibabel
from pathlib import Path
import numpy as np


atlasdir = '/data/brainsss/atlas'

atlasfiles = [i.as_posix() for i in Path(atlasdir).glob('*.nii')]

for file in atlasfiles:
    print(file)
    outfile = file.replace('.nii', '_fixed.nii')
    assert file != outfile, f'outfile should be different from file: {outfile}'
    img = nibabel.load(file)
    print(img.header)
    resolution = [0.00038, 0.00038, 0.00038, 1]
    img.header.set_zooms(resolution[:3])
    img.header.set_xyzt_units(xyz='mm')
    img.header.set_qform(np.diag(np.array(resolution)))
    img.to_filename(outfile)
