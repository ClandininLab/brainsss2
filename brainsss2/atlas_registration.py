import ants
import nibabel as nib
import scipy.stats
import numpy as np
from sklearn.preprocessing import quantile_transform


def make_clean_anat(anatfile, sigma=5, thresh_pct=40, normalize=False):
    anat_img = nib.load(anatfile)
    anat_ants = ants.image_read(anatfile)
    low_thresh = scipy.stats.scoreatpercentile(anat_ants.numpy(), thresh_pct)
    anat_ants_masked = ants.get_mask(anat_ants,
        low_thresh=low_thresh,
        cleanup=4)
    maskfile = anatfile.replace('.nii', '_mask.nii')
    assert maskfile != anatfile, 'maskfile should not be the same as anatfile'
    anat_ants_masked.to_filename(maskfile)

    brain = anat_img.get_fdata(dtype='float32') * anat_ants_masked[...]

    # ### Blur brain and mask small values ###
    # brain_copy = brain.copy()
    # brain_copy = gaussian_filter(brain_copy, sigma=sigma)
    # threshold = triangle(brain_copy)
    # brain_copy[np.where(brain_copy < threshold/2)] = 0

    # ### Remove blobs outside contiguous brain ###
    # labels, label_nb = scipy.ndimage.label(brain_copy)
    # brain_label = np.bincount(labels.flatten())[1:].argmax()+1
    # brain_copy = brain.copy().astype('float32')
    # brain_copy[np.where(labels != brain_label)] = np.nan

    ### Perform quantile normalization ###
    if normalize:
        brain_out = quantile_transform(brain.flatten().reshape(-1, 1), n_quantiles=500, random_state=0, copy=True)
        brain_out = brain_out.reshape(brain.shape)
        brain = np.nan_to_num(brain_out, copy=False)

    clean_anatfile = anatfile.replace('.nii', '_clean.nii')
    assert clean_anatfile != anatfile, 'clean_anatfile is same as anatfile'
    nib.save(nib.Nifti1Image(brain, anat_img.affine, anat_img.header), clean_anatfile)
    return clean_anatfile, maskfile
