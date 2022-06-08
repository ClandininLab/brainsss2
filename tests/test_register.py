# tests of registration between func/anat/atlas
# strategy: transform image and then apply inverse transform
# checking to make sure that the correlation between the original image and the transformed image is high

import os
import ants
import numpy as np
from brainsss2.atlas_registration import make_clean_anat
import pytest

TESTDATADIR = 'testdata'
INTERP = 'lanczosWindowedSinc'


@pytest.fixture
def funcimg():
    funcimgfile = os.path.join(TESTDATADIR, 'functional_channel_1_moco_mean.nii')
    funcimg = ants.image_read(funcimgfile)
    return(funcimg)


@pytest.fixture
def anatimg():
    anatimgfile = os.path.join(TESTDATADIR, 'anatomy_channel_1_res-2.0mu_moco_mean.nii')
    clean_anatimgfile, _ = make_clean_anat(anatimgfile, normalize=True)
    anatimg = ants.image_read(clean_anatimgfile)
    return(anatimg)


@pytest.fixture
def atlasimg():
    atlasimgfile = os.path.join(TESTDATADIR, '20220301_luke_2_jfrc_affine_zflip_2umiso.nii')
    atlasimg = ants.image_read(atlasimgfile)
    atlasimg.set_spacing([2, 2, 2])  # make sure the spacing is correct
    return(atlasimg)


@pytest.fixture
def func_anat_reg(funcimg, anatimg):

    func_to_anat = ants.registration(
        fixed=anatimg,
        moving=funcimg,
        type_of_transform='SyN',
        interpolator=INTERP,
        total_sigma=3,
        flow_sigma=3)

    inverse_func_to_anat = ants.apply_transforms(
        fixed=funcimg,
        moving=func_to_anat['warpedmovout'],
        transformlist=func_to_anat['invtransforms'],
        whichtoinvert=[True, False]
    )
    return(func_to_anat, inverse_func_to_anat)


@pytest.fixture
def anat_atlas_reg(anatimg, atlasimg):

    anat_to_atlas = ants.registration(
        fixed=atlasimg,
        moving=anatimg,
        type_of_transform='SyN',
        interpolator=INTERP,
        syn_sampling=64,
        total_sigma=3,
        flow_sigma=3,
        grad_step=.1,
        reg_iterations=[200, 200, 50]
    )

    inverse_anat_to_atlas = ants.apply_transforms(
        fixed=anatimg,
        moving=anat_to_atlas['warpedmovout'],
        transformlist=anat_to_atlas['invtransforms'],
        whichtoinvert=[True, False]
    )
    return(anat_to_atlas, inverse_anat_to_atlas)


def test_funcimg(funcimg):
    assert funcimg is not None


def test_anatimg(anatimg):
    assert anatimg is not None


def test_atlasimg(atlasimg):
    assert atlasimg is not None


def test_func_to_anat(func_anat_reg, funcimg):
    func_to_anat, inverse_func_to_anat = func_anat_reg
    assert inverse_func_to_anat.shape == funcimg.shape
    assert np.corrcoef(funcimg.numpy().flatten(), inverse_func_to_anat.numpy().flatten())[0, 1] > .99


def test_anat_to_atlas(anat_atlas_reg, anatimg):
    anat_to_atlas, inverse_anat_to_atlas = anat_atlas_reg
    assert inverse_anat_to_atlas.shape == anatimg.shape
    assert np.corrcoef(anatimg.numpy().flatten(), inverse_anat_to_atlas.numpy().flatten())[0, 1] > .99


def test_func_to_atlas(func_anat_reg, anat_atlas_reg, funcimg, atlasimg):
    func_to_anat, inverse_func_to_anat = func_anat_reg
    anat_to_atlas, inverse_anat_to_atlas = anat_atlas_reg

    func_to_atlas_transforms = anat_to_atlas['invtransforms'] + func_to_anat['invtransforms']  # noqa
    func_to_atlas = ants.apply_transforms(
        fixed=atlasimg, moving=funcimg,
        transformlist=func_to_atlas_transforms,
        interpolator=INTERP)

    # take back to func space from atlas space
    atlas_to_func_transforms = func_to_anat['invtransforms'] + anat_to_atlas['invtransforms']
    atlas_to_func_transforms

    inverse_func_to_atlas = ants.apply_transforms(
        fixed=funcimg, moving=func_to_atlas,
        transformlist=atlas_to_func_transforms,
        whichtoinvert=[True, False, True, False, ],
        interpolator=INTERP)

    # allow more leeway here because of the dual interpolation
    assert np.corrcoef(funcimg.numpy().flatten(), inverse_func_to_atlas.numpy().flatten())[0, 1] > .9
