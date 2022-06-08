# tests of registration between func/anat/atlas
# strategy: transform image and then apply inverse transform
# checking to make sure that the correlation between the original image and the transformed image is high

import os
import ants
import numpy as np
from brainsss2.atlas_registration import make_clean_anat
import pytest

TESTDATADIR = 'testdata'


@pytest.fixture
def funcimg():
    funcimgfile = os.path.join(TESTDATADIR, 'functional_channel_1_moco_mean.nii')
    funcimg = ants.image_read(funcimgfile)
    return(funcimg)


@pytest.fixture
def anatimg():
    anatimgfile = os.path.join(TESTDATADIR, 'anatomy_channel_1_res-2.0mu_moco_mean.nii')
    clean_anatimgfile, _ = make_clean_anat(anatimgfile)
    anatimg = ants.image_read(clean_anatimgfile)
    return(anatimg)


@pytest.fixture
def atlasimg():
    atlasimgfile = os.path.join(TESTDATADIR, '20220301_luke_2_jfrc_affine_fixed_2um.nii')
    atlasimg = ants.image_read(atlasimgfile)
    return(atlasimg)


@pytest.fixture
def func_anat_reg(funcimg, anatimg):

    func_to_anat = ants.registration(
        fixed=anatimg,
        moving=funcimg,
        type_of_transform='SyN',
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
        syn_sampling=64,
        total_sigma=1,
        flow_sigma=3,
        grad_step=.1,
        reg_iterations=[100, 100, 20]
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


