# test json settings

import argparse
from brainsss2.preprocess_utils import (
    load_default_settings_from_json,
    load_user_settings_from_json
)
import pytest

ALL_STEPS = [
    'motion_correction_anat',
    'motion_correction_func',
    'regression_XYZ',
    'regression_confound',
    'STA',
    'atlasreg',
    'supervoxels',
    'smoothing'
]


@pytest.fixture
def args():
    args = argparse.Namespace()
    args = load_default_settings_from_json(args)
    return args


def test_load_default_settings_from_json(args):
    assert args.default_settings is not None
    for step in ALL_STEPS:
        assert step in args.default_settings
        assert args.default_settings[step] is not None


def test_extend_default_settings_with_user_settings(args):
    args = load_user_settings_from_json(args, user_settings_file='user_settings.json')
    assert args.preproc_settings is not None
    assert args.preproc_settings['motion_correction_func']['type_of_transform'] == 'Rigid'


def test_empty_user_settings(args):
    args = load_user_settings_from_json(args, user_settings_file='nonexistent_user_settings')
    assert args.preproc_settings is not None
    assert args.preproc_settings['motion_correction_func']['type_of_transform'] == 'SyN'
