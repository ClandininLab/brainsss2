# test components of preprocess.py

import pytest
import json
from brainsss2.preprocess_utils import (
    dict_to_args_list,
    load_user_settings_from_json,
    load_default_settings_from_json,
    setup_modules,
    default_modules
)
from brainsss2.argparse_utils import ( # noqa
    get_base_parser,
    add_builder_arguments,
    add_preprocess_arguments,
    add_fictrac_qc_arguments,
    add_moco_arguments
)  # noqa


@pytest.fixture
def args():
    parser = get_base_parser('preprocess')

    parser.add_argument(
        "-b", "--basedir",
        type=str,
        help="base directory for fly data",
        required=True)

    parser = add_builder_arguments(parser)

    parser = add_preprocess_arguments(parser)

    return parser.parse_args(['-b', '/tmp/test'])


def test_dict_to_args_list():
    d = {'a': '1', 'b': 2, 'c': [1, 2, 3]}
    assert dict_to_args_list(d) == ['--a', '1', '--b', '2', '--c', '1', '2', '3']


def test_parse_args(args):
    assert args.basedir == '/tmp/test'


def test_load_settings_from_file(args):
    d = {'motion_correction_func': {
        'type_of_transform': 'Rigid'
    }}
    testfile = '/tmp/test.json'
    with open(testfile, 'w') as f:
        json.dump(d, f)
    setattr(args, 'settings_file', testfile)
    args = load_default_settings_from_json(args)
    args = load_user_settings_from_json(args, testfile)
    assert args.preproc_settings[
        'motion_correction_func']['type_of_transform'] == 'Rigid'


def test_setup_modules_basic(args):
    args = setup_modules(args)
    for d in default_modules():
        assert d in args.module_string


def test_setup_modules_extend(args):
    setattr(args, 'modules', ['testmod'])
    args = setup_modules(args)
    assert 'testmod' in args.module_string
    for d in default_modules():
        assert d in args.module_string
