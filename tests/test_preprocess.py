# test components of preprocess.py

import pytest
import json
from preprocess_utils import (
    parse_args,
    dict_to_args_list,
    load_user_settings_from_json,
    setup_modules,
    default_modules
)


@pytest.fixture
def args():
    return parse_args(['-d', '/tmp/test'])


def test_dict_to_args_list():
    d = {'a': '1', 'b': 2, 'c': [1, 2, 3]}
    assert dict_to_args_list(d) == ['--a 1', '--b 2', '--c 1', '--c 2', '--c 3']


def test_parse_args(args):
    assert args.target_dir == '/tmp/test'


def test_load_settings_from_file(args):
    d = {'process_flydir': 'test'}
    testfile = '/tmp/test.json'
    with open(testfile, 'w') as f:
        json.dump(d, f)
    setattr(args, 'settings_file', testfile)
    args = load_user_settings_from_json(args)
    assert args.process_flydir == 'test'


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
