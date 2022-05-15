from brainsss2.logging_utils import get_flystring
import argparse


def parse_args(input):
    parser = argparse.ArgumentParser(description="testing")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="base directory for imaging session",
        required=True,
    )
    return parser.parse_args(input)



def test_flystring():
    teststring = '/data/foo/fly_008/rp/d1'
    args = parse_args(['-d', teststring])
    assert get_flystring(args) == '_fly008'