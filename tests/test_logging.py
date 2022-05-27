# test logging setup
import argparse
import logging
import os
from brainsss2.logging_utils import setup_logging
import pyfiglet
import datetime
from time import strftime


def test_setup_logging():
    args = argparse.Namespace()
    setattr(args, 'verbose', False)
    setattr(args, 'basedir', './')
    args = setup_logging(args, logtype='motion_correction')
    args.logger.error('TESTSTRING')
    args.logger.debug('DEBUGSTRING')
    print('handlers', args.logger.handlers)
    assert args.logger.level == 20
    assert os.path.exists(args.logfile)
    with open(args.logfile, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 0
    found_teststring = False
    found_debugstring = False
    for line in lines:
        if 'TESTSTRING' in line:
            found_teststring = True
        if 'DEBUGSTRING' in line:
            found_debugstring = True
    assert found_teststring
    assert not found_debugstring


def test_setup_logging_verbose():
    args = argparse.Namespace()
    setattr(args, 'verbose', True)
    setattr(args, 'basedir', './')
    args = setup_logging(args, logtype='motion_correction')
    args.logger.error('TESTSTRING')
    args.logger.debug('DEBUGSTRING')
    print('handlers', args.logger.handlers)
    assert args.logger.level == 10
    assert os.path.exists(args.logfile)
    with open(args.logfile, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 0
    found_teststring = False
    found_debugstring = False
    for line in lines:
        if 'TESTSTRING' in line:
            found_teststring = True
        if 'DEBUGSTRING' in line:
            found_debugstring = True
    assert found_teststring
    assert found_debugstring


if __name__ == "__main__":
    test_setup_logging()
    test_setup_logging_verbose()
