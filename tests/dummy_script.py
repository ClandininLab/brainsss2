# this dummy script does its own internal logging
# this is meant to test the slurm logging functions

import logging
import argparse
import sys

sys.path.append("../brainsss")
sys.path.append("../brainsss/scripts")
from logging_utils import setup_logging



print('testing')

# set up fake args for logging
args = argparse.Namespace()
setattr(args, 'logdir', 'logs')
args = setup_logging(args, 'dummy_script_test')
logging.info('testing logging within dummy script')