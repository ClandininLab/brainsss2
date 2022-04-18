# refactor of slurm utils
import argparse
from preprocess_utils import dict_to_args_list, dict_to_namespace
import subprocess
import logging
import time
import os


class SlurmBatchJob:
    def __init__(self, jobname: str, script: str, args: dict):
        self.jobname = jobname
        self.script = script
        self.args = args
        self.job_id = None
        self.output = None
        self.com_dir = None

    def submit(self):
        return None

    def wait(self):
        return None

    def get_output(self):
        return None

