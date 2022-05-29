# script to run long wait time on sherllock
# pyright: reportMissingImports=false

import sys
import os
import shutil
from pathlib import Path
import logging  # noqa
from brainsss2.slurm import SlurmBatchJob  # noqa
import pytest  # noqa

if __name__ == "__main__":
    
    argdict = {"foo": 1, 'verbose': True, 'time_hours': 24}
    sbatch = SlurmBatchJob("test", "dummy_script_long.py",
        argdict, logfile='logs/sbatch_test.log', verbose=True)

    sbatch.run()
    sbatch.wait()
