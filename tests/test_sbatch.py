# test basic sbatch functionality on single node host
# pyright: reportMissingImports=false

import sys
import os
import shutil
from pathlib import Path
sys.path.append("../brainsss")
sys.path.append("../brainsss/scripts")
import logging  # noqa
from slurm import SlurmBatchJob  # noqa
import pytest  # noqa


@pytest.fixture
def sbatch():
    shutil.rmtree('logs')
    argdict = {"foo": 1}
    return SlurmBatchJob("test", "dummy_script.py",
        argdict, logfile='logs/sbatch_test.log')


def test_sbatch_argdict(sbatch):
    argdict = {"foo": 1}
    sbatch = SlurmBatchJob("test", "dummy_script.py", argdict)
    assert sbatch.args["foo"] == 1


def test_sbatch_run(sbatch):
    sbatch.run()
    assert sbatch.job_id is not None


def test_sbatch_status(sbatch):
    status = sbatch.status()
    assert status is not None


def test_sbatch_wait(sbatch):
    sbatch.run()
    sbatch.wait()
    assert sbatch.status() == "COMPLETED"
    assert os.path.exists(sbatch.logfile)
    # find internal log file
    assert len(list(Path('logs').glob('dummy_script_test*'))) == 1