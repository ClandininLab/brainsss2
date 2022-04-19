# test basic sbatch functionality on single node host
# pyright: reportMissingImports=false

import sys
import os

sys.path.append("../brainsss")
sys.path.append("../brainsss/scripts")
import logging  # noqa
from slurm import SlurmBatchJob  # noqa
import pytest  # noqa


@pytest.fixture
def sbatch():
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
    assert os.path.exists(os.path.join(sbatch.logdir, f'test_{sbatch.job_id}.out'))
    assert os.path.exists(os.path.join(sbatch.logdir, f'test_{sbatch.job_id}.stderr'))
