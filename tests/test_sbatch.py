# test basic sbatch functionality on single node host

import sys
import os

sys.path.append("../brainsss")
sys.path.append("../brainsss/scripts")
import logging
from slurm import SlurmBatchJob
import pytest


@pytest.fixture
def sbatch():
    argdict = {"foo": 1}
    return SlurmBatchJob("test", "dummy_script.py", argdict)


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


def sbatch_older():
    jobname = "test"
    script = "sbatch.py"
    args = {"sleep": 10}
    if not os.path.exists("logs"):
        os.mkdir("logs")
    com_dir = "./com"
    if not os.path.exists(com_dir):
        os.mkdir(com_dir)
    logfile = "logs/sbatch_test.log"

    logging_handlers = [logging.FileHandler(logfile)]
    logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        handlers=logging_handlers,
        level=logging.INFO,
        format="%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    job_id = slurm_submit(jobname, script, args, logfile)

    if job_id is None:
        logging.info("job_id is None")
    else:
        logging.info(f"job_id: {job_id}")
        output = wait_for_job(job_id, com_dir)
        logging.info("completed")
        logging.info(f"output:\n{output}")

    assert "Hello, world!" in output
