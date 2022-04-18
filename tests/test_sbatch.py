# test basic sbatch functionality on single node host

import sys
import os
sys.path.append('../brainsss')
sys.path.append('../brainsss/scripts')
from slurm_utils import slurm_submit, wait_for_job
import logging


def test_sbatch():
    jobname = 'test'
    script = 'sbatch.py'
    args = {'sleep': 10}
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logfile = 'logs/sbatch_test.log'

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
        logging.info('job_id is None')
    else:
        logging.info(f'job_id: {job_id}')
        output = wait_for_job(job_id, './com')
        logging.info('completed')
        logging.info(f'output:\n{output}')
    
    assert "Hello, world!" in output