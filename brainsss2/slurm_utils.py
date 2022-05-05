# refactor of slurm utils
import argparse
from brainsss2.preprocess_utils import dict_to_args_list, dict_to_namespace
import subprocess
import logging
import time
import os


def slurm_submit(jobname, 
                 script,
                 errfile,
                 args=None,
                 dep=None,
                 modules=None): #
    # , time, mem, nodes, modules, global_resources):
    """
    submit slurm job

    Parameters:
    -----------
    jobname: str
        name of job
    script: str
        path to script to run
    args_dict: dict
        arguments to pass to script
    errfile: str
        path to error file
    dep: str
        job dependency
    modules: list
        list of modules to load
    time_hours: int
        time limit for job in hours
    com_path: str
        path to directory to store output
    """

    default_args = {
        'nice': False,
        'time_hours': 1,
        'com_path': './com'
    }
    if args_dict is None:
        args_dict = {}
    elif not isinstance(args, dict):
        raise TypeError('args must be dict or None')

    if com_path is None:
        com_path = './com'
    if not os.path.exists(com_path):
        os.mkdir(com_path)

    if modules is None:
        module_string = ''
    else:
        module_string = 'ml {" ".join(modules)}; '

    if dep is not None:
        logging.info(f'dependencies: {dep}')
        dep = "--dependency=afterok:{} --kill-on-invalid-dep=yes ".format(dep)

    command = f"{module_string}python3 {script} {' '.join(dict_to_args_list(args))}"
    logging.info(f'command: {command}')

    sbatch_command = (
        f"sbatch -J {jobname} -o ./com/%j.out "
        f"--nice={args.nice} {}--open-mode=append --cpus-per-task={args.nodes} "
        f"-e ./com/%j.stderr  -t {time_hours}:00:00 --wrap='{command}'"
    )
    #    jobname, logfile, time, nice, node_cmd, mem, begin, command, dep

    logging.info(f'sbatch_command: {sbatch_command}')
    logging.info(f'Redirecting stderr to {errfile}')

    sbatch_response = subprocess.getoutput(sbatch_command)
    logging.info(f'sbatch_response: {sbatch_response}')

    job_id = sbatch_response.split(" ")[-1].strip()

    return job_id


def get_job_status(job_id, return_full_output=False):
    """
    get status of slurm job

    Parameters: 
    -----------
    job_id: str
        slurm job id
    return_full_output: bool
        whether to return full output from sacct (default to only return status)
    """

    temp = subprocess.getoutput(
        f"sacct -n -P -j {job_id} --noconvert --format=State,Elapsed,MaxRSS,NCPUS,JobName"
    )

    status = None if temp == "" else temp.split("\n")[0].split("|")[0].upper()

    return temp if return_full_output else status


def wait_for_job(job_id, com_path, wait_time=5):

    logging.info(f'Waiting for job {job_id}')

    while True:
        status = get_job_status(job_id)
        logging.info(f'status: {status}')
        if status is not None and status not in ['PENDING', 'RUNNING']:
            status = get_job_status(job_id, return_full_output=True)
            logging.info(f'Job {job_id} finished with status: {status}')
            com_file = os.path.join(com_path, f'{job_id}.out')
            try:
                with open(com_file, "r") as f:
                    output = f.read()
            except FileNotFoundError:
                logging.warning('Could not find com file: {com_file}')
                output = None
            return output
        else:
            time.sleep(wait_time)


if __name__ == "__main__":
    jobname = 'test'
    script = 'sbatch.py'
    args = {'sleep': 10}
    logfile = 'sbatch_test.log'

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