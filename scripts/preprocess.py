#!/usr/bin/env python3
# Top-level script to build and/or process a single fly
# will be wrapped by another script to allow processing of multiple flies

# pyright: reportMissingImports=false

import sys
import os
import logging
import datetime

from pathlib import Path
sys.path.append("../brainsss")
sys.path.append("../brainsss/scripts")
from logging_utils import setup_logging, get_logfile_name  # noqa
from collections import OrderedDict # noqa
from logging_utils import remove_existing_file_handlers, reinstate_file_handlers # noqa
# THIS A HACK FOR DEVELOPMENT
sys.path.insert(0, '../brainsss')
from preprocess_utils import ( # noqa
    load_user_settings_from_json,
    setup_modules,
    dict_to_args_list,
    run_shell_command
)  # noqa
from argparse_utils import ( # noqa
    get_base_parser,
    add_builder_arguments,
    add_preprocess_arguments,
    add_fictrac_qc_arguments,
)  # noqa
from slurm import SlurmBatchJob  # noqa


def get_max_slurm_cpus():
    """get the max number of cpus for slurm"""
    return int(run_shell_command("sinfo -h -o %C").strip().split('/')[-1])


def parse_args(input):
    parser = get_base_parser('preprocess')

    parser.add_argument(
        "-b", "--basedir",
        type=str,
        help="base directory for fly data",
        required=True)

    parser = add_builder_arguments(parser)

    parser = add_preprocess_arguments(parser)

    parser = add_fictrac_qc_arguments(parser)

    return parser.parse_args(input)


def build_fly(args):
    """build a single fly"""
    logging.info(f"building flies from {args.import_path}")

    assert args.import_date is not None, "no import date specified"

    args_dict = {
        "import_date": args.import_date,
        'import_dir': args.import_dir,
        "target_dir": args.target_dir,
        "fly_dirs": args.fly_dirs,
        "user": args.user,
        "verbose": args.verbose,
        'basedir': args.basedir,
        'logfile': os.path.join(
            args.basedir,
            'logs',
            f"flybuilder_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    }

    logging.info(f'args_dict submitted to fly_builder: {args_dict}')
    if not args.local:

        sbatch = SlurmBatchJob('flybuilder', "fly_builder.py", args_dict, verbose=args.verbose)
        sbatch.run()
        sbatch.wait()
        _ = remove_existing_file_handlers()
        reinstate_file_handlers(sbatch.saved_handlers)

        logging.info(f'flybuilder job complete: {sbatch.status(return_full_output=True)}')
        # get fly directory from output
        if os.path.exists(sbatch.logfile):
            return(get_flydir_from_building_log(sbatch.logfile))
        else:
            return None
    else:
        # run locally
        logging.info('running fly_builder.py locally')
        args.logfile = args_dict['logfile']
        argstring = ' '.join(dict_to_args_list(args.__dict__))
        output = run_shell_command(f'python fly_builder.py {argstring}')
        return get_flydir_from_output(output)


def get_dirs_to_process(args):
    """get the directories to process from the fly directory"""
    return {
        'func': [i.as_posix() for i in Path(args.process).glob('func_*')],
        'anat': [i.as_posix() for i in Path(args.process).glob('anat_*')]
    }


def run_preprocessing_step(script, args, args_dict):
    """run a preprocessing step

    Parameters:
        script {str}:
            script to run
        args {argparse.Namespace}:
            parsed arguments from main script
        args_dict {dict}:
            dictionary of arguments to pass to step script

        """
    stepname = script.split(".")[0]
    logging.info(f"running {stepname}")

    if 'dirtype' not in args and 'dirtype' not in args_dict:
        args.dirtype = 'func'
    elif 'dirtype' in args_dict:
        args.dirtype = args_dict['dirtype']

    procdirs = get_dirs_to_process(args)[args.dirtype]
    procdirs.sort()

    assert len(procdirs) > 0, "no func directories found, somethign has gone wrong"

    sbatch = {}
    saved_handlers = []

    for procdir in procdirs:
        if args.func_dirs is not None and procdir.split('/')[-1] not in args.func_dirs:
            logging.info(f'skipping {procdir} - not included in --func_dirs')
            continue

        if 'logfile' not in args_dict:
            logfile = get_logfile_name(
                os.path.join(procdir, 'logs'),
                stepname
            )
        print(f'LOGGING to {logfile}')
        args_dict['logfile'] = logfile

        if not os.path.exists(os.path.dirname(logfile)):
            os.mkdir(os.path.dirname(logfile))

        if args.local:
            logging.info(f'running {script} locally')
        else:
            logging.info(f'running {script} via slurm')

        args_dict['partition'] = args.partition
        args_dict['dir'] = procdir
        args.dir = procdir
        sbatch[procdir] = SlurmBatchJob(stepname, script,
                                     args_dict, local=args.local)
        sbatch[procdir].run()
        if hasattr(sbatch[procdir], 'saved_handlers'):
            saved_handlers.extend(sbatch[procdir].saved_handlers)

    output = {}
    for procdir, job in sbatch.items():
        job.wait()
        output[procdir] = job.status()

    _ = remove_existing_file_handlers()
    if saved_handlers:
        reinstate_file_handlers(saved_handlers)
    else:
        logging.warning('no saved handlers found')

    logging.info(f'Completed step: {stepname}')
    return(output)


# def run_zscore():
#     # TODO: check that moco file exists

#     for func in funcs:
#         load_directory = os.path.join(func, "moco")
#         save_directory = os.path.join(func)
#         brain_file = "functional_channel_2_moco.h5"

#         args = {
#             "logfile": logfile,
#             "load_directory": load_directory,
#             "save_directory": save_directory,
#             "brain_file": brain_file,
#         }
#         script = "zscore.py"
#         job_id = brainsss.sbatch(
#             jobname="zscore",
#             script=os.path.join(scripts_path, script),
#             modules=modules,
#             args=args,
#             logfile=logfile,
#             time=1,
#             mem=2,
#             nice=nice,
#             nodes=nodes,
#         )
#         brainsss.wait_for_job(job_id, logfile, com_path)


# def run_highpass():
#     # TODO: check for file existence

#     for func in funcs:

#         load_directory = os.path.join(func)
#         save_directory = os.path.join(func)
#         brain_file = "functional_channel_2_moco_zscore.h5"

#         args = {
#             "logfile": logfile,
#             "load_directory": load_directory,
#             "save_directory": save_directory,
#             "brain_file": brain_file,
#         }
#         script = "temporal_high_pass_filter.py"
#         job_id = brainsss.sbatch(
#             jobname="highpass",
#             script=os.path.join(scripts_path, script),
#             modules=modules,
#             args=args,
#             logfile=logfile,
#             time=4,
#             mem=2,
#             nice=nice,
#             nodes=nodes,
#         )
#         brainsss.wait_for_job(job_id, logfile, com_path)


# def run_correlation():

#     for func in funcs:
#         load_directory = os.path.join(func)
#         save_directory = os.path.join(func, "corr")
#         brain_file = "functional_channel_2_moco_zscore_highpass.h5"
#         behavior = "dRotLabY"

#         args = {
#             "logfile": logfile,
#             "load_directory": load_directory,
#             "save_directory": save_directory,
#             "brain_file": brain_file,
#             "behavior": behavior,
#         }
#         script = "correlation.py"
#         job_id = brainsss.sbatch(
#             jobname="corr",
#             script=os.path.join(scripts_path, script),
#             modules=modules,
#             args=args,
#             logfile=logfile,
#             time=2,
#             mem=4,
#             nice=nice,
#             nodes=nodes,
#         )
#         brainsss.wait_for_job(job_id, logfile, com_path)


# def run_STA():

#     for func in funcs:
#         args = {"logfile": logfile, "func_path": func}
#         script = "stim_triggered_avg_neu.py"
#         job_id = brainsss.sbatch(
#             jobname="STA",
#             script=os.path.join(scripts_path, script),
#             modules=modules,
#             args=args,
#             logfile=logfile,
#             time=4,
#             mem=4,
#             nice=nice,
#             nodes=nodes,
#         )
#         brainsss.wait_for_job(job_id, logfile, com_path)


# def run_h5_to_nii():

#     for func in funcs:
#         args = {
#             "logfile": logfile,
#             "h5_path": os.path.join(
#                 func, "functional_channel_2_moco_zscore_highpass.h5"
#             ),
#         }
#         script = "h5_to_nii.py"
#         job_id = brainsss.sbatch(
#             jobname="h5tonii",
#             script=os.path.join(scripts_path, script),
#             modules=modules,
#             args=args,
#             logfile=logfile,
#             time=2,
#             mem=10,
#             nice=nice,
#             nodes=nodes,
#         )
#         brainsss.wait_for_job(job_id, logfile, com_path)


# def temporal_mean_brain_post():

#     for funcanat, dirtype in zip(funcanats, dirtypes):
#         directory = os.path.join(funcanat, "moco")

#         if dirtype == "func":
#             files = ["functional_channel_1_moco.h5", "functional_channel_2_moco.h5"]
#         if dirtype == "anat":
#             files = ["anatomy_channel_1_moco.h5", "anatomy_channel_2_moco.h5"]

#         args = {"logfile": logfile, "directory": directory, "files": files}
#         script = "make_mean_brain.py"
#         job_id = brainsss.sbatch(
#             jobname="meanbrn",
#             script=os.path.join(scripts_path, script),
#             modules=modules,
#             args=args,
#             logfile=logfile,
#             time=2,
#             mem=10,
#             nice=nice,
#             nodes=nodes,
#         )
#         brainsss.wait_for_job(job_id, logfile, com_path)


def process_fly(args):
    """process a single fly"""

    assert os.path.exists(args.process)
    logging.info(f"processing fly from {args.process}")

    if args.test:
        logging.info("test mode, not actually processing flies")

    workflow_dict = OrderedDict()

    # add each step to the workflow
    # should always include basedir to ensure proper logging
    if args.fictrac_qc:
        workflow_dict['fictrac_qc.py'] = {
            "fps": 100,
            'basedir': args.basedir,
            'dir': args.process
        }

    if args.STB:
        workflow_dict['stim_triggered_avg_beh.py'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'cores': 2
        }

    if args.bleaching_qc:
        workflow_dict['bleaching_qc.py'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'cores': 2
        }

    # leaving this out - mean will be computed automatically by motcorr
    # if set(args.temporal_mean).intersection({'both', 'pre'}):
    #     workflow_dict['make_mean_brain.py'] = {
    #         'basedir': args.basedir,
    #         'dir': args.process,
    #         'cores': 4,
    #         'dirtype': 'func'
    #     }

    if args.motion_correction is not None:
        workflow_dict['motion_correction.py'] = {
            'basedir': args.basedir,
            'dir': args.process,
            # use longer run with fewer cores if not using normal queue
            'time_hours': 48 if args.partition == 'normal' else 96,
            'cores': min(4 if args.partition == 'normal' else 4, get_max_slurm_cpus()),
            'dirtype': args.motion_correction
        }

    for script, step_args_dict in workflow_dict.items():
        logging.info(f'running step: {script}')
        args.dir = args.process
        run_preprocessing_step(script, args, step_args_dict)

    if args.highpass:
        workflow_dict['temporal_high_pass_filter.py'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'cores': 2
        }

    # if args.fictrac_qc:
    #     #fictrac_output = run_fictrac_qc(args)
    #     step_args_dict = {
    #         "fps": 100,
    #         'basedir': args.basedir,}
    #     run_preprocessing_step('fictrac_qc.py', args, step_args_dict)

    # if args.STB:
    #     stb_output = run_stim_triggered_beh()

    # if args.bleaching_qc:
    #     bleaching_output = run_bleaching_qc()

    # if 'pre' in args.temporal_mean or 'both' in args.temporal_mean:
    #     mean_brain_pre_output = run_temporal_mean_brain_pre()

    # if args.motion_correction:
    #     motion_correction_output = run_motion_correction()

    # if args.zscore:
    #     zscore_output = run_zscore()

    # if args.highpass:
    #     highpass_output = run_highpass()

    # if args.correlation:
    #     correlation_output = run_correlation()

    # if args.STA:
    #     STA_output = run_STA()

    # if args.h5_to_nii:
    #     h5_to_nii_output = run_h5_to_nii()

    # if 'post' in args.temporal_mean or 'both' in args.temporal_mean:
    #     mean_brain_post_output = run_temporal_mean_brain_post()


def setup_build_dirs(args):
    assert args.import_date is not None, "must specify import_date for building"

    if args.import_dir is None:
        args.import_dir = os.path.join(args.basedir, "imports")
    setattr(args, 'import_path', os.path.join(args.import_dir, args.import_date))
    assert os.path.exists(args.import_path), f"Import path does not exist: {args.import_path}"

    if args.fictrac_import_dir is None:
        args.fictrac_import_dir = os.path.join(args.basedir, "fictrac")
    assert os.path.exists(
        args.fictrac_import_dir
    ), f"fictrac import dir {args.fictrac_import_dir} does not exist"
    return args


def get_flydir_from_output(output):
    """get the fly directory from the output"""
    for line in output.split("\n"):
        if "flydir: " in line:
            return line.split(" ")[1].strip()


def get_flydir_from_building_log(logfile):
    """get the fly directory from the log file"""
    with open(logfile, 'r') as f:
        for line in f.readlines():
            if "flydir: " in line:
                return line.split(" ")[-1].strip()


if __name__ == "__main__":

    print('welcome to fly_preprocessing')
    args = parse_args(sys.argv[1:])
    print(args)

    args = setup_modules(args)

    if args.target_dir is None:
        if args.process:
            args.target_dir = os.path.dirname(args.process)
        else:
            args.target_dir = os.path.join(args.basedir, "processed")
            if not os.path.exists(args.target_dir):
                os.mkdir(args.target_dir)

    if 'dir' not in args or args.dir is None:
        setattr(args, 'dir', args.target_dir)

    args = setup_logging(args, logtype='preprocess',
        logdir=os.path.join(args.basedir, "logs"))

    if not args.ignore_settings:
        args = load_user_settings_from_json(args)

    print(args)

    if args.build:
        logging.info("building fly")
        args = setup_build_dirs(args)
        args.process = build_fly(args)
        if args.process is None:
            raise ValueError('fly building failed')
        logging.info(f'Built to flydir: {args.process}')
        if args.build_only:
            logging.info('build only, exiting')
            args.process = None
    # TODO: I am assuming that results of build_dirs should be passed along to fly_dirs after processing...

    if args.process is not None:
        logging.info(f'processing {args.process}')
        setattr(args, 'script_path', os.path.dirname(os.path.realpath(__file__)))
        process_fly(args)
