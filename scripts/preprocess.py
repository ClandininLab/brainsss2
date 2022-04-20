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
from logging_utils import setup_logging, get_flystring, get_logfile_name
from collections import OrderedDict
from logging_utils import remove_existing_file_handlers, reinstate_file_handlers
# THIS A HACK FOR DEVELOPMENT
sys.path.append('../brainsss')
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


def parse_args(input):
    parser = get_base_parser('preprocess')

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

    funcdirs = get_dirs_to_process(args)['func']
    funcdirs.sort()

    assert len(funcdirs) > 0, "no func directories found, somethign has gone wrong"

    sbatch = {}
    saved_handlers = []

    for func in funcdirs:
        if args.func_dirs is not None and func.split('/')[-1] not in args.func_dirs:
            logging.info(f'skipping {func}')
            continue
        if 'logfile' not in args_dict:
            logfile = get_logfile_name(
                func,
                stepname,
                get_flystring(args)
            )
            args_dict['logfile'] = logfile

        if not os.path.exists(os.path.dirname(logfile)):
            os.mkdir(os.path.dirname(logfile))

        if not args.local:
            args_dict['dir'] = func
            sbatch[func] = SlurmBatchJob(stepname, script, args_dict)
            sbatch[func].run()
            if hasattr(sbatch[func], 'saved_handlers'):
                saved_handlers.extend(sbatch[func].saved_handlers)

        else:  # run locally
            logging.info(f'running {script} locally')
            saved_handlers = remove_existing_file_handlers()
            setattr(args, 'dir', func)  # create required arg for fictrac_qc.py
            args.logdir = None
            argstring = ' '.join(dict_to_args_list(args.__dict__))
            print(argstring)
            output = run_shell_command(f'python {script} {argstring}')

    if not args.local:
        output = {}
        for func, job in sbatch.items():
            job.wait()
            output[func] = job.status()

    _ = remove_existing_file_handlers()
    if len(saved_handlers) > 0:
        reinstate_file_handlers(saved_handlers)
    else:
        logging.warning('no saved handlers found')

    logging.info(f'Completed step: {stepname}')
    return(output)


# def run_stim_triggered_beh():

#     for func in funcs:
#         args = {"logfile": logfile, "func_path": func}
#         script = "stim_triggered_avg_beh.py"
#         job_id = brainsss.sbatch(
#             jobname="stim",
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


# def run_bleaching_qc():

#     # job_ids = []
#     for funcanat, dirtype in zip(funcanats, dirtypes):
#         directory = os.path.join(funcanat, "imaging")
#         args = {"logfile": logfile, "directory": directory, "dirtype": dirtype}
#         script = "bleaching_qc.py"
#         job_id = brainsss.sbatch(
#             jobname="bleachqc",
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


# def run_temporal_mean_brain_pre():

#     for funcanat, dirtype in zip(funcanats, dirtypes):
#         directory = os.path.join(funcanat, "imaging")

#         if dirtype == "func":
#             files = ["functional_channel_1.nii", "functional_channel_2.nii"]
#         if dirtype == "anat":
#             files = ["anatomy_channel_1.nii", "anatomy_channel_2.nii"]

#         args = {"logfile": logfile, "directory": directory, "files": files}
#         script = "make_mean_brain.py"
#         job_id = brainsss.sbatch(
#             jobname="meanbrn",
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


# def run_motion_correction():

#     for funcanat, dirtype in zip(funcanats, dirtypes):

#         directory = os.path.join(funcanat, "imaging")
#         # NB: 1/2 are actually anatomy/functional
#         if dirtype == "func":
#             brain_master = "functional_channel_1.nii"
#             brain_mirror = "functional_channel_2.nii"
#         if dirtype == "anat":
#             brain_master = "anatomy_channel_1.nii"
#             brain_mirror = "anatomy_channel_2.nii"

#         args = {
#             "logfile": logfile,
#             "directory": directory,
#             "brain_master": brain_master,
#             "brain_mirror": brain_mirror,
#             "scantype": dirtype,
#         }

#         script = "motion_correction.py"
#         # if global_resources:
#         #     dur = 48
#         #     mem = 8
#         # else:
#         #     dur = 96
#         #     mem = 4
#         global_resources = True
#         dur = 48
#         mem = 8
#         job_id = brainsss.sbatch(
#             jobname="moco",
#             script=os.path.join(scripts_path, script),
#             modules=modules,
#             args=args,
#             logfile=logfile,
#             time=dur,
#             mem=mem,
#             nice=nice,
#             nodes=nodes,
#             global_resources=global_resources,
#         )
#     ### currently submitting these jobs simultaneously since using global resources
#     brainsss.wait_for_job(job_id, logfile, com_path)


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
            'dir': args.process
        }

    for script, step_args_dict in workflow_dict.items():
        logging.info(f'running step: {script}')
        run_preprocessing_step(script, args, step_args_dict)

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
            raise Exception('fly building failed')
        logging.info(f'Built to flydir: {args.process}')
        if args.build_only:
            logging.info('build only, exiting')
            args.process = None
    # TODO: I am assuming that results of build_dirs should be passed along to fly_dirs after processing...

    if args.process is not None:
        logging.info(f'processing {args.process}')
        setattr(args, 'script_path', os.path.dirname(os.path.realpath(__file__)))
        process_fly(args)
