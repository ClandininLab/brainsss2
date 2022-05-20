#!/usr/bin/env python3
# Top-level script to build and/or process a single fly
# will be wrapped by another script to allow processing of multiple flies

# pyright: reportMissingImports=false

import sys
import os
import logging
import datetime
from pathlib import Path

from brainsss2.argparse_utils import add_moco_arguments
from brainsss2.logging_utils import (
    setup_logging,
    get_logfile_name,
    remove_existing_file_handlers,
    reinstate_file_handlers
  )  # noqa
from collections import OrderedDict # noqa
from brainsss2.preprocess_utils import ( # noqa
    load_user_settings_from_json,
    setup_modules,
    dict_to_args_list,
    run_shell_command
)  # noqa
from brainsss2.argparse_utils import ( # noqa
    get_base_parser,
    add_builder_arguments,
    add_preprocess_arguments,
    add_fictrac_qc_arguments,
    add_moco_arguments,
    add_imgmath_arguments
)  # noqa
from brainsss2.slurm import SlurmBatchJob  # noqa
from brainsss2.imgmath import imgmath  # noqa


def get_max_slurm_cpus():
    """get the max number of cpus for slurm"""
    cmdout = run_shell_command("sinfo -h -o %C")
    try:
        maxcores = int(cmdout.strip().split('/')[-1])
    except AttributeError:
        maxcores = 1
    return maxcores


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

    parser = add_moco_arguments(parser)

    parser = add_imgmath_arguments(parser)

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
        'func_dirs': args.func_dirs,
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

    if 'dirtype' in args_dict and args_dict['dirtype'] is not None:
        args.dirtype = args_dict['dirtype']
    else:
        args.dirtype = 'func'

    procdirs = get_dirs_to_process(args)[args.dirtype]
    procdirs.sort()

    assert len(procdirs) > 0, "no func directories found, somethign has gone wrong"

    sbatch = {}
    saved_handlers = []

    for procdir in procdirs:
        if args.dirtype == 'func' and args.func_dirs is not None and procdir.split('/')[-1] not in args.func_dirs:
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
        args_dict['verbose'] = args.verbose
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


def process_fly(args):
    """process a single fly"""

    assert os.path.exists(args.process)
    logging.info(f"processing fly from {args.process}")

    if args.test:
        logging.info("test mode, not actually processing flies")

    workflow_dict = OrderedDict()

    # add each step to the workflow
    # should always include basedir to ensure proper logging
    # the specific files used in each step are built into the workflow components
    # so we don't need to track and pass output to input at each step ala nipype
    # but it means that the steps cannot be reordered and expected to run properly
    if args.fictrac_qc or args.run_all:
        workflow_dict['fictrac_qc'] = {
            'script': 'fictrac_qc.py',
            "fps": 100,
            'basedir': args.basedir,
            'dir': args.process
        }

    if args.STB or args.run_all:
        workflow_dict['stim_triggered_avg_beh'] = {
            'script': 'stim_triggered_avg_beh.py',
            'basedir': args.basedir,
            'dir': args.process,
            'cores': 2
        }

    if args.bleaching_qc or args.run_all:
        workflow_dict['bleaching_qc'] = {
            'script': 'bleaching_qc.py',
            'basedir': args.basedir,
            'dir': args.process,
            'cores': 2
        }

    if args.motion_correction in ['func', 'both'] or args.run_all:
        workflow_dict['motion_correction_func'] = {
            'script': 'motion_correction.py',
            'basedir': args.basedir,
            'type_of_transform': args.type_of_transform,
            'dir': args.process,
            # use longer run with fewer cores if not using normal queue
            # need to retest this with the new moco model
            'time_hours': 16 if args.partition == 'normal' else 24,
            'cores': args.cores if args.cores is not None else min(
                8 if args.partition == 'normal' else 4,
                get_max_slurm_cpus()),
            'dirtype': 'func'
        }

    if args.motion_correction in ['anat', 'both'] or args.run_all:
        workflow_dict['motion_correction_anat'] = {
            'script': 'motion_correction.py',
            'basedir': args.basedir,
            'type_of_transform': args.type_of_transform,
            'dir': args.process,
            # use longer run with fewer cores if not using normal queue
            # need to retest this with the new moco model
            'time_hours': 2 if args.partition == 'normal' else 24,
            'stepsize': 4,
            'downsample': True,
            'cores': min(
                args.cores,
                8 if args.partition == 'normal' else 4,
                get_max_slurm_cpus()),
            'dirtype': 'anat'
        }

    if args.smoothing or args.run_all:
        workflow_dict['smoothing'] = {
            'script': 'imgmath.py',
            'operation': 'smooth',
            'basedir': args.basedir,
            'fwhm': args.fwhm,
            'dir': args.dir,
            'cores': 8,
            'file': os.path.join(
                args.process,
                'func_0/preproc/functional_channel_2_moco.h5'
            )
        }
        print('workflow_dict:', workflow_dict)

    if args.regression or args.run_all:
        workflow_dict['regression_XYZ'] = {
            'script': 'regression.py',
            'basedir': args.basedir,
            'dir': args.process,
            'overwrite': True,
            'cores': 8,
            'label': 'model001_dRotLabXYZ',
            'confound_files': 'preproc/framewise_displacement.csv',
            'time_hours': 1,
            'behavior': ['dRotLabY', 'dRotLabZ+', 'dRotLabZ-'],
        }
        workflow_dict['regression_confound'] = {
            'script': 'regression.py',
            'basedir': args.basedir,
            'overwrite': True,
            'dir': args.process,
            'save_residuals': True,
            'cores': 8,
            'label': 'model000_confound',
            'confound_files': 'preproc/framewise_displacement.csv',
            'time_hours': 1
        }

    if args.STA or args.run_all:
        workflow_dict['STA'] = {
            'script': 'stim_triggered_avg_neu.py',
            'basedir': args.basedir,
            'overwrite': True,
            'dir': args.process,
            'cores': 8,
            'time_hours': 1
        }

    if args.atlasreg or args.run_all:
        if args.atlasdir is None:
            args.atlasdir = os.path.join(
                args.basedir,
                'atlas'
            )
        workflow_dict['atlasreg'] = {
            'script': 'atlas_registration.py',
            'basedir': args.basedir,
            'atlasfile': os.path.join(
                args.atlasdir,
                args.atlasfile),
            'type_of_transform': 'SyN',
            'atlasname': 'jfrc',
            'overwrite': args.overwrite,
            'dir': args.dir,
            'cores': 8,
            'time_hours': 8
        }
    # align_anat

    # make supervoxels
    if args.supervoxels or args.run_all:
        workflow_dict['supervoxels'] = {
            'script': 'make_supervoxels.py',
            'basedir': args.basedir,
            'overwrite': args.overwrite,
            'dir': args.dir,
            'cores': 4,
            'time_hours': 1
        }

    if args.h5_to_nii:
        #             time=2,
        #             mem=10, - this seems crazy...
        logging.warning('h5_to_nii not yet implemented!')

    for stepname, step_args_dict in workflow_dict.items():
        logging.info(f'running step: {step_args_dict["script"]}')
        args.dir = args.process  # NOTE: this is bad and confusing, but would take work to fix
        run_preprocessing_step(step_args_dict['script'], args, step_args_dict)


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
