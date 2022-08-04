#!/usr/bin/env python3
# Top-level script to build and/or process a single fly
# ashlu this statment is not true? (won't wrap flies outside of this script)
# will be wrapped by another script to allow processing of multiple flies

# pyright: reportMissingImports=false

import sys
import os
import datetime
from pathlib import Path
import json
from brainsss2.argparse_utils import add_moco_arguments
# ashlu listing these out explicitly is awesome
from brainsss2.logging_utils import (
    setup_logging,
    get_logfile_name,
    remove_existing_file_handlers,
    reinstate_file_handlers
  )  # noqa
from collections import OrderedDict # noqa
from brainsss2.preprocess_utils import ( # noqa
    load_user_settings_from_json,
    load_default_settings_from_json,
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
from brainsss2.slurm import SlurmBatchJob, get_max_slurm_cpus  # noqa
from brainsss2.imgmath import imgmath  # noqa


def parse_args(input):
    parser = get_base_parser('preprocess')

    parser.add_argument(
        "-b", "--basedir",
        type=str,
        help="base directory for fly data",
        required=True)

    # set up base parser, and then loading in separate arguments for
    # each processing step
    parser = add_builder_arguments(parser)

    parser = add_preprocess_arguments(parser)

    # ashlu why these commented out?
    # parser = add_fictrac_qc_arguments(parser)

    # parser = add_moco_arguments(parser)

    # parser = add_imgmath_arguments(parser)

    return parser.parse_args(input)


def build_fly(args):
    """build a single fly directory

    Parameters:
        args {argparse.Namespace}:
            parsed arguments from main script

    Returns:
        flydir {str}:
            path to fly directory
    """
    args.logger.info(f"building flies from {args.import_path}")

    assert args.import_date is not None, "no import date specified"

    args_dict = {
        "import_date": args.import_date,
        'import_dir': args.import_dir,
        'fictrac_import_dir': args.fictrac_import_dir,
        "target_dir": args.target_dir,
        "fly_dirs": args.fly_dirs,
        'func_dirs': args.func_dirs,
        "user": args.user,
        "verbose": args.verbose,
        "overwrite": args.overwrite,
        'basedir': args.basedir,
        "time_hours": 4,
        'logfile': os.path.join(
            args.basedir,
            'logs',
            f"flybuilder_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    }

    args.logger.debug(f'args_dict submitted to fly_builder: {args_dict}')
    if not args.local:
        sbatch = SlurmBatchJob('flybuilder', "fly_builder.py", args_dict, verbose=args.verbose)
        sbatch.run()
        sbatch.wait()
        _ = remove_existing_file_handlers()
        reinstate_file_handlers(sbatch.saved_handlers)

        args.logger.info(f'flybuilder job complete: {sbatch.status(return_full_output=True)}')
        # get fly directory from output
        if os.path.exists(sbatch.logfile):
            return(get_flydir_from_building_log(sbatch.logfile))
        else:
            return None
    else:
        # run locally
        args.logger.info('running fly_builder.py locally')
        args.logfile = args_dict['logfile']
        argstring = ' '.join(dict_to_args_list(args.__dict__))
        output = run_shell_command(f'python fly_builder.py {argstring}')
        return get_flydir_from_output(output)


def get_dirs_to_process(args):
    """get the directories to process from the fly directory

    Parameters:
        args {argparse.Namespace}:
            parsed arguments from main script

    Returns:
        dirs_to_process {dict}:
            dictionary of directories to process
    """
    return {
        'func': [i.as_posix() for i in Path(args.process).glob('func_*')],
        'anat': [i.as_posix() for i in Path(args.process).glob('anat_*')]
    }


# ashlu two different argument dicts is confusing. maybe just renaming them would help
# ashlu (like "global args / meta args idk" vs processing_step_args)
def run_preprocessing_step(script, args, args_dict):
    """run a preprocessing step

    Parameters:
        script {str}:
            script to run
        args {argparse.Namespace}:
            parsed arguments from main script
        args_dict {dict}:
            dictionary of arguments to pass to step script

    Returns:
        output {str}:
            output from step script
    """
    stepname = script.split(".")[0]
    args.logger.info(f"running {stepname}")

    if 'dirtype' in args_dict and args_dict['dirtype'] is not None:
        args.dirtype = args_dict['dirtype']
    else:
        # assume func if not specified
        # this requires that dirtype is always set for anat dirs
        args.dirtype = 'func'

    procdirs = get_dirs_to_process(args)[args.dirtype]
    procdirs.sort()

    assert len(procdirs) > 0, "no func directories found, somethign has gone wrong"

    sbatch = {}

    # turn off previous logging handlers
    saved_handlers = remove_existing_file_handlers()

    for procdir in procdirs:
        if args.dirtype == 'func' and args.func_dirs is not None and procdir.split('/')[-1] not in args.func_dirs:
            args.logger.info(f'skipping {procdir} - not included in --func_dirs')
            continue

        if 'logfile' not in args_dict:
            logfile = get_logfile_name(
                os.path.join(procdir, 'logs'),
                stepname
            )

        if not os.path.exists(os.path.dirname(logfile)):
            os.mkdir(os.path.dirname(logfile))

        if args.local:
            args.logger.info(f'running {script} locally')
        else:
            args.logger.info(f'running {script} via slurm')

        args_dict['partition'] = args.partition
        args_dict['dir'] = procdir
        args_dict['verbose'] = args.verbose
        args_dict['logfile'] = logfile

        args.logger.info(f'HELLO. user_args:{args_dict}')

        sbatch[procdir] = SlurmBatchJob(stepname, script,
                                        user_args=args_dict,
                                        local=args.local,
                                        verbose=args.verbose)
        sbatch[procdir].run()

    output = {}
    # loop through processes, waiting for each one
    for procdir, job in sbatch.items():
        job.wait()
        output[procdir] = job.status()
        job.disable_loggers()

    args.logger.info(f'Completed step: {stepname}\n\n')

    # hand logging back to the previous handlers
    if saved_handlers:
        reinstate_file_handlers(saved_handlers)
    else:
        args.logger.warning('no saved handlers found')

    return(output)


def process_fly(args):
    args.logger.info(f'E: args.cores = {args.cores}')
    """perform preprocessing on a single fly

    Parameters:
        args {argparse.Namespace}:
            parsed arguments from main script

    Returns:
        workflow_dict {dict}:
            dictionary of workflow specs
    """

    assert os.path.exists(args.process)
    args.logger.info(f"processing fly from {args.process}")

    if args.test:
        args.logger.info("test mode, not actually processing flies")

    workflow_dict = OrderedDict()

    # add each step to the workflow
    # should always include basedir to ensure proper logging
    # the specific files used in each step are built into the workflow components
    # so we don't need to track and pass output to input at each step ala nipype
    # but it means that the steps cannot be reordered and expected to run properly

    # each step starts with the session-dependent variables
    # which are then extended with the preproc settings (default or user-specified)
    # ashlu - a bit confusing some of these items are only here and not in settings (cores for example)
    # ashlu - or some hardcoded file names here. Maybe port more of these over to settings file
    if args.fictrac_qc or args.run_all:
        workflow_dict['fictrac_qc'] = {
            'basedir': args.basedir,
            'dir': args.process
        }
        workflow_dict['fictrac_qc'].update(args.preproc_settings['fictrac_qc'])
        args.logger.debug(f'fictrac_qc workflow dict: {workflow_dict}')

    if args.STB or args.run_all:
        workflow_dict['stim_triggered_avg_beh'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'cores': 2
        }
        workflow_dict['stim_triggered_avg_beh'].update(args.preproc_settings['stim_triggered_avg_beh'])
        args.logger.debug(f'stim_triggered_avg_beh workflow dict: {workflow_dict}')

    if args.bleaching_qc or args.run_all:
        workflow_dict['bleaching_qc'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'cores': min(16, get_max_slurm_cpus() - 1)
        }
        workflow_dict['bleaching_qc'].update(args.preproc_settings['bleaching_qc'])
        args.logger.debug(f'bleaching_qc workflow dict: {workflow_dict}')

    args.logger.info(f'HIIIII 1 args.cors = {args.cores}')
    if args.motion_correction in ['func', 'both'] or args.run_all:
        workflow_dict['motion_correction_func'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'time_hours': 24 if args.partition == 'normal' else 24,
            'cores': min(
                16 if args.partition == 'normal' else 4,
                get_max_slurm_cpus() - 1),
        }
        args.logger.info(f'HIIIII 2 args.cors = {args.cores}')
        workflow_dict['motion_correction_func'].update(args.preproc_settings['motion_correction_func'])
        args.logger.debug(f'motion_correction_func workflow dict: {workflow_dict}')

    if args.motion_correction in ['anat', 'both'] or args.run_all:
        workflow_dict['motion_correction_anat'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'time_hours': 8 if args.partition == 'normal' else 24,
            'cores': min(
                args.cores,
                8 if args.partition == 'normal' else 4,
                get_max_slurm_cpus() - 1),
        }
        workflow_dict['motion_correction_anat'].update(args.preproc_settings['motion_correction_anat'])
        args.logger.debug(f'motion_correction_anat workflow dict: {workflow_dict}')

    if args.smoothing or args.run_all:
        workflow_dict['smoothing'] = {
            'basedir': args.basedir,
            'dir': args.dir,
            'cores': min(args.cores, 8, get_max_slurm_cpus() - 1),
            'file': os.path.join(
                args.process,
                'func_0/preproc/functional_channel_2_moco.h5'
            )
        }
        workflow_dict['smoothing'].update(args.preproc_settings['smoothing'])
        args.logger.debug(f'smoothing workflow dict: {workflow_dict}')

    if args.regression or args.run_all:
        # run confound model first so that we can create delta r2 for full model
        workflow_dict['regression_confound'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'cores': min(args.cores, 8, get_max_slurm_cpus() - 1),
            'residfile': f'preproc/functional_channel_2_moco_smooth-{args.preproc_settings["smoothing"]["fwhm"]:.1f}mu_residuals.h5'
        }
        workflow_dict['regression_confound'].update(args.preproc_settings['regression_confound'])
        args.logger.debug(f'regression_confound workflow dict: {workflow_dict}')
        workflow_dict['regression_XYZ'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'cores': min(args.cores, 8, get_max_slurm_cpus() - 1),
            'baseline_r2': 'regression/model000_confound/rsquared.nii'
        }
        workflow_dict['regression_XYZ'].update(args.preproc_settings['regression_XYZ'])
        args.logger.debug(f'regression_XYZ workflow dict: {workflow_dict}')

    if args.STA or args.run_all:
        workflow_dict['STA'] = {
            'basedir': args.basedir,
            'dir': args.process,
            'filename': 'preproc/functional_channel_2_moco_smooth-2.0mu_residuals.h5',
            'cores': min(args.cores, 8, get_max_slurm_cpus() - 1),
        }
        workflow_dict['STA'].update(args.preproc_settings['STA'])
        args.logger.debug(f'STA workflow dict: {workflow_dict}')

    if args.atlasreg or args.run_all:
        if args.atlasdir is None:
            args.atlasdir = os.path.join(
                args.basedir,
                'atlas'
            )
        workflow_dict['atlasreg'] = {
            'basedir': args.basedir,
            'atlasfile': os.path.join(
                args.atlasdir,
                args.atlasfile),
            'overwrite': args.overwrite,
            'dir': args.dir,
            'cores': min(args.cores, 8, get_max_slurm_cpus() - 1),
        }
        workflow_dict['atlasreg'].update(args.preproc_settings['atlasreg'])
        args.logger.debug(f'atlasreg workflow dict: {workflow_dict}')

    # make supervoxels
    if args.supervoxels or args.run_all:
        workflow_dict['supervoxels'] = {
            'basedir': args.basedir,
            'overwrite': args.overwrite,
            'dir': args.dir,
            'funcfile': f'preproc/functional_channel_2_moco_smooth-{args.preproc_settings["smoothing"]["fwhm"]:.1f}mu_residuals.h5',
            'cores': min(args.cores, 8, get_max_slurm_cpus() - 1),
        }
        workflow_dict['supervoxels'].update(args.preproc_settings['supervoxels'])
        args.logger.debug(f'supervoxels workflow dict: {workflow_dict}')

    # run PCA
    if args.PCA or args.run_all:
        workflow_dict['PCA_resid'] = {
            'basedir': args.dir,
            'label': 'resid',
            'overwrite': args.overwrite,
            'dir': args.dir,
            'datafile': f'preproc/functional_channel_2_moco_smooth-{args.preproc_settings["smoothing"]["fwhm"]:.1f}mu_residuals.h5',
            'cores': min(16, get_max_slurm_cpus() - 1),
        }
        workflow_dict['PCA_resid'].update(args.preproc_settings['PCA_resid'])
        args.logger.debug(f'PCA_resid workflow dict: {workflow_dict}')
        workflow_dict['PCA_moco'] = {
            'basedir': args.dir,
            'label': 'moco',
            'overwrite': args.overwrite,
            'dir': args.dir,
            'cores': min(16, get_max_slurm_cpus() - 1),
        }
        workflow_dict['PCA_moco'].update(args.preproc_settings['PCA_moco'])
        args.logger.debug(f'PCA_moco workflow dict: {workflow_dict}')

    # always overwrite for these
    if args.report or args.run_all:
        workflow_dict['check_status'] = {
            'dir': os.path.dirname(args.process),
            'cores': 2,
        }
        workflow_dict['check_status'].update(args.preproc_settings['check_status'])
        args.logger.debug(f'check_status workflow dict: {workflow_dict}')
        workflow_dict['report'] = {
            'basedir': args.process,
            'cores': 2,
        }
        workflow_dict['report'].update(args.preproc_settings['report'])
        args.logger.debug(f'report workflow dict: {workflow_dict}')

    for stepname, step_args_dict in workflow_dict.items():
        args.logger.info(f'running step: {step_args_dict["script"]}')
        args.dir = args.process  # NOTE: this is bad and confusing, but would take work to fix

        # the overwrite arg dominates - otherwise existibg files will be left alone
        step_args_dict['overwrite'] = args.overwrite

        step_output = run_preprocessing_step(step_args_dict['script'], args, step_args_dict)
        if step_output is None:
            args.logger.error(f'{step_args_dict["script"]} failed')
            raise Exception(f'{step_args_dict["script"]} failed')
        for key, value in step_output.items():
            if value != 'COMPLETED':
                args.logger.error(f'{step_args_dict["script"]} failed')
                if not args.continue_on_error:
                    raise Exception(f'{step_args_dict["script"]} failed')
        workflow_dict[stepname]['output'] = step_output
    return(workflow_dict)


def setup_build_dirs(args):
    """setup args entries for build_dirs

    Parameters:
    -----------
    args : argparse.Namespace
        args object

    Returns:
    --------
    args : argparse.Namespace
        args object with entries for build_dirs
    """
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

    args.logger.info(f'A only')
    args.logger.info(f'A: args.cores = {args.cores}')
    print(f'print A: args.cores = {args.cores}')

    args = load_default_settings_from_json(args)

    args.logger.info(f'B: args.cores = {args.cores}')
    args.logger.info(F"args.process: {args.process}")

    if not args.ignore_settings:
        args = load_user_settings_from_json(args)

    args.logger.info(f'C: args.cores = {args.cores}')

    if args.settings_file is not None:
        args = load_user_settings_from_json(args, args.settings_file)

    args.logger.info(f'D: args.cores = {args.cores}')

    if args.build:
        args.logger.info("building fly")
        args = setup_build_dirs(args)
        args.process = build_fly(args)
        if args.process is None:
            raise ValueError('fly building failed')
        args.logger.info(f'Built to flydir: {args.process}')
        if args.build_only:
            args.logger.info('build only, exiting')
            args.process = None
    # TODO: I am assuming that results of build_dirs should be passed along to fly_dirs after processing...
    # ashlu what would these "results" be?

    if args.process is not None:
        args.logger.info(f'processing {args.process}')
        setattr(args, 'script_path', os.path.dirname(os.path.realpath(__file__)))
        workflow_dict = process_fly(args)

        # ashlu should this happen in process_fly after workflow_dict instead of waiting for processing to happen?
        with open(os.path.join(args.process, 'preproc_workflow.json'), 'w') as f:
            json.dump(workflow_dict, f, indent=4)
