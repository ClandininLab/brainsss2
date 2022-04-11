#!/usr/bin/env python3
# Top-level script to build and/or process a single fly
# will be wrapped by another script to allow processing of multiple flies

import time
import sys
import os
import json
import brainsss
import logging
import argparse
import pyfiglet
from pathlib import Path


def parse_args(input_args):
    """parse command line arguments
    - this also includes settings that were previously in the user.json file
    - some of these do not match the previous command line arguments, but match the user.json file settings"""

    parser = argparse.ArgumentParser(description='Preprocess fly data')
    # build_flies and flies are exclusive   
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-b', '--build_fly', help='directory to build fly from')
    group.add_argument('-f', '--process_flydir', help='fly directory to process')

    parser.add_argument('--build-only', action='store_true', help="don't process after building")
    parser.add_argument('-i', '--imports_path', type=str, help='path to imports directory')
    parser.add_argument('-d', '--dataset_path', type=str, help='path to processed datasets')

    parser.add_argument('-n', '--nodes', help='number of nodes to use', type=int, default=1)
    parser.add_argument('--dirtype', help='directory type (func or anat)',
        choices=['func', 'anat'])

    parser.add_argument('-m', '--motion_correction', action='store_true', help='run motion correction')
    parser.add_argument('-z', '--zscore', action='store_true', help='temporal zscore')
    # can't use '-h' because it's a reserved word
    parser.add_argument('--highpass', action='store_true', help='highpass filter')
    # TODO: check how filter cutoff is specified...
    parser.add_argument('--highpass_cutoff', type=float, help='highpass filter cutoff')

    # TODO: clarify this
    parser.add_argument('-c', '--correlation', action='store_true', help='???')
    parser.add_argument('--fictrac_qc', action='store_true', help='run fictrac QC')
    parser.add_argument('--bleaching_qc', action='store_true', help='run bleaching QC')

    parser.add_argument('--STA', action='store_true', help='run STA')
    parser.add_argument('--STB', action='store_true', help='run STB')
    # should these be a single argument that takes "pre" or "post" as arguments?
    parser.add_argument('--temporal_mean_brain_pre', action='store_true', help='run temporal mean pre')
    parser.add_argument('--temporal_mean_brain_post', action='store_true', help='run temporal mean post')
    parser.add_argument('--h5_to_nii', action='store_true', help='run h52nii')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-w', '--width', type=int, default=120, help='width')
    parser.add_argument('--nice', action='store_true', help='nice')
    parser.add_argument('-t', '--test', action='store_true', help='test mode (set up and exit')
    parser.add_argument('--no_require_settings', action='store_true', help="don't require settings file")
    parser.add_argument('--ignore_settings', action='store_true', help="ignore settings file")

    # TODO: default log dir should be within fly dir
    parser.add_argument('-l', '--logdir', type=str, help='log directory')
    #  get user from unix rather than inferring from directory path, or allow specification
    parser.add_argument('-u', '--user', help='user', type=str, default=os.getlogin())
    parser.add_argument('-s', '--settings', help='user settings file (JSON) - will default to users/<user>.json')

    # setup for building flies
    parser.add_argument('--stim_triggered_beh', action='store_true', help='run stim_triggered_beh')

    parser.add_argument('-o', '--output', type=str, help='output directory')

    args = parser.parse_args(input_args)
    return(args)


def setup_logging(args):
    assert args.logdir is not None  # this shouldn't happen, but check just in case

    args.logdir = os.path.realpath(args.logdir)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    #  RP: use os.path.join rather than combining strings
    setattr(args, 'logfile', os.path.join(args.logdir, time.strftime("preprocess_%Y%m%d-%H%M%S.txt")))

    #  RP: replace custom code with logging.basicConfig
    logging_handlers = [logging.FileHandler(args.logfile)]
    if args.verbose:
        #  use logging.StreamHandler to echo log messages to stdout
        logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(handlers=logging_handlers, level=logging.INFO,
        format='%(asctime)s\n%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    title = pyfiglet.figlet_format("Brainsss", font="doom")
    title_shifted = ('\n').join([' ' * 42 + line for line in title.split('\n')][:-2])
    logging.info(title_shifted)
    if args.verbose:
        logging.info(f'logging enabled: {args.logfile}')

    #  NOTE: removed the printing of datetime since it's included in the logging format
    #  if you prefer without it then you could remove asctime from the format
    #  message and then print the date as before
    return(args)


def get_users_dir():
    """get the users directory
    - assumes that users directory is in main repo, one level up from scripts directory where this script should live
    """
    brainsss_basedir = Path(os.path.realpath(__file__)).parents[1].as_posix()
    return(os.path.join(brainsss_basedir, 'users'))


def load_user_settings_from_json(args):
    """load user settings from JSON file, overriding command line arguments
    - first try ~/.brainsss/settings.json, then try users/<user>.json"""

    try:
        settings_file = os.path.join(os.path.expanduser('~'), '.brainsss', 'settings.json')
        with open(settings_file) as f:
            settings = json.load(f)
        logging.info('loaded settings from %s', settings_file)
    except FileNotFoundError:
        users_dir = get_users_dir()

        if args.settings is None:
            args.settings = os.path.join(
                users_dir,
                args.user + '.json'
            )
        logging.info('using settings file %s', args.settings)

    if args.no_require_settings and not os.path.exists(args.settings):
        if args.verbose:
            logging.info('settings file not found, using default settings')
        return(args)

    # if user file doesn't exist and no_require_settings not enabled, exit and give some info
    assert os.path.exists(args.settings), f'''
settings file {args.settings} does not exist

To fix this, copy {os.path.realpath(__file__)}/user.json.example to users/{args.user}.json and update the values
OR turn settings file off with --no_require_settings'''

    with open(args.settings) as f:
        user_settings = json.load(f)
    for k, v in user_settings.items():
        setattr(args, k, v)
    logging.info(f'loaded settings from {args.settings}')
    return(args)


def fix_flydir_names(dirname):
    """ensure that flydir names start with fly_"""
    if not dirname.startswith('fly_'):
        dirname = 'fly_' + dirname
    return(dirname)


def build_fly(dir_to_build, args):
    """build a single fly"""
    logging.info(f'building fly from {dir_to_build}')

    flagged_dir = os.path.join(imports_path, dir_to_build)
    args = {'logfile': logfile, 'flagged_dir': flagged_dir, 'dataset_path': dataset_path,
        'fly_dirs': fly_dirs, 'user': user}
    script = 'fly_builder.py'
    job_id = brainsss.sbatch(jobname='bldfly',
                            script=os.path.join(scripts_path, script),
                            modules=modules,
                            args=args,
                            logfile=logfile, time=1, mem=1, nice=nice, nodes=nodes)
    func_and_anats = brainsss.wait_for_job(job_id, logfile, com_path)
    func_and_anats = func_and_anats.split('\n')[:-1]
    funcs = [x.split(':')[1] for x in func_and_anats if 'func:' in x] # will be full paths to fly/expt
    anats = [x.split(':')[1] for x in func_and_anats if 'anat:' in x]
    return(None)


# generate a more generic runner for func processing

def run_func_job(funcfiles, args, batch_dict)
    job_ids = []
    if args.verbose:
        print(funcfiles)

    for funcfile in funcfiles:
        funcdir = os.path.dirname(funcfile)
        directory = os.path.join(func, 'fictrac')
        if os.path.exists(directory):
            args_dict = {'logfile': args.logfile, 'directory': directory, 'fps': 100}
            script = 'fictrac_qc.py'
            job_id = brainsss.sbatch(jobname='fictracqc',
                                    script=os.path.join(scripts_path, script),
                                    modules=modules,
                                    args=args_dict,
                                    logfile=logfile, time=1, mem=1, nice=nice, nodes=nodes)
            job_ids.append(job_id)
        else:
            logging.info(f'{directory} not found, skipping fictrac_qc')
    for job_id in job_ids:
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_fictrac_qc(funcfiles, args):

    ##################
    ### Fictrac QC ###
    ##################

    job_ids = []
    print(funcfiles)
    for funcfile in funcfiles:
        func = os.path.dirname(funcfile)
        directory = os.path.join(func, 'fictrac')
        if os.path.exists(directory):
            args_dict = {'logfile': args.logfile, 'directory': directory, 'fps': 100}
            script = 'fictrac_qc.py'
            job_id = brainsss.sbatch(jobname='fictracqc',
                                    script=os.path.join(scripts_path, script),
                                    modules=modules,
                                    args=args_dict,
                                    logfile=logfile, time=1, mem=1, nice=nice, nodes=nodes)
            job_ids.append(job_id)
        else:
            logging.info(f'{directory} not found, skipping fictrac_qc')
    for job_id in job_ids:
        brainsss.wait_for_job(job_id, logfile, com_path)

def run_stim_triggered_beh():

    ##########################
    ### Stim Triggered Beh ###
    ##########################

    for func in funcs:
        args = {'logfile': logfile, 'func_path': func}
        script = 'stim_triggered_avg_beh.py'
        job_id = brainsss.sbatch(jobname='stim',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=1, mem=2, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

def run_bleaching_qc():

    ####################
    ### Bleaching QC ###
    ####################

    #job_ids = []
    for funcanat, dirtype in zip(funcanats, dirtypes):
        directory = os.path.join(funcanat, 'imaging')
        args = {'logfile': logfile, 'directory': directory, 'dirtype': dirtype}
        script = 'bleaching_qc.py'
        job_id = brainsss.sbatch(jobname='bleachqc',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=1, mem=2, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

def run_temporal_mean_brain_pre():

    #######################################
    ### Create temporal mean brains PRE ###
    #######################################

    for funcanat, dirtype in zip(funcanats, dirtypes):
        directory = os.path.join(funcanat, 'imaging')

        if dirtype == 'func':
            files = ['functional_channel_1.nii', 'functional_channel_2.nii']
        if dirtype == 'anat':
            files = ['anatomy_channel_1.nii', 'anatomy_channel_2.nii']

        args = {'logfile': logfile, 'directory': directory, 'files': files}
        script = 'make_mean_brain.py'
        job_id = brainsss.sbatch(jobname='meanbrn',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=1, mem=2, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

def run_motion_correction():

    #########################
    ### Motion Correction ###
    #########################

    for funcanat, dirtype in zip(funcanats, dirtypes):

        directory = os.path.join(funcanat, 'imaging')
        # NB: 1/2 are actually anatomy/functional
        if dirtype == 'func':
            brain_master = 'functional_channel_1.nii'
            brain_mirror = 'functional_channel_2.nii'
        if dirtype == 'anat':
            brain_master = 'anatomy_channel_1.nii'
            brain_mirror = 'anatomy_channel_2.nii'

        args = {'logfile': logfile,
                'directory': directory,
                'brain_master': brain_master,
                'brain_mirror': brain_mirror,
                'scantype': dirtype}

        script = 'motion_correction.py'
        # if global_resources:
        #     dur = 48
        #     mem = 8
        # else:
        #     dur = 96
        #     mem = 4
        global_resources = True
        dur = 48
        mem = 8
        job_id = brainsss.sbatch(jobname='moco',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=dur, mem=mem, nice=nice, nodes=nodes, global_resources=global_resources)
    ### currently submitting these jobs simultaneously since using global resources
    brainsss.wait_for_job(job_id, logfile, com_path)

def run_zscore():
    # TODO: check that moco file exists
    ##############
    ### ZSCORE ###
    ##############

    for func in funcs:
        load_directory = os.path.join(func, 'moco')
        save_directory = os.path.join(func)
        brain_file = 'functional_channel_2_moco.h5'

        args = {'logfile': logfile, 'load_directory': load_directory, 'save_directory': save_directory, 'brain_file': brain_file}
        script = 'zscore.py'
        job_id = brainsss.sbatch(jobname='zscore',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=1, mem=2, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

def run_highpass():
    # TODO: check for file existence
    ################
    ### HIGHPASS ###
    ################

    for func in funcs:

        load_directory = os.path.join(func)
        save_directory = os.path.join(func)
        brain_file = 'functional_channel_2_moco_zscore.h5'

        args = {'logfile': logfile, 'load_directory': load_directory, 'save_directory': save_directory, 'brain_file': brain_file}
        script = 'temporal_high_pass_filter.py'
        job_id = brainsss.sbatch(jobname='highpass',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=4, mem=2, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_correlation():

    ###################
    ### CORRELATION ###
    ###################

    for func in funcs:
        load_directory = os.path.join(func)
        save_directory = os.path.join(func, 'corr')
        brain_file = 'functional_channel_2_moco_zscore_highpass.h5'
        behavior = 'dRotLabY'

        args = {'logfile': logfile, 'load_directory': load_directory, 'save_directory': save_directory, 'brain_file': brain_file, 'behavior': behavior}
        script = 'correlation.py'
        job_id = brainsss.sbatch(jobname='corr',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=2, mem=4, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

def run_STA():

    #########################################
    ### STIMULUS TRIGGERED NEURAL AVERAGE ###
    #########################################

    for func in funcs:
        args = {'logfile': logfile, 'func_path': func}
        script = 'stim_triggered_avg_neu.py'
        job_id = brainsss.sbatch(jobname='STA',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=4, mem=4, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

def run_h5_to_nii():
    # TODO: check for file existence
    #################
    ### H5 TO NII ###
    #################

    for func in funcs:
        args = {'logfile': logfile, 'h5_path': os.path.join(func, 'functional_channel_2_moco_zscore_highpass.h5')}
        script = 'h5_to_nii.py'
        job_id = brainsss.sbatch(jobname='h5tonii',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=2, mem=10, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

def temporal_mean_brain_post():
    # TODO: check that moco files exist
    #########################################
    ### Create temporal mean brains, POST ###
    #########################################

    for funcanat, dirtype in zip(funcanats, dirtypes):
        directory = os.path.join(funcanat, 'moco')

        if dirtype == 'func':
            files = ['functional_channel_1_moco.h5', 'functional_channel_2_moco.h5']
        if dirtype == 'anat':
            files = ['anatomy_channel_1_moco.h5', 'anatomy_channel_2_moco.h5']

        args = {'logfile': logfile, 'directory': directory, 'files': files}
        script = 'make_mean_brain.py'
        job_id = brainsss.sbatch(jobname='meanbrn',
                                script=os.path.join(scripts_path, script),
                                modules=modules,
                                args=args,
                                logfile=logfile, time=2, mem=10, nice=nice, nodes=nodes)
        brainsss.wait_for_job(job_id, logfile, com_path)

 
def process_fly(fly_dir, args):
    """process a single fly"""
    logging.info(f'processing fly from {fly_dir}')

    funcs = []
    anats = []
    if args.test:
        print('test mode, not actually processing flies')
        return(fly_dir)
    logging.info(f'processing fly dir: {fly_dir}')
    fly_directory = os.path.join(args.dataset_path, fly_dir)
    if args.dirtype == 'func' or args.dirtype is None:
        funcs.extend([os.path.join(fly_directory, x) for x in os.listdir(fly_directory) if 'func' in x])
    if args.dirtype == 'anat'or args.dirtype is None:
        anats.extend([os.path.join(fly_directory, x) for x in os.listdir(fly_directory) if 'anat' in x])
    logging.info(f'found {len(funcs)} functional files and {len(anats)} anatomical files')

    brainsss.utils.sort_nicely(funcs)
    brainsss.utils.sort_nicely(anats)
    funcanats = funcs + anats
    dirtypes = ['func']*len(funcs) + ['anat']*len(anats)
    
    if args.fictrac_qc:
        fictrac_output = run_fictrac_qc(funcs, args)

    if args.stim_triggered_beh:
        stb_output = run_stim_triggered_beh()

    if args.bleaching_qc:
        bleaching_output = run_bleaching_qc()

    if args.temporal_mean_brain_pre:
        mean_brain_pre_output = run_temporal_mean_brain_pre()

    if args.motion_correction:
        motion_correction_output = run_motion_correction()

    if args.zscore:
        zscore_output = run_zscore()

    if args.highpass:
        highpass_output = run_highpass()

    if args.correlation:
        correlation_output = run_correlation()

    if args.STA:
        STA_output = run_STA()

    if args.h5_to_nii:
        h5_to_nii_output = run_h5_to_nii()

    if args.temporal_mean_brain_post:
        mean_brain_post_output = run_temporal_mean_brain_post()

    ############
    ### Done ###
    ############

    brainsss.print_footer(logfile, width)
    return(None)


def setup_modules(args):
    # change modules from a list to a single string
    if args.modules is None:
        module_list = ['gcc/6.3.0',
            'python/3.6', 'py-numpy/1.14.3_py36',
            'py-pandas/0.23.0_py36', 'viz',
            'py-scikit-learn/0.19.1_py36', 'antspy/0.2.2']
    else:
        module_list = args.modules
    setattr(args, 'modules', ' '.join(module_list))


# NOTE: moving the main() function contents out of main() function to make debugging easier
if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    args = setup_modules(args)

    args = setup_logging(args)

    if not args.ignore_settings:
        args = load_user_settings_from_json(args)

    print(args)

    assert args.dataset_path is not None, 'dataset_path is required'

    if args.build_fly is not None:
        assert args.imports_path is not None, 'imports_path is required'
        built_flydir = build_all_flies(args)
        args.process_flydir = built_flydir

    # TODO: I am assuming that results of build_dirs should be passed along to fly_dirs after processing...

    if args.process_flydir is not None:
        processed_flies = process_all_flies(args)
    
    


