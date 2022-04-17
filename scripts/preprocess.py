#!/usr/bin/env python3
# Top-level script to build and/or process a single fly
# will be wrapped by another script to allow processing of multiple flies

import sys
import os
import json
import brainsss
import logging
import argparse
from pathlib import Path
from logging_utils import setup_logging
from preprocess_utils import (
    parse_args,
    dict_to_args_list,
    load_user_settings_from_json
)


def build_fly(dir_to_build, args, use_sbatch=False):
    """build a single fly"""
    logging.info(f"building fly from {dir_to_build}")

    args = {
        "logfile": args.logfile,
        "import_date": args.import_date,
        "target_dir": args.target_dir,
        "fly_dirs": fly_dirs,
        "user": user,
    }

    script = "fly_builder.py"
    job_id = brainsss.sbatch(
        jobname="bldfly",
        script=os.path.join(scripts_path, script),
        modules=modules,
        args=args,
        logfile=logfile,
        time=1,
        mem=1,
        nice=nice,
        nodes=nodes,
    )
    func_and_anats = brainsss.wait_for_job(job_id, logfile, com_path)
    func_and_anats = func_and_anats.split("\n")[:-1]
    funcs = [
        x.split(":")[1] for x in func_and_anats if "func:" in x
    ]  # will be full paths to fly/expt
    anats = [x.split(":")[1] for x in func_and_anats if "anat:" in x]
    return None


# generate a more generic runner for func processing


def run_job(funcdir, args, sbatch_dict):
    job_ids = []
    if args.verbose:
        print(funcfiles)

    for funcfile in funcfiles:
        funcdir = os.path.dirname(funcfile)
        directory = os.path.join(func, "fictrac")
        if os.path.exists(directory):
            args_dict = {"logfile": args.logfile, "directory": directory, "fps": 100}
            script = "fictrac_qc.py"
            job_id = brainsss.sbatch(
                jobname="fictracqc",
                script=os.path.join(scripts_path, script),
                modules=modules,
                args=args_dict,
                logfile=logfile,
                time=1,
                mem=1,
                nice=nice,
                nodes=nodes,
            )
            job_ids.append(job_id)
        else:
            logging.info(f"{directory} not found, skipping fictrac_qc")
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
        directory = os.path.join(func, "fictrac")
        if os.path.exists(directory):
            args_dict = {"logfile": args.logfile, "directory": directory, "fps": 100}
            script = "fictrac_qc.py"
            job_id = brainsss.sbatch(
                jobname="fictracqc",
                script=os.path.join(scripts_path, script),
                modules=modules,
                args=args_dict,
                logfile=logfile,
                time=1,
                mem=1,
                nice=nice,
                nodes=nodes,
            )
            job_ids.append(job_id)
        else:
            logging.info(f"{directory} not found, skipping fictrac_qc")
    for job_id in job_ids:
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_stim_triggered_beh():

    ##########################
    ### Stim Triggered Beh ###
    ##########################

    for func in funcs:
        args = {"logfile": logfile, "func_path": func}
        script = "stim_triggered_avg_beh.py"
        job_id = brainsss.sbatch(
            jobname="stim",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=1,
            mem=2,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_bleaching_qc():

    ####################
    ### Bleaching QC ###
    ####################

    # job_ids = []
    for funcanat, dirtype in zip(funcanats, dirtypes):
        directory = os.path.join(funcanat, "imaging")
        args = {"logfile": logfile, "directory": directory, "dirtype": dirtype}
        script = "bleaching_qc.py"
        job_id = brainsss.sbatch(
            jobname="bleachqc",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=1,
            mem=2,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_temporal_mean_brain_pre():

    #######################################
    ### Create temporal mean brains PRE ###
    #######################################

    for funcanat, dirtype in zip(funcanats, dirtypes):
        directory = os.path.join(funcanat, "imaging")

        if dirtype == "func":
            files = ["functional_channel_1.nii", "functional_channel_2.nii"]
        if dirtype == "anat":
            files = ["anatomy_channel_1.nii", "anatomy_channel_2.nii"]

        args = {"logfile": logfile, "directory": directory, "files": files}
        script = "make_mean_brain.py"
        job_id = brainsss.sbatch(
            jobname="meanbrn",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=1,
            mem=2,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_motion_correction():

    #########################
    ### Motion Correction ###
    #########################

    for funcanat, dirtype in zip(funcanats, dirtypes):

        directory = os.path.join(funcanat, "imaging")
        # NB: 1/2 are actually anatomy/functional
        if dirtype == "func":
            brain_master = "functional_channel_1.nii"
            brain_mirror = "functional_channel_2.nii"
        if dirtype == "anat":
            brain_master = "anatomy_channel_1.nii"
            brain_mirror = "anatomy_channel_2.nii"

        args = {
            "logfile": logfile,
            "directory": directory,
            "brain_master": brain_master,
            "brain_mirror": brain_mirror,
            "scantype": dirtype,
        }

        script = "motion_correction.py"
        # if global_resources:
        #     dur = 48
        #     mem = 8
        # else:
        #     dur = 96
        #     mem = 4
        global_resources = True
        dur = 48
        mem = 8
        job_id = brainsss.sbatch(
            jobname="moco",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=dur,
            mem=mem,
            nice=nice,
            nodes=nodes,
            global_resources=global_resources,
        )
    ### currently submitting these jobs simultaneously since using global resources
    brainsss.wait_for_job(job_id, logfile, com_path)


def run_zscore():
    # TODO: check that moco file exists
    ##############
    ### ZSCORE ###
    ##############

    for func in funcs:
        load_directory = os.path.join(func, "moco")
        save_directory = os.path.join(func)
        brain_file = "functional_channel_2_moco.h5"

        args = {
            "logfile": logfile,
            "load_directory": load_directory,
            "save_directory": save_directory,
            "brain_file": brain_file,
        }
        script = "zscore.py"
        job_id = brainsss.sbatch(
            jobname="zscore",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=1,
            mem=2,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_highpass():
    # TODO: check for file existence
    ################
    ### HIGHPASS ###
    ################

    for func in funcs:

        load_directory = os.path.join(func)
        save_directory = os.path.join(func)
        brain_file = "functional_channel_2_moco_zscore.h5"

        args = {
            "logfile": logfile,
            "load_directory": load_directory,
            "save_directory": save_directory,
            "brain_file": brain_file,
        }
        script = "temporal_high_pass_filter.py"
        job_id = brainsss.sbatch(
            jobname="highpass",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=4,
            mem=2,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_correlation():

    ###################
    ### CORRELATION ###
    ###################

    for func in funcs:
        load_directory = os.path.join(func)
        save_directory = os.path.join(func, "corr")
        brain_file = "functional_channel_2_moco_zscore_highpass.h5"
        behavior = "dRotLabY"

        args = {
            "logfile": logfile,
            "load_directory": load_directory,
            "save_directory": save_directory,
            "brain_file": brain_file,
            "behavior": behavior,
        }
        script = "correlation.py"
        job_id = brainsss.sbatch(
            jobname="corr",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=2,
            mem=4,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_STA():

    #########################################
    ### STIMULUS TRIGGERED NEURAL AVERAGE ###
    #########################################

    for func in funcs:
        args = {"logfile": logfile, "func_path": func}
        script = "stim_triggered_avg_neu.py"
        job_id = brainsss.sbatch(
            jobname="STA",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=4,
            mem=4,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def run_h5_to_nii():
    # TODO: check for file existence
    #################
    ### H5 TO NII ###
    #################

    for func in funcs:
        args = {
            "logfile": logfile,
            "h5_path": os.path.join(
                func, "functional_channel_2_moco_zscore_highpass.h5"
            ),
        }
        script = "h5_to_nii.py"
        job_id = brainsss.sbatch(
            jobname="h5tonii",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=2,
            mem=10,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def temporal_mean_brain_post():
    # TODO: check that moco files exist
    #########################################
    ### Create temporal mean brains, POST ###
    #########################################

    for funcanat, dirtype in zip(funcanats, dirtypes):
        directory = os.path.join(funcanat, "moco")

        if dirtype == "func":
            files = ["functional_channel_1_moco.h5", "functional_channel_2_moco.h5"]
        if dirtype == "anat":
            files = ["anatomy_channel_1_moco.h5", "anatomy_channel_2_moco.h5"]

        args = {"logfile": logfile, "directory": directory, "files": files}
        script = "make_mean_brain.py"
        job_id = brainsss.sbatch(
            jobname="meanbrn",
            script=os.path.join(scripts_path, script),
            modules=modules,
            args=args,
            logfile=logfile,
            time=2,
            mem=10,
            nice=nice,
            nodes=nodes,
        )
        brainsss.wait_for_job(job_id, logfile, com_path)


def process_fly(fly_dir, args):
    """process a single fly"""
    logging.info(f"processing fly from {fly_dir}")

    funcs = []
    anats = []
    if args.test:
        print("test mode, not actually processing flies")
        return fly_dir
    logging.info(f"processing fly dir: {fly_dir}")
    fly_directory = os.path.join(args.target_dir, fly_dir)
    if args.dirtype == "func" or args.dirtype is None:
        funcs.extend(
            [
                os.path.join(fly_directory, x)
                for x in os.listdir(fly_directory)
                if "func" in x
            ]
        )
    if args.dirtype == "anat" or args.dirtype is None:
        anats.extend(
            [
                os.path.join(fly_directory, x)
                for x in os.listdir(fly_directory)
                if "anat" in x
            ]
        )
    logging.info(
        f"found {len(funcs)} functional files and {len(anats)} anatomical files"
    )

    brainsss.utils.sort_nicely(funcs)
    brainsss.utils.sort_nicely(anats)
    funcanats = funcs + anats
    dirtypes = ["func"] * len(funcs) + ["anat"] * len(anats)

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
    return None


def setup_modules(args):
    # change modules from a list to a single string
    if args.modules is None:
        module_list = [
            "gcc/6.3.0",
            "python/3.6",
            "py-numpy/1.14.3_py36",
            "py-pandas/0.23.0_py36",
            "viz",
            "py-scikit-learn/0.19.1_py36",
            "antspy/0.2.2",
        ]
    else:
        module_list = args.modules
    setattr(args, "modules", " ".join(module_list))


# NOTE: moving the main() function contents out of main() function to make debugging easier
if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    args = setup_modules(args)

    args = setup_logging(args, logtype='preprocess',
        logdir=os.path.join(args.target_dir, "logs"))

    if not args.ignore_settings:
        args = load_user_settings_from_json(args)

    print(args)
    sdfklj

    if args.build_fly is not None:
        assert args.import_dir is not None, "imports_path is required"
        built_flydir = build_all_flies(args)
        args.process_flydir = built_flydir

    # TODO: I am assuming that results of build_dirs should be passed along to fly_dirs after processing...

    if args.process_flydir is not None:
        processed_flies = process_all_flies(args)
