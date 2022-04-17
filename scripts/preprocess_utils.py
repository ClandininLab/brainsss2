import argparse
import os
import logging
import json


def parse_args(input_args):
    """parse command line arguments
    - this also includes settings that were previously in the user.json file
    - some of these do not match the previous command line arguments, but match the user.json file settings"""

    parser = argparse.ArgumentParser(description="Preprocess fly data")
    # build_flies and flies are exclusive
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--build_fly", help="directory to build fly from")
    group.add_argument("-f", "--process_flydir", help="fly directory to process")

    parser.add_argument(
        "--build-only", action="store_true", help="don't process after building"
    )
    parser.add_argument(
        "-i", "--import_dir", type=str, help="imports directory"
    )
    parser.add_argument(
        "-d", "--target_dir", type=str, help="path to processed datasets",
        required=True
    )

    parser.add_argument(
        "-n", "--nodes", help="number of nodes to use", type=int, default=1
    )
    parser.add_argument(
        "--dirtype", help="directory type (func or anat)", choices=["func", "anat"]
    )

    parser.add_argument(
        "-m", "--motion_correction", action="store_true", help="run motion correction"
    )
    parser.add_argument("-z", "--zscore", action="store_true", help="temporal zscore")
    # can't use '-h' because it's a reserved word
    parser.add_argument("--highpass", action="store_true", help="highpass filter")
    # TODO: check how filter cutoff is specified...
    parser.add_argument("--highpass_cutoff", type=float, help="highpass filter cutoff")

    # TODO: clarify this
    parser.add_argument("-c", "--correlation", action="store_true", help="???")
    parser.add_argument("--fictrac_qc", action="store_true", help="run fictrac QC")
    parser.add_argument("--bleaching_qc", action="store_true", help="run bleaching QC")

    parser.add_argument("--STA", action="store_true", help="run STA")
    parser.add_argument("--STB", action="store_true", help="run STB")
    # should these be a single argument that takes "pre" or "post" as arguments?
    parser.add_argument(
        "--temporal_mean_brain_pre", action="store_true", help="run temporal mean pre"
    )
    parser.add_argument(
        "--temporal_mean_brain_post", action="store_true", help="run temporal mean post"
    )
    parser.add_argument("--h5_to_nii", action="store_true", help="run h52nii")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-w", "--width", type=int, default=120, help="width")
    parser.add_argument("--nice", action="store_true", help="nice")
    parser.add_argument(
        "-t", "--test", action="store_true", help="test mode (set up and exit"
    )
    parser.add_argument(
        "--no_require_settings", action="store_true", help="don't require settings file"
    )
    parser.add_argument(
        "--ignore_settings", action="store_true", help="ignore settings file"
    )

    # TODO: default log dir should be within fly dir
    parser.add_argument("-l", "--logdir", type=str, help="log directory")
    #  get user from unix rather than inferring from directory path, or allow specification
    parser.add_argument("-u", "--user", help="user", type=str, default=os.getlogin())
    parser.add_argument(
        "-s",
        "--settings_file",
        help="user settings file (JSON) - defaults to ~/.brainsss/settings.json",
    )
    # setup for building flies
    parser.add_argument(
        "--stim_triggered_beh", action="store_true", help="run stim_triggered_beh"
    )
    parser.add_argument('--modules', help='modules to load', type=str, nargs="+")
    parser.add_argument("-o", "--output", type=str, help="output directory")

    return parser.parse_args(input_args)


def dict_to_args_list(d):
    # convert a dict to a set of command line args
    argslist = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                argslist.append(f"--{k}")
        elif isinstance(v, (int, float, str)):
            argslist.append(f"--{k} {v}")
        elif isinstance(v, list):
            argslist.extend(f"--{k} {vv}" for vv in v)
        else:
            raise ValueError(f"unsupported type {type(v)}")
    return argslist


def load_user_settings_from_json(args):
    """load user settings from JSON file, overriding command line arguments
    - first try ~/.brainsss/settings.json, then try users/<user>.json"""

    if args.settings_file is None:
        # first try ~/.brainsss/settings.json
        user_settings_file = os.path.join(
            os.path.expanduser("~"), ".brainsss", "settings.json"
        )
        if os.path.exists(user_settings_file):
            args.settings_file = user_settings_file
    elif args.settings_file is not None and not args.no_require_settings:
        # if user file doesn't exist and no_require_settings not enabled, exit and give some info
        assert os.path.exists(args.settings_file), f"""
settings file {args.settings_file} does not exist

To fix this, copy {os.path.realpath(__file__)}/user.json.example to ~/.brainsss/settings.json 
and update the values
OR turn settings file off with --no_require_settings"""
    else:
        logging.info("settings file not found, using default settings from command line")
        return args

    with open(args.settings_file) as f:
        user_settings = json.load(f)
    for k, v in user_settings.items():
        setattr(args, k, v)
    logging.info(f"loaded settings from {args.settings_file}")
    return args
