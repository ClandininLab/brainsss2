import os
import logging
import json
import shlex
import subprocess
import argparse


# from https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
def run_shell_command(command_line):
    command_line_args = shlex.split(command_line)

    logging.info('Subprocess: "' + command_line + '"')

    try:
        command_line_process = subprocess.Popen(
            command_line_args,
            stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
        )

        process_output, _ = command_line_process.communicate()

        logging.info(f'got line from subprocess: {process_output.decode("utf-8")}')
    except Exception as e:
        logging.info(f'Exception occured: {e}')
        logging.info('Subprocess failed')
        return False
    else:
        # no exception was raised
        logging.info('Subprocess finished')

    return process_output.decode("utf-8")


def dict_to_namespace(d):
    # convert dict to argparse.Namespace
    args = argparse.Namespace()
    for k, v in d.items():
        setattr(args, k, v)
    return args


def dict_to_args_list(d):
    # convert a dict to a set of command line args
    argslist = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                argslist.append(f"--{k}")
        elif isinstance(v, (int, float, str)):
            argslist.extend((f"--{k}", f"{v}"))
        elif isinstance(v, list):
            argslist.append(f"--{k}")
            argslist.extend(f"{vv}" for vv in v)
        elif v is not None:
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

    if args.no_require_settings and args.settings_file is None:
        logging.info("settings file not found, using default settings from command line")
        return args
    elif args.settings_file is None:
        raise FileNotFoundError('settings file not found, use --no_require_settings to use default settings from command line')

    assert os.path.exists(args.settings_file), f"""
settings file does not exist

To fix this, copy {os.path.realpath(__file__)}/user.json.example to ~/.brainsss/settings.json
and update the values
OR turn settings file off with --no_require_settings"""

    with open(args.settings_file) as f:
        user_settings = json.load(f)
    for k, v in user_settings.items():
        setattr(args, k, v)
    logging.info(f"loaded settings from {args.settings_file}")
    return args


def default_modules():
    return(
        [
            "gcc/6.3.0",
            "python/3.6",
            "py-numpy/1.14.3_py36",
            "py-pandas/0.23.0_py36",
            "viz",
            "py-scikit-learn/0.19.1_py36",
            "antspy/0.2.2",
        ]
    )


def setup_modules(args):
    # change modules from a list to a single string
    if args.modules is None:
        module_list = default_modules()
    else:
        module_list = args.modules + default_modules()
    module_list = list(set(module_list))
    setattr(args, "module_string", " ".join(module_list))
    return(args)
