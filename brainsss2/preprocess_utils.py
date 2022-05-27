import os
import logging
import json
import shlex
import subprocess
import argparse
import pkg_resources


# from https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
def run_shell_command(command_line, verbose=False):
    command_line_args = shlex.split(command_line)

    if verbose:
        logging.info('Subprocess: "' + command_line + '"')

    try:
        command_line_process = subprocess.Popen(
            command_line_args,
            stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
        )

        process_output, _ = command_line_process.communicate()

        if verbose:
            logging.info(f'got line from subprocess: {process_output.decode("utf-8")}')
    except Exception as e:
        logging.error(f'Exception occured: {e}')
        return False
    else:
        # no exception was raised
        if verbose:
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
            pass  # raise ValueError(f"unsupported type {type(v)}")
    return argslist


def load_default_settings_from_json(args):
    """load default settings from the package
    settings are structured as a dict of dicts
    each top level dict is a preprocessing step (named as in the workflow dict in preprocess.py)
    each next level dict contains settings for that step, with the same keys as the workflow dict
    """
    default_settings_file = pkg_resources.resource_filename('brainsss2', 'settings/settings.json')
    with open(default_settings_file) as f:
        setattr(args, 'default_settings', json.load(f))
    logging.info("loaded default settings")
    return args


def load_user_settings_from_json(args, user_settings_file=None):
    """load user settings from JSON file ( ~/.brainsss/settings.json),
    and overwrite default settings with any user settings

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    user_settings_file : str
        path to settings file (defaults to user file)

    Returns
    -------
    args : argparse.Namespace
    """

    assert args.default_settings is not None, "default settings just be created before adding user settings"
    # start with default settings
    setattr(args, 'preproc_settings', args.default_settings)

    if user_settings_file is None:
        user_settings_file = os.path.join(
            os.path.expanduser("~"), ".brainsss", "settings.json"
        )

    if not os.path.exists(user_settings_file):
        logging.info(f"settings file {user_settings_file} not found, using package default settings")
        return args

    with open(user_settings_file) as f:
        user_settings = json.load(f)
    logging.info(f"loaded user settings from {user_settings_file}")

    # overwrite default settings with user settings
    for step in user_settings:
        for key, value in user_settings[step].items():
            args.preproc_settings[step][key] = value
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
