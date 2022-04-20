# logging utils
import os
from time import strftime
import logging
import pyfiglet
import datetime
from git_utils import get_current_git_hash


def get_logfile_name(logdir, logtype, flystring):
    return os.path.join(
        logdir,
        strftime(f"{logtype}{flystring}_%Y%m%d-%H%M%S.txt")
    )


def remove_existing_file_handlers():
    l = logging.getLogger()
    saved_handlers = []
    for h in l.handlers:
        print(h)
        if isinstance(h, logging.FileHandler):
            saved_handlers.append(h)
            logging.getLogger().removeHandler(h)
    return saved_handlers


def reinstate_file_handlers(saved_handlers):
    """
    Reinstate saved file handlers.

    Parameters:
    -----------
    saved_handlers: list
        list of saved file handlers
    """
    
    for h in list(set(saved_handlers)):
        logging.getLogger().addHandler(h)


def get_flystring(args):
    """
    Get flystring from args.

    Parameters:
    -----------
    args: argparse.Namespace
        command line arguments

    Returns:
    --------
    flystring: str
        string to use for logfile naming
    """
    flystring = ""

    if "flystring" in args and args.flystring is not None:
        return(args.flystring)

    if 'dir' in args:
        dir_split = args.dir.split('/')
        for part in dir_split:
            if 'fly_' in part:
                flystring = '_' + part.replace('_', '').replace('-', '')
                break
    return flystring


def setup_logging(args, logtype, logdir=None, preamble=True):
    """
    Setup logging for the script.

    Parameters:
    -----------
    args: argparse.Namespace
        command line arguments
    logtype: str
        name of process to use for logfile naming
    logdir: str
        directory to save logfile in
    preamble: bool
        whether to print preamble to logfile

    Returns:
    --------
    args: argparse.Namespace
        command line arguments (updated with logging info)
    """
    if logdir is None and "logdir" in args:
        logdir = args.logdir
    elif logdir is not None:
        setattr(args, "logdir", logdir)
    else:
        setattr(args, "logdir", None)

    if args.logdir is None:
        if args.dir is not None:
            args.logdir = os.path.join(args.dir, 'logs')
        elif args.basedir is not None:
            args.logdir = os.path.join(args.basedir, 'logs')
        else:
            raise ValueError("args.dir or args.basedir must be specified if args.logdir is not")
    args.logdir = os.path.realpath(args.logdir)

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    if "verbose" not in args:
        setattr(args, "verbose", False)

    setattr(args, 'flystring', get_flystring(args))

    #  RP: use os.path.join rather than combining strings
    setattr(
        args,
        "logfile",
        get_logfile_name(args.logdir, logtype, args.flystring),
    )

    #  RP: replace custom code with logging.basicConfig
    setattr(args, 'file_handler', logging.FileHandler(args.logfile))
    logging_handlers = [args.file_handler]
    if args.verbose:
        #  use logging.StreamHandler to echo log messages to stdout
        logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        handlers=logging_handlers,
        level=logging.INFO,
        format="%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    title = pyfiglet.figlet_format("Brainsss", font="doom")
    title_shifted = ("\n").join([" " * 42 + line for line in title.split("\n")][:-2])
    logging.info(title_shifted)
    logging.info(f"jobs started: {datetime.datetime.now()}")
    setattr(args, "git_hash", get_current_git_hash())
    logging.info(f"git commit: {args.git_hash}")
    if args.verbose:
        logging.info(f"logging enabled: {args.logfile}")

    logging.info("\n\nArguments:")
    args_dict = vars(args)
    for key, value in args_dict.items():
        logging.info(f"{key}: {value}")
    logging.info("\n")
    return args
