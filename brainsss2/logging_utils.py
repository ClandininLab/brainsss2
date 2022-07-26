# logging utils

# pyright: reportMissingImports=false

import os
from time import strftime
import logging
import pyfiglet
import datetime


def get_logfile_name(logdir, logtype, flystring=''):
    return os.path.join(
        logdir,
        strftime(f"{logtype}{flystring}_%Y%m%d-%H%M%S.txt")
    )


def remove_existing_file_handlers():
    logger = logging.getLogger()
    saved_handlers = []
    for h in logger.handlers:
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

    if 'dir' in args and args.dir is not None:
        dir_split = args.dir.split('/')
        for part in dir_split:
            if 'fly_' in part:
                flystring = '_' + part.replace('_', '').replace('-', '')
                break
    return flystring


def setup_logging(args, logtype, logdir=None, logfile=None, preamble=True):
    """
    Setup logging for the script.

    Parameters:
    -----------
    args: argparse.Namespace
        command line arguments
    logtype: str
        name of process to use for logfile naming
    logfile: str
        file to save logfile in
    preamble: bool
        whether to print preamble to logfile

    Returns:
    --------
    args: argparse.Namespace
        command line arguments (updated with logging info)
    """

    if 'dir' not in args:
        setattr(args, 'dir', None)

    # fx argument has priority over args.logfile
    if logfile is not None:
        if args.verbose:
            print('setting log file using fx argument')
        setattr(args, 'logfile', logfile)
        setattr(args, 'logdir', os.path.dirname(logfile))
    elif 'logfile' in args and args.logfile is not None:
        if args.verbose:
            print('setting log file from args')
        setattr(args, 'logfile', args.logfile)
        setattr(args, 'logdir', os.path.dirname(args.logfile))
    else:
        # come up with a logfile name
        if logdir is not None:
            setattr(args, "logdir", logdir)
        elif logdir is None and "logdir" in args:
            logdir = args.logdir
        else:
            setattr(args, "logdir", None)

        if args.logdir is None:
            # try to put into dir if possible
            if args.dir is not None:
                if args.verbose:
                    print('setting log dir from args.dir')
                args.logdir = os.path.join(args.dir, 'logs')
            # fall back on basedir
            elif args.basedir is not None:
                if args.verbose:
                    print('setting log dir from args.basedir')
                args.logdir = os.path.join(args.basedir, 'logs')
            else:
                raise ValueError("args.dir or args.basedir must be specified if args.logdir is not")
        args.logdir = os.path.realpath(args.logdir)
        if args.verbose:
            print('logdir:', args.logdir)
        setattr(
            args,
            "logfile",
            get_logfile_name(args.logdir, logtype),
        )

    if args.verbose:
        print(f'{logtype}: logging to {args.logfile}')

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    if "verbose" not in args:
        setattr(args, "verbose", False)

    #  RP: replace custom code with logging.basicConfig
    setattr(args, 'file_handler', logging.FileHandler(args.logfile))
    logging_handlers = [args.file_handler]
    # set level of individual handlers
    logger = logging.getLogger('brainsss')
    for handler in logging_handlers:
        logger.addHandler(handler)

    # logging.basicConfig(
    #     handlers=logging_handlers,
    #     format="%(message)s",
    #     datefmt="%m/%d/%Y %I:%M:%S %p",
    # )
    formatter = logging.Formatter("%(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)
        if args.verbose:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)

    # set level of root logger
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    print(f"logger (should be {'DEBUG' if args.verbose else 'INFO'}):", logger)
    title = pyfiglet.figlet_format("Brainsss", font="doom")
    title_shifted = ("\n").join([" " * 20 + line for line in title.split("\n")][:-2])
    logger.info(title_shifted)
    logger.info(f"jobs started: {datetime.datetime.now()}")
    if args.verbose:
        logger.debug(f"verbose logging enabled: {args.logfile}")

    logger.info("abc")
    logger.info("\n\nArguments:")
    args_dict = vars(args)
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("\n")

    logger.info("defgh")
    setattr(args, 'logger', logger)
    logger.info("1")
    args.logger.info("2")

    return args
