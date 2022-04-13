# logging utils
import os
from time import strftime
import logging
import pyfiglet
import datetime
from git_utils import get_current_git_hash


def setup_logging(args, logtype="moco", logdir=None):
    if logdir is None and "logdir" in args:
        logdir = args.logdir
    elif logdir is not None:
        setattr(args, "logdir", logdir)
    else:
        setattr(args, "logdir", None)

    if "verbose" not in args:
        setattr(args, "verbose", False)

    if args.logdir is None:
        args.logdir = os.path.join(args.dir, "logs")
    args.logdir = os.path.realpath(args.logdir)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    #  RP: use os.path.join rather than combining strings
    setattr(
        args,
        "logfile",
        os.path.join(args.logdir, strftime(f"{logtype}_%Y%m%d-%H%M%S.txt")),
    )

    #  RP: replace custom code with logging.basicConfig
    logging_handlers = [logging.FileHandler(args.logfile)]
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
