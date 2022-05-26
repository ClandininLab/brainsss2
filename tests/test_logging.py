# test logging setup
import argparse
import logging
import os
# from brainsss2.logging_utils import setup_logging
import pyfiglet
import datetime
from time import strftime


def get_logfile_name(logdir, logtype, flystring=''):
    return os.path.join(
        logdir,
        strftime(f"{logtype}{flystring}_%Y%m%d-%H%M%S.txt")
    )


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

    logging.basicConfig(
        handlers=logging_handlers,
        format="%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    print(f"logger (should be {'DEBUG' if args.verbose else 'INFO'}):", logging.getLogger())
    title = pyfiglet.figlet_format("Brainsss", font="doom")
    title_shifted = ("\n").join([" " * 20 + line for line in title.split("\n")][:-2])
    logging.info(title_shifted)
    logging.info(f"jobs started: {datetime.datetime.now()}")
    if args.verbose:
        logging.debug(f"verbose logging enabled: {args.logfile}")

    logging.info("\n\nArguments:")
    args_dict = vars(args)
    for key, value in args_dict.items():
        logging.info(f"{key}: {value}")
    logging.info("\n")
    return args


def test_setup_logging():
    args = argparse.Namespace()
    setattr(args, 'verbose', False)
    setattr(args, 'basedir', './')
    args = setup_logging(args, logtype='motion_correction')
    logging.error('TESTSTRING')
    assert logging.getLogger().level == 20
    print('handlers', logging.getLogger().handlers)
    assert os.path.exists(args.logfile)
    with open(args.logfile, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 0
    found_teststring = False
    print('lines:')
    for line in lines:
        print(line)
        if 'TESTSTRING' in line:
            found_teststring = True
    assert found_teststring


if __name__ == "__main__":
    test_setup_logging()
