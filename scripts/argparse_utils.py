import argparse
import os

def get_base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-b", "--basedir", type=str, help="base directory for fly data", required=True
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-l", "--logdir", type=str, help="log directory")
    parser.add_argument('-t', '--test', action='store_true', help='test mode')
    return parser


# the following will set up arguments for specific components
# they are done this way so that they can be imported into preprocess.py
# NOTE: do not use single letter args here (except in preprocess)
def add_preprocess_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build", help="build_flies", action="store_true")
    group.add_argument("--process", type=str, help="fly directory to process")
    parser.add_argument(
        "--build-only", action="store_true", help="don't process after building"
    )
    parser.add_argument(
        "-n", "--nodes", help="number of nodes to use", type=int, default=1
    )
    parser.add_argument('-a', '--run_all', action='store_true', help='run all preprocessing steps')

    # flags to run each component
    parser.add_argument(
        "--motion_correction", action="store_true", help="run motion correction"
    )
    parser.add_argument("--correlation", action="store_true", help="???")
    parser.add_argument("--fictrac_qc", action="store_true", help="run fictrac QC")
    parser.add_argument("--bleaching_qc", action="store_true", help="run bleaching QC")

    parser.add_argument("--STA", action="store_true", help="run STA")
    parser.add_argument("--STB", action="store_true", help="run STB")
    # should these be a single argument that takes "pre" or "post" as arguments?
    parser.add_argument("--temporal_mean", type=str, help="run temporal mean (pre, post, both or None)",
                        choices=['pre', 'post', 'both', 'None'])
    parser.add_argument("--h5_to_nii", action="store_true", help="run h52nii")
    parser.add_argument("--nice", action="store_true", help="nice")
    parser.add_argument(
        "--no_require_settings", action="store_true", help="don't require settings file"
    )
    parser.add_argument(
        "--ignore_settings", action="store_true", help="ignore settings file"
    )
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
    return(parser)


def add_builder_arguments(parser):
    """add arguments for fly_builder"""
    parser.add_argument(
        "--import_date", type=str, help="date of import (YYYYMMDD)"
    )
    parser.add_argument(
        "--fly_dirs",
        type=str,
        nargs='+',
        help="specific fly dirs to process for import date"
    )
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing fly dir')
    parser.add_argument(
        "--fictrac_import_dir",
        type=str,
        help="fictrac import directory (defaults to <basedir>/fictrac)"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="directory for built flies (defaults to <basedir>/processed)"
    )
    parser.add_argument(
        '--import_dir',
        type=str,
        help='imaging import directory (defaults to <basedir>/imports)'
    )
    parser.add_argument(
        "--no_visual", action="store_true", help="do not copy visual data"
    )
    parser.add_argument('--xlsx_file', type=str, help='xlsx file to use for fly data')
    return parser


