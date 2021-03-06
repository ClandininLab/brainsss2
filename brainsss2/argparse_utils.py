# arg parsing setup for brainss tools
# strategy:
# all tools should start with the base parser
# each tool should have a subparser for each component
# required arguments should be added to the subparser
# or added within the parse_args for each tools
# otherwise we can get requirements across tools that are not appropriate

import argparse
import getpass


# from https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            if group_action.dest == arg:
                action._group_actions.remove(group_action)
                return


# generic arguments for all components
def get_base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-l", "--logfile", type=str, help="log file")
    parser.add_argument('-t', '--test', action='store_true', help='test mode')
    parser.add_argument('-o', '--overwrite',
        action='store_true',
        help='overwrite existing fly dir')
    return parser


# the following will set up arguments for specific components
# they are done this way so that they can be imported into preprocess.py
# NOTE: do not use single letter args here (except in preprocess)
def add_preprocess_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build", help="build_flies", action="store_true")
    group.add_argument("--process", type=str, help="fly directory to process")

    # flags to run each component
    comps = parser.add_argument_group('workflow components')
    comps.add_argument('-a', '--run_all', action='store_true', help='run all preprocessing steps')
    comps.add_argument("--fictrac_qc", action="store_true", help="run fictrac QC")
    comps.add_argument("--STB", action="store_true", help="run run stimulus-triggered behavioral averaging")
    comps.add_argument("--bleaching_qc", action="store_true", help="run bleaching QC")
    comps.add_argument(
        "--motion_correction",
        help="run motion correction (func, anat, or both - defaults to func)",
        choices=['func', 'anat', 'both']
    )
    comps.add_argument("--smoothing", action="store_true", help="run spatial smoothing")
    comps.add_argument("--STA", action="store_true", help="run stimulus-triggered neural averaging")
    comps.add_argument("--regression", action="store_true", help="run regression")
    comps.add_argument("--supervoxels", action="store_true", help="run supervoxels")
    comps.add_argument("--atlasreg", action="store_true", help="run registration to/from atlas")
    comps.add_argument("--PCA", action="store_true", help="run PCA on func data")
    comps.add_argument("--report", action="store_true", help="generate report")

    atlas = parser.add_argument_group('atlas arguments')
    atlas.add_argument(
        '--atlasfile',
        type=str,
        help='atlas file for atlasreg',
        default='20220301_luke_2_jfrc_affine_zflip_2umiso.nii')
    atlas.add_argument('--atlasname',
        type=str,
        default='jfrc',
        help='identifier for atlas space for atlasreg')
    atlas.add_argument('--atlasdir',
        type=str,
        default='/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/anat_templates',
        help='directory containing atlas files for atlasreg')

    flags = parser.add_argument_group('execution flags')
    flags.add_argument(
        "--build_only", action="store_true", help="don't process after building"
    )
    flags.add_argument(
        '--local',
        action='store_true',
        help='run locally (rather than using slurm)'
    )
    flags.add_argument(
        "--cores", help="number of cores to use", type=int, default=1
    )
    flags.add_argument(
        "--partition", help="slurm partition to use for running jobs", type=str, default='normal'
    )
    flags.add_argument("-u", "--user", help="user name (default to system username)", type=str, default=getpass.getuser())
    flags.add_argument('--modules', help='modules to load', type=str, nargs="+")
    flags.add_argument('--continue_on_error', action='store_true', help='keep going if a step fails')

    flags.add_argument(
        "--func_dirs",
        type=str,
        nargs='+',
        help="specific func dirs to process"
    )
    flags.add_argument(
        "-s",
        "--settings_file",
        help="custom settings file (overriding user/package defaults)",
    )
    flags.add_argument(
        "--ignore_settings", action="store_true", help="ignore settings file"
    )
    return(parser)

# ashlu move above preprocess arguments for logical flow
def add_builder_arguments(parser):
    """add arguments for fly_builder"""
    builder = parser.add_argument_group('fly_builder arguments')
    builder.add_argument(
        "--import_date", type=str, help="date of import (YYYYMMDD)"
    )
    builder.add_argument(
        "--fly_dirs",
        type=str,
        nargs='+',
        help="specific fly dirs to process for import date"
    )
    builder.add_argument(
        "--fictrac_import_dir",
        type=str,
        help="fictrac import directory (defaults to <basedir>/fictrac)"
    )
    builder.add_argument(
        "--target_dir",
        type=str,
        help="directory for built flies (defaults to <basedir>/processed)"
    )
    builder.add_argument(
        '--import_dir',
        type=str,
        help='imaging import directory (defaults to <basedir>/imports)'
    )
    builder.add_argument(
        "--no_visual", action="store_true", help="do not copy visual data"
    )
    builder.add_argument('--xlsx_file', type=str, help='xlsx file to use for fly data')
    return parser


def add_fictrac_qc_arguments(parser):
    parser.add_argument(
        "--fps", type=float, default=100, help="frame rate of fictrac camera"
    )
    parser.add_argument("--resolution", type=float, help="resolution of fictrac data")
    return(parser)


def add_moco_arguments(parser):
    parser.add_argument('--type_of_transform', type=str, default='SyN',
        help='type of transform to use')
    parser.add_argument('--interpolation_method', type=str, default='linear')
    parser.add_argument('--output_format', type=str, choices=['h5', 'nii'],
        default='h5', help='output format for registered image data')
    parser.add_argument('--flow_sigma', type=float, default=3,
        help='flow sigma for registration - higher sigma focuses on coarser features')
    parser.add_argument('--total_sigma', type=float, default=3,
        help='total sigma for registration - higher values will restrict the amount of deformation allowed')
    parser.add_argument('--meanbrain_n_frames', type=int, default=None,
        help='number of frames to average over when computing mean/fixed brain')
    parser.add_argument('--stepsize', type=int, help='stepsize for chunking registration')
    parser.add_argument('--save_nii', action='store_true', help='save nifti files')
    parser.add_argument('--downsample', action='store_true',
        help='downsample to lower res before moco')
    parser.add_argument('--new_resolution', type=float, default=2.,
        help='resolution to downsample to before moco')
    parser.add_argument('--dirtype', type=str, choices=['func', 'anat'])
    parser.add_argument('--use_existing', action='store_true', help='use existing transforms')
    return(parser)


def add_highpassfilter_arguments(parser):
    parser.add_argument(
        "--sigma", default=200, type=float, help="sigma for gaussian filter"
    )
    parser.add_argument(
        "--hpf_filename",
        default='preproc/functional_channel_2_moco.h5',
        type=str,
        help="filename for filtering"
    )
    return(parser)


def add_imgmath_arguments(parser):
    parser.add_argument('--fwhm', type=float, default=2.0,
        help='fwhm for smoothing')
    return(parser)


def add_dr_args(parser):
    parser.add_argument('--label', type=str, help='label for data', default='')
    parser.add_argument('--ncomps', type=int, default=10,
        help='number of components to plot')
    parser.add_argument('--ncuts', type=int, default=8,
        help='number of cuts to plot')
    parser.add_argument('--threshpct', type=int, default=90,
        help='threshold percentile for plotting')
    parser.add_argument('-d', '--basedir', type=str, required=True,
        help='fly directory')
    parser.add_argument('--outdir', type=str,
        help='output directory for plots (defaults to report/images/PCA)')
    parser.add_argument('--funcdir', type=str, default='func_0',
        help='func dir to process')
    parser.add_argument('--imgwidth', type=int, default=800)
    parser.add_argument('--imgheight', type=int, default=350)
    parser.add_argument('--datafile', type=str,
        default='preproc/functional_channel_2_moco_smooth-2.0mu.h5',
        help='func data file subpath (defaults to preproc/functional_channel_2_moco_smooth-2.0mu.h5')
    parser.add_argument('--meanfile', type=str,
        default='preproc/functional_channel_1_moco_mean.nii')
    parser.add_argument('--maskfile', type=str,
        default='preproc/functional_channel_1_moco_mask.nii')
    return(parser)
