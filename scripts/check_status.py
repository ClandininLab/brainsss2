#!/usr/bin/env python3
# Top-level script to check status of preprocessing
# pyright: reportMissingImports=false

import sys
import os
import json
import h5py
import re
import pandas as pd
from pathlib import Path
from brainsss2.argparse_utils import get_base_parser
from brainsss2.fictrac import load_fictrac


def parse_args(input):
    parser = get_base_parser('preprocess')

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="fly directory to check",
        required=True)

    return parser.parse_args(input)


def get_flyinfo(flydir):
    with open(os.path.join(flydir, 'fly.json')) as f:
        flyinfo = json.load(f)

    return flyinfo


def get_conversion_info(flydir):
    conversion_db_file = os.path.join(os.path.dirname(flydir), 'conversion_db.csv')
    print(conversion_db_file)
    if not os.path.exists(conversion_db_file):
        print('No conversion database found')
        return None
    conversion_db = pd.read_csv(conversion_db_file)
    if conversion_db is None:
        print('something went wrong reading conversion database')
        return None
    fly_conversion = conversion_db.query(f'processed_dir == "{flydir}"')
    if fly_conversion.shape[0] == 0:
        print('No conversion info found for fly')
        print(conversion_db.processed_dir)
        return None
    fly_conversion = pd.Series(fly_conversion.iloc[0, :])
    return(fly_conversion.to_dict())


def find_dirs(flydir):
    return({dirtype: {i.as_posix(): {} for i in Path(flydir).glob(f'{dirtype}_*')}
        for dirtype in ['anat', 'func']})


def check_dir(funcdir):
    if not os.path.exists(funcdir):
        print(f'{funcdir} does not exist')
        return None

    funcinfo = {'imaging': check_imaging(funcdir)}

    fictrac_dir = os.path.join(funcdir, 'fictrac')
    funcinfo['fictrac'] = check_fictrac(fictrac_dir)

    funcinfo['bleaching'] = check_bleaching_qc(funcdir)

    funcinfo['moco'] = check_moco(funcdir)

    funcinfo['visual'] = check_visual(funcdir)

    funcinfo['smoothing'] = check_smoothing(funcdir)

    funcinfo['regression'] = check_regression(funcdir)

    funcinfo['STA'] = check_STA(funcdir)

    funcinfo['atlasreg'] = check_atlasreg(funcdir)

    funcinfo['supervoxels'] = check_supervoxels(funcdir)

    return funcinfo


def check_atlasreg(funcdir):
    atlasreg_dir = os.path.join(
        os.path.dirname(funcdir),
        'registration'
    )
    if not os.path.exists(atlasreg_dir):
        return None
    reg_files = [i.as_posix() for i in Path(atlasreg_dir).glob('*nii')]

    if len(reg_files) == 0:
        return None
    else:
        return({'files': {i: {} for i in reg_files}})


def check_supervoxels(funcdir):
    supervoxel_dir = os.path.join(funcdir, 'clustering')
    if not os.path.exists(supervoxel_dir):
        return None
    sv_files = [i.as_posix() for i in Path(supervoxel_dir).glob('*nii*')]

    if len(sv_files) == 0:
        return None
    else:
        return({'files': {i: {} for i in sv_files}})


def check_smoothing(funcdir):
    preproc_dir = os.path.join(funcdir, 'preproc')
    if not os.path.exists(preproc_dir):
        return None

    moco_files = [i.as_posix() for i in Path(preproc_dir).glob('functional_channel_2_moco_smooth-*.h5')]

    if len(moco_files) == 0:
        return None
    else:
        return({'files': {i: {} for i in moco_files}})


def check_STA(funcdir):
    STA_dir = os.path.join(funcdir, 'STA')
    if not os.path.exists(STA_dir):
        return None

    sta_files = [i.as_posix() for i in Path(STA_dir).glob('sta*.npy')]

    if len(sta_files) == 0:
        return None
    else:
        return({'files': {i: {} for i in sta_files}})


def check_regression(funcdir):
    regression_dir = os.path.join(funcdir, 'regression')
    if not os.path.exists(regression_dir):
        return None

    reg_dirs = [i.as_posix() for i in Path(regression_dir).glob('model*')]

    if len(reg_dirs) == 0:
        return None
    else:
        results = {}
        for reg_dir in reg_dirs:
            if os.path.exists(
                os.path.join(regression_dir, reg_dir, 'rsquared.nii')
            ):
                results[reg_dir] = {'completed': True}
                # TODO: get some summary stats
        # NEED TO CHECK FOR FILES!
        return(results)


def check_moco(funcdir):
    preproc_dir = os.path.join(funcdir, 'preproc')
    if not os.path.exists(preproc_dir):
        return None

    moco_files = [i.as_posix() for i in Path(preproc_dir).glob('*moco.h5')]

    if not moco_files:
        return None

    moco_files.sort()

    settings_file = os.path.join(preproc_dir, 'moco_settings.json')
    if not os.path.exists(settings_file):
        return None
    with open(settings_file) as f:
        moco_info = {'settings': json.load(f),
                     'completed': True}

    files = ['motion_correction.png',
             'motion_parameters.csv',
             'framewise_displacement.csv']
    for file in files:
        if os.path.exists(os.path.join(preproc_dir, file)):
            moco_info[file] = os.path.join(preproc_dir, file)
        else:
            moco_info['completed'] = False

    # make sure that each frame has nonzero value in the middle
    # heuristic to catch empty values if moco dies part way through
    moco_info['files'] = {}
    for moco_file in moco_files:
        with h5py.File(moco_file, 'r') as f:
            moco_info['files'][moco_file] = {'shape': f['data'].shape,
                                    'completed': False}
            x, y, z = [int(i / 2) for i in f['data'].shape[:3]]
            tpvals = [f['data'][x, y, z, i] for i in range(f['data'].shape[-1])]
            if all(i > 0 for i in tpvals):
                moco_info['files'][moco_file]['completed'] = True

    return(moco_info)


def check_imaging(funcdir):
    imaging_dir = os.path.join(funcdir, 'imaging')
    if not os.path.exists(imaging_dir):
        return None
    imaging_info = {}
    infofile = os.path.join(imaging_dir, 'scan.json')
    if os.path.exists(infofile):
        with open(infofile) as f:
            imaging_info['scan'] = json.load(f)
    imaging_info['files'] = [
        os.path.join(imaging_dir, f) for f in os.listdir(imaging_dir)
        if re.search(r'(functional|anatomy)_channel_(1|2).nii', f)]
    imaging_info['files'].sort()
    return imaging_info


def check_bleaching_qc(funcdir):
    qc_dir = os.path.join(funcdir, 'QC')
    if not os.path.exists(qc_dir):
        return None
    bleaching_file = os.path.join(qc_dir, 'bleaching.png')
    bleaching_info = {}
    if os.path.exists(bleaching_file):
        bleaching_info['bleaching.png'] = bleaching_file
    return(bleaching_info)


def check_visual(funcdir):
    visual_dir = os.path.join(funcdir, 'visual')
    if not os.path.exists(visual_dir):
        return None
    visual_info = {'hdf5file': [i.as_posix() for i in Path(visual_dir).glob('*.hdf5')]}

    files_to_check = {'metadata': 'stimulus_metadata.pkl',
                      'photodiode': 'photodiode.csv'}
    for filetype, filename in files_to_check.items():
        filepath = os.path.join(visual_dir, filename)
        visual_info[filetype] = filepath if os.path.exists(filepath) else None
    qc_dir = os.path.join(funcdir, 'QC')
    if not os.path.exists(qc_dir) or not os.path.exists(
            os.path.join(qc_dir, 'stim_triggered_turning.png')):
        visual_info['STB'] = None
    else:
        visual_info['STB'] = os.path.join(qc_dir, 'stim_triggered_turning.png')

    return(visual_info)


def check_fictrac(fictrac_dir):
    if not os.path.exists(fictrac_dir):
        return None
    fictrac_data = load_fictrac(fictrac_dir)

    fictrac_info = {'dir': fictrac_dir,
                    'shape': fictrac_data.shape}

    qa_dir = os.path.join(os.path.dirname(fictrac_dir), 'QC')
    if not os.path.exists(qa_dir):
        fictrac_info['QC'] = None
    else:
        fictrac_info['QC'] = {}
        for file in ['fictrac_2d_hist_fixed.png',
                     'fictrac_2d_hist.png',
                     'fictrac_velocity_trace.png']:
            if os.path.exists(os.path.join(qa_dir, file)):
                fictrac_info['QC'][file] = os.path.join(qa_dir, file)
    return(fictrac_info)


def check_all_status(flyinfo):
    """check status of all processing steps"""
    for dirtype in ['anat', 'func']:
        print(f'\nchecking {dirtype}')
        for dir, dirinfo in flyinfo['dirs'][dirtype].items():
            print(f'found {os.path.basename(dir)}')
            if dirinfo['imaging'] is None:
                print("No imaging data found")
            else:
                print('Found imaging files:')
                for file in dirinfo['imaging']['files']:
                    print(f'    {os.path.basename(file)}')

            if dirtype == 'func' and dirinfo['fictrac'] is None:
                print("No fictrac data found")
            elif dirtype == 'func':
                print(f"Found fictrac data (shape: {dirinfo['fictrac']['shape']})")
                if len(dirinfo['fictrac']['QC']) == 3:
                    print('    Fictrac QC complete')
                else:
                    print('    Fictrac QC incomplete')

            if dirtype == 'func' and dirinfo['visual'] is None:
                print("No visual data found")
            elif dirtype == 'func':
                print("Found visual data")
                if dirinfo['visual']['metadata'] is not None \
                        and dirinfo['visual']['photodiode'] is not None:
                    print('    Stimulus metadata complete')
                else:
                    print('    Stimulus metadata incomplete')
                if dirinfo['visual']['STB'] is not None:
                    print('    Stimulus-triggered behavior complete')
                else:
                    print('    Stimulus-triggered behavior incomplete')

            if dirtype == 'func' and (
                dirinfo['bleaching'] is None or
                dirinfo['bleaching']['bleaching.png'] is None
            ):
                print("Bleaching QC not incomplete")
            elif dirtype == 'func':
                print("Bleaching QC complete")

            if dirinfo['moco'] is None:
                print("No motion correction data found")
            else:

                moco_completed = dirinfo['moco']['completed']
                for file, fileinfo in dirinfo['moco']['files'].items():
                    if not fileinfo['completed']:
                        moco_completed = False
                if not moco_completed:
                    print("Motion correction incomplete")
                else:
                    print('Motion correction complete')
                    for file in dirinfo['moco']['files']:
                        print(f'    {os.path.basename(file)}')

                if dirinfo['smoothing'] is None:
                    print("No smoothed data found")
                else:
                    print("Smoothing complete")
                    for file in dirinfo['smoothing']['files']:
                        print(f'    {os.path.basename(file)}')

            if dirinfo['regression'] is None and dirtype == 'func':
                print("No regression results found")
            elif dirtype == 'func':
                print("Regression complete")
                for model in dirinfo['regression']:
                    print(f'    {os.path.basename(model)}')

            if dirinfo['STA'] is None and dirtype == 'func':
                print("No STA results found")
            elif dirtype == 'func':
                print("STA complete")
                for file in dirinfo['STA']['files']:
                    print(f'    {os.path.basename(file)}')

            if dirinfo['atlasreg'] is None and dirtype == 'func':
                print("No registration results found")
            elif dirtype == 'func':
                print("Registration complete")
                for file in dirinfo['atlasreg']['files']:
                    print(f'    {os.path.basename(file)}')

            if dirinfo['supervoxels'] is None and dirtype == 'func':
                print("No supervoxel results found")
            elif dirtype == 'func':
                print("Supervoxel creation complete")
                for file in dirinfo['supervoxels']['files']:
                    print(f'    {os.path.basename(file)}')


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    assert os.path.exists(args.dir), f"Directory {args.dir} does not exist"
    print(f'Checking status of building/preprocessing for {args.dir}')

    flyinfo = get_flyinfo(args.dir)

    flyinfo['conversion'] = get_conversion_info(args.dir)

    flyinfo['dirs'] = find_dirs(args.dir)

    # check func dirs

    for funcdir in flyinfo['dirs']['func']:
        flyinfo['dirs']['func'][funcdir] = check_dir(funcdir)

    for anatdir in flyinfo['dirs']['anat']:
        flyinfo['dirs']['anat'][anatdir] = check_dir(anatdir)

    flyinfo_file = os.path.join(args.dir, 'fly_processing_info.json')
    with open(flyinfo_file, 'w') as f:
        json.dump(flyinfo, f, indent=4)

    check_all_status(flyinfo)
