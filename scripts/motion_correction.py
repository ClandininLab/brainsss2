# pyright: reportMissingImports=false

import os
import sys
import json
import logging
import datetime
import shutil
from brainsss2.motion_correction import (
    parse_args,
    load_data,
    create_moco_output_dir,
    save_motion_parameters,
    set_stepsize,
    setup_h5_datasets,
    run_motion_correction,
    apply_moco_parameters_to_channel_2,
    save_motcorr_settings_to_json,
    moco_plot,
    save_nii,
    get_temp_dir
)
from brainsss2.argparse_utils import get_base_parser, add_moco_arguments # noqa
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.imgmath import imgmath

if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    args = get_temp_dir(args)

    args = setup_logging(args, logtype='moco')

    args = create_moco_output_dir(args)

    files, args = load_data(args)

    logging.info(f'files: {files}')

    if args.stepsize is None:
        args = set_stepsize(args)

    logging.info('set up h5 datsets')
    h5_files = setup_h5_datasets(args, files)

    if not args.use_existing:
        logging.info('running motion correction')
        transform_files, motion_parameters = run_motion_correction(args, files, h5_files)
        with open(os.path.join(args.moco_output_dir, 'transform_files.json'), 'w') as f:
            json.dump(transform_files, f, indent=4)
    else:
        with open(os.path.join(args.moco_output_dir, 'transform_files.json'), 'r') as f:
            transform_files = json.load(f)

    if 'channel_2' in files and files['channel_2'] is not None:
        logging.info('applying motion correction for channel 2')
        apply_moco_parameters_to_channel_2(args, files, h5_files, transform_files)

    logging.info('saving to json')
    save_motcorr_settings_to_json(args, files, h5_files)

    logging.info('saving motion parameters')
    motion_file = save_motion_parameters(args, motion_parameters)

    logging.info('plotting motion')
    moco_plot(args, motion_file)

    logging.info('saving mean channel 1 file')
    imgmath(h5_files['channel_1'], 'mean', outfile_type='nii')

    if args.save_nii:
        logging.info('saving nifti')
        save_nii(args, h5_files)

    shutil.rmtree(args.temp_dir)

    logging.info(f'Motion correction complete: {datetime.datetime.now()}')
