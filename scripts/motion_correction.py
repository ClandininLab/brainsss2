# pyright: reportMissingImports=false

import os
import sys
import datetime
import logging
from brainsss2.motion_correction import (
    parse_args,
    load_data,
    save_motion_parameters,
    set_stepsize,
    get_dirtype,
    setup_h5_datasets,
    run_motion_correction,
    save_motcorr_settings_to_json,
    moco_plot,
    save_nii
)
from brainsss2.argparse_utils import get_base_parser, add_moco_arguments # noqa
from brainsss2.logging_utils import setup_logging # noqa
from brainsss2.preprocess_utils import check_for_existing_files
from brainsss2.imgmath import imgmath
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype='motion_correction')

    setattr(args, 'moco_output_dir', os.path.join(args.dir, 'preproc'))
    args.logger.info(f'Moco output directory: {args.moco_output_dir}')

    check_for_existing_files(args, args.moco_output_dir, ['moco_settings.json'])

    files, args = load_data(args)

    args.logger.info(f'files: {files}')

    args = get_dirtype(args, files)

    args = set_stepsize(args)

    args.logger.info('set up h5 datsets')
    h5_files = setup_h5_datasets(args, files)

    args.logger.info('running motion correction')
    motion_parameters = run_motion_correction(args, files, h5_files)

    args.logger.info('saving motion parameters')
    motion_file = save_motion_parameters(args, motion_parameters)

    args.logger.info('plotting motion')
    moco_plot(args, motion_file)

    args.logger.info('saving mean channel 1 file')
    imgmath(h5_files['channel_1'], 'mean', outfile_type='nii')

    if args.save_nii:
        args.logger.info('saving nifti')
        save_nii(args, h5_files)

    args.logger.info('saving parameters to json')
    save_motcorr_settings_to_json(args, files, h5_files)

    args.logger.info(f'Motion correction complete: {datetime.datetime.now()}')
