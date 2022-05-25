# create package file with default workflow args
import os
import json

if __name__ == "__main__":
    workflow_dict = {}

    workflow_dict['fictrac_qc'] = {
        'script': 'fictrac_qc.py',
        "fps": 100
    }

    workflow_dict['stim_triggered_avg_beh'] = {
        'script': 'stim_triggered_avg_beh.py',
        'cores': 2
    }

    workflow_dict['bleaching_qc'] = {
        'script': 'bleaching_qc.py',
    }

    workflow_dict['motion_correction_func'] = {
        'script': 'motion_correction.py',
        'type_of_transform': 'SyN',
        'time_hours': 24,
        'dirtype': 'func'
    }

    workflow_dict['motion_correction_anat'] = {
        'script': 'motion_correction.py',
        'type_of_transform': 'SyN',
        'time_hours': 8,
        'stepsize': 4,
        'downsample': True,
        'dirtype': 'anat'
    }

    workflow_dict['smoothing'] = {
        'script': 'imgmath.py',
        'operation': 'smooth',
        'fwhm': 2.0,
    }

    workflow_dict['regression_XYZ'] = {
        'script': 'regression.py',
        'overwrite': True,
        'label': 'model001_dRotLabXYZ',
        'confound_files': 'preproc/framewise_displacement.csv',
        'time_hours': 1,
        'behavior': ['dRotLabY', 'dRotLabZ+', 'dRotLabZ-'],
    }

    workflow_dict['regression_confound'] = {
        'script': 'regression.py',
        'overwrite': True,
        'save_residuals': True,
        'label': 'model000_confound',
        'confound_files': 'preproc/framewise_displacement.csv',
        'time_hours': 1
    }

    workflow_dict['STA'] = {
        'script': 'stim_triggered_avg_neu.py',
        'overwrite': True,
        'time_hours': 1
    }

    workflow_dict['atlasreg'] = {
        'script': 'atlas_registration.py',
        'type_of_transform': 'SyN',
        'atlasname': 'jfrc',
        'overwrite': True,
        'time_hours': 8
    }

    workflow_dict['supervoxels'] = {
        'script': 'make_supervoxels.py',
        'overwrite': True,
        'time_hours': 1
    }

    user_settings_file = os.path.join("../brainsss2/settings/settings.json")
    if not os.path.exists(os.path.dirname(user_settings_file)):
        os.makedirs(os.path.dirname(user_settings_file))

    with open(user_settings_file, 'w') as f:
        json.dump(workflow_dict, f, indent=4)
