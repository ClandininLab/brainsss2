# create static html report for fly preprocessing
from jinja2 import Environment, FileSystemLoader
import os
import json
import shutil
import nibabel as nib
import numpy as np
import nilearn.plotting
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# generic class to create an object from a dictionary
# from https://stackoverflow.com/questions/56092894/iterating-over-a-list-of-dictionaries-in-jinja2-template
class Item:
    def __init__(self, vals):
        self.__dict__ = vals


# class to represent an anatomical directory
# use this to make jinja templating easier
class DataDir:
    def __init__(self, datadir, data_dict, reportdir):
        # raw files
        self.files = data_dict['imaging']['files']
        for f in self.files:
            self.files[f]['channel'] = 1 if 'channel_1' in f else 2
        self.label = os.path.basename(datadir)

        if 'func' in self.label:
            qcdir = os.path.join('images', self.label, 'QC')
            qcdir_full = os.path.join(reportdir, qcdir)
            if not os.path.exists(qcdir_full):
                os.makedirs(qcdir_full)
            # fictrac
            if 'fictrac' in data_dict and 'QC' in data_dict['fictrac']:
                self.fictrac = data_dict['fictrac']
                for key, qcfile in data_dict['fictrac']['QC'].items():
                    shutil.copy(qcfile, qcdir_full)
                    setattr(self, key.replace('.png', ''), os.path.join(qcdir, key))

            else:
                self.fictrac = None

            # bleaching QC

            if 'bleaching' in data_dict:
                self.bleaching = data_dict['bleaching']
                if data_dict['bleaching']['bleaching.png'] is not None:
                    shutil.copy(data_dict['bleaching']['bleaching.png'], qcdir_full)
                self.bleaching_plot = os.path.join(qcdir, 'bleaching.png')
            else:
                self.bleaching = None
                self.bleaching_plot = None

            if 'smoothing' in data_dict and data_dict['smoothing'] is not None:
                self.smoothed_files = data_dict['smoothing']['files']
            else:
                self.smoothed_files = {}

            if 'regression' in data_dict and data_dict['regression'] is not None:
                regdir = os.path.join('images', self.label, 'regression')
                regdir_full = os.path.join(reportdir, regdir)
                if not os.path.exists(regdir_full):
                    os.makedirs(regdir_full)
                self.regression = data_dict['regression']
                for modeldir, model in self.regression.items():
                    print(modeldir, model)
                    if not model['completed']:
                        continue
                    if 'confound' in modeldir:
                        self.regression_confound = model
                        self.regression_confound_rsquared = os.path.join(
                            regdir, 'confound_rsquared.png')
                        shutil.copy(model['rsquared'], os.path.join(
                            regdir_full, 'confound_rsquared.png'))
                    else:
                        self.regression_model = model
                        self.regression_pvals = {}
                        # make desmtx correlation plot
                        desmtx = pd.read_csv(model['desmtx'])
                        fig = plt.figure(figsize=(10, 10))
                        ax = plt.gca()
                        _ = sns.heatmap(desmtx.corr(), cmap='viridis', ax=ax)
                        fig.tight_layout(rect=[0, 0, .9, 1])
                        fig.savefig(os.path.join(regdir_full, 'desmtx_corr.png'))
                        del fig, ax
                        self.regression_model_desmtx_corr = os.path.join(regdir, 'desmtx_corr.png')
                        self.regression_model_rsquared = os.path.join(
                            regdir, 'confound_rsquared.png')
                        shutil.copy(model['rsquared'], os.path.join(
                            regdir_full, 'model_rsquared.png'))
                        for k, v in model['pvals'].items():
                            shutil.copy(v, os.path.join(regdir_full, k + '.png'))
                            self.regression_pvals[k] = os.path.join(regdir, k + '.png')
            else:
                self.regression = None

            if 'atlasreg' in data_dict and data_dict['atlasreg'] is not None:
                regdir = os.path.join('images', self.label, 'registration')
                regdir_full = os.path.join(reportdir, regdir)
                if not os.path.exists(regdir_full):
                    os.makedirs(regdir_full)
                self.registration = Item(data_dict['atlasreg'])
                # create png images for the registration outcomes
                # first create original anatomical image
                anat_img = nib.load(os.path.join(self.registration.dir, self.registration.anatfile))
                self.registration.orig_anat_file_png = os.path.join(regdir, 'orig_anat.png')
                cut_coords = cut_coords = np.arange(8, 49, 8) * anat_img.header.get_zooms()[2]
                nilearn.plotting.plot_anat(anat_img, cut_coords=cut_coords,
                    display_mode='z', draw_cross=False,
                    output_file=os.path.join(regdir_full, 'orig_anat.png'))
                # then create functional image aligned to anatomy
                func_img = nib.load(self.registration.func_reg_to_anat_file)
                self.registration.func_to_anat_png = os.path.join(regdir, 'func_to_anat.png')
                nilearn.plotting.plot_anat(func_img, cut_coords=cut_coords,
                    display_mode='z', draw_cross=False,
                    output_file=os.path.join(regdir_full, 'func_to_anat.png'))
                # then create atlas aligned to anatomy
                atlas_img = nib.load(self.registration.atlas_to_mean_reg_file)
                self.registration.atlas_to_anat_png = os.path.join(regdir, 'atlas_to_anat.png')
                nilearn.plotting.plot_anat(atlas_img, cut_coords=cut_coords,
                    display_mode='z', draw_cross=False,
                    output_file=os.path.join(regdir_full, 'atlas_to_anat.png'))


        self.moco_completed = data_dict['moco']['completed']
        if not self.moco_completed:
            return None
        if os.path.exists(data_dict['moco']['motion_correction.png']):
            imgdir = os.path.join('images', self.label, 'moco')
            imgdir_full = os.path.join(reportdir, imgdir)
            if not os.path.exists(imgdir_full):
                os.makedirs(imgdir_full)
            shutil.copy(data_dict['moco']['motion_correction.png'], imgdir_full)

        self.moco_plot = os.path.join(imgdir, 'motion_correction.png')
        self.moco_files = data_dict['moco']['files']


if __name__ == '__main__':

    root = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(os.path.dirname(root), 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('index.html')

    flynum = 'fly_001'
    flydir = f'/data/brainsss/processed/{flynum}'
    assert os.path.exists(flydir), f'{flydir} does not exist'

    metadata_file = os.path.join(flydir, 'fly_processing_info.json')
    assert os.path.exists(metadata_file), f'{metadata_file} does not exist'
    with open(metadata_file, 'r') as f:
        flyinfo = json.load(f)

    reportdir = os.path.join(flydir, 'report')
    if not os.path.exists(reportdir):
        os.mkdir(reportdir)

    anat_info = {k: DataDir(k, v, reportdir) for k, v in flyinfo['dirs']['anat'].items()}
    func_info = {k: DataDir(k, v, reportdir) for k, v in flyinfo['dirs']['func'].items()}

    # # create slice images for anat and func means
    # for k, v in anat_info.items():
    #     for file in v.files:
    #         print(f'loading file {file}')
    #         img = nib.load(file)

    filename = os.path.join(reportdir, 'report.html')
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as fh:
        fh.write(template.render(
            h1=f"Fly preprocessing report: {flynum}",
            fly_metadata=flyinfo['metadata'],
            anat_info=anat_info,
            func_info=func_info,
            imgwidth=800,
            imgheight=350
        ))
