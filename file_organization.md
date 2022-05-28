## File organization scheme

The overall idea here is to develop a scheme that departs minimally from the existing scheme, while also moving towards a bit more systematicity in the organization of the various files.

### Original data

The original data are located in basedir/imports

Let's say that we want fly_2 recorded on March 29, 2022.  The organization in the imports folder looks like this:

```
imports
└── 20220329
    └── fly_2
        ├── anat_0  # higher resolution image collected for anatomical registration
        │   └── TSeries-12172018-1322-009
        │       ├── References
        │       ├── TSeries-12172018-1322-009.xml
        │       ├── TSeries-12172018-1322-009_channel_1.nii  # tdTomato channel - used for primary anatomy
        │       └── TSeries-12172018-1322-009_channel_2.nii  # GCaMP channel - not used 
        ├── fly.json
        └── func_0  # lower resolution image collected for functional imaging
            ├── 2022-03-29.hdf5
            └── TSeries-12172018-1322-005
                ├── References
                ├── TSeries-12172018-1322-005.xml
                ├── TSeries-12172018-1322-005_Cycle00001_VoltageOutput_001.xml
                ├── TSeries-12172018-1322-005_Cycle00001_VoltageRecording_001.csv
                ├── TSeries-12172018-1322-005_Cycle00001_VoltageRecording_001.xml
                ├── TSeries-12172018-1322-005_channel_1.nii # tdTomato channel - used for motion correction and registration
                └── TSeries-12172018-1322-005_channel_2.nii # GCaMP channel - used for functional imaging readout
```

In addition, there is one additional necessary directory for fly building:

basedir/fictrac: this contains pairs of .log/.dat files that contain readout of ball movements

```
fictrac
├── fictrac-20220329_141649.dat
└── fictrac-20220329_141649.log
```


### Fly builder

The fly_builder.py script copies the data from the imports directory into a new fly directory, creating a new directory structure within the processed data directory (which sits in the same base directory as the imports directory).  The name of the directory is generated automatically based on the existing fly directories, which have the naming scheme "fly_XXX" where XXX is a zero-padded three digit number.  A newly built fly is given the next available number.  Conversions are tracked within processed/conversion_db.csv; this is checked during building, the same fly number will be used if a particular import date/fly number combination is built again.

The result from the fly_builder.py script will be as follows:

```
fly_009
├── anat_0
│   └── imaging
│       ├── anatomy.xml
│       └── anatomy_channel_1.nii
├── fly.json
├── func_0
    ├── QC
    ├── fictrac
    │   ├── fictrac-20220329_173736.dat
    │   ├── fictrac-20220329_173736.log
    │   └── fictrac.xml
    ├── imaging
    │   ├── TSeries-12172018-1322-005_Cycle00001_VoltageRecording_001.xml
    │   ├── functional.xml
    │   ├── functional_channel_1.nii
    │   ├── functional_channel_2.nii
    │   ├── scan.json
    │   ├── timestamps.h5
    │   └── voltage_output.xml
    ├── logs
    ├── preproc
    └── visual
        ├── 2022-03-29.hdf5
        ├── photodiode.csv
        ├── photodiode.h5
        └── stimulus_metadata.pkl

```

Logging for fly_builder.py is in the `logs` directory in the base directory (since the fly builder doesn't know the fly number until after it has started converting); all other logging happens within the specific imaging directory (e.g. fly_XXX/func_0/logs).  Additional directories are also created:

- preproc: location of preprocessed outputs from subsequent preprocessing steps
- QC: location of QC outputs from various steps

## Preprocessing steps


### Fictrac QC

outputs:

```
func_0/
├── logs
│   └── fictrac_qc_20220420-074747.log
└── QC
    ├── fictrac_2d_hist_fixed.png
    ├── fictrac_2d_hist.png
    └── fictrac_velocity_trace.png
```

### Stimulus-triggered average behavior


outputs: 

```
func_0/
├── logs
│   └── stim_triggered_avg_beh_20220420-074802.log
└── QC
    └── stim_triggered_turning.png
```

### Bleaching QC

outputs: 

```
func_0/
├── logs
│   └── bleaching_qc_20220420-074802.log
└── QC
    └── bleaching.png
```

### Motion correction

Motion correction is performed on both the functional and anatomical data.  For the functional data, the motion parameters are computed based on the tdTomato channel, and then applied both channels.

output:

```
preproc/
├── framewise_displacement.csv
├── functional_channel_1_moco.h5
├── functional_channel_1_moco_mask.nii
├── functional_channel_1_moco_mean.nii
├── functional_channel_2_moco.h5
├── moco_settings.json
├── motion_correction.png
├── motion_parameters.csv
└── transform_files.json
```

All of the settings used for motion correction are stored to `moco_settings.json`.

### Spatial smoothing

A small amount of spatial smoothing (2 mm FWHM by default) is applied to the functional channel to reduce noise.  

outputs:

```
preproc/
└── functional_channel_2_moco_smooth-2.0mu.h5
```

### Regression modeling

Two linear regression models are fitted to each voxel within the brain mask.  

#### Confound model

This model includes motion-related confounds and low-frequency discrete cosine transform regressors, which implements high-pass filtering.  The model residuals are saved (with the mean added back in) for use in later analysis steps.

outputs:

```
regression/model000_confound/
├── model000_confound_desmtx.csv
├── rsquared.nii
└── rsquared.png
preproc
└── functional_channel_2_moco_smooth-2.0mu_residuals.h5
```

#### Behavioral regression model

This model includes regressors for three behavioral features (dRotLabY, dRotLabZ+, and dRotLabZ-) as well as the confounds included in the confound model.  The p-value images are stored as 1-p for easier thresholding and visualization.

outputs:

```
regression/model001_dRotLabXYZ/
├── 1-p_dRotLabY.nii
├── 1-p_dRotLabY.png
├── 1-p_dRotLabZ+.nii
├── 1-p_dRotLabZ-.nii
├── 1-p_dRotLabZ+.png
├── 1-p_dRotLabZ-.png
├── beta_dRotLabY.nii
├── beta_dRotLabZ+.nii
├── beta_dRotLabZ-.nii
├── fdr_1-p_dRotLabY.nii
├── fdr_1-p_dRotLabY.png
├── fdr_1-p_dRotLabZ+.nii
├── fdr_1-p_dRotLabZ-.nii
├── fdr_1-p_dRotLabZ+.png
├── fdr_1-p_dRotLabZ-.png
├── model001_dRotLabXYZ_desmtx.csv
├── rsquared.nii
├── rsquared.png
├── tstat_dRotLabY.nii
├── tstat_dRotLabZ+.nii
└── tstat_dRotLabZ-.nii
```

### stimulus-triggered average neural response

outputs:

```
STA/
├── sta_0.npy
├── sta_0.png
├── sta_180.npy
└── sta_180.png
```

### Atlas registration

This step registers the functional (channel 1) data to the anatomical data, and then registers the anatomical data to the specified atlas template.
Because this combines anatomical and functional data, the results are placed in a directory called `registration` in the base fly directory.  The transform files are also saved for later use.

outputs:
```
registration/
├── 20220301_luke_2_jfrc_affine_zflip_2umiso_space-anat.nii
├── 20220301_luke_2_jfrc_affine_zflip_2umiso_space-func.nii
├── anatomy_channel_1_res-2.0mu_moco_mean_space-func.nii
├── anatomy_channel_1_res-2.0mu_moco_mean_space-jfrc.nii
├── functional_channel_1_moco_mean_space-anat.nii
├── registration.json
└── transforms
    ├── anat_to_atlas_0GenericAffine.mat
    ├── anat_to_atlas_1InverseWarp.nii.gz
    ├── anat_to_atlas_1Warp.nii.gz
    ├── func_to_anat_0GenericAffine.mat
    ├── func_to_anat_1InverseWarp.nii.gz
    └── func_to_anat_1Warp.nii.gz
```

### Clustering (i.e. "supervoxels")

Hierarchical clustering is performed separately within each slice (due to slice timing differences), and the signal from each cluster is averaged to provide a lower-dimensional summary of the data.

outputs:

```
clustering/
├── cluster_labels.nii.gz
├── cluster_labels.npy
└── cluster_signals.npy
```

### Principal component analysis

PCA is performed at the voxel level, primarily for purposes of quality control and artifact detection.  Due to the number of voxels, randomized PCA is used, so the results will likely differ across runs.   PCA is run twice, first on the motion-corrected data and then on the confound-regression residuals.

Because these data are primarily used in the QC report, they are stored in the `report` directory located in the base fly directory.

outputs:
```
report/images/func_0/PCA
├── PCA_moco_comp_000.png
...
├── PCA_moco_components.csv
├── PCA_moco.json
├── PCA_moco_timeseries_comp_000.png
...
├── PCA_moco_timeseries_comp_009.png
├── PCA_resid_comp_000.png
...
├── PCA_resid_comp_009.png
├── PCA_resid_components.csv
├── PCA_resid.json
├── PCA_resid_timeseries_comp_000.png
...
└── PCA_resid_timeseries_comp_009.png
```

### Report generation

A report is generated within the `report` directory; this can be viewed by loading `report.html` in a web browser.


