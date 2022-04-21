## File organization scheme

The overall idea here is to develop a scheme that departs minimally from the existing scheme, while also moving towards a bit more systematicity in the organization of the various files.

### Original data

The original data are located in basedir/imports

Let's say that we want fly_2 recorded on March 29, 2022.  The organization in the imports folder looks like this:

imports
├── 20220329
│   └── fly_2
│       ├── anat_0  # higher resolution image collected for anatomical registration
│       │   └── TSeries-12172018-1322-009
│       │       ├── References
│       │       ├── TSeries-12172018-1322-009.xml
│       │       ├── TSeries-12172018-1322-009_channel_1.nii  # tdTomato channel - used for primary anatomy
│       │       └── TSeries-12172018-1322-009_channel_2.nii  # GCaMP channel - not used 
│       ├── fly.json
│       ├── func_0  # lower resolution image collected for functional imaging
│       │   ├── 2022-03-29.hdf5
│       │   └── TSeries-12172018-1322-005
│       │       ├── References
│       │       ├── TSeries-12172018-1322-005.xml
│       │       ├── TSeries-12172018-1322-005_Cycle00001_VoltageOutput_001.xml
│       │       ├── TSeries-12172018-1322-005_Cycle00001_VoltageRecording_001.csv
│       │       ├── TSeries-12172018-1322-005_Cycle00001_VoltageRecording_001.xml
│       │       ├── TSeries-12172018-1322-005_channel_1.nii # tdTomato channel - used for motion correction and registration
│       │       └── TSeries-12172018-1322-005_channel_2.nii # GCaMP channel - used for functional imaging readout
│       └── func_1  # ???
│           └── TSeries-12172018-1322-006
│               ├── TSeries-12172018-1322-006.xml
│               ├── TSeries-12172018-1322-006_Cycle00001_VoltageOutput_001.xml
│               ├── TSeries-12172018-1322-006_Cycle00001_VoltageRecording_001.csv
│               ├── TSeries-12172018-1322-006_Cycle00001_VoltageRecording_001.xml
│               ├── TSeries-12172018-1322-006_channel_1.nii
│               └── TSeries-12172018-1322-006_channel_2.nii
└── build_logs  # this is where logs for fly building will be put

In addition, there is one additional necessary directory for fly building:

basedir/fictrac: this contains pairs of .log/.dat files that contain readout of ball movements

fictrac
├── fictrac-20220329_141649.dat
└── fictrac-20220329_141649.log

Conversions are tracked within processed/conversion_db.csv - this is checked during building, and any imports already built will not be rebuilt unless the --overwrite flag is set.

### Fly builder

The fly_builder.py script copies the data from the imports directory into a new fly directory, creating a new directory structure within the processed data directory (which sits in the same base directory as the imports directory).  The name of the directory is generated automatically based on the existing fly directories, which have the naming scheme "fly_XXX" where XXX is a zero-padded three digit number.  A newly built fly is given the next available number.

TODO: devise a way to ensure a one-one mapping between import datasets and fly numbers.  currently running fly_builder.py twice on the same import dataset would generate two fly numbers.

The result from the fly_builder.py script will be as follows:


fly_009
├── anat_0
│   └── imaging
│       ├── anatomy.xml
│       ├── anatomy_channel_1.nii
│       └── anatomy_channel_2.nii
├── fly.json
├── func_0
│   ├── QC
│   ├── fictrac
│   │   ├── fictrac-20220329_173736.dat
│   │   ├── fictrac-20220329_173736.log
│   │   └── fictrac.xml
│   ├── imaging
│   │   ├── TSeries-12172018-1322-005_Cycle00001_VoltageRecording_001.xml
│   │   ├── functional.xml
│   │   ├── functional_channel_1.nii
│   │   ├── functional_channel_2.nii
│   │   ├── scan.json
│   │   ├── timestamps.h5
│   │   └── voltage_output.xml
│   ├── logs
│   ├── preproc
│   └── visual
│       ├── 2022-03-29.hdf5
│       ├── photodiode.csv
│       ├── photodiode.h5
│       └── stimulus_metadata.pkl
└── logs
    └── flybuilder_20220411-170431.txt

Logging for fly_builder.py is in fly_XXX/logs; all other logging happens within the specific imaging directory (e.g. fly_XXX/func_0/logs).  Additional directories are also created:

- preproc: location of preprocessed outputs from subsequent preprocessing steps
- QC: location of QC outputs from various steps

## Preprocessing steps


### Fictrac QC

outputs:

func_0/
├── logs
│   └── fictrac_qc_20220420-074747.log
└── QC
    ├── fictrac_2d_hist_fixed.png
    ├── fictrac_2d_hist.png
    └── fictrac_velocity_trace.png

### Stimulus-triggered average behavior


outputs: 

func_0/
├── logs
│   └── stim_triggered_avg_beh_20220420-074802.log
└── QC
    └── stim_triggered_turning.png


### Bleaching QC


### Temporal mean (pre preprocessing)


### Motion correction


### Z-scoring
- drop?


### highpass filtering


### correlation


### stimulus-triggered average neural response


### Convert h5 to nii


### Temporal mean (post preprocessing)



## Additional steps (not current included)


### Anatomical motion correction
- only to channel 1


### Co-registration of anatomical to functional


### Anatomical registration to atlas




