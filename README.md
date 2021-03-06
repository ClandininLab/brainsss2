# brainsss2

This package performs preprocessing and analysis of volumetric 2-photon imaging data from Drosophila melanogaster. It is based on a previous package called [brainsss](https://github.com/ClandininLab/brainsss).  

The workflow includes the following steps:

- *fly building*: importing the necessary data (imaging, behavioral, and stimulus) to generate the file structure for a single acquisition
- *FicTrac QC*: Quality control on motion tracking data generated by [FicTrac](https://rjdmoore.net/fictrac/).
- *Stimulus-triggered behavior*: Plotting of stimulus-triggered behavior
- *Bleaching QC*: Plotting of signal intensity changes over time due to photobleaching
- *Motion correction*: Alignment of timeseries images (computed on "structural" TdTomato images and applied to "functional" GCaMP images)
- *Spatial smoothing*: Application of a small amount of spatial smoothing to reduce noise
- *Confound regression*: Modeling of head motion and temporal drift and generation of a denoised timeseries
- *Behavioral regression*: Modeling of neural response to behavioral features (motion on the ball).
- *Stimulus-triggered averaging*: Averaging of neural response timeseries in relation to stimulus presentation
- *Atlas registration*: Alignment of functional timeseries, high-resolution anatomical image, and atlas template.
- *Clustering*: Generation of clusters ("supervoxels") within each slice for dimensionality reduction

The workflow also generates an html report showing the main results.

This package is meant for use on high-performance computing systems that implement the [SLURM scheduling system](https://slurm.schedmd.com/documentation.html), but may not work on all such systems due to differences in SLURM configuration.  It uses a custom SLURM launcher defined in [slurm.py](brainsss2/slurm.py).  

## Installing:

To install the package, first clone the repository and then install using pip:

```shell
> git clone https://github.com/ClandininLab/brainsss2
> cd brainsss2
> pip install -U .
```

This will also install the necessary dependencies.

Package installation is best done within a virtual environment, e.g. using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).  

On [Sherlock](https://www.sherlock.stanford.edu/), it's first necessary to load the base Python module:

```shell
module load python/3.9.0
```

## File structure

The code assumes that the data have been organized according to the Clandinin lab file structure; we hope in the future to transition this to supporting the [BIDS Microscopy Standard](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/10-microscopy.html).  See more information on this file organization scheme [here](file_organization.md).

## Usage:

Examples of usage can be found in the [example Makefile](scripts/Makefile.dell).

The script should be launched by running the [preprocess.sh](scripts/preprocess.sh) script.  This is simply a wrapper for [preprocess.py](scripts/preprocess.py), which is necessary in order to properly configure the slurm job.  The shell script may require modifications for the local SLURM configuration, such as queue names.

### Full run (fly building + preprocessing)
# ashlu clarify what "fly building" means
# ashlu "base directory for flies" in confusing because "for" sounds like loop
In order to perform the entire building and preprocessing workflow, the command would be:

```shell
sbatch preprocess.sh -b <base directory for flies> --import_date <date of acquisition>  --fly_dirs <fly directory within import dir> 
 --build --run_all --atlasdir /data/brainsss/atlas
```

### Fly building only

To build a fly without preprocessing, simply specify the `--build_only` flag:
```shell
sbatch preprocess.sh -b <base directory for flies> --import_date <date of acquisition>  --fly_dirs <fly directory within import dir> 
 --build --build_only
```


### Preprocessing only (full pipeline)

In order to perform the entire preprocessing workflow on a previously built fly, the command would be:

```shell
sbatch preprocess.sh -b <base directory for flies> --process <built fly directory to process> --run_all  --atlasdir <atlas directory>
```

If the default atlas file (20220301_luke_2_jfrc_affine_zflip_2umiso.nii) is not present in the atlas directory, then one can specify an alternative atlas file and atlas name using the `--atlasfile` and `--atlasname` flags.

## Individual preprocessing steps

Each individual preprocessing step can also be launched using individual flags that are available from [preprocess.py](scripts/preprocess.py):

```shell
usage: preprocess.py [-h] [-v] [-l LOGFILE] [-t] [-o] -b BASEDIR [--import_date IMPORT_DATE] [--fly_dirs FLY_DIRS [FLY_DIRS ...]]
                     [--fictrac_import_dir FICTRAC_IMPORT_DIR] [--target_dir TARGET_DIR] [--import_dir IMPORT_DIR] [--no_visual] [--xlsx_file XLSX_FILE]
                     [--build | --process PROCESS] [-a] [--fictrac_qc] [--STB] [--bleaching_qc] [--motion_correction {func,anat,both}] [--smoothing] [--STA]
                     [--regression] [--supervoxels] [--atlasreg] [--atlasfile ATLASFILE] [--atlasname ATLASNAME] [--atlasdir ATLASDIR] [--build_only]
                     [--local] [--cores CORES] [--partition PARTITION] [-u USER] [--modules MODULES [MODULES ...]] [--continue_on_error]
                     [--func_dirs FUNC_DIRS [FUNC_DIRS ...]] [-s SETTINGS_FILE] [--ignore_settings]

preprocess

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         verbose output
  -l LOGFILE, --logfile LOGFILE
                        log file
  -t, --test            test mode
  -o, --overwrite       overwrite existing fly dir
  -b BASEDIR, --basedir BASEDIR
                        base directory for fly data
  --build               build_flies
  --process PROCESS     fly directory to process

fly_builder arguments:
  --import_date IMPORT_DATE
                        date of import (YYYYMMDD)
  --fly_dirs FLY_DIRS [FLY_DIRS ...]
                        specific fly dirs to process for import date
  --fictrac_import_dir FICTRAC_IMPORT_DIR
                        fictrac import directory (defaults to <basedir>/fictrac)
  --target_dir TARGET_DIR
                        directory for built flies (defaults to <basedir>/processed)
  --import_dir IMPORT_DIR
                        imaging import directory (defaults to <basedir>/imports)
  --no_visual           do not copy visual data
  --xlsx_file XLSX_FILE
                        xlsx file to use for fly data

# ashlu call this processing for consistency?
workflow components:
  -a, --run_all         run all preprocessing steps
  --fictrac_qc          run fictrac QC
  --STB                 run run stimulus-triggered behavioral averaging
  --bleaching_qc        run bleaching QC
  --motion_correction {func,anat,both}
                        run motion correction (func, anat, or both - defaults to func)
  --smoothing           run spatial smoothing
  --STA                 run stimulus-triggered neural averaging
  --regression          run regression
  --supervoxels         run supervoxels
  --atlasreg            run registration to/from atlas

atlas arguments:
  --atlasfile ATLASFILE
                        atlas file for atlasreg
  --atlasname ATLASNAME
                        identifier for atlas space for atlasreg
  --atlasdir ATLASDIR   directory containing atlas files for atlasreg

execution flags:
  --build_only          don't process after building
  --local               run locally (rather than using slurm)
  --cores CORES         number of cores to use
  --partition PARTITION
                        slurm partition to use for running jobs
  -u USER, --user USER  user name (default to system username)
  --modules MODULES [MODULES ...]
                        modules to load
  --continue_on_error   keep going if a step fails
  --func_dirs FUNC_DIRS [FUNC_DIRS ...]
                        specific func dirs to process
  -s SETTINGS_FILE, --settings_file SETTINGS_FILE
                        custom settings file (overriding user/package defaults)
  --ignore_settings     ignore settings file
```

## Workflow settings

The python package has a set of default settings for each step that are defined in [settings.json](brainsss2/settings/settings.json).  Note that these are only used when the workflow is run via `preprocess.py`; if the steps are run via their individual scripts then these values must be specified as command line arguments.  

The default settings can be overridden in two ways:

- *user settings*: the user can specify a settings file in `~/.brainsss/settings.json`; these will override any default settings.
- *workflow-specific settings*: the user can specify a settings file on the command line using the `-s` argument to the preprocessing script; any settings in this file will override both default settings and user settings.

## Rerunning existing workflows

In cases where a workflow crashes, it may be necessary to rerun portions of the workflow.  The default behavior is for each step to perform a <cursory and non-exhaustive> check for whether the step was previously performed, and if so then it simply returns and moves on to the next step.  This will only rerun any steps that have not been completed.

If one wishes to rerun all steps, then they can specify the `--overwrite` flag to the preprocessing script.

## Logging

Output from each step is logged in one of two locations depending on the processing step.

The output from the preprocess script and the fly_builder script are saved to `<basedir>/logs`; this is necessary because the script doesn't yet know the identity of the processed fly when it is executed.

The output from all other steps is put in a log file directory within the functional or anatomical subdirectory (e.g. `fly_001/func_0/logs`).  

In addition, a global log file is generated for the SLURM job output, which is generated within the directory where the job is started, named `sbatch_preproc_<slurm job ID>.log`.
