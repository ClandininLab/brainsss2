from setuptools import setup

setup(
    name='brainsss2',
    version='0.0.1',
    long_description="""
    This package is meant for preprocessing and analysis of 2-photon
    imaging data.  It is a refactor of https://github.com/ClandininLab/brainsss/
    """,
    packages=['brainsss2'],
    package_data={'': ['templates/index.html', 'settings/settings.json']},
    scripts=[
        'scripts/preprocess.py',
        'scripts/fly_builder.py',
        'scripts/fictrac_qc.py',
        'scripts/motion_correction.py',
        'scripts/bleaching_qc.py',
        'scripts/regression.py',
        'scripts/imgmath.py',
        'scripts/make_supervoxels.py',
        'scripts/atlas_registration.py',
        'scripts/check_status.py',
        'scripts/h5_to_nii.py',
        'scripts/dimensionality_reduction.py',
        'scripts/check_status.py',
        'scripts/generate_report.py',
        'scripts/stim_triggered_avg_beh.py',
        'scripts/stim_triggered_avg_neu.py'],
    include_package_data=True,
    install_requires=[
        'pyfiglet',
        'psutil',
        'lxml',
        'openpyxl',
        'nibabel',
        'seaborn',
        'numpy',
        'pandas',
        'pytest',
        'scikit-image',
        'scipy',
        'h5py',
        'matplotlib',
        'antspyx',
        'nilearn',
        'jinja2',
        'jsonschema',
        'opencv-python',
        'GitPython'
    ],
)
