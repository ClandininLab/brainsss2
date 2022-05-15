from setuptools import setup

setup(
    name='brainsss2',
    version='0.0.1',
    long_description="""
    This package is meant for preprocessing and analysis of 2-photon
    imaging data.  It is a refactor of https://github.com/ClandininLab/brainsss/
    """,
    packages=['brainsss2'],
   scripts=['scripts/preprocess.py',
       'scripts/fly_builder.py',
       'scripts/fictrac_qc.py',
       'scripts/motion_correction.py',
       'scripts/bleaching_qc.py',
       'scripts/regression.py',
       'scripts/imgmean.py',
       'scripts/imgstd.py',
       ],
    include_package_data=True,
    install_requires=[
        'pyfiglet',
        'psutil',
        'lxml',
        'openpyxl',
        'nibabel',
        'numpy',
        'pandas',
        'pytest',
        'scikit-image',
        'scipy',
        'h5py',
        'matplotlib',
        'antspyx',
        'nilearn',
        'GitPython'
    ],
)
