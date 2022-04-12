from setuptools import setup

setup(
    name='brainsss',
    version='0.0.1',
    long_description="""
    This package performs preprocessing and analysis of 
    volumetric neural data on sherlock. At its core, brainsss is a wrapper to 
    interface with Slurm via python. It can handle complex submission of batches of 
    jobs with job dependencies and makes it easy to pass variables between jobs. 
    It also has full logging of job progress, output, and errors.
    """,
    packages=['brainsss'],
    scripts=['scripts/preprocess.py',
        'scripts/fictrac_qc.py'],
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
        'GitPython'
    ],
)
