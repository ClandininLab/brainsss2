# check whether the necessary modules are installed
# any missing modules should be added to install_requires in setup.py
# this must be run from the root directory of the project or the tests directory

import sys
import os

def test_imports():
    import brainsss2.fictrac
    import brainsss2.utils
    import brainsss2.visual
