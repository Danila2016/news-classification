import os
import numpy as np

os.environ['CPPFLAGS'] = os.getenv('CPPFLAGS', "") + " -I" + np.get_include()

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("intersection.pyx")
)
