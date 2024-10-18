# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Samson Wang
# --------------------------------------------------------

from __future__ import print_function

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import setup

import numpy as np


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

#with open("README.md", "r") as fh:
#    long_description = fh.read()

ext_modules = [
    Extension(
        name='cython_gibbs',
        sources=['src/cython_gibbs.pyx'],
        extra_compile_args = {'gcc': ['/Qstd=c99']},
        include_dirs=[numpy_include],
        language='c++'
    )
]

setup(
    name='cython_gibbs',
    ext_modules=cythonize(ext_modules),
    version = '0.1.3',
    description = 'Standalone cython_gibbs',
    #long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Samson Wang',
    author_email = '',
    url = '',
    keywords = ['cython_gibbs']
)

