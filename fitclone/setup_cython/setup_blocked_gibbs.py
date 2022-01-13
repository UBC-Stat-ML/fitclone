stuff='hello'
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = os.environ["cythonCC"]

ext_modules = [Extension(
        "blocked_gibbs_sample",
        ["blocked_gibbs_sample.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])]

setup(
    name='blocked_gibbs_sample',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
