stuff='hello'
from distutils.core import setup
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = os.environ["cythonCC"]
ext_modules = [
    Extension(
        "epsilon_ball_emission_parallel",
        ["epsilon_ball_emission_parallel.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])]

setup(
    name='epsilon_ball_emission_parallel',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
