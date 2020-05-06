stuff='hello'
from distutils.core import setup
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os
#os.environ["CC"] = "gcc"
os.environ["CC"] = os.environ["cythonCC"]
ext_modules = [
    Extension(
        "gp_llhood_parallel",
        ["gp_llhood_parallel.pyx"],
#            extra_compile_args=["-O0", '-fopenmp'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])]
        
setup(
    name='gp_llhood_parallel',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
