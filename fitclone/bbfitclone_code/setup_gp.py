stuff='hello'
from distutils.core import setup
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os
#os.environ["CC"] = "gcc-6"
#

# TODO: REMOVE THIS!!! TEMPORARY HACK platforms that don't have opemMPI
try:
    env = os.environ['HOST']
except:
    env = 'AZURECN'

if env == '' or env is None: env = 'local'

if env == 'local' or env == 'MOMAC39' or env == 'azure' or env == 'noah' or env == 'AZURECN':
    os.environ["CC"] = "gcc-7"
    ext_modules = [
        Extension(
            "gp_llhood_parallel",
            ["gp_llhood_parallel.pyx"],
#            extra_compile_args=["-O0", '-fopenmp'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'])]

elif env == 'grex' or env == 'bugaboo': 
    ext_modules = [Extension(
            "gp_llhood_parallel",
            ["gp_llhood_parallel.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'])]
    
elif env == 'shahlab': 
    ext_modules = [Extension(
            "gp_llhood_parallel",
            ["gp_llhood_parallel.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'])]
else:
    ext_modules = [Extension(
            "gp_llhood_parallel",
            ["gp_llhood_parallel.pyx"])]    


setup(
    name='gp_llhood_parallel',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
