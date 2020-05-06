stuff='hello'
from distutils.core import setup
from distutils.extension import Extension
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

if env == 'local':
    os.environ["CC"] = "gcc-6"
    ext_modules = [
        Extension(
            "wf_sample_parallel",
            ["wf_sample_parallel.pyx"])]
if env == 'MOMAC39' or env == 'azure' or env == 'noah':
    os.environ["CC"] = "gcc-7"
    ext_modules = [
        Extension(
            "wf_sample_parallel",
            ["wf_sample_parallel.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'])]

elif env == 'grex' or env == 'bugaboo': 
    ext_modules = [Extension(
            "wf_sample_parallel",
            ["wf_sample_parallel.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'])]
    
elif env == 'shahlab' or env == 'AZURECN': 
    ext_modules = [Extension(
            "wf_sample_parallel",
            ["wf_sample_parallel.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'])]
else:
    ext_modules = [Extension(
            "wf_sample_parallel",
            ["wf_sample_parallel.pyx"])]    


setup(
    name='wf_sample_parallel',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)