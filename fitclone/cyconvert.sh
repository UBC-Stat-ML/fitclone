#!/bin/bash

rm *.c
rm -rf build 
export cythonCC="gcc"
python3 setup_cython/setup.py build_ext --inplace
python3 setup_cython/setup_gp.py build_ext --inplace
python3 setup_cython/setup_epsilon.py build_ext --inplace
python3 setup_cython/setup_post_epsilon.py build_ext --inplace
python3 setup_cython/setup_blocked_gibbs.py build_ext --inplace
python3 setup_cython/setup_gaussian_emission.py build_ext --inplace


