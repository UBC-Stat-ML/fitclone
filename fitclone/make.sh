#!/bin/bash

# install python requirements
while read requirement; do conda install --yes $requirement; done < requirements.txt

# install cython and gcc
conda install -c anaconda hdf5
conda install -c conda-forge/label/cf202003 gcc
conda install -c anaconda cython
conda install -c intel scikit-learn

# compile the cythonized files
./cyconvert.sh 