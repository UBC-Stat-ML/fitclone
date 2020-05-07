#!/bin/bash

# install python requirements
while read requirement; do conda install --yes $requirement; done < requirements.txt

# install cython and gcc
conda install -c anaconda hdf5
conda install -c conda-forge gcc
conda install -c anaconda cython

# compile the cythonized files
./cyconvert.sh 