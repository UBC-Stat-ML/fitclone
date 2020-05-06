# fitclone

Repository for methods used in paper:
`Single cell fitness landscapes induced by genetic and pharmacologic perturbations in cancer`.

For `sitka` the Bayesian phylogenetic inference method see 
[sitka](https://github.com/UBC-Stat-ML/nowellpack).

`fitClone` needs Python3 to be installed. We recommend using a conda installation.

```
brew install gcc --without-multilib
conda install -c anaconda hdf5
conda install -c conda-forge gcc
conda install -c anaconda cython

./cyconvert.sh 
python3 fitclone.py
```
 
