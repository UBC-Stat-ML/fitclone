# fitclone

Repository for methods used in paper:
`Single cell fitness landscapes induced by genetic and pharmacologic perturbations in cancer`.

For `sitka` the Bayesian phylogenetic inference method see 
[sitka](https://github.com/UBC-Stat-ML/nowellpack).

`fitClone` needs Python3 to be installed. We recommend using a conda installation.



```
brew install gcc --without-multilib
conda install -c anaconda cython

conda install -c anaconda hdf5
conda install -c conda-forge gcc

#export PATH=$PATH:/usr/local/Cellar/gcc@6/6.5.0_5/bin/
./cyconvert.sh 
python3.

python3 run_experiment.py  /path/to/data/SA501/sa501_config.yaml

python3 run_experiment.py /Users/sohrabsalehi/projects/fitness_material/fitness_code_repo/fitclone/data/SA501/sa501_config.yaml

```
 
