# fitClone

Repository for methods used in paper:
`Single cell fitness landscapes induced by genetic and pharmacologic perturbations in cancer`.

BioRxive link here []

For `sitka` the Bayesian phylogenetic inference method see 
[sitka](https://github.com/UBC-Stat-ML/nowellpack).


`fitClone` is a Bayesian framework for inference in timeseries clonal abundance observations.
It is an implementation of the Wright-Fisher diffusion process. It simultaneously estimates growth trajectories and fitness coefficients for each clone in the population. The model accounts for drift as well as selection, with fitness estimated relative to a reference population. As a generative process, the model can be used for forecasting evolutionary trajectories of specific clones. 


## Installation

`fitClone` needs Python3 to be installed. We recommend using a conda installation.
It uses `Cython` and requires `openMP` to compile.

``` bash
# install requirements
conda install -c anaconda hdf5
conda install -c conda-forge gcc
conda install -c anaconda cython

# compile the cythonized files
./cyconvert.sh 

# run an example
chmox +x fitclone.py 
./fitclone.py
```
 
## Inputs
To run `fitClone` two prepare two inputs

1. A config file that sets the hyper-parameters and directory settings in [yaml](https://en.wikipedia.org/wiki/YAML) (e.g., [sa501_config.yaml](data/SA501/sa501_config.yaml))
2. A clonal fractions file that describes the fractions over time (e.g., [SA501_Ne_500.tsv](data/SA501/fractions/SA501_Ne_500.tsv))
The clone fractions file is a tab separated (tsv) file with three columns, `time`, `K`, and `X` as below:

``` table
time    K       X
0         0 0.135    
0.118     0 0.145    
0.295     0 0.425    
0.413     0 0.0745   
0         1 0.00409  
0.118     1 0.159    
0.295     1 0.0418   
0.413     1 0.142    
```

### The config file

This file should be written in yaml and can have the following values:

``` yaml

K: number of clonal that are updated simultaneously (recommended value 1)
K_prime: number number of clones sans reference
MCMC_in_Gibbs_nIter: number of iterations used in the MH step 
Ne: estimated effective population size
disable_ancestor_bridge: 
bridge_n_cores: number of cpu cores to use in bridge computation
do_predict: whether to perform prediction 
gp_epsilon: parameter for the Gaussian Process (GP) bridge (0.005)
gp_n_opt_restarts: number of attempts to fit the GP
h: discretisation constant
infer_epsilon: observation epsilon
inference_n_iter: number of MCMC iterations to use for inference
learn_time: if want to predict, set the learning horizon
lower_s: lower_bound for prior on s
n_cores: number of cpu cores to use 
obs_num: number of timepoints in the input
original_data: path/to/clonal_fractions.tsv
out_path: output directory
pf_n_particles: number of particles for the prediction particle filter
pf_n_theta: number of iterations for prediction
pgas_n_particles: number of particles for the inference particle filter
proposal_step_sigma: the sd of the proposal distribution used in the MH step
seed: seed to use for reproducibility 
upper_s: upper_bound for prior on s

```


### The clone fractions file


`time` denotes the diffusion time and should start from 0. `K` is the clone ID and should be a non-negative integer. 
If there are `M` clones, then K should take exactly `M-1`.
The missing clone is assumed to be the reference clone and will have a selective coefficient of 0. 
`X` records the abundance of the clone at that time.
At each time `t`, the sum of clonal fractions has to be between zero and one. 



[Example dataset underlying the analysis in Figure 1E is provided](data/SA501).

## Outputs
`fitClone` will write the following outputs in the `out_path` directory (set in the `config.yaml`):

``` yaml

config.yaml: an augmented copy of the config file used to run fitClone along with a few the runtimeinfer_theta.tsv.gz: the posterior of selection coefficientsinfer_x.tsv.gz: the posterior of imputed trajectoriesllhood_theta.tsv.gz: the likelihoodspredict.tsv.gz: the predicted trajectories if do_predict = 1
```
Moreover if `Rscript` is available, it will produce some summary plots in the `plots`.

