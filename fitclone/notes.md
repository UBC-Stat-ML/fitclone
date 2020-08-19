# Notes
This is a brief documentation of the class structure and the code-math connection. 

This will then be moved into docstrings inside classes.

## Tutorial
1. Run a caliberation round until reaching some end criteria

``` yaml
inference_n_iter: 500
pf_n_particles: 1000
pgas_n_particles: 1000
MCMC_in_Gibbs_nIter: 10  
pf_n_theta: 10000
```

2. End criteria
    - min ESS > threshold
    - RR about a good 
    - too many iters have passed

## Coding TODOs

1. Return more theta samples per trajectory 
2. Add a caliberation round
    * What are the params for that? 
    * it needs to be as fast as possible
    * disable prediction
    * disable plotting
    * add the utils inside of it
3. Add a quiet mode
4. ln -s results/latest to always point to the last experiment

## TODOs

1. Break down pgas_dir and pgas
2. Update names
3. Add a command line interface
4. Add input validity checks
5. Add accepting input from timeseries format
6. Add improvement lists
7. Convert to propor package structure

## Class structure

## Connection of classes to math