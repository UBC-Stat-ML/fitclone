import os
import yaml
import sys

exec(open(os.path.expanduser('scalable_computing.py')).read())

def generic_submit_batch(batch_name, params, days=7):
    
    if 'use_conditional' in params:
        use_conditional = params['use_conditional'][0]
    else:
        use_conditional = False
        
    if use_conditional is True:
        sources = ['revive_conditional_method_comparison.py']
    else:
        sources = ['SC_DLP_sa501_exp.py']        

    print('use_conditional = {}'.format(use_conditional))
    print('sources = {}'.format(sources))
        
    func_name = 'run_model_comparison_ne'
    
    # Batch params
    mem_per_job=15
    n_jobs=None
    time_per_job=days*24*60 # Minutes
    ncores=1
    
    expt = BatchExp(params=params, dependencies=sources, func_name=func_name)
    
    n_params = expt.get_n_params()
    print('Cross n_params = {}'.format(n_params))
    if n_jobs is None:
        n_jobs = int(n_params)    
    
    batch_params = Experiment.get_batch_params_dict(batch_name=batch_name, mem_per_job=mem_per_job, n_jobs=n_jobs, time_per_job=time_per_job, ncpus_per_job=ncores)
    expt.submit_batch(batch_params=batch_params)    


def batch_experiment_DLP_model_comparison(batch_config_path):
    stream = open(batch_config_path, "r")
    params = yaml.load(stream)
    
    # Batch params
    batch_name = os.path.splitext(os.path.basename(batch_config_path))[0]
    generic_submit_batch(batch_name, params)


batch_experiment_DLP_model_comparison('/scratch/shahlab_tmp/ssalehi/fitness/batchconfigs/{}'.format(sys.argv[1]))
