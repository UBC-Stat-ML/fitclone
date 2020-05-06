#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import yaml
import sys

#exec(open(os.path.expanduser('SC_DLP_sa501_exp.py')).read())
exec(open(os.path.expanduser('scalable_computing.py')).read())


# # Model comparison for DLP SA532, SA609, and SA501

# In[7]:


def generic_submit_batch(batch_name, params, days=7):
    
    if 'use_conditional' in params:
        use_conditional = params['use_conditional'][0]
    else:
        use_conditional = False
        
    if use_conditional is True:
        #sources = ['revive_conditional.py']
        sources = ['revive_conditional_method_comparison.py']
    else:
        sources = ['SC_DLP_sa501_exp.py']        

    print('use_conditional = {}'.format(use_conditional))
    print('sources = {}'.format(sources))
        
    func_name = 'run_model_comparison_ne'
    
    #params['the_num_par'] = [100000]
    #params['the_num_itr'] = [10000]
    #params['the_num_par'] = [1000]
    #params['the_num_itr'] = [500]
    #params['the_h_config'] = [os.path.join(os.path.dirname(params['the_original_data'][0]), 'h_config.tsv')]
    
    
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


# In[8]:


def batch_experiment_DLP_532_model_comparison():
    params = {}    
    
    # August 18th - August 22nd
    params['the_original_data'] = ['~/projects/fitclone/figures/raw/supp/SA532/steps_10/SA532_dropped_one_sc_dlp_4_clones_Ne']
    
    params['the_infer_epsilon'] = [.01, .05, 0.1]
    params['the_proposal_step_sigma'] = [.005, .01, .02, .05, .1]
    
    # Batch params
    batch_name = 'S532Aug'
    generic_submit_batch(batch_name, params)


# In[9]:


def batch_experiment_DLP_model_comparison(batch_config_path):
    stream = open(batch_config_path, "r")
    params = yaml.load(stream)
    
    # Batch params
    batch_name = os.path.splitext(os.path.basename(batch_config_path))[0]
    generic_submit_batch(batch_name, params)


# In[11]:


#batch_experiment_DLP_model_comparison(os.path.expanduser('~/projects/fitness/batchconfigs/dec29_SA609_pred_conditional.yaml'))


# In[ ]:


#batch_experiment_DLP_model_comparison(os.path.expanduser('~/projects/fitness/batchconfigs/{}'.format(sys.argv[1])))
batch_experiment_DLP_model_comparison('/scratch/shahlab_tmp/ssalehi/fitness/batchconfigs/{}'.format(sys.argv[1]))


# In[52]:


def write_yaml_example():
    params = {}    
    params['the_original_data'] = ['~/projects/fitclone/figures/raw/supp/SA609/steps_10/SA609_dropped_one_sc_dlp_4_clones_Ne']                    
    params['the_proposal_step_sigma'] = [.005, .01, .02, .05, .1]
    params['the_Ne'] = [i for i in [50, 100, 250, 500, 5000, 10000]]

    with open('data.yaml', 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)

