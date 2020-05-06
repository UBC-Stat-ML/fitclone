
# coding: utf-8

# In[6]:

import os
exec(open(os.path.expanduser('model_comparison_exp.py')).read())
exec(open(os.path.expanduser('hierarchical_full_experiment.py')).read())


# //# Model comparison for K = 4, versus strawmen
# # Model comparison for K = 10, versus strawmen

# In[4]:

def batch_experiment_1_model_comparison():
    sources = ['model_comparison_exp.py']
    func_name = 'run_model_comparison'
    params = {}
    params['some_seed'] = [i*10 for i in range(10, 50)]
    #params['some_seed'] = [i*10 for i in range(50, 90)]
    params['the_K'] = [10]
    
    batch_name = 't90MdlK10'
    mem_per_job=15
    n_jobs=None
    time_per_job=78*60
    ncores=1
    
    expt = BatchExp(params=params, dependencies=sources, func_name=func_name)
    
    n_params = expt.get_n_params()
    print(n_params)
    if n_jobs is None:
        n_jobs = int(n_params)    
    
    batch_params = Experiment.get_batch_params_dict(batch_name=batch_name, mem_per_job=mem_per_job, n_jobs=n_jobs, time_per_job=time_per_job, ncpus_per_job=ncores)
    expt.submit_batch(batch_params=batch_params)


# In[ ]:

def batch_experiment_2_hierarchical():
    sources = ['hierarchical_full_experiment.py']
    func_name = 'run_model_hie'
    params = {}
    params['some_seed'] = [i*10 for i in range(10, 50)]
    #params['some_seed'] = [i*10 for i in range(50, 100)]
    params['the_K'] = [10]
    
    batch_name = 'feb90hieK10'
    mem_per_job=15
    n_jobs=None
    time_per_job=78*60
    ncores=1
    
    expt = BatchExp(params=params, dependencies=sources, func_name=func_name)
    
    n_params = expt.get_n_params()
    print(n_params)
    if n_jobs is None:
        n_jobs = int(n_params)    
    
    batch_params = Experiment.get_batch_params_dict(batch_name=batch_name, mem_per_job=mem_per_job, n_jobs=n_jobs, time_per_job=time_per_job, ncpus_per_job=ncores)
    expt.submit_batch(batch_params=batch_params)


# In[5]:

batch_experiment_2_hierarchical()
#batch_experiment_1_model_comparison()


# 

# In[2]:

# Model comparison for K = 9, versus strawmen


# In[33]:

#doc = yaml.load_all(open('/Users/sohrabsalehi/projects/fitness/batch_runs/testMdlCmp_201802-01-163932.126875/yaml/param_chunk_0.yaml', 'r'))


# In[34]:

# for dd in doc:
#     print(dd['some_seed'])


# In[ ]:



