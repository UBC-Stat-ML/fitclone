
# coding: utf-8

# # Data generator
# ### Note
# `%run pgas-dir.ipynb` does not work on my machine. Perhaps too computationally demanding.
# Resolving to notebook to script conversion and using as input Python.
# 

# # Libraries

# In[4]:

import os
import itertools

# Dependencies
#path = os.path.expanduser('~/Google Drive/Masters/Thesis/scripts/fitness/pgas-dir.py')
#~/projects/fitness/
exec(open(os.path.expanduser('pgas-dir.py')).read())
exec(open(os.path.expanduser('scalable_computing.py')).read())


# In[7]:




# In[ ]:

class Test_import_integrity():
    def __init__(self):
        print('')
    
    def test():
        the_dir = os.path.expanduser("~/Google Drive/BCCRC/wright_fisher_experiments/simulated_data/sample_data.tsv")
        dat = TimeSeriesDataUtility.read_time_series(os.path.expanduser(the_dir))
        dat_profile = TimeSeriesDataUtility.parse_meta_data(the_dir)
        return(dat_profile)


# In[ ]:




# In[ ]:




# # Data generation
# Generate data for a variety of parameter settings. 
# The output should include a YAML header section, commented using "#", to detail data generation parameters including seed.
# A meaningful `filename` could signal seed and model (which K).
# Maintain a list of simulated data with parameter configuration and data file name to make running the methods over a specific simulation scenario easier.
# 
# ## Scenario 1
# Varying values of selection coeficients and 
# obs_num=5, silent=False, h = .001, selectionCoefficients=[.2, .3], end_time = .1, epsilon = .05, x0=[.5, .4], Ne = 200
# 
# ## Data generation implementation

# In[454]:




# In[ ]:




# In[446]:

class Data_Generator(Experiment):
    def __init__(self, batch_path, data_name):
        print('')
        self.db = None
        self.batch_path = batch_path
        self.db_file_path = os.path.join(self.batch_path, '{}.tsv'.format(data_name))
        
#     def dict_product(dicts):
#         return(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    # expand the columns to incorporate multi-dimensional variables
    def update_db(self, param, dat_file_path):
        temp_dic = param
        temp_dic['file_path'] = dat_file_path
        multi_dim_keys = ['s', 'true_x0']
        for i in range(len(multi_dim_keys)):
            for j in range(len(temp_dic[multi_dim_keys[i]])):
                temp_dic['{}{}'.format(multi_dim_keys[i], j)] = temp_dic[multi_dim_keys[i]][j]
                
        for i in range(len(multi_dim_keys)):
            #temp_dic.pop(multi_dim_keys[i], None)
            #if multi_dim_keys[i] in temp_dic: 
                #del temp_dic[multi_dim_keys[i]]
            temp_dic[multi_dim_keys[i]] = str(temp_dic[multi_dim_keys[i]])
                
        #temp_db = pn.DataFrame.from_dict(temp_dic)
        #print("temp_db['s'] is {}".format(temp_db['s']))
        
        if self.db is None:
            self.db = pn.DataFrame(temp_dic, index=[0])
        else: 
            #print(set(list(self.db)) == set(list(temp_db)))
            #self.db.loc[self.db.shape[0]] = temp_db
            self.db.loc[self.db.shape[0]] = temp_dic
            
                
    # params is a dictionary of prameters
    def generate_simulated_dataset(self, params):
        # Generate YAML profile string
        #print(params)
        # Generate YAML
        return('some_file.path')
    
    def run_with_config_file(self, config_file):
        # load the config_file
        params = ''
        return(self.run(params))
    
    
    def get_params(self):
        params = {}
        params['Ne'] = np.array([50, 200])
        params['seed'] = [i*10 for i in range(1,10)]
        params['h'] = [.001]
        params['epsilon'] = [.01]

        # Data
        #params['s'] = list(itertools.product(np.linspace(0, .5, 10), np.linspace(0, .5, 10)))
        #params['s'] = [list(i) for i in (itertools.product(np.linspace(0, .5, 10), np.linspace(0, .5, 10)))]
        params['s'] = [[j.tolist() for j in i] for i in (itertools.product(np.linspace(0, .5, 10), np.linspace(0, .5, 10)))]
        params['obs_num'] = [7]
        params['end_time'] = [.14]

        # Inference
        params['gp_epsilon'] = [.005]
        params['learn_time'] = [.9]
        params['true_x0'] = [[.4, .2]]
        params['gp_n_opt_restarts'] = [10]

        ## Sampler options
        params['pgas_n_particles'] = [1000]
        params['inference_n_iter'] = [10000]

        # Prediction
        params['pf_n_particles'] = [1000]
        ## Uses the last 10 theta for simulation
        params['pf_n_theta'] = [500]
        
        return(params)
        
   
    def get_dependencies(self):
        print('')

    def run(self, params):
        # Set parameter ranges
        #Ne = np.array([50, 100, 200, 500, 1000, 10000])
        
        
        expanded_params = list(Experiment.expand_cartesian_product(params))
        index = 0
        for param in expanded_params:
            # Generate & save the dataset
            dat_file_path = self.generate_simulated_dataset(param)
            
            # Update the db
            self.update_db(param, dat_file_path)
            
            index += 1
            if index > 10:
                print('Early breaking to test...')
                break
                
        TimeSeriesDataUtility.save_time_series(self.db, self.db_file_path)
                
                
        return(expanded_params)
            
    def test(self):
        s = Data_Generator()
        ss = s.run()
        len(ss)
        
        


# In[5]:

# jm = Job_manager(batch_name='test_data_generate', source_dir='~/Desktop/')
# s = Data_Generator(batch_path=jm.batch_path, data_name='FirstTestYAML')
# ss = s.run()
# jm.generate_yaml_chunks(nChunks= kk, params=ss)


# In[404]:




# In[ ]:




# # Parse input arguments

# In[184]:

# #!/usr/bin/python
# Useage: test.py -h
import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is "', inputfile)
   print('Output file is "', outputfile)

if __name__ == "__main__":
   #main(sys.argv[1:])
    print('')
  



# In[ ]:




# # The *work* block in the job submission.sh

# ```bash
# #!/bin/bash
# _base="/some/base/address"
# _dfiles="${base}/path/to/data/or/configfiles/*.yaml"
#  
# for f in $_dfiles
# do
#         test.py -i $f
#         lb2file="/tmp/${f##*/}.$$"   #tmp file
#         sed 's/Load_Balancer-1/Load_Balancer-2/' "$f" > "${lb2file}"   # update signature 
#         scp "${lb2file}" nginx@lb2.nixcraft.net.in:${f}   # scp updated file to lb2
#         rm -f "${lb2file}"
#         echo $f
# done
# ```

# ```bash
# #!/bin/bash
# _basedir="/shahlab/ssalehi/scratch"
# _files="$_basedir/improve/batch.runs/batchGDOct8/RFiles/*.rds"
# for f in $_files
# do
# 	echo $f
#     # my_script.py -i $f
# done
# ```

# In[ ]:

import os
import subprocess
import time

exec(open(os.path.expanduser('experiments-prediction.py')).read())

# Standard Prediction/Inference experiment
class DataGenerator(BayesianLearningExp):
    
        
    def generate_starting_value(self, k, count=1):
        res = [None]*count
        for i in range(count):
            res[i] = (np.random.dirichlet([1]*(k+1), 1)[0][0:k]).tolist()
        return(res)

    def get_params(self):
        params = {}
        params['Ne'] = [500]
        params['seed'] = [self.the_seed]
        params['simul_h'] = [.0001]
        params['epsilon'] = [.01]
        params['K'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        temp_k = params['K'][0]

        # Data
        params['s'] = [np.random.uniform(0, .5, temp_k).tolist()]
        params['obs_num'] = [5]
        params['end_time'] = [.05]

        # Parallelisation
        params['true_x0'] = self.generate_starting_value(temp_k, 1)
        params['out_path'] = ['~/Desktop/pgas_sanity_check/exp']
        return(params)

    def logic(self, resume):
        self._load_data(resume)
        get_ipython().magic('matplotlib inline')
        TimeSeriesDataUtility.plot_tall(self.full_dat, legend='', title='Real value')
        #savefig('foo.png', bbox_inches='tight')
        
import time
np.random.seed(50)
expt = OneStepExp(); expt.the_k = 4; expt.ncpus_per_job = 1; tt = expt._run_default()

