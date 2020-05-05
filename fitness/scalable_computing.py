# # A number of classes to enable easy running of experiments on clusters/cloud?

# ## Concepts and core classes
# An experiment would be implemented as follows:
# 1. A class `My_Experiment` inherits from `Experiment` and implements custom logic in the `run(param_set)` method for one set of parameter configuration. 
# 2. A list of set of parameters is defined, either via a `batch_config_file` or by overriding the `_get_params(self)` method.
# 3. `My_Experiment().submit_batch(batch_params)` or `My_Experiment().submit_batch(batch_config_file, batch_params)` is invoked.
# 
# ## Implementation
# 1. The `Experiment` class first `_expands()` the parameters.
# 2. Generates **chunked** config files (in yaml) to support multi-experiments per each job.
# 3. Generates the corresponding Python files (each .py file corresponds to a .yaml file, mirroring each others `Index`).
# 4. Generates the corresponding .sh file, with the cluster specific parameters (e.g., nJobs, timePerJob, memPerJob, etc.).
# 5. Generates a submit.sh script that gathers the correct submit commands in one go.
# 6. Finally echos the line to run the submit.sh file (including chmod+x file_name).
# 
# 
# ### Python files
# They consist in an import block, a for block that proceeds the correct index, and finally a object().run(param_set) command.
# 
# Given a list of param lists, divide them into `nFiles` yaml files, each having the appropriate number of yaml docs.
# These are imported from `queue_run.R` developed by Sohrab Salehi.
# 


import datetime
import yaml
import math
import pickle
import os
import itertools
import numpy as np
import time
import string
import random
import profile
import pstats
import sys

try:
    env = os.environ['HOST']
except:
    env = 'AZURECN'
fitness_dir = ''
if env == '' or env is None: env = 'local'
if env == 'local': fitness_dir = '/Users/sohrab/Google Drive/Masters/Thesis/scripts/fitness'
elif env == 'beast': fitness_dir = '/home/ssalehi/projects/fitness'
elif env == 'rocks3': fitness_dir = '/home/ssalehi/projects/fitness'
elif env == 'grex': fitness_dir = '/home/sohrab/projects/fitness'
elif env == 'bugaboo': fitness_dir = '/home/sohrab/projects/fitness'
elif env == 'shahlab': fitness_dir = '/scratch/shahlab_tmp/ssalehi/fitness'
elif env == 'MOMAC39': fitness_dir = '/Users/ssalehi/projects/fitness'
elif env == 'azure': fitness_dir = '/home/ssalehi/projects/fitness'
elif env == 'noah': fitness_dir = '/Users/sohrabsalehi/projects/fitness'
else:
    fitness_dir = '.'
    print('On unrecognised environment. Doing nothing.')

os.chdir(fitness_dir)
sys.path.insert(0, fitness_dir)

exec(open(os.path.expanduser('Utilities.py')).read())


class Configurable(object):
    """
    Set your variables from a config file. 
    """
    def __init__(self, conf_file):
        self._read_config_file(conf_file)
        
    def _read_config_file(self, conf_file):
        stream = open(conf_file, "r")
        doc = yaml.load(stream)
        for k,v in doc.items():
            setattr(self, k, v)
    
    def _set_config_from_dict(self, config_dict):
        for k,v in config_dict.items():
            setattr(self, k, v)
        self.configs = config_dict
    
    def get_test_path():
         return('/Users/ssalehi/projects/fitness/scripts/parameters/config.yaml') 
    
    def test():
        bb = Configurable(Configurable.get_test_path())
        print(bb.N0)


# In[5]:


class Env_setup(): 
    def get_host():
        return(os.environ['HOST'])
    
    def get_batch_dir():
        env = os.environ['HOST']
        if env == '' or env is None: env = 'local'
        if env == 'local': return('/Users/sohrab/Google Drive/Masters/Thesis/scripts/fitness/batch_runs')
        elif env == 'noah': return('/Users/sohrabsalehi/projects/fitness/batch_runs')
        elif env == 'AZURECN': return('.')
        elif env == 'grex': return('/global/scratch/sohrab/fitness/batch_runs')
        elif env == 'bugaboo': return('/global/scratch/sohrab/fitness/batch_runs')
        elif env == 'shahlab': return('/scratch/shahlab_tmp/ssalehi/fitness/batch_runs')
        elif env == 'MOMAC39': return('/Users/ssalehi/projects/fitness/batch_runs')
        else:
            print('On unrecognised environment. Setting to current dir.')
            return('.')
    
    def get_simulated_data_path():
        return(get_path(some_path="wright_fisher_experiments/simulated_data"))
    
    def get_path(some_path):
        main_path = os.path.expanduser("~/Google\ Drive/BCCRC/")
        return(os.path.join(main_path, some_path))


class Param_handler(object):
    def __init__(self, batch_name, batch_path):
        self.batch_name = batch_name
        self.batch_path = batch_path
        self.yaml_path = os.path.join(self.batch_path, 'yaml')
        self.chunks_ranges = None
            
    def simple_chunk(aSet, aSize):
        if aSize == 1:
            return([range(i,i+1) for i in aSet])
        theMin = min(aSet)
        starts = np.array(list(itertools.compress(aSet, [((i-theMin) % aSize == 0) for i in aSet])))
        ends = starts + aSize
        res = [i for i in (map(range, starts, ends))]
        return(res)

    # N: numberOfParameters; C: numberOfJobs
    def make_chunks(N, C):
        # Create the most uniform load balance
        # N = (C-k)*m + k(m-1): C-k groups of size m and k groups of size m-1 -- 
        # N = C*m - k 
        if C > N: 
            C = N
        m = math.floor(N/C) + 1
        k = C - (N % C)
        chunks = []
        if C != k:
            chunks = Param_handler.simple_chunk(list(range(0,(m*(C-k)))), m)
        chunks.extend(Param_handler.simple_chunk(list(range(m*(C-k),N)), m-1))
        return(chunks)
        
    def generate_yaml_chunks(self, params, nChunks):
        nParams = len(params)
        self.chunks_ranges = Param_handler.make_chunks(nParams, nChunks)

        index = 0
        file_paths = []
        
        if not os.path.exists(self.yaml_path):
            os.makedirs(self.yaml_path)
            
        for chunk_range in self.chunks_ranges:
            params_chunk = [params[i] for i in chunk_range]            
            final_path = os.path.join(self.yaml_path, 'param_chunk_{}.yaml'.format(index))
            file_paths.append(final_path)
            
            # add out_path to the dictionary
            secondary_index = 0
            for param_dict in params_chunk:
                param_dict['out_path'] = os.path.join(self.batch_path, 'outputs/o_{}_{}'.format(index, secondary_index))
                param_dict['yaml_index'] = index
                param_dict['yaml_secondary_index'] = secondary_index
                # Todo: this is kinda unncessary, eh?
                param_dict['config_chunk_path'] = final_path
                secondary_index += 1

            yaml.dump_all(params_chunk, open(final_path , 'w')) # default_flow_style=False
            index += 1
        return(file_paths)


class Code_generator(object):
    def __init__(self, owner):
        self.owner = owner
    
    # param_chunk_yaml
    def generate_code(self, chunked_config_file_paths):
        file_paths = []
        for param_chunk_yaml_path in chunked_config_file_paths:
            lines = ['# This is an auto-generated Python file for batch {} for bundle {}'.format(
                self.owner.batch_params['batch_name'], os.path.basename(param_chunk_yaml_path))]

            # The one essential library, obviously ;-)
            lines.append("import os")
            
            # The import bloc
            for i in self.owner.get_dependencies():
                lines.append( "exec(open(os.path.expanduser('{}')).read())".format(i))

            # The for bloc
            lines.append("docs = yaml.load_all(open('{}', 'r'))".format(param_chunk_yaml_path))
            lines.append("for doc in docs:")
            lines.append("\tres = {}().run(doc)".format(self.owner.get_class_name()))
            lines.append("\t")
            lines.append("\tif res is not None:")
            lines.append("\t\tpickle.dump(res, open('{}.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)".format(os.path.splitext(param_chunk_yaml_path)[0]))
            
            file_path = os.path.join(self.owner.python_path, '{}.py'.format(os.path.basename(os.path.splitext(param_chunk_yaml_path)[0])))
            file_paths.append(file_path)
            with open(file_path, 'w') as f:
                f.writelines("{}\n".format(l) for l in lines)
        return(file_paths)
    
    # Function based version
    def generate_function_code(self, func_name, chunked_config_file_paths):
        file_paths = []
        for param_chunk_yaml_path in chunked_config_file_paths:
            lines = ['# This is an auto-generated Python file for batch {} for bundle {}'.format(
                self.owner.batch_params['batch_name'], os.path.basename(param_chunk_yaml_path))]

            # The one essential library, obviously ;-)
            lines.append("import os")
            
            # The import bloc
            for i in self.owner.get_dependencies():
                lines.append( "exec(open(os.path.expanduser('{}')).read())".format(i))

            # The for bloc
            lines.append("docs = yaml.load_all(open('{}', 'r'))".format(param_chunk_yaml_path))
            lines.append("for doc in docs:")
            lines.append("\tres = {}(doc, '{}')".format(func_name, self.owner.batch_path))
            lines.append("\t")
            
            file_path = os.path.join(self.owner.python_path, '{}.py'.format(os.path.basename(os.path.splitext(param_chunk_yaml_path)[0])))
            file_paths.append(file_path)
            with open(file_path, 'w') as f:
                f.writelines("{}\n".format(l) for l in lines)
        return(file_paths)


class Job_script_generator(object):
    def __init__(self, owner):
        self.owner = owner
        
    # 200 -> 3:20:00
    def time_string(minutes):
        seconds = minutes % 1
        minutes = math.floor(minutes)
        return('{}:{:02d}:{:02d}'.format(minutes // 60, minutes % 60, math.ceil(seconds*60)))
    
    def generate_batch_code_shahlab(self, python_file_paths):
        index = 0
        file_paths = []
        print('Restricting jobs to node0502|node0504|node0506|node0507|node0509|node0510')
        for python_file_path in python_file_paths:
            target_env = 'SGE'
            lines = ['#$ -S /bin/sh']
            lines.append('#$ -r n ')
            lines.append('#$ -N j{}_{} '.format(index, self.owner.batch_params['batch_name']))
            lines.append('#$ -l h_rt={}'.format(Job_script_generator.time_string(self.owner.batch_params['time_per_job']*self.owner.get_n_jobs_per_node() + 1)))
            lines.append('#$ -l s_rt={}'.format(Job_script_generator.time_string(self.owner.batch_params['time_per_job']*self.owner.get_n_jobs_per_node())))
            lines.append('#$ -l h_vmem={}G'.format(self.owner.batch_params['mem_per_job']))
            #lines.append('#$ -l ncpus={}'.format(self.owner.batch_params['ncpus_per_job']))
            lines.append("#$ -l hostname='node0502|node0504|node0506|node0507|node0509|node0510'")
            lines.append('#$ -e {}'.format(os.path.join(self.owner.qsub_path, '{}_error.txt'.format(index))))
            lines.append('#$ -o {}'.format(os.path.join(self.owner.qsub_path, '{}_output.txt'.format(index))))
            #lines.append('#$ -M sohrab.salehi@gmail.com')
            #lines.append('#$ -m bes')
            lines.append('cd $SGE_O_WORKDIR')
            lines.append("#$ -v HOST='shahlab'")
            lines.append('/gsc/software/linux-x86_64-centos6/python-3.5.2/bin/python3 {}'.format(python_file_path))
            index += 1
            
            file_path = os.path.join(self.owner.qsub_path, '{}.sh'.format(os.path.basename(os.path.splitext(python_file_path)[0])))
            file_paths.append(file_path)
            with open(file_path, 'w') as f:
                f.writelines("{}\n".format(l) for l in lines)
        return(file_paths)

    def generate_batch_code_grex(self, chunked_config_file_paths):
        index = 0
        file_paths = []
        for python_file_path in chunked_config_file_paths:
            target_env = 'PBS'
            lines = ['#!/bin/bash']
            lines.append('#PBS -r n ')
            #lines.append('#PBS -m bea ')
            #lines.append('#PBS -M sohrab.salehi@gmail.com')
            lines.append('#PBS -l walltime={}'.format(Job_script_generator.time_string(self.owner.batch_params['time_per_job']*self.owner.get_n_jobs_per_node())))
            lines.append('#PBS -l procs={}'.format(self.owner.batch_params['ncpus_per_job']))
            lines.append('#PBS -l mem={}gb'.format(self.owner.batch_params['mem_per_job']))
            lines.append('#PBS -e {}'.format(os.path.join(self.owner.qsub_path, '{}_error.txt'.format(index))))
            lines.append('#PBS -o {}'.format(os.path.join(self.owner.qsub_path, '{}_output.txt'.format(index))))
            lines.append('cd $PBS_O_WORKDIR ')
            lines.append('module load python/3.5.1-intel')
            lines.append('python {}'.format(python_file_path))
            index += 1
            
            # For the local host
            print('python {}'.format(python_file_path))
            
            file_path = os.path.join(self.owner.qsub_path, '{}.sh'.format(os.path.basename(os.path.splitext(python_file_path)[0])))
            file_paths.append(file_path)
            with open(file_path, 'w') as f:
                f.writelines("{}\n".format(l) for l in lines)
        return(file_paths)

    def generate_batch_code(self, chunked_config_file_paths):
        if Env_setup.get_host() == 'shahlab': 
            return(self.generate_batch_code_shahlab(chunked_config_file_paths))
        else:
            return(self.generate_batch_code_grex(chunked_config_file_paths))
        
    # Genesis has no R, so make a bash script with the submission commands    
    def generate_submission_command(self, batch_file_paths):
        if Env_setup.get_host() == 'shahlab': 
            submit_path = os.path.join(self.owner.qsub_path, 'submit.sh')
            lines = ['#!/bin/bash']
            for file in batch_file_paths:
                lines.append('qsub -hard -q shahlab.q -P shahlab_high -pe ncpus {} -l shah_io=1 {}'.format(self.owner.batch_params['ncpus_per_job'], file))
                #lines.append('qsub -hard -q shahlab.q -l shah_io=1 {}'.format(file))

            with open(submit_path, 'w') as f:
                f.writelines("{}\n".format(l) for l in lines)

            print('Done creating Scripts!')
            print('RUN THE BASHSCRIPT YOURSELF!!!')
            print('chmod +x {}'.format(submit_path))
            print('{}'.format(submit_path))
        else:
            submit_path = os.path.join(self.owner.qsub_path, 'submit.sh')
            lines = ['#!/bin/bash']
            for file in batch_file_paths:
                lines.append('qsub {}'.format(file))

            with open(submit_path, 'w') as f:
                f.writelines("{}\n".format(l) for l in lines)

            print('Done creating Scripts!')
            print('RUN THE BASHSCRIPT YOURSELF!!!')
            print('chmod +x {}'.format(submit_path))
            print('{}'.format(submit_path))


class Sample_processor():
    '''
    Hack: Only the instance with m_id == 0 writes the llhood and theta to disk
    '''
    def __init__(self, time_points, owner, bias=0, m_id=0):
        self.owner = owner
        self.time_points = time_points 
        self.sample_buffer = []
        self.bias = bias
        self.m_id = m_id
        
        
    def handle_samples(self, tag, iteration, nIter, start, end):
        start = self.bias + start
        if tag == 'sample':
            res_theta = [self.sample_buffer[mm][0] for mm in range(len(self.sample_buffer))]
            res_x = [self.sample_buffer[mm][1] for mm in range(len(self.sample_buffer))]
            res_llhood = [self.sample_buffer[mm][2] for mm in range(len(self.sample_buffer))]
            # X
            fine_grained_times = np.linspace(0, self.time_points[-1], num=len(res_x[0]))
            df = TimeSeriesDataUtility.list_TK_to_tall(dat_list=res_x, times=fine_grained_times, base_index=start)
            if hasattr(self.owner, 'M_data'):
                TimeSeriesDataUtility.save_time_series(df, self.owner.inference_x_file_path[self.m_id], use_header=(start==0))
            else:
                TimeSeriesDataUtility.save_time_series(df, self.owner.inference_x_file_path, use_header=(start==0))
            # theta
            if self.m_id == 0:
                df = pn.DataFrame(data=np.array(res_theta), dtype=res_theta[0].dtype)
                TimeSeriesDataUtility.save_time_series(df, self.owner.inference_theta_file_path, use_index=False, use_header=(start==0))
            # llhood
                df = pn.DataFrame(data=np.array(res_llhood), dtype=res_llhood[0].dtype)
                TimeSeriesDataUtility.save_time_series(df, self.owner.inference_llhood_file_path, use_index=False, use_header=(start==0))
        elif tag == 'predict':
            x_predict = [self.sample_buffer[mm] for mm in range(len(self.sample_buffer))]
            fine_grained_times = np.linspace(0, self.time_points[-1], num=len(x_predict[0]))
            df = TimeSeriesDataUtility.list_TK_to_tall(dat_list=x_predict, times=fine_grained_times,  base_index=start)
            TimeSeriesDataUtility.save_time_series(df, self.owner.prediction_file_path, use_header=(start==0))
        else:
            None

        self.sample_buffer = []
    
    # Periodically write the results to disk
    def sample_processor(self, rvs, tag, iteration, nIter):
        #self.sample_buffer.append([rvs[i].copy() for i in range(len(rvs))])
        if isinstance(rvs, list):
            self.sample_buffer.append([rvs[i].copy() for i in range(len(rvs))])
        else:
            self.sample_buffer.append(rvs.copy())
            
        if iteration == 0 and nIter != 1:
            return(None)
        if iteration == 0 and nIter == 1:
            return(self.handle_samples(tag, iteration, nIter, start=0, end=1))
        
        r = iteration % self.owner.processing_interval
        if r == 0:
            zero_bias = 1 if iteration-self.owner.processing_interval == 0 else 0
            return(self.handle_samples(tag, iteration, nIter, start=iteration-self.owner.processing_interval+1-zero_bias, end=iteration+1))
        elif nIter <= self.owner.processing_interval and iteration == (nIter-1):
            return(self.handle_samples(tag, iteration, nIter, start=0, end=nIter))
        elif iteration == (nIter-1) and r != 0:
            return(self.handle_samples(tag, iteration, nIter, start=nIter-r, end=nIter))


class Particle_processor():
    def __init__(self, time_points, owner, m_id=0, dims=['n', 'time', 'K']):
        self.time_points = time_points
        self.owner = owner
        self.m_id = m_id
        self.dims = dims
        self.iter_index = 0
        self.bridge_iter_index = 0
    
    def process_bridge_counts(self, bridge_counts):
        print('Processing bridge', bridge_counts)
        arr = np.array(bridge_counts)
        x_ar = xr.DataArray(arr, dims=['time'], name='X')
        df = x_ar.to_dataframe()
        df['np'] = self.bridge_iter_index
        if hasattr(self.owner, 'M_data'):
            TimeSeriesDataUtility.save_time_series(df, self.owner.inference_bridge_file_path[self.m_id], use_header=(self.bridge_iter_index==0))
        else:
            TimeSeriesDataUtility.save_time_series(df, self.owner.inference_bridge_file_path, use_header=(self.bridge_iter_index==0))
        self.bridge_iter_index = self.bridge_iter_index + 1
    
    # N x T x K
    def process_particles(self, particle_time):
        N, T, K = particle_time.shape
        fine_grained_times = np.linspace(self.time_points[0], self.time_points[2], num=T)
        x_ar = xr.DataArray(particle_time, dims=self.dims, name='X', coords={'time':fine_grained_times})
        df = x_ar.to_dataframe()

        df['np'] = self.iter_index
        if hasattr(self.owner, 'M_data'):
            TimeSeriesDataUtility.save_time_series(df, self.owner.inference_particle_file_path[self.m_id], use_header=(self.iter_index==0))
        else:
            TimeSeriesDataUtility.save_time_series(df, self.owner.inference_particle_file_path, use_header=(self.iter_index==0))
        self.iter_index = self.iter_index + 1


class Experiment(Configurable):
    """
    Accept an object, that has a run method. with an optional config file. 
    Set all the cluster specific job configurations.
    """
    def __init__(self, exp_description=''):
        self.batch_config_file = None
        self.configs = None
        self.processing_interval = 10
        self.sample_buffer = None
        self.exp_descunription = exp_description
        
    
    def get_class_name(self):
        return(self.__class__.__name__)
    
    
    # Abstract methods
    def get_dependencies(self):
        """
        Should be the full path of the dependency
        """
        raise NotImplementedError('')
    
    def logic(self, params_dict):
        raise NotImplementedError('params_dict = {}'.format(params_dict))
    
    def get_params(self):
        raise NotImplementedError('')

    def parse_param_descriptor(param_descriptor_file):
        # Todo: implement reading range of params from a file
        raise NotImplementedError('To be implemented.')
    
    
    # Optional methods
    def _post_process(self):
        # Runs right after the logic() is finisehd and the re-saving of the config file
        # Use to wrap up the experiment, e.g., make plots, compute evaluation scores, etc...
        return('')
    
    # Job submission logic
    def _create_chunked_config_files(self):
        expanded_params = list(Experiment.expand_cartesian_product(self.get_param_descriptor()))
        self.chunked_config_file_paths = self._param_handler.generate_yaml_chunks(nChunks= self.batch_params['n_jobs'], params=expanded_params)
        
    def _generate_python_driver_files(self):
        self.python_file_paths = self._code_generator.generate_code(self.chunked_config_file_paths)
     
    def _generate_bash_files(self):
        self.batch_file_paths = self._jsg.generate_batch_code(self.python_file_paths)
    
    def _generate_master_submit_bash_file(self):
        self._jsg.generate_submission_command(self.batch_file_paths)
    
    
    # Utilities
    def get_n_jobs_per_node(self):
        chunks_ranges = self._param_handler.chunks_ranges
        last_length = ([len(i) for i in chunks_ranges])[len(chunks_ranges)-1] 
        last_length = 1 if last_length == 0 else last_length
        return(last_length)
    
    def expand_cartesian_product(param_dict):
        return(dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))
    
    def get_n_params(self):
        return(Experiment.number_of_params(self.get_param_descriptor()))
        
    def print_parameters(self):
        temp = list(Experiment.expand_cartesian_product(self.get_param_descriptor()))
        print(temp)
        return(temp)

    def number_of_params(param_dict):
        return(len(list(Experiment.expand_cartesian_product(param_dict))))
    
    def get_batch_params_dict(n_jobs, time_per_job, mem_per_job, batch_name, ncpus_per_job):
        """
        time_per_job in minutes
        mem_per_job in GBs
        """
        return({'n_jobs':n_jobs, 'time_per_job':time_per_job, 'mem_per_job':mem_per_job, 'batch_name':batch_name, 'ncpus_per_job':ncpus_per_job})      
        
    def get_time_stamp():
        return(datetime.datetime.now().strftime('%Y%m-%d-%H%-M%S.%f'))
    
    def get_random_str(N):
        return(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N)))
    
    # dictionary of range of parameters
    def get_param_descriptor(self):
        if self.batch_config_file is None:
            # safety check, every key has to be enclosed in a list, otherwise unexpected behaviour
            params_desc_dic = self.get_params()
            for key, value in params_desc_dic.items():
                if not isinstance(value, list):
                    raise ValueError('params_desc_dic[{}]={} is not a list. This will result in unexpected behaviour.'.format(key, value))
            return(params_desc_dic)
        else:
            return(parse_param_descriptor(self.batch_config_file))
        
        
    def get_file_name(generic_name, suffix=''):
        """
        Adds a time stamp to the name
        """
        return('{}_{}_{}{}'.format(generic_name, Experiment.get_random_str(5), Experiment.get_time_stamp(), suffix))
    
    def _setup_dirs(self, is_resume=False):
        if is_resume is False:
            self.out_path = Experiment.get_file_name(os.path.expanduser(self.out_path)) if is_resume == False else os.path.expanduser(self.out_path)
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
        print(self.out_path)
        # Save configs 
        self.configs_path = os.path.join(self.out_path, 'config.yaml')
        
        # Inference
        if hasattr(self, 'M_data'):
            self.inference_x_file_path = [None]*self.M_data
            self.inference_particle_file_path = [None]*self.M_data
            self.inference_bridge_file_path = [None]*self.M_data
            self.inference_xprime_file_path = [None]*self.M_data
            for m in range(self.M_data):
                self.inference_x_file_path[m] = os.path.join(self.out_path, 'infer_x_{}.tsv'.format(m))
                self.inference_particle_file_path[m] = os.path.join(self.out_path, 'infer_particle_{}.tsv'.format(m))
                self.inference_bridge_file_path[m] = os.path.join(self.out_path, 'infer_bridge_{}.tsv'.format(m))
                self.inference_xprime_file_path[m] = os.path.join(self.out_path, 'infer_xprime_{}.tsv'.format(m))
        else:
            self.inference_x_file_path = os.path.join(self.out_path, 'infer_x.tsv')
            self.inference_particle_file_path = os.path.join(self.out_path, 'infer_particle.tsv')
            self.inference_bridge_file_path = os.path.join(self.out_path, 'infer_bridge.tsv')
            self.inference_xprime_file_path = os.path.join(self.out_path, 'infer_xprime.tsv')
            
        self.inference_theta_file_path = os.path.join(self.out_path, 'infer_theta.tsv')
        self.inference_llhood_file_path = os.path.join(self.out_path, 'llhood_theta.tsv')
        # Prediction
        self.prediction_file_path = os.path.join(self.out_path, 'predict.tsv')

        # Evaluation (TODO)
            
    def _initialise(self, batch_params):
        self.batch_params = batch_params
        self._time_stamp = Experiment.get_time_stamp()
        self.batch_path = os.path.expanduser(os.path.join(Env_setup.get_batch_dir(), '{}_{}'.format(self.batch_params['batch_name'], Experiment.get_time_stamp())))
        if not os.path.exists(self.batch_path):
            os.makedirs(self.batch_path)
        self.qsub_path = os.path.join(self.batch_path, 'qsubFiles')
        self.python_path = os.path.join(self.batch_path, 'pythonFiles')
        if not os.path.exists(self.qsub_path):
            os.makedirs(self.qsub_path)
            
        if not os.path.exists(self.python_path):
            os.makedirs(self.python_path)
        
        self._param_handler = Param_handler(batch_name=self.batch_params['batch_name'], batch_path=self.batch_path)
        self._code_generator = Code_generator(self)
        self._jsg = Job_script_generator(self)
        
  
    def _save_config(self):
        try:
            if self.exp_description is not "":
                self.configs['exp_description'] = self.exp_description
        except Exception:
            yaml.dump(self.configs, open(self.configs_path, 'w'))
        return(self.configs_path)

    
    def _profile_run_func(self):
        import profile
        import pstats
        file_name = '{}_profile_stats'.format(self.get_class_name())
        res = profile.run('{}()._run_default()'.format(self.get_class_name()), file_name)
        p = pstats.Stats(file_name)
        p.strip_dirs().sort_stats(-1).print_stats()
        print(p.sort_stats('cumulative').print_stats(10))
        print(p.sort_stats('time').print_stats(10))

    
    def timer_dec(f):
        def decorator(*args, **kwargs):
            ref = None
            for arg in args:
                if isinstance(arg, Experiment):
                    ref = arg
                    break
            startTime = int(round(time.time()*1000))
            res = f(*args, **kwargs)
            endTime = int(round(time.time()*1000))  
            # run_time in ms
            ref.run_time = endTime-startTime
            return(res)
        return(decorator)

    # Only for test purposes
    def _run_default(self):
        dic = self.get_params()
        for k,v in dic.items():
            if isinstance(v, list):
                dic[k] = v[0]
        print(dic)
        return(self.run(dic))

    
    # Support standard behaviour, such as timing, saving of config files (input), etc... 
    #@timer_dec
    def run(self, params_dict, resume=False):
        self._set_config_from_dict(params_dict)
        self._setup_dirs(resume)
        if resume is False:
            self.configs['out_path'] = self.out_path
        
        self._save_config()
        startTime = int(round(time.time()))
        res = self.logic(resume)
        endTime = int(round(time.time()))  
        self.configs['run_time'] = endTime-startTime
        self.configs['run_time_str'] = time_string_from_seconds(endTime-startTime)
        self._save_config()
        self._post_process()
        return(res)
            

    def resume_exp(self, exp_path):
        self.configs_path = os.path.join(os.path.expanduser(exp_path), 'config.yaml')
        print(self.configs_path)
        stream = open(self.configs_path, "r")
        doc = yaml.load(stream)
        return(self.run(params_dict=doc, resume=True))
    
    def submit_batch(self, batch_params, batch_config_file=None):
        self._initialise(batch_params)
        
        self.batch_config_file = batch_config_file
        
        self._create_chunked_config_files()
        self._generate_python_driver_files()
        self._generate_bash_files()
        self._generate_master_submit_bash_file()
        


class BatchExp(Experiment):

    def __init__(self, params, dependencies, func_name):
        super().__init__()
        self._params = params
        self._dependencies = dependencies
        self._func_name = func_name
        
    def get_params(self):
        return(self._params)
    
    def get_dependencies(self):
        return(self._dependencies)
    
    def _generate_python_driver_files(self):
        self.python_file_paths = self._code_generator.generate_function_code(self._func_name, self.chunked_config_file_paths)

class Toy_exp(Experiment):
        
    def easy_submit():
        exp = Toy_exp()
        n_params = exp.get_n_params()
        batch_params = Experiment.get_batch_params_dict(batch_name='JustATest', mem_per_job=10, n_jobs=int(n_params/3), time_per_job=500)
        exp.submit_batch(batch_params=batch_params)
        
    def get_dependencies(self):
        return(['scalable_computing.py'])
    
    def get_params(self):
        params = {}
        params['Ne'] = [200, 50]
        params['h'] = [.001, 33, 34, 54]
        return(params)
    
    def logic(self, params_dict):
        self._set_config_from_dict(params_dict)
        
        print('Some logic for params_dict = {}'.format(params_dict))
        
        

