
# coding: utf-8

# # Main prediction experiment

# In[ ]:

import os
import subprocess
import time
exec(open(os.path.expanduser('scalable_computing.py')).read())
exec(open(os.path.expanduser('pgas-dir.py')).read())
#exec(open(os.path.expanduser('experiments-prediction.py')).read())

# Standard Prediction/Inference experiment
class BayesianLearningExp(Experiment):
    def __init__(self):
        super().__init__()
        self.dat = None
        self.dat_inference = None
        self.time_points_inference = None
        self.obs_inference = None
        self.the_orig_data = None
        self.filter_dim_index = None
        self.the_infer_h = None
        
        self.ncpus_per_job = 1
        
     
    
    def _post_process(self):
        try:
            absolute_path = ''
            try:
                env = os.environ['HOST']
            except:
                env = 'AZURECN'
            #env = os.environ['HOST']
            #if env == '' or env is None: env = 'local'
            if env == '' or env is None: env = 'AZURECN'
            if env == 'local': absolute_path = ''
            elif env == 'MOMAC39': absolute_path = ''
            elif env == 'grex': absolute_path = '' # /global/software/R-3.1.1-rh6/bin/
            elif env == 'shahlab': absolute_path = '/gsc/software/linux-x86_64-centos6/R-3.2.3/bin/'
            
            if env == 'AZURECN':
                print('Summarising for the AZURECN node...')
                try:
                    shared_dir = os.path.join(os.environ['AZ_BATCH_NODE_STARTUP_DIR'], 'wd')
                except:
                    shared_dir = '/mnt/batch/tasks/startup/wd'
                #os.chdir(shared_dir)

                subprocess.call(["sudo", "{}Rscript".format(absolute_path), "--vanilla", os.path.join(shared_dir, "time_series_summariser_exp.R"), os.path.realpath(self.out_path)])
                subprocess.call(["cat", '{}/{}'.format(self.out_path, 'predictsummary.yaml')])
            else:
                subprocess.call(["{}Rscript".format(absolute_path), "--vanilla", "time_series_driver_exp.R", self.out_path])
                subprocess.call(["cat", '{}/{}'.format(self.out_path, 'predictsummary.yaml')])
        except Exception as e:
            print('Some error happened while trying to call Rscript {}'.format(e))
    
            
    def get_dependencies(self):
        return(['scalable_computing.py', 'pgas-dir.py', 'experiments-prediction.py'])
    
    def run_with_config_file(self, config_file):
        self.config_file = os.path.expanduser(config_file)
        stream = open(self.config_file, "r")
        doc = yaml.load(stream)
        print(doc)
        return(self.run(doc))
    
    def _load_original_config_file(self):
        origin_dir = os.path.dirname(os.path.expanduser(self.original_data))            
        stream = open(os.path.join(origin_dir, 'config.yaml'), "r")
        doc = yaml.load(stream)
        self.s = doc['s']
        self.full_original_data = doc['full_original_data']
        
        self.obs_num = doc['obs_num']
        self.true_x0 = doc['true_x0']
        self.configs['full_original_data'] = self.full_original_data
        self.configs['obs_num'] = self.obs_num
        self.configs['true_x0'] = self.true_x0
        self.configs['s'] = self.s
    
    # The data loading logic
    def _load_data(self, resume):
        #print('self.K is {}'.format(self.K))
        random.seed(self.seed)
        try:
            # Does self.original_data exist?
            self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            
            # Read out the correct s value
            self._load_original_config_file()
            
            print('self._load_original_foncifg_file went smoothly.')
            
            if self.K == 1 and len(self.dat.loc[:, 'K'].unique()) > 1 and self.filter_dim_index is not None:
                # Filter the data to keep only k == self.filter_dim_index
                print('before self.dat looks like', self.dat)
                self.dat = self.dat[self.dat['K'] == self.filter_dim_index]
                self.dat['K'] = 0
                print('updated self.dat looks like', self.dat)
                
                
                self.s = self.s[self.filter_dim_index]
                self.true_x0 = self.true_x0[self.filter_dim_index]
                self.configs['s'] = [self.s]
                self.configs['true_x0'] = [self.true_x0]
                # Resave the updated dataframes to ease post processing
                self.original_data = os.path.join(self.out_path, 'sample_data.tsv')
                self.configs['original_data'] = self.original_data
                TimeSeriesDataUtility.save_time_series(self.dat, self.original_data)
                
                #raise ValueError('')
                
        except AttributeError:
            print('There was some error!!!')
            self.original_data = os.path.join(self.out_path, 'sample_data.tsv')
            self.full_original_data = os.path.join(self.out_path, 'sample_data_full.tsv')
            self.configs['original_data'] = self.original_data
            self.configs['full_original_data'] = self.full_original_data
            if resume:
                self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            else:
                #self.dat = WrightFisherDiffusion.generate_sample_data(silent=True, h=self.simul_h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.epsilon, Ne=self.Ne, x0=self.true_x0, K=self.K)   
                #print('****x0 is ', self.true_x0)
                self.dat, self.full_dat = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=self.simul_h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.epsilon, Ne=self.Ne, x0=self.true_x0, K=self.K)   
                TimeSeriesDataUtility.save_time_series(self.dat, self.original_data)
                TimeSeriesDataUtility.save_time_series(self.full_dat, self.full_original_data)

        if resume:
            self.mcmc_loader = MCMC_loader(self.out_path)
        # Filter the data for inference
        self.dat_inference = self.dat.loc[self.dat.time <= self.learn_time, ['time', 'K', 'X']]
        self.time_points_inference = self.dat_inference.loc[:, 'time'].unique()
        # Sanity check
        
        if self.K != len(self.dat_inference.loc[:, 'K'].unique()):
            raise ValueError('Inconsistent K')
        
        self.obs_inference = TimeSeriesDataUtility.tall_to_TK(self.dat_inference)
        
        # backward compatibility
        try:
            rejuvenaion_prob=self.rejuvenaion_prob
        except AttributeError:
            self.rejuvenaion_prob = 1
        
    
    # The infernce (learning of \theta) logic
    def _infer(self, resume):
        # Setup samplers
        if resume and os.path.isfile('{}.gz'.format(self.prediction_file_path)):
            last_x = MCMC_loader.get_last_infer_x(self.out_path)
            last_theta = MCMC_loader.get_last_infer_theta(self.out_path)
            return([last_theta, last_x])
        
        g = InformativeEmission(alpha=[1]*self.K, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = WrightFisherDiffusion(h=self.infer_h, K=self.K, Ne=self.Ne)
        #print('ERROR! PROVIDING TRUE X as X0')
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.infer_h, full_original_dat=None)
        
        pg_bridge_kernel = ParticleBridgeKernel(h=self.infer_h, K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = PGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference), observations=self.obs_inference, transitionDistribution=f, h=self.infer_h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)

        print('### Warning! Using negative s...')
        #lower_s = -0.3
        lower_s = -.5
        theta_init_sampler = s_uniform_sampler(dim_max=.5, dim_min=lower_s, K=self.K)
        adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([lower_s]*self.K), upper_bound=np.array([.5]*self.K))
        
        theta_sampler = MH_Sampler(adapted_proposal_distribution=adapted_theta_proposal, likelihood_distribution=f)
        fitness_learner = OutterPGAS(initial_distribution_theta=theta_init_sampler, initial_distribution_x=gp_sampler, observations=self.obs_inference, smoothing_kernel=pgas_sampler, parameter_proposal_kernel=theta_sampler, h=self.infer_h, MCMC_in_Gibbs_nIter=self.MCMC_in_Gibbs_nIter)
        fitness_learner.sample_processor = Sample_processor(self.time_points_inference, self)
        
        is_resume = resume and os.path.isfile('{}.gz'.format(self.inference_x_file_path))
        if is_resume:
            fitness_learner.data_loader = self.mcmc_loader
                
        return(fitness_learner.sample(self.inference_n_iter, xprime, is_resume=is_resume))

        
    # The prediction logic
    def _predict(self, xprime, theta_vector, resume):
        time_points = self.dat.loc[:, 'time'].unique()
        T = len(time_points)
        obs = TimeSeriesDataUtility.tall_to_TK(self.dat)
        
        #print('Testing prediction blindness')
        #obs['value'][obs['times'] > self.learn_time, ] = .876
        #print('obs[values] ={}'.format(obs['value']))
        
        g = InformativeEmission(alpha=[1]*self.K, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = WrightFisherDiffusion(h=self.infer_h, K=self.K, Ne=self.Ne)
        #print('ERROR! PROVIDING TRUE X as X0')
        #print('__ERROR!__')
        #print('USING FULL ORIGINAL PATH TO START...')
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.infer_h, full_original_dat=None)
        
        pg_bridge_kernel = ParticleBridgeKernel(h=self.infer_h, K=self.K, N=self.pf_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = PGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K=self.K, N=self.pf_n_particles, T=T, observations=obs, transitionDistribution=f, h=self.infer_h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores, disable_ancestor_bridge=self.disable_ancestor_bridge)
        pgas_sampler.particle_processor = Particle_processor(time_points=time_points, owner=self)
        
        fitness_learner = OutterPGAS(initial_distribution_theta=None, initial_distribution_x=gp_sampler, observations=obs, smoothing_kernel=pgas_sampler, parameter_proposal_kernel=None, h=self.infer_h)
        fitness_learner.sample_processor = Sample_processor(time_points, self)
        filtered_res_theta = theta_vector[-self.pf_n_theta:None,]
        if resume and os.path.isfile('{}.gz'.format(self.inference_x_file_path)):
            fitness_learner.data_loader = self.mcmc_loader
        is_resume = resume and os.path.isfile('{}.gz'.format(self.prediction_file_path))
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)    
        list_x_predict = fitness_learner.predict(t_learn=self.learn_time, theta_vector=filtered_res_theta, x0=xprime, is_resume=is_resume)
        
    def example_logic(self, resume=False, filtering=False):
        ## Load data
        self._load_data(resume)
        
        ## Inference
        if ~filtering:
            print('Inferring...')
            last_theta, last_x = self._infer(resume)
            print('Saving inference results to\n {}'.format(self.inference_x_file_path))

        ## Predict
        print('Predicting...')
        # Load the theta vector
        res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        #res_theta = TimeSeriesDataUtility.read_time_series('/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201706-30-17635.860598/infer_theta.tsv.gz').values

        self._predict(xprime=None, theta_vector=res_theta, resume=resume)
     
    
class TimingExp(BayesianLearningExp):
    
    #def _post_process(self):
    #    pass
    
    def _easy_submit(batch_name, mem_per_job=50, n_jobs=None, time_per_job=80*60, ncores=1, k=2, seed=2):
        expt = TimingExp()
        expt.the_k = k
        expt.ncpus_per_job = ncores
        expt.the_seed = seed
        n_params = expt.get_n_params()
        print(n_params)
        if n_jobs is None:
            n_jobs = int(n_params)
        batch_params = Experiment.get_batch_params_dict(batch_name=batch_name, mem_per_job=mem_per_job, n_jobs=n_jobs, time_per_job=time_per_job, ncpus_per_job=ncores)
        expt.submit_batch(batch_params=batch_params)
        
    def generate_starting_value(self, k, count=1):
        res = [None]*count
        for i in range(count):
            res[i] = (np.random.dirichlet([1]*(k+1), 1)[0][0:k]).tolist()
        return(res)

    def get_params(self):
        params = {}
        #params['Ne'] = [50, 200]
        #params['Ne'] = [50] # THE MOST IMPORTANT!!!
        params['Ne'] = [500] 
        
        #params['seed'] = [i*10 for i in range(5)]
        params['seed'] = [10]
        params['simul_h'] = [.001]
        
        params['epsilon'] = [.001]
        #params['K'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        params['K'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        temp_k = params['K'][0]

        # Data
        #params['s'] = [np.random.uniform(-1, 10, temp_k).tolist()]
        #params['s'] = [[.1]]
        #params['s'] = [[.1]*temp_k]
        # Use Truncated nomral
        mu = 0
        scale = .3
        a, b = (-.5 - mu)/scale, (10 - mu)/scale
        temp_s = sp.stats.truncnorm.rvs(loc=mu, scale=scale, a=a, b=b, size=temp_k).tolist()
        params['s'] = [temp_s]
        #params['s'] = [np.random.uniform(-1, .5, temp_k).tolist()]
        #params['obs_num'] = [i for i in range(4, 10)]
        params['obs_num'] = [5]
        params['end_time'] = [0.1]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        #params['learn_time'] = [.06, .09, .12]
        params['learn_time'] = [0.1]
        params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+1), 1)[0][0:temp_k]).tolist()]
        #params['true_x0'] = [[.1]*temp_k]
        #params['true_x0'] = self.generate_starting_value(temp_k, 1)

        
        ## Sampler options
        TESTING = False
        if TESTING is True:
            params['MCMC_in_Gibbs_nIter'] = [10]
            params['gp_n_opt_restarts'] = [20]
            
            params['pgas_n_particles'] = [10000]
            params['inference_n_iter'] = [500]
            params['pf_n_particles'] = [1000]
            params['pf_n_theta'] = [50]
        else:
            params['gp_n_opt_restarts'] = [20]
            params['pgas_n_particles'] = [10000]
            params['inference_n_iter'] = [10000]
            params['MCMC_in_Gibbs_nIter'] = [20]
            params['pf_n_particles'] = [10000]
            params['pf_n_theta'] = [500]

        # For K=1 experiments, sets which alelle should be kept
        #params['filter_dim_index'] = [1]
        # Sanity check: Only set this if K == 1
        if temp_k == 1:
            params['filter_dim_index'] = [self.the_filter_dim_index]
        
    
        if self.the_infer_h is not None:
            params['infer_h'] = [self.the_infer_h]
        else:
            params['infer_h'] = params['simul_h']
        
        # sampler
        params['proposal_step_sigma'] = [[.1]*temp_k]
    
        # Prediction
        params['disable_ancestor_bridge'] = [True]
        #params['infer_epsilon'] = [.01, .02, .05]=
        #params['infer_epsilon'] = [.005]
        params['infer_epsilon'] = [.01]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0.0]

        params['out_path'] = ['~/Desktop/pgas_sanity_check/jan17/exp/']
        if self.the_orig_data is not None:
            params['original_data'] = [self.the_orig_data]
            #params['original_data'] = ['~/Desktop/pgas_sanity_check/clump_k3_exp_Y0CQ6_201711-14-154502.161354/sample_data.tsv']
        return(params)

    
    def logic(self, resume):
        self._load_data(resume)
        
        #%matplotlib inline
        #TimeSeriesDataUtility.plot_tall(self.full_dat, legend='', title='Real value')
        last_theta, last_x = self._infer(resume)
        
        #print('Saving inference results to\n {}'.format(self.inference_x_file_path))
        
        res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        #res_theta = TimeSeriesDataUtility.read_time_series('/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_C14AN_201801-13-092510.981043/infer_theta.tsv.gz').values
        #res_theta = np.array([self.s]*self.pf_n_theta)
        self._predict(xprime=None, theta_vector=res_theta, resume=resume)

import time
#TimingExp()._run_default()

#TimingExp().resume_exp('/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201707-16-145503.332905')
for the_seed in range(10):
    np.random.seed(1+the_seed * 10)
#expt = TimingExp(); expt.the_k = 1; expt.ncpus_per_job = 1; expt.the_filter_dim_index = 1; tt = expt._run_default()
    expt = TimingExp(); 
#expt.original_data = '/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_C14AN_201801-13-092510.981043/sample_data.tsv.gz'
    expt.the_k = 4; expt.ncpus_per_job = 1; tt = expt._run_default()
#np.random.seed(90)
#expt = TimingExp(); expt.the_k = 4; expt.ncpus_per_job = 1; tt = expt._run_default()
# [0.19676098732281286, 0.03684986266960172, 0.05328435425432679, 0.13779794583366614],


# In[ ]:




# In[ ]:




# In[ ]:




# In[14]:

get_ipython().magic('matplotlib inline')
#TimeSeriesDataUtility.plot_tall(expt.full_dat, legend='', title='Full Real value')
#TimeSeriesDataUtility.plot_tall(expt.dat, legend='', title='Real value')
#TimeSeriesDataUtility.plot_tall(expt.dat_inference, legend='', title='Real Infer value')


# In[9]:

#expt.dat_inference


# In[15]:

def test_k_full(the_K, some_seed):
    import time
    np.random.seed(some_seed)    
    expt=TimingExp(); 
    expt.the_k=the_K; 
    expt.ncpus_per_job = 4; 
    tt = expt._run_default()
    return(expt.original_data)

def test_k_one(some_seed, data_path):
    import time
    np.random.seed(some_seed)
    for i in range(0,4):
        expt=TimingExp(); 
        expt.the_k=1; 
        expt.the_filter_dim_index=i; 
        #expt.the_orig_data = os.path.expanduser('~/Desktop/pgas_sanity_check/exp_Y0CQ6_201711-15-142602.897583/sample_data.tsv')
        expt.the_orig_data = os.path.expanduser(data_path)
        expt.ncpus_per_job = 1; 
        tt = expt._run_default()
        
def test_one_step(the_K, some_seed, data_path):
    import time
    np.random.seed(some_seed)
    expt=TimingExp(); 
    expt.the_k=the_K; 
    expt.the_orig_data = os.path.expanduser(data_path)
    expt.the_infer_h = .01
    expt.ncpus_per_job = 1; 
    tt = expt._run_default()
    return(expt.original_data)

#test_k_one()


# In[ ]:




# # Model comparison experiment set 1:
# Running the full model, then the K=1 model and finally the one step model over the same same datasets.
# 
# First full model is run over 5 seeds > 20. 
# Then the h model is run over exactly that dataset
# Then the K=1 is run over exactly that dataset
# 

# In[11]:

the_K = 4
data_files = []
#for some_seed in [30, 40, 50, 60, 70]:
#for some_seed in [30, 40]:
#for some_seed in [50, 60, 70, 80, 90, 100]:
#for some_seed in [60, 70, 90]:
#seed=90 exp_Y0CQ6_201711-17-133721.774863 
#for some_seed in [20, 30, 40, 50, 80, 100]:
#for some_seed in [90]:
#for some_seed in [i*10 for i in range(11, 50)]:
for some_seed in [i*10 for i in range(50, 100)]:
    # Run the full model 
    data_path = test_k_full(the_K, some_seed)
    # Keep record
    data_files.append(data_path)
    # Run the one_step model
    test_one_step(the_K, some_seed, data_path)
    # Run the K_1 model
    test_k_one(some_seed, data_path)
    
print(data_files)
#thefile = open('/Users/sohrabsalehi/Google Drive/BCCRC/Meeting_alex_sohrab/data_sets_nov19.txt', 'w+')
thefile = open('/Users/ssalehi/Google Drive/BCCRC/Meeting_alex_sohrab/data_sets_nov_22_long.txt', 'w+')
for item in data_files:
    thefile.write("%s\n" % item)
thefile.close()


# In[ ]:




# In[ ]:

print(data_files)
#thefile = open('/Users/sohrabsalehi/Google Drive/BCCRC/Meeting_alex_sohrab/data_sets_nov19.txt', 'w+')
thefile = open('/Users/ssalehi/Google Drive/BCCRC/Meeting_alex_sohrab/data_sets_nov19.txt', 'w+')
for item in data_files:
    thefile.write("%s\n" % item)
thefile.close()


# In[ ]:

data_files


# In[173]:

thefile = open('/Users/sohrabsalehi/Google Drive/BCCRC/Meeting_alex_sohrab/data_sets.txt', 'w')
for item in data_files:
    thefile.write("%s\n" % item)


# In[178]:

thefile.close()


# In[ ]:




# In[40]:

import time
#TimingExp()._run_default()

#np.random.seed(90)
#expt = TimingExp(); expt.the_k = 4;  
#expt.the_orig_data = os.path.expanduser('~/Desktop/pgas_sanity_check/exp_Y0CQ6_201711-15-142602.897583/sample_data.tsv')
#expt.ncpus_per_job = 1; tt = expt._run_default()


# In[141]:

#expt.original_data
#TimingExp._easy_submit(batch_name='PSweep{}_'.format(2), mem_per_job=12, n_jobs=None, time_per_job=30*60, k = 2, ncores=1)


# In[2]:

# expt = TimingExp()
# expt.ncpus_per_job = 1
# expt.the_k = 2

# n_params = expt.get_n_params()
# print(n_params)


# In[44]:

# for k in range(2, 5):
#     TimingExp._easy_submit(batch_name='PSweep{}_'.format(k), mem_per_job=12, n_jobs=None, time_per_job=30*60, k = k, ncores=1)
#k=7; TimingExp._easy_submit(batch_name='PSweep{}_'.format(k), mem_per_job=12, n_jobs=None, time_per_job=30*60, k = k, ncores=1)


# In[43]:

# out_path = '/Users/sohrabsalehi/projects/fitness/batch_runs/PSweep7__201708-21-154820.123063/yaml/test_case_4'
# yamls = os.listdir(out_path)
# for file in yamls:
#     print(os.path.join(out_path, file))
#     conf = yaml.load(open(os.path.join(out_path, file), 'r'))
#     #conf['out_path'] = os.path.basename(conf['out_path'])
#     conf['n_cores'] = 4
#     conf['bridge_n_cores'] = 4
#     yaml.dump(conf, open(os.path.join(out_path, file), 'w'))


# In[ ]:




# In[46]:

# ss = '/Users/sohrabsalehi/projects/fitness/batch_runs/PSweep7__201708-21-154820.123063/yaml/param_chunk_194.yaml'
# conf = yaml.load(open(ss, 'r'))
# os.path.basename(conf['out_path'])


# In[45]:

# conf['n_cores']
# TimingExp._easy_submit(batch_name='PSweep{}_'.format(2), mem_per_job=12, n_jobs=None, time_per_job=30*60, k = 2, ncores=1)
# # expt = TimingExp()
# # expt.the_k = 2
# # expt.ncpus_per_job = 3
# # expt._run_default()


# In[ ]:




# In[5]:

#k=2;TimingExp._easy_submit(batch_name='ex{}_'.format(k), mem_per_job=5, n_jobs=None, time_per_job=24*60, k=k, ncores=12)


# In[6]:

# >>> for k in range(5, 10):
# ...     TimingExp._easy_submit(batch_name='PSweep{}_'.format(k), mem_per_job=12, n_jobs=None, time_per_job=30*60, k = k, ncores=1)
# ... 
# 540
# Restricting jobs to node0502|node0504|node0506|node0507|node0509|node0510
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep5__201708-05-153155.088674/qsubFiles/submit.sh
# /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep5__201708-05-153155.088674/qsubFiles/submit.sh
# 540
# Restricting jobs to node0502|node0504|node0506|node0507|node0509|node0510
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep6__201708-05-153158.330779/qsubFiles/submit.sh
# /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep6__201708-05-153158.330779/qsubFiles/submit.sh
# 540
# Restricting jobs to node0502|node0504|node0506|node0507|node0509|node0510
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep7__201708-05-153201.644334/qsubFiles/submit.sh
# /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep7__201708-05-153201.644334/qsubFiles/submit.sh
# 540
# Restricting jobs to node0502|node0504|node0506|node0507|node0509|node0510
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep8__201708-05-153205.083605/qsubFiles/submit.sh
# /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep8__201708-05-153205.083605/qsubFiles/submit.sh
# 540
# Restricting jobs to node0502|node0504|node0506|node0507|node0509|node0510
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep9__201708-05-153208.649750/qsubFiles/submit.sh
# /shahlab/ssalehi/scratch/fitness/batch_runs/PSweep9__201708-05-153208.649750/qsubFiles/submit.sh


# In[7]:

# >>> for k in range(2, 5):
# ...     TimingExp._easy_submit(batch_name='PSweep{}_'.format(k), mem_per_job=12, n_jobs=None, time_per_job=30*60, k = k, ncores=1)
# ... 
# 540
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /global/scratch/sohrab/fitness/batch_runs/PSweep2__201708-05-174351.744451/qsubFiles/submit.sh
# /global/scratch/sohrab/fitness/batch_runs/PSweep2__201708-05-174351.744451/qsubFiles/submit.sh
# 540
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /global/scratch/sohrab/fitness/batch_runs/PSweep3__201708-05-174357.788512/qsubFiles/submit.sh
# /global/scratch/sohrab/fitness/batch_runs/PSweep3__201708-05-174357.788512/qsubFiles/submit.sh
# 540
# Done creating Scripts!
# RUN THE BASHSCRIPT YOURSELF!!!
# chmod +x /global/scratch/sohrab/fitness/batch_runs/PSweep4__201708-05-174404.492719/qsubFiles/submit.sh
# /global/scratch/sohrab/fitness/batch_runs/PSweep4__201708-05-174404.492719/qsubFiles/submit.sh


# In[31]:

# import numpy as np
# from scipy.stats import multivariate_normal
# x = np.linspace(0, 5, 10, endpoint=False)
# y = multivariate_normal.logpdf(x, mean=2.5, cov=0.5);
# y


# cov_mat = np.array([[  8.35323250e-05,  -1.20673318e-05,  -1.59806342e-05,  -4.25027511e-05],
#  [ -1.20673318e-05,   1.13966697e-04,  -2.27862119e-05,  -6.06031450e-05],
#  [ -1.59806342e-05,  -2.27862119e-05 ,  1.43535521e-04 , -8.02560754e-05],
#  [ -4.25027511e-05,  -6.06031450e-05,  -8.02560754e-05,   2.48556687e-04]])
# mu = np.array([ 0.01064356, -0.01766087, -0.01964779,  0.026194 ])*2
# xi= np.array([ 0.12258389,  0.11011092 , 0.17390026 , 0.48995754])
# xim1 = np.array([ 0.0919955 ,  0.13117307 , 0.17371105,  0.46200904] )
# sp.stats.multivariate_normal.pdf(x=xi, mean=xim1+mu, cov=cov_mat, allow_singular=True)
# ss = multivariate_normal.logpdf([.2, .3, .3], mean=[.4, .5, .5], cov=np.array([[1,0,0], [0,1,0], [0,0,1]]))
# print(ss)

# tt = multivariate_normal.pdf([.2, .3, .3], mean=[.4, .5, .5], cov=np.array([[1,0,0], [0,1,0], [0,0,1]]))
# tt
# log(tt)

