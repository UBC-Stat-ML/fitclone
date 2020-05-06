
# coding: utf-8

# In[48]:

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
            #res[i] = (np.random.dirichlet([10]*(k+1), 1)[0][0:k]).tolist()
            shuffleMap = np.array(list(range(k+1))) + 1
            np.random.shuffle(shuffleMap)
            shuffleMap = shuffleMap/np.sum(shuffleMap)
            #res[i] = (np.random.dirichlet([5]*(k+1), 1)[0][0:k]).tolist()
            res[i] = (shuffleMap[0:k]).tolist()
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
        params['obs_num'] = [10]
        params['end_time'] = [0.1]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        #params['learn_time'] = [.06, .09, .12]
        params['learn_time'] = [0.08]
        #params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+1), 1)[0][0:temp_k]).tolist()]
        #params['true_x0'] = [[.1]*temp_k]
        params['true_x0'] = self.generate_starting_value(temp_k, 1)

        
        ## Sampler options
        TESTING = False
        if TESTING is True:
            params['MCMC_in_Gibbs_nIter'] = [1]
            params['gp_n_opt_restarts'] = [10]
            
            params['pgas_n_particles'] = [100]
            params['inference_n_iter'] = [20]
            params['pf_n_particles'] = [100]
            params['pf_n_theta'] = [50]
        else:
            params['gp_n_opt_restarts'] = [100]
            params['pgas_n_particles'] = [10000]
            params['inference_n_iter'] = [10000]
            params['MCMC_in_Gibbs_nIter'] = [20]
            params['pf_n_particles'] = [10000]
            params['pf_n_theta'] = [5000]

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

        #params['out_path'] = ['~/Desktop/pgas_sanity_check/jan17/exp/']
        params['out_path'] = [self.the_out_path]
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
        self._predict(xprime=None, theta_vector=res_theta, resume=resume)

import time


# In[49]:

def test_k_full(the_K, some_seed, the_out_path):
    import time
    np.random.seed(some_seed)    
    expt=TimingExp(); 
    expt.the_k=the_K; 
    expt.ncpus_per_job = 1;
    expt.the_out_path = the_out_path
    tt = expt._run_default()
    return(expt.original_data)

def test_k_one(some_seed, data_path, the_out_path):
    import time
    np.random.seed(some_seed)
    for i in range(0,4):
        expt=TimingExp(); 
        expt.the_k=1; 
        expt.the_filter_dim_index=i; 
        expt.the_orig_data = os.path.expanduser(data_path)
        expt.ncpus_per_job = 1; 
        expt.the_out_path = the_out_path
        tt = expt._run_default()
        
def test_one_step(the_K, some_seed, data_path, the_out_path):
    import time
    np.random.seed(some_seed)
    expt=TimingExp(); 
    expt.the_k=the_K; 
    expt.the_orig_data = os.path.expanduser(data_path)
    expt.the_infer_h = .01
    expt.ncpus_per_job = 1; 
    expt.the_out_path = the_out_path
    tt = expt._run_default()
    return(expt.original_data)

#test_k_one()


# In[45]:

#def run_model_comparison(the_K, some_seed, batch_path):
def run_model_comparison(options, batch_path):
    the_K = options['the_K']
    some_seed = options['some_seed']
    # Run the full model 
    data_path = test_k_full(the_K, some_seed, options['out_path'])
    # Write the data_path
    data_dir = os.path.join(batch_path, 'data_files/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    thefile = open(os.path.join(data_dir, 'data_file_{}.txt'.format(some_seed)), 'w+')
    thefile.write("%s\n" % data_path)
    thefile.close()

    # Run the one_step model
    test_one_step(the_K, some_seed, data_path, options['out_path'])
    # Run the K_1 model
    test_k_one(some_seed, data_path, options['out_path'])


# In[50]:

#test_k_full(the_K=4, some_seed=10, the_out_path='/Users/sohrabsalehi/Desktop/model_comp/')


# In[43]:




# In[41]:




# In[42]:




# In[29]:




# In[30]:




# In[ ]:



