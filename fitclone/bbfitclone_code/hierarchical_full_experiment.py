
# coding: utf-8

# # Supporting multiple repeats

# In[1]:

import os
import subprocess
import time
exec(open(os.path.expanduser('experiments-prediction.py')).read())


# Standard Prediction/Inference experiment
class HierarchicalBayesianLearning(BayesianLearningExp):
    
    def _load_data(self, resume):
        for m in range(self.M_data):
            random.seed(self.seed[m])

        self.dat = [None]*self.M_data
        self.full_dat = [None]*self.M_data
        self.dat_inference = [None]*self.M_data
        self.time_points_inference = [None]*self.M_data
        self.obs_inference = [None]*self.M_data
        self.original_data = [None]*self.M_data
        self.full_original_data = [None]*self.M_data

        for m in range(self.M_data):                
#             try:
#                 # Does self.original_data exist?
#                 self.dat[m] = TimeSeriesDataUtility.read_time_series(self.original_data[m])
#             except (AttributeError, TypeError) as e:
            self.original_data[m] = os.path.join(self.out_path, 'sample_data_{}.tsv'.format(m))
            self.full_original_data[m] = os.path.join(self.out_path, 'sample_data_full_{}.tsv'.format(m))
            if resume:
                self.dat[m] = TimeSeriesDataUtility.read_time_series(self.original_data[m])
            else:
                print('self.true_x0[m] is ', self.true_x0[m])
                self.dat[m], self.full_dat[m] = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=self.simul_h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.epsilon, Ne=self.Ne, x0=self.true_x0[m], K=self.K)   
                print('self.dat[m] is ', self.dat[m])
                TimeSeriesDataUtility.save_time_series(self.dat[m], self.original_data[m])
                TimeSeriesDataUtility.save_time_series(self.full_dat[m], self.full_original_data[m])
            
            if resume:
                print('WARNING! Resume not supported!!!')
                self.mcmc_loader = MCMC_loader(self.out_path)
            # Filter the data for inference
            self.dat_inference[m] = self.dat[m].loc[self.dat[m].time <= self.learn_time, ['time', 'K', 'X']]
            self.time_points_inference[m] = self.dat_inference[m].loc[:, 'time'].unique()

            # Sanity check
            if self.K != len(self.dat_inference[m].loc[:, 'K'].unique()):
                raise ValueError('Inconsistent K'.format(self.K, en(self.dat_inference[m].loc[:, 'K'].unique())))

            self.obs_inference[m] = TimeSeriesDataUtility.tall_to_TK(self.dat_inference[m])

        self.configs['original_data'] = self.original_data
        self.configs['full_original_data'] = self.full_original_data
        
        # backward compatibility
        try:
            rejuvenaion_prob=self.rejuvenaion_prob
        except AttributeError:
            self.rejuvenaion_prob = 1
            
        
    
    # The infernce (learning of \theta) logic
    def _infer(self, resume):
        if resume and os.path.isfile('{}.gz'.format(self.prediction_file_path)):
            last_x = MCMC_loader.get_last_infer_x(self.out_path)
            last_theta = MCMC_loader.get_last_infer_theta(self.out_path)
            return([last_theta, last_x])
        
        g = InformativeEmission(alpha=[1]*self.K, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = WrightFisherDiffusion(h=self.infer_h, K=self.K, Ne=self.Ne)
        
        gp_sampler = [None]*self.M_data
        pg_bridge_kernel = [None]*self.M_data
        pgas_sampler = [None]*self.M_data
        xprime = [None]*self.M_data
        sample_processors = [None]*self.M_data
        
        for m in range(self.M_data):
            gp_sampler[m] = GP_sampler(dat=self.obs_inference[m]['value'], obsTimes=self.time_points_inference[m], epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.infer_h)
            pg_bridge_kernel[m] = ParticleBridgeKernel(h=self.infer_h, K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler[m], transitionDistribution=f, n_cores=self.bridge_n_cores)
            pgas_sampler[m] = PGAS(bridgeKernel=pg_bridge_kernel[m], emissionDistribution=g, proposalDistribution=g, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference[m]), observations=self.obs_inference[m], transitionDistribution=f, h=self.infer_h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
            xprime[m] = pgas_sampler[m].generate_dummy_trajectory(self.learn_time)

        #print('### Warning! Using negative s...')
        #lower_s = -0.3
        lower_s = -.5
        upper_s = .5
        theta_init_sampler = s_uniform_sampler(dim_max=upper_s, dim_min=lower_s, K=self.K)
        adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K), sigma=np.array(self.proposal_step_sigma), lower_bound=np.array([lower_s]*self.K), upper_bound=np.array([upper_s]*self.K))
        theta_sampler = MH_Sampler(adapted_proposal_distribution=adapted_theta_proposal, likelihood_distribution=f)
        fitness_learner = HierarchicalOuterPGAS(initial_distribution_theta=theta_init_sampler, initial_distribution_list_x=gp_sampler, observations_list=self.obs_inference, smoothing_kernels=pgas_sampler, 
                                     parameter_proposal_kernel=theta_sampler, h=self.infer_h, MCMC_in_Gibbs_nIter=self.MCMC_in_Gibbs_nIter)
        for m in range(self.M_data):
            sample_processors[m] = Sample_processor(time_points=self.time_points_inference[m], owner=self, m_id=m)
        fitness_learner.sample_processors = sample_processors
        
        is_resume = resume and os.path.isfile('{}.gz'.format(self.inference_x_file_path[0]))
        if is_resume:
            fitness_learner.data_loader = self.mcmc_loader
                
        return(fitness_learner.sample_hierarchical(self.inference_n_iter, xprime, is_resume=is_resume))
        
        
    # The prediction logic
    def _predict(self, xprime, theta_vector, resume):
        time_points = self.dat[0].loc[:, 'time'].unique()
        T = len(time_points)
        obs = TimeSeriesDataUtility.tall_to_TK(self.dat[0])
        
        #print('Testing prediction blindness')
        #obs['value'][obs['times'] > self.learn_time, ] = .876
        #print('obs[values] ={}'.format(obs['value']))
        
        g = InformativeEmission(alpha=[1]*self.K, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = WrightFisherDiffusion(h=self.infer_h, K=self.K, Ne=self.Ne)
        #print('ERROR! PROVIDING TRUE X as X0')
        gp_sampler = GP_sampler(dat=self.obs_inference[0]['value'], obsTimes=self.time_points_inference[0], epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.infer_h, full_original_dat=None)
        
        pg_bridge_kernel = ParticleBridgeKernel(h=self.infer_h, K=self.K, N=self.pf_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = PGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K=self.K, N=self.pf_n_particles, T=T, observations=obs, transitionDistribution=f, h=self.infer_h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores, disable_ancestor_bridge=self.disable_ancestor_bridge)
        
        fitness_learner = OutterPGAS(initial_distribution_theta=None, initial_distribution_x=gp_sampler, observations=obs, smoothing_kernel=pgas_sampler, parameter_proposal_kernel=None, h=self.infer_h)
        fitness_learner.sample_processor = Sample_processor(time_points, self)
        filtered_res_theta = theta_vector[-self.pf_n_theta:None,]
        #if resume and os.path.isfile('{}.gz'.format(self.inference_x_file_path[0])):
        #    fitness_learner.data_loader = self.mcmc_loader
        is_resume = resume and os.path.isfile('{}.gz'.format(self.prediction_file_path))
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)    
        list_x_predict = fitness_learner.predict(t_learn=self.learn_time, theta_vector=filtered_res_theta, x0=xprime, is_resume=is_resume)
        
        
class HierNormalExp(HierarchicalBayesianLearning):
    
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
        
#     def generate_starting_value(self, k, count=1):
#         res = [None]*count
#         for i in range(count):
#             res[i] = (np.random.dirichlet([1]*(k+1), 1)[0][0:k]).tolist()
#         return(res)
    
    
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
    
    # OLD
    def get_params_old(self):
        params = {}
        params['Ne'] = [500]
        params['seed'] = [[i*10 for i in range(self.M_data)]]
        params['h'] = [.001]
        params['epsilon'] = [.01]
        params['K'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        temp_k = params['K'][0]
        params['M_data'] = [self.M_data]

        # Data
        params['s'] = [np.random.uniform(0, .5, temp_k).tolist()]
        params['obs_num'] = [10]
        params['end_time'] = [.05]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['learn_time'] = [.05]
        params['true_x0'] = [self.generate_starting_value(temp_k, self.M_data)]

        params['gp_n_opt_restarts'] = [100]

        ## Sampler options
        params['pgas_n_particles'] = [10000]
        params['inference_n_iter'] = [5000]

        # sampler
        params['proposal_step_sigma'] = [[.1]*temp_k]

        # Prediction
        params['disable_ancestor_bridge'] = [True]
        params['infer_epsilon'] = [.01]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0.0]
        params['pf_n_particles'] = [10000]
        params['pf_n_theta'] = [1000]
        params['MCMC_in_Gibbs_nIter'] = [10]
        params['out_path'] = ['~/Desktop/pgas_sanity_check/exp']
        return(params)

    def get_params(self):
        params = {}
        params['Ne'] = [500] 
        params['seed'] = [[i*10 for i in range(self.M_data)]]
        params['simul_h'] = [.001]
        
        params['epsilon'] = [.001]
        params['K'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        temp_k = params['K'][0]
        params['M_data'] = [self.M_data]

        # Data
        # Use Truncated nomral
        mu = 0
        scale = .3
        a, b = (-.5 - mu)/scale, (10 - mu)/scale
        temp_s = sp.stats.truncnorm.rvs(loc=mu, scale=scale, a=a, b=b, size=temp_k).tolist()
        params['s'] = [temp_s]
        params['obs_num'] = [10]
        params['end_time'] = [0.1]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['learn_time'] = [0.08]
        params['true_x0'] = [self.generate_starting_value(temp_k, self.M_data)]

        
        print('ss_self.M_data = ', self.M_data)
        print('ss_self.true_x0 = ', params['true_x0'])
        
        ## Sampler options
        TESTING = False
        if TESTING is True:
            params['MCMC_in_Gibbs_nIter'] = [10]
            params['gp_n_opt_restarts'] = [100]
            params['pgas_n_particles'] = [100]
            params['inference_n_iter'] = [100]
            params['pf_n_particles'] = [100]
            params['pf_n_theta'] = [50]
        else:
            params['gp_n_opt_restarts'] = [100]
            params['pgas_n_particles'] = [10000]
            params['inference_n_iter'] = [10000]
            params['MCMC_in_Gibbs_nIter'] = [20]
            params['pf_n_particles'] = [10000]
            params['pf_n_theta'] = [5000]

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
        params['infer_epsilon'] = [.01]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0.0]

        params['out_path'] = [self.the_out_path]
        if self.the_orig_data is not None:
            params['original_data'] = [self.the_orig_data]
            
        return(params)


    def logic(self, resume):
        self._load_data(resume)
        last_theta, last_x = self._infer(resume)
        #print('Saving inference results to\n {}'.format(self.inference_x_file_path[0]))
        
        #res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        #res_theta = np.array([self.s]*self.pf_n_theta)
        #self._predict(xprime=None, theta_vector=res_theta, resume=resume)

import time
#TimingExp()._run_default()

#TimingExp().resume_exp('/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201707-16-145503.332905')
#np.random.seed(2)
#expt = TimingExp(); expt.the_k = 10; expt.ncpus_per_job = 8; tt = expt._run_default()
#np.random.seed(100)
#expt = HierNormalExp(); expt.M_data = 3; expt.the_k = 4; expt.ncpus_per_job = 1; tt = expt._run_default()


# In[2]:

def test_k_full_hie(the_K, some_seed, the_out_path):
    import time
    # Make sure initial params are identical to seed
    np.random.seed(some_seed)    
    expt = HierNormalExp(); 
    expt.the_k=the_K; 
    expt.ncpus_per_job = 1;
    expt.the_out_path = the_out_path
    expt.M_data = 3;
    tt = expt._run_default()
    return(expt.original_data)


# In[3]:

#test_k_full_hie(4, 12, '/Users/sohrabsalehi/Desktop/hie_test')


# In[27]:

def run_model_hie(options, batch_path):
    the_K = options['the_K']
    some_seed = options['some_seed']
    # Run the full model 
    data_path = test_k_full_hie(the_K, some_seed, options['out_path'])
    # Write the data_path
    data_dir = os.path.join(batch_path, 'data_files/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    thefile = open(os.path.join(data_dir, 'data_file_{}.txt'.format(some_seed)), 'w+')
    thefile.write("%s\n" % data_path)
    thefile.close()


# In[ ]:



