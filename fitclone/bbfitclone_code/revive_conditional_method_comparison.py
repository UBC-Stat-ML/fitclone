
# coding: utf-8

# In[7]:

import os

exec(open(os.path.expanduser('SC_DLP_sa501_exp.py')).read())
exec(open(os.path.expanduser('extended_wf_model.py')).read())

class ConditionalBayesianLearning(Bayesian_learning_exp):
    def _compute_one_step_h(self, time_points):
        h = np.empty([len(time_points), 1])
        for time in range(1, len(time_points)+1):
            h[time] = time_points[time] - time_points[time-1]
        return(h)
    
    def _load_data(self, resume):
        # Set one-step h
        self.one_step = False
        if self.one_step != False:
            self.h = self._compute_one_step_h(self.time_points_inference)        
        else:
            #self.h = np.repeat(self.h, len(self.time_points_inference))
            self.h = np.repeat(self.h, self.obs_num)
        
        print('self.h is {}'.format(self.h))
        random.seed(self.seed)
        try:
            # Does self.original_data exist?
            self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            # Load trueX and S from there too
            # 1. 
            # old_configs_path = os.path.dirname(self.original_data)
            # old_configs_path = os.path.join(old_configs_path, 'config.yaml')
            # print(old_configs_path)
            # stream = open(old_configs_path, "r")
            # doc = yaml.load(stream)
            # self.configs['s'] = doc['s']
            # self.configs['true_x0'] = doc['true_x0']
            # self.configs['full_original_data'] = doc['full_original_data']
            
            # self.s = self.configs['s']
            # self.full_original_data = self.configs['full_original_data']
            # self.true_x0 = self.configs['true_x0']
            
            
        except AttributeError:
            self.original_data = os.path.join(self.out_path, 'sample_data.tsv')
            self.full_original_data = os.path.join(self.out_path, 'sample_data_full.tsv')
            self.configs['original_data'] = self.original_data
            self.configs['full_original_data'] = self.full_original_data
            if resume:
                self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            else:
                self.dat, self.full_dat = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=self.h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.infer_epsilon, Ne=self.Ne, x0=self.true_x0, K=self.K_prime)   
                TimeSeriesDataUtility.save_time_series(self.dat, self.original_data)
                TimeSeriesDataUtility.save_time_series(self.full_dat, self.full_original_data)
                
        if resume:
            self.mcmc_loader = MCMC_loader(self.out_path)
        # Filter the data for inference
        self.dat_inference = self.dat.loc[self.dat.time <= self.learn_time, ['time', 'K', 'X']]
        self.time_points_inference = self.dat_inference.loc[:, 'time'].unique()
        # Sanity check
        
        if self.K_prime != len(self.dat_inference.loc[:, 'K'].unique()):
            raise ValueError('Inconsistent K')
        
        self.obs_inference = TimeSeriesDataUtility.tall_to_TK(self.dat_inference)
        
        # backward compatibility
        try:
            rejuvenaion_prob=self.rejuvenaion_prob
        except AttributeError:
            self.rejuvenaion_prob = 1
        
    def _test_xprime(self):
        g = InformativeEmission(alpha=[1]*self.K_prime, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = ConditionalWrightFisherDisffusion(h=self.h, K_prime=self.K_prime, K=self.K, Ne=self.Ne)
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h)
        pg_bridge_kernel = BlockedParticleBridgeKernel(K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = BlockedPGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K_prime=self.K_prime, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference), observations=self.obs_inference, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)
        self.T = self.obs_inference['times'].shape[0]
        deltaT = self.obs_inference['times'][self.T-1] - self.obs_inference['times'][0]
        tau = int((round(deltaT/self.h, _TIME_ROUNDING_ACCURACY)))
        x_full_path = gp_sampler.sample_full_path(xprime[0, :], self.h, deltaT, tau)
        
        return(x_full_path)
    
    # The infernce (learning of \theta) logic
    def _infer(self, resume):
        # Setup samplers
        if resume and os.path.isfile('{}.gz'.format(self.prediction_file_path)):
            last_x = MCMC_loader.get_last_infer_x(self.out_path)
            last_theta = MCMC_loader.get_last_infer_theta(self.out_path)
            return([last_theta, last_x])
        
        g = InformativeEmission(alpha=[1]*self.K_prime, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = ConditionalWrightFisherDisffusion(h=self.h, K_prime=self.K_prime, K=self.K, Ne=self.Ne)
        #print('__ERROR!__')
        #print('USING FULL ORIGINAL PATH TO START...')
        #gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=self.full_original_data)
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        pg_bridge_kernel = BlockedParticleBridgeKernel(K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)        
        pgas_sampler = BlockedPGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K_prime=self.K_prime, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference), observations=self.obs_inference, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
        pgas_sampler.particle_processor = Particle_processor(time_points=self.time_points_inference, owner=self)        
        
        print('TODO: generate a valid path...')
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)
        
        #np.save('/Users/sohrabsalehi/Desktop/revive_tests/xprime', xprime)

        #theta_init_sampler = s_uniform_sampler(dim_max=.5, dim_min=-.1, K=self.K_prime)
        theta_init_sampler = s_uniform_sampler(dim_max=.5, dim_min=self.lower_s, K=self.K_prime)
        #adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K_prime), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([-.1]*self.K_prime), upper_bound=np.array([.5]*self.K_prime))
        adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K_prime), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([self.lower_s]*self.K_prime), upper_bound=np.array([self.upper_s]*self.K_prime))
        theta_sampler = MH_Sampler(adapted_proposal_distribution=adapted_theta_proposal, likelihood_distribution=f)
        
        fitness_learner = OutterPGAS(initial_distribution_theta=theta_init_sampler, initial_distribution_x=gp_sampler, observations=self.obs_inference, smoothing_kernel=pgas_sampler, parameter_proposal_kernel=theta_sampler, h=self.h, MCMC_in_Gibbs_nIter=self.MCMC_in_Gibbs_nIter)
        
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
        print('obs are=',obs['value'])
        g = InformativeEmission(alpha=[1]*self.K_prime, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = ConditionalWrightFisherDisffusion(h=self.h, K_prime=self.K_prime, K=self.K, Ne=self.Ne)
        #print('ERROR! GIVING full_original_dat to GP')
        
        #print('__ERROR!__')
        #print('USING FULL ORIGINAL PATH TO START...')
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        
        pg_bridge_kernel = BlockedParticleBridgeKernel(K=self.K, N=self.pf_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = BlockedPGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K_prime=self.K_prime, K=self.K, N=self.pf_n_particles, T=T, observations=obs, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores, disable_ancestor_bridge=self.disable_ancestor_bridge)
        pgas_sampler.particle_processor = Particle_processor(time_points=time_points, owner=self)
        
        fitness_learner = OutterPGAS(initial_distribution_theta=None, initial_distribution_x=gp_sampler, observations=obs, smoothing_kernel=pgas_sampler, parameter_proposal_kernel=None, h=self.h)
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


# In[2]:

class CondExp(ConditionalBayesianLearning):
    
    def _easy_submit(batch_name, mem_per_job=50, n_jobs=None, time_per_job=80*60, ncores=1, k=2, seed=2):
        expt = CondExp()
        expt.the_k = k
        expt.ncpus_per_job = ncores
        expt.the_seed = seed
        n_params = expt.get_n_params()
        print(n_params)
        if n_jobs is None:
            n_jobs = int(n_params)
        batch_params = Experiment.get_batch_params_dict(batch_name=batch_name, mem_per_job=mem_per_job, n_jobs=n_jobs, time_per_job=time_per_job, ncpus_per_job=ncores)
        expt.submit_batch(batch_params=batch_params)

    def get_params(self):
        params = {}
        params['K_prime'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        
        temp_k = params['K_prime'][0]

        params['s'] = [np.random.uniform(-.5, .5, temp_k).tolist()]
        params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+1), 1)[0][0:temp_k]).tolist()]
        print('sum(x0) = {}'.format(np.sum(params['true_x0'])))

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['gp_n_opt_restarts'] = [20]

        # Prediction
        params['disable_ancestor_bridge'] = [True]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0]
        #params['MCMC_in_Gibbs_nIter'] = [10]
        params['MCMC_in_Gibbs_nIter'] = [20]
        
        # S prior
        #params['upper_s'] = [.5]
        #params['lower_s'] = [-.5]
        
        params['upper_s'] = [self.the_upper_s]
        params['lower_s'] = [self.the_lower_s]
        
        #############################
        # Sampler options
        #############################
        params['seed'] = [self.the_seed]
        params['pgas_n_particles'] = [self.the_num_par]
        params['inference_n_iter'] = [self.the_num_itr]
        
        params['pf_n_particles'] = [self.the_num_par]
        params['pf_n_theta'] = [int(self.the_num_itr*.5)]
        
        params['original_data'] = [self.the_original_data]
        params['out_path'] = [self.the_out_path]
        params['learn_time'] = [self.the_learn_time]
        params['end_time'] = [self.the_end_time]
        params['obs_num'] = [self.the_obs_num]
        params['Ne'] = [self.the_Ne]
        params['h'] = [self.the_h]
        params['infer_epsilon'] = [self.the_infer_epsilon]
        params['proposal_step_sigma'] = [[self.the_proposal_step_sigma]*temp_k]
        params['do_predict'] = [self.the_do_predict]
        params['one_step'] = [self.the_one_step]
        
        # Conditional Blocked Gibbs sampler (size of the block update)
        params['K'] = [self.the_block_size] #  How many trajectories are updated at the same time...
        
        # Multinomial error params
        params['multinomial_error'] = [self.the_multinomial_error]
        if self.the_multinomial_error: 
            params['Dir_alpha'] = [[self.the_Dir_alpha]*(temp_k+1)]
            params['Y_sum_total'] = [self.the_Y_sum_total]
        
        return(params)

    def logic(self, resume):
        self._load_data(resume)
        last_theta, last_x = self._infer(resume)
        #print('Saving inference results to\n {}'.format(self.inference_x_file_path))
        #self._test_xprime()
        #return()

        res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        #res_theta = np.array([self.s]*self.pf_n_theta)
        #self._predict(xprime=None, theta_vector=res_theta, resume=resume)

import time

#np.random.seed(100)
#[0.24730082276901072, 0.11404155222466811, 0.12773696187860567],
#expt = CondExp(); expt.the_k = 7; expt.ncpus_per_job = 1; expt._run_default()


# In[ ]:

def read_h_for_Ne(file_path, dat_path):
    print(os.path.expanduser(file_path))
    dat = TimeSeriesDataUtility.read_time_series(os.path.expanduser(file_path))
    print(dat)
    
    base = [os.path.splitext(os.path.basename(x))[0] for x in dat.file_name.values]
    print('{}{}'.format('base', base))
    print('{}{}'.format('dat_path', dat_path))
    print(dat.h[[os.path.basename(dat_path) == t for t in base]])
    return(float(dat.h[[os.path.basename(dat_path) == t for t in base]]))
    
def read_last_time_point(file_path, pred_index=0):
    #file_path = '/Users/sohrabsalehi/projects/fitclone/figures/raw/supp/SA532_dropped_one_sc_dlp_6_clones.tsv.gz'
    dat = TimeSeriesDataUtility.read_time_series(file_path)
    return([np.asscalar(dat.time.max()), np.asscalar(dat.time.unique()[-(1+pred_index)]), len(dat.K.unique()), len(dat.time.unique())])

def run_model_comparison_ne(options, batch_path):
    import time
    #np.random.seed(some_seed)    
    expt = CondExp(); 

    for k,v in options.items():
        setattr(expt, k, v)
    
    expt.the_original_data = options['the_original_data']
    expt.the_h = float(options['the_h'])
    
    end_point, pred_point, expt.the_k, expt.the_obs_num = read_last_time_point(expt.the_original_data, int(options['the_do_predict']))
    
    # Default values
    if options['the_do_predict'] == 0:
        expt.the_learn_time = end_point
    else:
        expt.the_learn_time = pred_point
        
    if 'the_one_step' in options:
        expt.the_one_step = options['the_one_step']
    else:
        expt.the_one_step = False
    
    
    if 'the_seed' in options:
        expt.the_seed = options['the_seed']
    else:
        expt.the_seed = 100
    

    if 'the_upper_s' in options:
        expt.the_upper_s = options['the_upper_s']
    else:
        expt.the_upper_s = 10
        
        
    if 'the_lower_s' in options:
        expt.the_lower_s = options['the_lower_s']
    else:
        expt.the_lower_s = -10
        
        
    expt.the_end_time = end_point
    expt.the_do_predict = int(options['the_do_predict'])
    expt.ncpus_per_job = 1
    
    expt.the_multinomial_error = False
    if 'the_multinomial_error' in options:
        expt.the_multinomial_error = True
        expt.the_Dir_alpha = options['the_dir_alpha']
        expt.the_Y_sum_total = options['the_Y_sum_total']
            
    expt.the_out_path = options['out_path']
                                
    tt = expt._run_default()

