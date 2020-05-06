
# coding: utf-8

# # Conditional experiments with repeat
# The setup is identical to the normal setup except that biological repeats are allowed. 
# This will affect the likelihood the MH step, where the product of repeats is used. 

# In[ ]:

import os
exec(open(os.path.expanduser('experiments-prediction.py')).read())
exec(open(os.path.expanduser('extended_wf_model.py')).read())

class HierarchicalConditionalBayesianLearning(Bayesian_learning_exp):
    def _load_data(self, resume):
        for m in range(self.M_data):
            random.seed(self.seed[m])
            
        self.dat = [None]*self.M_data
        self.full_dat = [None]*self.M_data
        self.dat_inference = [None]*self.M_data
        self.time_points_inference = [None]*self.M_data
        self.obs_inference = [None]*self.M_data
        for m in range(self.M_data):                
            try:
                # Does self.original_data exist?
                self.dat[m] = TimeSeriesDataUtility.read_time_series(self.original_data[m])
            except (AttributeError, TypeError) as e:
                self.original_data = [None]*self.M_data
                self.full_original_data = [None]*self.M_data
                self.original_data[m] = os.path.join(self.out_path, 'sample_data_{}.tsv'.format(m))
                self.full_original_data[m] = os.path.join(self.out_path, 'sample_data_full_{}.tsv'.format(m))
                if resume:
                    self.dat[m] = TimeSeriesDataUtility.read_time_series(self.original_data[m])
                else:
                    print('self.true_x0[m] is ', self.true_x0[m])
                    self.dat[m], self.full_dat[m] = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=self.h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.epsilon, Ne=self.Ne, x0=self.true_x0[m], K=self.K_prime)   
                    print('self.dat[m] is ', self.dat[m])
                    TimeSeriesDataUtility.save_time_series(self.dat[m], self.original_data[m])
                    TimeSeriesDataUtility.save_time_series(self.full_dat[m], self.full_original_data[m])
            self.configs['original_data'] = self.original_data
            self.configs['full_original_data'] = self.full_original_data
            if resume:
                print('WARNING! Resume not supported!!!')
                self.mcmc_loader = MCMC_loader(self.out_path)
            # Filter the data for inference
            self.dat_inference[m] = self.dat[m].loc[self.dat[m].time <= self.learn_time, ['time', 'K', 'X']]
            self.time_points_inference[m] = self.dat_inference[m].loc[:, 'time'].unique()
            
            # Sanity check
            if self.K_prime != len(self.dat_inference[m].loc[:, 'K'].unique()):
                raise ValueError('Inconsistent K'.format(self.K_prime, en(self.dat_inference[m].loc[:, 'K'].unique())))
            
            self.obs_inference[m] = TimeSeriesDataUtility.tall_to_TK(self.dat_inference[m])
        
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
        
        g = InformativeEmission(alpha=[1]*self.K_prime, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = ConditionalWrightFisherDisffusion(h=self.h, K_prime=self.K_prime, K=self.K, Ne=self.Ne)
        
        gp_sampler = [None]*self.M_data
        pg_bridge_kernel = [None]*self.M_data
        pgas_sampler = [None]*self.M_data
        xprime = [None]*self.M_data
        sample_processors = [None]*self.M_data
        
        for m in range(self.M_data):
            gp_sampler[m] = GP_sampler(dat=self.obs_inference[m]['value'], obsTimes=self.time_points_inference[m], epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h)
            pg_bridge_kernel[m] = BlockedParticleBridgeKernel(h=self.h, K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler[m], transitionDistribution=f, n_cores=self.bridge_n_cores)
            pgas_sampler[m] = BlockedPGAS(bridgeKernel=pg_bridge_kernel[m], emissionDistribution=g, proposalDistribution=g, K_prime=self.K_prime, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference[m]), observations=self.obs_inference[m], transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
            xprime[m] = pgas_sampler[m].generate_dummy_trajectory(self.learn_time)

        theta_init_sampler = s_uniform_sampler(dim_max=.5, dim_min=-.1, K=self.K_prime)
        adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K_prime), sigma=np.array(self.proposal_step_sigma), lower_bound=np.array([-.1]*self.K_prime), upper_bound=np.array([.5]*self.K_prime))
        theta_sampler = MH_Sampler(adapted_proposal_distribution=adapted_theta_proposal, likelihood_distribution=f)
        fitness_learner = HierarchicalOuterPGAS(initial_distribution_theta=theta_init_sampler, initial_distribution_list_x=gp_sampler, observations_list=self.obs_inference, smoothing_kernels=pgas_sampler, 
                                     parameter_proposal_kernel=theta_sampler, h=self.h, MCMC_in_Gibbs_nIter=self.MCMC_in_Gibbs_nIter)
        for m in range(self.M_data):
            sample_processors[m] = Sample_processor(self.time_points_inference, self)
        fitness_learner.sample_processors = sample_processors
        
        is_resume = resume and os.path.isfile('{}.gz'.format(self.inference_x_file_path))
        if is_resume:
            fitness_learner.data_loader = self.mcmc_loader
                
        return(fitness_learner.sample_hierarchical(self.inference_n_iter, xprime, is_resume=is_resume))

        
    # The prediction logic
    def _predict(self, xprime, theta_vector, resume):
        time_points = self.dat.loc[:, 'time'].unique()
        T = len(time_points)
        obs = TimeSeriesDataUtility.tall_to_TK(self.dat)
        print('obs are=',obs['value'])
        g = InformativeEmission(alpha=[1]*self.K_prime, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = ConditionalWrightFisherDisffusion(h=self.h, K_prime=self.K_prime, K=self.K, Ne=self.Ne)
        #print('ERROR! GIVING full_original_dat to GP')
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        
        pg_bridge_kernel = BlockedParticleBridgeKernel(h=self.h, K=self.K, N=self.pf_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = BlockedPGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K_prime=self.K_prime, K=self.K, N=self.pf_n_particles, T=T, observations=obs, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores, disable_ancestor_bridge=self.disable_ancestor_bridge)
        
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
        
    
class HieCondExp(HierarchicalConditionalBayesianLearning):
    
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
    
    
    def generate_starting_value(self, k, count=1):
        res = [None]*count
        for i in range(count):
            res[i] = (np.random.dirichlet([1]*(k+1), 1)[0][0:k]).tolist()
        return(res)
    
    def get_params(self):
        params = {}
        params['Ne'] = [500]
        params['seed'] = [[10, 20, 30]]
        params['h'] = [.001]
        params['epsilon'] = [.01]
        params['K_prime'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        params['K'] = 2
        temp_k = params['K_prime'][0]
        params['M_data'] = [self.M_data]

        # Data
        params['s'] = [np.random.uniform(0, .5, temp_k).tolist()]
        #params['s'] = [[.1]*temp_k]
        params['obs_num'] = [5]
        params['end_time'] = [.05]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['learn_time'] = [.05]
        #print('StARTING with EXtrA ROOM!')
        params['true_x0'] = [self.generate_starting_value(temp_k, self.M_data)]
        print(params['true_x0'])
        print(len(params['true_x0']))
        #params['true_x0'] = [[.2]*temp_k]
        for m in range(self.M_data):
            print('sum(x0) = {}'.format(np.sum(params['true_x0'][0][m])))
        params['gp_n_opt_restarts'] = [1]

        ## Sampler options
        params['pgas_n_particles'] = [10]
        params['inference_n_iter'] = [200]

        # sampler
        params['proposal_step_sigma'] = [[.1]*temp_k]

        # Prediction
        params['disable_ancestor_bridge'] = [True]
        params['infer_epsilon'] = [.01]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0]
        params['pf_n_particles'] = [1000]
        params['pf_n_theta'] = [50]
        params['MCMC_in_Gibbs_nIter'] = [5]
        params['out_path'] = ['~/Desktop/pgas_sanity_check/exp']
        return(params)


    def logic(self, resume):
        self._load_data(resume)
        last_theta, last_x = self._infer(resume)
        #print('Saving inference results to\n {}'.format(self.inference_x_file_path))
        #self._test_xprime()
        #return()

        #res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        #res_theta = np.array([self.s]*self.pf_n_theta)
        #self._predict(xprime=None, theta_vector=res_theta, resume=resume)

import time
np.random.seed(50)
expt = HieCondExp(); expt.M_data = 2; expt.the_k = 3; expt.ncpus_per_job = 1; expt._run_default()


# In[ ]:




# In[ ]:



