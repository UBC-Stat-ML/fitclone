
# coding: utf-8

# In[27]:

import os
import subprocess
import time
exec(open(os.path.expanduser('scalable_computing.py')).read())
exec(open(os.path.expanduser('pgas-dir.py')).read())
#exec(open(os.path.expanduser('experiments-prediction.py')).read())

# Standard Prediction/Inference experiment
class Bayesian_learning_exp(Experiment):
    def __init__(self):
        super().__init__()
        self.dat = None
        self.dat_inference = None
        self.time_points_inference = None
        self.obs_inference = None
     
    
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
    
    # The data loading logic
    def _load_data(self, resume):
        #print('self.K is {}'.format(self.K))
        random.seed(self.seed)
        try:
            # Does self.original_data exist?
            self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
        except AttributeError:
            self.original_data = os.path.join(self.out_path, 'sample_data.tsv')
            self.full_original_data = os.path.join(self.out_path, 'sample_data_full.tsv')
            self.configs['original_data'] = self.original_data
            self.configs['full_original_data'] = self.full_original_data
            if resume:
                self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            else:
                #print(self.K)
                #print('self.K is {}'.format(self.K))
                #self.dat = WrightFisherDiffusion.generate_sample_data(silent=True, h=self.h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.epsilon, Ne=self.Ne, x0=self.true_x0, K=self.K)   
                self.dat, self.full_dat = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=self.h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.epsilon, Ne=self.Ne, x0=self.true_x0, K=self.K)   
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
        f = WrightFisherDiffusion(h=self.h, K=self.K, Ne=self.Ne)
        #print('ERROR! PROVIDING TRUE X as X0')
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        
        pg_bridge_kernel = ParticleBridgeKernel(h=self.h, K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = PGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference), observations=self.obs_inference, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)

        theta_init_sampler = s_uniform_sampler(dim_max=.5, dim_min=.01, K=self.K)
        adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([0.01]*self.K), upper_bound=np.array([.5]*self.K))
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
        
        #print('Testing prediction blindness')
        #obs['value'][obs['times'] > self.learn_time, ] = .876
        #print('obs[values] ={}'.format(obs['value']))
        
        g = InformativeEmission(alpha=[1]*self.K, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = WrightFisherDiffusion(h=self.h, K=self.K, Ne=self.Ne)
        #print('ERROR! PROVIDING TRUE X as X0')
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        
        pg_bridge_kernel = ParticleBridgeKernel(h=self.h, K=self.K, N=self.pf_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = PGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K=self.K, N=self.pf_n_particles, T=T, observations=obs, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores, disable_ancestor_bridge=self.disable_ancestor_bridge)
        
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
        
class TimingExp(Bayesian_learning_exp):
    
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
    

    def get_params(self):
        params = {}
        params['Ne'] = [200]
        params['seed'] = [i*10 for i in range(5)]
        params['h'] = [.001]
        params['epsilon'] = [.01]
        params['K'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        temp_k = params['K'][0]

        # Data
        params['s'] = [np.random.uniform(0, .5, temp_k).tolist()]
        params['obs_num'] = [4]
        params['end_time'] = [0.07]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['learn_time'] = [.06]
        params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+1), 1)[0][0:temp_k]).tolist()]

        params['gp_n_opt_restarts'] = [20]

        ## Sampler options
        params['pgas_n_particles'] = [500]
        params['inference_n_iter'] = [100]

        # sampler
        params['proposal_step_sigma'] = [[.1]*temp_k]

        # Prediction
        params['disable_ancestor_bridge'] = [True]
        params['infer_epsilon'] = [.02]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0.0]
        params['pf_n_particles'] = [500]
        params['pf_n_theta'] = [50]
        params['MCMC_in_Gibbs_nIter'] = [10]
        params['out_path'] = ['~/Desktop/crispr/exp']
        #params['original_data'] = ['~/Desktop/crispr/CX23_200.tsv']
        params['original_data'] = ['/Users/sohrabsalehi/Desktop/CX23_200.tsv']
        return(params)

    
    def logic(self, resume):
        self._load_data(resume)
        last_theta, last_x = self._infer(resume)
        #print('Saving inference results to\n {}'.format(self.inference_x_file_path))
        
        res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        #res_theta = np.array([self.s]*self.pf_n_theta)
        self._predict(xprime=None, theta_vector=res_theta, resume=resume)

import time
#TimingExp()._run_default()

#TimingExp().resume_exp('/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201707-16-145503.332905')
#np.random.seed(2)
#expt = TimingExp(); expt.the_k = 10; expt.ncpus_per_job = 8; tt = expt._run_default()
#expt = TimingExp(); expt.the_k = 10; expt.ncpus_per_job = 8; tt = expt._run_default()


# In[28]:

expt = TimingExp(); expt.the_k = 5; expt.ncpus_per_job = 1; tt = expt._run_default()


# In[14]:

TimeSeriesDataUtility.read_time_series('/Users/sohrabsalehi/Desktop/crispr/CX23_200 3.tsv')


# In[ ]:



