import os
import subprocess
import time

exec(open('scalable_computing.py').read())
exec(open('pgas-dir.py').read())

# Standard Prediction/Inference experiment
class Bayesian_learning_exp(Experiment):
    def __init__(self):
        super().__init__()
        self.dat = None
        self.dat_inference = None
        self.time_points_inference = None
        self.obs_inference = None
        self.Y_sum_total = None
        self.Dir_alpha = None # DirMult concentration parameter (just for the Dir part) 

    def _post_process(self):
        try:
            absolute_path = ''
            try:
                env = os.environ['HOST']
            except:
                env = 'AZURECN'
            if env == '' or env is None: env = 'AZURECN'
            if env == 'local': absolute_path = ''
            elif env == 'MOMAC39': absolute_path = ''
            elif env == 'grex': absolute_path = ''
            elif env == 'shahlab': absolute_path = '/gsc/software/linux-x86_64-centos6/R-3.2.3/bin/'
            
            if env == 'AZURECN':
                print('Summarising for the AZURECN node...')
                try:
                    shared_dir = os.path.join(os.environ['AZ_BATCH_NODE_STARTUP_DIR'], 'wd')
                except:
                    shared_dir = '/mnt/batch/tasks/startup/wd'

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
        import time
        self.config_file = os.path.expanduser(config_file)
        stream = open(self.config_file, "r")
        doc = yaml.load(stream)
        print(doc)
        return(self.run(doc))
    
    def _compute_one_step_h(self, time_points):
        h = np.empty([len(time_points), 1])
        for time in range(1, len(time_points)+1):
            h[time] = time_points[time] - time_points[time-1]
        return(h)
        
    
    # The data loading logic
    def _load_data(self, resume):
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
                self.dat, self.full_dat = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=self.h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.infer_epsilon, Ne=self.Ne, x0=self.true_x0, K=self.K)   
                TimeSeriesDataUtility.save_time_series(self.dat, self.original_data)
                TimeSeriesDataUtility.save_time_series(self.full_dat, self.full_original_data)

        if resume:
            self.mcmc_loader = MCMC_loader(self.out_path)
        # Filter the data for inference
        self.dat_inference = self.dat.loc[self.dat.time <= self.learn_time, ['time', 'K', 'X']]
        self.time_points_inference = self.dat_inference.loc[:, 'time'].unique()
        
        print(self.time_points_inference)
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
        # Set one-step h
        if self.one_step != False:
            self.h = self._compute_one_step_h(self.time_points_inference)        
        else:
            self.h = np.repeat(self.h, len(self.time_points_inference))
        
        # Setup samplers
        print('self.h is {}'.format(self.h))
        if resume and os.path.isfile('{}.gz'.format(self.prediction_file_path)):
            last_x = MCMC_loader.get_last_infer_x(self.out_path)
            last_theta = MCMC_loader.get_last_infer_theta(self.out_path)
            return([last_theta, last_x])
        
        print('self.multinomial_error = {}'.format(self.multinomial_error))
        
        if self.multinomial_error == True:
            print('Using Dirichlet Emission!')
            g = DirMultEmission(alpha=self.Dir_alpha, N_total=self.Y_sum_total)
        else:
            g = InformativeEmission(alpha=[1]*self.K, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)

        f = WrightFisherDiffusion(h=self.h, K=self.K, Ne=self.Ne)
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        pg_bridge_kernel = ParticleBridgeKernel(K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
        pgas_sampler = PGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference), observations=self.obs_inference, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
        
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)
        upper_s = 10
        lower_s = -10
        theta_init_sampler = s_uniform_sampler(dim_max=upper_s, dim_min=lower_s, K=self.K)

        adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([lower_s]*self.K), upper_bound=np.array([upper_s]*self.K))
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
        
        # Set one-step h
        if self.one_step != False:
            self.h = self._compute_one_step_h(time_points)        
        else:
            self.h = np.repeat(self.h, T)
        
        # setup the emissiom model
        if self.multinomial_error == True:
            g = DirMultEmission(alpha=self.Dir_alpha, N_total=self.Y_sum_total)
        else:
            g = InformativeEmission(alpha=[1]*self.K, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)

        f = WrightFisherDiffusion(h=self.h, K=self.K, Ne=self.Ne)
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        pg_bridge_kernel = ParticleBridgeKernel(K=self.K, N=self.pf_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)
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

        self._predict(xprime=None, theta_vector=res_theta, resume=resume)
        
class TimingExp(Bayesian_learning_exp):
    
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
        
        params['seed'] = [100]
        params['K'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        temp_k = params['K'][0]

        # Data
        #params['s'] = [np.random.uniform(-1.0, .5, temp_k).tolist()]
        print('Start s from a different value..')
        #params['s'] = [[0.0]*temp_k]
        params['s'] = [np.random.uniform(-.5, .5, temp_k).tolist()]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+1), 1)[0][0:temp_k]).tolist()]
        params['gp_n_opt_restarts'] = [50]

        # Prediction
        params['disable_ancestor_bridge'] = [True]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0.0]
        
        params['MCMC_in_Gibbs_nIter'] = [20]

        ## Sampler options
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
        params['blocked_gibbs'] = [self.the_blocked_gibbs]
        
        # Multinomial error params
        params['multinomial_error'] = [self.the_multinomial_error]
        if self.the_multinomial_error: 
            params['Dir_alpha'] = [[self.the_Dir_alpha]*(temp_k+1)]
            params['Y_sum_total'] = [self.the_Y_sum_total]

        return(params)

    
    def logic(self, resume):
        self._load_data(resume)
        last_theta, last_x = self._infer(resume)        
        res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        self._predict(xprime=None, theta_vector=res_theta, resume=resume)


def read_h_for_Ne(file_path, dat_path):
    print(os.path.expanduser(file_path))
    dat = TimeSeriesDataUtility.read_time_series(os.path.expanduser(file_path))
    print(dat)
    
    base = [os.path.splitext(os.path.basename(x))[0] for x in dat.file_name.values]
    return(float(dat.h[[os.path.basename(dat_path) == t for t in base]]))
    
def read_last_time_point(file_path, pred_index=0):
    dat = TimeSeriesDataUtility.read_time_series(file_path)
    return([np.asscalar(dat.time.max()), np.asscalar(dat.time.unique()[-(1+pred_index)]), len(dat.K.unique()), len(dat.time.unique())])

def run_model_comparison_ne(options, batch_path):
    import time
    #np.random.seed(some_seed)    
    expt = TimingExp(); 

    for k,v in options.items():
        setattr(expt, k, v)
    
    expt.the_original_data = '{}_{}.tsv'.format(options['the_original_data'], options['the_Ne'])
    
    # Add the_h_config
    options['the_h_config'] = os.path.join(os.path.dirname(options['the_original_data']), 'h_config.tsv')
    expt.the_h = read_h_for_Ne(options['the_h_config'], expt.the_original_data)
    end_point, pred_point, expt.the_k, expt.the_obs_num = read_last_time_point(expt.the_original_data, int(options['the_do_predict']))
    
    if options['the_do_predict'] == 0:
        expt.the_learn_time = end_point
    else:
        expt.the_learn_time = pred_point
    
    
    if 'the_one_step' in options:
        expt.the_one_step = options['the_one_step']
    else:
        expt.the_one_step = False
        
    # Add the blocked version
    if 'the_blocked_gibbs' in options:
        expt.the_blocked_gibbs = options['the_blocked_gibbs']
    else:
        expt.the_blocked_gibbs = False
    
    
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
