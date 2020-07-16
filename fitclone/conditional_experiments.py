import os
import time
import numpy as np
from random import random
import yaml

#exec(open('full_exp.py').read())
#exec(open('extended_wf_model.py').read())


from Utilities import MCMC_loader
from Models import WrightFisherDiffusion, InformativeEmission, _TIME_ROUNDING_ACCURACY
from extended_wf_model import ConditionalWrightFisherDisffusion
from pgas_dir import GP_sampler, SUniform_Sampler, MH_Sampler, RandomWalk_Proposal, OutterPGAS
from pgas import BlockedParticleBridgeKernel, BlockedPGAS
from scalable_computing import Particle_processor, Sample_processor


from conditional_experiments import BayesianLearningExp
from Utilities import TimeSeriesDataUtility


class ConditionalBayesianLearning(BayesianLearningExp):

    def _compute_one_step_h(self, time_points):
        h = np.empty([len(time_points), 1])
        for time in range(1, len(time_points)+1):
            h[time] = time_points[time] - time_points[time-1]
        return(h)
    
    def _load_data(self, resume):
        print('In _load_data')
        # Set one-step h
        self.one_step = False
        if self.one_step != False:
            self.h = self._compute_one_step_h(self.time_points_inference)        
        else:
            self.h = np.repeat(self.h, self.obs_num)
        
        print('self.h is {}'.format(self.h))
        random.seed(self.seed)
        try:
            # Does self.original_data exist?
            self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            old_configs_path = os.path.dirname(self.original_data)
            old_configs_path = os.path.join(old_configs_path, 'config.yaml')
            print(old_configs_path)
            stream = open(old_configs_path, "r")
            doc = yaml.load(stream)
            self.configs['s'] = doc['s']
            self.configs['true_x0'] = doc['true_x0']
            self.configs['full_original_data'] = doc['full_original_data']
            
            self.s = self.configs['s']
            self.full_original_data = self.configs['full_original_data']
            self.true_x0 = self.configs['true_x0']
            
            
        except AttributeError:
            self.original_data = os.path.join(self.out_path, 'sample_data.tsv')
            self.full_original_data = os.path.join(self.out_path, 'sample_data_full.tsv')
            self.configs['original_data'] = self.original_data
            self.configs['full_original_data'] = self.full_original_data
            if resume:
                self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            else:
                self.dat, self.full_dat = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=self.h, selectionCoefficients=self.s, obs_num=self.obs_num, end_time=self.end_time, epsilon=self.epsilon, Ne=self.Ne, x0=self.true_x0, K=self.K_prime)   
                TimeSeriesDataUtility.save_time_series(self.dat, self.original_data)
                TimeSeriesDataUtility.save_time_series(self.full_dat, self.full_original_data)
                
        if resume:
            self.mcmc_loader = MCMC_loader(self.out_path)
        # Filter the data for inference
        self.dat_inference = self.dat.loc[self.dat.time <= self.learn_time, ['time', 'K', 'X']]
        self.time_points_inference = self.dat_inference.loc[:, 'time'].unique()
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
        gp_sampler = GP_sampler(dat=self.obs_inference['value'], obsTimes=self.time_points_inference, epsilon=self.gp_epsilon, nOptRestarts=self.gp_n_opt_restarts, h=self.h, full_original_dat=None)
        pg_bridge_kernel = BlockedParticleBridgeKernel(K=self.K, N=self.pgas_n_particles, proposalDistribution=gp_sampler, transitionDistribution=f, n_cores=self.bridge_n_cores)        
        pgas_sampler = BlockedPGAS(bridgeKernel=pg_bridge_kernel, emissionDistribution=g, proposalDistribution=g, K_prime=self.K_prime, K=self.K, N=self.pgas_n_particles, T=len(self.time_points_inference), observations=self.obs_inference, transitionDistribution=f, h=self.h, rejuvenaion_prob=self.rejuvenaion_prob, n_cores=self.n_cores)
        pgas_sampler.particle_processor = Particle_processor(time_points=self.time_points_inference, owner=self)                
        xprime = pgas_sampler.generate_dummy_trajectory(self.learn_time)
        lower_s = -.5
        theta_init_sampler = SUniform_Sampler(dim_max=.5, dim_min=lower_s, K=self.K_prime)
        adapted_theta_proposal = RandomWalk_Proposal(mu=np.array([0]*self.K_prime), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([lower_s]*self.K_prime), upper_bound=np.array([.5]*self.K_prime))
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
        g = InformativeEmission(alpha=[1]*self.K_prime, epsilon=self.infer_epsilon, b=self.infer_epsilon_tolerance)
        f = ConditionalWrightFisherDisffusion(h=self.h, K_prime=self.K_prime, K=self.K, Ne=self.Ne)
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
        self._predict(xprime=None, theta_vector=res_theta, resume=resume)
        
    
class CondExp(ConditionalBayesianLearning):
    import time
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
        params['Ne'] = [500]
        params['seed'] = [10]
        params['h'] = [.01]
        params['epsilon'] = [.01]
        params['K_prime'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        params['K'] = 2
        temp_k = params['K_prime'][0]

        # Data
        params['s'] = [np.random.uniform(-.5, .5, temp_k).tolist()]
        params['obs_num'] = [6]
        params['end_time'] = [.06]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['learn_time'] = [.05]
        params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+1), 1)[0][0:temp_k]).tolist()]
        print('sum(x0) = {}'.format(np.sum(params['true_x0'])))
        params['gp_n_opt_restarts'] = [20]

        ## Sampler options
        params['pgas_n_particles'] = [10000]
        params['inference_n_iter'] = [1000]

        # sampler
        params['proposal_step_sigma'] = [[.1]*temp_k]

        # Prediction
        params['disable_ancestor_bridge'] = [True]
        params['infer_epsilon'] = [.01]
        params['rejuvenaion_prob'] = [1]
        params['infer_epsilon_tolerance'] = [0]
        params['pf_n_particles'] = [10000]
        params['pf_n_theta'] = [500]
        params['MCMC_in_Gibbs_nIter'] = [10]
        params['out_path'] = ['~/Desktop/pgas_sanity_check/exp']        
      
        return(params)

    def logic(self, resume):
        self._load_data(resume)
        last_theta, last_x = self._infer(resume)
        res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values

        self._predict(xprime=None, theta_vector=res_theta, resume=resume)





