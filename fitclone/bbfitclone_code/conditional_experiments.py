
# coding: utf-8

# In[38]:

import os

exec(open(os.path.expanduser('SC_DLP_sa501_exp.py')).read())
exec(open(os.path.expanduser('extended_wf_model.py')).read())
#exec(open(os.path.expanduser('conditional_experiments.py')).read())

#class ConditionalBayesianLearning(BayesianLearningExp):
class ConditionalBayesianLearning(Bayesian_learning_exp):

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
            #self.h = np.repeat(self.h, len(self.time_points_inference))
            self.h = np.repeat(self.h, self.obs_num)
        
        print('self.h is {}'.format(self.h))
        random.seed(self.seed)
        try:
            # Does self.original_data exist?
            self.dat = TimeSeriesDataUtility.read_time_series(self.original_data)
            # Load trueX and S from there too
            # 1. 
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

        #theta_init_sampler = s_uniform_sampler(dim_max=.5, dim_min=-.1, K=self.K_prime)
        lower_s = -.5
        theta_init_sampler = s_uniform_sampler(dim_max=.5, dim_min=lower_s, K=self.K_prime)
        #adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K_prime), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([-.1]*self.K_prime), upper_bound=np.array([.5]*self.K_prime))
        adapted_theta_proposal = Random_walk_proposal(mu=np.array([0]*self.K_prime), sigma=np.array(self.proposal_step_sigma), lower_bound = np.array([lower_s]*self.K_prime), upper_bound=np.array([.5]*self.K_prime))
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
        params['Ne'] = [500]
        #params['seed'] = [i*10 for i in range(5)]
        params['seed'] = [10]
        params['h'] = [.01]
        params['epsilon'] = [.01]
        params['K_prime'] = [self.the_k] # set one less than the nonminal K (e.g. K=2 means 3 alleles )
        params['K'] = 2
        temp_k = params['K_prime'][0]

        # Data
        #params['s'] = [np.random.uniform(0, .5, temp_k).tolist()]
        params['s'] = [np.random.uniform(-.5, .5, temp_k).tolist()]
        #params['s'] = [[.1]*temp_k]
        params['obs_num'] = [6]
        params['end_time'] = [.06]

        # Parallelisation
        params['n_cores'] = [self.ncpus_per_job]
        params['bridge_n_cores'] = [self.ncpus_per_job]

        # Inference
        params['gp_epsilon'] = [.005]
        params['learn_time'] = [.05]
        #print('StARTING with EXtrA ROOM!')
        params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+1), 1)[0][0:temp_k]).tolist()]
        #params['true_x0'] = [(np.random.dirichlet([1]*(temp_k+2), 1)[0][0:temp_k]).tolist()]
        #params['true_x0'] = [[.1, .2, .3]]
        #params['true_x0'] = [[0.82248785,  0.        ,  0.        ,  0.12918921,  0.02075464]]
        #params['true_x0'] = [[.1]*temp_k]
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
        #params['original_data'] = ['/Users/ssalehi/Desktop/pgas_sanity_check/exp_3KKJH_201710-25-171934.409470/sample_data.tsv.gz']
        #params['original_data'] = ['/Users/sohrabsalehi/Desktop/pgas_sanity_check/K=4,single_data_seed=50_exp_Y0CQ6_201710-26-184727.911479/sample_data.tsv.gz'] 
        #params['original_data'] = ['/Users/sohrabsalehi/Desktop/pgas_sanity_check/exp_Y0CQ6_201710-26-183929.707763/sample_data.tsv.gz']         
        
      
        return(params)

    def logic(self, resume):
        self._load_data(resume)
        last_theta, last_x = self._infer(resume)
        #print('Saving inference results to\n {}'.format(self.inference_x_file_path))
        #self._test_xprime()
        #return()

        res_theta = TimeSeriesDataUtility.read_time_series(self.inference_theta_file_path).values
        #res_theta = np.array([self.s]*self.pf_n_theta)
        self._predict(xprime=None, theta_vector=res_theta, resume=resume)

import time

#np.random.seed(100)
#[0.24730082276901072, 0.11404155222466811, 0.12773696187860567],
#expt = CondExp(); expt.the_k = 7; expt.ncpus_per_job = 1; expt._run_default()

#np.random.seed(30)
#[0.24730082276901072, 0.11404155222466811, 0.12773696187860567],
#expt = CondExp(); expt.the_k = 4; expt.ncpus_per_job = 1; expt._run_default()


# In[ ]:




# In[16]:




# In[ ]:




# In[ ]:

# Cashed with path check 100/100 --- OutterPGAS --- 0:04:09.
# Dumb with path check 100/100 --- OutterPGAS --- 0:06:27.
# Cashed withOUT path check 100/100 --- OutterPGAS --- 0:03:41.


# In[3]:

# On iteration 100/100 --- OutterPGAS --- 0:03:41.


# In[15]:




# In[348]:

import os
exec(open(os.path.expanduser('experiments-prediction.py')).read())
exec(open(os.path.expanduser('extended_wf_model.py')).read())
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.ticker as mticker


# In[355]:

the_seed = 10
random.seed(the_seed)
np.random.seed(the_seed)
K_prime = 4
K = 2
gp_epsilon = .005
gp_n_opt_restarts = 10
h = .001
#s = [.1]*K_prime
s = [np.random.uniform(0, .5, K_prime).tolist()]
s = np.array(s).reshape(-1, )
#true_x0 = [.1]*K_prime
true_x0 = [(np.random.dirichlet([1]*(K_prime+1), 1)[0][0:K_prime]).tolist()]
true_x0 = np.asarray(true_x0).reshape(-1, )


obs_num = 5
end_time = .05
learn_time = .05
epsilon = .01
Ne = 500
out_path = os.path.expanduser('~/Desktop/tgp')
original_data = os.path.join(out_path, 'sample_data.tsv')
full_original_data = os.path.join(out_path, 'sample_data_full.tsv')

dat, full_dat = WrightFisherDiffusion.generate_full_sample_data(silent=True, h=h, selectionCoefficients=s, obs_num=obs_num, end_time=end_time, epsilon=epsilon, Ne=Ne, x0=true_x0, K=K_prime)   
TimeSeriesDataUtility.save_time_series(dat, original_data)
TimeSeriesDataUtility.save_time_series(full_dat, full_original_data)

# Filter the data for inference
dat_inference = dat.loc[dat.time <= learn_time, ['time', 'K', 'X']]
time_points_inference = dat_inference.loc[:, 'time'].unique()
obs_inference = TimeSeriesDataUtility.tall_to_TK(dat_inference)


# In[356]:

# What is wrong with the starting values?
gp_sampler = GP_sampler(dat=obs_inference['value'], obsTimes=time_points_inference, epsilon=gp_epsilon, nOptRestarts=gp_n_opt_restarts, h=h, full_original_dat=None)
print(obs_inference['value'])
time_length = learn_time
tau = len(full_dat['time'].unique())
x0 = obs_inference['value'][0, :]


# In[357]:

def sample_full_path(x0, h, time_length, tau, gp_sampler):
    Xi = np.empty([tau, x0.shape[0]])
    Xi[0, ] = x0
    time_mesh = np.linspace(0, time_length, tau)
    print('taus is ', tau)
    for k in range(gp_sampler.K):
        means, stds = gp_sampler.gps[k].predict(time_mesh.reshape(-1,1), return_std=True)
        #Xi[:, k] = sp.stats.norm.rvs(loc=means.reshape(tau), scale=stds).reshape(tau)
        Xi[:, k] = means[:, 0]
        #Xi[Xi[:, k] < 0, k] = -Xi[Xi[:, k] < 0, k]
        #Xi[Xi[:, k] < 0, k] = 0
    for t in range(tau):
        if np.sum(Xi[t, ]) > 1:
            pass
            #Xi[t,] /= np.sum(Xi[t,])            
    return(Xi)


# In[358]:

gp_full_path = sample_full_path(x0=x0, h=h, time_length=time_length, tau=tau, gp_sampler=gp_sampler)


# In[359]:

get_ipython().magic('matplotlib inline')
TimeSeriesDataUtility.plot_TK(gp_full_path, legend='', title='')


# In[354]:

TimeSeriesDataUtility.plot_tall(full_dat, legend='', title='Real value')


# In[1]:

#gp_full_path


# In[160]:

Xi = np.empty([tau, x0.shape[0]])
Xi[0, ] = x0
time_mesh = np.linspace(0, time_length, tau)
means, stds = gp_sampler.gps[0].predict(time_mesh.reshape(-1,1), return_std=True)
print(means)
print(stds)
Xi[:, 0] = sp.stats.norm.rvs(loc=means.reshape(tau), scale=stds).reshape(tau)
Xi[Xi[:, 0] < 0, ] = -Xi[Xi[:, 0] < 0, ]
TimeSeriesDataUtility.plot_TK(Xi, legend='', title='', xlabels=time_mesh)


# In[161]:

# What if using our own sample?
xx = np.empty([tau, K_prime])
xx[0, ] = x0
for i in range(1, tau):
    xx[i, :] = gp_sampler.sample(deltaT=h, x=xx[i-1, ])
xx
TimeSeriesDataUtility.plot_TK(xx, legend='', title='', xlabels=time_mesh)


# In[159]:




# In[218]:

#TimeSeriesDataUtility.plot_tall(full_dat.loc[full_dat.K == 0, ], legend='', title='')


# In[85]:

TimeSeriesDataUtility.plot_tall(dat.loc[dat.K == 0, ], legend='', title='')
#TimeSeriesDataUtility.plot_tall(dat)


# In[188]:

t = np.asarray(dat.loc[dat.K == 0, 'time'])
x = np.asarray(dat.loc[dat.K == 3, 'X'])
print(t)
print(x)

x = x.reshape(-1, 1)
t = t.reshape(-1, 1)


# In[201]:

#kernel = C(.1, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e5))
#kernel =  C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernel =  C(.1, (1e-5, 1e5)) * RBF(1, (1e-10, 1e5))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gp_n_opt_restarts)
gp = gp.fit(t, x)


# In[203]:

time_mesh = np.linspace(0, time_length, tau)
means, stds = gp.predict(time_mesh.reshape(-1,1), return_std=True)
means


# In[152]:

TimeSeriesDataUtility.plot_TK(means, legend='', title='')


# In[28]:

import numpy as np
dummy = np.zeros([10, 0, 2])


# In[29]:

dummy


# In[367]:

os.path.dirname('/Users/ssalehi/Desktop/pgas_sanity_check/exp_3KKJH_201710-25-171934.409470/sample_data.tsv.gz')


# In[ ]:



