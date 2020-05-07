import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.ticker as mticker
import math
import os
from time import time
import pandas as pn
import scipy as sp
import scipy.stats
import time as tm

# Global Constants
_LLHOOD_INF_WARNING = False
_TOLERATED_ZERO_PRECISION = 1e-20

# Dependencies 
exec(open('Utilities.py').read())
exec(open('Models.py').read())
exec(open('pgas.py').read())


from gp_llhood_parallel import *


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C



class GP_sampler:
    def __init__(self, dat, obsTimes, epsilon, h, nOptRestarts=10, full_original_dat=None):
        '''
        dat should be in TK format
        Fit a GP to each dimension independently, 
        then extract alpha and beta, and finally expose a conditional normal with these parameters
        '''
        self.T , self.K = dat.shape
        self.h = h
        self.epsilon = epsilon
        self.alpha = np.empty([self.K])
        self.beta = np.empty([self.K])
        self.C = np.empty([self.T-1, self.K]) # T-1 by K (since \delta T-s are different potentially, i.e., the step size h is a vector)
        self.sigma2 = np.empty([self.T-1, self.K])
        self.dat = dat
        self.obsTimes = obsTimes
        self.gps = [None]*self.K
        self.full_original_dat = full_original_dat
        
        for k in range(self.K):
            # X: observation times; y: observed values
            t = obsTimes.reshape(-1, 1)
            x = dat[:, k].reshape(-1,1)
            self.alpha[k], self.beta[k], self.gps[k] = self.fit_gp(t, x, nOptRestarts)
            for t in range(self.T-1):
                self.C[t, k] = self.alpha[k]*np.exp((-1/(2*self.beta[k]))*(self.h[t])**2)
                self.sigma2[t, k] = self.alpha[k] - ((self.C[t, k]**2)/self.alpha[k])
            
    def fit_gp(self, t, x, nOptRestarts):    
        kernel =  C(1.0, (1e-1, 1e1)) * RBF(10, (1e-1, 1e1)) # less wiggle
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=nOptRestarts)
        gp.fit(t, x)
        alpha = gp.kernel_.get_params()['k1__constant_value']
        beta = sqrt(gp.kernel_.get_params()['k2__length_scale'])
        return([alpha, beta, gp])
        
    def sample_full_path(self, x0, h, time_length, tau):
        Xi = np.empty([tau, x0.shape[0]])
        Xi[0, ] = x0
        if self.full_original_dat is not None:
            # load Xi
            true_dat = TimeSeriesDataUtility.read_time_series(self.full_original_dat)
            the_trajectory = TimeSeriesDataUtility.tall_to_TK(true_dat)
            print('the_trajectory.shape = ', the_trajectory['value'].shape)
            Xi = the_trajectory['value'][0:tau, ]
            return(Xi)
        print('Generati.ng gp.regressor sample:')
        time_mesh = np.linspace(0, time_length, tau)
        for k in range(self.K):
            means, stds = self.gps[k].predict(time_mesh.reshape(-1,1), return_std=True)
            Xi[:, k] = sp.stats.norm.rvs(loc=means.reshape(tau), scale=stds).reshape(tau)
            Xi[Xi[:, k] < 0, k] = -Xi[Xi[:, k] < 0, k]
        for t in range(tau):
            if np.sum(Xi[t, ]) > 1:
                Xi[t,] /= np.sum(Xi[t,])            
        return(Xi)
    
    def sample(self, x, deltaT, t):
        """
            x is a K by 1 array 
        """
        K = x.shape[0]
        sample_x = np.empty(x.shape)
        for k in range(K):
            mu = (self.C[t, k]/self.alpha[k]) * x[k]
            scale = sqrt(self.sigma2[t, k]+self.epsilon)
            a, b = (0 - mu)/scale, (1 - mu)/scale
            sample_x[k] = sp.stats.truncnorm.rvs(loc=mu, scale=scale, size=1, a=a, b=b)
        
        theSum = np.sum(sample_x)
        if theSum > 1:
            sample_x /= theSum
        return(sample_x)
    
    def compute_loglikelihood(self, x_M, x_t, deltaT, t):
        """
        x_M is the end point of the bridge
        x_t is the suggested state
        deltaT is their time difference
        Compute the conditional density of a normal $\Phi(x_m \mid \mu_{GP}(x_t), \sigma_{GP}(x_t))$
        Assuming independence of dimensions
        """
        K = x_t.shape[0]
        loglikelihood = 0.0
        for k in range(K):
            mu = (self.C[t, k]/self.alpha[k]) * x_t[k]
            
            loglikelihood += sp.stats.truncnorm.logpdf(x=x_M[k], loc=mu, scale=sqrt(self.sigma2[t, k]+self.epsilon), a=0, b=1)
            if math.isinf(loglikelihood) and _LLHOOD_INF_WARNING == True:
                print('llhood is Inf!')
                print('k is {}'.format(k))
                print('x_t is {}'.format(x_t))
                print('x_M is {}'.format(x_M))
                print('deltaT is {}'.format(deltaT))
                print('C = {}, mu = {}, sigma2 = {}'.format(self.C, mu, self.sigma2))
                print('self.alpha[k] = {}, self.beta[k] = {}'.format(self.alpha[k], self.beta[k]))           
                
        return(loglikelihood)  
    
    def compute_loglikelihood_vectorised(self, x_M, X_t, deltaT):
        raise ValueError('Dont use old and wrong and inefficient compute_loglikelihood_vectorised.')
        N, K = X_t.shape
        loglikelihood = np.empty([N, K])
        for i in range(N):
            for k in range(K):
                mu = (self.C[k]/self.alpha[k]) * X_t[i, k]
                scale = sqrt(self.sigma2[k]+self.epsilon)
                a, b = (0 - mu)/scale, (1 - mu)/scale
                loglikelihood[i, k] = sp.stats.truncnorm.logpdf(x=x_M[k], loc=mu, scale=scale, a=a, b=b)
        return(np.sum(loglikelihood, axis=1)) 
    
    
    def compute_loglikelihood_parallel(self, x_M, X_t, deltaT, llhood, n_cores=1, shuffle_map=None):
        the_C = np.empty(self.C.shape[1])
        the_sigma2 = np.empty(self.C.shape[1])
        for k in range(self.K):
            the_C[k] = self.alpha[k]*np.exp((-1/(2*self.beta[k]))*(deltaT)**2)
            the_sigma2[k] = self.alpha[k] - ((the_C[k]**2)/self.alpha[k])
            
        """
        K in the loop is derived from the shape of X_M
        """
        if shuffle_map is None:
            shuffle_map = np.arange(self.K)
        gp_compute_loglikelihood_parallel(x_M=x_M, X_t=X_t, alpha=self.alpha[shuffle_map], sigma2=the_sigma2[shuffle_map], C=the_C[shuffle_map], epsilon=self.epsilon, loglikelihood=llhood, n_threads=n_cores)
    
    
    def plot_gp_fit(self):
        self.plot(k=0)
        self.plot(k=1)
        
    def plot_gp_dimension(self, k, M=1000):       
        t = self.obsTimes.reshape(-1, 1)
        x = self.dat[:, k].reshape(-1,1)
        kernel = C(.1, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
        gp.fit(t, x)
        times = np.linspace(self.obsTimes[0], self.obsTimes[T-1], num=M)
        y_pred, sigma = gp.predict(times.reshape(-1,1), return_std=True)
        #%matplotlib inline
        plt.fill(np.concatenate([times, times[::-1]]), np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), alpha=.5, fc='b', ec='None', label='95% confidence interval')

class tune_proposal(object):
    def __init__(self):
        print('')
    
    def tune(self):
        print('')
 



class OutterPGAS(ParticleGibbs):
    def __init__(self, smoothing_kernel, parameter_proposal_kernel, initial_distribution_theta, initial_distribution_x, observations, h, MCMC_in_Gibbs_nIter=20):          
        self.q0 = initial_distribution_theta
        self.p0 = initial_distribution_x
        self.smoothing_kernel = smoothing_kernel
        self.p_theta = parameter_proposal_kernel
        self.obs = observations
        self.T, self.K = self.obs['value'].shape
        # Assuming that the time step h overlaps the observation times
        self.h = h
        self.time_length = self.obs['times'][self.T-1] - self.obs['times'][0]
        self.tau = self._get_tau()
        self.sample_processor = None
        self.data_loader = None
        self._current_iter = 1 # iteration zero handles starting values
        self.x0 = None
        self.nIter = None
        self.MCMC_in_Gibbs_nIter = MCMC_in_Gibbs_nIter
        
        print('OutterPGAS: time_length, tau = {} {}'.format(self.time_length, self.tau))
        
        
    def __pad_x_to_Xi(self, x, Xi):
        print('not implemented')
        # if the input x is only the observations, ignore the auxuliary Xi
         
            
    def sample_initial_values(self):
        # Initialise
        print('in sample_initial_values')
        self.x[:, :] = self.p0.sample_full_path(self.x0[0,], self.h, self.time_length, self.tau)
        try:
            self.theta[:] = self.q0.sample()
        except AttributeError:
            pass
    
    def load_initial_values(self):
        self._current_iter = self.data_loader.get_last_iter()+1
        self.x[:, :] = self.data_loader.get_x()
        try:
            self.theta[:] = self.data_loader.get_theta()
        except AttributeError:
            pass
   
    def _initialise(self, is_resume, tag):
        if is_resume:
            self.load_initial_values()            
        else:
            self.sample_initial_values()
            rvs = [self.theta, self.x, np.array([-np.Inf])] if tag == 'sample' else self.x
            self.call_sample_processor(rvs, 0, self.nIter, tag=tag)

    
    def call_sample_processor(self, rvs, iteration, nIter, tag):
        if self.sample_processor is not None:
            self.sample_processor.sample_processor(rvs=rvs, tag=tag, iteration=iteration, nIter=nIter)
    
    def sample(self, nIter, x0, is_resume=False):
        """
        x0 is a skeleton path over observed times
        """

        # T by K
        self.theta = np.empty([self.K])
        # \tau by K
        self.x = np.empty([self.tau, self.K])
        print('OutterPGAS: sample: self.tau={}'.format(self.tau))
        self.x0 = x0
        self.nIter = nIter
        self._initialise(is_resume, 'sample')
        startTime = int(round(time()))
        llhood = -np.Inf
        
        should_update_epsilon = False
        original_epsilon = .01
        waiting = 0
        for i in range(self._current_iter, nIter):
            endTime = int(round(time()))
            print('On iteration {}/{} --- OutterPGAS --- {} -- (llhood = {} )'.format(i+1, nIter, time_string_from_seconds(endTime-startTime), llhood))
            _, self.x, n_passed = self.smoothing_kernel.sample_non_parallel(self.x.copy(), self.theta.copy())
            self.theta[:], llhood = self.p_theta.sample(nIter=self.MCMC_in_Gibbs_nIter, theta=self.theta.copy(), x=self.x[:,:].copy())
            
            # Write-down or otherwise process samples every so iterations
            self.call_sample_processor([self.theta, self.x.copy(), np.array([llhood])], i, nIter, tag='sample')

        return([self.theta, self.x])
    
    def predict(self, theta_vector, t_learn, x0, sample_parallel=False, is_resume=False):
        """
        The time difference between timepoints is encoded in the observation vector
        """
        nIter, _dummyK = theta_vector.shape
        # Init particles
        # N by \tau by K
        self.x = np.empty([self.tau, self.K])
        self.x0 = x0
        self.nIter = nIter
        # Initialise
        self._initialise(is_resume, 'predict')
        startTime = int(round(time()))
        self.starting_x = self.x[:,:].copy()
        
        for i in range(self._current_iter, nIter):
            endTime = int(round(time()))
            print('On iteration {}/{} --- OutterPGAS --- {}'.format(i+1, nIter, time_string_from_seconds(endTime-startTime)))
            
            if sample_parallel is False:
                self.x[:,:] = self.smoothing_kernel.sample_non_parallel(self.x[:,:].copy(), theta_vector[i-1, ], T_learn = t_learn)[1]
            else:
                self.x[:,:] = self.smoothing_kernel.sample_parallel(self.x[:,:].copy(), theta_vector[i-1, ], T_learn = t_learn)[1]                
            
            self.call_sample_processor(self.x, i, nIter, tag='predict')
        return(self.x)
    



class Random_walk_proposal:
    def __init__(self, mu, sigma, upper_bound, lower_bound):
        self.mu = mu
        self.sigma = sigma
        self.ub = upper_bound
        self.lb = lower_bound
    
    def compute_loglikelihood(self, loc, value):
        """
        loc is the to be mu on which the value is coditioned
        """
        K = loc.shape[0]
        llhoods = np.empty([K])
        for k in range(K):
            a, b = (self.lb[k] - loc[k]) / self.sigma[k], (self.ub[k] - loc[k]) / self.sigma[k]
            llhoods[k] = sp.stats.truncnorm.logpdf(a=a, b=b, loc=loc[k], x=value[k], scale=self.sigma[k])
        return(np.sum(llhoods))
        
    def sample(self, loc):
        K = loc.shape[0]
        samples = np.empty([K])
        for k in range(K):
            a, b = (self.lb[k] - loc[k]) / self.sigma[k], (self.ub[k] - loc[k]) / self.sigma[k]
            samples[k] = sp.stats.truncnorm.rvs(a=a, b=b,loc=loc[k], scale=self.sigma[k], size=1)
            if samples[k] < self.lb[k] or samples[k] > self.ub[k]:
                raise ValueError('The sample is not in range {}, loc={}, sigma={}'.format(samples[k], loc[k], self.sigma[k]))

        return(samples)
        

class MH_Sampler:
    def __init__(self, adapted_proposal_distribution, likelihood_distribution):
        self.phi = adapted_proposal_distribution
        self.l = likelihood_distribution
        
    def sample(self, nIter, theta, x):
        """
        Even though the proposal is a random walk, since it may be truncated, we won't cancel them out
        x is the scaffold of discritised values
        """
        theta_prime = theta
        llhood_old_cache = None
        
        # Precompute sufficient statistics
        B_inv = self.l.compute_B_inv(x)
        for j in range(nIter):
            theta_new = self.phi.sample(theta_prime)
            # Don't re-compute llhood_old if it hsan't been rejected
            llhood_old = self.l.compute_loglikelihood_cache_x(s=theta_prime, x=x, B_inv = B_inv);
            
            llhood_new = self.l.compute_loglikelihood_cache_x(s=theta_new, x=x, B_inv = B_inv);
            q_llhood = self.phi.compute_loglikelihood(loc=theta_new, value=theta_prime)
            q_llhood_reverse = self.phi.compute_loglikelihood(loc=theta_prime, value=theta_new)
            A = np.minimum(1.0, np.exp(llhood_new + q_llhood - llhood_old - q_llhood_reverse))
            if np.isnan(A):
                print('MH acceptance probability is NaN!')
                print('theta_new = {}, theta_prime = {}'.format(theta_new, theta_prime))
                print('llhood_new={} + q_llhood={} - llhood_old={} - q_llhood_reverse={}'.format(llhood_new, q_llhood, llhood_old, q_llhood_reverse))
                print(x)
                raise ValueError('MH acceptance probability is NaN!')

            b = np.random.binomial(size=1, p=A, n=1)
            if b == 1:
                theta_prime = theta_new
        return([theta_prime, llhood_new])
    
    def sample_risky(self, nIter, theta, x):
        """
        Even though the proposal is a random walk, since it may be truncated, we won't cancel them out
        x is the scaffold of discritised values
        """
        theta_prime = theta
        llhood_old_cache = None
        for j in range(nIter):
            theta_new = self.phi.sample(theta_prime)
            # Don't re-compute llhood_old if it hsan't been rejected
            if llhood_old_cache is None:
                llhood_old = self.l.compute_loglikelihood(s=theta_prime, x=x);
            else:
                llhood_old = llhood_old_cache
                
            llhood_new = self.l.compute_loglikelihood(s=theta_new, x=x);
  
            A = np.minimum(1.0, np.exp(llhood_new-llhood_old))

            if np.isnan(A):
                print('MH acceptance probability is NaN!')
                print('theta_new = {}, theta_prime = {}'.format(theta_new, theta_prime))
                print('llhood_new={} + q_llhood={} - llhood_old={} - q_llhood_reverse={}'.format(llhood_new, q_llhood, llhood_old, q_llhood_reverse))
                print(x)
                raise ValueError('MH acceptance probability is NaN!')
            
            b = np.random.binomial(size=1, p=A, n=1)
            if b == 1:
                theta_prime = theta_new
                llhood_old_cache = None
            else:
                llhood_old_cache = llhood_old
            
        return([theta_prime, llhood_new])
    
    def sample_hierarchical(self, nIter, theta, Xs):
        """
        Even though the proposal is a random walk, since it may be truncated, we won't cancel them out
        Xs is a list of x,
        x is the scaffold of discritised values
        """
        theta_prime = theta
        llhood_old_cache = None
        for j in range(nIter):
            theta_new = self.phi.sample(theta_prime)
            # Don't re-compute llhood_old if it hsan't been rejected
            if llhood_old_cache is None:
                llhood_old = 0.0
                for x in Xs:
                    llhood_old = llhood_old + self.l.compute_loglikelihood(s=theta_prime, x=x);
            else:
                llhood_old = llhood_old_cache
            llhood_new = 0.0
            for x in Xs:
                llhood_new = llhood_new + self.l.compute_loglikelihood(s=theta_new, x=x);
            q_llhood = self.phi.compute_loglikelihood(loc=theta_new, value=theta_prime)
            q_llhood_reverse = self.phi.compute_loglikelihood(loc=theta_prime, value=theta_new)
            A = np.minimum(1.0, np.exp(np.sum(np.array([llhood_new, q_llhood, -llhood_old, -q_llhood_reverse]))))

            if np.isnan(A):
                print('theta_new = {}, theta_prime = {}'.format(theta_new, theta_prime))
                print('llhood_new={} + q_llhood={} - llhood_old={} - q_llhood_reverse={}'.format(llhood_new, q_llhood, llhood_old, q_llhood_reverse))
                print(Xs)
                raise ValueError('MH acceptance probability is NaN!')
            b = np.random.binomial(size=1, p=A, n=1)
            if b == 1:
                theta_prime = theta_new
                llhood_old_cache = None
            else:
                llhood_old_cache = llhood_old
            
        return([theta_prime, llhood_new])


class s_uniform_sampler:
    def __init__(self, K, dim_min, dim_max):
        self.K = K
        self.min = dim_min
        self.max = dim_max
    def sample(self):
        theta = np.empty([self.K])
        for i in range(self.K):
            theta[i] = np.random.uniform(high=self.max, low=self.min, size=1)
        return(theta)

