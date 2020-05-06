
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.ticker as mticker


# # Issue tracker
# Types:
# May be mutation processes
# 
# ## Questions
# ***- WHY IS THERE A DENT toward the observation in PREDICTION TIME in lower iterations?***
# - Check that when the sum of previous ones has reached one, all the others are set to zero...
#     - That's why the dynamic X_new ,min, max is in place, to enforce the boundary conditions. 
# - ***Checkout multinomial regression***
#     - The Vanila Wright-Fisher model is nothing more than this. 
# 
# - **CSMC within particle MCMC**?
# - **Use a bridge sampler for each particle in the PGAS sweep. This is very amenable to multiprocessing. How about that?**
# - What is the best way to do prediction using a model learnt by PGAS? Just a PF? 
# - **Should we fit a GP to the Reference Trajectory in each run of the PGAS**?
# - How to best pick the starting reference trajectory?
# - Senstitive to Epsilon, how to adapt it as we go?
# - The diffusion is independent of the parameters (namely, S values). How can we exploit this feature in the inference?
# - How to tune the parameters of the proposal distribution?
#     - LibBi's strategy of changing till we get a target acceptance rate?
# - What if we propagate from GP (i.e., $r_{\theta}(.) = GP$ and only weight the particles using the Wright-Fisher discretization (i.e., $f_{\theta}(x_t \mid x_{t-1})$)
#     - Pros: 
#         - The suggested trajectories are guaranteed to be very close to a bridge.
# - Does the conditional distribution of $\Theta$ given the current state, a Gaussian with modified $\mu_i$ and $\sigma$?
# See `Bromiley P. Products and convolutions of gaussian probability density functions. Tina-Vision Memo. 2003;3.`
# in http://www.tina-vision.net/docs/memos/2003-003.pdf
# More generally see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
# 
# - Discuss the merits of using a DNN to learn the WR model with multiple parameters over a grid using the Diffusion approximation with very fine discretisation
# 
# ## ToDos
# - Simplify the model when an element reaches 0.
# - **Make it much faster by using a courser step size while computing the log likelihood of theta!**
# - Add strict test for the ball around observation after inference is done. 
# - Add submit batch to genesis 
# - We then computerunningmeansofthelatentvariablesx1:T and,fromthese,wecomputetherunning root-mean-squared errors (RMSEs) âœn relative to the true posterior means (computed with a modified Bryson-Frazier smoother, Bierman, 1973).
# - Implement tests in Gweke Getting it right paper.
# - According to `profile`, `_multivariate' is a bottleneck of the algorithm. See if an in-house implementation, without the need for the cholesky decom is feasible.
# - Break down experiment into Experiment and Prediction/Inference experiment to allow support for data generation, plotting, etc batch jobs.
# - Decouple the number of particles in the bridge sampler from that of the PGAS
# - Use `Tracer()()` for debugging.
# - **How similar are the posterior theta histograms to the priors?**
# - Write MCMC results to file as you go.
# - In MH-Sampler, don't re-compute llhood_old if it hsan't been rejected
# - Add job submission to cluster classes.
# - Extend the initialisation model to Libbi! The current Gamma normalisation is suboptimial.
# - Add implementation for higher K values
# - Add MCMC diagnostics, minimum of coda
#     - Look at porting py-coda https://github.com/surhudm/py-coda to Python3 (it's written in Python2.6)
#     - Look at PyMC3 here: https://pymc-devs.github.io/pymc3/notebooks/posterior_predictive.html
# - Determine criteria to assess prediction accuracy
# - Add a review of forecasting methods in time-series
# - Refactor the Wright-Fisher transmission as a vector operation
# - Explore memory saving schemes for particles, especially for the $\Xi$ auxiliary variables that are only needed as place holders; one idea is to only save the random walk and $\Theta$ pair and then recreate them when needed.
#     - See exp-Golomb coding and https://cs.stackexchange.com/questions/20156/compressing-normally-distributed-data
#     - Memory complexity of the PGAS is  $O(NTK\frac{h}{\tau})$ where $\tau$ is the time length. 
#     - Simulate all the random walks at the beginning of a run
# - Add a review session on diffusion bridge simulation methods.
# - Compare Choleskey and system of linear equations to see which one is better for loglikelihood computation.
# - Give the guided proposal method paper another read. Perhaps simulating once from a SDE with modified drift term would be much easier (How correct will that be?)?
# - Devise a clever resampling for the Kernel.
# - Implement `def __str__(self)` for all classes.
# - Investigate machine precision making varibles negative, especially in the posterior for X|Y. For example use, `np.iffinite()`
# - Parallelize the Particle Gibbs, at least for particle propagation
# - Verify the particle weighting strategy
# - Write testcases, see e.g., http://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/

# # Code snippets
# ## Multiprocesing
# ```python
# from multiprocessing import Pool
# pool = multiprocessing.Pool(4)
# out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))```
# See documentation here:
# https://docs.python.org/3/library/multiprocessing.html

# # Libraries

# In[2]:

import math
import os
import pandas as pn
#import multiprocessing
import scipy as sp
import scipy.stats
#from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool
#from IPython.core.debugger import Tracer

# Global Constants
_LLHOOD_INF_WARNING = False
_TOLERATED_ZERO_PRECISION = 1e-20


# Global debugging
_XX_x = None


# Dependencies 
#exec(open(os.path.expanduser('scalable_computing.py')).read())
exec(open(os.path.expanduser('Utilities.py')).read())
exec(open(os.path.expanduser('Models.py')).read())
exec(open(os.path.expanduser('pgas.py')).read())


#from gp_llhood_parallel import *
from gp_llhood_parallel import *


# In[ ]:




# In[ ]:




# # PGAS 
# 
# ## Different distributions involved in the PGAS+ParticleRejuvenation
# - $r_{\theta, 1}(x_1 \mid y_1)$
#     - Sampling:
#          - IterativeSampling to ensure $x_1$ complies with the simplex condition
#     - Likelihood:
#         - All equal to $1/N$
# - $g_{\theta}(y_t|x_t^i)$
#     - Sampling: 
#         - NA
#     - Likelihood: Making sure that particles that are closely resemble a bridge have higher weights.
#     \begin{equation}
#     g_{\theta}(y_t|x_t^i) = \prod_{j=1}^K g_i(y_{t,j}|x_t^{i,j}), \text{ where } g_i(y_{t,j}|x_t^{i,j}) =  
#     \begin{cases}
#       1, & \text{if}\ \lvert y-x \rvert <\epsilon \\
#       \text{exp}(-\lambda y), & \text{otherwise}
#     \end{cases}
#     \end{equation}
#     with e.g., $\lambda = 10$
# - $\mu_{\theta}(x_1^i) = 1$
# - $w_1^i$ just set it to the likelihood of $g_{\theta}(y_t|x_t^i)$
# - $r_{\theta, t}(x_t \mid \tilde{x}_{t-1}^i, y_t)$
#     - Sampling:
#         - Ignore the y, and use the black box simulator, or $f_{\theta}$ to generate $x_t$
#     - Likelihood:
#         - Set to 1
# - $f_{\theta}(x_t^i \mid \tilde{x}_{t-1}^i)$
#     - Sampling:
#         - Use the black box to propagate forward as in the case of $r_{\theta, t}(x_t \mid \tilde{x}_{t-1}^i, y_t)$
#     - Likelihoood
#         - Equal to 1, unless the proposal is different, in that case it will be product of Gaussians of the Euler-Maruyama approximation.        
# 
# ## Note
# In the standard PGAS, the reference ancestor resampling step is:
# \begin{equation}
# P(a_t^{b_t} = i) \propto w_{t-1}^i f(x_t \mid x_{t-1})
# \end{equation}
# The _PGAS+Rejuvenation_, replaces this with resampling an entirely new bridge.
# While this means that the _PGAS+Rejuvenation_ does not have to keep track of states at disceitise steps, 
# it may lead to redundancies, since the $K(.)$ bridge kernel has to it all over again.
# 
# ## Pseudo code
# 0. Get $\theta \in \Theta$ and $x'_{1:T}$ the parameters and reference trajectory respectively.
# 1. Draw $x_0^i$ using the string chopping precedure above for the first $N-1$ particles
# 2. $x_0^N \leftarrow x'_0$, i.e., set the $N$-th particle to the first leg of the reference trajectory
# 3. $w_0^i \leftarrow g_{\theta}(y_t|x_t^i)$ For all particles, weight them according to how close they are to the observation
# 4. **For** t in 1:(T-1) **Do**
#     - **For** i in $0:N-2$ particles **Do**
#         - $a_t^i \propto w_{t-1}$, from all particles 
#         // Ancestor sampling and resampling
#     - **For** i in $0:N-2$ particles **Do**
#         - $x_t^i \sim f_{\theta}(. \mid x_{t-1}^{a_t^i})$
#         // Propagate particles
#     7. $(a_t^N, \Xi_t) \sim K_t(.)$
#     // Rejuvenate the ancestor and $Xi_t$
#     - **For** i in $0:N-1$ particles **Do**
#         - $w_t^i \leftarrow g_{\theta}(y_t|x_t^i)$, similar to step 3
# 7. Draw $k \propto w_T^i$ from all particles
# 8. **return** $x_{1:T}^k$ where the trajectory is constrcuted recursively using the ancestor indeces$
# 
# ### Prediction special case:
# The input would be an additional time value, indicating the prediction moment, where observations are ignored. 
# 

# ## Particle Gibbs sampler implementation

# In[28]:




# 

# In[ ]:




# In[3]:




# # Kernel $K(.,.)$ to resample ancestor and rejuvenate $\Xi$
# 
# In our case, that is sampling a bridge between two consecutive observation, with a slight abuse of notation, 
# by indexing variables by t we mean their corresponding value in $\{T-1, T-1+\Delta \tau, T-1+2\Delta \tau,..., T \}$, so that 
# $x_t := x_{T-1+t \Delta \tau}$
# 
# We ignore observations.
# $p_0(.)$ is a multivariate with weights $w_{t-1}^{a_t}$ 
# 
# 0. Get $p_0$ vector, $N$ particles
# 1. $x_0^i \sim p_0(.)$
# 2. $w_0^i \leftarrow 1/N$
# 3. **For** t in 1:M **Do** 
#     \\ M is the number of discretised steps in between two timepoints
#     - **For** i in 1:N **Do**
#         - **If** resampling is triggered **Then** 
#             - $a_t^i \sim \text{Mult}(. \mid w_{t-1}^{1:N})$
#             - $w_t^i \leftarrow 1/N$
#             
#         - **else**
#             - $a_t^i \leftarrow i$
#             - $w_t^i \leftarrow w_{t-1}^i/ \sum_{j}^N w_{t-1}^j$
#     - **For** i in 1:N **Do**            
#         - $x_t^i \leftarrow f_{\theta}(x_t \mid x_{t-1}^{a_t^i})$
#         
#         - $w_t^i \leftarrow r(x_{M} \mid x_{t}^i) \frac{w_t^i}{w_{t-1}^{a_t^i}}$
# 
# 
# where $r()$ is from a fitted GP and note that it measures how well this one is on the path to become $x_M$, the bridge endpoint.
# 

# 

# In[30]:




# In[ ]:




# ## GP Handler implementation
# When using a GP to sample new states, we need to make sure that the final suggestion is dirichlet compatible, i.e., the values fall into [0,1] and their sum does not exceed 1.

# In[1]:

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
        #self.C = np.empty([self.K])
        self.C = np.empty([self.T-1, self.K]) # T-1 by K (since \delta T-s are different potentially, i.e., the step size h is a vector)
        self.sigma2 = np.empty([self.T-1, self.K])
        self.dat = dat
        self.obsTimes = obsTimes
        self.gps = [None]*self.K
        self.full_original_dat = full_original_dat
        
        #print('GP.dat =', dat)
        for k in range(self.K):
            # X: observation times; y: observed values
            t = obsTimes.reshape(-1, 1)
            x = dat[:, k].reshape(-1,1)
            #print('GP:t,x=', t, x)
            self.alpha[k], self.beta[k], self.gps[k] = self.fit_gp(t, x, nOptRestarts)
            for t in range(self.T-1):
                self.C[t, k] = self.alpha[k]*np.exp((-1/(2*self.beta[k]))*(self.h[t])**2)
                self.sigma2[t, k] = self.alpha[k] - ((self.C[t, k]**2)/self.alpha[k])
            
    def fit_gp(self, t, x, nOptRestarts):    
        #kernel = C(.1, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e5))
        #kernel =  C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        #kernel =  C(1.0, (1e-1, 1e1)) * RBF(10, (1e-2, 1e2))
        kernel =  C(1.0, (1e-1, 1e1)) * RBF(10, (1e-1, 1e1)) # less wiggle
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=nOptRestarts)
        gp.fit(t, x)
        alpha = gp.kernel_.get_params()['k1__constant_value']
        beta = sqrt(gp.kernel_.get_params()['k2__length_scale'])
        #print('GP:params=', gp.kernel_.get_params())
        return([alpha, beta, gp])
        
    def sample_full_path(self, x0, h, time_length, tau):
        #WrightFisherDiffusion.check_state(x0)
        #tau = int(round(time_length/h, _TIME_ROUNDING_ACCURACY)) + 1
        Xi = np.empty([tau, x0.shape[0]])
        #print('x0.shape[0] = {}'.format(x0.shape[0]))
        Xi[0, ] = x0
#         print('Returning old style X0')
#         for t in range(1, tau):
#             Xi[t, ] = self.sample(Xi[t-1,], h)
#         return(Xi)
        if self.full_original_dat is not None:
            print('ERROR!!! Returning the TRUE Xi')
            # load Xi
            true_dat = TimeSeriesDataUtility.read_time_series(self.full_original_dat)
            the_trajectory = TimeSeriesDataUtility.tall_to_TK(true_dat)
            print('the_trajectory.shape = ', the_trajectory['value'].shape)
            print('Xi.shape = ', Xi.shape)
            Xi = the_trajectory['value'][0:tau, ]
            return(Xi)
        print('Generati.ng gp.regressor sample:')
        # TODO: @Sohrab generate a gaussian using the std and mean instead of the means...
        time_mesh = np.linspace(0, time_length, tau)
        for k in range(self.K):
            means, stds = self.gps[k].predict(time_mesh.reshape(-1,1), return_std=True)
            Xi[:, k] = sp.stats.norm.rvs(loc=means.reshape(tau), scale=stds).reshape(tau)
            #Xi[:, k] = self.gps[k].predict(time_mesh.reshape(-1,1), return_std=False).reshape(tau)
            # TODO: Sohrab: CHanged this as well"
            #Xi[Xi[:, k] < 0, k] = 0
            Xi[Xi[:, k] < 0, k] = -Xi[Xi[:, k] < 0, k]
        for t in range(tau):
            if np.sum(Xi[t, ]) > 1:
                Xi[t,] /= np.sum(Xi[t,])            
        #WrightFisherDiffusion.check_path(Xi)    
        #print('sample_full_path Xi[0, ] = {}'.format(Xi[0, ]))
        #print('sample_full_path Xi = {}'.format(Xi))
        
        #%matplotlib inline
        #TimeSeriesDataUtility.plot_TK(Xi, legend='', title='GP sample')
        
        #raise ValueError('Stop after plotting...')
        
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
        
        #WrightFisherDiffusion.check_state(sample_x)    
        
        return(sample_x)
    
    def compute_loglikelihood(self, x_M, x_t, deltaT, t):
        """
        x_M is the end point of the bridge
        x_t is the suggested state
        deltaT is their time difference
        Compute the conditional density of a normal $\Phi(x_m \mid \mu_{GP}(x_t), \sigma_{GP}(x_t))$
        Assuming independence of dimensions
        """
        print('ERROR! C[k] for the end point needs to be recomputed, since it deltaT is between bridge_end_point and current point which is almost always > self.h')
        K = x_t.shape[0]
        loglikelihood = 0.0
        for k in range(K):
            mu = (self.C[t, k]/self.alpha[k]) * x_t[k]
            
            #a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            
            
            loglikelihood += sp.stats.truncnorm.logpdf(x=x_M[k], loc=mu, scale=sqrt(self.sigma2[t, k]+self.epsilon), a=0, b=1)
            #loglikelihood = loglikelihood + sp.stats.norm.logpdf(x=x_M[k], loc=mu, scale=sqrt(sigma2+self.epsilon))
            #if loglikelihood < -10000.00:
            if math.isinf(loglikelihood) and _LLHOOD_INF_WARNING == True:
                print('llhood is Inf!')
                print('k is {}'.format(k))
                print('x_t is {}'.format(x_t))
                print('x_M is {}'.format(x_M))
                print('deltaT is {}'.format(deltaT))
                print('C = {}, mu = {}, sigma2 = {}'.format(self.C, mu, self.sigma2))
                print('self.alpha[k] = {}, self.beta[k] = {}'.format(self.alpha[k], self.beta[k]))
                #raise ValueError('llhood is {}'.format(loglikelihood))
                
            #print("sp.stats.norm.logpdf(x=x_M[{}], loc=mu, scale=sqrt(sigma2), a=0, b=1) = {}".format(k, loglikelihood))
        #print('loglikelihood is {}'.format(loglikelihood))
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
                # TODO: handle infinity 
        return(np.sum(loglikelihood, axis=1)) 
    
    
    def compute_loglikelihood_parallel(self, x_M, X_t, deltaT, llhood, n_cores=1, shuffle_map=None):
        ### TODO: @Sohrab!!! Update C to recompute according to deltaT. As is, it assumes deltaT = self.h, while x_M and X_t are almost surely further apart...
        ### October 18, 2018: Why not recompute sigma2? It depends on the new C...
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
        #gp_compute_loglikelihood_parallel(x_M=x_M, X_t=X_t, alpha=self.alpha[shuffle_map], sigma2=self.sigma2[shuffle_map], C=self.C[shuffle_map], epsilon=self.epsilon, loglikelihood=llhood, n_threads=n_cores)
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
        


# In[8]:




# In[ ]:




# # Bayesian learining of the SSM
# ## Pseudo code
# 
# M iterations.
# 
# 0. $\theta[0] \sim q_0(.)$ and $x_{1:T}[0] \sim p_0(.)$
# 1. **For** n in 1:M **Do**
#     1. $x_{1:T}[n] \sim \text{PGAS}(x_{1:T}[n-1], \theta[n-1])$
#     2. $\theta[n] \sim p(\theta \mid x_{1:T}[n], y_{1:T})$  
# 
# 
# To part B in the above algorithm, use a MH algorithm, with say, 10 steps, that proposes from a random walk Gaussian and weighs by the approximation to the conditional distribution 
# This requires outputing of the $\Xi$ and computing the $\prod \prod \Phi_{\theta}(\Xi_{i,j}) $
# 
# 

# In[33]:

class tune_proposal(object):
    def __init__(self):
        print('')
    
    def tune(self):
        print('')
 


# 
# ## Alternative Kernel, Conditional importance sampling
# Another method, based on the Conditional importance sampling (Alg. 3 of Lindsten et al. 2015), is as follows:
# Propose from GP and weight by WF.
# More precisely 
# $$\nu^{a_t} = \Phi_{GP}(. \mid \mu_t, \sigma_t)$$
# $$\psi(\Xi) = \Phi_{GP}(.)$$
# Then weight $(a_{t-1},\Xi)$-s by 
# $$\frac{w(a_{t-1}) f_{\theta}(x_t \mid x_{t-1}))}{v(a_{t-1}) \psi(\Xi)}$$
# Finally return the $i$-th pair $\propto w_i$
# 
# ### This is probably not that desirable, since the dynamics are not reflective of the underlying process (paths are not simulated from WF diffusion approximation)
# 

# ## OutterPGAS implementation
# 
# ### Consider renaming to Metropolis-within-Gibbs algorithm

# In[1]:

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
        #self.tau = int(round(self.time_length/h, _TIME_ROUNDING_ACCURACY)) + 1
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
        
        #print('ERROR- USING TRUE S VALUE FOR SEED =9')
        #self.theta = np.array([-0.15808434270324956, -0.3787244359894405, -0.4224185247188207, -0.4759770259429087])
        #self.theta = np.array([-0.3469458012338531, -0.3423143389905927, 0.4854911141831526, 0.4044977308052937])
        
#         print("Sampling starting value:")
#         print(self.x)
        #%matplotlib inline
        #TimeSeriesDataUtility.plot_TK(self.x, legend='', title='Starting value')        
        
        
        startTime = int(round(time.time()))
        llhood = -np.Inf
        
        should_update_epsilon = False
        original_epsilon = .01
        waiting = 0
        for i in range(self._current_iter, nIter):
            endTime = int(round(time.time()))
            print('On iteration {}/{} --- OutterPGAS --- {} -- (llhood = {} )'.format(i+1, nIter, time_string_from_seconds(endTime-startTime), llhood))
            #self.x = self.smoothing_kernel.sample_non_parallel(self.x.copy(), self.theta.copy())[1]
            _, self.x, n_passed = self.smoothing_kernel.sample_non_parallel(self.x.copy(), self.theta.copy())
            
            
            
#             if should_update_epsilon:
#                 if np.mean(np.array(n_passed)) < 1 or np.min(np.array(n_passed)) == 0:
#                     self.smoothing_kernel.g.epsilon = self.smoothing_kernel.g.epsilon*2
#                     waiting = 0
#                     print('doubling epsilon')
#                 elif np.mean(np.array(n_passed)) > 100 and np.min(np.array(n_passed)) > 0 and self.smoothing_kernel.g.epsilon > original_epsilon and waiting > 10:
#                     self.smoothing_kernel.g.epsilon = self.smoothing_kernel.g.epsilon/2
#                     waiting = 0
#                     print('halfing epsilon')
#                 print('Epsilon is ', self.smoothing_kernel.g.epsilon)
#                 waiting = waiting + 1
                
            # Check the full path
            #WrightFisherDiffusion.check_path(self.x[i, :,:])
            #WrightFisherDiffusion.check_path(self.x)
            
            self.theta[:], llhood = self.p_theta.sample(nIter=self.MCMC_in_Gibbs_nIter, theta=self.theta.copy(), x=self.x[:,:].copy())
            # Test the speed, doesn't matter
            #self.theta[:], llhood = [np.array([.004,.039, .032]), -10]

            
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
        startTime = int(round(time.time()))
        
        #print("Predict starting value:")
        #print(self.x[:,:])
        
        self.starting_x = self.x[:,:].copy()
        
        for i in range(self._current_iter, nIter):
            endTime = int(round(time.time()))
            print('On iteration {}/{} --- OutterPGAS --- {}'.format(i+1, nIter, time_string_from_seconds(endTime-startTime)))
            
            if sample_parallel is False:
                #print('Error! Debugging! Not using the previous path...')
                self.x[:,:] = self.smoothing_kernel.sample_non_parallel(self.x[:,:].copy(), theta_vector[i-1, ], T_learn = t_learn)[1]
                #self.x[:,:] = self.smoothing_kernel.sample_non_parallel(self.starting_x.copy(), theta_vector[i-1, ], T_learn = t_learn)[1]
            else:
                self.x[:,:] = self.smoothing_kernel.sample_parallel(self.x[:,:].copy(), theta_vector[i-1, ], T_learn = t_learn)[1]                
                print('Warning! Sampling Parallel!')
            
            self.call_sample_processor(self.x, i, nIter, tag='predict')
            
            # TODO: write-down samples every J iterations
        return(self.x)
    


# In[3]:




# # HierarchicalOuterPGAS class with support for multiple repeats

# In[ ]:

class HierarchicalOuterPGAS(ParticleGibbs):
    def __init__(self, smoothing_kernels, parameter_proposal_kernel, initial_distribution_theta, initial_distribution_list_x, observations_list, h, MCMC_in_Gibbs_nIter=10):          
        self.M_repeats = len(smoothing_kernels)
        self.q0 = initial_distribution_theta
        self.p0 = initial_distribution_list_x
        self.smoothing_kernels = smoothing_kernels
        self.p_theta = parameter_proposal_kernel
        self.obs = observations_list[0]
        self.obs_list = observations_list
        self.T, self.K = self.obs['value'].shape
        # Assuming that the time step h overlaps the observation times
        self.h = h
        self.time_length = self.obs['times'][self.T-1] - self.obs['times'][0]
        #self.tau = int(round(self.time_length/h, _TIME_ROUNDING_ACCURACY)) + 1
        self.tau = self._get_tau()
        self.sample_processors = None
        self.data_loader = None
        self._current_iter = 1 # iteration zero handles starting values
        
        self.x0 = [None]*self.M_repeats
        self.x = [None]*self.M_repeats
        
        self.nIter = None
        self.MCMC_in_Gibbs_nIter = MCMC_in_Gibbs_nIter
                
    def __pad_x_to_Xi(self, x, Xi):
        print('not implemented')
        # if the input x is only the observations, ignore the auxuliary Xi
         
            
    def sample_initial_values(self):
        # Initialise
        for m in range(self.M_repeats):
            self.x[m][:, :] = self.p0[m].sample_full_path(self.x0[m][0,], self.h, self.time_length, self.tau)
            try:
                self.theta[:] = self.q0.sample()
            except AttributeError:
                pass
    
    def load_initial_values(self):
        self._current_iter = self.data_loader.get_last_iter()+1
        for m in range(self.M_repeats):
            self.x[m][:, :] = self.data_loader.get_x()
        try:
            self.theta[:] = self.data_loader.get_theta()
        except AttributeError:
            pass
   
    def _initialise(self, is_resume, tag):
        if is_resume:
            self.load_initial_values()            
        else:
            self.sample_initial_values()
            for m in range(self.M_repeats):
                rvs = [self.theta, self.x[m], np.array([-np.Inf])] if tag == 'sample' else self.x[m]
                self.call_sample_processor(rvs, 0, self.nIter, tag=tag, repeat_index=m)

    
    def call_sample_processor(self, rvs, iteration, nIter, tag, repeat_index):
        if self.sample_processors is not None:
            self.sample_processors[repeat_index].sample_processor(rvs=rvs, tag=tag, iteration=iteration, nIter=nIter)
        
    def sample_hierarchical(self, nIter, x0, is_resume=False, share_shuffle_map=False):
        """
        x0 is a skeleton path over observed times
        """
        # T by K
        self.theta = np.empty([self.K])
        # \tau by K
        for m in range(self.M_repeats):
            self.x[m] = np.empty([self.tau, self.K])
            self.x0[m] = x0[m]
        self.nIter = nIter
        self._initialise(is_resume, 'sample')
        
        startTime = int(round(time.time()))
        llhood = -np.Inf
        for i in range(self._current_iter, nIter):
            if share_shuffle_map == True:
                shuffle_map = np.random.permutation(self.K)
            for m in range(self.M_repeats):
                if share_shuffle_map == True:
                    self.x[m] = self.smoothing_kernels[m].sample_non_parallel(self.x[m].copy(), self.theta.copy(), shuffle_map=shuffle_map)[1]
                else:
                    self.x[m] = self.smoothing_kernels[m].sample_non_parallel(self.x[m].copy(), self.theta.copy())[1]
            
            # Check the full path
            #WrightFisherDiffusion.check_path(self.x[i, :,:])
            
            self.theta[:], llhood = self.p_theta.sample_hierarchical(nIter=self.MCMC_in_Gibbs_nIter, theta=self.theta.copy(), Xs=self.x.copy())
            
            endTime = int(round(time.time()))
            print('On iteration {}/{} --- OutterPGAS --- {} -- (llhood = {} )'.format(i+1, nIter, time_string_from_seconds(endTime-startTime), llhood))
            
            # Write-down or otherwise process samples every so iterations
            for m in range(self.M_repeats): 
                self.call_sample_processor([self.theta, self.x[m].copy(), np.array([llhood])], i, nIter, tag='sample', repeat_index=m)

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
        startTime = int(round(time.time()))
        
        print("Predict starting value:")
        print(self.x[:,:])
        
        for i in range(self._current_iter, nIter):
            endTime = int(round(time.time()))
            print('On iteration {}/{} --- OutterPGAS --- {}'.format(i+1, nIter, time_string_from_seconds(endTime-startTime)))
            
            if sample_parallel is False:
                
                self.x[:,:] = self.smoothing_kernel.sample_non_parallel(self.x[:,:].copy(), theta_vector[i-1, ], T_learn = t_learn)[1]
            else:
                self.x[:,:] = self.smoothing_kernel.sample_parallel(self.x[:,:].copy(), theta_vector[i-1, ], T_learn = t_learn)[1]                
                print('Warning! Sampling Parallel!')
            
            self.call_sample_processor(self.x, i, nIter, tag='predict')
            
            # TODO: write-down samples every J iterations
        return(self.x)
    


# ## Proposal distribution for $\Theta$
# ### MH algorithm for resampling $\theta$
# 0. $\theta' \leftarrow \theta_{current}$
# 1. **For** j in 1:J **Do**
#     - $\theta_{new} = \Phi_{\text{adapted}}(. \mid \mu(\theta_{old}), \sigma )$
#     - $a = min(1, \frac{l(\theta_{new})q(\theta_{old} \mid \theta_{new}) }{l(\theta_{old}) q(\theta_{new} \mid \theta_{old})  })   $
#     - $A \sim \text{Bin}(a)$
#     - **If** $A == 1$ **Then**
#         - $\theta' \leftarrow \theta$
#     
# 2. return $\theta'$
# 
# ### Tune the parameter sampling proposal
# One way is to use the method of LibBi https://github.com/sbfnk/RBi.helpers/blob/master/R/adapt_proposal.R and https://github.com/sbfnk/RBi.helpers/blob/master/R/output_to_proposal.R#L65.
# Briefly, change parameters until a desired acceptance rate is achieved or stopping criteria are reached.
# Also see here https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_mcmc_sect022.htm.
# We'll use a Gaussian random walk proposal for each dimension of the parameter space.
# Look at this for alternative ideas: TUNING OF MARKOV CHAIN MONTE CARLO ALGORITHMS USING COPULAS
# Acceptance rate approaches 0.234 in higher dimensions and 0.45 in one dimension (Rosenthal), but any number between 0.15 and 0.5 is at least 80% efficient.
# 
# LibBi: Use sample mean and SD to adapt the current proposal values.
# It could be independent Gaussians or a multivariate one with correlation's taken into account.
# 
# 
# #### Scale tuning
# One way: Suggested by SAS
# $c_{i+1} = \frac{c_i \Phi^{-1}(p_{\text{target}} /2)}{\Phi^{-1}(p_i /2)}$
# where $p_i$ is the current acceptance rate, $p_{\text{target}}$ is the desired acceptance rate, $c_{i}$ and $c_{i+1}$ are current and new scale values respectively.
# 
# Libbi's way:
# std = scale * std(MCMC_sample)
# 
# #### Location tuning
# Do we need this?
# 

# ### Random walk proposal implementation

# In[35]:

class Random_walk_proposal:
    def __init__(self, mu, sigma, upper_bound, lower_bound):
        self.mu = mu
        self.sigma = sigma
        #print('SELF>SIGMA = ', self.sigma)
        self.ub = upper_bound
        self.lb = lower_bound
    
    def compute_loglikelihood(self, loc, value):
        """
        loc is the to be mu on which the value is coditioned
        """
        #print('loc {}, vlaue={}, self.sigma {}'.format(loc, value, self.sigma))
        #sp.stats.truncnorm.logpdf(x=x_M[k], loc=mu, scale=sqrt(sigma2), a=0, b=1)
        #llhoods = np.array([sp.stats.norm.logpdf(x=value, loc=loc, scale=self.sigma)  for val,loc,scale in [value, mu, self.sigma]])
        #seq = map(sp.stats.norm.logpdf, value, loc, self.sigma)
        K = loc.shape[0]
        llhoods = np.empty([K])
        for k in range(K):
            a, b = (self.lb[k] - loc[k]) / self.sigma[k], (self.ub[k] - loc[k]) / self.sigma[k]
            llhoods[k] = sp.stats.truncnorm.logpdf(a=a, b=b, loc=loc[k], x=value[k], scale=self.sigma[k])
        # logpdf(x, a, b, loc=0, scale=1)
        #seq = map(sp.stats.truncnorm.logpdf, value, self.lb, self.ub, loc, self.sigma)
        
        #llhoods = np.fromiter(seq, dtype=np.float128)
        return(np.sum(llhoods))
    def sample(self, loc):
        #print('RW sampling...')
        #return(sp.stats.truncnorm.rvs(loc=mu, scale=sqrt(sigma2), size=1, a=0, b=1))
        #seq = map(sp.stats.norm.rvs, loc, self.sigma, [1]*len(self.sigma) )
        
        # rvs(a, b, loc=0, scale=1, size=1, random_state=None)
        K = loc.shape[0]
        samples = np.empty([K])
        for k in range(K):
            a, b = (self.lb[k] - loc[k]) / self.sigma[k], (self.ub[k] - loc[k]) / self.sigma[k]
            samples[k] = sp.stats.truncnorm.rvs(a=a, b=b,loc=loc[k], scale=self.sigma[k], size=1)
            #samples[k] = np.random.uniform(high=self.ub[k], low=self.lb[k], size=1)
            if samples[k] < self.lb[k] or samples[k] > self.ub[k]:
                raise ValueError('The sample is not in range {}, loc={}, sigma={}'.format(samples[k], loc[k], self.sigma[k]))
        #seq = map(sp.stats.truncnorm.rvs, self.lb, self.ub, loc, self.sigma, [1]*len(self.sigma))
        
        #print('list(map) {}'.format(list(seq)))
        #samples = np.fromiter(np.array(list(seq)).flatt, dtype=np.float64)
        #samples = np.array(np.array(list(seq)).flatten())
        #print('sampels are {}, and bounds = [{}, {}]'.format(samples, self.lb, self.ub))
        return(samples)
        


# In[ ]:




# ### Metropolis-Hastings implementation for proposing $\Theta$

# In[5]:

class MH_Sampler:
    def __init__(self, adapted_proposal_distribution, likelihood_distribution):
        self.phi = adapted_proposal_distribution
        self.l = likelihood_distribution
        
    def sample(self, nIter, theta, x):
        #qq = np.random.uniform(high=1, low=0, size=1)
        #if qq > .8:
        #    print('Using old sampler')
        """
        Even though the proposal is a random walk, since it may be truncated, we won't cancel them out
        x is the scaffold of discritised values
        """
        theta_prime = theta
        llhood_old_cache = None
        
        # Precompute sufficient statistics
        B_inv = self.l.compute_B_inv(x)
        #B_inv = None
        for j in range(nIter):
            theta_new = self.phi.sample(theta_prime)
            # Don't re-compute llhood_old if it hsan't been rejected
            #if llhood_old_cache is None:
            #llhood_old = self.l.compute_loglikelihood(s=theta_prime, x=x);
            llhood_old = self.l.compute_loglikelihood_cache_x(s=theta_prime, x=x, B_inv = B_inv);
            
            #else:
             #   llhood_old = llhood_old_cache
            #llhood_new = self.l.compute_loglikelihood(s=theta_new, x=x);
            llhood_new = self.l.compute_loglikelihood_cache_x(s=theta_new, x=x, B_inv = B_inv);
            
            #if llhood_new == -np.Inf:
            #    llhood_new = -500
            
            #if llhood_old == -np.Inf:
            #    llhood_old = -500
            
            q_llhood = self.phi.compute_loglikelihood(loc=theta_new, value=theta_prime)
            q_llhood_reverse = self.phi.compute_loglikelihood(loc=theta_prime, value=theta_new)
            #print('llhood_new={} - llhood_old={}'.format(llhood_new, llhood_old))
            #A = np.minimum(1.0, np.exp(np.sum(np.array([llhood_new, q_llhood, -llhood_old, -q_llhood_reverse]))))
            # Sohrab: Added back on August 13th
            A = np.minimum(1.0, np.exp(llhood_new + q_llhood - llhood_old - q_llhood_reverse))
            #A = np.minimum(1, np.exp(llhood_new - llhood_old))
            if np.isnan(A):
                print('MH acceptance probability is NaN!')
                print('theta_new = {}, theta_prime = {}'.format(theta_new, theta_prime))
                print('llhood_new={} + q_llhood={} - llhood_old={} - q_llhood_reverse={}'.format(llhood_new, q_llhood, llhood_old, q_llhood_reverse))
                print(x)
                raise ValueError('MH acceptance probability is NaN!')

            
            #print('A is ...', A)
            b = np.random.binomial(size=1, p=A, n=1)
            if b == 1:
                theta_prime = theta_new
                #llhood_old_cache = None
                #print('Accepted theta = {}'.format(theta_prime))
#            else:
                #llhood_old_cache = llhood_old
            
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
            
            #if llhood_new == -np.Inf:
            #    llhood_new = -500
            
            #if llhood_old == -np.Inf:
            #    llhood_old = -500
            
            #q_llhood = self.phi.compute_loglikelihood(loc=theta_new, value=theta_prime)
            #q_llhood_reverse = self.phi.compute_loglikelihood(loc=theta_prime, value=theta_new)
            
            #print('llhood_new={} - llhood_old={}'.format(llhood_new, llhood_old))
            #A = np.minimum(1.0, np.exp(np.sum(np.array([llhood_new, q_llhood, -llhood_old, -q_llhood_reverse]))))
            
            A = np.minimum(1.0, np.exp(llhood_new-llhood_old))
            #A = np.minimum(1, np.exp(llhood_new - llhood_old))
            if np.isnan(A):
                print('MH acceptance probability is NaN!')
                print('theta_new = {}, theta_prime = {}'.format(theta_new, theta_prime))
                print('llhood_new={} + q_llhood={} - llhood_old={} - q_llhood_reverse={}'.format(llhood_new, q_llhood, llhood_old, q_llhood_reverse))
                print(x)
                raise ValueError('MH acceptance probability is NaN!')
            
            #print('A is ...', A)
            b = np.random.binomial(size=1, p=A, n=1)
            if b == 1:
                theta_prime = theta_new
                llhood_old_cache = None
                #print('Accepted theta = {}'.format(theta_prime))
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
            #print('llhood_new={} - llhood_old={}'.format(llhood_new, llhood_old))
            A = np.minimum(1.0, np.exp(np.sum(np.array([llhood_new, q_llhood, -llhood_old, -q_llhood_reverse]))))
            #A = np.minimum(1, np.exp(llhood_new - llhood_old))
            if np.isnan(A):
                print('theta_new = {}, theta_prime = {}'.format(theta_new, theta_prime))
                print('llhood_new={} + q_llhood={} - llhood_old={} - q_llhood_reverse={}'.format(llhood_new, q_llhood, llhood_old, q_llhood_reverse))
                print(Xs)
                raise ValueError('MH acceptance probability is NaN!')
            b = np.random.binomial(size=1, p=A, n=1)
            if b == 1:
                theta_prime = theta_new
                llhood_old_cache = None
                #print('Accepted theta = {}'.format(theta_prime))
            else:
                llhood_old_cache = llhood_old
            
        return([theta_prime, llhood_new])


# In[37]:

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


# 

# # Prediction (forecasting) in time-series using SMC
# ## Outline
# The aim is to predict the future state (configuration) of the system given past observations, 
# that is for some time interval $\tau \gt T$, 
# $$X_{\tau} \mid Y_{0:T}$$
# 
# For a fixed $\theta$, it amounts to 
# $$ p_{\theta}(X_{\tau} \mid Y_{0:T}) = \int p_{\theta}(X_{\tau} \mid X_{T})p_{\theta}(X_{T} \mid Y_{0:T})dX_{T} $$
# 
# When $\theta$ is not fixed, we need to marginalise over it as well, that is:
# $$p(X_{\tau} \mid Y_{0:T}, \theta) = \iint p_{\theta}(X_{\tau} \mid X_{T})p_{\theta}(X_{T} \mid Y_{0:T})dX_{T}d_{\theta} $$
# 
# The prediction task can be imagined as propagating the system without any obserations.
# 
# ## Recipe
# Learn the SSM parameters using the Bayesian learning routine. Then run the particle filter from the begining, each time using a set $\{\Theta_1, \Theta_2, .., \Theta_I\}$, perhaps the last $I$ sampled $\Theta$, and for each, run the PGAS+PR algorithm.
# In each PGAS+PR run, up until the last observation, do as above.
# For the time points with no observation (i.e, $t>T$), weight particles by $1/N$ and stop special treatment of the reference trajectory; alternatively, append a newly sampled state to it.
# 
# This is similar to LibBi's method, where it picks all the Theta's from a previous posterior sample.
# 
# ## Some references
# See _Bayesian Learning and Predictability in a Stochastic Nonlinear Dynamical Model_ https://arxiv.org/abs/1211.1717 and _A sequential Monte Carlo approach for marine ecological prediction_ 
# https://arxiv.org/abs/1211.1717
# an old tutorial on the classic SMC method derivations A Tutorial on Particle Filters for Online
# Nonlinear/Non-Gaussian Bayesian Tracking http://www.irisa.fr/aspi/legland/ref/arulampalam02a.pdf
# 
# ## Pseudo code - Bayesian prediction
# 0. Get $\{\theta\}_{i=0}^{I}$, $T$, and $T_{\text{learn}}$
# 1. Generate $x_\text{reference} = x_{1:T}[0]$, the reference trajectory
# 2. **For** $i$ in $1:I$ **Do**
#     - $x_{1:T}[i] \sim \text{PGAS}(x_{1:T}[i-1], \theta[i], $T_{\text{learn}}$)$
#     
# 
# 

# 
