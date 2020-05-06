#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import scipy as sp
import scipy.stats
import math

exec(open(os.path.expanduser('Utilities.py')).read())

from epsilon_ball_emission_parallel import *
from epsilon_ball_posterior_parallel import *
from wf_sample_parallel import *
from gaussian_emission_parallel import *

_TIME_ROUNDING_ACCURACY = 7


# In[2]:


class GenericDistribution:
     def __init__(self):
            self.data = 'hello'
            
     def sample(self):
        print('this is a sample.')
        
     def likelihood(self, rvs, observations):
        print('return observation.')


# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# ## Emission model implmentation
# ### $\epsilon$-ball emission

# # String chopping and the emission model
# The current emission model is as follows. This is akin to an ABC procedure, where a synthetic noise model is added to make simulation of bridges in highly informative (directly observed models) easier. 
# \begin{equation}
# X \sim \text{Dir}(1) \\
# y_i \sim \text{Uniform}(x_i - \epsilon, x_i + \epsilon)
# \end{equation}

# One way to sample from the posterior of this distribution  ($p(X_t \mid Y_t)$) is using the string chopping as follows. This covers the case where all the K types are observed. If one is missing, then the sum doesn't have to be exactly one, it should be at most 1.
# 1. Construct boundaries for $y_i$, i.e., $x_i' \in [y_i - \epsilon, y_i + \epsilon]$
# 2. Enforce that $x_j, j \ne i$ sum to one, i.e., $x_i' \in [1-\sum_{j>i}x_j' - \sum_{j > i}y_j-(k-i)\epsilon, 1-\sum_{j>i}x_j' - \sum_{j > i}y_j+(k-i)\epsilon]$
# 3. $x_i'\in [max(y_i - \epsilon, 1-\sum_{j>i}x_j' - \sum_{j > i}y_j-(k-i)\epsilon), min(y_i + \epsilon, 1-\sum_{j>i}x_j' - \sum_{j > i}y_j+(k-i)\epsilon)]$
# 
# This iterative procedure ensures that the sampled $X_i := (x_1, ..., x_K)$ sums to one.
# The likelihood of each sampled vector $X$ would then be $\prod_{i=1}^K \frac{1}{b-a}$ where $a$ and $b$ are the lower and upper bounds in step 3 of the above procedure respectively.
# When $\epsilon$ is small enough, the likelihood of the $X$ vector is approximately Dirichlet.

# In[3]:


class InformativeEmission(GenericDistribution):
    def __init__(self, alpha, epsilon, b=1):
        super().__init__();
        self.alpha = alpha
        self.epsilon = epsilon
        self.x = np.empty(len(alpha))
        self.y = np.empty(len(alpha))
        self.loglikelihood = -math.inf
        self.b = b
        
        # TODO : $\Xi$ auxiliary r.v.s are not the same length for none-equi-distance. observations.
        
        
    def sample(self):
        """
        Samples an x vector from Dirichlet anad then a Y vector from Unif(x+-\epsilon)
        """
        # Sample X from a Dirichlet distribution and then for each x_i, sample y_i from a Unif with \epsilon
        print('self.alpha = {}'.format(self.alpha))
        self.x = np.random.dirichlet(size=1, alpha=self.alpha)
        for index, xi in np.ndenumerate(self.x):
            self.y[index[1]] = np.random.uniform(xi-self.epsilon, xi+self.epsilon)
        return({'obs':self.y, 'x':self.x})
    
    def sample_obs(self, some_x, t):
        result_y = np.empty(shape=some_x.shape)
        for index, xi in np.ndenumerate(some_x):
            #print('xi = {} at index = {}'.format(xi, index))
            result_y[index[0]] = np.random.uniform(xi-self.epsilon, xi+self.epsilon)
        return({'obs':result_y, 'x':some_x})
    
    def sample_posterior(self, observation, t, is_one_missing=True, shuffle=True):
        # Sample from the conditional distribution, given the observation
        #y = np.array(observation)
        y_obs = np.array(observation).copy()
        #print('y_obs is {}'.format(y_obs))
        k = len(y_obs)
        xprime = np.zeros(k)
        
        # Shuffle the vectors to counter bias for the first component
        if shuffle:
            shuffleMap = np.array(list(range(k)), dtype=int)
            np.random.shuffle(shuffleMap)
            y_obs = y_obs[shuffleMap.flatten()]
        
        test_collection = []
        lower_y = lambda i: np.array([max(0.0, y-self.epsilon) for y in y_obs[i:k]])
        upper_y = lambda i: np.array([min(1.0, y+self.epsilon) for y in y_obs[i:k]])    
        
        for i in range(k):
            a = max(0.0, y_obs[i]-self.epsilon)
            a = min(1.0, a)
            b = min(1.0, y_obs[i]+self.epsilon)
            b = max(0.0, b)
            #c = max(0.0, 1 - np.sum(xprime[0:(i+1)]) - np.sum(y_obs[(i+1):k]) - (k-(i+1))*self.epsilon)
            c = max(0.0, 1 - np.sum(xprime[0:i]) - np.sum(upper_y(i+1)))
            c = min(1.0, c)
            #d = min(1.0, 1 - np.sum(xprime[0:(i+1)]) - np.sum(y_obs[(i+1):k]) + (k-(i+1))*self.epsilon)
            d = min(1.0, 1 - np.sum(xprime[0:i]) - np.sum(lower_y(i+1)))
            d = max(0.0, d)
            test_collection.append([a, b, c, d])
            lower_bound = max(a,c) if is_one_missing == False else a
            upper_bound = min(b,d)
            #print([a,c,b,d])
            xprime[i] = np.random.uniform(lower_bound, upper_bound)
        
        if any(t < 0 for t in xprime):
            print('test_collection is {}'.format(test_collection))
            if shuffle:
                print('shuffleMap is {}'.format(shuffleMap))
            print('y_obs is {}'.format(y_obs))
            print('xprime is {}'.format(xprime))
            print('np.sum(xprime) is {}'.format(np.sum(xprime)))
            raise ValueError('x is not positive!')
            
        if any(t > 1 for t in xprime):
            print('test_collection is {}'.format(test_collection))
            if shuffle:
                print('shuffleMap is {}'.format(shuffleMap))
            print('y_obs is {}'.format(y_obs))
            print('xprime is {}'.format(xprime))
            print('np.sum(xprime) is {}'.format(np.sum(xprime)))
            raise ValueError('x is not under 1!')
            
        if np.sum(xprime) > 1.0:
            print('test_collection is {}'.format(test_collection))
            if shuffle:
                print('shuffleMap is {}'.format(shuffleMap))
            print('y_obs is {}'.format(y_obs))
            print('xprime is {}'.format(xprime))
            print('np.sum(xprime) is {}'.format(np.sum(xprime)))
            raise ValueError('x is not Dirichlet!')
            
        if any(t > self.epsilon for t in (abs(xprime - y_obs))):
            print('test_collection is {}'.format(test_collection))
            if shuffle:
                print('shuffleMap is {}'.format(shuffleMap))
            print('y_obs is {}'.format(y_obs))
            print('xprime is {}'.format(xprime))
            print('np.sum(xprime) is {}'.format(np.sum(xprime)))
            raise ValueError('x not in epsilon ball')
        
        #print('input was {}'.format(observation))
        #print('sampled x was {}'.format(xprime))        
        # Revert
        if shuffle:
            mapBack = np.array([np.where(shuffleMap==i) for i in range(k)]).flatten()
            #print('mapBack is {}'.format(mapBack))
            y_obs = y_obs[mapBack]
            xprime = xprime[mapBack]
        
        #WrightFisherDiffusion.check_state(xprime)
        return(xprime)
        
        
    def sample_posterior_vectorised(self, observation, t, N=1, is_one_missing=True, free_freq=1.0):
        #print('free_freq = {}'.format(free_freq))
        # Sample from the conditional distribution, given the observation
        #print('observations = {}'.format(observation))
        y_obs = np.array(observation).copy()
        K = len(y_obs)
        X = np.zeros([N, K])

        # Pre-compute the lower_y and upper_y arrays
        sum_lower_y = np.empty([K])
        sum_upper_y = np.empty([K])
        for i in range(1, K+1):        
            sum_lower_y[i-1] = np.sum(np.array([max(0.0, y-self.epsilon) for y in y_obs[i:K]]))
            sum_upper_y[i-1] = np.sum(np.array([min(free_freq, y+self.epsilon) for y in y_obs[i:K]]))   
        test_collection = []
        
        for n in range(N):
            for i in range(K):
                a = max(0.0, y_obs[i]-self.epsilon)
                a = min(free_freq, a)
                b = min(free_freq, y_obs[i]+self.epsilon)
                b = max(0, b)
                c = max(0.0, free_freq - np.sum(X[n, 0:i]) - sum_upper_y[i])
                c = min(free_freq, c)
                d = min(free_freq, free_freq - np.sum(X[n, 0:i]) - sum_lower_y[i])
                d = max(0.0, d)
                test_collection.append([a, b, c, d])
                lower_bound = max(a,c) if is_one_missing == False else a
                upper_bound = min(b,d)
                X[n, i] = np.random.uniform(lower_bound, upper_bound)

            if np.sum(X[n, ]) > free_freq:
                
                print('n=', n)
                print('(abs(X[n, ] - y_obs))=', (abs(X[n, ] - y_obs)))
                print('a,b,c,d=',test_collection)
                print('free_freq', free_freq)
                print('observation=',observation)
                print('X[n, ]=', X[n,])
                print('np.sum(X[n, ])=', np.sum(X[n, ]))
                raise ValueError('x is not Dirichlet!')

            if any(t > self.epsilon for t in (abs(X[n, ] - y_obs))):
                print(n)
                print((abs(X[n, ] - y_obs)))
                print(test_collection)
                raise ValueError('x not in epsilon ball')
        
        return(X)
    
    def sample_posterior_parallel(self, observation, X, t, is_one_missing=True, n_threads=1):
        #raise ValueError('sample_posterior_parallel')
        epsilon_ball_sample_posterior_parallel(y_obs=np.array(observation), epsilon=self.epsilon, 
                                        unif_rands=np.random.uniform(0,1,X.shape),
                                        X=X, n_threads=n_threads, is_one_missing=(1 if is_one_missing == True else 0))
    
    def compute_loglikelihood(self, params, observation, t, lambdaVal = 10):
        # Compute the likelihood of the parameters, given the observations
        k = len(params)
        g_theta = np.empty(k)
        for i in range(0, k):
            g_theta[i] = 1 if abs(params[i]-observation[i]) <= self.epsilon else self.b*math.exp(-lambdaVal*abs(params[i]-observation[i]))
            #g_theta[i] = 1.0 if abs(params[i]-observation[i]) <= self.epsilon else 0.0
        #print(g_theta)
        #self.loglikelihood = log(np.prod(g_theta))
        self.loglikelihood = np.sum(np.log(g_theta))
        return(self.loglikelihood)
    
    def compute_loglikelihood_vectorise(self, X, y, t, lambdaVal=10):
        N, K = X.shape
        loglikelihood = np.zeros([N])
        for i in range(N):
            for k in range(K):
                if abs(X[i, k]-y[k]) <= self.epsilon:
                    continue
                elif self.b != 0:
                    loglikelihood[i] = loglikelihood[i] - lambdaVal*abs(X[i, k]-y[k])
                else:
                    loglikelihood[i] = -np.inf
                    break
                    
        return(loglikelihood)
    
    
    def compute_loglikelihood_parallel(self, X, y, t, loglikelihood, lambdaVal=10, n_cores=1):
        #epsilon_ball_emission_parallel(X=X, y=y, lambdaVal=lambdaVal, epsilon=self.epsilon, b=self.b, loglikelihood=loglikelihood, n_threads=n_cores)
        gaussian_emission_parallel(X=X, y=y, epsilon=self.epsilon, loglikelihood=loglikelihood, n_threads=n_cores)
    
    def test():
        emission = InformativeEmission(alpha = [1]*4, epsilon=.05)
        xprime = emission.sample_posterior(observation=[0.0657, .728, .095, .103], is_one_missing = False)
        print("{} {}".format(xprime, np.sum(xprime)))
        #xprime = emission.sample_posterior(observation=[0.0657, .095, .728, .103])
        #print(np.sum([0.0657, .728, .095, .103]))
        xprime = emission.sample_posterior(observation=[0.0657, .728, .095], is_one_missing = True)
        print("{} {}".format(xprime, np.sum(xprime)))
        #xprime = emission.sample_posterior(observation=[.2, .2, .2, .2])
        xprime = emission.sample_posterior(observation=[0.0657, .728, .095, .103])
        #xprime = emission.sample_posterior(observation=[0.0657, .095, .728, .103])
        print(np.sum([0.0657, .728, .095, .103]))
        print(xprime)
        print(np.sum(xprime))
        sample_y = emission.sample()
        print(sample_y)
        xpost = emission.sample_posterior(observation=sample_y['obs'])
        print(np.sum(sample_y['obs']))
        print(sample_y['obs'])
        print(xpost)
        print(sample_y['obs']-xpost)
        np.sum(xpost)
        obs = sample_y['obs']

        #emission.compute_loglikelihood(params=xpost, observation=obs)
        #print("the x = {}".format(sample_y['x'])) 
        #print("the obs = {}".format(sample_y['obs'])) 
        type(sample_y['x'])
        type(obs)
        emission.compute_loglikelihood(params=sample_y['x'][0], observation=obs)


# # Just a Gaussian emission (e.g. from bulk)

# References: # https://en.wikipedia.org/wiki/Normal_distribution
# 
# Loglikelihood for a given vector X and set of observations y and a fixed variance $\sigma^2$ \\
# 
# $$llhood(X \mid y, sigma) = log(\prod^{K})  = \frac{-K}{2} ln(2\pi) \frac{-K}{2} ln(\sigma^2) - \frac{1}{2\sigma^2}\sum^K(x_i-y_i)^2 $$

# In[ ]:





# ### Dirichlet-Multinomial emission for DLP sequencing

# In DLP sequencing we assume $N_\text{total}$ cells are extracted from the tumour+normal tissue, sequenced, aligned, and copy number analyzed. 
# Then via phylogenetic analysis and based on copy number data, each cell is assigned to a clone. 
# This will result in $(N_1, N_2, ..., N_K)$.
# 
# \begin{equation}
# \alpha = 1 \\
# X \sim Dir(\alpha) \\
# Y \mid X \sim Mult(N_\text{total}, X)
# \end{equation}
# 
# where $N_\text{total} = \sum_k^K N_k$.
# 
# #### Source of error
# 1. Distorted sampling (including missing underrepresented clones, and biased sampling)
# 2. Miss-assignment of cells to clones (Errors in estimating the phylogenetic tree)
# 3. Lumping clones together (where the cut is made on the phylogenetic tree)
# 
# #### Distorted sampling
# Including biased spatial sampling.
# Due to limited sampling of the cells, about 500 at the moment, we may miss some of the diversity. 
# We assume a Dirichlet-Multinomial observation model. 
# The concentration parameter $\lambda$ accounts for the distortion in sampling, the larger the $\lambda$, the less the distortion is.
# 
# #### Miss-assignment of cells to clones
# Unlike double-barcoding experiments, the true lineage of cells is often not known and needs to be estimated. 
# Errors in the phylogenetic reconstruction distorts the clonal abundances. 
# Methods that provided bootstrap support at each branch can provide a measure of uncertainty. 
# Could inform the hyperparams of the Dirichlet distribution?
# 
# #### Lumping clones together (over-under estimation of K, the number of clones)
# This is when the number of clones/types is not correctly estimated and therefore some interactions are ignored. 
# I'm not sure how to account for this. 
# The most straight forward version would be Bayesian model selection, over some different cuts of the phylogenetic tree (i.e. tryout different K-s.)
# A rule of thumb could be to pick the cut that represents the clones of interest through the time-series. 
# 
# 

# In[50]:


class DirMultEmission(GenericDistribution):
    '''
    Assumes that all observations will be in K-1 dimension, and the value of the last one is encoded in the N_total
    '''
    def __init__(self, N_total, alpha):
        super().__init__();
        self.alpha = np.array(alpha) # This has to have dimension one more than x and Y
        self.N_total = N_total  # Is a vector, one for each timepoint
        self.K = self.alpha.shape[0]-1
        print('SELF.K is , ', self.K)
        print('SELF.N_total is , ', self.N_total)

    def sample(self, t):
        """
        Samples an x vector from Dirichlet anad then a Y vector from Multinomial(x)
        """
        x = np.random.dirichlet(size=1, alpha=self.alpha)
        y = np.random.multinomial(self.N_total[t], self.x, 1)[0:self.K]
        return({'obs':y, 'x':x})

    # NOTE: The y vector will be missing the last one...
    def sample_obs(self, some_x, t):
        # add the last x
        result_y = np.empty(shape=[self.K])
        full_x = np.append(some_x, 1-np.sum(some_x))
        result_y[:] = np.random.multinomial(self.N_total[t], full_x, 1).reshape(self.K+1)[0:self.K]
        return({'obs':result_y, 'x':some_x})

    def sample_posterior(self, observation, t):
        # Sample from the conditional distribution, given the observation
        # X \sim Dir(alpha+Y)
        last_y = self.N_total[t]-np.sum(observation)
        y_obs = np.append(observation, last_y)
        xprime = np.random.dirichlet(size=1, alpha=self.alpha+y_obs).reshape(self.K+1)[0:self.K]
        
        #WrightFisherDiffusion.check_state(xprime)
        return(xprime)

    # Return X: N by T, which are N different params for the same observation
    def sample_posterior_vectorised(self, observation, t, N=1):
        # Sample from the conditional distribution, given the observation
        # X \sim Dir(alpha+Y)
        #print('sample_posterior_vectorised: t=', t)
        last_y = self.N_total[t]-np.sum(observation)
        y_obs = np.append(observation, last_y)
        #X = np.zeros([N, self.K])
        X = np.random.dirichlet(size=N, alpha=self.alpha+y_obs)[:, 0:self.K]
        return(X)

    def sample_posterior_parallel(self, observation, t, X, n_threads=1):
        #print('WARNING, parallel version not IMPLEMENTED YET! Using vectorised...')
        #raise ValueError('sample_posterior_parallel')
        X[:] = self.sample_posterior_vectorised(observation=observation, t=t, N=X.shape[0])
        
    
    # Dirichlet distribution's support is on strictly positive real values
    def replace_zeros(self, X):
        X[X < 0] = 0.0
        # Find zeros in full_X & replace them with .0001
        X[X == 0.0] = .00001
        X = X/np.sum(X)
        return(X)
    
    def compute_loglikelihood(self, params, observation, t):
        X = params
        last_y = self.N_total[t]-np.sum(observation)
        y_obs = np.append(observation, last_y)
        full_X = np.append(X, 1-np.sum(X))
        self.replace_zeros(full_X)
        # Compute the likelihood of the parameters, given the observations
        return(scipy.stats.dirichlet.logpdf(full_X, self.alpha + y_obs))

    def compute_loglikelihood_vectorise(self, X, y, t):
        N = X.shape[0]
        loglikelihood = np.zeros([N])
        last_y = self.N_total[t] - np.sum(y)
        y_obs = np.append(y, last_y)
        dir_mult_alpha = self.alpha + y_obs
        for i in range(N):
            full_X = np.append(X[i, ], 1-np.sum(X[i, ]))
            full_X = self.replace_zeros(full_X)                
            #print(full_X)
            loglikelihood[i] = scipy.stats.dirichlet.logpdf(full_X, dir_mult_alpha)
        return(loglikelihood)


    def compute_loglikelihood_parallel(self, X, y, t, loglikelihood, n_cores=1):
        #raise ValueError('Not implemented!')
        #print('Warning! Not implemented! Using the vecrorised version...')
        loglikelihood[:] = self.compute_loglikelihood_vectorise(X, y, t)


# ### BetaBinomial emission for CRISPR-CAS9 knockout screens (Bulk/PCR sequencing method)
# 
# In these experiments a barcode and CRISPR-CAS9 complex are integrated into a random part of the genome. 
# The complex is then expressed and assembled to mutate a target in the genome, with some unknown efficiency $e_\text{CRISPR}$.
# Then primers are designed to target the barcode (or the double barcode) using PCR-amplification and targeted deep sequencing (depth about 10K-20K reads). 
# The the reads are aligned and those that are $100$ percent match to the barcode are counted and reported. 

# To account for read count overdispersed distribution, we could either use the median normalization and variance modeling of  
# (Li W, Xu H, Xiao T, Cong L, Love MI, Zhang F, Irizarry RA, Liu JS, Brown M, Liu XS. MAGeCK enables robust identification of essential genes from genome-scale CRISPR/Cas9 knockout screens. Genome biology. 2014 Dec 5;15(12):554.)
# or use, similar to PyClone, a product of BetaBinomials. 
# For the latter, the precision $s$ parameter could be fixed for all sgRNAs, or set separately. 
# 
# Library size := Sequencing depth
# 
# Some other normalisation steps:
# Total count (TC): Gene counts are divided by the total number of mapped reads (or library size) associated with their lane and multiplied by the mean total count across all the samples of the dataset.
# 
# If the only considered effect is the library size, we can use a Binomial emission. Any other effect needs to be specified. 
# 

# In[27]:


# import numpy as np
# import scipy.stats
# N = 10
# alpha = np.array([1,1,1,1])
# y_obs = np.array([200, 100, 150, 50])
# kk = np.random.dirichlet(size=N, alpha=alpha+y_obs)


# scipy.stats.dirichlet.logpdf(kk[0, 0:3], alpha+y_obs)


# In[47]:





# In[46]:





# In[ ]:





# ## The Wright-Fisher diffusion approximation
# This is used to propagate particles forward in a Particle filter routine.
# 
# The likelihood, $f(x_t \mid x_{t-1})$ equals the likelihood of the driving noise, namely the underlying multivariate Brownian motion, since given the driving noise, the components are deterministically computed.
# \begin{equation}
# f(x_t \mid x_{t-1}) = \prod_{i=0}^{N-1} \prod_{d=0}^{K-1} \Phi(dW_{i,d} \mid 0, 1)
# \end{equation}
# where $dW_{i,d}$ is the $d$-th component of the standard Brownian motion (i.e, Wiener process) at the $i$-th step of the Euler-Maruyama discretization and 
# $\Phi$ is the conditional density of the standard Normal distribution.
# 
# When only X is given, compute:
# $p(Z_{\tau + \Delta \tau} \mid Z_{\tau}) \approx \mathcal{N} (Z_{\tau} + \mu(Z_\tau) \Delta \tau, \sigma^2 (Z_{\tau}) \Delta \tau)$
# One way of computing this, is to first solve for $dW_1, dW_2, ..., $ algebraically and then compute $\Phi(dW_1, dW_2, ...)$ from a standard normal distribution.
# Alternatively, gather $\sigma^2(X)$ and compute multivariate Gaussian distribution.
# 
# The second approach above seems inefficient, since we already have the inverted covariance matrix form and don't need it to be computed. 
# 
# ## Dealing with instability (ill-conditionedness) of the $\Sigma$ matrix
# We have faced situations where $\Sigma$ is _singular_ and therefore lacks an invert.
# In such cases the Multivariate Normal distribution has no PDF and we'll resort to pseudo-invert and pseudo-determinant for computation of the log-pdf.
# 
# ### Log-likelihood as solution to system of linear equations
# 
# $\Sigma^{-1}.(X_t - (X_{t-1} - \mu(X_{t-1}))= dW $  where $\Sigma$ is the Cholesky decomposition at our disposal, from here we'll denote it $B$.
# llhood = $-dW^T.dW$
# 
# By elimination: 
# 
# $\nu_i = (X_t - (X_{t-1} - \mu(X_{t-1}))_i$
# $$dW_0 = \frac{\nu_0}{B_{0,0}} $$
# $$dW_1 = \frac{\nu_1 - B_{1_0}dW_0}{B_{1,1}}$$
# $$ \cdots $$
# $$dW_{K-1} = \frac{\nu_{K-1} - \Sigma_{j=0}^{K-1} B_{K-1,j}dW_j }{B_{K-1,K-1}}  $$
# 
# **The diagonal elements of $B$, i.e., $B_{i,i} = \sqrt{d_i}$ appear in the denominator and have to be non-zero. **  
# They will be zero only if $x_i = 0$ ($q_i$ has to be zero by that point, since if the 
# 
# 
# In such cases:
# 
# $$  dW = (B[dj>0, dj>0])^{-1} . \nu  $$
# and 
# $$ \text{llhood(dW) = -dW^T.dW} $$
# 
# 
# It occurs when there are zero rows in the lower triangular part of the $B$. 
# These zeros indicate that that dimension has stopped being updated.
# It results in an under-specified system of linear equations. 
# Depending on the underlying reason for the zeros, different remedies may be applied:
# 
# 
# A row i in B will be zero only and only if 
# q_i is zero  
# x_i is zero  
# There may be occasional situations where only a component j of row i is zero, and it's when x_j = 0  
# 
# 
# 
# 

# 
# ## Wright-Fisher diffusion approximation implementation

# In[4]:


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
    return c


# In[ ]:





# In[ ]:





# In[1]:


from math import *
# A factory class to generate WF with desriable K
class WrightFisherDiffusion(GenericDistribution):
    def __init__(self, K, Ne, h):        
        # h is the discritization timestep
        self.h = h
        self.K = K
        self.Ne = Ne
        #print('Warning... Only using h[0] for simulation...')
        self._ne_times_h = Ne*h[0]
    
    def check_state(x):
        theSum = np.sum(x)
        if np.round(theSum, _TIME_ROUNDING_ACCURACY) > 1 or np.round(theSum, _TIME_ROUNDING_ACCURACY) < 0:
            print('theSum is {}'.format(theSum))
            print(x)
            raise ValueError('The path violates the sum rule.')
        
        if np.any([(piece > 1.0 or piece < 0.0) for piece in x]) == True:
            print(x)
            raise ValueError('The path violates simplex condition.')
            
        if np.any([(piece > 0.0 and piece < _TOLERATED_ZERO_PRECISION) for piece in x]) == True:
            print(x)
            raise ValueError('Potential underflow in the path.')
    
    def check_path(x_path):
        global _XX_x
        theT = x_path.shape[0]
        for t in range(theT):
            try:
                WrightFisherDiffusion.check_state(x_path[t, ])
            except ValueError as e:
                # TODO: Check if the fold.change from last state is too large (too small)
                _XX_x = x_path
                print('State at time {} has isses\n{}'.format(t, x_path))
                print("Value error: {}".format(e))
                np.save('/Users/sohrabsalehi/Desktop/revive_tests/bad_x', x_path)
                raise e
    
    def sample(self, x0, time_length, selectionCoefficients, seed=1, deltaWs=None):
        """
        :param x0: Starting frequencies
        :type x0: T by K Numpy.ndarray
        :param time_length: the duration of time to simulate the diffusion, often the time between observation
        :type time_length: float
        :returns: The K-dimensional Wright-Fisher model simulated forward in time for a duration of *time_length* 
        :rtype: T by K Numpy.ndarray
        """
        # Assuming that K = 3 for now
        # TODO: add the simulation class from the code-generator
        # 1. sample a wiener process over the grid
        # 2. numerically advance the states over h timesteps
        #np.random.seed(seed)
        #random.seed(seed)
        self.s = selectionCoefficients
        tau = int((round(time_length/self.h[0], _TIME_ROUNDING_ACCURACY)))
        
        Xi = np.empty([tau+1, self.K])
        Xi[0, ] = x0
        for step in range(1, tau+1):
            if deltaWs is None:
                deltaW = np.random.normal(0, sqrt(self.h[0]), self.K)
            else:
                deltaW = deltaWs[step-1, :]
            _drift = WrightFisherDiffusion.compute_mu(x=Xi[step-1, ], s=self.s, K=self.K, _ne_times_h=self._ne_times_h)
            B = WrightFisherDiffusion.compute_sigma(Xi[step-1, ], self.K)
            # Don't update when dj = 0,
            # WHY? DO UPDATE, MAYBE D_{i, i} = 0 BECAUSE Q_I WAS ZERO, IN WHICH CASE THE PREVIOUS ONE GENERATION HAD REACHED 1. EXAMINE THE CASE WHERE K = 2 AND X_1 = X_2 = 0.5 => B = .5[[1, 0], [-1, 0]]
            #B[np.diag(B) == 0, ] = 0
            _diffusion = np.dot(B, deltaW)
            #print('Single.X = {}'.format(Xi[step-1, ]))
            #print('Single.B = {}'.format(B))
            #print('Single._diffusion = {}'.format(_diffusion))
            #print('Single._drift = {}'.format(_drift))
            deltaX = _drift + _diffusion
            #print('Single._deltaX = {}'.format(deltaX))
            deltaX[_diffusion == 0] = 0

            Xi[step, ] = Xi[step-1, ] + deltaX

            # Enforce boundaries
            xSum = 0
            for i in range(self.K):
                Xi[step, i] = min(1-xSum, min(1.0, max(Xi[step, i], 0.0)))
                xSum = xSum + Xi[step, i]

        return([Xi[tau, ], -1.23, Xi[1:(tau), ]])

    def compute_sigma(x, K):
        q = 1 - np.cumsum(x)
        #print('Single.q.before = {}'.format(q))
        q[q < 0] = 0
        #print('Single.q.after = {}'.format(q))
        q_prime = np.roll(q, 1)
        q_prime[0] = 1.0
        #dj = np.divide(np.multiply(x, q), q_prime)
        #dj[~np.isfinite(dj)] = 0
        sj = np.sqrt(div0(np.multiply(x, q), q_prime))

        #sj_over_q = np.divide(sj, q)
        #sj_over_q[~np.isfinite(sj_over_q)] = 0
        sj_over_q = div0(sj, q)
        B = np.outer(-x.T, sj_over_q)
        B = B*np.tri(*B.shape, k=-1) + np.diag(sj)

        return(B)
    
    def compute_mu(x, s, K, _ne_times_h):
        _mu_fixed = -np.dot(x, s)
        return(np.multiply((_mu_fixed + s), _ne_times_h*x))
                                                                                 
        
    def compute_loglikelihood_old(self, x, s):
        N, K = x.shape
        temp = np.empty([N])
        for i in range(1, N):
            cov_mat = self.compute_sigma2(x[i-1,])*self.h
            #temp[i] = sp.stats.multivariate_normal.logpdf(x=x[i,], mean=x[i-1,]+WrightFisherDiffusion.compute_mu(x[i-1,], s, K, self._ne_times_h), cov=cov_mat, allow_singular=True)
            #temp[i] = sp.stats.multivariate_normal.logpdf(x=x[i,], mean=x[i-1,]+WrightFisherDiffusion.compute_mu(x[i-1,], s, K, self._ne_times_h), cov=cov_mat, allow_singular=False)
            temp[i] = np.log(sp.stats.multivariate_normal.pdf(x=x[i,], mean=x[i-1,]+WrightFisherDiffusion.compute_mu(x[i-1,], s, K, self._ne_times_h), cov=cov_mat, allow_singular=True))
        res = temp.sum()
        return(res)

    
    def hack_for_sa501_dlp(self, x, s):
        N, K = x.shape
        sqrt_h = sqrt(self.h)
        log_2pi = np.log(2*np.pi)
        log_h = np.log(self.h)
        temp = np.empty([N-1])
        x_l = x[0:(N-1), ]
        deltaX = x[1:N, ] - x_l
        _mu_fixed = np.dot(x_l, -s)
        _mu = (np.multiply(x_l.T, _mu_fixed).T + np.multiply(x_l, s))*self._ne_times_h
        _dev = deltaX - _mu
        
        for i in range(0, N-1):
            B = WrightFisherDiffusion.compute_sigma(x[i, ], K)
            mask = np.diag(B) > 0
            B = B[mask, ][:, mask]
            dW = np.matmul(np.linalg.inv(B), _dev[i,][mask])
            temp[i] = -.5*((self.h**-1)*np.dot(dW, dW.T) + K*log_2pi + K*log_h)
            
        return(temp.sum())
        
    
    def compute_loglikelihood_hack(self, x, s):
        ignore_before = 1
        #ignore_before = None
        if (ignore_before is not None):
            N, K = x.shape
            ## SETTING MANUALLY THE CLONE TO BE IGNORED
            # remove the clone
            k = 1
            t1 = int((.46/.78) * N)
            x_before = np.delete(x, k, 1)[0:t1, :]
            s_before = np.delete(s, k, 0)
            x_after = x[t1:N, :]
            ll1 = self.hack_for_sa501_dlp(x_before, s_before)
            ll2 = self.hack_for_sa501_dlp(x_after, s)
            return(ll1+ll2)
        else:
            return(self.hack_for_sa501_dlp(x, s))
            
        
    
    # Used to compute llhood of the trajectory (OuterPGAS) 
    def compute_loglikelihood(self, x, s, ignore_before=None, ignore_after=None):
        #print('compute_loglikelihood...')
        # TODO: check here...
        N, K = x.shape # N here is actaully tau, i.e., number of discretisations
        #print('N of x is {}'.format(N))
        #print('Check if this matches size of h!')
        #print('self.h.shape = {}'.format(self.h.shape))
        #print('##########################\n##########################\n')
        
        # Either N = len(h): using one-step method
        # Or, h is fixed
        _IS_ONE_STEP = False
        
        the_h = self.h
        
        if N == len(self.h):
            _IS_ONE_STEP = True
        else:
            the_h = np.repeat(self.h[0], N) # Just expand the identical value
         
        sqrt_h = np.sqrt(the_h)
        log_h = np.log(the_h)
        log_2pi = np.log(2*np.pi)
        temp = np.empty([N-1])
        x_l = x[0:(N-1), ]
        deltaX = x[1:N, ] - x_l
        _mu_fixed = np.dot(x_l, -s)
        
        if _IS_ONE_STEP is True:
            #print('IS ONE STEP...')
            _mu = (np.multiply(x_l.T, _mu_fixed).T + np.multiply(x_l, s))*self.Ne
            _dev = np.empty(_mu.shape)
            for i in range(0, N-1):
                _dev[i, :] = deltaX[i] - _mu[i]*the_h[i]
        else:
            _mu = (np.multiply(x_l.T, _mu_fixed).T + np.multiply(x_l, s))*self.Ne*the_h[0]
            _dev = deltaX - _mu

        for i in range(0, N-1):
            B = WrightFisherDiffusion.compute_sigma(x[i, ], K)
            mask = np.diag(B) > 0
            B = B[mask, ][:, mask]
            qqq = np.linalg.inv(B)
            dW = np.matmul(qqq, _dev[i,][mask])
            temp[i] = -.5*((the_h[i]**-1)*np.dot(dW, dW.T) + K*log_2pi + K*log_h[i])
            
        return(temp.sum())
    
    # Uses pre-computed sufficient statistics for x
    # See here for a Cythonised version: https://stackoverflow.com/questions/31994879/improving-cython-lapack-performance-with-internal-array-definitions
    def compute_loglikelihood_cache_x(self, x, s, B_inv=None, ignore_before=None, ignore_after=None):
        if B_inv is None:
            return(self.compute_loglikelihood(x, s))
    
        N, K = x.shape # N here is actaully tau, i.e., number of discretisations

        _IS_ONE_STEP = False

        the_h = self.h

        if N == len(self.h):
            _IS_ONE_STEP = True
        else:
            the_h = np.repeat(self.h[0], N) # Just expand the identical value

        sqrt_h = np.sqrt(the_h)
        log_h = np.log(the_h)
        log_2pi = np.log(2*np.pi)
        temp = np.empty([N-1])
        x_l = x[0:(N-1), ]
        deltaX = x[1:N, ] - x_l
        _mu_fixed = np.dot(x_l, -s)

        if _IS_ONE_STEP is True:
            _mu = (np.multiply(x_l.T, _mu_fixed).T + np.multiply(x_l, s))*self.Ne
            _dev = np.empty(_mu.shape)
            for i in range(0, N-1):
                _dev[i, :] = deltaX[i] - _mu[i]*the_h[i]
        else:
            _mu = (np.multiply(x_l.T, _mu_fixed).T + np.multiply(x_l, s))*self.Ne*the_h[0]
            _dev = deltaX - _mu

        for i in range(0, N-1):
            #dW = np.matmul(B_inv[i, :,:], _dev[i,][mask])
            mask = np.diag(B_inv[i, :,:]) > 0
            dW = np.matmul(B_inv[i, :,:][mask, ][:, mask], _dev[i,][mask])
            temp[i] = -.5*((the_h[i]**-1)*np.dot(dW, dW.T) + K*log_2pi + K*log_h[i])
            #temp[i] = -.5*((the_h[i]**-1)*np.dot(dW, dW.T))

        #llhood = temp.sum() - .5*K*(log_2pi + log_h[0])*(N-1)
        '''
        if any(np.isnan(temp)):
            print('llhood is NaN!')
            nan_index = np.where(np.isnan(temp) == True)
            print('NaN index = {}'.format(nan_index))
            print('B_inv at Nan_index = {}'.format(B_inv[nan_index, :, :]))
            np.save('/Users/sohrabsalehi/Desktop/revive_tests/x', x)
        '''
        return(temp.sum())

    def compute_B_inv(self, x):            
        N, K = x.shape # N here is actaully tau, i.e., number of discretisations
        B_inv = np.zeros([N-1, K, K])
        for i in range(0, N-1):
            temp = WrightFisherDiffusion.compute_sigma(x[i, ], K)
            mask = np.diag(temp) > 0
            #B_inv[i, ][np.ix_(np.squeeze(np.where(mask)),np.squeeze(np.where(mask)))] = np.linalg.inv(temp[mask, ][:, mask])
            B_inv[i, ][np.ix_(mask,mask)] = np.linalg.inv(temp[mask, ][:, mask])

        return(B_inv)

    
    def compute_sigma2(self, x):
        return(np.diag(x)-np.outer(x.T, x))

    def set_np_seed():
        if seed is None:
            seed = int(np.random.uniform(0, 1, 1)*10000) 
        np.random.seed(seed)
        
    def sample_vectorised(self, X0, time_length, selectionCoefficients, seed=None, deltaWs=None):
        self.s = selectionCoefficients
        tau = int((round(time_length/self.h, _TIME_ROUNDING_ACCURACY)))
        
        N, _ = X0.shape
        
        Xi = np.empty([N, tau+1, self.K])
        Xi[:, 0, :] = X0
        #print('sample_vectorised: Xi.shape = {}'.format(Xi.shape))
        for step in range(1, tau+1):
            if deltaWs is None:
                deltaW = np.random.normal(0, sqrt(self.h), N*self.K).reshape(N, self.K)
            else:
                deltaW = deltaWs[:, step-1, :]
            #print('deltaW ({}) is {}'.format(deltaWs == None, deltaW))
            x_l = Xi[:, step-1, :]
            _mu_fixed = np.dot(x_l, -self.s)
            _mu = (np.multiply(x_l.T, _mu_fixed).T + np.multiply(x_l, self.s))*self._ne_times_h
            
            for j in range(N):
                B = WrightFisherDiffusion.compute_sigma(x_l[j,], self.K)
                #B[np.diag(B) == 0, ] = 0
                _diffusion = np.dot(B, deltaW[j, :])
                deltaX = _mu[j, ] + _diffusion
                deltaX[_diffusion == 0] = 0
                
                Xi[j, step, ] = Xi[j, step-1, ] + deltaX
                Xi_new = Xi[j, step, ]

                # Enforce boundaries
                # all negative values back to 0
                # when qsum >=1 for i, then all x_j, j>i = 0
                # and x_i = x_i - 1
                #Xi_new[Xi_new < 0] = 0
                #q_new = np.cumsum(Xi_new)
                
                xSum = 0
                for i in range(self.K):
                    Xi[j, step, i] = min(1-xSum, min(1.0, max(Xi[j, step, i], 0.0)))
                    xSum = xSum + Xi[j, step, i]

        return([Xi[:, tau, :], -1.23, Xi[:, 1:(tau), :]])   

    def sample_parallel(self, X0, t, time_length, selection_coeffs, seed, Xi, x_out, n_threads):
        #tau = Xi.shape[1]
        #if time_length/self.h != tau + 1:
        #    print('Warning! Time_length and tau+1 are NOT equal {} != {}.'.format(tau, time_length/self.h))
        wf_sample_parallel(x0=X0, time_length=time_length, s=selection_coeffs, seed=seed, h=self.h[t], Xi=Xi, ne=self.Ne, x_out=x_out,  n_threads=n_threads)
    
    
    def _get_tau(self, T, obsTimes):
        return(T + sum([len(self._Xi_slice(i, obsTimes)) for i in range(T-1)]))
    
    # Indexing helper functions for $\Xi$
    def _t_to_tau(self, t, obsTimes):
        # TODO: BUG: @Sohrab: why is the index not time in the for loop below?
        # quick hack
        if (t == 0):
            return(int(round(obsTimes[t]/self.h[0], _TIME_ROUNDING_ACCURACY)))
        i = 0
        for time in range(1, t+1):
            #print('BUG BUG in _t_to_tau() in Models.')
            #diff_t = obsTimes[t] - obsTimes[t-1]
            diff_t = obsTimes[time] - obsTimes[time-1]
            i += int(round(diff_t/self.h[t-1], _TIME_ROUNDING_ACCURACY))
        return(i)
    
    def _Xi_slice(self, t, obsTimes):
        return(range(self._t_to_tau(t, obsTimes)+1, self._t_to_tau(t+1, obsTimes)))
    
    
    def generate_full_data(self, obsTimes, emisssionDistribution, theta, x0=None, K=2):
        T = len(obsTimes)
        tau = self._get_tau(T, obsTimes)

        x_trajectory = np.empty(shape=[1, T, K])
        x_full_path = np.empty(shape=[1, tau, K])
        x_full_path[:, :, :] = -1.666
        y = np.empty(shape=[T, K])
        
        if x0 is None:
            x_trajectory[0, 0, ] = emisssionDistribution.sample()['x']
        else:
            x_trajectory[0, 0, ] = x0
        
        full_obs_times = np.linspace(0, obsTimes[T-1], num=tau)
        y[0, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, 0, ], t=0)['obs']    
        
        for t in range(1, len(obsTimes)):  
            x_trajectory[0, t, ], _, x_full_path[0, self._Xi_slice(t-1, obsTimes), ] = self.sample(x0=x_trajectory[0, t-1,], time_length=obsTimes[t]-obsTimes[t-1], selectionCoefficients=theta)
                
            x_full_path[0, self._t_to_tau(t, obsTimes),:] = x_trajectory[0, t, :]
            y[t, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, t, ], t=t)['obs']
        
        x_full_path[0, self._t_to_tau(0, obsTimes),] = x_trajectory[0, 0, ]
        return({'x':x_trajectory[0, :,:], 'obs':y, 'x_full_path':x_full_path[0, :,:], 'x_full_times':full_obs_times})
    
    
    def generate_full_data_old(self, obsTimes, emisssionDistribution, theta, x0=None, K=2):
        T = len(obsTimes)
        tau = self._get_tau(T, obsTimes)
        #x_trajectory = np.empty(shape=[T, K])
        x_trajectory = np.empty(shape=[1, T, K])
        x_full_path = np.empty(shape=[1, tau, K])
        x_full_path[:, :, :] = -1.666
        y = np.empty(shape=[T, K])
        if x0 is None:
            x_trajectory[0, 0, ] = emisssionDistribution.sample()['x']
        else:
            x_trajectory[0, 0, ] = x0
        
        #x_full_path = np.empty([tau, K])
        full_obs_times = np.linspace(0, obsTimes[T-1], num=tau)
        y[0, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, 0, ], t=0)['obs']    
        #print('obsTimes[1:] is {}'.format(obsTimes[1:]))
        
        for t in range(1, len(obsTimes)):  
            x_trajectory[0, t, ], _, x_full_path[0, self._Xi_slice(t-1, obsTimes), ] = self.sample(x0=x_trajectory[0, t-1,], time_length=obsTimes[t]-obsTimes[t-1], selectionCoefficients=theta)
            # Use sample_parallel
#             deltaT = obsTimes[t]-obsTimes[t-1]
#             the_slice = self._Xi_slice(t-1, obsTimes)
#             self.sample_parallel(X0 = x_trajectory[:, t-1,], time_length=deltaT, 
#                                  selection_coeffs = np.array(theta), 
#                                  seed=1, 
#                                  Xi = x_full_path[:, the_slice.start:the_slice.stop, :], 
#                                  x_out = x_trajectory[:, t, ], n_threads=1)
         
            x_full_path[0, self._t_to_tau(t, obsTimes),:] = x_trajectory[0, t, :]
            y[t, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, t, ], t=t)['obs']
        
        x_full_path[0, self._t_to_tau(0, obsTimes),] = x_trajectory[0, 0, ]
        
#         if (x_full_path.any(-1.666)):
#             raise ValueError('x_full_path is not fully filled.')
        return({'x':x_trajectory[0, :,:], 'obs':y, 'x_full_path':x_full_path[0, :,:], 'x_full_times':full_obs_times})
    
    def generate_data(self, obsTimes, emisssionDistribution, theta, x0=None, K=2):
        """
        :param obsTimes: the first element has to be 0
        :type obsTimes:
        """
        #print('theta = {}'.format(theta))
        #print('x0 = {}'.format(x0))
        #print('K = {}'.format(K))        
        # returns a dictionary with x and y keys
        T = len(obsTimes)
        x_trajectory = np.empty(shape=[T, K])
        y = np.empty(shape=[T, K])
        if x0 is None:
            x_trajectory[0, ] = emisssionDistribution.sample()['x']
        else:
            x_trajectory[0, ] = x0
            
        y[0, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, ], t=0)['obs']    
        #print('obsTimes[1:] is {}'.format(obsTimes[1:]))
        for t in range(1, len(obsTimes)):  
            x_trajectory[t, ] = self.sample(x0=x_trajectory[t-1,], time_length=obsTimes[t]-obsTimes[t-1], selectionCoefficients=theta)[0]
            y[t, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[t, ], t=t)['obs']
            
        return({'x':x_trajectory, 'obs':y})
    
    
    def generate_full_sample_data(obs_num=5, silent=True, h = .001, selectionCoefficients=[.2, .3], end_time = .1, epsilon = None, x0=[.5, .4], Ne = 200, K=2, Dir_alpha=None, N_total=None):
        WF_diff_gen = WrightFisherDiffusion(h=h, K=K, Ne=Ne)
        
        #print('We are here')
        
        obsTimes = np.linspace(0, end_time, num=obs_num)
        if epsilon is None:
            emission = DirMultEmission(alpha=Dir_alpha, N_total=N_total)
        else:    
            emission = InformativeEmission(alpha=[1]*K, epsilon=epsilon)
        
        dat_gen = WF_diff_gen.generate_full_data(obsTimes=obsTimes, emisssionDistribution=emission, x0=np.array(x0), theta=selectionCoefficients, K=K)
        
        x_gen_full_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['x_full_path'], times=dat_gen['x_full_times'])
        x_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['x'], times=obsTimes)
        y_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['obs'], times=obsTimes)
        
        y_gen_tall['X_true'] = x_gen_tall['X']
        
        if silent is False:
            print(dat_gen)
            TimeSeriesDataUtility.plot_tall(x_gen_tall)
            TimeSeriesDataUtility.plot_tall(y_gen_tall)

        return([y_gen_tall, x_gen_full_tall])
        
    
    def generate_sample_data(obs_num=5, silent=False, h = .001, selectionCoefficients=[.2, .3], end_time = .1, epsilon = .05, x0=[.5, .4], Ne = 200, K=2):
        WF_diff_gen = WrightFisherDiffusion(h=h, K=K, Ne=Ne)
        obsTimes = np.linspace(0, end_time, num=obs_num)
        emission = InformativeEmission(alpha=[1]*K, epsilon=epsilon)
        dat_gen = WF_diff_gen.generate_data(obsTimes = obsTimes, emisssionDistribution=emission, x0=np.array(x0), theta=selectionCoefficients, K=K)
        
        x_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['x'], times=obsTimes)
        y_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['obs'], times=obsTimes)
        y_gen_tall['X_true'] = x_gen_tall['X']
        
        if silent is False:
            print(dat_gen)
            TimeSeriesDataUtility.plot_tall(x_gen_tall)
            TimeSeriesDataUtility.plot_tall(y_gen_tall)

        return(y_gen_tall)


# In[ ]:





# In[4]:





# # Dummy WF with constant variance

# In[5]:


from math import *
class DummyWrightFisherDiffusion(GenericDistribution):
    def __init__(self, K, Ne, h):
        # h is the discritization timestep
        self.h = h
        self.K = K
        self.Ne = Ne
        self._ne_times_h = Ne*h
    
    def check_state(x):
        theSum = np.sum(x)
        if np.round(theSum, _TIME_ROUNDING_ACCURACY) > 1 or np.round(theSum, _TIME_ROUNDING_ACCURACY) < 0:
            print('theSum is {}'.format(theSum))
            print(x)
            raise ValueError('The path violates the sum rule.')
        
        if np.any([(piece > 1.0 or piece < 0.0) for piece in x]) == True:
            print(x)
            raise ValueError('The path violates simplex condition.')
            
        if np.any([(piece > 0.0 and piece < _TOLERATED_ZERO_PRECISION) for piece in x]) == True:
            print(x)
            raise ValueError('Potential underflow in the path.')
    
    def check_path(x_path):
        global _XX_x
        theT = x_path.shape[0]
        for t in range(theT):
            try:
                DummyWrightFisherDiffusion.check_state(x_path[t, ])
            except ValueError as e:
                # TODO: Check if the fold.change from last state is too large (too small)
                _XX_x = x_path
                print('State at time {} has isses\n{}'.format(t, x_path))
                print("Value error: {}".format(e))
                raise e
    
    def sample(self, x0, time_length, selectionCoefficients, seed=1, deltaWs=None):
        """
        :param x0: Starting frequencies
        :type x0: T by K Numpy.ndarray
        :param time_length: the duration of time to simulate the diffusion, often the time between observation
        :type time_length: float
        :returns: The K-dimensional Wright-Fisher model simulated forward in time for a duration of *time_length* 
        :rtype: T by K Numpy.ndarray
        """
        # TODO: add the simulation class from the code-generator
        # 1. sample a wiener process over the grid
        # 2. numerically advance the states over h timesteps
        #np.random.seed(seed)
        #random.seed(seed)
        self.s = selectionCoefficients
        tau = int((round(time_length/self.h, _TIME_ROUNDING_ACCURACY)))
        Xi = np.empty([tau+1, self.K])
        Xi[0, ] = x0
        for step in range(1, tau+1):
            if deltaWs is None:
                deltaW = np.random.normal(0, sqrt(self.h), self.K)
            else:
                deltaW = deltaWs[step-1, :]
            
            _drift = DummyWrightFisherDiffusion.compute_mu(x=Xi[step-1, ], s=self.s, K=self.K, _ne_times_h=self._ne_times_h)
            # _drift[:] =  .1 *self._ne_times_h
            # Just set it to zero
            _drift[:] = 0
            #B = np.eye(self.K)
            B = sqrt(Xi[step-1, ]*(1.0-Xi[step-1, ]))
            
            _diffusion = np.dot(B, deltaW)
            deltaX = _drift + _diffusion
            Xi[step, ] = Xi[step-1, ] + deltaX

        return([Xi[tau, ], -1.23, Xi[1:(tau), ]])

    def compute_sigma(x, K):
        return(np.eye(K))
    
    def compute_mu(x, s, K, _ne_times_h):
        _mu_fixed = -np.dot(x, s)
        return(np.multiply((_mu_fixed + s), _ne_times_h*x))
                                                                                 

    def set_np_seed():
        if seed is None:
            seed = int(np.random.uniform(0, 1, 1)*10000) 
        np.random.seed(seed)
     
    def compute_loglikelihood(self, x, s):
        N, K = x.shape
        temp = np.empty([N])
        for i in range(1, N):
            #cov_mat = self.compute_sigma2(x[i-1,])*self.h
            #temp[i] = np.log(sp.stats.multivariate_normal.pdf(x=x[i,], mean=x[i-1,] + 0.0, cov=cov_mat, allow_singular=True))
            temp[i] = sp.stats.norm.logpdf(x=x[i,], loc=x[i-1,] + 0.0, scale=sqrt(x[i-1, ]*(1.0-x[i-1, ]))*self.h)
        res = temp.sum()
        return(res)
    
    def sample_vectorised(self, X0, time_length, selectionCoefficients, seed=None, deltaWs=None):
        self.s = selectionCoefficients
        tau = int((round(time_length/self.h, _TIME_ROUNDING_ACCURACY)))
        
        N, _ = X0.shape
        Xi = np.empty([N, tau+1, self.K])
        Xi[:, 0, :] = X0
        #print('sample_vectorised: Xi.shape = {}'.format(Xi.shape))
        for step in range(1, tau+1):
            if deltaWs is None:
                deltaW = np.random.normal(0, sqrt(self.h), N*self.K).reshape(N, self.K)
            else:
                deltaW = deltaWs[:, step-1, :]
            #print('deltaW ({}) is {}'.format(deltaWs == None, deltaW))
            x_l = Xi[:, step-1, :]
            _mu_fixed = np.dot(x_l, -self.s)
            _mu = (np.multiply(x_l.T, _mu_fixed).T + np.multiply(x_l, self.s))*self._ne_times_h
            
            # Also, set mu to be fixed
            #_mu[:] =  .1 *self._ne_times_h
            _mu[:] = 0.0
            
            for j in range(N):
                #B = np.eye(self.K)
                if Xi[j, step-1, ] < 0 or Xi[j, step-1, ] > 1:
                    Xi[j, step, ] = Xi[j, step-1, ]
                    Xi_new = Xi[j, step, ]
                else:
                    B = sqrt(Xi[j, step-1, ]*(1.0-Xi[j, step-1, ]))
                    _diffusion = np.dot(B, deltaW[j, :])
                    deltaX = _mu[j, ] + _diffusion

                    Xi[j, step, ] = Xi[j, step-1, ] + deltaX
                    Xi_new = Xi[j, step, ]

                # Don't!!!
                # Enforce boundaries
                # all negative values back to 0
                # when qsum >=1 for i, then all x_j, j>i = 0
                # and x_i = x_i - 1
                #Xi_new[Xi_new < 0] = 0
                #q_new = np.cumsum(Xi_new)
                
#                 xSum = 0
#                 for i in range(self.K):
#                     Xi[j, step, i] = min(1-xSum, min(1.0, max(Xi[j, step, i], 0.0)))
#                     xSum = xSum + Xi[j, step, i]

        return([Xi[:, tau, :], -1.23, Xi[:, 1:(tau), :]])   

    def sample_parallel(self, X0, time_length, selection_coeffs, seed, Xi, x_out, n_threads):
        #tau = Xi.shape[1]
        #if time_length/self.h != tau + 1:
        #    print('Warning! Time_length and tau+1 are NOT equal {} != {}.'.format(tau, time_length/self.h))
        x_out_c, _, Xi_c = self.sample_vectorised(X0=X0.copy(), time_length=time_length, selectionCoefficients=selection_coeffs, seed=seed)
        x_out = x_out_c.copy()
        Xi = Xi_c.copy()
        #wf_sample_parallel(x0=X0, time_length=time_length, s=selection_coeffs, seed=seed, h=self.h, Xi=Xi, ne=self.Ne, x_out=x_out,  n_threads=n_threads)
    
    
    def _get_tau(self, T, obsTimes):
        return(T + sum([len(self._Xi_slice(i, obsTimes)) for i in range(T-1)]))
    
     # Indexing helper functions for $\Xi$
    def _t_to_tau(self, t, obsTimes):
        # quick hack
        if (t == 0):
            return(int(round(obsTimes[t]/self.h, _TIME_ROUNDING_ACCURACY)))
        i = 0
        for time in range(1, t+1):
            diff_t = obsTimes[t] - obsTimes[t-1]
            i += int(round(diff_t/self.h, _TIME_ROUNDING_ACCURACY))
        return(i)
    
    def _Xi_slice(self, t, obsTimes):
        return(range(self._t_to_tau(t, obsTimes)+1, self._t_to_tau(t+1, obsTimes)))
    
    
    def generate_full_data(self, obsTimes, emisssionDistribution, theta, x0=None, K=2):
        T = len(obsTimes)
        tau = self._get_tau(T, obsTimes)
        #x_trajectory = np.empty(shape=[T, K])
        x_trajectory = np.empty(shape=[1, T, K])
        x_full_path = np.empty(shape=[1, tau, K])
        x_full_path[:, :, :] = -1.666
        y = np.empty(shape=[T, K])
        if x0 is None:
            x_trajectory[0, 0, ] = emisssionDistribution.sample()['x']
        else:
            x_trajectory[0, 0, ] = x0
        
        #x_full_path = np.empty([tau, K])
        full_obs_times = np.linspace(0, obsTimes[T-1], num=tau)
        y[0, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, 0, ], t=0)['obs']    
        
        for t in range(1, len(obsTimes)):  
            x_trajectory[0, t, ], _, x_full_path[0, self._Xi_slice(t-1, obsTimes), ] = self.sample(x0=x_trajectory[0, t-1,], time_length=obsTimes[t]-obsTimes[t-1], selectionCoefficients=theta)
            x_full_path[0, self._t_to_tau(t, obsTimes),:] = x_trajectory[0, t, :]
            y[t, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, t, ], t=t)['obs']
        
        x_full_path[0, self._t_to_tau(0, obsTimes),] = x_trajectory[0, 0, ]
        
        return({'x':x_trajectory[0, :,:], 'obs':y, 'x_full_path':x_full_path[0, :,:], 'x_full_times':full_obs_times})
    
    def generate_data(self, obsTimes, emisssionDistribution, theta, x0=None, K=2):
        """
        :param obsTimes: the first element has to be 0
        :type obsTimes:
        """
        #print('theta = {}'.format(theta))
        #print('x0 = {}'.format(x0))
        #print('K = {}'.format(K))        
        # returns a dictionary with x and y keys
        T = len(obsTimes)
        x_trajectory = np.empty(shape=[T, K])
        y = np.empty(shape=[T, K])
        if x0 is None:
            x_trajectory[0, ] = emisssionDistribution.sample()['x']
        else:
            x_trajectory[0, ] = x0
            
        y[0, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, ], t=0)['obs']    
        #print('obsTimes[1:] is {}'.format(obsTimes[1:]))
        for t in range(1, len(obsTimes)):  
            x_trajectory[t, ] = self.sample(x0=x_trajectory[t-1,], time_length=obsTimes[t]-obsTimes[t-1], selectionCoefficients=theta)[0]
            y[t, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[t, ], t=t)['obs']
            
        return({'x':x_trajectory, 'obs':y})
    
    
    def generate_full_sample_data(obs_num=5, silent=True, h = .001, selectionCoefficients=[.2, .3], end_time = .1, epsilon = None, x0=[.5, .4], Ne = 200, K=2, Dir_alpha=None, N_total=None, obsTimes=None):
        WF_diff_gen = DummyWrightFisherDiffusion(h=h, K=K, Ne=Ne)
        if obsTimes is None:
            obsTimes = np.linspace(0, end_time, num=obs_num)
            
        if epsilon is None:
            emission = DirMultEmission(alpha=Dir_alpha, N_total=N_total)
        else:    
            emission = InformativeEmission(alpha=[1]*K, epsilon=epsilon)
        dat_gen = WF_diff_gen.generate_full_data(obsTimes=obsTimes, emisssionDistribution=emission, x0=np.array(x0), theta=selectionCoefficients, K=K)
        
        x_gen_full_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['x_full_path'], times=dat_gen['x_full_times'])
        x_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['x'], times=obsTimes)
        y_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['obs'], times=obsTimes)
        
        y_gen_tall['X_true'] = x_gen_tall['X']
        
        if silent is False:
            print(dat_gen)
            TimeSeriesDataUtility.plot_tall(x_gen_tall)
            TimeSeriesDataUtility.plot_tall(y_gen_tall)

        return([y_gen_tall, x_gen_full_tall])
        
    
    def generate_sample_data(obs_num=5, silent=False, h = .001, selectionCoefficients=[.2, .3], end_time = .1, epsilon = .05, x0=[.5, .4], Ne = 200, K=2):
        WF_diff_gen = DummyWrightFisherDiffusion(h=h, K=K, Ne=Ne)
        obsTimes = np.linspace(0, end_time, num=obs_num)
        emission = InformativeEmission(alpha=[1]*K, epsilon=epsilon)
        dat_gen = WF_diff_gen.generate_data(obsTimes = obsTimes, emisssionDistribution=emission, x0=np.array(x0), theta=selectionCoefficients, K=K)
        
        x_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['x'], times=obsTimes)
        y_gen_tall = TimeSeriesDataUtility.TK_to_tall(dat_gen['obs'], times=obsTimes)
        y_gen_tall['X_true'] = x_gen_tall['X']
        
        if silent is False:
            print(dat_gen)
            TimeSeriesDataUtility.plot_tall(x_gen_tall)
            TimeSeriesDataUtility.plot_tall(y_gen_tall)

        return(y_gen_tall)


# In[78]:


def dummyWF_test():
    dwf = DummyWrightFisherDiffusion(h=.001, K=1, Ne=200)
    res, res_full = DummyWrightFisherDiffusion.generate_full_sample_data(obs_num=5, h = .001, selectionCoefficients=[.2], end_time = .1, epsilon = 0.01, x0=[.5], Ne = 200, K=1)
    s1, _, s2 = dwf.sample_vectorised(X0=np.array([.1]).reshape(-1,1), time_length=.1, selectionCoefficients=np.array([.2]))


# In[ ]:





# In[1]:





# In[12]:





# In[13]:





# In[14]:





# In[17]:





# In[18]:





# In[21]:





# In[ ]:




