
# coding: utf-8

# # Extended Wright-Fisher (WF) model
# ## Main idea
# The K-allele WF model takes a long time or will not converge when K is large. 
# Break each update to two random partitions, (i) a $K'$=3 partition that will be updated, and (ii) a $K'$=K-3 partition that will be held fixed.
# 
# Since the first partition can be depleted, $N_e$ has to be variable. 
# 
# ## Roadmap
# 1. Conceive the discrete process and its conditional distributions
# 2. Derive the diffusion approximation
# 3. Implement
# 
# ## Conceive the discrete process
# ### Generative form
# 0. Initialise
# 1. Sample $N_e \sim P(N_e)$, perhaps an exponential
# 2. Conditioned on this $N_e$, grow the types from a multinomial distribution
# 3. Repeat
# 
# ### Conditional form
# 1. Assume two random partitions of K types, $B_1$ and $B_2$, where $\left\vert{B_1}\right\vert=2$ and $B_2=K \setminus B_1$. 
# 2. Assume a valid vector X is available. 
# Fact: Multinomial sampling is repeated sampling with replacement.  
# Selection coefficients and current prevalence can be combined to give a set $\{b\}_{i=1}^K$ of multinomial *category* probabilities.
# 
# 
# Fix $N_e$
# Fix $B_1$ and $B_2$ partitions
# Run vanila WF and scale by $B_1/Ne$
# 
# 
# 
# 
# 
# 

# In[ ]:




# # Conditional multivariate Gaussian
# 
# Reference: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
# 
# 
# $$ 
# \boldsymbol{x} = 
# \begin{pmatrix}
# \boldsymbol{x_1} \\
# \boldsymbol{x_2} 
# \end{pmatrix}
# $$
# 
# $$
# \boldsymbol{\mu} = 
# \begin{pmatrix}
# \boldsymbol{\mu_1} \\
# \boldsymbol{\mu_2} 
# \end{pmatrix}
# $$
# 
# $$
# \boldsymbol{\Sigma} = 
# \begin{pmatrix}
# \boldsymbol{\Sigma_{11}} && \boldsymbol{\Sigma_{12}} \\
# \boldsymbol{\Sigma_{21}} && \boldsymbol{\Sigma_{22}}
# \end{pmatrix}
# $$
# 
# 
# 
# $$
# \boldsymbol{x_1} \mid \boldsymbol{x_2}=a \sim \mathcal{N}(\bar{\mu}, \bar{\Sigma})
# $$
# 
# 
# $$
# \bar{\mu} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(a-\mu_2)
# $$
# 
# $$
# \bar{\Sigma} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
# $$
# 
# $\bar{\Sigma}$ is the Schur's complement of $\Sigma_{22}$ in $\Sigma$ and thus can be computed by inverting the latter, removing rows and columns corresponding to $x_2$ and inverting back the result.
# 
# ## Rational
# The quantity $\Sigma_{22}$ needs to be computed only once for all the particles.
# Moreover, since only a small dimentionality is used, it is easier to establish a bridge and less particles are needed.
# 

# ## Best computation method
# What is the best way (fastest vs. most accurate) to sample from a conditional multivariate Gaussian distribution?
# 
# ### Test performance
# Start with a **MVGD** (Multivariate Gaussian Distribution) and Gibbs sample the conditional to see if it recovers the true parameters. 
# 

# # Draft implementation

# In[1]:

import os
exec(open(os.path.expanduser('Utilities.py')).read())
exec(open(os.path.expanduser('Models.py')).read())

from blocked_gibbs_sample import *


# In[2]:

def sample_conditional_MVGD(mu, sigma, x, partition_size=2):
    # randomly pick a
    # sort mu and sigma such that a is at the end
    # compute mu_bar and sigma_bar
    # sort the result back 
    K = x.shape[0]
    shuffle_map = np.random.permutation(K)
    x_shuffled = x[shuffle_map]
    a = x_shuffled[(partition_size):K]
    mu_shuffled = mu[shuffle_map]
    sigma_shuffled = sigma[shuffle_map, ][:, shuffle_map]
    mu_bar = compute_mu_bar(mu=mu_shuffled, sigma=sigma_shuffled, a=a)
    sigma_bar = compute_sigma_bar(sigma=sigma_shuffled, a=a)
    x_cond = sp.stats.multivariate_normal.rvs(mean=mu_bar, cov=np.matmul(sigma_bar, sigma_bar.T))
    x_shuffled[0:partition_size] = x_cond
    x_res = np.empty(x.shape)
    for j in range(x_res.shape[0]):
        x_res[shuffle_map[j]] = x_shuffled[j]
    return(x_res)


# In[3]:

def test():
    ### Compute Schur complement step by step
    Ne = 200
    h = .001
    K = 5
    s = np.array([.1, .2, .3, .4, .5])
    x = np.array((np.random.dirichlet([1]*(K+1), 1)[0][0:K]).tolist())
    seed = 10
    np.random.seed(seed)

    wf = WrightFisherDiffusion(h=h, K=K, Ne=Ne)

    mu = WrightFisherDiffusion.compute_mu(_ne_times_h=Ne*h, K=K, s=s, x=x)
    sigma2 = wf.compute_sigma2(K=K, x=x)
    sigma = WrightFisherDiffusion.compute_sigma(x=x, K=K)
    
    compute_mu_bar(mu, sigma, x[2:K])
    compute_sigma_bar(sigma, x[2:K])
    sample_conditional_MVGD(mu=mu, sigma=sigma, partition_size=2, x=x)


# # Implementation of the Conditional Wright-Fisher model
# Only works over multiple particles, to save computational cost of having to compute precision_22
# 
# 1. Randomly pick the two partitions
# 2. 
# 
# ## TODO: make it more memory efficient by making particles only keep the first 2 components
# 
# 
# 
# ## Caching linear algebra operations
# 
# $\bar{\Sigma} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$
# Given that $\Sigma_{12} = \Sigma_{21}^{T}$ and $\Sigma_{12} = [x_0, x_1] \times a$, we'll have:
# $$ \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} =  [x_0, x_1] \times a \times \Sigma_{22}^{-1} \times a^T [x_0, x_1]^{T} = q \times [x_0, x_1] \times [x_0, x_1]^{T} = q\times \text{cov}([x0, x1])$$
# where $q = a \times \Sigma_{22}^{-1} \times a^T$ is a quadratic form (scalar), and
# $cov([x_0, x_1]) = [x_i x_j]_{i,j \in \{0,1\}  }$.
# 
# Observe that $\Sigma_{11} = \text{diag}([x0, x1]) - \text{cov}([x0, x1]) $ thus
# $$\bar{\Sigma} = \text{diag}([x0, x1]) - (q+1)\text{cov}([x0, x1])$$
# 
# ### For $\bar{\mu}$
# 

# In[ ]:




# In[1]:

class ConditionalWrightFisherDisffusion(WrightFisherDiffusion):
    def __init__(self, K_prime, K, Ne, h):
        super().__init__(K, Ne, h)
        self.K_prime = K_prime
    
    def _compute_mu_bar(mu, sigma, a):
        """
        We assume mu and sigam are sorted such that x2 is at the end and is the same length as a
        """
        # \bar{\mu} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(a-\mu_2)
        N = mu.shape[0]
        q = a.shape[0]
        p = N - q
        mu_1 = mu[0:p]
        mu_2 = mu[(p):N]
        sigma_12 = sigma[0:p, (p):N]
        sigma_22 = sigma[(p):N, (p):N]
        mu_bar = mu_1 + np.matmul(np.matmul(sigma_12, np.linalg.pinv(sigma_22)), (a - mu_2))
        return(mu_bar)
    
    def _compute_sigma2_bar(sigma, a):
        """
        We assume mu and sigam are sorted such that x2 is at the end and is the same length as a
        """
        # \bar{\Sigma} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
        N = sigma.shape[0]
        q = a.shape[0]
        p = N - q
        sigma_11 = sigma[0:p, 0:p]
        sigma_12 = sigma[0:p, p:N]
        sigma_21 = sigma[p:N, 0:p]
        sigma_22 = sigma[p:N, p:N]
        sigma_bar = sigma_11 - np.matmul(np.matmul(sigma_12, np.linalg.pinv(sigma_22)), sigma_21)
        return(sigma_bar)
    
    # Note: deltaW has to be ~ Normal(0, sqrt(h)), otherwise the code breaks
    def sample_parallel(self, X0, t, Xi, x_out, A, s, time_length, seed=None, n_threads=1, deltaW=None):
        tau = int((round(time_length/self.h[t], _TIME_ROUNDING_ACCURACY)))
        n_particles, _, K = Xi.shape
        if deltaW is None:
            # So that h er
            deltaW = np.random.normal(0, sqrt(self.h[t]), n_particles*tau*K).reshape([n_particles, tau, K])
         
        #print('tau = {}'.format(tau))
        #print('self.h = {}, self.t = {}, selt.h[t] = {}, tau = {}'.format(self.h[t], t, self.h[t]), tau)
        
#         print('###################')
#         print(tau)
#         print(X0)
#         print(time_length)
#         print(s)
          
            
#         print(self.Ne)
#         print(Xi)
#         print(A)
#         print(x_out)
#         print(n_threads)
#         print(deltaW)        
#         print('###################')
            
        wf_blocked_gibbs_parallel_ss(tau=tau, X0=X0, time_length=time_length, s=s, h=self.h[t], Ne=self.Ne, Xi=Xi, A=A, x_out=x_out, n_threads=n_threads, deltaW=deltaW)
        
#         print('************************')
#         print('************************')
#         print('x_out after parallel is ', x_out)
#         print('Xi after parallel is ', Xi)
#         print('************************')
#         print('************************')
    
    def _sample_conditional_MVGD(mu, sigma, x, partition_size=2):
        # randomly pick a
        # sort mu and sigma such that a is at the end
        # compute mu_bar and sigma_bar
        # sort the result back 
        K = x.shape[0]
        shuffle_map = np.random.permutation(K)
        x_shuffled = x[shuffle_map]
        a = x_shuffled[(partition_size):K]
        mu_shuffled = mu[shuffle_map]
        sigma_shuffled = sigma[shuffle_map, ][:, shuffle_map]
        mu_bar = self.compute_mu_bar(mu=mu_shuffled, sigma=sigma_shuffled, a=a)
        sigma_bar = self.compute_sigma2_bar(sigma=sigma_shuffled, a=a)
        x_cond = sp.stats.multivariate_normal.rvs(mean=mu_bar, cov=sigma_bar)
        x_shuffled[0:partition_size] = x_cond
        x_res = np.empty(x.shape)
        for j in range(x_res.shape[0]):
            x_res[shuffle_map[j]] = x_shuffled[j]
        return(x_res)
    
    def sample_vectorised_dumbest(self, X0, A, s, time_length, seed=None):
        '''
        Like the dumb version, but will just precompute \Sigma_{2,2}^{-1}
        '''
        if (self.K_prime != s.shape[0]):
            raise ValueError('s ({}) has the wrong size ({}).'.format(s.shape[0], self.K_prime))
        if (self.K_prime - self.K != A.shape[1]):
            raise ValueError('A ({}) has the wrong size ({}).'.format(A.shape[1], self.K_prime-self.K))
        tau = int((round(time_length/self.h[0], _TIME_ROUNDING_ACCURACY)))
        N, _ = X0.shape
        Xi = np.empty([N, tau+1, self.K])
        Xi[:, 0, :] = X0
        # A quick hack to handle the case where the last deltaT doesn't exactly coordinate with T_learn
        is_last_time_point = (A.shape[0] == tau)
        for step in range(1, tau+1):
            if step == tau and is_last_time_point:
                break
            free_freq = 1 - np.sum(A[step, ])
            if free_freq < 0:
                print("Warning! Free freq was < 0")
                free_freq = 0
            a = A[step-1, ]
            for n in range(N):
                x = np.append(Xi[n, step-1, ], a)
                # compute mu_bar
                wf = WrightFisherDiffusion(h=self.h, K=self.K_prime, Ne=self.Ne)
                #mu = WrightFisherDiffusion.compute_mu(_ne_times_h=self._ne_times_h, K=self.K_prime, s=s, x=x)
                mu = WrightFisherDiffusion.compute_mu(_ne_times_h=self.Ne, K=self.K_prime, s=s, x=x)
                sigma2 = wf.compute_sigma2(x=x)
                
                mu_bar = ConditionalWrightFisherDisffusion._compute_mu_bar(mu=mu, sigma=sigma2, a=a)
                sigma2_bar = ConditionalWrightFisherDisffusion._compute_sigma2_bar(sigma=sigma2, a=a)
                    
                Xi[n, step, :] = sp.stats.multivariate_normal.rvs(mean=Xi[n, step-1, ] + mu_bar*self.h[0], cov=sigma2_bar*self.h[0])
            
                # Clip the frequencies
                x_sum = 0
                for k in range(self.K):
                    # Don't propagate zeros
                    if Xi[n, step-1, k] == 0:
                        Xi[n, step, k] = 0
                    else:
                        rem = free_freq-x_sum
                        if rem < 0: rem = 0
                        Xi[n, step, k] = min(rem, max(Xi[n, step, k], 0.0))
                        x_sum = x_sum + Xi[n, step, k]
                    
        for n in range(N):            
            WrightFisherDiffusion.check_path(Xi[n, :, :])
        if is_last_time_point:
            print('IS LAST TIME POINT, PASSING OVER...')
            Xi[:, tau, :] = Xi[:, tau-1, :]
        #print(Xi)
        
        return([Xi[:, tau, :], -1.23, Xi[:, 1:(tau), :]])   
    
    def sample_vectorised_dumber(self, X0, A, s, time_length, seed=None):
        if (self.K_prime != s.shape[0]):
            raise ValueError('s ({}) has the wrong size ({}).'.format(s.shape[0], self.K_prime))
        if (self.K_prime - self.K != A.shape[1]):
            raise ValueError('A ({}) has the wrong size ({}).'.format(A.shape[1], self.K_prime-self.K))
        tau = int((round(time_length/self.h, _TIME_ROUNDING_ACCURACY)))
        N, _ = X0.shape
        Xi = np.empty([N, tau+1, self.K])
        Xi[:, 0, :] = X0
        # A quick hack to handle the case where the last deltaT doesn't exactly coordinate with T_learn
        is_last_time_point = (A.shape[0] == tau)
        for step in range(1, tau+1):
            if step == tau and is_last_time_point:
                break
            free_freq = 1 - np.sum(A[step, ])
            if free_freq < 0:
                print("Warning! Free freq was < 0")
                free_freq = 0
            a = A[step-1, ]
            for n in range(N):
                x = np.append(Xi[n, step-1, ], a)
                # compute mu_bar
                wf = WrightFisherDiffusion(h=self.h, K=self.K_prime, Ne=self.Ne)
                #mu = WrightFisherDiffusion.compute_mu(_ne_times_h=self._ne_times_h, K=self.K_prime, s=s, x=x)
                mu = WrightFisherDiffusion.compute_mu(_ne_times_h=self.Ne, K=self.K_prime, s=s, x=x)
                sigma2 = wf.compute_sigma2(x=x)
                
                mu_bar = ConditionalWrightFisherDisffusion._compute_mu_bar(mu=mu, sigma=sigma2, a=a)
                sigma2_bar = ConditionalWrightFisherDisffusion._compute_sigma2_bar(sigma=sigma2, a=a)
                    
                Xi[n, step, :] = sp.stats.multivariate_normal.rvs(mean=Xi[n, step-1, ] + mu_bar*self.h, cov=sigma2_bar*self.h)
            
                # Clip the frequencies
                x_sum = 0
                for k in range(self.K):
                    # Don't propagate zeros
                    if Xi[n, step-1, k] == 0:
                        Xi[n, step, k] = 0
                    else:
                        rem = free_freq-x_sum
                        if rem < 0: rem = 0
                        Xi[n, step, k] = min(rem, max(Xi[n, step, k], 0.0))
                        x_sum = x_sum + Xi[n, step, k]
                    
        for n in range(N):            
            WrightFisherDiffusion.check_path(Xi[n, :, :])
        if is_last_time_point:
            print('IS LAST TIME POINT, PASSING OVER...')
            Xi[:, tau, :] = Xi[:, tau-1, :]
        #print(Xi)
        
        return([Xi[:, tau, :], -1.23, Xi[:, 1:(tau), :]])   
    
    def sample_vectorised_dumb(self, X0, A, s, time_length, seed=None, deltaW=None):
        """
        s is 1 by K_prime
        """
        if (self.K_prime != s.shape[0]):
            raise ValueError('s ({}) has the wrong size ({}).'.format(s.shape[0], self.K_prime))
        if (self.K_prime - self.K != A.shape[1]):
            raise ValueError('A ({}) has the wrong size ({}).'.format(A.shape[1], self.K_prime-self.K))
        tau = int((round(time_length/self.h[0], _TIME_ROUNDING_ACCURACY)))
        N, _ = X0.shape
        Xi = np.empty([N, tau+1, self.K])
        Xi[:, 0, :] = X0
        if deltaW is None:
            deltaW = np.random.normal(0, sqrt(self.h[0]), N*tau*self.K).reshape([N, tau, self.K])
        # A quick hack to handle the case where the last deltaT doesn't exactly coordinate with T_learn
        is_last_time_point = (A.shape[0] == tau)
        for step in range(1, tau+1):
            if step == tau and is_last_time_point:
                break
            free_freq = 1 - np.sum(A[step, ])
            a = A[step-1, ]
            sigma_22 = np.diag(a) - np.outer(a.T, a)
            # Remove those that are zero
            #mask = np.diag(sigma_22) > 0
            #sigma_22 = sigma_22[mask, ][:, mask]
            #B = np.linalg.inv(sigma_22)
            # TODO: @Sohrab: use the mask, keep record of the removed one and assume it is zero in all upcoming updates.
            B = np.linalg.pinv(sigma_22)
            
            # based on state before propagation
            #a = A[step-1, mask]
            
            # Precomputation
            ## For \mu
            mu_pc = np.dot(a, B)
            ## For \sigma
            sigma_pc = np.dot(mu_pc, a.T) # gotta be a scalar

            for n in range(N):
                # Compute mu_bar
                mu_fixed = np.dot(s, np.append(Xi[n, step-1, :], a))
                mu_1 = np.multiply(Xi[n, step-1, :], s[0:self.K]-mu_fixed) * self.Ne # pointwise multiplication
                mu_2 = np.multiply(a, s[self.K:s.shape[0]]-mu_fixed) * self.Ne
                # NOTE: This has to be negative! Precomputation is missing a negative from sigma_12 == -x \times a
                mu_bar = mu_1 - np.matmul(np.outer(Xi[n, step-1, :], mu_pc), (a-mu_2))
                
                #print('mu_bar is ', mu_bar)
                
                # Compute Sigma_bar
                cov_X = np.outer(Xi[n, step-1, :], Xi[n, step-1, :].T)
                sigma2_bar = np.diag(Xi[n, step-1, :]) - (sigma_pc+1)*cov_X
                
                
                #print('sigma2_bar is ', sigma2_bar)
                
                #Xi[n, step, :] = sp.stats.multivariate_normal.rvs(mean=Xi[n, step-1, ] + mu_bar*self._ne_times_h, cov=sigma2_bar*self.h[0])
                the_mean = Xi[n, step-1, ] + mu_bar*self.h[0]
                the_cov = sigma2_bar*self.h[0]
                Xi[n, step, :] = sp.stats.multivariate_normal.rvs(mean=the_mean, cov=the_cov)
                
                ### Testing the equality with the parallel version
#                 normals = deltaW[n, step-1,:]
#                 L = np.linalg.cholesky(the_cov)
#                 diffusion = np.dot(L, normals)
#                 for kk in range(mu_bar.shape[0]):
#                     #diffusion = 0
#                     #for jj in range(mu_bar.shape[0]):
#                     #    diffusion = diffusion + L[kk, jj]*normals[jj]
#                     Xi[n, step, kk] = the_mean[kk] + diffusion[kk]
                
                
                # Clip the frequencies
#                 x_sum = 0
#                 for k in range(self.K):
#                     Xi[n, step, k] = min(free_freq-x_sum, max(Xi[n, step, k], 0.0))
#                     x_sum = x_sum + Xi[n, step, k]
                x_sum = 0
                for k in range(self.K):
                    # Don't propagate zeros
                    if Xi[n, step-1, k] == 0:
                        Xi[n, step, k] = 0
                    else:
                        rem = free_freq-x_sum
                        if rem < 0: rem = 0
                        Xi[n, step, k] = min(rem, max(Xi[n, step, k], 0.0))
                        x_sum = x_sum + Xi[n, step, k]
                                        
        if is_last_time_point:
            Xi[:, tau, :] = Xi[:, tau-1, :]
        return([Xi[:, tau, :], -1.24, Xi[:, 1:(tau), :]])   
    


# In[5]:

# sigma_22 = .03*np.array([[7,-4], [-4, 8]])
# sigma_22_inv = np.linalg.inv(sigma_22)
# sigma_22_inv


# In[6]:

# sigma_12 = np.array([[-3, -4], [-6, -8]])
# sigma_12


# In[7]:

# first_half = np.matmul(sigma_12, sigma_22_inv)
# print(first_half)
# np.matmul(first_half, sigma_12.T)/33.34


# In[ ]:




# In[ ]:




# In[139]:

# Ne = 200
# h = .001
# s = np.array([.1, .1, .1, .1])
# A = np.array([.1, .2, .3, .4])
# K = 2
# K_prime = 4
# X0 = np.array(A[0:K_prime]*1).reshape(1, K_prime)
# x0 = X0[:, 0:K]
# n_particles = 2
# tau = 3
# time_length = tau*h
# Xi = np.empty([n_particles, tau, K_prime])
# x_out = np.empty([n_particles, K_prime])
# np.random.seed(2)
# A = np.asarray([A[K:K_prime]]*tau)


# In[140]:

# print(x0, A)


# In[141]:

# A


# In[142]:

# wf = WrightFisherDiffusion(h=h, K=K_prime, Ne=Ne)
# wf.sample_parallel(X0=X0, time_length=time_length, selection_coeffs=s, seed=1, Xi=Xi, x_out=x_out, n_threads=1)
# print(Xi, x_out)


# In[143]:

# X = np.empty([Xi.shape[0], Xi.shape[1]+2, Xi.shape[2]])
# X[:, 1:Xi.shape[1]+1, :] = Xi
# X[:, 0, :] = x0
# X[:, Xi.shape[1]+1, :] = x_out
# X


# In[76]:

# x0.shape
# A
# x_out


# In[77]:




# In[ ]:




# In[46]:

# A = x0[K:K_prime].reshape(1, K_prime-K)
# A = X[0,:, K:K_prime]
# print(A.shape)
# A


# In[122]:

#cwf = ConditionalWrightFisherDisffusion(h=h, Ne=Ne, K=K, K_prime=K_prime)


# In[132]:

#x, _, Xi = cwf.sample_vectorised_dumb(A=A[0:7, ], s=s, seed=2, time_length=7*h, X0=x0)
#print(x, Xi, X0)


# In[117]:




# In[93]:

#np.sum(x)
# print(x0)
# print(Xi)
# print(x)


# In[15]:

# array([[[ 0.1       ,  0.2       ,  0.3       ,  0.4       ],
#         [ 0.09335522,  0.1960415 ,  0.2968504 ,  0.41375287],
#         [ 0.10162158,  0.17878144,  0.29136166,  0.42823532],
#         [ 0.09497842,  0.17138128,  0.28902201,  0.44461829],
#         [ 0.09712859,  0.1752141 ,  0.28758534,  0.44007198],
#         [ 0.10309451,  0.15228197,  0.29101199,  0.45361153],
#         [ 0.11896263,  0.14326119,  0.30192772,  0.43584846]]])

