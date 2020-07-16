import os
import numpy as np
import scipy as sp
import scipy.stats
import math

# exec(open('Utilities.py').read())
#exec(open('Models.py').read())

from blocked_gibbs_sample import *

from Models import WrightFisherDiffusion, _TIME_ROUNDING_ACCURACY


class ConditionalWrightFisherDisffusion(WrightFisherDiffusion):
    def __init__(self, K_prime, K, Ne, h):
        super().__init__(K, Ne, h)
        self.K_prime = K_prime
    
    @staticmethod
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
    
    @staticmethod
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
            deltaW = np.random.normal(0, math.sqrt(self.h[t]), n_particles*tau*K).reshape([n_particles, tau, K])
   
        wf_blocked_gibbs_parallel_ss(tau=tau, X0=X0, time_length=time_length, s=s, h=self.h[t], Ne=self.Ne, Xi=Xi, A=A, x_out=x_out, n_threads=n_threads, deltaW=deltaW)
        

    def _sample_conditional_MVGD(self, mu, sigma, x, partition_size=2):
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
        mu_bar = self._compute_mu_bar(mu=mu_shuffled, sigma=sigma_shuffled, a=a)
        sigma_bar = self._compute_sigma2_bar(sigma=sigma_shuffled, a=a)
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
            B = np.linalg.pinv(sigma_22)
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
                # Compute Sigma_bar
                cov_X = np.outer(Xi[n, step-1, :], Xi[n, step-1, :].T)
                sigma2_bar = np.diag(Xi[n, step-1, :]) - (sigma_pc+1)*cov_X
                the_mean = Xi[n, step-1, ] + mu_bar*self.h[0]
                the_cov = sigma2_bar*self.h[0]
                Xi[n, step, :] = sp.stats.multivariate_normal.rvs(mean=the_mean, cov=the_cov)
                

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
                                        
        if is_last_time_point:
            Xi[:, tau, :] = Xi[:, tau-1, :]
        return([Xi[:, tau, :], -1.24, Xi[:, 1:(tau), :]])   
    