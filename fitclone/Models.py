import os
import random
import numpy as np
import scipy as sp
import scipy.stats
import math
#import pyximport

# exec(open('Utilities.py').read())
from Utilities import TimeSeriesDataUtility

#from epsilon_ball_posterior_parallel import epsilon_ball_sample_posterior_parallel
#from gaussian_emission_parallel import gaussian_emission_parallel

    
from epsilon_ball_emission_parallel import *
from epsilon_ball_posterior_parallel import *
from wf_sample_parallel import *
from gaussian_emission_parallel import *
from gp_llhood_parallel import *

# Global Constants
_TIME_ROUNDING_ACCURACY = 7
_TOLERATED_ZERO_PRECISION = 1e-20


class GenericDistribution:
    def __init__(self):
        self.data = ''
        self.alpha = 0
        self.epsilon = 0
        self.x = 0
        self.y = 0
        self.loglikelihood = 0
        self.b = 0
            
    def sample(self):
        print('this is a sample.')
        
    def likelihood(self, rvs, observations):
        print('return observation.')



class InformativeEmission(GenericDistribution):
    def __init__(self, alpha, epsilon, b=1):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.x = np.empty(len(alpha))
        self.y = np.empty(len(alpha))
        self.loglikelihood = -math.inf
        self.b = b
        
    def sample(self):
        """
        Samples an x vector from Dirichlet anad then a Y vector from Unif(x+-epsilon)
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
            result_y[index[0]] = np.random.uniform(xi-self.epsilon, xi+self.epsilon)
        return({'obs':result_y, 'x':some_x})
    
    def sample_posterior(self, observation, t, is_one_missing=True, shuffle=True):
        # Sample from the conditional distribution, given the observation
        y_obs = np.array(observation).copy()
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
            c = max(0.0, 1 - np.sum(xprime[0:i]) - np.sum(upper_y(i+1)))
            c = min(1.0, c)
            d = min(1.0, 1 - np.sum(xprime[0:i]) - np.sum(lower_y(i+1)))
            d = max(0.0, d)
            test_collection.append([a, b, c, d])
            lower_bound = max(a,c) if is_one_missing == False else a
            upper_bound = min(b,d)
            xprime[i] = np.random.uniform(lower_bound, upper_bound)
        
        if any(t < 0 for t in xprime):
            raise ValueError('x is not positive!')
            
        if any(t > 1 for t in xprime):
            raise ValueError('x is not under 1!')
            
        if np.sum(xprime) > 1.0:
            raise ValueError('x is not Dirichlet!')
            
        if any(t > self.epsilon for t in (abs(xprime - y_obs))):
            raise ValueError('x not in epsilon ball')
        
        # Revert
        if shuffle:
            mapBack = np.array([np.where(shuffleMap==i) for i in range(k)]).flatten()
            y_obs = y_obs[mapBack]
            xprime = xprime[mapBack]
        return(xprime)
        
        
    def sample_posterior_vectorised(self, observation, t, N=1, is_one_missing=True, free_freq=1.0):
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
                raise ValueError('x is not Dirichlet!')

            if any(t > self.epsilon for t in (abs(X[n, ] - y_obs))):
                raise ValueError('x not in epsilon ball')
        
        return(X)
    
    def sample_posterior_parallel(self, observation, X, t, is_one_missing=True, n_threads=1):
        epsilon_ball_sample_posterior_parallel(y_obs=np.array(observation), epsilon=self.epsilon, 
                                        unif_rands=np.random.uniform(0,1,X.shape),
                                        X=X, n_threads=n_threads, is_one_missing=(1 if is_one_missing == True else 0))
    
    def compute_loglikelihood(self, params, observation, t, lambdaVal = 10):
        # Compute the likelihood of the parameters, given the observations
        k = len(params)
        g_theta = np.empty(k)
        for i in range(0, k):
            g_theta[i] = 1 if abs(params[i]-observation[i]) <= self.epsilon else self.b*math.exp(-lambdaVal*abs(params[i]-observation[i]))
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
        gaussian_emission_parallel(X=X, y=y, epsilon=self.epsilon, loglikelihood=loglikelihood, n_threads=n_cores)
    
    @staticmethod
    def test():
        emission = InformativeEmission(alpha = [1]*4, epsilon=.05)
        t = 1
        xprime = emission.sample_posterior(observation=[0.0657, .728, .095, .103], t = t, is_one_missing = False)
        print("{} {}".format(xprime, np.sum(xprime)))
        xprime = emission.sample_posterior(observation=[0.0657, .728, .095], t = t, is_one_missing = True)

        xprime = emission.sample_posterior(observation=[0.0657, .728, .095, .103], t = 1)
        print(np.sum([0.0657, .728, .095, .103]))
        print(xprime)
        print(np.sum(xprime))
        sample_y = emission.sample()
        print(sample_y)
        xpost = emission.sample_posterior(observation=sample_y['obs'], t = t)
        print(np.sum(sample_y['obs']))
        print(sample_y['obs'])
        print(xpost)
        print(sample_y['obs']-xpost)
        np.sum(xpost)
        obs = sample_y['obs']

        type(sample_y['x'])
        type(obs)
        emission.compute_loglikelihood(params=sample_y['x'][0], observation=obs, t = t)




class DirMultEmission(GenericDistribution):
    '''
    Assumes that all observations will be in K-1 dimension, and the value of the last one is encoded in the N_total
    '''
    def __init__(self, N_total, alpha):
        super().__init__()
        self.alpha = np.array(alpha) # This has to have dimension one more than x and Y
        self.N_total = N_total  # Is a vector, one for each timepoint
        self.K = self.alpha.shape[0]-1

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
        
        return(xprime)

    # Return X: N by T, which are N different params for the same observation
    def sample_posterior_vectorised(self, observation, t, N=1):
        # Sample from the conditional distribution, given the observation
        # X \sim Dir(alpha+Y)
        last_y = self.N_total[t]-np.sum(observation)
        y_obs = np.append(observation, last_y)
        X = np.random.dirichlet(size=N, alpha=self.alpha+y_obs)[:, 0:self.K]
        return(X)

    def sample_posterior_parallel(self, observation, t, X, n_threads=1):
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
            loglikelihood[i] = scipy.stats.dirichlet.logpdf(full_X, dir_mult_alpha)
        return(loglikelihood)


    def compute_loglikelihood_parallel(self, X, y, t, loglikelihood, n_cores=1):
        loglikelihood[:] = self.compute_loglikelihood_vectorise(X, y, t)



def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
    return c

# A factory class to generate WF with desriable K
class WrightFisherDiffusion(GenericDistribution):
    def __init__(self, K, Ne, h):        
        # h is the discritization timestep
        self.h = h
        self.K = K
        self.Ne = Ne
        self._ne_times_h = Ne*h[0]
    
    @staticmethod
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
    
    @staticmethod
    def check_path(x_path: np.ndarray) -> None:
        theT = x_path.shape[0]
        for t in range(theT):
            try:
                WrightFisherDiffusion.check_state(x_path[t, ])
            except ValueError as e:
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
        # 1. sample a wiener process over the grid
        # 2. numerically advance the states over h timesteps
        self.s = selectionCoefficients
        tau = int((round(time_length/self.h[0], _TIME_ROUNDING_ACCURACY)))
        
        Xi = np.empty([tau+1, self.K])
        Xi[0, ] = x0
        for step in range(1, tau+1):
            if deltaWs is None:
                deltaW = np.random.normal(0, math.sqrt(self.h[0]), self.K)
            else:
                deltaW = deltaWs[step-1, :]
            _drift = WrightFisherDiffusion.compute_mu(x=Xi[step-1, ], s=self.s, K=self.K, _ne_times_h=self._ne_times_h)
            B = WrightFisherDiffusion.compute_sigma(Xi[step-1, ], self.K)
            _diffusion = np.dot(B, deltaW)
            deltaX = _drift + _diffusion
            deltaX[_diffusion == 0] = 0

            Xi[step, ] = Xi[step-1, ] + deltaX

            # Enforce boundaries
            xSum = 0
            for i in range(self.K):
                Xi[step, i] = min(1-xSum, min(1.0, max(Xi[step, i], 0.0)))
                xSum = xSum + Xi[step, i]

        return([Xi[tau, ], -1.23, Xi[1:(tau), ]])

    @staticmethod
    def compute_sigma(x, K):
        q = 1 - np.cumsum(x)
        q[q < 0] = 0
        q_prime = np.roll(q, 1)
        q_prime[0] = 1.0
        sj = np.sqrt(div0(np.multiply(x, q), q_prime))
        sj_over_q = div0(sj, q)
        B = np.outer(-x.T, sj_over_q)
        B = B*np.tri(*B.shape, k=-1) + np.diag(sj)

        return(B)
    
    @staticmethod
    def compute_mu(x, s, K, _ne_times_h):
        _mu_fixed = -np.dot(x, s)
        return(np.multiply((_mu_fixed + s), _ne_times_h*x))
                                                                                 
        
    def compute_loglikelihood_old(self, x, s):
        N, K = x.shape
        temp = np.empty([N])
        for i in range(1, N):
            cov_mat = self.compute_sigma2(x[i-1,])*self.h
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
            
        
    
    def compute_loglikelihood(self, x, s, ignore_before=None, ignore_after=None):
        """ Used to compute llhood of the trajectory (OuterPGAS) """
        N, K = x.shape # N here is actaully tau, i.e., number of discretisations
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
            mask = np.diag(B_inv[i, :,:]) > 0
            dW = np.matmul(B_inv[i, :,:][mask, ][:, mask], _dev[i,][mask])
            temp[i] = -.5*((the_h[i]**-1)*np.dot(dW, dW.T) + K*log_2pi + K*log_h[i])
     
        return(temp.sum())

    def compute_B_inv(self, x):            
        N, K = x.shape # N here is actaully tau, i.e., number of discretisations
        B_inv = np.zeros([N-1, K, K])
        for i in range(0, N-1):
            temp = WrightFisherDiffusion.compute_sigma(x[i, ], K)
            mask = np.diag(temp) > 0
            B_inv[i, ][np.ix_(mask,mask)] = np.linalg.inv(temp[mask, ][:, mask])

        return(B_inv)

    
    def compute_sigma2(self, x):
        return(np.diag(x)-np.outer(x.T, x))

    @staticmethod
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
        for step in range(1, tau+1):
            if deltaWs is None:
                deltaW = np.random.normal(0, math.sqrt(self.h), N*self.K).reshape(N, self.K)
            else:
                deltaW = deltaWs[:, step-1, :]
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
                xSum = 0
                for i in range(self.K):
                    Xi[j, step, i] = min(1-xSum, min(1.0, max(Xi[j, step, i], 0.0)))
                    xSum = xSum + Xi[j, step, i]

        return([Xi[:, tau, :], -1.23, Xi[:, 1:(tau), :]])   

    def sample_parallel(self, X0, t, time_length, selection_coeffs, seed, Xi, x_out, n_threads):
        wf_sample_parallel(x0=X0, time_length=time_length, s=selection_coeffs, seed=seed, h=self.h[t], Xi=Xi, ne=self.Ne, x_out=x_out,  n_threads=n_threads)
    
    
    def _get_tau(self, T, obsTimes):
        return(T + sum([len(self._Xi_slice(i, obsTimes)) for i in range(T-1)]))
    
    # Indexing helper functions for $\Xi$
    def _t_to_tau(self, t, obsTimes):
        # quick hack
        if (t == 0):
            return(int(round(obsTimes[t]/self.h[0], _TIME_ROUNDING_ACCURACY)))
        i = 0
        for time in range(1, t+1):
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
            # Use sample_parallel
            x_full_path[0, self._t_to_tau(t, obsTimes),:] = x_trajectory[0, t, :]
            y[t, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, t, ], t=t)['obs']
        
        x_full_path[0, self._t_to_tau(0, obsTimes),] = x_trajectory[0, 0, ]
        
        return({'x':x_trajectory[0, :,:], 'obs':y, 'x_full_path':x_full_path[0, :,:], 'x_full_times':full_obs_times})
    
    def generate_data(self, obsTimes, emisssionDistribution, theta, x0=None, K=2):
        """
        :param obsTimes: the first element has to be 0
        :type obsTimes:
        """
        # returns a dictionary with x and y keys
        T = len(obsTimes)
        x_trajectory = np.empty(shape=[T, K])
        y = np.empty(shape=[T, K])
        if x0 is None:
            x_trajectory[0, ] = emisssionDistribution.sample()['x']
        else:
            x_trajectory[0, ] = x0
            
        y[0, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[0, ], t=0)['obs']    
    
        for t in range(1, len(obsTimes)):  
            x_trajectory[t, ] = self.sample(x0=x_trajectory[t-1,], time_length=obsTimes[t]-obsTimes[t-1], selectionCoefficients=theta)[0]
            y[t, ] = emisssionDistribution.sample_obs(some_x = x_trajectory[t, ], t=t)['obs']
            
        return({'x':x_trajectory, 'obs':y})
    
    
    @staticmethod
    def generate_full_sample_data(obs_num=5, silent=True, h = .001, selectionCoefficients=[.2, .3], end_time = .1, epsilon = None, x0=[.5, .4], Ne = 200, K=2, Dir_alpha=None, N_total=None):
        WF_diff_gen = WrightFisherDiffusion(h=h, K=K, Ne=Ne)
        
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
        
    @staticmethod
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
