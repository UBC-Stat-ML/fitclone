import math
import os
import pandas as pn
import numpy as np


# Dependencies 
#exec(open('Utilities.py').read())
#exec(open('Models.py').read())
#exec(open('extended_wf_model.py').read())

from Models import _TIME_ROUNDING_ACCURACY, _TOLERATED_ZERO_PRECISION


class ParticleGibbs(object):
    def __init__(self, N, T, K):
        # The N particles for T timepoints of K-dimensional states are kept in a 3 dimensional matrix part[N*T*K]
        self.N = N
        self.T = T
        self.K = K
        self.particles = np.empty([N, T, K])
        self.x = np.empty([self.N, self.T, self.K])
        self.a = np.empty([self.N, self.T], dtype=int)
        self.w = np.empty([self.N, self.T])
        self.tau = 0
        self.loglikelihood = np.empty([self.N, self.T])
        self.Xi = np.empty([self.N, self.tau, self.K])
        self.obs = {}
        self.h = []




       
    def deb_assemble_particles(self, the_slice, t):
        deb_tau = the_slice.stop - the_slice.start + 1
        deb_full_path = np.empty([self.N, deb_tau+1, self.K])
        deb_full_path[0:(self.N-1), 0, :] = self.x[self.a[0:(self.N-1), t], t-1, ]
        deb_full_path[0:(self.N-1), 1:deb_tau, :] = self.Xi[0:(self.N-1), the_slice.start:the_slice.stop, :]
        deb_full_path[0:(self.N-1), deb_tau, :] = self.x[0:(self.N-1), t, :]
        # Reconstruct the path for the Ancestor particle xprime
        deb_full_path[(self.N-1), 0, :] = self.x[self.N-1, t-1, ]
        deb_full_path[(self.N-1), 1:deb_tau, :] = self.Xi[self.N-1, self._Xi_slice(t-1), ]
        deb_full_path[(self.N-1), deb_tau, :] = self.x[self.N-1, t, ]
        return(deb_full_path)
             
        
    # Debug: aseemble particles that have passed, only taking the first tt into account
    # dbc: number of particles passed
    def deb_assemble_passed_particles(self, dbc, tt, passed_index):
        x_result = np.empty([dbc, tt, self.K])
        the_length = 0
        for i in range(tt):
            the_length += len(self._Xi_slice(i))
        x_full_final = np.empty([dbc, the_length, self.K])
        # Record the ancestor indeces
        b_result = np.empty([dbc, tt], dtype=int)
        loop_index = 0
        for k in passed_index:
            x_result[loop_index, tt-1,] = self.x[k, tt-1,]
            b_result[loop_index, tt-1] = k
            curr_index = k
            for t in range(2, tt+1):
                curr_index = self.a[curr_index, tt-t+1]
                x_result[loop_index, tt-t,] = self.x[curr_index, tt-t,]
                b_result[loop_index, tt-t] = curr_index
            loop_index +=1
            
        # Assemble Xi-s
        for loop_index in range(len(passed_index)):
            for i in range(tt):
                if i != tt-1:
                    x_full_final[loop_index, self._Xi_slice(i), ] = self.Xi[b_result[loop_index, i], self._Xi_slice(i), ]
                x_full_final[loop_index, self._t_to_tau(i), ] = x_result[loop_index, i]
        
        return(x_full_final)

    def sample(self, referenceTrajectory):
        self.Xi_prime = referenceTrajectory
        print('this is a sample.')

    def likelihood(self, rvs, observations):
        print('return observation.')
    
    def _get_tau(self):
        return(self.T + sum([len(self._Xi_slice(i)) for i in range(self.T-1)]))
    
     # Indexing helper functions for $\Xi$
    def _t_to_tau(self, t):
   
        # quick hack
        if (t == 0):
            return(int(round(self.obs['times'][0]/self.h[0], _TIME_ROUNDING_ACCURACY)))
        i = 0
        for time in range(1, t+1):
            diff_t = self.obs['times'][time] - self.obs['times'][time-1]
            i += int(round(diff_t/self.h[t-1], _TIME_ROUNDING_ACCURACY))
        return(i)
    
    def _Xi_slice(self, t):
        return(range(self._t_to_tau(t)+1, self._t_to_tau(t+1)))
    
    def _pick_trajectory(self, k):
        x_result = np.empty([self.T, self.K])
        # Record the ancestor indeces
        b_result = np.empty([self.T], dtype=int)
        
        x_result[self.T-1,] = self.x[k, self.T-1,]
        b_result[self.T-1] = k
        curr_index = k
        for t in range(2, self.T+1):
            curr_index = self.a[curr_index, self.T-t+1]
            x_result[self.T-t,] = self.x[curr_index, self.T-t,]
            b_result[self.T-t] = curr_index
        return([x_result, b_result])
        

class PGAS(ParticleGibbs):
    # K dimensions (clones), T timepoints, and N particles
    def __init__(self, N, T, K, bridgeKernel, emissionDistribution, proposalDistribution, transitionDistribution, observations, h, rejuvenaion_prob=1, n_cores=0, disable_ancestor_bridge=False):
        """
        The N particles for T timepoints of K-dimensional states are kept in a 3 dimensional matrix part[N*T*K]
        """
        self.N = N
        self.K = K
        self.T_last_obs = observations['value'].shape[0]
        # Could be the prediction time
        self.T = T 
        self.h = h
        # Set the distributions
        self.r = proposalDistribution
        self.f = transitionDistribution
        self.g = emissionDistribution
        self.bridge = bridgeKernel
        # T by K
        self.obs = observations
        self.rejuvenaion_prob = rejuvenaion_prob
        self.n_cores = n_cores
        self.disable_ancestor_bridge = disable_ancestor_bridge
        self.particle_processor = None
    
    # Take into account the non-equi-distance observation case
    def _init_particles(self):
        self.x = np.empty([self.N, self.T, self.K])
        self.a = np.empty([self.N, self.T], dtype=int)
        self.w = np.empty([self.N, self.T])
        
        # Keep the auxiliary r.v.s.
        self.total_time_length = self.obs['times'][self.T-1] - self.obs['times'][0]
        self.tau = self._get_tau()

        # N by \tau by K (The first ones, Xi[:, self.__Xi_slice(0),], are useless)
        self.Xi = np.empty([self.N, self.tau, self.K])
        self.Xi[:,:,:] = -1.234
        
        # Check if Xi's size is completely covered
        emprical_tau = 1
        for i in range(0, self.T-1):
            emprical_tau += 1 + len(self._Xi_slice(i))
        if self.tau != emprical_tau:
            raise ValueError('Xi is not covered. tau={} != emprical_tau={}'.format(self.tau, emprical_tau))
     
        # llhood of the f(x_t|x_{t-1})
        self.loglikelihood = np.empty([self.N, self.T])
            
    def generate_dummy_trajectory(self, T_learn):
        learn_obs = self.obs['value'][self.obs['times'] < T_learn, ]
        xdummy = np.empty(learn_obs.shape)
        for t in range(xdummy.shape[0]):
            xdummy[t, ] = self.g.sample_posterior(learn_obs[t, ], t)
        return(xdummy)
    
    
    def sample_non_parallel(self, ReferenceTrajectory, theta, T_learn=None):
        self._init_particles()
        # xpime should be a T by K vector
        xprime = ReferenceTrajectory
        self.theta = theta
        # Reserve the full path to return
        x_full_final = np.empty([self.tau, self.K])
        x_full_final[:,:] = -4.23
        
        if x_full_final.shape != xprime.shape:
            raise ValueError('Incomplete Reference trajectory provided... {}, {}', x_full_final.shape, xprime.shape)
        if self.K != self.obs['value'].shape[1]:
            print(self.K, self.obs['value'].shape[1])
            raise ValueError('Inconsistent K in model and observations.')

        # The time after which observations are to be ignored
        if T_learn is None:
            # Then use all observations to predict
            self.T_learn = self.obs['times'][self.T_last_obs-1] + 1.0
        else:
            self.T_learn = T_learn
        
        # Initialization
        self.r.sample_posterior_parallel(observation=self.obs['value'][0,], t=0, X=self.x[0:(self.N-1), 0, :], n_threads=self.n_cores)        
        self.x[self.N-1, 0, ] = xprime[self._t_to_tau(0), ]
        
        llhood_weights = np.zeros([self.N])
        self.g.compute_loglikelihood_parallel(X=self.x[:, 0, :], y=self.obs['value'][0,], t=0, loglikelihood=llhood_weights, n_cores=self.n_cores)
        self.w[:, 0] = np.exp(llhood_weights)
        
        # Ensure weights sum to one
        self.w[:, 0] /= np.sum(self.w[:, 0])
        
        ## DEBUG
        deb_bridge_counts = [None]*(self.T-1)
        
        # Resample, Propagate, Rejuvenate, Weighit
        for t in range(1, self.T):
            is_predicting = (True if self.obs['times'][t] > self.T_learn else False)
            deltaT = self.obs['times'][t]-self.obs['times'][t-1]
            
            # Ensure reference trajectory has values for prediction
            if is_predicting == True and xprime.shape[0] <= self._t_to_tau(t):
                print('Warning! Extending the Reference Trajectory for prediction...')
                xprime_x_t, _dummy, xprime_Xi_t = self.f.sample(xprime[self._t_to_tau(t-1), ].copy(), deltaT, self.theta)
                xprime = np.append(xprime, xprime_Xi_t, xprime_x_t) 
            
            # Pick ancestors for the first N-1 particles
            self.a[0:(self.N-1), t] = np.random.choice(a=range(self.N), p=self.w[:, t-1], size=self.N-1)
            
            # Propagate particles
            the_slice = self._Xi_slice(t-1)
            self.f.sample_parallel(X0=self.x[self.a[0:(self.N-1), t], t-1, ], t = t, 
                                   time_length=deltaT, selection_coeffs=self.theta, seed=1, Xi=self.Xi[0:(self.N-1), the_slice.start:the_slice.stop, :], 
                                   x_out=self.x[0:(self.N-1), t, :],  n_threads=self.n_cores)
                
            # Rejuvenate the ancestor and Xi r.v.s
            if is_predicting is False and np.random.uniform(0, 1, 1) < self.rejuvenaion_prob and self.disable_ancestor_bridge is False:
                self.a[self.N-1, t], xprime[self._Xi_slice(t-1), ] = self.bridge.sample_joint_path_ancestor_not_parallel(obs_index=t, particles=self.x[:,t-1,], initial_weights=self.w[:, t-1], time_length=deltaT, h=self.h[t], xprime=xprime[self._t_to_tau(t),], theta=self.theta)    
            else:
                self.a[self.N-1, t] = np.random.choice(a=range(0, self.N), p=self.w[:, t-1], size=1)
                
            self.x[self.N-1, t, ] = xprime[self._t_to_tau(t), ]
            self.Xi[self.N-1, self._Xi_slice(t-1), ] = xprime[self._Xi_slice(t-1), ]
            
            # Weighting  
            if is_predicting == True:
                self.w[:, t] = 1.0/self.N
            else:
                llhood_weights = np.zeros([self.N])
                self.g.compute_loglikelihood_parallel(X=self.x[:, t, :], y=self.obs['value'][t,], t=t, loglikelihood=llhood_weights, n_cores=self.n_cores)
                self.w[:, t] = np.exp(llhood_weights)
            
            # Report the number of particles that passed through
            do_report = False
            if do_report is True:
                ww1 = self.w[:,t]>0
                ww2 = np.isfinite(self.w[:, t])
                ww3 = np.logical_and(ww1, ww2)
                dbc = np.sum(ww3)
                deb_bridge_counts[t-1] = dbc
                print('# of particles passed through the epsilon ball = {}'.format(dbc))    
           
            # Ensure weights sum to 1
            self.w[:, t] = self.w[:, t]/np.sum(self.w[:, t])    
        
        # Check if the bridge was successfull at all...
        temp =  self.w[:, t].copy()
        # Ensure weights sum to 1
        self.w[:, t] = self.w[:, t]/np.sum(self.w[:, t])
        if all(~np.isfinite(ww) for ww in self.w[:,t]) and is_predicting is True:
            print(temp)
            print(any(temp != 0))
            print(self.w[:,t])
            raise ValueError('All ws are nan')

        # Pick the resulting trajectory
        k = np.random.choice(a=range(self.N), p=self.w[:, self.T-1], size=1)
        
        x_final, b_final = self._pick_trajectory(k)
        
        # Construct the full path 
        for i in range(self.T):
            x_full_final[self._t_to_tau(i), ] = x_final[i, :]
            if i > 0:
                x_full_final[self._Xi_slice(i-1), ] = self.Xi[b_final[i], self._Xi_slice(i-1), ]

        return([x_final, x_full_final, deb_bridge_counts])
    
   
class BlockedPGAS(PGAS):
    def __init__(self, N, T, K_prime, K, bridgeKernel, emissionDistribution, proposalDistribution, transitionDistribution, observations, h, rejuvenaion_prob=1, n_cores=0, disable_ancestor_bridge=False):
        """
        K_prime is the full sise
        K is the block_size
        """
        super().__init__(N, T, K, bridgeKernel, emissionDistribution, proposalDistribution, transitionDistribution, observations, h, rejuvenaion_prob, n_cores, disable_ancestor_bridge)
        self.K_prime = K_prime
  
    def sample_non_parallel(self, xprime_full_path, theta, T_learn=None, shuffle_map=None):
        """
        xprime_full_path is a tau by K vector
        """
        # 1. Pick the partitions
        # 2. Sort s, xprime_full_path, and observations accordingly
        # 3. Comptue X_final and Xi_final
        # 4. Sort X_fianl and Xi_final back to the original
    
        if (self.K_prime != xprime_full_path.shape[1]):
            raise ValueError('xprime_full_path has the wrong size ({} != {})'.format(self.K_prime, xprime_full_path.shape[1]))
        if shuffle_map is None:
            shuffle_map = np.random.permutation(self.K_prime)
        xprime_full_path = xprime_full_path[:, shuffle_map]
        theta = theta[shuffle_map]
        self.obs['value'] = self.obs['value'][:, shuffle_map]
        A = xprime_full_path[:, self.K:self.K_prime].copy()

        self._init_particles()
        # xpime should be a T by K vector
        self.theta = theta

        # Reserve the full path to return
        x_full_final = np.empty([self.tau, self.K])
        x_full_final[:,:] = -4.23

        if x_full_final.shape[0] != xprime_full_path.shape[0]:
            raise ValueError('Incomplete Reference trajectory provided...')

        # The time after which observations are to be ignored
        if T_learn is None:
            # Then use all observations to predict
            self.T_learn = self.obs['times'][self.T_last_obs-1] + 1.0
        else:
            self.T_learn = T_learn

        # Initialization
        self.x[0:(self.N-1), 0, :] = self.r.sample_posterior_vectorised(observation=self.obs['value'][0, 0:self.K], t=0, N=self.N-1, free_freq=1-np.sum(A[0, :])) # HAS TO BE A[0,:], since it's initialisation with observations
        self.x[self.N-1, 0, ] = xprime_full_path[self._t_to_tau(0), 0:self.K]

        llhood_weights = np.zeros([self.N])
        self.g.compute_loglikelihood_parallel(X=self.x[:, 0, :], y=self.obs['value'][0, 0:self.K], t= 0, loglikelihood=llhood_weights, n_cores=self.n_cores)
        self.w[:, 0] = np.exp(llhood_weights)

        # Ensure weights sum to one
        self.w[:, 0] /= np.sum(self.w[:, 0])
    
        deb_bridge_counts = None
        
        # Resample, Propagate, Rejuvenate, Weighit
        for t in range(1, self.T):
            is_predicting = (True if self.obs['times'][t] > self.T_learn else False)
            deltaT = self.obs['times'][t]-self.obs['times'][t-1]

            # Ensure reference trajectory has values for prediction
            if is_predicting == True and xprime_full_path.shape[0] <= self._t_to_tau(t):
                print('Warning! Extending the Reference Trajectory for prediction...')
                xprime_x_t, _dummy, xprime_Xi_t = self.f.sample(xprime_full_path[self._t_to_tau(t-1), ].copy(), deltaT, self.theta)
                xprime_full_path = np.append(xprime_full_path, xprime_Xi_t, xprime_x_t) 

            # Pick ancestors for the first N-1 particles
            self.a[0:(self.N-1), t] = np.random.choice(a=range(0, self.N), p=self.w[:, t-1], size=self.N-1)

            # Propagate particles
            the_slice = self._Xi_slice(t-1)
            self.f.sample_parallel(X0=self.x[self.a[0:(self.N-1), t], t-1, ], t=t,
                                   Xi=self.Xi[0:(self.N-1), the_slice.start:the_slice.stop,:], 
                                   x_out=self.x[0:(self.N-1), t, :], 
                                   A=A[(the_slice.start-1):(the_slice.stop+1), :], 
                                   s=self.theta, time_length=deltaT, seed=1, 
                                   n_threads=self.n_cores, deltaW=None)
            
            # Rejuvenate the ancestor and Xi r.v.s
            if is_predicting is False and np.random.uniform(0, 1, 1) < self.rejuvenaion_prob and self.disable_ancestor_bridge is False:
                self.a[self.N-1, t], xprime_full_path[self._Xi_slice(t-1), 0:self.K] = self.bridge.sample_joint_path_ancestor_blocked(obs_index = t, 
                                                                                                                                      particles=self.x[:,t-1, ], 
                                                                                                                                      initial_weights=self.w[:, t-1], 
                                                                                                                                      time_length=deltaT, h=self.h[t],
                                                                                                                                      xprime=xprime_full_path[self._t_to_tau(t),], theta=self.theta, 
                                                                                                                                      A=A[the_slice.start:(the_slice.stop+2), :], 
                                                                                                                                      shuffle_map=shuffle_map)    
            else:
                self.a[self.N-1, t] = np.random.choice(a=range(0, self.N), p=self.w[:, t-1], size=1)
            self.x[self.N-1, t, ] = xprime_full_path[self._t_to_tau(t), 0:self.K]
            self.Xi[self.N-1, self._Xi_slice(t-1), ] = xprime_full_path[self._Xi_slice(t-1), 0:self.K]
            # Weighting  
            if is_predicting == True:
                self.w[:, t] = 1.0/self.N
            else:
                llhood_weights = np.zeros([self.N])
                self.g.compute_loglikelihood_parallel(X=self.x[:, t, :], y=self.obs['value'][t, 0:self.K], t=t, loglikelihood=llhood_weights, n_cores=self.n_cores)
                self.w[:, t] = np.exp(llhood_weights)

            # Report the number of particles that passed through
            ww1 = self.w[:,t]>0
            ww2 = np.isfinite(self.w[:, t])
            ww3 = np.logical_and(ww1, ww2)
            dbc = np.sum(ww3)
            if (dbc < ww1.shape[0]):
                print('# of particles passed through the epsilon ball = {}'.format(dbc))
                
            # Ensure weights sum to 1
            self.w[:, t] = self.w[:, t]/np.sum(self.w[:, t])   

        # Check if the bridge was successfull at all...
        temp =  self.w[:, t].copy()
        # Ensure weights sum to 1
        self.w[:, t] = self.w[:, t]/np.sum(self.w[:, t])
        if all(~np.isfinite(ww) for ww in self.w[:,t]) and is_predicting is True:
            print(temp)
            print(any(temp != 0))
            print(self.w[:,t])
            raise ValueError('All ws are nan')
            
        
        # Pick the resulting trajectory
        k = np.random.choice(a=range(0, self.N), p=self.w[:, self.T-1], size=1)

        x_final, b_final = self._pick_trajectory(k)

        # Construct the full path 
        for i in range(self.T):
            x_full_final[self._t_to_tau(i), ] = x_final[i, :]
            if i > 0:
                x_full_final[self._Xi_slice(i-1), ] = self.Xi[b_final[i], self._Xi_slice(i-1), ]
        
        x_res = np.empty([x_final.shape[0], self.K_prime])
        x_res_temp = np.empty([x_final.shape[0], self.K_prime])
        for i in range(self.T):
            x_res[i, 0:self.K] = x_final[i]
            x_res[i, self.K:self.K_prime] = xprime_full_path[self._t_to_tau(i), self.K:self.K_prime]
        
        X_full_res = np.empty([x_full_final.shape[0], self.K_prime])
        #print('ERROR! NOT APPLYING THE CHANGE!')
        xprime_full_path[:, 0:self.K] = x_full_final[:, :]
        
        obs_temp = np.empty(self.obs['value'].shape)
        
        # Shuffle back
        for j in range(self.K_prime):
            x_res_temp[:, shuffle_map[j]] = x_res[:, j]
            X_full_res[:, shuffle_map[j]] = xprime_full_path[:, j]
            obs_temp[:, shuffle_map[j]] = self.obs['value'][:, j]
        
        self.obs['value'] = obs_temp
        return([x_res_temp, X_full_res, deb_bridge_counts])


## ParticleBridge kernel implementation
class BridgeKernel():
    def __init__(self):
        print('')
    def sample_joint_path_ancestor(self, time_length, particles, xprime):
        print('sample from the joint distribution of the ancestor and the future states')


class ParticleBridgeKernel(BridgeKernel, ParticleGibbs):
    
    def __init__(self, N, K, transitionDistribution, proposalDistribution, n_cores=1):
        """
        T, depends on the observation times, and will differ for non-equidistant models.
        """
        self.N = N
        self.K = K
        self.f = transitionDistribution
        self.g = proposalDistribution
        self.n_cores = n_cores

    def _init_particles(self, time_length, h):
        # mesh it by timestep
        self.T = int(round(time_length/h, _TIME_ROUNDING_ACCURACY))
        # N by T by K
        self.x = np.empty([self.N, self.T, self.K])
        self.dummy = np.empty([self.N, 0, self.K])
        self.a = np.empty([self.N, self.T], dtype=int)
        self.w = np.empty([self.N, self.T])
        
    def sample_ancestor(self, w, particles, xprime):
        N = len(w)
        print('NOT FINISHED! ONLY USING the ancestor weights.')
        w = w / np.sum(w)
        return (np.random.choice(a=range(0, N), size=1, p=w))
    
    
    def _should_resample(self, t):
        if (t < 1):
            return(False)
        b = np.random.binomial(size=1, p=.4, n=1)
        return(b==0)
        
    
    def sample_joint_path_ancestor_not_parallel(self, obs_index, time_length, h, particles, initial_weights, xprime, theta):
        """
        particles is a N by K vector, each row represeting one of the particles over multiple dimensions
        xprime is a K by 1 vector, representing the endPoint of the Brdige, on which the enclosing PGAS algorithm is conditioned
        """
        self._init_particles(time_length, h)
        self.theta = theta
                
        # Initialization
        self.a[:, 0] = np.random.choice(a=range(self.N), p=initial_weights, size=self.N)
        self.x[:, 0, ] = particles[self.a[:, 0]]
        self.w[:, 0] = 1/self.N

        # Ensure weights sum to one
        self.w[:, 0] /= np.sum(self.w[:, 0])
        
        for t in range(1, self.T):
            # Resample and ancestor sampling
            if self._should_resample(t-1):
                self.a[:, t] = np.random.choice(a=range(self.N), p=self.w[:, t-1]/np.sum(self.w[:, t-1]), size=self.N)
                self.w[:, t] = 1.0/self.N
            else:
                self.a[:, t] = np.array(range(self.N))
                self.w[:, t] = self.w[:, t-1]/np.sum(self.w[:, t-1])
                
            # Propagate particles; e.g., using the WrightFisherDistribution
            self.f.sample_parallel(X0=self.x[self.a[0:self.N, t], t-1,:], t = obs_index, time_length=h, selection_coeffs=self.theta, seed=1, Xi=self.dummy, x_out=self.x[:, t, :], n_threads=self.n_cores)
                
            
            # Weighting
            deltaT = time_length - h*t
            llhood = np.zeros([self.N])
            self.g.compute_loglikelihood_parallel(x_M=xprime, X_t=self.x[:, t-1, :], deltaT=deltaT, llhood=llhood, n_cores=self.n_cores)
            self.w[:, t] = np.divide(np.multiply(np.exp(llhood[:]), self.w[:, t]), self.w[self.a[:, t], t])
            
            
        # pick a random trajectory    
        # Ensure weights sum to one
        self.w[:, self.T-1] /= np.sum(self.w[:, self.T-1])
        k = np.random.choice(a=range(self.N), p=self.w[:, self.T-1], size=1)
        x_final, b_final = self._pick_trajectory(k)
        
        return([b_final[0], x_final[1:,] ])
    

    
    def __propagate_by_gp(self):
        print('not implemented')


class BlockedParticleBridgeKernel(ParticleBridgeKernel):
    
    def __init__(self, N, K, transitionDistribution, proposalDistribution, n_cores=1):
        super().__init__(N, K, transitionDistribution, proposalDistribution, n_cores)
        
    
    def sample_joint_path_ancestor_blocked(self, obs_index, time_length, h, particles, initial_weights, xprime, theta, A, shuffle_map):
        """
        particles is a N by K vector, each row represeting one of the particles over multiple dimensions
        xprime is a K by 1 vector, representing the endPoint of the Brdige, on which the enclosing PGAS algorithm is conditioned
        A is the remainder vector, has shape [1+time_length/h, K_prime-K], 
        """
        self._init_particles(time_length, h)
        self.theta = theta
        # Initialization
        # Hack: Allows one step caching 
        self.dummy = np.empty([self.N, 1, self.K])
        self.dummy[:,:,:] = -4.234
        
        self.a[:, 0] = np.random.choice(a=range(self.N), p=initial_weights, size=self.N)
        self.x[:, 0, ] = particles[self.a[:, 0]]
        self.w[:, 0] = 1/self.N

        # Ensure weights sum to one
        self.w[:, 0] /= np.sum(self.w[:, 0])
        
        for t in range(1, self.T):
            # Resample and ancestor sampling
            if self._should_resample(t-1):
                self.a[:, t] = np.random.choice(a=range(self.N), p=self.w[:, t-1]/np.sum(self.w[:, t-1]), size=self.N)
                self.w[:, t] = 1.0/self.N
            else:
                self.a[:, t] = np.array(range(self.N))
                self.w[:, t] = self.w[:, t-1]/np.sum(self.w[:, t-1])
                
            # Propagate particles
            self.f.sample_parallel(X0=self.x[self.a[0:self.N, t], t-1,:], 
                                   t = obs_index, 
                                   Xi=self.dummy, 
                                   x_out=self.x[:, t, :], 
                                   A=A[(t-1):t+1, :], 
                                   s=self.theta, 
                                   time_length=h, seed=1, n_threads=self.n_cores)

            # Weighting
            deltaT = time_length - h*t
            llhood = np.zeros([self.N])
            # self.g: the GP_sampler
            self.g.compute_loglikelihood_parallel(x_M=xprime, X_t=self.x[:, t-1, :], deltaT=deltaT, llhood=llhood, n_cores=self.n_cores, shuffle_map=shuffle_map)
            self.w[:, t] = np.divide(np.multiply(np.exp(llhood[:]), self.w[:, t]), self.w[self.a[:, t], t])
            
            
        # pick a random trajectory    
        # Ensure weights sum to one
        self.w[:, self.T-1] /= np.sum(self.w[:, self.T-1])
        k = np.random.choice(a=range(self.N), p=self.w[:, self.T-1], size=1)
        x_final, b_final = self._pick_trajectory(k)
        return([b_final[0], x_final[1:,] ])
