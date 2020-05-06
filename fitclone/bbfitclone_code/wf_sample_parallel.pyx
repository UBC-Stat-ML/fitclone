import numpy as np
cimport numpy as np
cimport openmp
import cython
from cython.parallel import prange, parallel

from libc.math cimport sqrt
ctypedef np.float_t DTYPE_t
ctypedef np.int_t ITYPE_t

cdef double _NUMERICAL_ACCURACY = 1e-8

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def wf_sample_parallel(np.ndarray[DTYPE_t, ndim=2] x0, double time_length, np.ndarray[DTYPE_t, ndim=1] s, int seed, double h, double[:,:,:] Xi, int ne, double[:,:] x_out, int n_threads):
        #np.random.seed(seed)
        cdef int n_particles = Xi.shape[0]
        cdef int tau = Xi.shape[1]+1
        cdef int K = Xi.shape[2]
        cdef int i, j, step, I
        cdef double _ne_times_h = ne*h
        
        cdef np.ndarray [DTYPE_t, ndim=3] deltaW = np.random.normal(0, sqrt(h), n_particles*tau*K).reshape([n_particles, tau, K])        
        cdef double[:,:] q = np.empty([n_threads, K], dtype=np.double)
        cdef double[:] _mu_fixed = np.empty([n_threads], dtype=np.double)
        cdef double[:] temp_cumsum = np.empty([n_threads], dtype=np.double)
        cdef double[:,:] x = np.empty([n_threads, K], dtype=np.double)
        cdef double[:,:] q_prime = np.empty([n_threads, K], dtype=np.double)
        cdef double[:,:] dj = np.empty([n_threads, K], dtype=np.double)
        cdef double[:,:] diffusion = np.empty([n_threads, K], dtype=np.double)
        cdef double[:,:] drift = np.empty([n_threads, K], dtype=np.double)
        cdef double[:,:,:] B = np.zeros([n_threads, K, K], dtype=np.double)
        cdef double[:] xSum = np.empty([n_threads], dtype=np.double)

        cdef long[:] buffer_id = np.zeros([n_particles], dtype=np.int)
        
        with nogil, parallel(num_threads=n_threads):
            # For each particle
            #for I in prange(n_particles, schedule='static', chunksize=1):
            for I in prange(n_particles, schedule='dynamic'):
                buffer_id[I] = cython.parallel.threadid()
                for i in range(K):
                    if tau > 1:
                        Xi[I, 0, i] = x0[I, i]
                
                for step in range(1, tau+1):
                    # Reset all arrays
                    for i in range(K):
                        q[buffer_id[I], i] = 0
                        q_prime[buffer_id[I], i] = 0
                        dj[buffer_id[I], i] = 0
                        diffusion[buffer_id[I], i] = 0
                        drift[buffer_id[I], i] = 0
                        if step > 1:
                            x[buffer_id[I], i] = Xi[I, step-2, i]
                        elif tau > 1:
                            x[buffer_id[I], i] = Xi[I, step-1, i]
                        else:
                            x[buffer_id[I], i] = x0[I, i]
                        xSum[buffer_id[I]] = 0
                        for j in range(i+1):
                            B[buffer_id[I], i, j] = 0

                    # Compute Mu
                    #compute_mu(x=Xi[I, step-1, ], s=s, K=K, _ne_times_h=_ne_times_h, out_array=drift)
                    _mu_fixed[buffer_id[I]] = 0.0
                    for i in range(K):
                        _mu_fixed[buffer_id[I]] = _mu_fixed[buffer_id[I]] - x[buffer_id[I],i]*s[i]

                    for i in range(K):
                        drift[buffer_id[I], i] = (_mu_fixed[buffer_id[I]] + s[i])*_ne_times_h*x[buffer_id[I], i]
                    # End Compute Mu

                    # Compute Sigma...
                    #compute_sigma(Xi[I, step-1, ], B=B)           
                    temp_cumsum[buffer_id[I]] = 0
                    for i in range(K):
                        temp_cumsum[buffer_id[I]] = temp_cumsum[buffer_id[I]] + x[buffer_id[I], i]
                        q[buffer_id[I], i] = 1.0 - temp_cumsum[buffer_id[I]]
                        if q[buffer_id[I], i] < _NUMERICAL_ACCURACY:
                            q[buffer_id[I], i] = 0

                    for i in range(K-1):
                        q_prime[buffer_id[I], i+1] = q[buffer_id[I], i]

                    q_prime[buffer_id[I], 0] = 1.0

                    for i in range(K):
                        if q_prime[buffer_id[I], i] > 0:
                            dj[buffer_id[I], i] = sqrt(x[buffer_id[I], i]*q[buffer_id[I], i]/q_prime[buffer_id[I], i])
                        else: 
                            dj[buffer_id[I], i] = 0

                    for i in range(K):
                        for j in range(i+1):
                            if i == j:
                                B[buffer_id[I], i, i] = dj[buffer_id[I], i]
                            elif q[buffer_id[I], j] == 0: # one violation, set the row to zero
                                B[buffer_id[I], i, j] = 0
                            else:
                                B[buffer_id[I], i, j] = -(x[buffer_id[I], i]/q[buffer_id[I], j])*dj[buffer_id[I], j]
                    # End of Compute Sigma
                    
                    # Compute _diffusion = np.dot(B, deltaW)
                    # TODO: Combine xSum calculation with dot(B, deltaW) to avoid computing changes in case xSum >= 1
                    for i in range(K):
                        if step < tau:
                            if step > 1:
                                Xi[I, step-1, i] = Xi[I, step-2, i]
                        else:
                            if tau > 1:
                                x_out[I, i] = Xi[I, step-2, i]
                            else:
                                x_out[I, i] = x0[I, i]
                            
                        for j in range(i+1):
                            # !!!! ACTUALLY DO UPDATE!!! Don't update when dj = 0
                            diffusion[buffer_id[I], i] = diffusion[buffer_id[I], i] + B[buffer_id[I], i, j]*deltaW[I, (step-1), j]
                
                        if diffusion[buffer_id[I], i] != 0:
                            if step < tau:
                                Xi[I, step-1, i] = Xi[I, step-1, i] + drift[buffer_id[I], i] + diffusion[buffer_id[I], i]
                            else:
                                x_out[I, i] = x_out[I, i] + drift[buffer_id[I], i] + diffusion[buffer_id[I], i]
                                
                    # Enforce boundaries
                    for i in range(K):
                        if step < tau:
                            Xi[I, step-1, i] = min(1-xSum[buffer_id[I]], min(1.0, max(Xi[I, step-1, i], 0.0)))
                            xSum[buffer_id[I]] = xSum[buffer_id[I]] + Xi[I, step-1, i]
                        else:
                            x_out[I, i] = min(1-xSum[buffer_id[I]], min(1.0, max(x_out[I, i], 0.0)))
                            xSum[buffer_id[I]] = xSum[buffer_id[I]] + x_out[I, i]
