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
def local_sample_parallel(np.ndarray[DTYPE_t, ndim=2] x0, double time_length, np.ndarray[DTYPE_t, ndim=1] s, int seed, double h, double[:,:,:] Xi, int ne, double[:,:] x_out, int n_threads):
        cdef int n_particles = Xi.shape[0]
        cdef int tau = Xi.shape[1]+1
        cdef int K = Xi.shape[2]
        cdef int i, j, step, I
        cdef double _ne_times_h = ne*h
        
        cdef np.ndarray [DTYPE_t, ndim=3] deltaW = np.random.normal(0, sqrt(h), n_particles*tau*K).reshape([n_particles, tau, K])        
        cdef double[:] q = np.empty([K], dtype=np.double)
        cdef double _mu_fixed
        cdef double temp_cumsum
        cdef double[:] x = np.empty([K], dtype=np.double)
        cdef double[:] q_prime = np.empty([K], dtype=np.double)
        cdef double[:] dj = np.empty([K], dtype=np.double)
        cdef double[:] diffusion = np.empty([K], dtype=np.double)
        cdef double[:] drift = np.empty([K], dtype=np.double)
        cdef double[:,:] B = np.zeros([K, K], dtype=np.double)
        cdef double xSum

        
        with nogil, parallel(num_threads=n_threads):
            # For each particle
            for I in prange(n_particles, schedule='static'):
                for i in range(K):
                    if tau > 1:
                        Xi[I, 0, i] = x0[I, i]
                
                for step in range(1, tau+1):
                    # Reset all arrays
                    for i in range(K):
                        q[i] = 0
                        q_prime[i] = 0
                        dj[i] = 0
                        diffusion[i] = 0
                        drift[i] = 0
                        if step > 1:
                            x[i] = Xi[I, step-2, i]
                        elif tau > 1:
                            x[i] = Xi[I, step-1, i]
                        else:
                            x[i] = x0[I, i]
                        xSum = 0
                        for j in range(i+1):
                            B[i, j] = 0

                    # Compute Mu
                    _mu_fixed = 0.0
                    for i in range(K):
                        _mu_fixed = _mu_fixed - x[i]*s[i]

                    for i in range(K):
                        drift[i] = (_mu_fixed + s[i])*_ne_times_h*x[i]
                    # End Compute Mu

                    # Compute Sigma...
                    temp_cumsum = 0
                    for i in range(K):
                        temp_cumsum = temp_cumsum + x[i]
                        q[i] = 1.0 - temp_cumsum
                        if q[i] < _NUMERICAL_ACCURACY:
                            q[i] = 0

                    for i in range(K-1):
                        q_prime[i+1] = q[i]

                    q_prime[0] = 1.0

                    for i in range(K):
                        if q_prime[i] > 0:
                            dj[i] = sqrt(x[i]*q[i]/q_prime[i])
                        else: 
                            dj[i] = 0

                    for i in range(K):
                        for j in range(i+1):
                            if i == j:
                                B[i, i] = dj[i]
                            elif q[j] == 0: # one violation, set the row to zero
                                B[i, j] = 0
                            else:
                                B[i, j] = -(x[i]/q[j])*dj[j]
                    # End of Compute Sigma
                    
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
                            diffusion[i] = diffusion[i] + B[i, j]*deltaW[I, (step-1), j]
                
                        if diffusion[i] != 0:
                            if step < tau:
                                Xi[I, step-1, i] = Xi[I, step-1, i] + drift[i] + diffusion[i]
                            else:
                                x_out[I, i] = x_out[I, i] + drift[i] + diffusion[i]
                                
                    # Enforce boundaries
                    for i in range(K):
                        if step < tau:
                            Xi[I, step-1, i] = min(1-xSum, min(1.0, max(Xi[I, step-1, i], 0.0)))
                            xSum = xSum + Xi[I, step-1, i]
                        else:
                            x_out[I, i] = min(1-xSum, min(1.0, max(x_out[I, i], 0.0)))
                            xSum = xSum + x_out[I, i]