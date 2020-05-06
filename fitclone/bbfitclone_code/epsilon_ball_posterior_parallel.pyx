import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel

ctypedef np.float_t DTYPE_t
ctypedef np.int_t ITYPE_t

from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def epsilon_ball_sample_posterior_parallel(double[:] y_obs,
                                    double epsilon,
                                    double [:,:] unif_rands,
                                    double[:,:] X, 
                                    int n_threads, 
                                    int is_one_missing=1):
    #print('n_threads={}'.format(n_threads))
    # Sample from the conditional distribution, given the observation
    cdef int i, n, K, N
    cdef double a, b, c, d, lower_bound, upper_bound, the_sum
    
    N = X.shape[0] 
    K = X.shape[1]
    the_sum = 0.0
    
    cdef double[:] sum_lower_y = np.empty([K])
    cdef double[:] sum_upper_y = np.empty([K])

    # Pre-compute the lower_y and upper_y arrays
    for i in range(1, K+1):        
        sum_lower_y[i-1] = np.sum(np.array([max(0.0, y-epsilon) for y in y_obs[i:K]]))
        sum_upper_y[i-1] = np.sum(np.array([min(1.0, y+epsilon) for y in y_obs[i:K]]))   

    with nogil, parallel(num_threads=n_threads):
        for n in prange(N, schedule='static'):
            the_sum = 0
            for i in range(K):
                a = max(0.0, y_obs[i]-epsilon)
                a = min(1.0, a)
                b = min(1.0, y_obs[i]+epsilon)
                b = max(0.0, b)
                c = max(0.0, 1 - the_sum - sum_upper_y[i])
                c = min(1.0, c)
                d = min(1.0, 1 - the_sum - sum_lower_y[i])
                d = max(0.0, d)
                lower_bound = max(a,c) if is_one_missing == False else a
                upper_bound = min(b,d)
                X[n, i] = unif_rands[n, i] * (upper_bound - lower_bound) + lower_bound
                the_sum = the_sum + X[n, i]
