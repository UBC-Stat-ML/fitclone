import numpy as np
cimport numpy as np
cimport openmp
import cython
from cython.parallel import prange, parallel

from libc.math cimport fabs

ctypedef np.float_t DTYPE_t
ctypedef np.int_t ITYPE_t

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def epsilon_ball_emission_parallel(double[:, :] X, double[:] y, double lambdaVal, double epsilon, double b, double[:] loglikelihood, int n_threads):
    cdef int N, K, i, k
    N = X.shape[0]
    K = X.shape[1]
    cdef double INFINITY_ = -np.inf
    cdef int *buffer_id = <int *>malloc(N*sizeof(int))
         
    for i in range(N):
        loglikelihood[i] = 0.0
    
    with nogil, parallel(num_threads=n_threads):
        for i in prange(N, schedule='dynamic'):
            for k in range(K):
                if fabs(X[i, k]-y[k]) <= epsilon:
                    continue
                elif b != 0:
                    loglikelihood[i] = loglikelihood[i] - lambdaVal*fabs(X[i, k]-y[k])
                else:
                    loglikelihood[i] = INFINITY_
                    break
    free(buffer_id)        
