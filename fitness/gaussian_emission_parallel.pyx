import numpy as np
cimport numpy as np
cimport openmp
import cython
from cython.parallel import prange, parallel

from libc.math cimport sqrt
from libc.math cimport M_PI
from libc.math cimport log

ctypedef np.float_t DTYPE_t
ctypedef np.int_t ITYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def gaussian_emission_parallel(double[:, :] X, double[:] y, double epsilon, double[:] loglikelihood, int n_threads):
    cdef int N, K, i, k
    N = X.shape[0]
    K = X.shape[1]
    cdef double const_ll  = -(K*.5)*log(2.0*M_PI) - K*log(epsilon)     
    cdef double const_sigma2_inv  = -.5*(epsilon**(-2))     

    for i in range(N):
        loglikelihood[i] = 0.0
    
    with nogil, parallel(num_threads=n_threads):
        for i in prange(N, schedule='dynamic'):
            for k in range(K):
                loglikelihood[i] = loglikelihood[i] + (X[i, k]-y[k])**2
            loglikelihood[i] = (loglikelihood[i] * const_sigma2_inv) + const_ll 