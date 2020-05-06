stuff = 'hello'
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
cimport openmp

from libc.math cimport sqrt
from libc.math cimport log
from libc.math cimport erfc
from libc.math cimport M_PI

ctypedef np.float_t DTYPE_t
ctypedef np.int_t ITYPE_t

from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def gp_compute_loglikelihood_parallel(double[:] x_M, double[:,:] X_t, double[:] alpha, double[:] sigma2, double[:] C, double epsilon, double[:] loglikelihood, int n_threads):
    cdef int N, K, i, k
    N = X_t.shape[0]
    K = X_t.shape[1]
    
    cdef double *mu = <double *>malloc(n_threads*sizeof(double))
    cdef int *buffer_id = <int *>malloc(N*sizeof(int))
    # math constant caches
    cdef double *sigma = <double *>malloc(K*sizeof(double))
    cdef double *log_sigma = <double *>malloc(K*sizeof(double))
    cdef double *inv_sigma_sqrt2 = <double *>malloc(K*sizeof(double))
    
    for i in range(N):
        loglikelihood[i] = 0
    
    for k in range(K):
        sigma[k] = sqrt(sigma2[k]+epsilon)    
        log_sigma[k] = log(sigma[k])
        inv_sigma_sqrt2[k] = 1/(sigma[k]*sqrt(2))
    
    cdef double const_log = -log(sqrt(2*M_PI)) + log(2)

    with nogil, parallel(num_threads=n_threads):
        for i in prange(N, schedule='dynamic'):
            buffer_id[i] = cython.parallel.threadid()
            for k in range(K):
                mu[buffer_id[i]] = C[k]*X_t[i, k]/alpha[k]
                loglikelihood[i] = loglikelihood[i]  - log_sigma[k] + const_log - .5*((x_M[k]-mu[buffer_id[i]])/sigma[k])**2 - log(erfc((mu[buffer_id[i]]-1)*inv_sigma_sqrt2[k]) - erfc(mu[buffer_id[i]]*inv_sigma_sqrt2[k]))

                # TODO: handle infinity 
                #openmp.omp_get_num_threads()
    free(mu)
    free(buffer_id)        
    free(sigma)
    free(log_sigma)
    free(inv_sigma_sqrt2)
