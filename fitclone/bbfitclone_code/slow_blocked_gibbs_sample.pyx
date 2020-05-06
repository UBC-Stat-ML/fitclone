import numpy as np
cimport numpy as np
import scipy as sp
import scipy.stats
import cython
from cython.parallel import prange, parallel

from libc.math cimport fabs, sqrt

ctypedef np.float_t DTYPE_t
ctypedef np.int_t ITYPE_t
cdef double _NUMERICAL_ACCURACY = 1e-4
cdef double _ALMOST_ONE = 1.0 - _NUMERICAL_ACCURACY 
cdef double _RIDGE_CORRECTION = 1e-2

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double array_dot(double[:] a_1, double[:] a_2, int K) nogil:
    cdef double res = 0.0
    for k in range(K):
        res = res + a_1[k]*a_2[k]
    return(res)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double safe_division(double nu, double de) nogil:
    if de == _NUMERICAL_ACCURACY:
        return(0.0)
    else:
        return(nu/de)
        


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void cholesky(double [:,:] A, double[:,:] L, int n) nogil:
    cdef int i, j, k
    cdef double s = 0.0
    L[:,:] = 0.0
    
    for i in range(n):
        for j in range(i+1):
            s = 0
            for k in range(j):
                s = s + L[i, k]*L[j, k]
            if i == j:
                L[i, j] = sqrt(fabs(A[i, i] - s))
                if L[i, j] < _NUMERICAL_ACCURACY:
                    L[i, j] = 0.0
            else:
                #L[i, j] = (1.0 / L[j,j] * (A[i, j] - s))
                L[i, j] = (safe_division(1.0, L[j,j]) * (A[i, j] - s))
                


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void compute_mu_bar(int n, int K, int K_prime, int Ne, double[:] a, double[:] s, double[:] Xi, double[:] full_Xi, double[:] mu_pc, double[:] mu_bar, double[:] mu_1, double[:] mu_2) nogil:
    '''
    \mu_bar = \mu_1 - \Xi \corss \mu_{pc} \cross (a-\mu_2)
    Expanding the cross multiplications, we'll have:
    C = \Xi \corss \mu_{pc}, so c_{i,j} = \Xi_{i} \cross \mu_j
    S = C \cross (a-\mu_2), so s[k] = \sigma_{t=0}^{K'}c_{k, t}(a_t-\mu_2[t])
    '''
    cdef int k, t = 0
    full_Xi[0:K] = Xi[:]
    cdef double mu_fixed = array_dot(s, full_Xi, K+K_prime)
        
    for k in range(K):
        mu_1[k] = Xi[k]*(s[k]-mu_fixed)*Ne
    for k in range(0, K_prime):
        mu_2[k] = a[k]*(s[k+K]-mu_fixed)*Ne
    
    # NOTE: This has to be negative! Precomputation is missing a negative from sigma_12 == -x \times a
    for k in range(K):
        mu_bar[k] = 0
        for t in range(K_prime):
            mu_bar[k] = mu_bar[k] + mu_pc[t]*(a[t]-mu_2[t])
        mu_bar[k] = mu_1[k] - Xi[k]*mu_bar[k]

        
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef void compute_mu_bar_old(int n, int K, int K_prime, int Ne, double[:] a, double[:] s, double[:] Xi, double[:] full_Xi, double[:] mu_pc, double[:] mu_bar, double[:] mu_1, double[:] mu_2):        
#     mu_fixed = np.dot(s, np.append(Xi, a))
#     mu_1 = np.multiply(Xi[n, Xi_step, :], np.subtract(s[0:K], mu_fixed)) * Ne # pointwise multiplication
#     mu_2 = np.multiply(a, np.subtract(s[K:s.shape[0]],mu_fixed)) * Ne
#     # NOTE: This has to be negative! Precomputation is missing a negative from sigma_12 == -x \times a
#     mu_bar = np.subtract(mu_1, np.matmul(np.outer(Xi[n, Xi_step, :], mu_pc), (np.subtract(a, mu_2))))

                
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void compute_sigma_bar(int K, double q, double[:] x, double[:,:] sigma2_bar) nogil:
    cdef int k, j
    for k in range(K):
        for j in range(k+1):
            if k == j:
                sigma2_bar[k, k] = x[k]*(1-q*x[k])
            else:
                sigma2_bar[k, j] = -q*x[k]*x[j]
                sigma2_bar[j, k] = sigma2_bar[k, j] 
    

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)   
cdef void sample_multivariate_normal(double [:] mean, double[:,:] cov, double[:] x_out):
    #np.ndarray[DTYPE_t, ndim=1]
    cdef double[:] junk
    junk = sp.stats.multivariate_normal.rvs(mean=mean, cov=cov)    
    x_out[:] = junk

        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)   
cdef void propagate_with_noise(int K, double sqrt_h, double [:] mean, double[:,:] cov, double[:] normals, double[:,:] L, double[:] x_out) nogil:
    # See Evans's p.73 Evans LC. An introduction to stochastic differential equations version 1.2. Lecture Notes, UC Berkeley. 2006.
    # https://pdfs.semanticscholar.org/916a/5b03d0e6964f5efdfe6ef147815cfa2069c1.pdf
    ## TODO: use the SVD method and compute Cholesky from that
    # If the sum is 1, then scale back to K=K-1
    # Make sure that the removed portion is kept updated, but not directly
    # 1. 
    # 2. 
    
    # If the sum exceeds, remove it
    cdef int k = 0
    cdef int j = 0
#     cdef double the_sum = 0.0
#     for k in range(K+K_prime):
#         the_sum = the_sum + full_Xi[k]
#     if the_sum >= _ALMOST_ONE:
#         for k in range(K):
#             cov[k, k] = cov[k, k] + _RIDGE_CORRECTION
    
    
    cholesky(cov, L, K)
#     print('cov is ', np.asarray(cov))
#     print('np.chol is ')
#     cdef np.ndarray[DTYPE_t, ndim=2] kk = np.linalg.cholesky(cov)
#     print('it is ', kk)
    
    
    cdef double diffusion
    #cdef int k, j
    for k in range(K):
        diffusion = 0.0
        for j in range(K):
            diffusion = diffusion + L[k, j]*sqrt_h*normals[j]
        x_out[k] = mean[k] + diffusion
            
        
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)        
cdef void clip_frequencies(int step, int K, double[:] X0, double[:,:] Xi, double[:] x_out, double free_freq, double tau) nogil:
    cdef double x_sum = 0.0
    cdef double rem = 0.0
    
    for k in range(K):
        if step == 1:
            if X0[k] == 0:
                Xi[step-1, k] = 0
            else: 
                rem = free_freq-x_sum
                if rem < 0: rem = 0
                Xi[step-1, k] = min(rem, max(Xi[step-1, k], 0.0))
                x_sum = x_sum + Xi[step-1, k]
        elif step < tau:
            if Xi[step-2, k] == 0:
                Xi[step-1, k] = 0
            else:
                rem = free_freq-x_sum
                if rem < 0: rem = 0
                Xi[step-1, k] = min(rem, max(Xi[step-1, k], 0.0))
                x_sum = x_sum + Xi[step-1, k]
        else: # last step
            if Xi[step-2, k] == 0:
                x_out[k] = 0
            else:
                rem = free_freq-x_sum
                if rem < 0: rem = 0
                x_out[k] = min(rem, max(x_out[k], 0.0))
                x_sum = x_sum + x_out[k]
                

                
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def wf_blocked_gibbs_parallel1(int tau, double[:,:] X0, double time_length, double[:] s, double h, int Ne, double[:,:,:] Xi, double[:,:] A, double[:,:] x_out, int n_threads, double[:,:,:] deltaW):
        """
        s is 1 by K_prime
        """
        cdef int N, K, K_prime, step, is_last_time_point, n, k, Xi_step, j, test_i
        cdef double mu_fixed, temp_mean
        cdef double[:,:] sigma_22
        cdef double sqrt_h = sqrt(h)
        cdef double free_freq
        N = X0.shape[0]
        K = X0.shape[1]
        K_prime = A.shape[1]
        
        #cdef double[:,:,:] deltaW = np.random.normal(0, sqrt_h, N*tau*K).reshape([N, tau, K])        

        # Buffers, are stepped.
        cdef double[:,:] B
        cdef double[:] junk
        cdef double[:,:] mu_pc = np.empty([tau, K_prime])
        cdef double[:] sigma_pc = np.empty([tau])
        
        # Reuse them (TODO: Make thread-friendly by I)
        cdef double[:] mu_1 = np.empty([K])
        cdef double[:] mu_2 = np.empty([K_prime])
        cdef double[:] mu_bar = np.empty([K])
        cdef double[:,:] sigma2_bar = np.zeros([K, K])
        cdef double[:,:] cholesky_buffer = np.empty([K, K])
        cdef double[:, :] full_Xi = np.empty([A.shape[0], K+K_prime]) # stop enlarging an array evrytime we compute mu_bar
        full_Xi[:, K:(K_prime+K)] = A[:,:]
        
        # Use Xi as buffer for X0
        Xi[:, 0, :] = X0
        
        # A quick hack to handle the case where the last deltaT doesn't exactly coordinate with T_learn
        is_last_time_point = (A.shape[0] == tau)
          
        # Precompute inverse A-s, AKA Bs
        for step in range(0, tau):
            sigma_22 = np.diag(A[step, :]) - np.outer(np.transpose(A[step, :]), A[step, :])
            B = np.linalg.pinv(sigma_22)
            ## For \mu
            junk = np.dot(A[step, ], B)
            mu_pc[step, :] = junk
            
            ## For \sigma
            sigma_pc[step] = np.dot(mu_pc[step, :], np.transpose(A[step, ])) # gotta be a scalar
        
#         with nogil, parallel(num_threads=n_threads):
#             for n in prange(N, schedule='static'):             # For each particle
        for test_i in range(1):
            for n in range(N):
                for step in range(1, tau+1):
                    if step == tau and is_last_time_point:
                        break

                    if step > 1:
                        Xi_step = step - 2
                    elif tau > 1:
                        Xi_step = step - 1
                    else:
                        Xi_step = 0
            
                    free_freq = 0
                    for k in range(K_prime):
                        free_freq = free_freq + A[step, k]
                    free_freq = 1.0 - free_freq
                    
                    
                    # Compute mu_bar
                    compute_mu_bar(n=n, K=K, K_prime=K_prime, Ne=Ne, a=A[step-1,:], s=s, Xi=Xi[n, Xi_step, :], full_Xi=full_Xi[step-1,:], mu_pc=mu_pc[step-1,:], mu_bar=mu_bar, mu_1=mu_1, mu_2=mu_2)
                    #print('mu_bar is ', np.asarray(mu_bar))
                    
                    
                    #compute_mu_bar_old()
                    #print('mu_bar_old is', mu_bar_old)
                    
                    # Compute Sigma_bar
                    compute_sigma_bar(K=K, q=sigma_pc[step-1]+1, x=Xi[n, Xi_step, :], sigma2_bar=sigma2_bar)
                    #print('sigma2_bar is ', np.asarray(sigma2_bar))
                    
                    # Scale and shift the mean
                    for k in range(K):
                        mu_bar[k] = Xi[n, Xi_step, k] + mu_bar[k]*h
                        for j in range(K):
                            sigma2_bar[k, j] = sigma2_bar[k, j]*h
                            
                    # Propagate forward       
                    if step < tau:
                        sample_multivariate_normal(mean=mu_bar, cov=sigma2_bar, x_out=Xi[n, step-1, :])
                        #propagate_with_noise(K=K, sqrt_h=sqrt_h, mean=mu_bar, cov=sigma2_bar, normals=deltaW[n, step-1,], L=cholesky_buffer, x_out=Xi[n, step-1, :])
                    else:    
                        sample_multivariate_normal(mean=mu_bar, cov=sigma2_bar, x_out=x_out[n, :])
                        #propagate_with_noise(K=K, sqrt_h=sqrt_h, mean=mu_bar, cov=sigma2_bar, normals=deltaW[n, step-1,], L=cholesky_buffer, x_out=x_out[n, :])
                    
                    # Clip the frequencies
                    #print('x_out before clipping is ', np.asarray(x_out[n, :]))
                    clip_frequencies(step=step, K=K, X0=X0[n,:], Xi=Xi[n,:,:], x_out=x_out[n,:], free_freq=free_freq, tau=tau)
                    #print('x_out after clipping is ', np.asarray(x_out[n, :]))
                    
        if is_last_time_point:
            print('is_last_time_point')
            x_out[:,:] = Xi[:, tau-2, :]            