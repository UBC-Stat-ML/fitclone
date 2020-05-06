def sample_posterior_vectorised(self, observation, N=1, is_one_missing=True, shuffle=True):
    # Sample from the conditional distribution, given the observation
    y_obs = np.array(observation).copy()
    K = len(y_obs)
    X = np.zeros([N, K])
    
    # Pre-compute the lower_y and upper_y arrays
    sum_lower_y = np.empty([K])
    sum_upper_y = np.empty([K])
    for i in range(1, K+1):        
        sum_lower_y[i-1] = np.sum(np.array([max(0.0, y-self.epsilon) for y in y_obs[i:K]]))
        sum_upper_y[i-1] = np.sum(np.array([min(1.0, y+self.epsilon) for y in y_obs[i:K]]))   
    
    
    # Shuffle the vectors to counter bias for the first component
    # if shuffle:
    #     shuffleMap = np.array(list(range(K)), dtype=int)
    #     np.random.shuffle(shuffleMap)
    #     y_obs = y_obs[shuffleMap.flatten()]
    
    for n in range(N):
        for i in range(K):
            a = max(0.0, y_obs[i]-self.epsilon)
            a = min(1.0, a)
            b = min(1.0, y_obs[i]+self.epsilon)
            b = max(0.0, b)
            c = max(0.0, 1 - np.sum(X[n, 0:i]) - sum_upper_y[i])
            c = min(1.0, c)
            d = min(1.0, 1 - np.sum(X[n, 0:i]) - sum_lower_y[i])
            d = max(0.0, d)
            lower_bound = max(a,c) if is_one_missing == False else a
            upper_bound = min(b,d)
            X[n, i] = np.random.uniform(lower_bound, upper_bound)
                
        if np.sum(X[n, ]) > 1.0:
            raise ValueError('x is not Dirichlet!')
            
        if any(t > self.epsilon for t in (abs(X[n, ] - y_obs))):
            raise ValueError('x not in epsilon ball')
        
        # if shuffle:
        #     mapBacK = np.array([np.where(shuffleMap==i) for i in range(K)]).flatten()
        #     y_obs = y_obs[mapBacK]
        #     X[n, ] = X[n, mapBacK]
        
        return(X)