# set environment

source('time_series_utils.R')
library(reshape2)

# Time series evaluation 

## 1. Prediction accuracy
## RMSE, for points were we've observation, what is the root mean square error of observation and prediction?

## 2. Parameter accuracy
## MAE (Mean absolute error), for the post-burn sample mean of the infered params and their true-value

## 3. Convergenece of the MCMC chain
# gelman rubin statistics and others in CODA


euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))

# TxK 
RMSE <- function(X1, X2) {
  N = nrow(X1)
  if (N != nrow(X2)) stop('non-matching time-series')
  err = c()
  for (t in 1:N) {
    err = c(err, euc.dist(X1[t,], X2[t,])) 
  }
  
  return(sqrt(mean(err^2)))
}

# returns a set of #|some_array| values in the values array that are closest to the corresponding values in the some_array
ts_utils_get_closest <- function(values, some_array) {
  index = c()
  for (some_value in some_array) {
    index <- c(index, which.min(abs(values-some_value)))
  }
  values[index]
}


compute_RMSE_for_exp <- function(exp_path, for_inference=F, burn_in_fraction=.10, thinning=1) {
  burn_in = burn_in_fraction
  
  # read the learn_time
  mcmc_options = ts_utils_read_config(exp_path) #mcmc_options = list(seed=11, nMCMC=10, nParticles=5, s=(seq(K)/(K+1)))
  if (is.null(mcmc_options)) {
    mcmc_options = list(seed=11, nMCMC=10, nParticles=-1, s=(seq(K)/(K+1)))
    print('Warning! Using dummy config file')
  }
  
  # read the real data
  data_path <- file.path(exp_path, 'sample_data.tsv.gz')
  if (!file.exists(data_path)) {
    data_path = mcmc_options$original_data
  }
  dat <- read.table(data_path, sep='\t', header = T, stringsAsFactors = F)
  dat$X.1 <- NULL
  
  K = length(unique(dat$K))
  

  # T by K
  dat_wide = acast(dat, time ~ K, value.var = 'X')
  ref_times = as.numeric(rownames(dat_wide))

  # the predicted data
  exp_kind = 'predict'
  param_path = file.path(exp_path, paste0(exp_kind, ".tsv.gz"))
  if (!file.exists(param_path)) {
    param_path = file.path(exp_path, paste0('infer_x', '.tsv.gz'))
    print('Warning! Using inference instead of prediction!')
    for_inference=T
  }
  
  param_dat = read.table(param_path, sep='\t', header = T, stringsAsFactors = F)
  nIter = length(unique(param_dat$np))
  param_dat = param_dat[param_dat$np >= nIter*burn_in,]
  # thinning
  param_dat = param_dat[param_dat$np %in% unique(param_dat$np)[seq(1, length(unique(param_dat$np)), thinning)], ]
  
  param_dat_wide = acast(param_dat, time ~ K, value.var = 'X', mean)
  inferred_times = as.numeric(rownames(param_dat_wide))
  
  param_dat_wide_final = param_dat_wide[inferred_times %in% ts_utils_get_closest(values = inferred_times, some_array = ref_times), , drop=F]
  
  learn_time = mcmc_options$learn_time
  if (!for_inference) {
    rmse = RMSE(dat_wide, param_dat_wide_final)
    
    # Calculate rmse after learn time
    # Check if it was a prediction task 
    if (!all(ref_times > learn_time) == FALSE) {
      dat_lt = dat_wide[ref_times > learn_time, ,drop=F]
      inferred_times = as.numeric(rownames(param_dat_wide_final))
      param_dat_wide_final_lt = param_dat_wide_final[inferred_times > learn_time, ,drop=F]
      rmse_after_learn_time = RMSE(dat_lt, param_dat_wide_final_lt)
    } else {
      param_dat_wide_final_lt = NULL
    }
      
    
    res = list(rmse_after_learn_time=NaN, rmse=rmse)
  } else {
    dat_wide = dat_wide[ref_times < learn_time,, drop=F]
    inferred_times = as.numeric(rownames(param_dat_wide_final))
    param_dat_wide_final = param_dat_wide_final[inferred_times < learn_time, ,drop=F]
    rmse = RMSE(dat_wide, param_dat_wide_final)
    res = list(rmse_infer_before_learn_time=rmse)
  }
  
  
  ## Compute RMSE for each dimension
  ### 1. everything
  K = length(unique(dat$K))
  rmse_dim = c()
  for (k in unique(dat$K)) {
    # k = 0
    inf_temp = param_dat_wide_final[, which(colnames(param_dat_wide_final)==k), drop=F]
    ref_temp = dat_wide[, which(colnames(dat_wide)==k), drop=F]
    rmse_dim <- c(rmse_dim, RMSE(inf_temp, ref_temp))
  }
  names(rmse_dim) = paste0('s', unique(dat$K))
  if (!for_inference) {
    res$indiv_dim$full_rmse = as.list(rmse_dim)
  } else {
    res$indiv_dim$rmse_infer_before_learn_time = as.list(rmse_dim)
  }
  
  if (!for_inference && !is.null(param_dat_wide_final_lt)) {
    ### 2. just prediction
    rmse_pred_dim = c()
    for (k in unique(dat$K)) {
      # k = 0
      inf_temp = param_dat_wide_final_lt[, which(colnames(param_dat_wide_final_lt)==k), drop=F]
      ref_temp = dat_lt[, which(colnames(dat_lt)==k), drop=F]
      rmse_pred_dim <- c(rmse_pred_dim, RMSE(inf_temp, ref_temp))
    }
    names(rmse_pred_dim) = paste0('s', unique(dat$K))
    res$indiv_dim$rmse_after_learn_time = as.list(rmse_pred_dim)
  }
  
  
  yml = as.yaml(res)
  
  file_name = file.path(exp_path, paste0(exp_kind, 'summary', '.yaml'))
  writeLines(yml, file_name)
  return(res)
}

compute_MAE_for_exp <- function(exp_path, burn_in_fraction=.10, thinning=1) {
  # Read the inferred theta
  param_path = file.path(exp_path, 'infer_theta.tsv.gz')
  param_dat = read.table(param_path, sep='\t', header=TRUE)
  K = ncol(param_dat)
  colnames(param_dat) = paste0('s', seq(K))
  n = nrow(param_dat)
  param_dat = param_dat[seq(burn_in_fraction*n, n, thinning),, drop=F]
  mean_theta = colMeans(param_dat)
  
  # Read reference values
  mcmc_options = ts_utils_read_config(exp_path) #mcmc_options = list(seed=11, nMCMC=10, nParticles=5, s=(seq(K)/(K+1)))
  ref_theta = unlist(mcmc_options$s)
  
  MAE = mean(abs(ref_theta - mean_theta))
  res = list(s_MAE=MAE)
  res$indiv_dim = abs(ref_theta - mean_theta)
  yml = as.yaml(res)
  
  file_name = file.path(exp_path, paste0('MAE', '.yaml'))
  writeLines(yml, file_name)
  return(MAE)
}


## Utils -- tar all the plots files and move them to a plots .gz in the output
handle_plots_for_exp <- function(exp_path) {
  ## batch_path/outputs/exp_dir_name
  dest_dir = file.path(dirname(exp_path), 'plots')
  dir.create(dest_dir, showWarnings = F)
  
  dest_file = file.path(dest_dir, paste0(basename(exp_path), '_plots'))
  cmd_string = sprintf('tar -czf %s.gz -C %s .', dest_file, file.path(exp_path, 'plots/'))
  print(cmd_string)
  system(cmd_string)
}


