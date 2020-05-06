# set environment
env <- Sys.getenv('HOST')
if (env == '') env <- 'AZURECN'
switch(env, local=setwd('/Users/sohrab/Google Drive/Masters/Thesis/scripts/fitness'),
       beast=setwd('/home/ssalehi/projects/fitness'),
       AZURECN=setwd('/mnt/batch/tasks/startup/wd'),
       rocks3=setwd('/home/ssalehi/projects/fitness'),
       grex=setwd('/home/sohrab/projects/fitness'),
       bugaboo=setwd('/home/sohrab/projects/fitness'),
       shahlab=setwd('/scratch/shahlab_tmp/ssalehi/fitness'),
       azure=setwd('/home/ssalehi/projects/fitness'),
       noah=setwd('/Users/sohrabsalehi/projects/fitness'),
       MOMAC39=setwd('/Users/ssalehi/projects/fitness'))


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


# Source e.g., http://www.geosci-model-dev.net/7/1247/2014/gmd-7-1247-2014.pdf
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
  #write.table(df, file.path(exp_path, paste0(exp_kind, '_summary', '.tsv')), sep='\t', row.names = F)
  return(res)
}

#ss = compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_201706-19-185106')
#ss = compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_OM489_201706-26-16226.085942', burn_in_fraction = 2)
#sss = compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_OM489_201706-26-16226.085942', burn_in_fraction = 10)
#print(ss$rmse_infer_before_learn_time )
#print(sss$rmse_infer_before_learn_time)


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

eval_all_for_batch <- function(batch_path, burn_in_fraction=.10, thinning=1) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    is_err = tryCatch({
      compute_RMSE_for_exp(dir, burn_in_fraction = burn_in_fraction, thinning=thinning)
      compute_MAE_for_exp(dir, burn_in_fraction = burn_in_fraction, thinning=thinning)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
}


handle_all_for_batch <- function(batch_path) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    if (!dir.exists(file.path(dir, 'plots'))) next
    is_err = tryCatch({
      #compute_RMSE_for_exp(dir)
      handle_plots_for_exp(dir)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
  
  cmd_string = sprintf('tar -czf %s.gz -C %s .', file.path(batch_path, 'outputs/plots/'), file.path(batch_path, 'outputs/plots/'))
  print(cmd_string)
  system(cmd_string)
}


#exp_path='/Users/ssalehi/Desktop/pgas_sanity_check/exp_5RXP4_201707-19-16801.571058/'
#compute_RMSE_for_exp(exp_path)
#compute_MAE_for_exp(exp_path)

#eval_all_for_batch('/shahlab/ssalehi/scratch/fitness/batch_runs/TenKmore_201706-25-234420.450537')
#handle_all_for_batch('/shahlab/ssalehi/scratch/fitness/batch_runs/TenKmore_201706-25-234420.450537')

#compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/not_too_bad_either_exp_IEQH5_201706-28-14014.234892', for_inference = T)
#compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201706-28-15559.928666', for_inference = T)
#compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201706-28-171112.650169', for_inference = T)
#compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201706-28-171112.650169', for_inference = T)

#compute_RMSE_for_exp(exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201707-07-215713.414551', for_inference = T)



#compute_RMSE_for_exp(exp_path="/Users/sohrab/Desktop/testplots/o_0_0_OKXPI_201707-11-01828.021046")

#eval_all_for_batch(batch_path='/Users/ssalehi/Desktop/pgas_sanity_check/K2500short_201707-03-134111.238639', burn_in_fraction=10)
