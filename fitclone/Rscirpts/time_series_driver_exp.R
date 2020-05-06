
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

if (env == 'AZURECN') {
  Sys.setenv('R_LIBS_USER'='/mnt/batch/tasks/startup/wd/site-library')
  .libPaths(Sys.getenv("R_LIBS_USER"))
}

source('time_series_evaluation.R')
source('violin_viz.R')
#source('time_series_plotting.R')

analyze_all_exp <- function(exp_path, burn_in_fraction=.10, thinning = 1, should_plot=TRUE) {
  print('Compuing RMSE')
  is_err = tryCatch({
    compute_RMSE_for_exp(exp_path, burn_in_fraction = burn_in_fraction, thinning=thinning)
  }, error=function(err) {
    print(sprintf('Error - %s', err))
    errMsg <- err
  })
  
  print('Compuing MAE')
  is_err = tryCatch({
    compute_MAE_for_exp(exp_path, burn_in_fraction = burn_in_fraction, thinning=thinning)
  }, error=function(err) {
    print(sprintf('Error - %s', err))
    errMsg <- err
  })
  
  print('JUS PLotting')
  #if(inherits(is_err, "error")) next
  if (should_plot) {
    is_err = tryCatch({
      plot_violin_for_exp_path(exp_path=exp_path, burn_in_fraction = burn_in_fraction, thinning=thinning, datatag = '')
      plot_all_for_exp_path(exp_path=exp_path, burn_in_fraction = burn_in_fraction, thinning=thinning)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
  }
}


options <- commandArgs(trailingOnly = TRUE)
print(options)
analyze_all_exp(tail(options, n=1))


#exp_path="~/Desktop/crispr/exp_Y0CQ6_201711-09-165527.694305"

#exp_path="~/Desktop/pgas_sanity_check/exp_Y0CQ6_201711-17-133026.581438"
#analyze_all_exp(exp_path=exp_path,   burn_in_fraction = .1, thinning = 1)

#analyze_all_exp(exp_path=exp_path,   burn_in_fraction = .01, thinning = 1)
#analyze_all_exp(exp_path=exp_path,   burn_in_fraction = .5, thinning = 1)

#compute_MAE_for_exp(exp_path, burn_in_fraction = .1, thinning=10)
#compute_MAE_for_exp(exp_path, burn_in_fraction = .01, thinning=1)


