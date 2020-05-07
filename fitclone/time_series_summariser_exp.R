## Does NO plotting, only computes numerical statistics. 
## Designed for cloud purposes

source('time_series_evaluation.R')

analyze_all_exp <- function(exp_path, burn_in_fraction=.10, thinning = 1) {
  is_err = tryCatch({
    compute_RMSE_for_exp(exp_path, burn_in_fraction = burn_in_fraction, thinning=thinning)
  }, error=function(err) {
    print(sprintf('Error - %s', err))
    errMsg <- err
  })
  
  is_err = tryCatch({
    compute_MAE_for_exp(exp_path, burn_in_fraction = burn_in_fraction, thinning=thinning)
  }, error=function(err) {
    print(sprintf('Error - %s', err))
    errMsg <- err
  })
}


options <- commandArgs(trailingOnly = TRUE)
print(options)
analyze_all_exp(tail(options, n=1))


#exp_path="~/Desktop/crispr/exp_Y0CQ6_201710-26-184727.911479"
#exp_path="~/Desktop/pgas_sanity_check/exp_Y0CQ6_201711-15-133916.865830"
#analyze_all_exp(exp_path=exp_path,   burn_in_fraction = .01, thinning = 1)
#analyze_all_exp(exp_path=exp_path,   burn_in_fraction = .1, thinning = 1)
