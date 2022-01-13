source('time_series_evaluation.R')
source('violin_viz.R')

analyze_all_exp <- function(exp_path, burn_in_fraction=.10, thinning = 1, should_plot=TRUE) {
  print('Compuing RMSE')
  is_err = tryCatch({
  }, error=function(err) {
    print(sprintf('Error - %s', err))
    errMsg <- err
  })
  
  print('Compuing MAE')
  is_err = tryCatch({
  }, error=function(err) {
    print(sprintf('Error - %s', err))
    errMsg <- err
  })
  
  print('Plotting')
  if (should_plot) {
    is_err = tryCatch({
      plot_violin_for_exp_path(exp_path=exp_path, burn_in_fraction = burn_in_fraction, thinning=thinning)
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
