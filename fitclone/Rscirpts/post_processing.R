# Utilities for post processing
library(plyr)
library(tidyr)
library(yaml)
library(coda)
library(ggplot2)
library(stringr)
library(tools)


source('~/projects/fitness/fitclone_name_utils.R')

#------------------
## Path utils
#------------------
get_main_path_for_datatag <- function(datatag) {
  if (datatag %in% c('SA609', 'SA532', 'htert'))
    return(file.path('~/Desktop/SC-1311/', datatag))
  else
    stop(sprintf('datatag [%s] not recognized.'))
}




#------------------
## Batch utils
#------------------
extract_step_size_for_confing <- function(config) {
  #od = '~/projects/fitclone/figures/raw/supp/SA609/corrupt/steps_4/SA609_dropped_one_sc_dlp_4_clones_Ne_50.tsv'
  as.numeric(gsub('.*/steps_([0-9]+)/.*', '\\1', config[['original_data']]))
}

extact_reference_for_config <- function(config) {
  #od = '~/projects/fitclone/figures/raw/supp/SA609/corrupt/references/1/steps_10/SA609_dropped_one_sc_dlp_4_clones_Ne_500.tsv'
  as.numeric(gsub('.*/references/([0-9]+)/.*', '\\1', config[['original_data']]))
}

extact_times_for_config <- function(config) {
  dp <- config[['original_data']]
  # dp = '/Users/sohrabsalehi/projects/fitclone/figures/raw/supp/SA609/corrupt/fake/autocut/7/steps_4/SA609_dropped_one_sc_dlp_6_clones_Ne_1000.tsv'
  if (!file.exists(dp)){
    dp <- paste0(dp, '.gz')
  }
  dat = read.table(dp, header = T)
  # ut = unique(dat$time)
  return(unique(dat$time))
}

extact_predict_for_config <- function(config) {
  # config = yaml.load_file('/Users/sohrabsalehi/projects/fitclone/presentations/September_25/cumulative/plots_all/o_4_0_OGG2Q_201809-17-195827.927245/config.yaml')
  if (!is.null(config[['do_predict']]))
    return(config[['do_predict']])
  else {
    # see how many timepoints back was the learn time
    lt = config[['learn_time']]
    et = config[['end_time']]
    ut <- extact_times_for_config(config)
    #return(abs(which(ut == et) - which(ut == lt)))
    return(abs(which.min( abs(ut-et)) - which.min(abs(ut-lt))))
  }
}


get_s_for_exp_path <- function(exp_path) {
  param_path = file.path(exp_path, 'infer_theta.tsv.gz')
  dat <- read.table(param_path, stringsAsFactors = F)
  colnames(dat) <- paste0('s', seq(ncol(dat)))
  dat
}

get_s_means_for_exp_path <- function(exp_path) {
  s_dat <- get_s_for_exp_path(exp_path)
  colMeans(s_dat[ceiling(.1*nrow(s_dat):nrow(s_dat)), ])
}


get_params_for_exp_path <- function(exp_path, paramNameList, df = NULL) {
  config = yaml.load_file(file.path(exp_path, 'config.yaml'))
  key_vals = list()
  for (paramName in paramNameList) {
    if (paramName == 'step') {
      df[[paramName]] = extract_step_size_for_confing(config)
    } else if (paramName == 'ref') {
      df[[paramName]] = extact_reference_for_config(config)
    } else if (paramName == 'block_size') {
      df[[paramName]] = config[['K']][1]
    } else if (paramName == 'do_predict') {
      df[[paramName]] = extact_predict_for_config(config)
    } else if (paramName == 's_mean') {
      
    }else if (is.null(config[[paramName]]))  {
      next
    } else {
      df[[paramName]] = config[[paramName]][1]
    }
  }
  df
}

get_batch_path_from_batch_name <- function(batch_name) {
  file.path('/shahlab/ssalehi/scratch/fitness/batch_runs', batch_name)
}

get_exp_names_in_batch <- function(batch_name) {
  batch_path = get_batch_path_from_batch_name(batch_name)
  outputs = file.path(batch_path, 'outputs')
  system(sprintf('ssh shahlab15 ls -1 %s', outputs), intern = T)
}

#------------------
## Batch utils - downloading files from server
#------------------
extract_datatag_from_batch_name <- function(batch_name) {
  datatags = c('SA609', 'SA532', 'SA906a', 'SA906b')
  for (dt in datatags) {
    if (grepl(dt, batch_name))
      return(dt)
  }
}
# For internal use
# These don't belong to any specific experiments
download_batch_specific_files <- function(batch_names, file_names = NULL, force_redownload=FALSE) {
  if (is.null(file_names))
    file_names <- c('all_niter.rds', 'all_theta.rds', 'plots_all.tar.gz')

  for (batch_name in batch_names) {
    # batch_name = batch_names[[1]]
    dt <- extract_datatag_from_batch_name(batch_name)
    out_dir = file.path('~/Desktop/SC-1311', dt, 'batch_runs', batch_name)
    for (fn in file_names) {
      # fn = file_names[[1]]
      out_path = file.path(out_dir, fn)
      
      if (file.exists(out_path) & !force_redownload) {
        print(sprintf('File %s exists. Skipping...', fn))
        next
      }
      
      print(sprintf('Downloading %s', fn))
      
      in_path = file.path(get_batch_path_from_batch_name(batch_name), fn)
      
      dir.create(file.path(out_dir), recursive = T, showWarnings = F)
      cmdStr = sprintf('scp shahlab15:%s %s/', in_path, out_dir)
      system(cmdStr)
    }
  }
}


download_experiment_in_batch <- function(batch_path, exp_names, out_dir, files=NULL) {
  for (exp_dir in exp_names) {
    # exp_dir = exp_names[[1]]
    for (file in files) {
      # file = files[1]
      if (file != 'all_theta.rds') {
        in_path = file.path(batch_path, 'outputs', exp_dir, file)
        dir.create(file.path(out_dir, exp_dir), recursive = T, showWarnings = F)
        cmdStr = sprintf('scp shahlab15:%s %s/', in_path, file.path(out_dir, exp_dir))
      }
      else {
        in_path = file.path(batch_path, file)
        dir.create(file.path(out_dir), recursive = T, showWarnings = F)
        cmdStr = sprintf('scp shahlab15:%s %s/', in_path, file.path(out_dir))
      }
      system(cmdStr)
    }
  }
}


download_file_for_batch <- function(batch_name, exp_name=NULL, file_name) {
  dt <- extract_datatag_from_batch_name(batch_name)
  if (!is.null(exp_name)) {
    in_path = file.path(batch_path, 'outputs', exp_name, file_name)
    out_dir = file.path('~/Desktop/SC-1311', dt, 'batch_runs', batch_name, 'outputs', exp_name) 
  } else {
    in_path = file.path(batch_path, file_name)
    out_dir = file.path('~/Desktop/SC-1311', dt, 'batch_runs', batch_name) 
  }
  
  dir.create(out_dir, recursive = T, showWarnings = F)
  cmdStr = sprintf('scp shahlab15:%s %s/', in_path, file.path(out_dir))
  system(cmdStr)
}




download_exp_files_for_batch <- function(batch_name, datatag=NULL, exp_names = NULL, files=NULL) {
  # batch_name = 'SA609_corrupt_autocut_201809-17-185811.547073'
  #exp_names = as.vector(na.omit(unlist(lapply(1:5, function(ref) res$exp_dir[res$ref == ref][1]))))
  if (is.null(exp_names))
    exp_names = get_exp_names_in_batch(batch_name)
  batch_path = get_batch_path_from_batch_name(batch_name)
  if (is.null(files))
    files = c('full_original_data_trace_.pdf', 'config.yaml', 'predict_trace.pdf', 'theta_hists.pdf')
  out_dir = file.path(get_main_path_for_datatag(datatag), batch_name, generate_random_str())
  dir.create(out_dir)
  download_experiment_in_batch(batch_path = batch_path, exp_names = exp_names, out_dir = out_dir, files = files)
}

# grab_all_s_for_batch(batch_path = batch_path, extra_params = c('ref', 'step', 'do_predict'))

grab_all_s_for_batch <- function(batch_path, datatag='', extra_params=NULL) {
  params_to_grab = c('infer_epsilon', 'Ne', 'proposal_step_sigma', 'seed')
  if (!is.null(extra_params))
    params_to_grab <- append(params_to_grab, extra_params)
  
  res = NULL
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    print(dir)
    is_err = tryCatch({
      temp = get_s_for_exp_path(dir)
      # melt
      temp = reshape2::melt(data=temp, value.name='s', variable.name=c('K'), stringsAsFactors=F)
      temp$K = gsub('s', '', temp$K)
      temp <- get_params_for_exp_path(dir, params_to_grab, temp)
      temp$exp_dir = basename(dir)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
    
    if (is.null(res))
      res = temp
    else 
      res = dplyr::bind_rows(res, temp)
  }
  saveRDS(res, file.path(batch_path, 'all_theta.rds'))
  
  # Print a help command
  print(sprintf('scp shahlab15:%s .', file.path(batch_path, 'all_theta.rds')))
}

grab_all_delta_for_batch <- function(batch_path, datatag='', K = c(0, 5), extra_params=NULL) {
  params_to_grab = c('infer_epsilon', 'Ne', 'proposal_step_sigma', 'seed')
  if (!is.null(extra_params))
    params_to_grab <- append(params_to_grab, extra_params)
  
  res = NULL
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    print(basename(dir))
    is_err = tryCatch({
      pred_path = file.path(dir, 'predict.tsv.gz')
      # pred_path = '/Users/sohrabsalehi/Desktop/viz_sa532_75/SC-1035/re_re_run/exponential_growth/SA532_dropped_one_sc_dlp_3_clones_Ne_500/predict.tsv.gz'
      temp = read.table(pred_path, header = T, stringsAsFactors = F, sep='\t')
      time = unique(temp$time)
      temp = temp[temp$time == max(time), ]
      # ss = temp
      # K = c(0, 5)
      # K = c(0, 2)
      delta = temp$X[temp$K == K[1]] - temp$X[temp$K == K[2]]
      
      temp <- data.frame(delta=delta, np=seq(length(delta)), time=max(time))
      temp <- get_params_for_exp_path(dir, params_to_grab, temp)
      temp$exp_dir = basename(dir)
    }, error=function(err) {
      print(dir)
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
    
    if (is.null(res))
      res = temp
    else 
      res = dplyr::bind_rows(res, temp)
  }
  saveRDS(res, file.path(batch_path, 'all_delta.rds'))
  
  # Print a help command
  print(sprintf('scp shahlab15:%s .', file.path(batch_path, 'all_delta.rds')))
}


# grab_all_mcmc_for_batch(batch_path=batch_path, datatag='SA609', extra_params=c('step','do_predict'))

grab_all_mcmc_for_batch <- function(batch_path, datatag='', extra_params=NULL, grab_s = FALSE) {
  params_to_grab = c('infer_epsilon', 'Ne', 'proposal_step_sigma', 'seed')
  if (!is.null(extra_params))
    params_to_grab <- append(params_to_grab, extra_params)
  
  res = NULL
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    print(dir)
    is_err = tryCatch({
      temp = get_s_for_exp_path(dir)
      ess = coda::effectiveSize(as.mcmc(temp))
      rr = coda::rejectionRate(as.mcmc(temp))
      if (grab_s) {
        temp <- data.frame(ess = ess, rejection_rate=rr, mean_s = colMeans(temp[ceiling(.1*nrow(temp)):nrow(temp), ]),  K = colnames(temp))
      } else {
        temp <- data.frame(ess = ess, rejection_rate=rr)
      }
      
      temp <- get_params_for_exp_path(dir, params_to_grab, temp)
      temp$exp_dir = basename(dir)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
    
    if (is.null(res))
      res = temp
    else 
      res = dplyr::bind_rows(res, temp)
  }
  saveRDS(res, file.path(batch_path, 'all_mcmc.rds'))
  
  # Print a help command
  print(sprintf('scp shahlab15:%s .', file.path(batch_path, 'all_mcmc.rds')))
}

# batch_path = '/shahlab/ssalehi/scratch/fitness/batch_runs/dec_SA609_pred_conditional_201812-17-134345.395964'
# grab_all_niter_for_batch(batch_path = batch_path, extra_params = c('step', 'do_predict', 'pf_n_particles'))

grab_all_niter_for_batch <- function(batch_path, datatag='', extra_params=NULL, burn_in = .1) {
  params_to_grab = c('infer_epsilon', 'Ne', 'proposal_step_sigma', 'run_time', 'seed')
  if (!is.null(extra_params))
    params_to_grab <- append(params_to_grab, extra_params)
  
  res = NULL
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    print(dir)
    is_err = tryCatch({
      temp = get_s_for_exp_path(dir)
      temp <- temp[(round(burn_in*nrow(temp))):nrow(temp), ]
      ess = coda::effectiveSize(as.mcmc(temp))[1]
      rr = coda::rejectionRate(as.mcmc(temp))[1]
      niter = nrow(temp)
      sorder = mean(temp$s1 > temp$s6)
      temp <- data.frame(niter = niter, sorder = sorder, ess = ess, rejection_rate=rr)
      temp <- get_params_for_exp_path(dir, params_to_grab, temp)
      temp$exp_dir = basename(dir)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
    
    if (is.null(res))
      res = temp
    else 
      res = dplyr::bind_rows(res, temp)
  }
  saveRDS(res, file.path(batch_path, 'all_niter.rds'))
  
  # Print a help command
  print(sprintf('scp shahlab15:%s .', file.path(batch_path, 'all_niter.rds')))
}


# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_4_0_51JNT_201903-29-224044.150150'))
# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_5_0_3ZVP2_201903-29-224044.150556'))
# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_6_0_TW3I3_201903-29-224044.150931'))
# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_7_0_CB35Y_201903-29-224044.150100'))
# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_12_0_L6RIV_201903-29-224044.151006'))
# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_13_0_MGWH5_201903-29-224044.148823'))
# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_14_0_7JSAH_201903-29-224044.150676'))
# replot_for_exp_path(file.path(batch_path, 'outputs', 'o_15_0_ZR6AK_201903-29-224044.148346'))


replot_for_exp_path <- function(exp_path) {
  plot_all_for_exp_path(exp_path, ignore_hist=T, ignore_inference=T)
  plot_violin_for_exp_path(exp_path)
  gz_all_for_exp_path(exp_path)
}


#------------------
## Data wrangling
#------------------
ss_spread <- function(ss, value_col_name, var_col_name) {
  ss %>%  
    group_by_at(vars(-value_col_name)) %>%  # group by everything other than the value column. 
    dplyr::mutate(row_id=1:n()) %>% ungroup() %>%  # build group index
    spread(key=var_col_name, value=value_col_name) %>%    # spread 
    dplyr::select(-row_id) %>%
    as.data.frame()
}

# Sort the num_array by the numeric values and then add the tag
# Attempts to convert num_array into numeric array if num_array is string by removing letters
common_sense_factorise <- function(num_array, tag) {
  # num_array = res$Ne
  # tag = 'Ne = '
  if (!is.numeric(num_array)) {
    num_array = as.numeric(gsub('[A-Z]+', '', num_array))
  }
  the_levels = sort(unique(num_array))
  factor(num_array, levels = the_levels, labels = paste0(tag, the_levels), ordered = T)
}

compute_p_greater_than <- function(vec1, vec2) {
  if (length(vec1) != length(vec2)) return(NULL) 
  sum(vec1>vec2)/length(vec1)
}

#------------------
## Plotting
#------------------
GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, draw_group = function(self, data, ..., draw_quantiles = NULL){
  data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
  grp <- data[1,'group']
  newdata <- plyr::arrange(transform(data, x = if(grp%%2==1) xminv else xmaxv), if(grp%%2==1) y else -y)
  newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
  newdata[c(1,nrow(newdata)-1,nrow(newdata)), 'x'] <- round(newdata[1, 'x']) 
  if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
    stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <= 
                                              1))
    quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
    aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
    aesthetics$alpha <- rep(1, nrow(quantiles))
    both <- cbind(quantiles, aesthetics)
    quantile_grob <- GeomPath$draw_panel(both, ...)
    ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
  }
  else {
    ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
  }
})

geom_split_violin <- function (mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE, show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, position = position, show.legend = show.legend, inherit.aes = inherit.aes, params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}

