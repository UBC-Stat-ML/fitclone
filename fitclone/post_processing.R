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
  as.numeric(gsub('.*/steps_([0-9]+)/.*', '\\1', config[['original_data']]))
}

extact_reference_for_config <- function(config) {
  as.numeric(gsub('.*/references/([0-9]+)/.*', '\\1', config[['original_data']]))
}

extact_times_for_config <- function(config) {
  dp <- config[['original_data']]
  if (!file.exists(dp)){
    dp <- paste0(dp, '.gz')
  }
  dat = read.table(dp, header = T)

  return(unique(dat$time))
}

extact_predict_for_config <- function(config) {
  if (!is.null(config[['do_predict']]))
    return(config[['do_predict']])
  else {
    # see how many timepoints back was the learn time
    lt = config[['learn_time']]
    et = config[['end_time']]
    ut <- extact_times_for_config(config)
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

