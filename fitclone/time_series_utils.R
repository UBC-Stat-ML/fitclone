library(yaml)
library(plyr)
ts_utils_read_config <- function(exp_path) {
  #exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_201706-19-132958/'
  config_path = file.path(exp_path, 'config.yaml')
  if (file.exists(config_path))
    config = yaml.load_file(config_path)
  else
    return(NULL)
  return(config)
}


ts_utils_flatten_config <- function(config_list, list_valued_keys=NULL) {
  if (is.null(list_valued_keys)) {
    list_valued_keys = list()
    for (key in names(config_list)) {
      if (length(config_list[[key]]) > 1 )
        list_valued_keys = append(list_valued_keys, key)
    }
  }
  
  temp_list = config_list[!(names(config_list) %in%  unlist(list_valued_keys))]
  for (key in list_valued_keys) {
    index = 1
    for (x in config_list[[key]]) {
      temp_list[[paste0(key, index)]] = x
      index = index + 1
    }
  }
  
  print('ignoring true_x0')
  temp_list$true_x0=NULL
  
  return(data.frame(temp_list))
}

ts_utils_wrap_up_batch <- function(batch_path) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  dat = NULL
  for (dir in dirs) {
    is_err = tryCatch({
      #dir='/Users/ssalehi/Desktop/pgas_sanity_check/FiveK_201707-01-031144.187799/outputs/o_6_0_F3K27_201707-01-031220.860928'
      mae = yaml.load_file(file.path(dir, 'MAE.yaml'))$s_MAE
      rmse = yaml.load_file(file.path(dir, 'predictsummary.yaml'))
      config = yaml.load_file(file.path(dir, 'config.yaml'))
      
      config$s_mae = mae
      
      if (is.null(rmse$rmse_infer_before_learn_time))
        rmse$rmse_infer_before_learn_time = NA
      
      if (is.null(rmse$rmse_after_learn_time))
        rmse$rmse_after_learn_time = NA
      
      if (is.null(rmse$rmse))
        rmse$rmse = NA
      
      config$x_predict_rmse = rmse$rmse
      config$x_predict_rmse_after_learn_time = rmse$rmse_after_learn_time
      
      config$x_infer_rmse_before_learn_time = rmse$rmse_infer_before_learn_time
      #config$x_infer_rmse_after_learn_time = ifelse(!is.na(rmse$rmse_infer_before_learn_time), rmse$rmse_after_learn_time, NA)
      
      temp = ts_utils_flatten_config(config)

      #print(temp[, !(colnames(temp) %in% c('config_chunk_path', 'out_path', 'original_data'))])
      
      if (is.null(dat))
        dat = temp
      else
        dat = plyr::rbind.fill(dat, temp)
        #dat = rbind(dat, temp)
      
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
  
  if (!is.null(dat)) {
    gz1 <- gzfile(file.path(batch_path, 'outputs', 'summary.tsv.gz'), "w")
    write.table(dat, gz1, sep='\t')
    close(gz1)
  }
}


ts_utils_apply_to_batch <- function(batch_path, func_list) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    is_err = tryCatch({
      
      for (a_func in func_list) {
        is_func_err = tryCatch({
          res = a_func(dir)
          
        }, error=function(err) {
          print(sprintf('Error - %s', err))
          errMsg <- err
        })
        
        if(inherits(is_func_err, "error")) next
      }
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
}




tt_utils_wrap_up_batch_plot <- function(batch_path) {
  #ss = read.table('/Users/ssalehi/Desktop/pgas_sanity_check/TenKLooseBall_201706-28-172437.967739/outputs/summary.tsv.gz')
  ss = read.table(file.path(batch_path, 'outputs/summary.tsv.gz'))
  ss$id = paste0(ss$infer_epsilon, '_', ss$infer_epsilon_tolerance)
  height = 3000
  height=height
  width=1.0/.44 * height
  pdfCoef=220
  file_path = file.path(batch_path, 'outputs', paste0('summary', '.pdf'))
  pdf(file = file_path, width=width/pdfCoef, height=height/pdfCoef)
  
  p <- ggplot(ss, aes(id, x_predict_rmse_after_learn_time)) +
    geom_boxplot() +
    theme(text = element_text(size=20), plot.title = element_text(size = 20), 
          axis.text.x = element_text(angle=90, hjust=1),
          legend.text = element_text(face = 'bold')) 
  print(p)
  p <- ggplot(ss, aes(id, s_mae)) +
    geom_boxplot() +
    theme(text = element_text(size=20), plot.title = element_text(size = 20), 
          axis.text.x = element_text(angle=90, hjust=1),
          legend.text = element_text(face = 'bold'))
  print(p)
  dev.off()
}

tt_utils_list_sort_predictions <- function(batch_path, nItems=10, give_best=T) {
  summary_path = file.path(batch_path, 'outputs/summary.tsv.gz')
  if (!file.exists(summary_path))
    ts_utils_wrap_up_batch(batch_path=batch_path)
  
  dat = read.table(summary_path, stringsAsFactors = F)
  dat$out_name = basename(dat$out_path)
  head(dat)
  #dat.filtered = dat[order(dat$x_predict_rmse_after_learn_time, decreasing = !give_best)[1:nItems], ]
  dat.filtered = dat[order(dat$x_predict_rmse_after_learn_time, decreasing = !give_best), ]
  dat.filtered = dat.filtered[, !(colnames(dat.filtered) %in% c('config_chunk_path', 'out_path', 'original_data'))]
  write.table(dat.filtered, file.path(batch_path, 'outputs/summary.sorted.tsv'), sep='\t', row.names = F)
  write.table(dat.filtered[,c('out_name', 'x_predict_rmse_after_learn_time')], file.path(batch_path, 'outputs/short.summary.sorted.tsv'), sep='\t', row.names = F)
  dat.filtered$out_name[1:nItems]
}

#dat = read.table(file.path('/Users/ssalehi/Desktop/pgas_sanity_check/500fine_201707-02-173623.104472/outputs', 'summary.sorted.tsv'), header=T, sep='\t')
#head(dat)
#mean(as.numeric(dat$run_time), na.rm = T)/3600
#ts_utils_wrap_up_batch(batch_path="/Users/ssalehi/Desktop/pgas_sanity_check/K2500short_201707-03-134111.238639")
#tt_utils_wrap_up_batch_plot(batch_path ='/Users/ssalehi/Desktop/pgas_sanity_check/K2500short_201707-03-134111.238639')
#tt_utils_list_sort_predictions(batch_path ='/Users/ssalehi/Desktop/pgas_sanity_check/K2500short_201707-03-134111.238639', give_best = F)
#tt_utils_list_sort_predictions(batch_path ='/Users/ssalehi/Desktop/pgas_sanity_check/K2500short_201707-03-134111.238639', give_best = T)
#tt_utils_list_sort_predictions(batch_path ='/Users/ssalehi/Desktop/pgas_sanity_check/500fine_201707-02-173623.104472', give_best = T)
#tt_utils_list_sort_predictions(batch_path, give_best = T)
