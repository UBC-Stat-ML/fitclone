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




