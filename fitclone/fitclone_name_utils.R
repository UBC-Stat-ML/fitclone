library(tools)

#------------------
## File rnd utils
#------------------
has_tag <- function(fpath) {
  # Look for alphanumeric string of length 10 possibly enclosed by two underscores
  pattern <- paste0(rep('[0-9|A-Z]', 10), collapse = '')
  # 1. Check for before and after underscore
  if (grepl(paste0('_', pattern, '_'), basename(fpath)) |
      grepl(paste0('_', pattern), basename(fpath))|
      grepl(paste0(pattern, '_'), basename(fpath))) {
    return(TRUE)
  }
  
  return(FALSE)
}

detag_filepath <- function(fpath, remove_analyis_tag = FALSE, analysis_tag=NULL) {
  pattern <- paste0(rep('[0-9|A-Z]', 10), collapse = '')
  pattern <- sprintf('(_%s_|_%s|%s_)', pattern, pattern, pattern)
  temp_name <- gsub(pattern, '',  basename(fpath))
  
  if (remove_analyis_tag & !is.null(analysis_tag)) {
    pattern <- sprintf('(_%s_|_%s|%s_)', analysis_tag, analysis_tag, analysis_tag)
    temp_name <- gsub(pattern, '', temp_name)
  }
    
  file.path(dirname(fpath),  temp_name)
}

remove_delimiters <- function(fpath) {
  temp_name <- basename(fpath)
  temp_name <- gsub('_|-| ', '', temp_name)
  file.path(dirname(fpath), temp_name)
}


generate_random_str <- function(n = 1) {
  a <- do.call(paste0, replicate(5, base::sample(LETTERS, n, TRUE), FALSE))
  paste0(a, sprintf("%04d", base::sample(9999, n, TRUE)), base::sample(LETTERS, n, TRUE))
}

generate_time_str <- function(n = 1) {
  #format(Sys.time(), "%Y%m%d %a %b %d %X %Y")
  format(Sys.time(), "%Y%m%d%H-%M-%S")
}

generate_date_str <- function(lower_case=TRUE) {
  #format(Sys.time(), "%Y%m%d %a %b %d %X %Y")
  if (lower_case)
    tolower(format(Sys.time(), "%b%d"))
  else
    format(Sys.time(), "%b%d")
}


get_pretty_str_for_time_str <- function(timestr) {
  posix_time <- as.POSIXct(x = timestr, format = "%Y%m%d%H-%M-%S")
  format(posix_time, "%Y%m%d%H-%M-%S")
  
  format(posix_time, "%a %b %X %Y")
}

# Add an adjective_animal_rnd_str
generate_random_funny_string <- function() {
  r1 <- as.character(base::sample(1:1000, 1))
  r2 <- as.character(base::sample(1:100, 1))
  
  name_str <- gsub(' ', '_', paste0('__', r1, '_', r2, '__'))
  paste0(name_str, generate_time_str())
}

get_pretty_str_from_funny_str <- function(input_str = NULL, keep_timestamp = TRUE) {
  if (is.null(input_str))
    input_str <- generate_random_funny_string()
  # Remove underscore and add the tag in paranthesis
  tokens <- strsplit(input_str, '__')[[1]]
  pretty_str <- stringr::str_to_title(strsplit(tokens[[2]], '_')[[1]])
  res_str <- paste0(pretty_str, collapse = ' ')
  if (keep_timestamp) {
    timestr <- tokens[length(tokens)]
    timestr <- get_pretty_str_for_time_str(timestr)
    res_str <- paste0(res_str, ' (', timestr,  ')')
  } 
  res_str
}


get_ID_from_edge_list_path <- function(edge_list_path, keep_timestamp = TRUE) {
  if (!is.null(edge_list_path)) {
    if (edge_list_path == '') return('')
    tokens <- strsplit(file_path_sans_ext(basename(edge_list_path)), '__')[[1]]
    temp_str <- tokens[[2]]
    timestamp <- tokens[[3]]
    get_pretty_str_from_funny_str(input_str = paste0('__', temp_str, '__', timestamp), keep_timestamp)
  } else {
    get_pretty_str_from_funny_str(generate_random_funny_string(), keep_timestamp)
  }
}

extract_funny_str_from_edge_list_path <- function(edge_list_path, keep_underscore=FALSE) {
  tokens <- strsplit(file_path_sans_ext(basename(edge_list_path)), '__')[[1]]
  if (!keep_underscore)
    gsub('_', '', tokens[[2]])
  else
    tokens[[2]]
}



get_file_name_postfix <- function(datatag, edge_list_path, tag = '', file_extension = '', add_rnd_str = TRUE) {
  if (edge_list_path == '') {
    id = ''
  } else {
    id <- strsplit(file_path_sans_ext(basename(edge_list_path)), '__')[[1]]
    id <- id[[2]]
  }
  
  rnd_str <-ifelse(add_rnd_str,  paste0('_', generate_random_str()), '')
  sprintf('%s_%s%s__%s%s', tag, datatag, rnd_str, id, file_extension)
}

get_title_str <- function(datatag, n_cells, clone_name = NULL) {
  if (is.null(clone_name)) 
    sprintf('%s (n = %d)', datatag, n_cells)
  else 
    sprintf('%s - Clone %s (n = %d)', datatag, clone_name, n_cells)
}

get_rnd_str_from_file_path <- function(fp) {
  tail(strsplit(strsplit(basename(fp), '__')[[1]][1], '_')[[1]], 1)
}


get_convert_cmd <- function(prefix = 'collection') sprintf('convert $(ls -v1 *.png) %s_%s.pdf', prefix, tolower(format(Sys.time(), "%b_%d")))
get_collection_cmd <- function(prefix = 'collection') sprintf('convert $(ls -v1 *.png) %s.pdf', prefix)



