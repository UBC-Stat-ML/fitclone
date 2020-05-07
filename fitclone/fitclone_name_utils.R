library(tools)
#------------------
## Clone dic utils
#------------------

get_clone_dic_for_seq <- function(a_seq) {
  data.frame(old_K = a_seq, 
             is_ref = FALSE, letters = LETTERS[seq(length(a_seq))], color = get_cluster_colours(length(a_seq))[1:length(a_seq)],
             pretty_names = get_pretty_names_for_loci(a_seq), stringsAsFactors = F)
}

get_dummy_clone_dic <- function(candi_edge) {
  #candi_edge <- candi_edge[order(candi_edge$clone_id), ]
  data.frame(old_K = candi_edge$clone_id, 
             is_ref = FALSE, letters = LETTERS[seq(nrow(candi_edge))],
             pretty_names = get_pretty_names_for_loci(candi_edge$clone_id), stringsAsFactors = F)
}

sugar_dummy_clone_dic <- function(ts, aug_cut, new_clades, tcc) {
  candi_edge <- get_candi_edge(the_cut = NULL, aug_cut = aug_cut, ts = ts, tcc = tcc, new_clades = new_clades)
  # candi_edge = ts[ts$clone_id %in% names(aug_cut), ]
  # 
  # candi_edge <- candi_edge[!(candi_edge$clone_id %in% new_clades), ]
  # for (nc in new_clades) {
  #   temp <- data.frame(clone_id = nc, frac = NA)
  #   candi_edge <- bind_rows(candi_edge, temp)
  # }
  get_dummy_clone_dic(candi_edge)
}


get_aug_cut <- function(g = NULL, datatag, edge_list_path) {
  if (is.null(g)) {
    g <- read_ltm_tree(edge_list_path)
  }
  the_cut <- get_the_cut(datatag, edge_list_path, g=g)
  add_siblings_cut(datatag = datatag, edge_list_path = edge_list_path, the_cut = the_cut, the_g = g)
}

# 
# get_new_aug <- function(dt) {
#   split_by_genotype_driver(aug_path = )
# }

get_cells_intersect_aug_cut_datatag <- function(aug_cut, datatag) {
  cells <- colnames(load_new_cn_data(datatag))
  for (cn in names(aug_cut)) {
    aug_cut[[cn]] <- aug_cut[[cn]][aug_cut[[cn]] %in% cells]
    if (length(aug_cut[[cn]]) == 0) aug_cut[[cn]] <- NULL
  }
  aug_cut
}

subtract_lists <- function(list1, list2) {
  res <- list()
  for (cn in names(list1)) {
    res[[cn]] <- setdiff(list1[[cn]], list2[[cn]])
    if (length(res[[cn]]) == 0) res[[cn]] <- NULL
  }
  res
}


get_dummy_candid_edge <- function(g, aug_cut) {
  ntotal <- sum(get_tcc(g)$Freq)
  frac <- unlist(lapply(names(aug_cut), function(x) length(aug_cut[[x]])/ntotal))
  data.frame(clone_id = names(aug_cut), frac = frac, stringsAsFactors = F)
}

# TODO: Add the additional fraction data for missing edges
# TODO: Add the label infront of the clade
get_candi_edge <- function(the_cut, aug_cut, ts, tcc, new_clades) {
  #print(names(aug_cut))
  if (is.null(aug_cut) & !is.null(the_cut)) {
    candi_edge <- ts[ts$clone_id %in% unlist(the_cut), ]
  } else {
    #new_clades <- setdiff(names(aug_cut), unlist(the_cut))
    candi_edge <- ts[ts$clone_id %in% names(aug_cut), ]
    candi_edge <- candi_edge[!(candi_edge$clone_id %in% new_clades), ]
    for (nc in new_clades) {
      # nc = new_clades[[1]]
      frac <- length(aug_cut[[nc]]) / sum(tcc$Freq)
      temp <- data.frame(clone_id = nc, frac = frac)
      #temp <- data.frame(clone_id = nc, frac = NA)
      candi_edge <- bind_rows(candi_edge, temp)
    }
  }
  candi_edge
}

universal_clone_dic <- function(ts, aug_cut, new_clades, tcc) {
  config_path <- file.path(exp_path, 'config.yaml')
  if (!is.null(exp_path) & file.exists(config_path)) {
    fitness_exp_path <- file.path(dirname(exp_path), fitclone_exp_dir)
    clone_dic <- get_clone_dic(datatag, fitness_exp_path)
    clone_dic$colour <- get_cluster_colours(nrow(clone_dic))
    clone_dic$K <- as.character(clone_dic$K)
    dummy_clone_dic <- clone_dic
  } else {
    dummy_clone_dic <- sugar_dummy_clone_dic(ts = ts, aug_cut = aug_cut, new_clades = new_clades, tcc = tcc)
  }
  dummy_clone_dic
}

fp_get_all_datatags <- function(include_holiday = FALSE) {
  dd <- data.frame(datatags = c('SA039', 'SA906a', 'SA906b', 'SA532', 'SA609', 'SA609X3X8a', 'SA000'), 
             labels = c('p53 wildtype', 'p53-/-a', 'p53-/-b', 'Her2+', 'TNBC', 'TNBC mixture', 'TNBC Rx'), 
             labels_2 = c('p53 WT', 'p53-/-a', 'p53-/-b', 'Her2+', 'TNBC', 'TNBC-mixture', 'TNBC-Rx'), 
             modelsystem = c('Cell lines', 'Cell lines', 'Cell lines', 'PDX', 'PDX', 'PDX', 'PDX'),
             stringsAsFactors = F)
  dd$latex_labels = c(sprintf("\\textit{%s}", dd$labels_2[1:3]), dd$labels_2[4:7])
  dd$datatag <- dd$datatags
  
  if (include_holiday) {
    dd <- data.frame(datatags = c('SA039', 'SA906a', 'SA906b', 'SA532', 'SA609', 'SA609X3X8a', 'SA000', 'SA001'), 
                     labels = c('p53 wildtype', 'p53-/-a', 'p53-/-b', 'Her2+', 'TNBC', 'TNBC mixture', 'TNBC Rx', 'TNBC Rx Un'), 
                     labels_2 = c('p53 WT', 'p53-/-a', 'p53-/-b', 'Her2+', 'TNBC', 'TNBC-mixture', 'TNBC-Rx', 'TNBC-Rx-Un'), 
                     modelsystem = c('Cell lines', 'Cell lines', 'Cell lines', 'PDX', 'PDX', 'PDX', 'PDX', 'PDX'),
                     stringsAsFactors = F)
    dd$datatag <- dd$datatags
  }
  
  dd
}


list_2_df <- function(the_list, col_names = NULL) {
  df <- data.frame(v1 = names(the_list), v2 = unlist(unname(the_list)), stringsAsFactors = F)
  if (!is.null(col_names)) {
    colnames(df) <- col_names
  }
  df %>% as_tibble()
}


list_to_data_frame <- function(clust_list, clone_dic = NULL) {
  # TODO: incorporate the clone_dic
  dat <- data.frame(single_cell_id = unlist(clust_list, use.names = F), cluster = NA, stringsAsFactors = F)
  for (nl in names(clust_list)) {
    dat$cluster[dat$single_cell_id %in% clust_list[[nl]]] <- nl
  }
   
  if (!is.null(clone_dic)) {
    dat <- dplyr::left_join(dat, clone_dic, by=c("cluster"="old_K"))
    dat$cluster <- dat$letters
    dat <- dat[, c('single_cell_id', 'cluster')]
  }
    
  dat
}

data_frame_to_list <- function(clustering, clone_dic) {
  if (!is.null(clone_dic)) {
    clustering = dplyr::inner_join(clustering, clone_dic, by=c('cluster'= 'letters'))
    clustering$cluster <- clustering$old_K
  }
  clusts = unique(clustering$cluster)
  res <- list()
  for (cl in clusts) {
    res[[cl]] <- clustering$single_cell_id[clustering$cluster == cl]
  }
  res
}


#------------------
## Datatag utils
#------------------
get_treatment_datanames <- function() {
  #c("SA609aRx8p", "SA609bRx8p", "SA609cRx8p", "SA609aRx8p", "SA609bRx4p", "SA609cRx8p")
  c("SA609aRx8p", "SA609aRx4p", "SA609bRx8p")
}



get_treatment_sane_name <- function() {
   list('SA609aRx4p' = 'SA609-Rx4-X4', 
                              'SA609aRx8p' = 'SA609-Rx8-X6', 
                              'SA609bRx8p' = 'SA609-Rx8-X4')
}


get_datatag_sane_names <- function() {
  res <- list()
  res[['SA609X3X8a']] <- 'Mixture'
  res[['SA609']] <- 'Original'
  res <- append(res, get_treatment_sane_name())
  res
}

get_core_dat_path <- function() '~/Desktop/SC-1311/core_fitness_data.rds'

load_core_dat <- function(datatag = NULL) {
  core_dat <- readRDS(get_core_dat_path())
  
  if (!is.null(datatag)) {
    if (datatag == 'SA666') {
      #core_dat <- core_dat %>% dplyr::filter(!is.na(drug))
      core_dat <- core_dat %>% dplyr::filter(datatag %in% get_treatment_datanames())
    } else if (datatag %in% c('SA906a', 'SA906b')) {
      core_dat <- core_dat[core_dat$datatag == 'SA906' & core_dat$branch == gsub('SA906', '', datatag), ]
    } else {
      core_dat <- core_dat %>% dplyr::filter(datatag == !!datatag) %>% as.data.frame()
    }
  }
  
  core_dat
}


load_cell_cycle_dat <- function() {
  readRDS('~/Desktop/SC-1311/cell_cycle_state_all.rds')
  # Only keep uique rows
  # qq = readRDS('~/Desktop/SC-1311/cell_cycle_state_all.rds')
  # qq <- qq[!duplicated(qq$cell_id), ]
  # saveRDS(qq, '~/Desktop/SC-1311/cell_cycle_state_all.rds')
}


function() {
  dat <- readRDS('~/Desktop/SC-1311/cell_cycle_state_all.rds')
  #new_dat <- read.csv('~/Desktop/SC-1311/cell_cycle_rest_NEW_missing_SA777.csv')
  new_dat <- read.csv('~/Desktop/SC-1311/cell_cycle_rest_NEW_SA609X3X8b.csv')
  head(new_dat)
  table(new_dat$library_id)
  core_dat <- load_core_dat()
  cd <- core_dat[, c('library_id', 'datatag')]
  new_dat <- left_join(new_dat, cd, by=c())
  #which(!(unique(new_dat$library_id) %in% cd$library_id))
  
  dat <- rbind(dat, new_dat)
  saveRDS(dat, '~/Desktop/SC-1311/cell_cycle_state_all.rds')
}

# From aug_cut to timepoints
cell_name_2_timepoint <- function(res) {
  tmp <- list()
  for (cn in names(res)) {
    tmp[[cn]] <- parse_cell_names(res[[cn]])
  }
  tmp
}

get_edge_list_path_for_datatag <- function(dt) {
  sa609_config <- load_main_configs(datatag = dt)
  sa609_config$all_cells$edge_list_path
}

get_exp_path_for_datatag <- function(dt) {
  sa609_config <- load_main_configs(datatag = dt)
  dt_batch_path <- file.path('~/Desktop/SC-1311', dt, 'batch_runs', sa609_config$all_cells$batch_name)
  file.path(dt_batch_path, 'outputs', sa609_config$all_cells$exp_dir)
}

get_original_tree_path_for_datatag <- function(dt) {
  # dt = 'SA609'
  sa609_config <- load_main_configs(datatag = dt)
  sa609_config$original_tree_path
}

get_fitness_exp_path_for_datatag <- function(dt) {
  # dt = 'SA609'
  sa609_config <- load_main_configs(datatag = dt)
  dt_batch_path <- file.path('~/Desktop/SC-1311', dt, 'batch_runs', sa609_config$all_cells$batch_name)
  dt_exp_path <-  file.path(dt_batch_path, 'outputs', sa609_config$all_cells$exp_dir)
  file.path(dirname(dt_exp_path), sa609_config$all_cells$fitclone_exp_dir) 
}

get_theta_path_for_datatag <- function(dt) {
  dt_fitclone_exp_dir <- get_fitness_exp_path_for_datatag(dt)
  file.path(dt_fitclone_exp_dir, 'infer_theta.tsv.gz')
}


get_fitness_datatags <- function() return(c('SA609', 'SA532', 'SA906a', 'SA906b', 'SA039', 'SA666', 'SA609X3X8a', 'SA000'))

get_SA1000_datatags <- function() return(c('SA609', 'SA609X3X8a', 'SA000', get_treatment_datanames()))


get_main_config_path <- function() return('~/projects/fitness/fitclone_path_configs.yaml')
get_backup_path <- function() return('~/projects/fitness/config_backups')
get_backup_name_for_file <- function(file_name) paste0('backup_', generate_random_str(), '_', generate_time_str(), '_', basename(file_name))

save_backup_for_file <- function(file_path) {
  file.copy(file_path, file.path(get_backup_path(), get_backup_name_for_file(file_path)))
}

load_main_configs <- function(datatag = NULL, backuppath = NULL) {
  configs <- yaml.load_file(get_main_config_path())
  if (!is.null(backuppath)) {
    configs <- yaml.load_file(backuppath)
    print('WARNING! USING backup path to load the configs...')
  }
  
  if (!is.null(datatag)) {
    configs <- configs[[datatag]]
  }
  configs
}


load_split_cut_configs <- function(datatag = NULL) {
  configs <- yaml.load_file('~/projects/fitness/fitclone_split_cut.yaml')
  if (!is.null(datatag)) {
    configs <- configs[[datatag]]
  }
  configs
}

load_cut_configs <- function(datatag = NULL) {
  configs <- yaml.load_file('~/projects/fitness/fitclone_cut_configs.yaml')
  if (!is.null(datatag)) {
    configs <- configs[[datatag]]
  }
  configs
}

get_skeleton_main_config <- function(datatag) {
  cond_list <- list(exp_dir='', batch_name='', edge_list_path='')
  
  skelton <- list(
    original_tree_path = '', 
    high_quality_condition = cond_list,
    all_cells = cond_list,
    #fitness_analysis = cond_list
    fitclone_batch_name = '',
    fitclone_exp_dir = ''
  )
}


reset_main_config <- function(datatag = NULL) {
  if (is.null(datatag)) stop('Please set datatag.')
  save_backup_for_file(file_path = get_main_config_path())
  configs <- load_main_configs()
  configs[[datatag]] <- get_skeleton_main_config()
  write_yaml(configs, get_main_config_path())
}

update_main_config <- function(datatag, keyval, use_all_cells=TRUE) {
  configs <- load_main_configs()
  save_backup_for_file(file_path = get_main_config_path())
  kk = names(keyval)
  for (kkk in kk) {
    vv = keyval[[kkk]]
    # Special treatment for original path
    if (kkk == 'original_tree_path') {
      configs[[datatag]][[kkk]] <- vv
    } else {
      if (use_all_cells) {
        configs[[datatag]]$all_cells[[kkk]] <- vv
      } else {
        configs[[datatag]]$high_quality_condition[[kkk]] <- vv
      }
    }
  }
  
  write_yaml(configs, get_main_config_path())
}

suggest_dir_structure <- function(use_all_cells=NULL) {
  if (is.null(use_all_cells)) stop('Need to set a condition')
  exp_dir <- ifelse(use_all_cells, 'plots', 'report')
  
  list(batch_name=generate_date_str(lower_case = F),
       exp_dir=exp_dir)
}


get_passages_for_isogenic <- function() {
  #lib_ids <- c('A96225B', 'A96225C', 'A96181C')
  #passages <- c('X10', 'X30', 'X50')
  core_dat = readRDS('~/Desktop/SC-1311/core_fitness_data.rds')
  core_dat %>% dplyr::select(library_id, timepoint) %>% dplyr::rename(lib_ids=library_id, passages=timepoint)
  #data.frame(lib_ids, passages)
}

# Find the library_id and cross_reference with the ticket to find passage number
# TODO: Cross reference with Emma to get days
parse_cell_names_for_isogenic <- function(cell_names) {
  # cell_names = colnames(mat)[-c(1,2,3,4)]
  # Main referene: https://www.bcgsc.ca/jira/browse/SC-721
  # 184-hTERT L9, 99.5 TP53 null
  
  # We've modified isogenic cell lines to distinguish the branch (either a or b)
  libids <- gsub('(SA[0-9]+)(a|b)-([A-Z0-9]+)-.*', '\\3', cell_names)
  passage_ref <- get_passages_for_isogenic()
  res <- passage_ref$passages[match(libids, passage_ref$lib_ids)]
  # Check for GM cells
  if (any(is.na(res))) {
    res[which(is.na(res))] <- parse_cell_names(cell_names[which(is.na(res))])
  }
  res
}

parse_cell_names_for_treatment <- function(cell_names) {
  gsub('SA[0-9][0-9][0-9][a-z]Rx[0-9]+[a-z](X[0-9]+).*', '\\1', cell_names)
  #SA609bRx8pX4XB01729-A95635D-R65-C39
}


get_drug_dic <- function() {
  list('p' = 'Paclitaxel')
}

get_drug_for_treatment <- function(cell_names) {
  gsub('SA[0-9][0-9][0-9][a-z]Rx[0-9]+([a-z])(X[0-9]+).*', '\\1', cell_names)
  
}

get_dosage_for_treatment <- function(cell_names) {
  as.numeric(gsub('SA[0-9][0-9][0-9][a-z]Rx([0-9]+)([a-z])(X[0-9]+).*', '\\1', cell_names))
}

get_branch_for_treatment <- function(cell_names) {
  gsub('SA[0-9][0-9][0-9]([a-z])Rx([0-9]+)([a-z])(X[0-9]+).*', '\\1', cell_names)
}

get_isogenic_SA_values <- function() {
  return(c('SA1101', 'SA906a', 'SA906b', 'SA906'))
}


parse_cell_names_default <- function(cell_names) {
  gsub('SA[0-9]+(X[0-9]+).*', '\\1', cell_names)
}

parse_cell_names_default_old <- function(cell_names) {
  cells <- c()
  
  for (cell_label in cell_names) {
    time_point <- strsplit(cell_label, '-')[[1]][1]
    time_point <- gsub('SA[0-9]+(X[0-9]+).*', '\\1', time_point)
    if (length(cells) == 0)
      cells <- c(time_point)
    else 
      cells <- c(cells, time_point)
  }
  return(cells)
}


is_treatment_cell <- function(cell_names, rm_loci = TRUE) {
  res <- !grepl('SA[0-9][0-9][0-9]X[0-9].*', cell_names)
  if (rm_loci) {
    res <- res & !grepl('locus_', cell_names)
  }
  res
}




get_libid_from_cell_names <- function(cell_names) {
  # TODO: add the condition for SA906 and SA666
  gsub('SA([0-9]|[A-Z]|[a-z])+-(A([0-9]|[A-Z])+)-.*', '\\2', cell_names)
}


get_sampleid_from_cell_names <- function(cell_names) {
  gsub('([A-Z]*|[0-9]*)-.*', '\\1', cell_names)
}


parse_cell_names <- function(cell_names) {
  # Find libids
  #libids <- libid_from_cell_id(cell_names)
  libids <- get_libid_from_cell_names(cell_names)
  cell.dat <- data.frame(library_id = libids, cellids = cell_names, stringsAsFactors = F)
  if (datatag %in% c('SA922', 'SA922n')) {
    core_dat <- load_core_dat(datatag)
  } else {
    core_dat <- load_core_dat()
  }
  
  cell.dat <- dplyr::left_join(cell.dat, core_dat[, c('timepoint', 'library_id')], by=c('library_id'))
  stopifnot(all(cell_names == cell.dat$cellids, na.rm = T))
  
  # Set SA928 to itself
  cell.dat$timepoint[grepl('SA928', cell.dat$cellids)] <- cell.dat$cellids[grepl('SA928', cell.dat$cellids)]
  
  cell.dat$timepoint
}

# Returns timepoint
parse_cell_names_old <- function(cell_names) {
  # Check for isgenic cells
  for (isocell in get_isogenic_SA_values()) {
    # isocell = get_isogenic_SA_values()[2]
    if (any(grep(isocell, cell_names))) {
      return(parse_cell_names_for_isogenic(cell_names))
    }
  }
  
  # Handle a mixed situation
  # vgn <- V(g)$name; cell_names <- vgn[!grepl('locus_', vgn) & !grepl('root', vgn)]
  # 
  has_treatment <- any(unlist(lapply(get_treatment_datanames(), function(x)  any(grep(x, cell_names)))))
  if (has_treatment) {
    res <- data.frame(cell_names = cell_names, tp = NA)
    
    # Parse the rest
    default_i <- !is_treatment_cell(res$cell_names)
    res$tp[default_i] <- parse_cell_names_default(res$cell_names[default_i])
    res$tp[!default_i] <- parse_cell_names_for_treatment(res$cell_names[!default_i])
    
    return(res$tp)
  } 
  
  # Handle the mixture case
  if (any(grepl('SA609X3X8', cell_names))) {
    libids <- gsub('(SA([0-9]|[A-Z])+)-([A-Z0-9]+)-.*', '\\3', cell_names)
    core_dat <- load_core_dat() %>% dplyr::filter(grepl('SA609X3X8', datatag))
    res <- core_dat$timepoint[match(libids, core_dat$library_id)]
    # Check for GM cells
    if (any(is.na(res))) {
      res[which(is.na(res))] <- parse_cell_names(cell_names[which(is.na(res))])
    }
    return(res)
  }
  
  
  return(parse_cell_names_default(cell_names))
}

get_in_dir_for_datatag <- function(datatag) {
  sprintf('~/Desktop/SC-1311/%s/processed_data', datatag)
}



load_new_cn_data <- function(datatag, filter_by_timepoint=NULL) {
  in_dir <- get_in_dir_for_datatag(datatag)
  
  out_path <- file.path(in_dir,  'cnv_data.rds')
  if (file.exists(out_path)) 
    return(readRDS(out_path))
  
  dat <- as.data.frame(fread(file.path(in_dir, 'cnv_data.csv')))
  #dat <- read.csv(file.path(in_dir, 'cnv_data.csv'))
  
  # Filter for cells in timepoint
  if (!is.null(filter_by_timepoint)) {
    dat <- dat[grepl(paste0(datatag, timepoint), dat$single_cell_id), ]
  }
  
  # tidy to wide
  value.var = 'copy_numer'
  if (is.null(dat[[value_var]])) 
    value.var = 'state'
  
  if (is.null(dat$single_cell_id)) {
    dat$single_cell_id <- dat$cell_id
    dat$cell_id <- NULL
  }
  
  
  mat <- reshape2::dcast(dat, chr+start+end ~ single_cell_id, value.var = c(value.var))
  rownames(mat) <- paste0(mat$chr, '_', mat$start, '_', mat$end)
  mat <- mat[, -c(1:3)]
  mat <- mat[, grepl(datatag, colnames(mat))]
  saveRDS(mat, out_path)
  mat
}


#------------------
## Peek alias
#------------------
headm <- function(amat, n = 10, ncol= 10) head(amat[, 1:ncol], n = n)

peekm <- function(amat, n=10, ncol=10) {
  #head(amat[sample(1:nrow(amat), min(nrow(amat), n)), 1:(min(ncol,ncol(amat)))], n = n)
  rnd_rows <- sample(1:nrow(amat), min(nrow(amat), n))
  rnd_cols <- sample(1:ncol(amat), min(ncol,ncol(amat)))
  head(amat[rnd_rows, rnd_cols], n = n)
}

len <- function(...) length(...)

lenu <- function(...) length(unique(...))

'%==%' <- function(x, y) {(!is.null(x) && x == y)}

'%ni%' <- function(x,y)!('%in%'(x,y))

fasttable <- function(...) as.data.frame(fread(..., stringsAsFactors = F))

fasttablet <- function(...) {
  fread(..., stringsAsFactors = F) %>% as_tibble()
}

nloci <- function(somepath) lenu(fasttable(somepath)$loci)

openm <- function(somepath) system(sprintf('open %s', somepath))


#------------------
## File rnd utils
#------------------
has_tag <- function(fpath) {
  # fpath = files[[1]]
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
  #fpath = '/Users/sohrabsalehi/Desktop/SC-1311/SA906b/batch_runs/Feb15/outputs/plots/tree_traj_SA906b_CSEDK4610D__coarse_raccoon.png'
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
  # timestr = generate_time_str()
  posix_time <- as.POSIXct(x = timestr, format = "%Y%m%d%H-%M-%S")
  format(posix_time, "%Y%m%d%H-%M-%S")
  
  format(posix_time, "%a %b %X %Y")
}

# Add an adjective_animal_rnd_str
generate_random_funny_string <- function() {
  #adj <- readRDS('~/Desktop/SC-1311/etc/eng_adjectives.rds')
  #animals <- readRDS('~/Desktop/SC-1311/etc/eng_animals.rds')
  #set.seed(9)
  #set.seed(NULL)
  #r1 <- tolower(base::sample(adj, 1))
  #r2 <- tolower(base::sample(animals, 1))
  
  
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

get_rnd_str_from_edge_list_path <- function() {
  
}

old_generate_random_funny_string <- function() {
  adj = readRDS('~/Desktop/SC-1311/etc/eng_adjectives.rds')
  animals = readRDS('~/Desktop/SC-1311/etc/eng_animals.rds')
  #set.seed(9)
  r1 <- tolower(sample(adj, 1))
  r2 <- tolower(sample(animals, 1))
  
  paste0(r1, '_', r2, '_', generate_random_str())
  gsub(' ', '_', paste(r1, r2, generate_random_str(), sep = '_'))
}

old_get_pretty_str_from_funny_str <- function(input_str = NULL) {
  if (is.null(input_str))
    input_str <- generate_random_funny_string()
  # Remove underscore and add the tag in paranthesis
  tokens = strsplit(input_str, '_')[[1]]
  pretty_str <- stringr::str_to_title(tokens[-c(length(tokens))])
  paste0(pretty_str[1], ' ', pretty_str[2], ' (', tokens[length(tokens)],  ')')
}

curate_funny_strings <- function() {
  adj <- readLines('~/Desktop/SC-1311/etc/eng_adjectives.txt')
  saveRDS(adj, '~/Desktop/SC-1311/etc/eng_adjectives.rds')
  
  animals <- readLines('~/Desktop/SC-1311/etc/eng_animals.txt')
  saveRDS(tolower(animals), '~/Desktop/SC-1311/etc/eng_animals.rds')
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
  # fp = '/Users/sohrabsalehi/Desktop/SC-1311/SA609/phylogeny/cuts/tree_traj_SA609_LVQLJ2710I__new_falcon__2019012113-59-52.png'
  tail(strsplit(strsplit(basename(fp), '__')[[1]][1], '_')[[1]], 1)
}


get_convert_cmd <- function(prefix = 'collection') sprintf('convert $(ls -v1 *.png) %s_%s.pdf', prefix, tolower(format(Sys.time(), "%b_%d")))
get_collection_cmd <- function(prefix = 'collection') sprintf('convert $(ls -v1 *.png) %s.pdf', prefix)



