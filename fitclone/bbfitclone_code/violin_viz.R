# set environment
setup_env <- function() {
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
}
setup_env()

library(RColorBrewer)
library(dplyr)
library(coda)

source('time_series_plotting.R')

#myColors = c( "#a6cee3" ,"#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c" ,"#fdbf6f", "#ff7f00" ,"#cab2d6" ,"#6a3d9a", "#ffff99" ,"#c0c0c0")


get_time_point_labels <- function(datatag, keep_all_tps=FALSE) {
  
  mat = load_new_cn_data(datatag)
  tps = unique(parse_cell_names(colnames(mat)))
  # Remove GM
  tps <- tps[grep('X[0-9]+', tps)]
  tps <- levels(common_sense_factorise(tps, 'X'))
  return(tps)
  
  # if (datatag == 'SA609') {
  #   if (keep_all_tps)
  #     return(c('X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'))
  #   
  #   return(c('X3', 'X4', 'X5', 'X6', 'X7', 'X8'))
  # }
  #   
  # if (datatag == 'SA532') 
  #   return(c('X2', 'X4', 'X5', 'X6', 'X7', 'X8'))
}

get_cluster_colours <- function(nClusters) {
  if (nClusters > 8) {
    #clust_colours <- colorRampPalette(brewer.pal(12, "Set3"))(nClusters)
    clust_colours <- colorRampPalette(brewer.pal(8, "Set2"))(nClusters)
  } else {
    clust_colours <- brewer.pal(nClusters, "Set2")
  }
  clust_colours
} 

get_param_trace <- function(exp_path, burn_in_fraction=.10, thinning=1) {
  print(exp_path)
  mcmc_options = ts_utils_read_config(exp_path) #mcmc_options = list(seed=11, nMCMC=10, nParticles=5, s=(seq(K)/(K+1)))
  if (!is.null(mcmc_options$K_prime)) {
    mcmc_options$block_size = mcmc_options$K
    mcmc_options$K = mcmc_options$K_prime
  }
  
  if (is.null(mcmc_options)) {
    print('WAS NULL')
    mcmc_options = list(seed=11, nMCMC=10, nParticles=-1, s=(seq(K)/(K+1)))
    print('Warning! Using dummy config file')
    return(NULL)
  }
  
  out_dir = file.path(exp_path, 'plots')
  dir.create(out_dir, showWarnings = F)
  
  # plot theta
  param_path = file.path(exp_path, 'infer_theta.tsv.gz')
  if (file.exists(param_path)) {
    param_dat = read.table(param_path, sep='\t', header=TRUE)
    K = ncol(param_dat)
    colnames(param_dat) = paste0('s', seq(K))
    n = nrow(param_dat)
    param_dat = param_dat[seq(burn_in_fraction*n, n, thinning), ]
    param_dat_tall = melt(param_dat, value.name='s', varname=c('k'))
    param_dat_tall$K = gsub('s', '', param_dat_tall$variable)
    param_dat_tall$variable = NULL
    
    return(list(param_dat_tall, param_dat))
  }
}

# WF -> Heatmap colours
get_clone_dic_for_datatag <- function(datatag=NULL, nClones=NULL) {
  dic <- NULL
  if (datatag == 'SA609') {
    #dic <- list('1' ='D', '2' = '6', '3' = 'F', '4' = 'E', '5' = 'C')
    #dic <- list('1'='1', '2'='4', '3'='3', '4'='5', '5'='6')
    dic <- list('1'='2', '2'='4', '3'='3', '4'='5')
    # dic <- list()
    # clonea_names <- paste0(1:6)[-c(2)]
    # for (i in seq_along(clonea_names)) {
    #   dic[paste0(i)] <- clonea_names[i]
    # }
  } else if (datatag == 'SA532') {
    dic <- list('1'='1', '2'='5', '3'='3', '4'='2', '5'='6')
    if (!is.null(nClones)) {
      if (nClones == 4)
        dic <- list('1'='2', '2'='1', '3'='4')
      else 
        dic <- list('1'='6', '2'='2', '3'='5', '4'='3', '5'='4', '6'='7')
    } 
    #dic <- list('1' ='A', '2' = 'D', '3' = 'C', '4' = 'S1', '5' = 'B')
    # dic <- list()
    # clonea_names <- paste0(1:6)[-c(4)]
    # for (i in seq_along(clonea_names)) {
    #   dic[paste0(i)] <- clonea_names[i]
    # }
  } else {
    print(paste0('ERROR! datatag = ', datatag, ' is not implemented.'))
    # TODO: return a default version
    dic <- list()
    index = 1
    for (ll in seq(nClones)) {
      dic[[paste0(index)]] <- ll
      index = index + 1
    }
  }
  dic
}

get_colour_dic_for_datatag <- function(datatag=NULL, full=FALSE, nClones=NULL) {
  colour_dic <- NULL
  if (datatag == 'SA609') {
    #colour_dic = c('C'=myColors[1], '6'=myColors[6], 'D'=myColors[3], 'F'=myColors[4], 'E'=myColors[5])
    if (!is.null(nClones)) {
      ref_clone <- 1
      colour_dic <- get_cluster_colours(nClones)
      names(colour_dic) <- paste0(1:nClones)
      if (!full)
        colour_dic <- colour_dic[-c(ref_clone)]
    } else {
      colour_dic <- get_cluster_colours(6)
      names(colour_dic) <- paste0(1:6)
      if (!full)
        colour_dic <- colour_dic[-c(2)]
    }
  } else if (datatag == 'SA532') {
    #colour_dic = c('S1'=myColors[1], 'A'=myColors[2], 'C'=myColors[3], 'D'=myColors[5], 'B'=myColors[6])
    if (!is.null(nClones)) {
      ref_clone <- 3
      if (nClones == 7)
        ref_clone <- 1
        
      colour_dic <- get_cluster_colours(nClones)
      names(colour_dic) <- paste0(1:nClones)
      if (!full)
        colour_dic <- colour_dic[-c(ref_clone)]
    } else {
      colour_dic <- get_cluster_colours(6)
      names(colour_dic) <- paste0(1:6)
      if (!full)
        colour_dic <- colour_dic[-c(4)]
    }
  } else {
    print(paste0('ERROR! datatag = ', datatag, ' is not implemented.'))
    # TODO: return a default version
    myColors = get_cluster_colours(nClones)
    index = 1
    for (cc in myColors) {
      colour_dic[[seq(nClones)[index]]] <- cc
      index = index + 1
    }
  }
  colour_dic
}


plot_violin_for_exp_path <- function(exp_path, datatag='',  burn_in_fraction=.10, thinning=1) {
  param_trace_list <- get_param_trace(exp_path, burn_in_fraction = burn_in_fraction, thinning = thinning)
  param_trace = param_trace_list[[1]]
  param_trace_wide = param_trace_list[[2]]
  post_burn_in_iters = nrow(param_trace_wide)
  param_trace_list <- NULL
  temp <- param_trace
  k_vals <- unique(param_trace$K)
  nclust <- length(k_vals)
  clone_dic <- get_clone_dic_for_datatag(datatag = datatag, nClones = nclust+1)
  colour_dic <- get_colour_dic_for_datatag(datatag = datatag, nClones = nclust+1)
  
  for (kk in k_vals) {
    temp$K[param_trace$K == kk] <- clone_dic[[kk]]
  }
  param_trace <- temp
  
  # Add some MCMC measures
  ess = coda::effectiveSize(as.mcmc(param_trace_wide))
  rr = coda::rejectionRate(as.mcmc(param_trace_wide))
  subtitle =  sprintf('ESS = %s\nRej. rate = %.2f\nPost-burn_in-iters. = %d', paste0(format(ess, digits = 2),  collapse = ', '), rr[[1]], post_burn_in_iters)
  
  p <- ggplot(data = param_trace, aes(y=s, x=K)) +
    labs(subtitle=subtitle) + 
    geom_violin(alpha=.5, aes(fill=K, colour=K)) + 
    scale_fill_manual(values=colour_dic) + 
    scale_colour_manual(values=colour_dic) + 
    xlab('Clone ID') + 
    ylab('s') + 
    theme_light(base_size = 45)
    #theme(text = element_text(size=45), plot.title = element_text(size = 20), 
          #legend.text = element_text(face = 'bold'))
    
  
  out_dir = file.path(exp_path, 'plots')
  dir.create(out_dir, showWarnings = F)
  file_path = file.path(out_dir, 'violin_hist.pdf')
  height = 3000
  height=height
  width=1.0/.44 * height
  pdfCoef=220
  pdf(file = file_path, width=width/pdfCoef, height=height/pdfCoef)
  print(p)
  dev.off()
}

# datatag='SA609'; exp_path='/Users/sohrabsalehi/Desktop/sc_dlp_sa501/SA609_exp_Y0CQ6_201806-15-162738.261879'
# datatag='SA532'; exp_path='/Users/sohrabsalehi/Desktop/sc_dlp_sa532/exp_Y0CQ6_201806-18-121554.708200'

# plot_violin_for_exp_path(exp_path, datatag)
# sa609_dlp_predict(NULL)
# sa532_dlp_predict(NULL)

# plot_violin_for_exp_path(exp_path, 'SA532')

plot_violin_for_batch <- function(batch_path, datatag='') {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    is_err = tryCatch({
      plot_violin_for_exp_path(dir, datatag)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
}

# the step_4 version
# batch_path = '/shahlab/ssalehi/scratch/fitness/batch_runs/SA532NehCor_201807-06-174749.937430'
# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/SA609NehCor_201807-06-174756.623344"

# the step_1 version
# batch_path = '/shahlab/ssalehi/scratch/fitness/batch_runs/SA532Steps1_201807-09-12627.382603'
# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/SA609Steps1_201807-09-12703.016327"

# the step_10 version
# batch_path = '/shahlab/ssalehi/scratch/fitness/batch_runs/SA532Steps10_201807-09-12650.083477'
# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/SA609Steps10_201807-09-12719.831442"


# step_4, new data, July 15th

# batch_path = '/shahlab/ssalehi/scratch/fitness/batch_runs/SA532Steps10_201807-09-12650.083477'

# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/S532S4P0e_201807-15-145425.767218"
# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/S609S4P0e_201807-15-145425.851427"

# Prediction

# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/BigS609Aug10P1e_201808-18-133558.59956"
# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/BigS609Aug10P0e_201808-18-133556.442448"

batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/BigS532Aug10P0e_201808-18-13100.421520"

## THIS IS THE NEXTONE  #batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/BigS532Aug10P1e_201808-18-133555.221785"


# plot_violin_for_batch(batch_path, 'SA532')
# plot_violin_for_batch(batch_path, 'SA609')


# plot_cumulative_for_batch(batch_path, 'SA532')
# plot_cumulative_for_batch(batch_path, 'SA609')
# gz_all_for_batch(batch_path)

######## Plot trajectories
grant.expand.trace_old <- function(prediction_trace_path, filter_before=NULL) {
  # prediction_trace_path = file.path(exp_path, 'predict.tsv.gz')
  df <- read.table(prediction_trace_path, header=T, sep='\t', row.names=NULL)
  
  if (!is.null(filter_before)) {
    df <- df[df$time <= filter_before, ]
  }
  
  df$X.1 = NULL
  head(df)
  last_K = max(as.numeric(df$K)) + 1
  np = max(df$np) + 1
  #burn.in = .10 * np
  burn.in = .8 * np
  df = df[df$np > burn.in, ]
  
  # Add K'th type
  # Approximate using the other K-1 means
  xk <- ddply(df, .(time, np), summarise, X = 1-sum(X))
  xk$K <- last_K
  colnames(xk) = c('time', 'np', 'X', 'K')
  head(xk)
  dat <- dplyr::bind_rows(df, xk)
  
  # TOO Expensive
  # for (i in unique(df$np)){
  #   for (j in unique(df$time)) {
  #     temp = data.frame(time=j, K = last_K, X = 1-sum(df$X[df$time == j & df$np == i]), np=i)
  #     df = rbind(df, temp)    
  #   }
  # }
  
  
  dat
}

grant.expand.trace <- function(prediction_trace_path, filter_before=NULL) {
  # prediction_trace_path = file.path(exp_path, 'predict.tsv.gz')
  dat <- read.table(prediction_trace_path, stringsAsFactors = F, header = T, sep = '\t')
  ref_k <- lenu(dat$K)
  dat1 <- dat %>% dplyr::select(time, K, X, np) %>% dplyr::filter(np > .1*max(np)) %>% ungroup()
  dat2 <- dat1 %>% dplyr::group_by(time, np) %>% dplyr::summarise(K = ref_k, X = 1 - sum(X))
  dat3 <- dplyr::bind_rows(dat1, dat2) 
  dat4 <- dat3 %>% dplyr::arrange(np, time) %>% dplyr::group_by(np, time) %>% dplyr::arrange(K, .by_group=TRUE)
  dat4
}






grant.compute.HDI <- function(df, prob = 0.95) {
  x_trace = df
  
  ###################
  n = nrow(x_trace)
  #x_trace = x_trace[x_trace$X >= 0 & x_trace$X <= 1,]
  if (any(x_trace$X < 0 | x_trace$X > 1)) {
    print('Warning! NAs found in the trajectory!!')
    print(x_trace$X[x_trace$X < 0 | x_trace$X > 1])
  }
  
  x_trace$X[x_trace$X < 0 | x_trace$X > 1] = NA
  print('Removing illegal x values...')
  
  # cast by time to become nSamples by Time
  x_trace_wide = acast(x_trace, np~time+K, value.var = 'X', mean)
  x_trace_wide = na.omit(x_trace_wide)
  
  colnames(x_trace_wide) = paste0('X', colnames(x_trace_wide))
  x_trace_mcmc = mcmc(x_trace_wide)
  x_trace_mcmc_colmeans = colMeans(x_trace_mcmc)
  dat = as.data.frame(HPDinterval(x_trace_mcmc, prob))
  
  dat$x = unlist(unname((x_trace_mcmc_colmeans)))
  dat$time <- as.numeric(gsub('X(.*)_.*', '\\1', rownames(dat)))
  dat$K = as.numeric(gsub('X.*_(.*)', '\\1', rownames(dat)))
  #dat$trueX <- model$x$value
  rownames(dat) <- NULL
  dat
}


# 1. Compute the N-th trace
# 2. Compute the means and the HDIs
# 3. Plot this new one
grant.generate.cumulative.plot <- function(dat, out_path, colours=NULL, time_labels, v_line=NULL) {
  # dat = hdi_dat
  p <- ggplot(dat, aes(time, x, group = K)) + 
    #geom_ribbon(aes(ymin=lower, ymax=upper, fill='credible_interval'), alpha=0.5) +
    geom_ribbon(aes(ymin=lower, ymax=upper, fill=factor(K)), alpha=0.5) +
    geom_line(linetype = 2, aes(colour=factor(K))) + 
    #scale_fill_manual(name='', values=c('credible_interval'='darkgrey'), labels=c('Credible Interval')) +
    #scale_fill_brewer(palette="Set3") + 
    
    #scale_colour_manual(name='', values=c('ph'='red','sample_mean'='black'), labels=c('Prediction Horizon','Sample Mean')) +
    #scale_x_continuous(name="Timepoint", breaks=time_labels$breaks, labels = time_labels$labels) +
    scale_x_continuous(name="Timepoint") +
    ylim(c(-.2,1.2)) +
    ylab('Cellular prevalence') + 
    labs(fill='', colour='', shape='') +
    theme(plot.title = element_text(size = 25), legend.text = element_text(face = 'bold', size=25),
          axis.text.x = element_text(hjust=1, size = 25, face = 'plain'),
          axis.text.y = element_text(size = 25, face = 'plain'),
          axis.title.x = element_text(size = 30),
          axis.title.y = element_text(size = 30),
          axis.title=element_text(size=15,face="bold"))
  # theme(plot.title = element_text(size = 25), legend.text = element_text(face = 'bold', size=15),
  #       axis.text.x = element_text(hjust=1, size = 15, face = 'plain'),
  #       axis.text.y = element_text(size = 15, face = 'plain'),
  #       axis.title=element_text(size=15,face="bold"))
  
  
  if (!is.null(colours)) {
    p <- p + scale_fill_manual(values=colours) + scale_colour_manual(values=colours) 
  } 
  
  
  if (!is.null(v_line)) {
    p <-   p + geom_vline(aes(xintercept=v_line),  linetype = 2, color='red')
  }
  
  height = 1500
  height=height
  width=1.0/.44 * height
  pdfCoef=220
  #file_path = file.path('~/Google Drive/BCCRC/grant_sep/', paste0('inference_trace_3.pdf'))
  pdf(file = out_path, width=width/pdfCoef, height=height/pdfCoef)
  
  print(p)
  
  dev.off()
  
}


plot_cumulative_for_batch <- function(batch_path, datatag=NULL) {
  dirs <- list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    is_err = tryCatch({
      paper_dlp_cumulative_for_exp_path(dir, datatag)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
}


paper_dlp_cumulative_for_exp_path <- function(exp_path, datatag) {
  file_path <- file.path(exp_path, 'predict.tsv.gz')
  out_path <- file.path(exp_path,  '/plots/', paste0(tolower(datatag), '_dlp_expanded.pdf'))
  dat <- grant.expand.trace(prediction_trace_path = file_path)
  hdi_dat <- grant.compute.HDI(dat)
  # The colours = last number is the reference
  K <- length(unique(dat$K))
  dic <- get_clone_dic_for_datatag(datatag, K)
  #K <- length(dic) + 1
  
  reference <- paste0(setdiff(1:K, unname(unlist(dic))))
  hdi_dat$K <- factor(hdi_dat$K, levels = c(0:(K-1)), labels = c(unname(unlist(dic)) , reference))
  # Sort the clones
  #hdi_dat$K = factor(hdi_dat$K, levels = c('A', 'B', 'C', 'D', 'S1', '4'))
  hdi_dat$K = factor(hdi_dat$K, levels = paste0(sort(unique(hdi_dat$K))))
  colours <- get_colour_dic_for_datatag(datatag, TRUE, K)
  grant.generate.cumulative.plot(hdi_dat, out_path, colours = colours)
}


# For each batch, find out which experiments have a specific param set to some value
wf_list_param_for_batch <- function(batch_path, paramNameList, paramValList, monitorKeys=NULL) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  dat = NULL
  for (dir in dirs) {
    config = yaml.load_file(file.path(dir, 'config.yaml'))
    key_vals = list()
    for (paramName in c(paramNameList, monitorKeys)) {
      #if (is.null(config[[paramName]])) next
      key_vals[[paramName]] = config[[paramName]][1]
    }
    
    temp = data.frame(key_vals, stringsAsFactors = F)
    temp$dir = dir
  
    if (is.null(dat)) 
      dat = temp
    else
      dat = rbind(dat, temp)
  }
  
  
  # Filter
  i = 0
  filter_array = rep(TRUE, nrow(dat))
  for (paramName in paramNameList) {
    i = i + 1
    filter_array <- filter_array & (dat[[paramName]] == paramValList[[i]])
  }
  
  write.table(dat, file.path(batch_path, 'param_val_list.tsv'), sep='\t')
  write.table(dat[filter_array, ], file.path(batch_path, paste0('param_val_list_filtered.tsv')), sep='\t')
  
  dat[filter_array, ]
}

# wf_list_param_for_batch('/shahlab/ssalehi/scratch/fitness/batch_runs/SA609_small_201808-22-191727.263033', 'proposal_step_sigma', '0.05')


# 
# sa609_dlp_predict <- function(exp_path) {
#   exp_path='/Users/sohrabsalehi/Desktop/sc_dlp_sa501/SA609_exp_Y0CQ6_201806-15-162738.261879'
#   param_trace <- get_param_trace(exp_path)
#   #myColors <- c('orange', 'lightblue', 'lightgreen', 'darkblue', 'darkgreen')
#   
#   param_trace$K[param_trace$K == '1'] <- 'D'
#   param_trace$K[param_trace$K == '2'] <- '6'
#   param_trace$K[param_trace$K == '3'] <- 'F'
#   param_trace$K[param_trace$K == '4'] <- 'E'
#   param_trace$K[param_trace$K == '5'] <- 'C'
#   
#   #1 -> C
#   #2 -> B
#   #3 -> D
#   #4 -> F
#   #5 -> E 
#   #? -> A
#   
#   colourList = c('C'=myColors[1], '6'=myColors[6], 'D'=myColors[3], 'F'=myColors[4], 'E'=myColors[5])
#   #labelList = c('C', '6', 'D', 'F', 'E')
#   #labelList = c('C', 'D', 'F', 'E', '6')
#   
#   #param_trace$colours_1 <- myColors[as.numeric(param_trace$K)]
#   p <-  ggplot(data = param_trace, aes(y=s, x=K))
#   p <- p + geom_violin(alpha=.5, aes(fill=K, colour=K))
#   p <- p + scale_fill_manual(values=colourList) 
#   p <- p + scale_colour_manual(values=colourList) 
#   p + xlab('Clone ID') + 
#     ylab('s') + 
#     theme(text = element_text(size=45), plot.title = element_text(size = 20), 
#           #axis.text.x = element_text(angle=90, hjust=1),
#           legend.text = element_text(face = 'bold'))
# }
# 
# 
# sa532_dlp_predict <- function(exp_path) {
#   exp_path='/Users/sohrabsalehi/Desktop/sc_dlp_sa532/exp_Y0CQ6_201806-18-121554.708200'
#   param_trace <- get_param_trace(exp_path)
# 
#   param_trace$K[param_trace$K == '1'] <- 'A'
#   param_trace$K[param_trace$K == '2'] <- 'D'
#   param_trace$K[param_trace$K == '3'] <- 'C'
#   param_trace$K[param_trace$K == '4'] <- 'S1'
#   param_trace$K[param_trace$K == '5'] <- 'B'
#   
#   #1 -> S1
#   #2 -> A
#   #3 -> C
#   #4 -> ?
#   #5 -> D
#   #6 -> B
#   
#   colourList = c('S1'=myColors[1], 'A'=myColors[2], 'C'=myColors[3], 'D'=myColors[5], 'B'=myColors[6])
#   #labelList = c('C', 'D', 'F', 'E', '6')
#   
#   #param_trace$colours_1 <- myColors[as.numeric(param_trace$K)]
#   p <-  ggplot(data = param_trace, aes(y=s, x=K))
#   p <- p + geom_violin(alpha=.5, aes(fill=K, colour=K))
#   p <- p + scale_fill_manual(values=colourList) 
#   p <- p + scale_colour_manual(values=colourList) 
#   p + xlab('Clone ID') + 
#     ylab('s') + 
#     theme(text = element_text(size=45), plot.title = element_text(size = 20), 
#           #axis.text.x = element_text(angle=90, hjust=1),
#           legend.text = element_text(face = 'bold'))
# }
# 
# 
# 



