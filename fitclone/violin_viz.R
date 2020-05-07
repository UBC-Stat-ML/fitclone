library(RColorBrewer)
library(dplyr)
library(coda)

source('time_series_plotting.R')


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


plot_violin_for_exp_path <- function(exp_path,  burn_in_fraction=.10, thinning=1) {
  param_trace_list <- get_param_trace(exp_path, burn_in_fraction = burn_in_fraction, thinning = thinning)
  param_trace = param_trace_list[[1]]
  param_trace_wide = param_trace_list[[2]]
  post_burn_in_iters = nrow(param_trace_wide)
  param_trace_list <- NULL
  temp <- param_trace
  k_vals <- unique(param_trace$K)
  nclust <- length(k_vals)
  clone_dic <- get_clone_dic_for_datatag(nClones = nclust+1)
  colour_dic <- get_colour_dic_for_datatag(nClones = nclust+1)
  
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


plot_violin_for_batch <- function(batch_path, datatag='') {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    is_err = tryCatch({
      plot_violin_for_exp_path(dir)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
}


get_clone_dic_for_datatag <- function(nClones=NULL) {
  dic <- NULL
  dic <- list()
  index = 1
  for (ll in seq(nClones)) {
    dic[[paste0(index)]] <- ll
    index = index + 1
  }

  dic
}

get_colour_dic_for_datatag <- function(full=FALSE, nClones=NULL) {
  colour_dic <- NULL
  myColors = get_cluster_colours(nClones)
  index = 1
  for (cc in myColors) {
    colour_dic[[seq(nClones)[index]]] <- cc
    index = index + 1
  }
  colour_dic
}
















