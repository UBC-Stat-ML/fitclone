library(ggplot2)
library(coda)
library(plyr); library(dplyr)
library(reshape2)
library(RColorBrewer)

source('time_series_utils.R')

ss_bi_create_x_plots <- function(bi_model, outpath) {
  # bi_model = dataSet.1
  # outpath <- '~/wf_3x.pdf'
  # for each dimensino K, (K starts from 0)
  # add (overlay) x[k]
  # overlay x[k+1] <- 1 - element_wise_sum(x[k])
  dat <- bi_read(bi_model, c('x'))$x
  K <- unique(dat$k)
  
  # find out what K is 
  xk <- ddply(dat, .(time), summarise, value = 1-sum(value))
  xk$k <- length(K)
  dat <- rbind(dat, xk)
  colnames(dat) <- c('k', 'time', 'x')
  # sanity check
  #ddply(dat, .(time), summarise, sum = sum(value))
  
  #ne = bi_read(bi_model, c('ne'))
  ne = 200
  s = 0.1
  #
  # add s value 
  dat$s <- '0'
  for (i in K) {
    #dat$s[dat$k == i] <- sprintf('%.1f', bi_read(bi_model, c(paste0('s', i+1)))[[paste0('s', i+1)]] )
    sDat <- bi_read(bi_model, 's')$s
    dat$s[dat$k == i] <- sprintf('%.1f',  sDat$value[sDat$k == i])
  }
  
  p <- ggplot(dat, aes(time, x, group = k, colour = s, size = s)) + 
    geom_line(linetype = 2) + 
    #scale_colour_manual(name = 'State', values = c('red', 'blue'), labels = c('Predicted', 'True(unseen)')) +
    scale_colour_manual(values = c("#9999CC", "#CC6666", "#66CC99")) + 
    scale_size_manual(values = 1+2*as.numeric(sort(unique(dat$s)))) +
    ggtitle(sprintf('Simulated prevalence paths\nNe = %d', ne)) +
    theme(plot.title = element_text(size = 20), legend.text = element_text(face = 'bold'))
  
  pdf(outpath)
  print(p)
  dev.off()
  p
}


quick_test_posterior_ordering <- function(exp_path) {
  file_path = file.path(exp_path, 'infer_theta.tsv.gz')
  ss = read.table(file_path, header=T)
  head(ss)
  ss <- ss[round(.1*nrow(ss)):nrow(ss), ]
  nrow(ss)
  mean(ss$X0 > ss$X5) 
  boxplot(ss)
}


wf.count.2.freq <- function(tallFormat, N=NULL) {
  t.index = 0
  for (t in unique(tallFormat$time)) {
    t.index <- t.index + 1
    if (!is.null(N[t.index])) {
      the_sum = N[t.index]
      # Sanity check
      if (sum(tallFormat$X[tallFormat$time == t]) > N[t.index]) {
        print('ERROR! The observed sum is greater than the provided sum!')
      }
    } else {
      the_sum = sum(tallFormat$X[tallFormat$time == t])
    }
    
    if (the_sum != 0)
      tallFormat$X[tallFormat$time == t] <- tallFormat$X[tallFormat$time == t]/the_sum
  }
  tallFormat
}




####### Trace plot
ss_bi_create_trace_plot_all <- function(x_trace, mcmc_options, outpath, vars) {
  traceCols <- c("#CC6666", "#66CC99", "#9999CC")
  
  pdf(outpath)
  traces <- get_traces(bi_object, all = T)
  p <- NULL
  xclump.list <- list()
  xclump.trueX <- list()
  for (var in vars) {
    # var = vars[[1]]
    #model <- bi_read(bi_model)
    #model$x$value
    varIndex <- gsub('x', '', var)
    traces.dat = as.data.frame(traces)
    # apply burn-in
    n = nrow(traces.dat)
    traces.dat <- traces.dat[(n/10):n, ]
    traces.dat = traces.dat[, grepl(paste0('x\\.', varIndex, '\\.*'), colnames(traces.dat))]
    
    if (var == 'xclump') {
      # generate xclump 
      traces.dat <- 1
      for (nn in names(xclump.list)) {
        traces.dat <- traces.dat - xclump.list[[nn]] 
      }
      varIndex <- gsub('x', '', vars[[1]])
      colnames(traces.dat) <- gsub(paste0('x\\.', varIndex), 'x\\.clump', colnames(xclump.list[[1]]))
      varIndex <- 'clump'
    } else {
      xclump.list[[varIndex]] <- traces.dat
    }
    
    traces.mcmc = mcmc(traces.dat)
    
    dat = as.data.frame(HPDinterval(mcmc(traces.mcmc)))
    dat$x = unlist(unname((colMeans(traces.mcmc))))
    dat$time <- rownames(dat)
    dat$time <- gsub(paste0('x\\.', varIndex, '_'), '', dat$time)
    
    if (var == 'xclump') {
      temp <- 1
      for (nn in names(xclump.trueX)) {
        temp <- temp - xclump.trueX[[nn]]
      }
      dat$trueX <- temp
    }
    else {
      dat$trueX <- bi_read(bi_model, c('x'))$x$value[bi_read(bi_model, c('x'))$x$k == varIndex]
      xclump.trueX[[varIndex]] <- dat$trueX
    }
    
    dat$group <- 1
    rownames(dat) <- NULL
    
    ## TODO: update this to support more colours than 3
    if (var == 'xclump')
      varIndex <- 2
    
    ne = 200
    s = bi_read(bi_model, c(paste0('s', as.numeric(varIndex)+1)))[[paste0('s', as.numeric(varIndex)+1)]]
    if (is.null(p)) {
      p <- ggplot(dat, aes(time, x, group = group)) + 
        geom_point(aes(time, x, color = 'red'), size = 3) + 
        geom_point(aes(time, trueX, color = 'blue'), size = 4) +
        scale_colour_manual(name = 'State', values = c('red', 'blue'), labels = c('Predicted', 'True(unseen)')) +
        geom_line(linetype = 2) + 
        geom_ribbon(aes(ymin=lower,ymax=upper), alpha=0.5, fill = traceCols[[as.numeric(varIndex)+1]])
    } else {
      p <- p + 
        geom_point(data = dat, aes(time, x, color = 'red'), size = 3) + 
        geom_point(data = dat, aes(time, trueX, color = 'blue'), size = 4) +
        scale_colour_manual(name = 'State', values = c('red', 'blue'), labels = c('Predicted', 'True(unseen)')) +
        geom_line(data = dat, linetype = 2) + 
        geom_ribbon(data = dat, aes(ymin=lower,ymax=upper), alpha=0.5, fill = traceCols[[as.numeric(varIndex)+1]]) +
        #ggtitle(sprintf('Posterior path with 95%% HPD credible interval\ns = %.2f, Ne = %d', s, ne)) + 
        ggtitle(sprintf('Posterior path with 95%% HPD credible interval\nNe = %d', ne)) + 
        theme(text = element_text(size=20), plot.title = element_text(size = 20), 
              axis.text.x = element_text(angle=90, hjust=1),
              legend.text = element_text(face = 'bold'))
    }
  }
  print(p)
  dev.off()
}



ts_utils_create_trace_plot_single <- function(x_trace, mcmc_options, index, plot_all = FALSE) {
  n = nrow(x_trace)
  if (any(x_trace$X < 0 | x_trace$X > 1)) {
    print('Warning! NAs found in the trajectory!!')
    print(x_trace$X[x_trace$X < 0 | x_trace$X > 1])
  }

  if (!plot_all) {
    x_trace$X[x_trace$X < 0 | x_trace$X > 1] = NA
    x_trace = x_trace[x_trace$K == index, ]
  }

  # cast by time to become nSamples by Time
  x_trace_wide = acast(x_trace, np~time, value.var = 'X', mean)
  x_trace_wide = na.omit(x_trace_wide)
  
  colnames(x_trace_wide) = paste0('X', colnames(x_trace_wide))
  x_trace_mcmc = mcmc(x_trace_wide)
  x_trace_mcmc_colmeans = colMeans(x_trace_mcmc)
  # HDI
  dat = as.data.frame(HPDinterval(x_trace_mcmc, prob=.95))
  dat$x = unlist(unname((x_trace_mcmc_colmeans)))
  dat$time <- rownames(dat)
  dat$time <- as.numeric(gsub('X', '', dat$time))
  dat$group = 1
  #dat$trueX <- model$x$value
  rownames(dat) <- NULL
  ne = mcmc_options$Ne
  s = mcmc_options$s[index+1]
  learn_time = mcmc_options$learn_time
  trueX.dat = mcmc_options$true_value$x
  trueX.dat$group = 2
  # Convert to freq if not in freq
  if (median(trueX.dat$X) > 1) {
    print(mcmc_options)
    #print(trueX.dat)
  }

  p <- ggplot(dat, aes(time, x, group = group)) + 
    geom_ribbon(aes(ymin=lower, ymax=upper, fill='credible_interval'), alpha=0.8) +
    geom_vline(linetype = 2, aes(xintercept = learn_time, colour='ph'), size=5) + # size = 5
    geom_line(linetype = 2, aes(colour='sample_mean'), size=5) + 
    geom_point(data = trueX.dat, mapping = aes(x = time, y = X, shape='observed_values'), size=5, color='red') + # size = 25
    scale_colour_manual(name='', values=c('ph'='red','sample_mean'='black'), labels=c('Prediction Horizon','Sample Mean')) +
    scale_fill_manual(name='', values=c('credible_interval'='darkgrey'), labels=c('Credible Interval')) +
    scale_shape_discrete(name='', labels=c('Observed Values', 'True Values')) +
    ylim(c(-.2,1.2)) +
    xlab('Time') +
    ylab(sprintf('x%d', index+1)) + 
    ggtitle(sprintf("Posterior path (index = %d) with 95%% HPD crediable interval\ntrue_s = %.2f, Ne = %d", index+1, s, ne)) + 
    labs(fill='', colour='', shape='') +
    theme_light(base_size = 40) +
    theme(plot.title = element_text(size=20), 
          legend.text = element_text(size=20, face='bold'),
          axis.text.x = element_text(hjust=1, size=20, face='plain'),
          axis.text.y = element_text(size=20, face='plain'),
          legend.position="bottom", legend.direction="horizontal",
          axis.title=element_text(size=20, face="bold"))

  p
}




####### Historgram
ts_utils_get_nice_range <- function(the_range) {
  the_len = abs(the_range[2] - the_range[1])
  return(c(the_range[1]-.1*the_len, the_len[2]+.1*the_len))
}

ts_utils_create_hist_plot = function(param_trace, mcmc_options, outpath, vars = NULL, plot_separate=F) {
  K = length(unique(param_trace$K))
  #print(param_trace)
  seed = mcmc_options$seed
  nMCMC = mcmc_options$inference_n_iter
  nParticles =  mcmc_options$pgas_n_particles
  
  #traceCols <- c("#CC6666", "#66CC99", "#9999CC")
  #YlOrRd Reds
  traceCols = colorRampPalette(brewer.pal(8, 'Set1'))(K+1)
  true_s = data.frame(s = unlist(mcmc_options$s), K = paste0(seq(K)))
  fill_colour <- 'skyblue'
  line_colour <- 'orange'

  if (plot_separate) {
    if (!dir.exists(dirname(outpath))) dir.create(dirname(outpath))
    pdf(outpath)
    the_range = range(param_trace$s)
    the_range = ts_utils_get_nice_range(the_range)
    print(the_range)
    for (k in unique(param_trace$K)) {
      temp_dat = param_trace[param_trace$K == k, ]
      estimated_s = mean(temp_dat$s)
      p = ggplot(data = temp_dat, aes(s), fill=K) + 
        geom_histogram(alpha = .9, aes(fill='estimated_s', y=..density..)) +
        geom_density() + 
        geom_vline(data=true_s[true_s$K == k, ,drop=F], aes(xintercept=s, color='true_s'),  linetype = 2) +
        ggtitle(sprintf('k = %s, Posterior distribution with 10%% burn-in \n#MCMC iter = %d, #particles = %d', k, nMCMC, nParticles)) + 
        labs(x=paste0('s', k)) + 
        scale_colour_manual(name = 'True s', values =c('true_s'=line_colour), labels = c(sprintf('%.3f',true_s$s[true_s$K == k]))) +
        scale_fill_manual(name = 'Estimated s', values =c('estimated_s'=fill_colour), labels = c(sprintf('%.3f',estimated_s))) +
        xlim(min(the_range[1], -1), the_range[2]) +
        theme_light(base_size = 25) + 
        theme(plot.title = element_text(size = 20),
              text = element_text(size=20),
              axis.text.x = element_text(angle=90, hjust=1),
              legend.text = element_text(face = 'bold'))
      print(p)
    }
  } else {
    pdf(outpath)
    p1 = ggplot(data = param_trace, aes(s, colour=K, fill=K)) +
      geom_histogram(alpha = .2, bins = 60) +
      facet_wrap(~K) + 
      geom_vline(data=true_s, aes(xintercept=s, group = K, color = K),  linetype = 2) +
      ggtitle(sprintf('Posterior distribution with 10%% burn-in \n#MCMC iter = %d, #particles = %d', nMCMC, nParticles)) +
      theme_light(base_size = 25) +
      theme(plot.title = element_text(size = 20),
            text = element_text(size=20),
            axis.text.x = element_text(angle=90, hjust=1),
            legend.text = element_text(face = 'bold'))
    print(p1)
    p = ggplot(data = param_trace, aes(s, group = K, colour=K, fill=K)) +
      geom_density(alpha = .2) +
      geom_vline(data=true_s, aes(xintercept=s, group = K, color = K),  linetype = 2) +
      ggtitle(sprintf('Posterior distribution with 10%% burn-in \n#MCMC iter = %d, #particles = %d', nMCMC, nParticles)) +
      theme_light(base_size = 25) +
      theme(plot.title = element_text(size = 20),
            text = element_text(size=20),
            axis.text.x = element_text(angle=90, hjust=1),
            legend.text = element_text(face = 'bold'))
    print(p)
    
  }
  dev.off()
}


ts_utils_test_hists <- function() {
  param_path = "~/Google Drive/BCCRC/wright_fisher_experiments/sample_outputs_201706-15-162441/infer_theta.tsv"
  param_dat = read.table(param_path, sep='\t')
  colnames(param_dat) = c('s1', 's2')
  mcmc_options = list(seed=11, nMCMC=10, nParticles=5, true_value=list('s1'=.2, 's2'=.3))
  out_dir = '~/Google Drive/BCCRC/wright_fisher_experiments/sample_outputs_201706-15-162441'
  ts_utils_create_hist_plot(param_trace = param_dat, mcmc_options=mcmc_options, outpath = file.path(out_dir, '/theta_hist.pdf'), vars = c('s1','s2'))
}


ts_utils_create_countour_plot <- function(params_dat) {
  commonTheme = list(labs(color="Density",fill="Density",
                          s1="s1",
                          s2="s2"),
                     theme_bw(),
                     theme(legend.position=c(1,0),
                           legend.justification=c(0,1)))
  
    #df = data.frame(params_dat); colnames(df) = c('x', 'y')
    
    # ggplot(data=df,aes(x,y)) + 
    # geom_density2d(aes(colour=..level..)) + 
    # scale_colour_gradient(low="green",high="red") + 
    # geom_point() + commonTheme
    
    
    ggplot(data=params_dat,aes(s1,s2)) + 
      stat_density2d(aes(fill=..level..,alpha=..level..),geom='polygon',colour='black') + 
      scale_fill_continuous(low="green",high="red") +
      geom_smooth(method=lm,linetype=2,colour="red",se=F) + 
      guides(alpha="none") +
      geom_point() + commonTheme
}




preprocess_dat <- function(dat) {
  return(dat)
  # remove time
  dat = dat[-c(which(dat$time == 'time')), ] 
  #dat = na.omit(dat)
  dat$X = as.numeric(dat$X)
  dat$time = as.numeric(dat$time)
  dat$K = as.numeric(dat$K)
  if (!is.null(dat$np)){
    dat$np = as.numeric(dat$np)
  }
  dat$X.1 = NULL
  dat$time = as.numeric(sprintf('%.3f', dat$time))
  dat
}



plot_true_trace_for_exp_path <- function(exp_path) {
  plot_with_data_path <- function(data_path, i='') {
    dat = read.table(data_path, header=T)
    if (length(s) == length(unique(dat$K)))
      dat$K = sprintf('s%d = %.3f', (dat$K+1), s[dat$K+1])
    p <- ggplot(dat, aes(time, X)) + geom_line(aes(colour=factor(K), group=K), size=2) + 
      theme_light(base_size = 45) +
      theme(legend.position="bottom", legend.direction="horizontal")
    
    file_path = file.path(exp_path, 'plots', paste0('full_original_data_trace_', i, '.pdf'))
    height = 3000
    height=height
    width=1.0/.44 * height
    pdfCoef=220
    pdf(file = file_path, width=width/pdfCoef, height=height/pdfCoef)
    print(p)
    dev.off()
  }
  
  # exp_path = '~/Desktop/pgas_sanity_check/exp_nov21_k10_fixed_p100k_C14AN_201712-14-151157.915699/'
  mcmc_options = ts_utils_read_config(exp_path)
  s = mcmc_options$s
  
  if (!is.null(mcmc_options$M_data)) {
    for (i in seq(mcmc_options$M_data)) {
      data_path <<- file.path(exp_path, paste0('sample_data_full_', i-1, '.tsv.gz'))
      if (!file.exists(data_path)) {
        data_path <<- mcmc_options$full_original_data[i]
        if (is.null(data_path))
          data_path <<- mcmc_options$original_data[i]
        if (!file.exists(data_path)) {
          data_path <<- paste0(data_path, '.gz')
        }
      }
      
      plot_with_data_path(data_path=data_path, i=i-1)
    }
  } 
  else {
    
    data_path <- file.path(exp_path, 'sample_data_full.tsv.gz')
    
    if (!file.exists(data_path)) {
      data_path <- mcmc_options$full_original_data
    }
    
    if (is.null(data_path)){
      data_path <- mcmc_options$original_data
    }
        
    if (!file.exists(data_path)) {
        data_path <- paste0(data_path, '.gz')
    }

    
    plot_with_data_path(data_path=data_path)
  }
  

}


plot_mcmc_traces_for_exp_path <- function(exp_path) {
  param_path = file.path(exp_path, 'infer_theta.tsv.gz')
  if (file.exists(param_path)) {
    param_dat = read.table(param_path, sep='\t', header=TRUE)
    K = ncol(param_dat)
    colnames(param_dat) = paste0('s', seq(K))
    n = nrow(param_dat)
    param_dat = param_dat[seq(burn_in*n, n, thinning), , drop=F]
    param_dat_tall = melt(param_dat, value.name='s', varname=c('k'))
    param_dat_tall$K = gsub('s', '', param_dat_tall$variable)
    param_dat_tall$variable = NULL
    
    s_mcmc = mcmc(param_dat)
    plot(s_mcmc)
  }
}

plot_all_for_exp_path <- function(exp_path, burn_in_fraction=.10, thinning=1, ignore_inference=FALSE, ignore_hist=FALSE) {
  print(exp_path)
  burn_in = burn_in_fraction
  mcmc_options = ts_utils_read_config(exp_path) #mcmc_options = list(seed=11, nMCMC=10, nParticles=5, s=(seq(K)/(K+1)))
  if (!is.null(mcmc_options$K_prime)) {
    mcmc_options$block_size = mcmc_options$K
    mcmc_options$K = mcmc_options$K_prime
  }
  
  if (is.null(mcmc_options)) {
    print('WAS NULL')
    mcmc_options = list(seed=11, nMCMC=10, nParticles=-1, s=(seq(K)/(K+1)))
    print('Warning! Using dummy config file')
  }
  
  out_dir = file.path(exp_path, 'plots')
  dir.create(out_dir, showWarnings = F)
  
  # plot true data
  plot_true_trace_for_exp_path(exp_path)
  
  # plot theta
  param_path = file.path(exp_path, 'infer_theta.tsv.gz')
  if (file.exists(param_path) & !ignore_hist) {
    param_dat = read.table(param_path, sep='\t', header=TRUE)
    K = ncol(param_dat)
    colnames(param_dat) = paste0('s', seq(K))
    n = nrow(param_dat)
    param_dat = param_dat[seq(burn_in*n, n, thinning), , drop=F]
    param_dat_tall = melt(param_dat, value.name='s', varname=c('k'))
    param_dat_tall$K = gsub('s', '', param_dat_tall$variable)
    param_dat_tall$variable = NULL
    
    ts_utils_create_hist_plot(param_trace = param_dat_tall, mcmc_options=mcmc_options, outpath=file.path(out_dir, 'theta_hist_single.pdf'), vars = seq(K))
    
    ## Plot individual histograms
    ts_utils_create_hist_plot(param_trace = param_dat_tall, mcmc_options=mcmc_options, outpath = file.path(out_dir, 'theta_hists.pdf'), vars = seq(K), plot_separate=T)
  } else {
    print("Warning! No theta infernce file found.")
  }
  
  inner_plot_traces <- function(the_suffix) {
    height = 3000
    height=height
    width=1.0/.44 * height
    pdfCoef=220
    for (exp_kind in c('predict', 'infer_x')) {
      if (ignore_inference & exp_kind == 'infer_x') {
        next
      }
    
      param_path = file.path(exp_path, paste0(exp_kind, the_suffix, ".tsv.gz"))
      if (!file.exists(param_path)) next
      file_path = file.path(out_dir, paste0(exp_kind, the_suffix, '_trace', '.pdf'))
      pdf(file = file_path, width=width/pdfCoef, height=height/pdfCoef)
      K = mcmc_options$K
      param_dat = read.table(param_path, sep='\t', header = T, stringsAsFactors = F)
      # burn-in
      np_set = unique(param_dat$np)
      nIter = length(np_set)
      param_dat = param_dat[param_dat$np >= nIter*burn_in,]
      # thinning
      param_dat = param_dat[param_dat$np %in% np_set[seq(1, length(np_set), thinning)], ]
      
      for (index in (seq(K)-1) ) {
        data_path <- file.path(exp_path, paste0('sample_data', the_suffix, '.tsv.gz'))
        if (!file.exists(data_path)) {
          data_path = mcmc_options$original_data
        }
        if (!file.exists(data_path)) {
          data_path = paste0(data_path, '.gz')
        }
        
        dat <- read.table(data_path, sep='\t', header = T, stringsAsFactors = F)
        dat$X.1 <- NULL
        # Convert counts to freq if the input is in counts
        is_mults = FALSE
        if (!is.null(mcmc_options$multinomial_error)) is_mults = mcmc_options$multinomial_error
          
        if (!is.null(mcmc_options$observation_model) | is.null(mcmc_options$multinomial_error)) {
          #if (mcmc_options$observation_model %in% c('dir_mult', 'mult') ) {
          dat = wf.count.2.freq(dat, N=mcmc_options$Y_sum_total)
          #}
        }
        #dat = preprocess_dat(dat)
        dat <- dat[dat$K == index, ]
        if (is.null(mcmc_options$learn_time)) mcmc_options$learn_time = .09
        mcmc_options$true_value$x <- dat
        
        # the predicted data
        param_dat_k = param_dat[param_dat$K == index,]
        
        p = ts_utils_create_trace_plot_single(x_trace = param_dat_k, mcmc_options=mcmc_options, index = index)
        print(p)
      }
      dev.off()
    }
  }
  
  
  # plot traces
  if (is.null(mcmc_options$M_data)) {
    mcmc_options$M_data = 1
  }

  for (ii in seq(mcmc_options$M_data)) {
    if (mcmc_options$M_data == 1)
      i_suffix = ''
    else 
      i_suffix = paste0('_', (ii-1))
    
    inner_plot_traces(the_suffix=i_suffix)
  }
  
  
}

