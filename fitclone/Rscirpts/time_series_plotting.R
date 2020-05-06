library(ggplot2)
library(coda)
library(plyr); library(dplyr)
library(reshape2)
library(RColorBrewer)

#setwd('/Users/sohrab/Google Drive/Masters/Thesis/scripts/fitness')
#source('~/projects/fitness/time_series_utils.R')
# set environment
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
  # exp_path = '/Users/sohrabsalehi/Desktop/SC-1311/SA609/batch_runs/SA609_long_test_counts_201810-22-192343.288569/o_1_0_2HMV9_201810-22-221746.960947/'
  # exp_path = '/Users/sohrabsalehi/projects/fitness/batch_runs/SA609_test_counts_201810-22-22533.849597/outputs/o_0_0_9V7NA_201810-22-22540.709293'
  # exp_path = '/Users/sohrabsalehi/projects/fitness/batch_runs/SA609_test_counts_201810-22-22533.849597/outputs/o_0_0_9V7NA_201810-22-22540.709293'
  # scp 
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
  # apply burn-in
  # x_trace = param_dat
  n = nrow(x_trace)
  #x_trace = x_trace[x_trace$X >= 0 & x_trace$X <= 1,]
  if (any(x_trace$X < 0 | x_trace$X > 1)) {
    print('Warning! NAs found in the trajectory!!')
    print(x_trace$X[x_trace$X < 0 | x_trace$X > 1])
  }

  if (!plot_all) {
    x_trace$X[x_trace$X < 0 | x_trace$X > 1] = NA
    x_trace = x_trace[x_trace$K == index, ]
  }

  
  #x_trace <- tail(x_trace[in (n/10):n, ])
  
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
  #trueX.dat$x = trueX.dat$X;
  #trueX.dat$x = trueX.dat$X_true;

  #print('Using True X not the Observed Value...')
  # Good reference for scales: http://www.hafro.is/~einarhj/education/ggplot2/scales.html
  p <- ggplot(dat, aes(time, x, group = group)) + 
    geom_ribbon(aes(ymin=lower, ymax=upper, fill='credible_interval'), alpha=0.8) +
    geom_vline(linetype = 2, aes(xintercept = learn_time, colour='ph'), size=5) + # size = 5
    geom_line(linetype = 2, aes(colour='sample_mean'), size=5) + 
    geom_point(data = trueX.dat, mapping = aes(x = time, y = X, shape='observed_values'), size=5, color='red') + # size = 25
    #geom_point(data = trueX.dat, mapping = aes(x = time, y = X_true, shape='true_values'), size=3, color='blue') +
    scale_colour_manual(name='', values=c('ph'='red','sample_mean'='black'), labels=c('Prediction Horizon','Sample Mean')) +
    scale_fill_manual(name='', values=c('credible_interval'='darkgrey'), labels=c('Credible Interval')) +
    scale_shape_discrete(name='', labels=c('Observed Values', 'True Values')) +
    ylim(c(-.2,1.2)) +
    xlab('Time') +
    ylab(sprintf('x%d', index+1)) + 
    ggtitle(sprintf("Posterior path (index = %d) with 95%% HPD crediable interval\ntrue_s = %.2f, Ne = %d", index+1, s, ne)) + 
    labs(fill='', colour='', shape='') +
    #xlab=('Time', text = element_text(size=20)) + 
    theme_light(base_size = 40) +
    theme(plot.title = element_text(size=20), 
          legend.text = element_text(size=20, face='bold'),
          axis.text.x = element_text(hjust=1, size=20, face='plain'),
          axis.text.y = element_text(size=20, face='plain'),
          legend.position="bottom", legend.direction="horizontal",
          axis.title=element_text(size=20, face="bold"))
  
  # Add the real X on top, since Dir_Mult could be so 
  #if (!is.null(mcmc_options$observation_model)) {
  #  p = p + geom_point(data = trueX.dat, mapping = aes(x = time, y = X_true, shape='true_values'), size=3, color='blue')
  #}
  
  # theme(plot.title = element_text(size=65), 
  #       legend.text = element_text(size=65, face='bold'),
  #       axis.text.x = element_text(hjust=1, size=65, face='plain'),
  #       axis.text.y = element_text(size=65, face='plain'),
  #       legend.position="bottom", legend.direction="horizontal",
  #       axis.title=element_text(size=65, face="bold"))
  # 
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


plot_all_for_batch <- function(batch_path, ignore_hist=F, burn_in=.1) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    is_err = tryCatch({
      plot_all_for_exp_path(dir, ignore_inference = T, ignore_hist = ignore_hist, burn_in_fraction = burn_in)
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
}

# batch_path = '/shahlab/ssalehi/scratch/fitness/batch_runs/SA532NehCor_201807-06-174749.937430'
# batch_path = "/shahlab/ssalehi/scratch/fitness/batch_runs/SA609NehCor_201807-06-174756.623344"
# plot_all_for_batch(batch_path)
# plot_all_for_exp_path(exp_path)

gz_all_for_batch <- function(batch_path) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  #out_path = file.path(batch_path, 'outputs', 'plots')
  out_path = file.path(batch_path, 'plots')
  dir.create(out_path, recursive = T, showWarnings=F)
  for (dir in dirs) {
    is_err = tryCatch({
      gz_file <- file.path(dir, paste0(basename(dir), '.tar.gz'))
      gz_in <- file.path(dir, 'plots')
      # Copy config into the plots dir
      system(paste0('cp  ', file.path(dir, 'config.yaml'), ' ', gz_in))
      system(paste0('tar -czvf ', gz_file,   ' -C ', gz_in, ' .'))
      system(paste0('mv  ', gz_file, ' ', out_path, '/'))
    }, error=function(err) {
      print(sprintf('Error - %s', err))
      errMsg <- err
    })
    
    if(inherits(is_err, "error")) next
  }
  
  #gz_out = file.path(batch_path, 'outputs', paste0('plots_all.tar.gz'))
  gz_out <- file.path(batch_path, paste0('plots_all.tar.gz'))
  system(paste0('tar -czvf ', gz_out,   ' -C ', out_path, ' .'))
  
  # Print a help command
  print(sprintf('scp shahlab15:%s .', gz_out))
}
# gz_all_for_batch(batch_path)


gz_all_for_exp_path <- function(exp_path) {
  is_err = tryCatch({
    gz_file <- file.path(dirname(exp_path), paste0(basename(exp_path), '.tar.gz'))
    gz_in <- file.path(exp_path, 'plots')
    
    # Copy config into the plots dir
    system(paste0('cp  ', file.path(exp_path, 'config.yaml'), ' ', gz_in))
    system(paste0('tar -czvf ', gz_file,   ' -C ', gz_in, ' .'))
    #system(paste0('mv  ', gz_file, ' ', out_path, '/'))
  }, error=function(err) {
    print(sprintf('Error - %s', err))
    errMsg <- err
  })
    
  if(inherits(is_err, "error")) {}
  
  # Print a help command
  print(sprintf('scp shahlab15:%s .', gz_file))
}



list_finished_exp_for_batch <- function(batch_path) {
  dirs = list.dirs(file.path(batch_path, 'outputs'), full.names = T, recursive = F)
  for (dir in dirs) {
    config_file_path = file.path(dir, 'config.yaml')
    if (file.exists(config_file_path)) {
      print(config_file_path)
      config = ts_utils_read_config(dir)
      print(paste0(config$pf_n_particles, ', ', config$pgas_n_particles))
    }
  }
}


#plot_all_for_batch(batch_path)

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

# exp_path = '/Users/sohrabsalehi/projects/fitness/batch_runs/test_fast_201810-18-23226.403744/outputs/o_0_0_91H9D_201810-19-105300.701898'
# exp_path = '/Users/sohrabsalehi/projects/fitness/batch_runs/test_fast_201810-18-23226.403744/outputs/o_0_0_BG0B5_201810-18-23335.068177'
# exp_path = '/Users/sohrabsalehi/projects/fitness/batch_runs/SA609_test_counts_201810-22-22533.849597/outputs/o_0_0_X3CKP_201810-23-113150.675203/'

# plot_all_for_exp_path(exp_path=exp_path)
plot_all_for_exp_path <- function(exp_path, burn_in_fraction=.10, thinning=1, ignore_inference=FALSE, ignore_hist=FALSE) {
  print(exp_path)
  burn_in = burn_in_fraction
  #burn_in = .10; thinning=1
  #exp_path = '/Users/ssalehi/Desktop/pgas_sanity_check/exp_IEQH5_201706-28-14014.234892'
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
      
      #exp_kind = 'infer_x'
      #exp_kind = 'predict'
      # the_suffix = ''
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
        #index = 0
        #exp_kind = 'infer_x'
        # load original data
        
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
          
        if (!is.null(mcmc_options$observation_model) | mcmc_options$multinomial_error) {
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
        
        #x_trace = param_dat; mcmc_options=mcmc_options; outpath = file_name; index = index
        #p = ts_utils_create_trace_plot_single(x_trace = param_dat_k, mcmc_options=mcmc_options, index = index, plot_all = TRUE)
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

# plot_all_for_exp_path(exp_path=exp_path)
# plot_all_for_exp_path(exp_path="/Users/sohrabsalehi/projects/fitness/batch_runs/S532July1P0e_201808-05-115822.941972/outputs/o_0_0_DRFWD_201808-05-115844.950033")
# plot_all_for_exp_path(exp_path="/Users/sohrabsalehi/Desktop/pgas_sanity_check/exp_nov21_k10_fixed_p100k_C14AN_201712-14-185538.155661/")

#plot_all_for_exp_path("/global/scratch/sohrab/fitness/batch_runs/TestSmallIters_201706-20-015916//outputs/output_201706-20-02012")
#plot_all_for_batch(batch_path="/global/scratch/sohrab/fitness/batch_runs/TestSmallIters_201706-20-015916/")
#plot_all_for_exp_path(exp_path="/Users/ssalehi/Desktop/pgas_sanity_check/exp_24YNG_201706-23-163320.669113/")
#plot_all_for_exp_path(exp_path="~/Desktop/pgas_sanity_check/exp_5RXP4_201708-05-143754.759926")
#
#plot_all_for_exp_path(exp_path="~/Desktop/xxx/exp_4R7WJ_201708-03-163443.721354", .2)


# read libbi output and prepare analysis for PGAS
prepare_pathological_case <- function() {
  for (file_name in c('seed_20.tsv', 'seed_30.tsv')) {
    base_path = "~/Google\ Drive/BCCRC/wright_fisher_experiments/pathological_cases/" 
    the_path = file.path(base_path, file_name)
    dat = read.table(the_path, header=T, row.names = 1)
    colnames(dat) = c('K', 'time', 'X')
    
    out_path = file.path(base_path, paste0('processed_', file_name))
    write.table(dat, out_path, row.names = F, sep='\t')
  }
}

#plot_all_for_batch(batch_path ='/Users/ssalehi/Desktop/pgas_sanity_check/K2500short_201707-03-134111.238639')


# lambdaVal = 500
# exp(-lambdaVal*abs(.01))
# 
# 
# 

temp_llhood <- function() {
  #exp_path='/Users/ssalehi/Desktop/pgas_sanity_check/exp_5RXP4_201707-19-182301.898848/'
  exp_path='/Users/ssalehi/Desktop/pgas_sanity_check/exp_5RXP4_201707-19-195022.972502/'
  
  llhood = read.table(file.path(exp_path, 'llhood_theta.tsv.gz'), header=T)
  theta = read.table(file.path(exp_path, 'infer_theta.tsv.gz'), header = T)
  
  
  #llhood = read.table('~/Desktop/llhoods_old.tsv.gz', header=T, sep='\t')
  llhood = read.table('~/Desktop/fine_llhoods_old.tsv.gz', header=T, sep='\t')
  colnames(llhood) = c('index', 's1', 's2', 'llhood')
  ss = llhood
  ss$exp = exp(ss$llhood)
  
  mat = matrix(data=ss$llhood,nrow = 51, ncol = 51, byrow = T)
  library(lattice)
  levelplot(mat )

  image(seq(0.0, .5, length.out = 51), seq(0.0, .5, length.out = 51), mat)
  
  
  
  v <- ggplot(ss, aes(s1, s2, colour=llhood)) + 
    geom_point(size=10, shape=15) +  scale_colour_gradient2(low = "black", high = 'red', midpoint = 87900, mid = 'grey') +
    theme(panel.background = element_rect(fill = 'black'))
  v
  
  
  
  nrow(llhood) == nrow(theta)
  ss = cbind(llhood, theta)
  colnames(ss)= c('llhood', 's1', 's2')
  ss$exp = exp(ss$llhood)
  tol = .1
  ss[abs(ss$s1-.1) < tol & abs(ss$s2-.3)< tol, ]
  ss[sample(nrow(ss), 10), ]
  hist(ss$exp)
  
  #ts_utils_create_countour_plot(params_dat = ss[200:500,])
  ts_utils_create_countour_plot(params_dat = ss)
  
  library(ggplot2)
  tt = faithful
  tt$z = runif(nrow(tt))
  tt$x_y = paste0(tt$waiting, tt$eruptions)
  
  ss$flash = ss$llhood - min(ss$llhood)
  
  
  
  
  
  
  v <- ggplot(ss, aes(s1, s2, z=exp))
  v + geom_point(aes(colour=exp))
  
  # tt = ss
  # tt$x_y = paste0(tt$s1, tt$s2)
  # tt = tt[!duplicated(tt$x_y), ]
  # 
  # v <- ggplot(tt, aes(s1, s2, z=llhood))
  # v + geom_contour()
}


plot_specific_mcmc_iter_for_exp_path <- function(exp_path, mcmc_iter=1, ignore_inference=F) {
  #exp_path = '/Users/sohrab/Desktop/pgas_sanity_check/exp_Y0CQ6_201708-10-222416.947661'
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
  }
  
  out_dir = file.path(exp_path, 'plots')
  dir.create(out_dir, showWarnings = F)
  
  # plot traces
  height = 3000
  height=height
  width=1.0/.44 * height
  pdfCoef=220
  for (exp_kind in c('predict', 'infer_x')) {
    if (ignore_inference & exp_kind == 'infer_x') {
      next
    }
    #for (exp_kind in c('infer_x')) {
    #exp_kind = 'infer_x'
    #exp_kind = 'predict'
    param_path = file.path(exp_path, paste0(exp_kind, ".tsv.gz"))
    if (!file.exists(param_path)) next
    file_path = file.path(out_dir, paste0(exp_kind, '_trace_single_iter', '.pdf'))
    pdf(file = file_path, width=width/pdfCoef, height=height/pdfCoef)
    K = mcmc_options$K
    
    param_dat = read.table(param_path, sep='\t', header = T, stringsAsFactors = F)
    for (index in (seq(K)-1) ) {
      #index = 0
      #exp_kind = 'infer_x'
      # load original data
      data_path <- file.path(exp_path, 'sample_data.tsv.gz')
      dat <- read.table(data_path, sep='\t', header = T, stringsAsFactors = F)
      dat$X.1 <- NULL
      dat <- dat[dat$K == index, ]
      if (is.null(mcmc_options$learn_time)) mcmc_options$learn_time = .09
      mcmc_options$true_value$x <- dat
      param_dat_k = param_dat[param_dat$K == index & param_dat$np == mcmc_iter,]
      p = ts_utils_create_trace_plot_single_iteration(x_trace=param_dat_k, mcmc_options=mcmc_options, index = index)
      print(p)
    }
    dev.off()
  }
}

ts_utils_create_trace_plot_single_iteration <- function(x_trace, mcmc_options, index) {
  
  n = nrow(x_trace)
  if (any(x_trace$X < 0 | x_trace$X > 1)) {
    print('Warning! NAs found in the trajectory!!')
    print(x_trace$X[x_trace$X < 0 | x_trace$X > 1])
  }
  
  x_trace$X[x_trace$X < 0 | x_trace$X > 1] = NA
  x_trace = x_trace[x_trace$K == index, ]
  
  # # cast by time to become nSamples by Time
  # x_trace_wide = acast(x_trace, np~time, value.var = 'X', mean)
  # x_trace_wide = na.omit(x_trace_wide)
  # 
  # colnames(x_trace_wide) = paste0('X', colnames(x_trace_wide))
  # dat = as.data.frame(x_trace_wide)
  # dat$x = unlist(unname((x_trace_wide)))
  # dat$time <- rownames(dat)
  # dat$time <- as.numeric(gsub('X', '', dat$time))
  # dat$group = 1
  # rownames(dat) <- NULL
  ne = mcmc_options$Ne
  s = mcmc_options$s[index+1]
  learn_time = mcmc_options$learn_time
  trueX.dat = mcmc_options$true_value$x
  trueX.dat$group = 2

  # Good reference for scales: http://www.hafro.is/~einarhj/education/ggplot2/scales.html
  p <- ggplot(x_trace, aes(time, X)) + 
    geom_vline(linetype = 2, aes(xintercept = learn_time, colour='ph')) +
    geom_line(linetype = 2, aes(colour='X')) + 
    geom_point(data = trueX.dat, mapping = aes(x = time, y = X, shape='observed_values'), size=5, color='red') +
    geom_point(data = trueX.dat, mapping = aes(x = time, y = X_true, shape='true_values'), size=3, color='blue') +
    scale_colour_manual(name='', values=c('ph'='red','X'='black'), labels=c('Prediction Horizon','Sample Mean')) +
    scale_shape_discrete(name='', labels=c('Observed Values', 'True Values')) +
    ylim(c(-.2,1.2)) +
    xlab('Time') +
    ylab(sprintf('x%d', index+1)) + 
    ggtitle(sprintf("Posterior path (index = %d) with 95%% HPD crediable interval\ntrue_s = %.2f, Ne = %d", index+1, s, ne)) + 
    labs(fill='', colour='', shape='') +
    theme(plot.title = element_text(size = 25), legend.text = element_text(face = 'bold', size=15),
          axis.text.x = element_text(angle=90, hjust=1, size = 15, face = 'plain'),
          axis.text.y = element_text(size = 15, face = 'plain'),
          axis.title=element_text(size=15,face="bold"))

  p
}

#exp_path='~/Desktop/pgas_sanity_check/exp_Y0CQ6_201708-11-135233.849283'
#plot_specific_mcmc_iter_for_exp_path(exp_path)
#plot_all_for_exp_path('/Users/ssalehi/Desktop/pgas_sanity_check/exp_Y0CQ6_201709-07-165123.394758/', burn_in_fraction = .2)
# exp_path = '/Users/sohrabsalehi/Google\ Drive/BCCRC/grant_sep/o_416_0_05PZ3_201708-10-022256.078076'
#plot_specific_mcmc_iter_for_exp_path(exp_path = exp_path, mcmc_iter = 4000)
#xyplot(X~time, param_dat[param_dat$K == 4 & param_dat$np %in% seq(1000,1010) ,], type='b', group=np)
#xyplot(X~time|K, param_dat[param_dat$np %in% seq(1000,1010, by=4) ,], type='b', group=np)


#exp_path='~/Desktop/pgas_sanity_check/exp_Y0CQ6_201711-15-16430.637758'
#plot_all_for_exp_path(exp_path=exp_path, burn_in_fraction = .1, thinning = 1, ignore_inference=FALSE, ignore_hist=FALSE)

#exp_path='~/Desktop/pgas_sanity_check/exp_Y0CQ6_201712-15-142843.393637'
#plot_all_for_exp_path(exp_path=exp_path, burn_in_fraction = .1, thinning = 1, ignore_inference=FALSE, ignore_hist=FALSE)


#exp_path='/Users/ssalehi/Desktop/pgas_sanity_check/exp_Y0CQ6_201711-15-133916.865830'
#plot_all_for_exp_path(exp_path=exp_path, burn_in_fraction = .2, thinning = 1, ignore_inference=FALSE, ignore_hist=FALSE)




#exp_path='/Users/ssalehi/Desktop/pgas_sanity_check/exp_C14AN_201801-11-001151.752864'
#plot_specific_mcmc_iter_for_exp_path(exp_path = exp_path, mcmc_iter = 40, ignore_inference=TRUE)


#plot_specific_mcmc_iter_for_exp_path(exp_path = exp_path, mcmc_iter = 4000)



# exp_path="/Users/sohrabsalehi/Desktop/pgas_sanity_check/exp_Y0CQ6_201712-15-142431.411778/"
# plot_all_for_exp_path(exp_path=exp_path, burn_in_fraction = .1, thinning = 1, ignore_inference=TRUE, ignore_hist=TRUE)
# ddd = read.table(exp_path, sep='\t', header = T)
# xyplot(X~time, ddd, group=K, type='l')
# 
# 
# exp_path="/Users/sohrabsalehi/Desktop/pgas_sanity_check/exp_nov21_k10_fixed_p100k_C14AN_201712-15-141224.472639/predict.tsv.gz"
# ddd = read.table(exp_path, sep='\t', header = T)
# xyplot(X~time|K, ddd, type='l')
# 
# 


#exp_path = '~/Desktop/pgas_sanity_check/exp_Y0CQ6_201712-15-161533.758952' # This is really bad!!!!!
#exp_path = '~/Desktop/pgas_sanity_check/exp_Y0CQ6_201712-15-162358.059152' # ALSO REALLY BAD
#exp_path = '~/Desktop/pgas_sanity_check/exp_Y0CQ6_201712-15-162932.792046' # ALSO bad 
#exp_path = '~/Desktop/pgas_sanity_check/exp_Y0CQ6_201712-19-015116.190621'

#exp_path = '~/Desktop/bulk_sa501/exp_Y0CQ6_201801-05-155003.099339'

#exp_path = '/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_5ZAX7_201801-10-011540.506520'

#exp_path = '/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_C14AN_201801-11-085308.572979'
#exp_path = '/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_C14AN_201801-11-16957.965661'

#exp_path = '/Users/sohrabsalehi/Desktop/hie_test_DRYHO_201802-07-195728.554429'

#plot_all_for_exp_path(exp_path=exp_path, ignore_inference=T, ignore_hist=T)
#plot_all_for_exp_path(exp_path=exp_path, ignore_inference=F, ignore_hist=F)


# Read the real dataset
test_analyze_bridge <- function() {
  obs_full = (read.table('/Users/sohrabsalehi/Desktop/compare_samples/sample_data_full.tsv.gz', header=T, sep='\t', stringsAsFactors = F))
  obs_full$X.1 = NULL
  obs_full$time = as.numeric(obs_full$time)
  obs_full$X = as.numeric(obs_full$X)
  
  xyplot(X~time, obs_full, type='l', groups=K)
  
  # Read the bridge paths
  dat = read.table('/Users/sohrabsalehi/Desktop/compare_samples/bridge_samples/bridge_paths.tsv.gz', header=T)
  head(dat)
  p1=xyplot(X~factor(time)|factor(K), dat, type='b', groups=n, main='bridge_samples')
  dat$type='bridge'
  
  
  # Read the pf samples
  dat_pf = read.table('/Users/sohrabsalehi/Desktop/compare_samples/pf_samples/pf_paths.tsv.gz', header=T)
  head(dat_pf)
  p2=xyplot(X~factor(time)|factor(K), dat_pf, type='b', groups=n, main='pf_samples')
  dat_pf$type = 'pf'
  
  
  # Both together
  p1+p2
  dat_both = rbind(dat_pf, dat)
  xyplot(X~factor(time)|factor(K), dat_both, type='b', groups=type, main='All samples', auto.key = T)
  
  
  ggplot(dat, aes(time)) + geom_line(aes(y=X)) + facet_wrap(~K)
  
  
  
  
  
  plot(dat$time[dat$n == 0 & dat$K == 0], dat$X[dat$n == 0 & dat$K == 0], type='b')
  for (i in seq(100)) {
    lines(dat$time[dat$n == i & dat$K == 0], dat$X[dat$n == i & dat$K == 0])
  }
  
  
  K = 2
  plot(dat$time[dat$n == 0 & dat$K == K], dat$X[dat$n == 0 & dat$K == K], type='b')
  for (i in seq(100)) {
    lines(dat$time[dat$n == i & dat$K == K], dat$X[dat$n == i & dat$K == K])
  }
  
  
  
  
  # Plot A lot of paths
  dat_wf = read.table('/Users/sohrabsalehi/Desktop/compare_samples/multi_simul/wf_paths.tsv.gz', header=T)
  xyplot(X~time|K, dat_wf, type='l', groups=n, main='dat_wf')
}


plot_paths <- function() {
  #exp_path = '/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_C14AN_201801-10-175305.791406/predict.tsv.gz'
  exp_path = '/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_C14AN_201801-12-163423.609295/predict.tsv.gz'
  exp_path = '/Users/sohrabsalehi/Desktop/pgas_sanity_check/nov22/exp_C14AN_201801-13-092510.981043/predict.tsv.gz'
  dat = read.table(exp_path, header = T, sep='\t')
  dat = dat[dat$np > 250, ]
  xyplot(X~time|K, dat, type='b', groups=np, ylim=c(0,1))
  
  kk = dat[dat$K == 1, ]
  kk$K <- NULL
  kk$X.1 <- NULL
  wide_dat <- acast(data = kk, formula = np~time, value.var = 'X')
  dat_hdi <- coda::HPDinterval(mcmc(wide_dat))
  
  summary_dat = data.frame(x=colMeans(wide_dat), lower=dat_hdi[, 1], higher=dat_hdi[, 2], time_point=as.numeric(colnames(wide_dat)))
  #summary_dat$time_point = factor(summary_dat$time_point)
  
  #wide_dat <- acast(data = dat.melt[dat.melt$np == 1, ], formula = np~time_point)
  ggplot(summary_dat, aes(time_point, x)) + 
    geom_ribbon(aes(ymin=lower, ymax=higher), alpha=0.8) +
    geom_line(linetype = 2, size=2) + ylim(-.01, 1.01)
}

plot_blang <- function() {
  file_path = '~/Downloads/process_p0.csv'
  dat = read.csv(file_path)
  #xyplot(value~index_0, dat, groups=sample, type='b')
  colnames(dat) = c('time_point', 'np', 'X')
  
  n_total = length(unique(dat$np))
  n_size = round(.01 * n_total)
  sub_sampled_np = sample(x=seq(n_total), size = n_size)
  sub_dat = dat[dat$np %in% sub_sampled_np, ]
  (p2 = xyplot(X~time_point, sub_dat, groups=np, type='l', col='black'))
  
  
  plot_hdi(dat.melt = sub_dat, closeObj = list(end_point=0.5, epsilon=.01))
}


### Find paths for which there is negative stuff
plot_illegal_paths <- function(exp_path) {
  #exp_kind = 'infer_x'
  #exp_kind = 'predict'
  param_path = file.path(exp_path, paste0(exp_kind, ".tsv.gz"))
  if (!file.exists(param_path)) next
  file_path = file.path(out_dir, paste0(exp_kind, the_suffix, '_trace', '.pdf'))
  K = mcmc_options$K
  param_dat = read.table(param_path, sep='\t', header = T, stringsAsFactors = F)
  # burn-in
  np_set = unique(param_dat$np)
  nIter = length(np_set)
  param_dat = param_dat[param_dat$np >= nIter*burn_in,]
  # thinning
  param_dat = param_dat[param_dat$np %in% np_set[seq(1, length(np_set), thinning)], ]
  
  for (index in (seq(K)-1) ) {
    #index = 0
    #exp_kind = 'infer_x'
    # load original data
    
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
    
    if (!is.null(mcmc_options$observation_model) | mcmc_options$multinomial_error) {
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

    illegal_nps <- param_dat_k$np[param_dat_k$X < 0 | param_dat_k$X > 1]
    param_dat_k$tag = 'legal'
    param_dat_k$tag[param_dat_k$np %in% illegal_nps] <- 'illegal'
    
    legal_nps = param_dat_k$np[param_dat_k$tag == 'legal']
    test_nps = sample(legal_nps, 1000)
    test_nps <- c(test_nps, illegal_nps)
    
    
    ggplot(param_dat_k[param_dat_k$np %in% test_nps, ], aes(x=time, y=X, group=factor(np), colour=factor(tag))) +
      geom_line(alpha = .2) +
      theme_light()
    
    
  }
  
    
    
}



