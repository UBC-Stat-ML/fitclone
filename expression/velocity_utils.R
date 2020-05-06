cell_2_sample <- function(cell_names) {
  gsub('(SA([0-9]|[A-Z])+):.*', '\\1', cell_names)
}

# Get Nick's embedding
get_nicks_embedding <- function() {
  emb.x <- read.csv('/datadrive/fitness/temp/tsne_fitness.csv', stringsAsFactors = FALSE) # 37114     3
  dim(emb.x)
  #stop('Bunch of cells with .1.1')
  #emb.x$X <- gsub('-1', 'x', emb.x$X)
  emb.x$X <- gsub('\\.1\\.1', '\\.1', emb.x$X)
  emb.x$X <- gsub('\\.1', 'x', emb.x$X)
  any(grepl('xx', emb.x$X))
  any(grepl('-1', emb.x$X))
  emb.x$X <- gsub('_', ':', emb.x$X)
  emb.t <- as.matrix(emb.x[, c(2,3)]); rownames(emb.t) <- emb.x$X; colnames(emb.t) <- NULL
  rownames(emb.t) <- gsub('SA609X4XB03083', 'SA609X4XB003083', rownames(emb.t))
  unique(cell_2_sample(rownames(emb.t)))
  table(cell_2_sample(rownames(emb.t)))
  emb.t <- emb.t[cell_2_sample(rownames(emb.t)) %in% c("SA609X7XB03505", "SA609X6XB03404", "SA609X5XB03230", "SA609X4XB003083"), ]
  table(cell_2_sample(rownames(emb.t)))
  emb.t
}



######################################
get_abs_vel <- function(vel, tag='', genes = NULL) {
  # vel is gene by cell
  if (!is.null(genes)) {
    genes <- intersect(genes, rownames(vel))
    vel <- vel[genes, ]
  }
  qq <- colSums(abs(vel))
  dd <- data.frame(cell_name = names(qq), abs_vel = abs(unname(qq)), timepoint = gsub('(SA([0-9]|[A-Z])+):.*', '\\1', names(qq)), 
                   stringsAsFactors = FALSE)
  dd$timepoint_name <- gsub('.*(X[0-9]).*', '\\1', dd$timepoint)
  dd <- dd %>% dplyr::group_by(timepoint_name) %>% 
    dplyr::mutate(median_abs_vel = median(abs_vel), 
                  mean_abs_vel = mean(abs_vel), 
                  timepoint_label = sprintf('%s(%d)', unique(timepoint_name), dplyr::n())) %>% 
    dplyr::ungroup()
  
  saveRDS(dd, sprintf("/datadrive/fitness/figures/vel_est_box_%s.rds", tag))
  print("Saving boxplot...")
  ncells <- ncol(vel); ngenes = nrow(vel); 
  tiff(sprintf("/datadrive/fitness/figures/vel_est_box_%s.tiff", tag), units="in", width=5, height=5, res=300)
  p1 <- dd %>% ggplot(aes(x = timepoint_label, y = abs_vel, group = timepoint)) + 
    ggtitle(label = 'RNA velocity in the TNBC Rx line', subtitle = sprintf('# Cells = %d; # Genes = %d', ncells, ngenes)) + 
    geom_violin(aes(fill = mean_abs_vel)) + 
    geom_boxplot(width=0.05, fill = NA, colour = 'white', outlier.stroke = 0, outlier.colour = NA, outlier.fill =  NA, size = .5, coef = 1) + 
    xlab('Timepoint') + ylab('Absolute cumulative RNA velocity') + 
    scale_fill_gradient(
      n.breaks = 3,
      low = "grey",
      high = "darkviolet",
      space = "Lab",
      na.value = "black",
      guide = "colourbar", 
      name = 'Mean velocity'
    ) + 
    cowplot::theme_cowplot() + 
    theme(legend.position = 'bottom')
  print(p1)
  dev.off()
  
  dd %>% dplyr::group_by(timepoint) %>% dplyr::summarise(mean_vel_per_cell = mean(abs_vel)/n(), ncells = n()) %>% dplyr::ungroup()
}
######################################

# x$vel is the same as nd
######################################
get_vel_mat <- function(rvel.cd, emb.m) {
  vel <- rvel.cd
  em <- as.matrix(vel$current);
  ccells <- intersect(rownames(emb.m),colnames(em));
  nd <- as.matrix(vel$deltaE[,ccells])
  cgenes <- intersect(rownames(em),rownames(nd));
  nd <- nd[cgenes,]; em <- em[cgenes,]
  scale <- "sqrt"
  if(scale=='log') {
    nd <- (log10(abs(nd)+1)*sign(nd))
  } else if(scale=='sqrt') {
    nd <- (sqrt(abs(nd))*sign(nd))
  }
  nd
}
######################################

######################################
# Get it using all the genes
######################################
# get_rvel <- function(ldat, r, cells, genes = NULL) {
#   emat <- ldat$spliced; nmat <- ldat$unspliced
#   # restrict to cells that passed p2 filter
#   emat <- emat[,rownames(r$counts)]; nmat <- nmat[,rownames(r$counts)];
#   emat <- emat[, cells]; nmat <- nmat[, cells]
#   if (!is.null(genes)) {
#     emat <- emat[genes, ]; nmat <- nmat[genes, ]
#   }
#   fit.quantile <- 0.02
#   rvel <- gene.relative.velocity.estimates(emat, nmat, deltaT=1, fit.quantile = fit.quantile,
#                                            #min.nmat.emat.correlation = 0.2, 
#                                            #min.nmat.emat.slope = 0.2, 
#                                            kCells = 20)
#   rvel                                           
# }



# Compute velocity for given genes and cells 
get_rvel <- function(ldat, mcells, mgenes) {
  fit.quantile <- 0.02
  emat <- ldat$spliced; nmat <- ldat$unspliced
  emat <- emat[intersect(rownames(emat), mgenes), intersect(colnames(emat), mcells)]; 
  nmat <- nmat[intersect(rownames(nmat), mgenes), intersect(colnames(nmat), mcells)]; 
  gene.relative.velocity.estimates(emat, nmat, deltaT=1, fit.quantile = fit.quantile, kCells = 20)
}




# Find genes with highest variance
# based on emat
# numCores <- detectCores()
# numCores
library(parallel)

get_gene_stds <- function(ldat, r, cells, genes = NULL) {
  numCores <- detectCores() - 10
  emat <- ldat$spliced; nmat <- ldat$unspliced
  # restrict to cells that passed p2 filter
  emat <- emat[,rownames(r$counts)]; nmat <- nmat[,rownames(r$counts)];
  emat <- emat[, cells]; nmat <- nmat[, cells]
  if (!is.null(genes)) {
    emat <- emat[genes, ]; nmat <- nmat[genes, ]
  }
  print(sprintf("ngenes = %d - ncells = %d", nrow(emat), ncol(emat)))
  stds <- matrix(NA, nrow = nrow(emat), ncol = 1)
  
  
  std_func <- function(i) {
    sd(emat[i, ])
  }
  
  results <- parallel::mclapply(1:nrow(emat), std_func, mc.cores = numCores)
  
  names(results) <- rownames(emat)
  unlist(results)
}

