# ssh transfer@scrna-pipeline2.canadacentral.cloudapp.azure.com
# cd /datadrive/fitness
# 
# Password123!
#   scp /Users/sohrabsalehi/Downloads/tsne_fitness.csv transfer@scrna-pipeline2.canadacentral.cloudapp.azure.com
# 
# scp ~/projects/fitness_material/tmp/figures/velocyto/figures/de_genes_R_X7_X6.rds tran:/datadrive/fitness/
#   
#   
#   `
# rsync -a tran:/datadrive/fitness/figures/ ~/projects/fitness_material/figures/velocyto/figures; open ~/projects/fitness_material/figures/velocyto/figures
# `
# 
# 
# ```
library(velocyto.R)
library(pagoda2)
library(biomaRt)

list.files('/datadrive/fitness/docker_out/')

loom_path <- '/datadrive/fitness/docker_out/combined_all__.loom'
ldat <- read.loom.matrices(loom_path)

emat <- ldat$spliced
emat <- emat[,colSums(emat)>=1e3]

# See here for gene filters
# https://jef.works/blog/2020/01/14/rna_velocity_analysis_tutorial_tips/

# check cell cycle marker
gs = c('MCM5','MCM6','CCNF','KIF2C')

## mRNA only
grab_mrans <- function(genes) {
  # BiocManager::install("biomaRt")
  # genes rownames(emat)
  mart <- useMart(biomart = "ENSEMBL_MART_ENSEMBL", dataset = 'hsapiens_gene_ensembl', host = "jul2015.archive.ensembl.org") 
  results <- getBM(attributes=c('hgnc_symbol', "transcript_biotype"),
                   filters = 'hgnc_symbol',
                   values = genes,
                   mart = mart)
  head(results); table(results$transcript)
  mrnas <- results$hgnc_symbol[results$transcript_biotype == 'protein_coding']
  length(mrnas)
  mrnas
}


mrnas <- grab_mrans(rownames(emat))


# Read the new embedding
emb.n <- read.csv('/datadrive/fitness/temp/tsne_fitness.csv', stringsAsFactors = FALSE)
cells <- emb.n$X



### Clustering analysis (test for the embedding form)
rownames(emat) <- make.unique(rownames(emat))
r <- Pagoda2$new(emat,modelType='plain',trim=10,log.scale=T)
r$adjustVariance(plot=T,do.par=T,gam.k=10)
r$calculatePcaReduction(nPcs=100,n.odgenes=3e3,maxit=300)
r$makeKnnGraph(k=30,type='PCA',center=T,distance='cosine');
r$getKnnClusters(method=multilevel.community,type='PCA',name='multilevel')
r$getEmbedding(type='PCA',embeddingType='tSNE',perplexity=50,verbose=T)





### Velocity estimation

#### Keep cells from the nick embeding
function() {
  # 34132 cells in nick's embedding
  samples <- gsub('(SA([0-9]|[A-Z])+)_.*', '\\1', emb.n$X)
  table(samples)
  # Rx samples
  samples_rx <- gsub('(SA([0-9]|[A-Z])+):.*', '\\1', colnames(emat))
  # SA609X4XB003083  SA609X5XB03230  SA609X6XB03404  SA609X7XB03505 
  #          1094            3046            3742            2945 
  samples <- samples[samples %in% unique(samples_rx)]
  
  # update nick's embedding to match the current cells
  # change SA609X4XB003083 into SA609X4XB03083
  emb.n$X <- gsub('SA609X4XB03083', 'SA609X4XB003083', emb.n$X)
  # Only keep the Rx samples
  emb.n <- emb.n[gsub('(SA([0-9]|[A-Z])+)_.*', '\\1', emb.n$X) %in%  c('SA609X4XB003083', 'SA609X5XB03230', 'SA609X6XB03404', 'SA609X7XB03505'), ]
  # Leaves us with 8960 cells
  # 10827 cells in emat
  # update the emb.n cell names to match "SA609X7XB03505:AAAGATGCAAGGACTGx"
  # from SA609X6XB03404_AAACCCAAGCCTGCCA-1
  emb.n$X <- gsub('-1', 'x', emb.n$X)
  emb.n$X <- gsub('_', ':', emb.n$X)
  # what is not matching...
  qq <- colnames(emat)[!(colnames(emat) %in% emb.n$X)]
  # check sample of qq 
  unique(gsub('(SA([0-9]|[A-Z])+):.*', '\\1', qq))
  # Cells not in the embedding
  # SA609X4XB003083  SA609X5XB03230  SA609X6XB03404  SA609X7XB03505 
  #        218             534             534             594 
  mm = sample(qq, 10)
  any(grepl("SA609X7XB003505.*", emb.n$X))
  any(grepl("SA609X6XB03404:GCACGGTGTG.*", emb.n$X))
  # accept for now
  cells <- intersect(colnames(emat), emb.n$X)
  cells
}


emat <- ldat$spliced; nmat <- ldat$unspliced
emat <- emat[,rownames(r$counts)]; nmat <- nmat[,rownames(r$counts)]; # restrict to cells that passed p2 filter

# just pick the intersect
# use mrna genes
emat <- emat[mrnas,cells]; nmat <- nmat[mrnas,cells]


# Clusters = timepoints
cluster.label  <- gsub('(SA([0-9]|[A-Z])+):.*', '\\1', colnames(emat))

# add colour
cell.colors <- as.character(factor(cluster.label, levels = c('SA609X4XB003083',  'SA609X5XB03230', 'SA609X6XB03404',  'SA609X7XB03505'), labels = as.character(wes_palette("GrandBudapest1", n = 4, type = 'continuous'))))

timepoints <- c('SA609X4XB003083',  'SA609X5XB03230', 'SA609X6XB03404',  'SA609X7XB03505')
cols <- rainbow(5)[-c(2)]
cell.colors <- as.character(factor(cluster.label, levels = timepoints, labels = cols))
names(cell.colors) <- colnames(emat)

cdat <- data.frame(clab = cluster.label, ccol = unname(cell.colors), stringsAsFactors = FALSE); cdat <- cdat[!duplicated(cdat$clab), ]; cdat <- cdat[order(cdat$clab), ]
myCol <- cdat$ccol; names(myCol) <- cdat$clab

png('/datadrive/fitness/figures/legend.png', width=1000, height=1000, res=300)
p <- cdat %>% ggplot(aes(x = clab, y = 1 , col = clab, label = gsub('.*(X[0-9]).*', '\\1', clab) )) + geom_text() + scale_color_manual(values = myCol) + theme_void(base_size = 2) + theme(legend.position = 'none')
print(p)
dev.off()

# transform emb.n to a matrix with X as its rownames
emb.m <- as.matrix(emb.n[, c(2,3)]); rownames(emb.m) <- emb.n$X; colnames(emb.m) <- NULL
emb.m <- emb.m[cells, ]


# take cluster labels
cluster.label <- r$clusters$PCA[[1]]
cell.colors <- pagoda2:::fac2col(cluster.label)
# take embedding
emb <- r$embeddings$PCA$tSNE


cell.dist <- as.dist(1-armaCor(t(r$reductions$PCA)))

emat <- filter.genes.by.cluster.expression(emat,cluster.label,min.max.cluster.average = 0.5)
nmat <- filter.genes.by.cluster.expression(nmat,cluster.label,min.max.cluster.average = 0.05)
length(intersect(rownames(emat),rownames(emat)))


fit.quantile <- 0.02
rvel.cd <- gene.relative.velocity.estimates(emat,nmat,deltaT=1,kCells=20,cell.dist=cell.dist,fit.quantile=fit.quantile)


## without pooling
rvel.cd <- gene.relative.velocity.estimates(emat, nmat, deltaT=1,fit.quantile = fit.quantile,
                                            #min.nmat.emat.correlation = 0.2, 
                                            #min.nmat.emat.slope = 0.2, 
                                            #                                           kCells = 1)
                                            kCells = 20)

saveRDS(rvel.cd, '/datadrive/fitness/rvel.cd.rds')
dir.create('/datadrive/fitness/figures')

celcol <- cell.colors[rownames(emb)]

defaultNCores()


#png('/datadrive/fitness/figures/rx_velocity.png',width=1000,height=1000, res=300)
tiff("/datadrive/fitness/figures/rx_velocity.tiff", units="in", width=5, height=5, res=300)
x <- show.velocity.on.embedding.cor(emb.m, 
                                    rvel.cd, 
                                    n=300, 
                                    scale='sqrt', 
                                    cell.colors = ac(cell.colors, alpha=0.5), 
                                    cex = 0.8, 
                                    arrow.scale = 10, 
                                    show.grid.flow = TRUE, 
                                    min.grid.cell.mass = 0.5, 
                                    grid.n = 40, 
                                    arrow.lwd = 1, 
                                    do.par = F, 
                                    cell.border.alpha = 0.1, 
                                    cc = x$cc, 
                                    return.details = FALSE)
dev.off()

#saveRDS(x, '/datadrive/fitness/detailed_output.rds')



# just velocity sizes (compare to rvel.cd$current)
# TODO: split velocity sizes into differentiation, maturation, and proliferation axis
#x$vel gene by cell matrix
# x$vel is the sqrt scaled version of rvel.cd$deltaE
tt <- x$vel[1:5, 1:5]  
# for each timepoint, what is the abs(sum) of the velocities? What is the 
saveRDS(x$vel, "/datadrive/fitness/figures/vel_est.rds")

# Show the sum(abs(vel)) across all genes in each timepoint normalised by the number of cells in each timepoint
qq <- colSums(abs(tt))


######################################
# todo: look at velocity of gene at each timepoint and do some sort of DE analysis?
######################################



# Use all genes with reads at least 1000
######################################
rcd <- get_rvel(ldat, r, cells)
saveRDS(rcd, '/datadrive/fitness/de_genes_R_X7_X6_all_genes.rds')
vv <- get_vel_mat(rcd, emb.m)
get_abs_vel(vv, 'all_genes')


######################################
# Use genes in the DE-R_X7_X6 
######################################
genes <- readRDS('/datadrive/fitness/de_genes_R_X7_X6.rds')
rcd <- get_rvel(ldat, r, cells, genes = genes)
vv <- get_vel_mat(rcd, emb.m)
get_abs_vel(vv, 'de_rx7x6_genes')



######################################
# Use high std genes
######################################
# Subsample the velocity already calculated
rcd_all <- readRDS('/datadrive/fitness/de_genes_R_X7_X6_all_genes.rds')
# rcd <- get_rvel(ldat, r, cells, genes = NULL)
qq <- get_gene_stds(ldat, r, cells)
vv <- get_vel_mat(rcd, emb.m)
qq_h <- qq[qq > .9]
qq_h <- qq[qq > .6]
get_abs_vel(vv, 'high_std_genes', genes = names(qq_h))



######################################
# Use Mirela's genes
######################################
local_pre_process <- function() {
  # Read genes
  mcells <- read.csv('/Users/sohrabsalehi/projects/fitness_material/tables/allsets_covthr200_cells.csv', stringsAsFactors = F) %>% as_tibble()
  mgenes <- read.csv('/Users/sohrabsalehi/projects/fitness_material/tables/allsets_covthr200_genes.csv', stringsAsFactors = F) %>% as_tibble()
  
  mcells <- mcells %>% 
    dplyr::mutate(timepoint = gsub('(X[0-9]+) [A-Z]+', '\\1', condition), 
                  datatag = gsub('(X[0-9]+) ([A-Z]+)', '\\2', condition)) %>% 
    dplyr::filter(str_sub(datatag, '-1') == 'T')
  nrow(mcells)
  table(mcells$condition)
  
  conv <- data.frame(condition = c('X4 UT',  'X5 UTT', 'X6 UTTT', 'X7 UTTTT'),
                     sample_name = rev(c("SA609X7XB03505", "SA609X6XB03404", "SA609X5XB03230", "SA609X4XB003083")), stringsAsFactors = F)
  mcells <- dplyr::left_join(mcells, conv)                   
  mcells$cell_name <- sprintf('%s:%s', mcells$sample_name, gsub('-1', 'x', mcells$Barcode))                 
  stopifnot(all(str_sub(mcells$cell_name, -1) == 'x'))

  saveRDS(list(vcells = mcells$cell_name, vgenes = mgenes$gene_symbol), '~/projects/fitness_material/results/volume/volume_cells_genes.RDS')
}

# All preamble
library(velocyto.R)
library(pagoda2)
library(biomaRt)
library(ggplot2)
library(dplyr)

list.files('/datadrive/fitness/docker_out/')

loom_path <- '/datadrive/fitness/docker_out/combined_all__.loom'
ldat <- read.loom.matrices(loom_path)

emat <- ldat$spliced
emat <- emat[,colSums(emat)>=1e3]
# cells names: SA609X4XB003083:TTCATGTTCGTTCTGCx
  
# Load Mirela's genes 
mlist <- readRDS('/datadrive/fitness/volume/volume_cells_genes.RDS')
mcells <- mlist$vcells; mgenes <- mlist$vgenes
# cell names: SA609X7XB03505:TTTGGTTGTGATAAGTx

emb.t <- get_nicks_embedding()
# SA609X7XB03505:TTTGTCACAGTGACAG.1


# Use white colour for all cells
#wcolours <- rep('white', length(cell.colors))
#names(wcolours) <- names(cell.colors)
cell.colors <- rep('white', nrow(emb.t))
names(cell.colors) <- rownames(emb.t)

rvel.all <- get_rvel(ldat, mcells, mgenes)

tag <- 'sync_volume_pure'

vv <- get_vel_mat(rvel.all, emb.t)
# Plot the cumulative abs velocity: datadrive/fitness/figures/vel_est_box_[].tiff
get_abs_vel(vv, tag)

# Use Mirela's cells
x_vals <- list()

for (tp in c("SA609X7XB03505", "SA609X6XB03404", "SA609X5XB03230", "SA609X4XB003083")) {
  # tp <- "SA609X7XB03505"
  cell.tp <- mcells[cell_2_sample(mcells) == tp]
  print(sprintf('Processing tp = %s at %d cells...', tp, length(cell.tp)))
  emb.t.tp <- emb.t[intersect(rownames(emb.t), cell.tp), ]
  
  tiff(sprintf("/datadrive/fitness/figures/rx_velocity_%s_%s.tiff", tag, tp), units="in", width=5, height=5, res=300)
  
  #plot(emb.t,bg='white',pch=21, col=ac(1,alpha=.1), xlab='', ylab='');
  x <- NULL
  if (tp %in% names(x_vals)) {
    x <- x_vals[[tp]]
  }
  #x_vals[[tp]] <- show.velocity.on.embedding.cor(emb.t.tp,
  
  # Filter the vel object to retain only these timepoint's cells
  rvel.tp <- rvel.all
  # Ensure the order of cells are correct
  oo <- match(cell.tp, colnames(rvel.tp$current)); oo <- oo[!is.na(oo)]
  rvel.tp$current <- rvel.tp$current[, oo]
  # Mind the NAs
  # stopifnot(all(match(cell.tp, colnames(rvel.tp$current)) == sort(1:length(cell.tp))))
  # Do we need to filter the other objects in rvel.tp?
  
  
  x_vals[[tp]] <- show.velocity.on.embedding.cor(emb.t,
                                                 rvel.tp, 
                                                 n=300, 
                                                 scale='sqrt', 
                                                 cell.colors = ac(cell.colors, alpha=0.0), 
                                                 cex = 0.8, 
                                                 arrow.scale = 5, 
                                                 show.grid.flow = TRUE, 
                                                 min.grid.cell.mass = 0.5, 
                                                 grid.n = 40, 
                                                 arrow.lwd = 1, 
                                                 do.par = F, 
                                                 cell.border.alpha = 0.02, 
                                                 cc = x$cc, 
                                                 return.details = FALSE)
  dev.off()
}















