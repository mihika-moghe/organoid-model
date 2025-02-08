library(BoolNet)
library(doParallel)
library(fmsb)

###### Use when do simulate in multi-condition
registerDoParallel(cores=20)
getDoParWorkers()
# https://cran.r-project.org/web/packages/doParallel/vignettes/gettingstartedParallel.pdf



################ convert to attractor matrix and scores function
calc_attr_score <- function(attractors, output_list){
  attr_num <- length(attractors$attractors)
  attr_mat <- matrix(nrow = length(net$genes), ncol = attr_num, dimnames = list(net$genes))
  attr_ratio <- matrix(nrow = 1, ncol = attr_num)
  for (attr_idx in 1:attr_num) {
    attr_seq <- t(getAttractorSequence(attractors, attr_idx))
    attr_mat[,attr_idx] <- apply(attr_seq,1,mean)
    attr_ratio[attr_idx] <- round(100 * sapply(attractors$attractors[attr_idx], function(attractor) {
      attractor$basinSize/ncol(attractors$stateInfo$table)}), attr_idx)
  }
  
  attr_ratio <- c(attr_ratio)/sum(attr_ratio)
  attr_ratio_mat <- matrix(rep(attr_ratio, each=length(output_list)),length(output_list),attr_num)
  node_activity <- apply(round(100*attr_mat[output_list,],2)*attr_ratio_mat,1,sum)
  return(node_activity)
}

################ single off-perturbation analysis
# pert_double <- function(cand_node, net, output_list, input, off_node=NaN, on_node=NaN){
pert_double <- function(cand_node, net, output_list, off_node=NaN, on_node=NaN){
  
  if(is.nan(off_node) & is.nan(on_node)){
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:dim(cand_node)[1], .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesOFF = cand_node[idx,])
        calc_attr_score(attractors, output_list)
      })
  }
  else if(is.nan(off_node) & !is.nan(on_node)){
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:dim(cand_node)[1], .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesON = c(on_node), genesOFF = cand_node[idx,])
        calc_attr_score(attractors, output_list)
      })
  }
  else if(!is.nan(off_node) & is.nan(on_node)){
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:dim(cand_node)[1], .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesOFF = c(off_node,cand_node[idx,]))
        calc_attr_score(attractors, output_list)
      })
  }
  else {
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:dim(cand_node)[1], .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesON = c(on_node), genesOFF = c(off_node,cand_node[idx,]))
        calc_attr_score(attractors, output_list)
      })
  }
  
  print(ptime_pert)
  # rownames(pert_result) <- cand_nsode
  
  data_total_activity <- as.data.frame(rbind(rep(100,length(output_list)),rep(0,length(output_list)),
                                             pert_result))
  stopImplicitCluster()
  
  return(data_total_activity)
  
}




###################################################################################### 

pert_single <- function(cand_node, net, output_list, off_node=NaN, on_node=NaN){

  if(is.nan(off_node) & is.nan(on_node)){
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:length(cand_node), .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesOFF = cand_node[idx])
        calc_attr_score(attractors, output_list)
      })
  }
  else if(is.nan(off_node) & !is.nan(on_node)){
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:length(cand_node), .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesON = c(on_node), genesOFF = cand_node[idx])
        calc_attr_score(attractors, output_list)
      })
  }
  else if(!is.nan(off_node) & is.nan(on_node)){
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:length(cand_node), .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesOFF = c(off_node,cand_node[idx]))
        calc_attr_score(attractors, output_list)
      })
  }
  else {
    ptime_pert <- system.time(
      pert_result <- foreach(idx=1:length(cand_node), .combine=rbind) %dopar%{
        attractors <- getAttractors(net, type = "synchronous", method = "random", startStates = 1000000,
                                    genesON = c(on_node), genesOFF = c(off_node,cand_node[idx]))
        calc_attr_score(attractors, output_list)
      })
  }

  print(ptime_pert)
  # rownames(pert_result) <- cand_node

  data_total_activity <- as.data.frame(rbind(rep(100,length(output_list)),rep(0,length(output_list)),
                                             pert_result))
  stopImplicitCluster()

  return(data_total_activity)
}

#####################################################################################################


############## https://www.r-graph-gallery.com/143-spider-chart-with-saveral-individuals.html
# Color vector
# colors_border=c( rgb(0.2,0.5,0.5,0.9), rgb(0.8,0.2,0.5,0.9) , rgb(0.7,0.5,0.1,0.9) )
# colors_in=c( rgb(0.2,0.5,0.5,0.4), rgb(0.8,0.2,0.5,0.4) , rgb(0.7,0.5,0.1,0.4) )
plotting <- function(data, title){
  colors_border = sample(colours(),dim(data)[1]-2, replace = FALSE)
  colors_in = colors_border
  radarchart( data  , axistype=1 , title = title,
              #custom polygon
              pcol=colors_border , plwd=2 , plty=1,
              #custom the grid
              # cglcol="grey", cglty=1, axislabcol="grey", caxislabels=seq(0,100,10), cglwd=0.8,
              #custom labels
              # vlcex=0.8
  )
  
  legend(x=1.3, y=1.3, legend = rownames(data[3:dim(data)[1],]), bty = "n", pch=2 , 
         col=colors_in , text.col = "black", cex=0.6, pt.cex=0.6, y.intersp = 0.25)
}

