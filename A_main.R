############# Reference site
# https://github.com/cran/BoolNet/blob/master/R/plotAttractors.R
# https://cran.r-project.org/web/packages/BoolNet/BoolNet.pdf


library(BoolNet)
library(gtools)

source('./A_helper.R')

net <- loadNetwork("A_model.txt")
output_list <- net$genes


#########################
#        Normal         #
#########################

ptime_Normal <- system.time(
  attractors_Normal <- getAttractors(net, type = "synchronous", method = "random",
                                     startStates = 1000000, genesOFF = c("APOE4"))
)
print(ptime_Normal)
# plotAttractors(attractors_Normal)
pheno_Normal <- calc_attr_score(attractors_Normal, output_list)


#########################
#       APOE4 SNP       #
#########################
ptime_APOE4 <- system.time(
  attractors_APOE4 <- getAttractors(net, type = "synchronous", method = "random",
                                    startStates = 1000000, genesON = c("APOE4"))
  )
print(ptime_APOE4)
# plotAttractors(attractors_APOE4)
pheno_APOE4 <- calc_attr_score(attractors_APOE4, output_list)


#########################
#        LPL SNP        #
#########################
ptime_LPL <- system.time(
    attractors_LPL <- getAttractors(net, type = "synchronous", method = "random",
                                    startStates = 1000000, genesOFF = c("APOE4","LPL"))
  )
pheno_LPL <- calc_attr_score(attractors_LPL, output_list)

#########################################
#        Make perturbation table        #
#########################################
source('./A_helper.R')


##################### APOE4
s_target <- "p53"
d_target <- rbind(c("PTEN","Dkk1"), c("MKK7","synj1"),c("PTEN","mTOR"))
pert_APOE4_s <- pert_single(s_target, net, output_list, off_node = , on_node = "APOE4")
pert_APOE4_d <- pert_double(d_target, net, output_list, off_node = , on_node = "APOE4")


pert_APOE4_s1 <- t(pert_APOE4_s[-c(1,2),])
pert_APOE4_d1 <- t(pert_APOE4_d[-c(1,2),])

APOE4_pert_res <- cbind(pert_APOE4_s1,pert_APOE4_d1)
colnames(APOE4_pert_res) <- c("p53","PTEN/Dkk1","MKK7/synj1","PTEN/mTOR")

write.table(APOE4_pert_res, "APOE4_pert_res_0429.txt", sep = "\t", row.names = TRUE, col.names = TRUE)

output_node_list <- c("Dkk1","LRP6","RhoA","ROCK","GSK3beta","DLK","MKK7","ERK","CREB","p38","JNK","PI3K","AKT",
                      "CREB","PTEN","FOXO","PP2A","Fyn","s_NMDAR","e_NMDAR","mGluR","Ca_ion","PP2B",
                      "CLU","SORL1","SREBP2","ABCA7","HMGCR","LPL","Sortilin","CYP46A1","Cholesterol","mTOR",
                      "ULK1","beclin1","LAMP2","LC3","p62","ATM","p53","MDM2","BAD","BAX","Bcl2","Bim","CASP2",
                      "CASP3","PUMA","LIMK1","Cofilin","MAPT","Cdk5","APP","BACE1","a_secretase","CIP2A")

output_idx <- match(output_node_list, rownames(APOE4_pert_res))

filt_res <- APOE4_pert_res[output_idx,]

##################### LPL
s_target <- "RhoA"
d_target <- rbind(c("JNK","Cdk5"),c("PTEN","Dkk1"))
pert_LPL_s <- pert_single(s_target, net, output_list, off_node = c("LPL","APOE4"), on_node = )
pert_LPL_d <- pert_double(d_target, net, output_list, off_node = c("LPL","APOE4"), on_node = )


pert_LPL_s1 <- t(pert_LPL_s[-c(1,2),])
pert_LPL_d1 <- t(pert_LPL_d[-c(1,2,4),])

LPL_pert_res <- cbind(pert_LPL_s1,pert_LPL_d1)
colnames(LPL_pert_res) <- c("RhoA","JNK/Cdk5")

output_idx <- match(output_node_list, rownames(LPL_pert_res))
filt_res <- LPL_pert_res[output_idx,]
