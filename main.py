from helper import calc_attr_score, pert_single, pert_double
import pandas as pd
import boolnet
import time
from helper import calc_attr_score, pert_single, pert_double


# Load network
net = boolnet.load_network("A_model.txt")
output_list = net('genes')

#########################
#        Normal         #
#########################

start_time = time.time()
attractors_Normal = net.get_attractors(
    type="synchronous",
    method="random",
    start_states=1000000,
    genes_off=["APOE4"]
)
print(f"Normal analysis time: {time.time() - start_time:.2f}s")
        

if not attractors_Normal.get('attractors'):
    raise ValueError("No attractors found in normal analysis")

pheno_Normal = calc_attr_score(attractors_Normal, output_list)


#########################
#       APOE4 SNP       #
#########################
start_time = time.time()
attractors_APOE4 = net.get_attractors(
    type="synchronous",
    method="random",
    start_states=1000000,
    genes_on=["APOE4"]
)
print(f"APOE4 analysis time: {time.time() - start_time:.2f}s")

if not attractors_APOE4.get('attractors'):
    raise ValueError("No attractors found in APOE4 analysis")

pheno_APOE4 = calc_attr_score(attractors_APOE4, output_list)

#########################
#        LPL SNP        #
#########################
start_time = time.time()
attractors_LPL = net.get_attractors(
    type="synchronous",
    method="random",
    start_states=1000000,
    genes_off=["APOE4", "LPL"]
)
print(f"LPL analysis time: {time.time() - start_time:.2f}s")

if not attractors_LPL.get('attractors'):
    raise ValueError("No attractors found in LPL analysis")

pheno_LPL = calc_attr_score(attractors_LPL, output_list)

#########################################
#        Make perturbation table        #
#########################################

# APOE4 analysis
s_target = ["p53"]
d_target = [
    ("PTEN", "Dkk1"),
    ("MKK7", "synj1"),
    ("PTEN", "mTOR")
]

pert_APOE4_s = pert_single(s_target, net, output_list, on_node=["APOE4"])
pert_APOE4_d = pert_double(d_target, net, output_list, on_node=["APOE4"])

pert_APOE4_s1 = pert_APOE4_s.iloc[2:].T
pert_APOE4_d1 = pert_APOE4_d.iloc[2:].T

APOE4_pert_res = pd.concat([pert_APOE4_s1, pert_APOE4_d1], axis=1)
APOE4_pert_res.columns = ["p53", "PTEN/Dkk1", "MKK7/synj1", "PTEN/mTOR"]

APOE4_pert_res.to_csv("APOE4_pert_res.txt", sep="\t")

output_node_list = [
    "Dkk1", "LRP6", "RhoA", "ROCK", "GSK3beta", "DLK", "MKK7", "ERK", "CREB",
    "p38", "JNK", "PI3K", "AKT", "CREB", "PTEN", "FOXO", "PP2A", "Fyn",
    "s_NMDAR", "e_NMDAR", "mGluR", "Ca_ion", "PP2B", "CLU", "SORL1", "SREBP2",
    "ABCA7", "HMGCR", "LPL", "Sortilin", "CYP46A1", "Cholesterol", "mTOR",
    "ULK1", "beclin1", "LAMP2", "LC3", "p62", "ATM", "p53", "MDM2", "BAD",
    "BAX", "Bcl2", "Bim", "CASP2", "CASP3", "PUMA", "LIMK1", "Cofilin",
    "MAPT", "Cdk5", "APP", "BACE1", "a_secretase", "CIP2A"
]

filt_res = APOE4_pert_res.loc[output_node_list]

# LPL analysis
s_target = ["RhoA"]
d_target = [
    ("JNK", "Cdk5"),
    ("PTEN", "Dkk1")
]

pert_LPL_s = pert_single(s_target, net, output_list, off_node=["LPL", "APOE4"])
pert_LPL_d = pert_double(d_target, net, output_list, off_node=["LPL", "APOE4"])

pert_LPL_s1 = pert_LPL_s.iloc[2:].T
pert_LPL_d1 = pert_LPL_d.iloc[2:, :-1].T

LPL_pert_res = pd.concat([pert_LPL_s1, pert_LPL_d1], axis=1)
LPL_pert_res.columns = ["RhoA", "JNK/Cdk5"]

output_idx = [LPL_pert_res.index.get_loc(node) for node in output_node_list if node in LPL_pert_res.index]
filt_res = LPL_pert_res.iloc[output_idx]
