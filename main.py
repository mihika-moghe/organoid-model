import pandas as pd
import boolnet
import time
from helper import calc_attr_score, pert_single, pert_double
import efficacy


print("Loading network...")
net = boolnet.load_network("A_model.txt")
output_list = net['genes'] 
net = boolnet.load_network("A_model.txt")
print(f"Loaded network with {len(net['genes'])} genes")
print(f"First gene {net['genes'][0]} has inputs: {net['interactions'][0]['input']}")


#########################
#        Normal         #
#########################

print("\nStarting Normal analysis...")
start_time = time.time()
attractors_Normal = boolnet.get_attractors(
    net, 
    type="synchronous",
    method="random",
    start_states=1000000,
    genes_off=["APOE4"]
)
print(f"Normal analysis time: {time.time() - start_time:.2f}s")
        

if not attractors_Normal.get('attractors'):
    raise ValueError("No attractors found in normal analysis")

pheno_Normal = calc_attr_score(attractors_Normal, output_list)
print("\nNormal phenotype scores:")

#########################
#       APOE4 SNP       #
#########################

print("\nStarting APOE4 analysis...")
start_time = time.time()
attractors_APOE4 = boolnet.get_attractors(
    net, 
    type="synchronous",
    method="random",
    start_states=1000000,
    genes_on=["APOE4"]
)
print(f"APOE4 analysis time: {time.time() - start_time:.2f}s")

if not attractors_APOE4.get('attractors'):
    raise ValueError("No attractors found in APOE4 analysis")

pheno_APOE4 = calc_attr_score(attractors_APOE4, output_list)
print("\nAPOE4 phenotype scores:")

#########################
#        LPL SNP        #
#########################

print("\nStarting LPL analysis...")
start_time = time.time()
attractors_LPL = boolnet.get_attractors(
    net,
    type="synchronous",
    method="random",
    start_states=1000000,
    genes_off=["APOE4", "LPL"]
)
print(f"LPL analysis time: {time.time() - start_time:.2f}s")

if not attractors_LPL.get('attractors'):
    raise ValueError("No attractors found in LPL analysis")

pheno_LPL = calc_attr_score(attractors_LPL, output_list)
print("\nLPL phenotype scores:")

#########################################
#        Make perturbation table        #
#########################################

print("\nStarting perturbation analysis...")
# APOE4 analysis
s_target = ["p53"]
d_target = [
    ("PTEN", "Dkk1"),
    ("MKK7", "synj1"),
    ("PTEN", "mTOR")
]
print("\nAnalyzing APOE4 single perturbations...")
pert_APOE4_s = pert_single(s_target, net, output_list, on_node=["APOE4"])
print("\nAnalyzing APOE4 double perturbations...")
pert_APOE4_d = pert_double(d_target, net, output_list, on_node=["APOE4"])

pert_APOE4_s1 = pert_APOE4_s.iloc[2:].T
pert_APOE4_d1 = pert_APOE4_d.iloc[2:].T
 
print(f"pert_APOE4_s1 shape: {pert_APOE4_s1.shape}")
print(f"pert_APOE4_d1 shape: {pert_APOE4_d1.shape}")

print("\nAPOE4 perturbation results:")
if pert_APOE4_s1.shape[1] == 1 and pert_APOE4_d1.shape[1] == 3:
     
    APOE4_pert_res = pd.concat([pert_APOE4_s1, pert_APOE4_d1], axis=1)
    APOE4_pert_res.columns = ["p53", "PTEN/Dkk1", "MKK7/synj1", "PTEN/mTOR"]
else:
     
    APOE4_pert_res = pd.concat([pert_APOE4_s1, pert_APOE4_d1], axis=1)
    print(f"Combined DataFrame has {APOE4_pert_res.shape[1]} columns")
    
    if APOE4_pert_res.shape[1] == 2:
        APOE4_pert_res.columns = ["p53", "Combined_Perturbations"]
print("Saved APOE4 results to APOE4_pert_res.txt")
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

common_nodes = [node for node in output_node_list if node in APOE4_pert_res.index]
print(f"Found {len(common_nodes)} nodes in DataFrame index")

if common_nodes:
    print("\nFiltered APOE4 perturbation results:")
    filt_res = APOE4_pert_res.loc[common_nodes]
else:
    print("\nWARNING: None of the output nodes found in results!")
    print("DataFrame index contains:", APOE4_pert_res.index.tolist()[:10], "...")
     
    filt_res = pd.DataFrame(columns=APOE4_pert_res.columns)

 
print("\nStarting LPL perturbation analysis...")

s_target = ["RhoA"]
d_target = [
    ("JNK", "Cdk5"),
    ("PTEN", "Dkk1")
]

print("\nAnalyzing LPL single perturbations...")
pert_LPL_s = pert_single(s_target, net, output_list, off_node=["LPL", "APOE4"])
print("\nAnalyzing LPL double perturbations...")
pert_LPL_d = pert_double(d_target, net, output_list, off_node=["LPL", "APOE4"])

 
print(f"pert_LPL_s shape: {pert_LPL_s.shape}")
print(f"pert_LPL_d shape: {pert_LPL_d.shape}")

 
if pert_LPL_s.shape[1] > 2:   
    pert_LPL_s1 = pert_LPL_s.iloc[2:].T
else:
    print("WARNING: pert_LPL_s appears to be empty or malformed")
    pert_LPL_s1 = pd.DataFrame()   

if pert_LPL_d.shape[1] > 2:
    pert_LPL_d1 = pert_LPL_d.iloc[2:, :-1].T
else:
    print("WARNING: Using all available columns from pert_LPL_d")
    pert_LPL_d1 = pert_LPL_d.iloc[2:].T if pert_LPL_d.shape[0] > 2 else pd.DataFrame()

print(f"pert_LPL_s1 shape after transformation: {pert_LPL_s1.shape}")
print(f"pert_LPL_d1 shape after transformation: {pert_LPL_d1.shape}")

if not pert_LPL_s1.empty or not pert_LPL_d1.empty:
    print("\nLPL perturbation results:")
    if pert_LPL_s1.empty:
        LPL_pert_res = pert_LPL_d1
    elif pert_LPL_d1.empty:
        LPL_pert_res = pert_LPL_s1
    else:
        LPL_pert_res = pd.concat([pert_LPL_s1, pert_LPL_d1], axis=1)
    
    num_cols = LPL_pert_res.shape[1]
    if num_cols == 1:
        if not pert_LPL_s1.empty:
            LPL_pert_res.columns = ["RhoA"]
        else:
            LPL_pert_res.columns = ["JNK/Cdk5"]
    elif num_cols == 2:
        LPL_pert_res.columns = ["RhoA", "JNK/Cdk5"]
    elif num_cols == 3:
        LPL_pert_res.columns = ["RhoA", "JNK/Cdk5", "PTEN/Dkk1"]
    else:
        LPL_pert_res.columns = [f"Column_{i}" for i in range(num_cols)]
    
    print(f"Index type: {LPL_pert_res.index[:5]}")
    if all(isinstance(idx, (int, float)) or idx == 'Base' for idx in LPL_pert_res.index[:5]):
        print("Index appears to be numeric or default - trying to set gene names")
        if hasattr(output_list, '__iter__') and len(output_list) >= len(LPL_pert_res):
            LPL_pert_res.index = output_list[:len(LPL_pert_res)]
            print(f"Set index to gene names from output_list")
    
    common_nodes = [node for node in output_node_list if node in LPL_pert_res.index]
    print(f"Found {len(common_nodes)} common nodes for LPL results")
    
    if common_nodes:
        print("\nFiltered LPL perturbation results:")
        filt_res = LPL_pert_res.loc[common_nodes]
    else:
        print("\nWARNING: No nodes from output_node_list found in LPL results!")
        print("LPL_pert_res index contains first few:", list(LPL_pert_res.index)[:10])
        filt_res = pd.DataFrame(columns=LPL_pert_res.columns)
else:
    print("ERROR: No valid perturbation data available for LPL analysis")
    filt_res = pd.DataFrame(columns=["RhoA", "JNK/Cdk5"])

print("\nAnalysis complete")