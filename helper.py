import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import networkx as nx
import boolnet

################ convert to attractor matrix and scores function
def calc_attr_score(attractors, output_list):
    attr_num = len(attractors['attractors'])
    attr_mat = np.zeros(len(output_list), attr_num)
    attr_ratio = np.zeros(attr_num)

    for attr_indx in range(attr_num):
        attr_seq = np.array(attractors['attractors'][attr_indx]['sequence'])
        
        for i, node in enumerate(output_list):
            node_idx = attractors['node_names'].index(node)
            attr_mat[i, attr_indx] = np.mean(attr_seq[:, node_idx])
       
        total_states = len(attractors['stateInfo']['table'])
        basin_size = attractors['attractors'][attr_indx]['basinSize']
        attr_ratio[attr_indx] = (100 * basin_size / total_states)

        attr_ratio = attr_ratio / np.sum(attr_ratio)
        attr_ratio_mat = np.tile(attr_ratio, (len(output_list), 1))
        node_activity = np.sum(np.round(100 * attr_mat * attr_ratio_mat, 2), axis=1)
    
        return node_activity

################ single off-perturbation analysis

def pert_double(cand_node, net, output_list, off_node = None, on_node = None):
    if off_node is None and on_node is None:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_off=cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)
    
    elif off_node is None and on_node is not None:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_on=[on_node], genes_off=cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)
    
    elif off_node is not None and on_node is None:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_off=[off_node] + cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)
    
    else:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_on=[on_node], genes_off=[off_node] + cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)

    pert_result = pd.DataFrame(pert_result)
    pert_result.insert(0, 'Base', np.zeros(len(pert_result)))  
    pert_result.loc[-1] = np.ones(len(output_list)) * 100   
    pert_result.index = range(len(pert_result))   
    
    return pert_result

def pert_single(cand_node, net, output_list, off_node=None, on_node=None):
    if off_node is None and on_node is None:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_off=cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)
    
    elif off_node is None and on_node is not None:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_on=[on_node], genes_off=cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)
    
    elif off_node is not None and on_node is None:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_off=[off_node] + cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)
    
    else:
        pert_result = []
        with ProcessPoolExecutor() as executor:
            for idx in range(len(cand_node)):
                attractors = boolnet.get_attractors(net, genes_on=[on_node], genes_off=[off_node] + cand_node[idx])
                result = calc_attr_score(attractors, output_list)
                pert_result.append(result)

    pert_result = pd.DataFrame(pert_result)
    pert_result.insert(0, 'Base', np.zeros(len(pert_result)))  
    pert_result.loc[-1] = np.ones(len(output_list)) * 100   
    pert_result.index = range(len(pert_result))   
    
    return pert_result

def plotting (data, title):
     labels = data.columns.tolist()
     num_vars = len(labels)
     
     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
     angles += angles[:1]
     
     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
     ax.set_theta_offset(np.pi / 2)
     ax.set_theta_direction(-1)
     
     ax.set_xticks(angles[:-1])
     ax.set_xticklabels(labels)
     
     ax.set_rscale("linear")
     ax.set_rlabel_position(0)
     plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
     plt.ylim(0, 100)
     
     for i in range(2, data.shape[0]):   
         values = data.iloc[i].tolist()
         values += values[:1]
         ax.plot(angles, values, linewidth=2, linestyle="solid", label=data.index[i])
         ax.fill(angles, values, alpha=0.25)
    
     plt.title(title, size=15, color="black", y=1.1)
     ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
     
     plt.show()

