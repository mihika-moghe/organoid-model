import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import boolnet
import matplotlib.pyplot as plt



################ convert to attractor matrix and scores function
def calc_attr_score(attractors, output_list):
    attr_num = len(attractors['attractors'])
    attr_mat = np.zeros((len(output_list), attr_num))
    attr_ratio = np.zeros(attr_num)

    for attr_indx in range(attr_num):
        attractor = attractors['attractors'][attr_indx]
        
        if 'involvedStates' in attractor:
            states = np.array(attractor['involvedStates'])
            if states.ndim == 1:
                states = states.reshape(1, -1)
        else:
            states = np.array(attractor.get('sequence', []))
   
        
        for i, node in enumerate(output_list):
            try: 
            
                if 'genes' in attractors['stateInfo']:
                        node_idx = attractors['stateInfo']['genes'].index(node)
                else:
                    node_idx = output_list.index(node)
                
                attr_mat[i, attr_indx] = np.mean(states[:, node_idx])
            
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not find index for node {node}")
                attr_mat[i, attr_indx] = 0
       
        total_states = len(attractors['stateInfo'].get('table', [1]))
        basin_size = attractor.get('basinSize', 1)
        
        if basin_size is None:
            basin_size = 1
        
        attr_ratio[attr_indx] = (100 * basin_size / total_states)
    
    total_ratio = np.sum(attr_ratio)
    if total_ratio > 0:
        attr_ratio = attr_ratio / total_ratio
    else:
        attr_ratio = np.ones_like(attr_ratio) / len(attr_ratio)
        
    attr_ratio_mat = np.tile(attr_ratio, (len(output_list), 1))
    node_activity = np.sum(np.round(100 * attr_mat * attr_ratio_mat, 2), axis=1)

    return node_activity

################ perturbation analysis

def process_perturbation(node, net, off_node, on_node):
    
    try:
        genes_off = [node] if off_node is None else [off_node, node]
        genes_on = [on_node] if on_node is not None else None
        
        attractors = boolnet.get_attractors(
            type="synchronous",
            method="random",
            start_states=1000000,
            genes_off=genes_off,
            genes_on=genes_off,
        )
    
    except Exception as e:
        print(f"Error processing node {node}: {str(e)}")
        return None


def pert_single(cand_node, net, output_list, off_node=None, on_node=None):
    pert_result = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(lambda node: process_perturbation(node), cand_node)
    
    for attractors in results:
        if attractors is not None:
            result = calc_attr_score(attractors, output_list)
            pert_result.append(result)

    pert_result = pd.DataFrame(pert_result)
    pert_result.insert(0, 'Base', np.zeros(len(pert_result)))  
    pert_result.loc[-1] = np.ones(len(output_list)) * 100   
    pert_result.index = range(len(pert_result))   
    
    return pert_result

def pert_double(cand_node, net, output_list, off_node=None, on_node=None):
    pert_result = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(lambda node: process_perturbation(node), cand_node)

    for attractors in results:
        if attractors is not None:
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