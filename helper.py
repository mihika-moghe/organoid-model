import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import boolnet
from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt



################ convert to attractor matrix and scores function
def calc_attr_score(attractors: dict, output_list: List[str]) -> np.ndarray:

    try:
        if not attractors.get('attractors'):
            print("No attractors found")
            return np.zeros(len(output_list))
            
        attr_num = len(attractors['attractors'])
        attr_mat = np.zeros((len(output_list), attr_num))
        attr_ratio = np.zeros(attr_num)
        
        print(f"\nCalculating scores for {attr_num} attractors and {len(output_list)} output nodes")
        print(f"StateInfo keys: {attractors['stateInfo'].keys()}")
        print(f"Network genes: {attractors['stateInfo']['genes']}")
        
        num_genes = len(attractors['stateInfo']['genes'])
        num_fixed = sum(1 for x in attractors['stateInfo']['fixedGenes'] if x != -1)
        total_states = 2 ** (num_genes - num_fixed)
        
        for attr_idx in range(attr_num):
            attractor = attractors['attractors'][attr_idx]
            print(f"\nProcessing attractor {attr_idx + 1}")
            print(f"Attractor keys: {attractor.keys()}")
            
            if 'involvedStates' in attractor:
                states = np.array(attractor['involvedStates'])
                print(f"Raw states shape: {states.shape}")
                
                if states.ndim == 1:
                    states = states.reshape(1, -1)
                elif states.ndim > 2:
                    states = states.reshape(states.shape[0], -1)
                    
                print(f"Processed states shape: {states.shape}")
                
                if states.shape[1] != num_genes:
                    print(f"Warning: State dimension mismatch. Expected {num_genes}, got {states.shape[1]}")
                    if states.shape[1] > num_genes:
                        states = states[:, :num_genes]
                    else:
                        pad_width = ((0, 0), (0, num_genes - states.shape[1]))
                        states = np.pad(states, pad_width, mode='constant')
            else:
                print("Warning: No involvedStates found, using empty array")
                states = np.zeros((1, num_genes))
            
            for i, node in enumerate(output_list):
                try:
                    node_idx = attractors['stateInfo']['genes'].index(node)
                    print(f"Processing node {node} at index {node_idx}")
                    attr_mat[i, attr_idx] = np.mean(states[:, node_idx])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not find index for node {node}: {str(e)}")
                    attr_mat[i, attr_idx] = 0
            
            basin_size = len(states) if 'basinSize' not in attractor else attractor.get('basinSize', len(states))
            if basin_size is None or basin_size == 0:
                basin_size = len(states)
                
            print(f"Basin size: {basin_size}, Total states: {total_states}")
            attr_ratio[attr_idx] = (100.0 * basin_size / total_states)
            
        total_ratio = np.sum(attr_ratio)
        if total_ratio > 0:
            attr_ratio = attr_ratio / total_ratio
        else:
            attr_ratio = np.ones_like(attr_ratio) / len(attr_ratio)
            
        print("\nFinal attr_ratio:", attr_ratio)
        print("Attr_mat shape:", attr_mat.shape)
            
        attr_ratio_mat = np.tile(attr_ratio, (len(output_list), 1))
        scores = np.sum(np.round(100 * attr_mat * attr_ratio_mat, 2), axis=1)
        
        print("Final scores shape:", scores.shape)
        return scores
        
    except Exception as e:
        print(f"Error calculating attractor scores: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(output_list))

def process_perturbation(node: str, network: dict, output_list: List[str], 
                        off_node: Optional[List[str]] = None, 
                        on_node: Optional[List[str]] = None) -> np.ndarray:
   
    try:
        print(f"Processing perturbation for node: {node}")
        
        genes_off = [node]
        if off_node:
            genes_off.extend(off_node)
            
        genes_on = on_node if on_node else []
        
        print(f"Genes OFF: {genes_off}")
        print(f"Genes ON: {genes_on}")
        
        attractors = boolnet.get_attractors(
            network,
            type="synchronous", 
            method="random",
            start_states=1000000,  
            genes_off=genes_off,
            genes_on=genes_on
        )
        
        if not attractors:
            print(f"No attractors found for node {node}")
            return np.zeros(len(output_list))
            
        return calc_attr_score(attractors, output_list)
        
    except Exception as e:
        print(f"Error processing node {node}: {str(e)}")
        print(f"Network genes: {network['genes']}")
        print(f"Output list: {output_list}")
        return np.zeros(len(output_list))

def pert_single(cand_nodes: List[str], network: dict, output_list: List[str],
                off_node: Optional[List[str]] = None, 
                on_node: Optional[List[str]] = None) -> pd.DataFrame:
     
    print("\nStarting single perturbation analysis...")
    print(f"Candidate nodes: {cand_nodes}")
    
    pert_result = []
    for node in cand_nodes:
        try:
            result = process_perturbation(node, network, output_list, off_node, on_node)
            pert_result.append(result)
            print(f"Completed perturbation for {node}")
        except Exception as e:
            print(f"Error in perturbation of {node}: {str(e)}")
            pert_result.append(np.zeros(len(output_list)))
    
    pert_result = pd.DataFrame(pert_result)
    
    baseline = pd.DataFrame([np.ones(len(output_list)) * 100])  
    pert_result = pd.concat([baseline, pert_result], ignore_index=True) 
    pert_result.insert(0, 'Base', np.zeros(len(pert_result))) 
    
    print("Completed single perturbation analysis")
    return pert_result

def pert_double(cand_pairs: List[Tuple[str, str]], network: dict, 
                output_list: List[str],
                off_node: Optional[List[str]] = None, 
                on_node: Optional[List[str]] = None) -> pd.DataFrame:
     
    print("\nStarting double perturbation analysis...")
    print(f"Candidate pairs: {cand_pairs}")
    
    pert_result = []
    for node1, node2 in cand_pairs:
        try:
            genes_off = [node1, node2]
            if off_node:
                genes_off.extend(off_node)
                
            result = process_perturbation(node1, network, output_list, 
                                        genes_off, on_node)
            pert_result.append(result)
            print(f"Completed perturbation for pair {node1}, {node2}")
        except Exception as e:
            print(f"Error in perturbation of pair {node1}, {node2}: {str(e)}")
            pert_result.append(np.zeros(len(output_list)))
    
    pert_result = pd.DataFrame(pert_result)
    
    baseline = pd.DataFrame([np.ones(len(output_list)) * 100])  
    pert_result = pd.concat([baseline, pert_result], ignore_index=True)  
    pert_result.insert(0, 'Base', np.zeros(len(pert_result)))  
    
    print("Completed double perturbation analysis")
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