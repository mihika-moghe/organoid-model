"""
Integration script to use baseline results from the main simulation file
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boolnet
import time
from PET import PET

import main 

def get_baseline_from_main():
    """
    Get baseline states from the main analysis file.
    Returns dictionary of conditions to attractors.
    """
    # This assumes your main file has these global variables
    # If your main file has a different structure, modify accordingly
    try:
        baselines = {
            "Normal": main.attractors_Normal,
            "APOE4": main.attractors_APOE4,
            "LPL": main.attractors_LPL
        }
        print("Successfully loaded baseline states from main analysis file")
        return baselines
    except AttributeError:
        print("WARNING: Could not load attractors from main analysis file")
        print("Will create fresh baseline simulations instead")
        return None

def load_perturbation_data():
    """
    Load perturbation data from saved files.
    """
    try:
        # Load APOE4 perturbation results
        apoe4_data = pd.read_csv("APOE4_pert_res.txt", sep="\t", index_col=0)
        print(f"Loaded APOE4 perturbation data with {apoe4_data.shape[0]} rows")
        
        # Load LPL perturbation results
        lpl_data = pd.read_csv("LPL_pert_res.txt", sep="\t", index_col=0)
        print(f"Loaded LPL perturbation data with {lpl_data.shape[0]} rows")
        
        return {
            "APOE4": apoe4_data,
            "LPL": lpl_data
        }
    except FileNotFoundError:
        print("WARNING: Could not load perturbation data files")
        return None

def convert_perturbation_to_network_state(pert_data, column=0):
    """
    Convert perturbation data to network state.
    
    Args:
        pert_data: DataFrame with perturbation results
        column: Column index to use (default is first column)
    
    Returns:
        Dictionary with node states
    """
    # Default threshold for activation (>50% = active)
    threshold = 50.0
    
    # Create network state dictionary
    network_state = {}
    
    # If data is empty, return empty dictionary
    if pert_data is None or pert_data.empty:
        return network_state
    
    # Get column name (if using index, get first column)
    col_name = pert_data.columns[column] if column < len(pert_data.columns) else pert_data.columns[0]
    
    # Convert percentage values to binary states
    for node, row in pert_data.iterrows():
        value = row[col_name]
        # Nodes with >50% activation are considered "on" (1), others are "off" (0)
        network_state[node] = 1 if value > threshold else 0
    
    print(f"Converted perturbation data to network state with {len(network_state)} nodes")
    return network_state

def generate_pet_from_main_results(drug_name, drug_targets, condition="APOE4", output_dir=None):
    """
    Generate PET scans using baseline results from the main file
    and simulating drug effects on them.
    
    Args:
        drug_name: Name of the drug
        drug_targets: List of (target, effect) tuples
        condition: Patient condition (Normal, APOE4, LPL)
        output_dir: Directory to save visualizations
    
    Returns:
        Dictionary with paths to visualizations
    """
    # Set output directory
    if output_dir is None:
        output_dir = f"pet_results_main/{drug_name.lower()}_{condition.lower()}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load network for simulation
    print("Loading network...")
    net = boolnet.load_network("A_model.txt")
    output_list = net['genes']
    
    # Try to get baseline from main file
    baselines = get_baseline_from_main()
    baseline_attractors = None
    
    if baselines and condition in baselines:
        baseline_attractors = baselines[condition]
        print(f"Using baseline from main file for {condition} condition")
    else:
        # If not available, simulate fresh
        print(f"Simulating fresh baseline for {condition}...")
        
        # Set up condition-specific parameters
        genes_on = []
        genes_off = []
        
        if condition == "APOE4":
            genes_on = ["APOE4"]
        elif condition == "Normal":
            genes_off = ["APOE4"]
        elif condition == "LPL":
            genes_off = ["APOE4", "LPL"]
        
        # Run attractor analysis
        baseline_attractors = boolnet.get_attractors(
            net,
            type="synchronous",
            method="random",
            start_states=100000,   
            genes_on=genes_on,
            genes_off=genes_off
        )
    
    # Try to load perturbation data
    pert_data = load_perturbation_data()
    baseline_state = None
    
    if pert_data and condition in pert_data:
        # Convert perturbation data to network state
        baseline_state = convert_perturbation_to_network_state(pert_data[condition])
        print(f"Using perturbation data for {condition} baseline state")
    else:
        # If perturbation data not available, use attractor state
        print("Perturbation data not available, using attractor state instead")
        
        # Extract state from attractor
        if baseline_attractors and 'attractors' in baseline_attractors:
            first_attractor = baseline_attractors['attractors'][0]
            first_state = first_attractor['involvedStates'][0]
            baseline_state = {output_list[i]: first_state[i] for i in range(len(first_state))}
        else:
            print("ERROR: No valid baseline state available")
            return None
    
    # Simulate drug effect
    print(f"Simulating effect of {drug_name}...")
    
    # Set up condition-specific parameters
    genes_on = []
    genes_off = []
    
    if condition == "APOE4":
        genes_on = ["APOE4"]
    elif condition == "Normal":
        genes_off = ["APOE4"]
    elif condition == "LPL":
        genes_off = ["APOE4", "LPL"]
    
    # Add drug targets
    for target, effect in drug_targets:
        if target in output_list:
            if effect == 1:
                genes_on.append(target)
            else:
                genes_off.append(target)
    
    # Run simulation with drug
    drug_attractors = boolnet.get_attractors(
        net,
        type="synchronous",
        method="random",
        start_states=100000,   
        genes_on=genes_on,
        genes_off=genes_off
    )
    
    # Extract drug state
    if drug_attractors and 'attractors' in drug_attractors:
        first_attractor = drug_attractors['attractors'][0]
        first_state = first_attractor['involvedStates'][0]
        drug_state = {output_list[i]: first_state[i] for i in range(len(first_state))}
    else:
        print("ERROR: Drug simulation produced no valid state")
        return None
    
    # Initialize PET generator
    pet_generator = PET()
    
    # Generate PET visualizations
    print(f"Generating PET visualizations for {drug_name} in {condition}...")
    
    # Generate baseline PET (using baseline_state from perturbation data)
    amyloid_baseline, anatomy = pet_generator.generate_pet_from_network_state(
        baseline_state, 'amyloid', condition
    )
    
    # Generate post-treatment PET (using drug_state from simulation)
    amyloid_post, _ = pet_generator.generate_pet_from_network_state(
        drug_state, 'amyloid', condition
    )
    
    # Generate visualizations
    visualization_paths = {}
    
    # Generate baseline visualization
    baseline_file = os.path.join(output_dir, "amyloid_baseline.png")
    visualization_paths['amyloid_baseline'] = pet_generator.create_clinical_pet_visualization(
        amyloid_baseline, anatomy, 'amyloid', baseline_file, drug_name, is_baseline=True
    )
    
    # Generate post-treatment visualization
    post_file = os.path.join(output_dir, "amyloid_post.png")
    visualization_paths['amyloid_post'] = pet_generator.create_clinical_pet_visualization(
        amyloid_post, anatomy, 'amyloid', post_file, drug_name, is_baseline=False
    )
    
    # Generate comparison visualization
    comparison_file = os.path.join(output_dir, "amyloid_comparison.png")
    visualization_paths['amyloid_comparison'] = pet_generator.create_clinical_pet_visualization(
        amyloid_baseline, anatomy, 'amyloid', comparison_file, drug_name, paired_volume=amyloid_post
    )
    
    # Generate difference map
    diff_file = os.path.join(output_dir, "amyloid_difference.png")
    visualization_paths['amyloid_difference'] = pet_generator.create_amyloid_difference_map(
        amyloid_baseline, amyloid_post, diff_file, drug_name
    )
    
    print(f"Created PET visualizations in {output_dir}")
    return visualization_paths

def test_all_drugs_with_main_data():
    """Test PET visualization with all FDA-approved drugs using main file data"""
    # List of all drugs to test
    drugs = {
        "Lecanemab": [
            ("APP", 0),    # Inhibit APP processing
            ("BACE1", 0)   # Indirect reduction of BACE1
        ],
        "Donepezil": [
            ("AChE", 0)    # Acetylcholinesterase inhibitor
        ],
        "Memantine": [
            ("e_NMDAR", 0) # NMDA receptor antagonist
        ],
        "Galantamine": [
            ("AChE", 0),   # Acetylcholinesterase inhibitor
            ("nAChR", 1)   # Nicotinic receptor modulator  
        ]
    }
    
    # All conditions to test
    conditions = ["APOE4", "Normal", "LPL"]
    
    results = {}
    
    for drug_name, drug_targets in drugs.items():
        drug_results = {}
        
        for condition in conditions:
            print(f"\n=== Testing {drug_name} in {condition} condition ===")
            
            visualizations = generate_pet_from_main_results(
                drug_name, drug_targets, condition
            )
            
            if visualizations:
                drug_results[condition] = visualizations
                print(f"Successfully generated PET scans for {drug_name} in {condition}")
            else:
                print(f"Failed to generate PET scans for {drug_name} in {condition}")
        
        results[drug_name] = drug_results
    
    return results

if __name__ == "__main__":
    # Test with main file baseline data
    print("\n=== Testing FDA-approved drugs with main file data ===")
    results = test_all_drugs_with_main_data()
    
    print("\nAll tests complete. PET visualizations saved to pet_results_main directory.")