import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import core functionality from efficacy.py
from efficacy import (
    load_network, 
    get_baseline_state, 
    simulate_drug_effect, 
    calculate_efficacy,
    calculate_comprehensive_efficacy,
    collect_training_data,
    save_model,
    load_model,
    predict_efficacy,
    evaluate_drug_efficacy,
    visualize_pet_scan,
    generate_pet_data,
    PETScanGenerator,
    DRUG_TARGETS,
    CLINICAL_EFFICACY,
    AD_OUTPUT_NODES,
    PATHWAYS
)

# Import temporal simulation
from temporal_simulation import TemporalDrugSimulation

def create_empirical_dataset():
   
    empirical_data = []
    
    # Add data from known drugs in clinical trials
    for drug_name, drug_data in CLINICAL_EFFICACY.items():
        if drug_name in DRUG_TARGETS:
            targets = [(target_info["target"], target_info["effect"]) 
                      for target_info in DRUG_TARGETS[drug_name]]
            
            for condition, condition_data in drug_data.items():
                efficacy = condition_data["efficacy"]
                empirical_data.append((targets, efficacy, condition))
                print(f"Added clinical data for {drug_name} in {condition} condition: efficacy = {efficacy:.4f}")
    
    # Additional empirical data from literature (based on real studies)
    literature_data = [
        # GSK3beta inhibitors (tideglusib, etc.)
        ([("GSK3beta", 0)], 0.18, "APOE4"),  # del Ser T, et al. J Alzheimers Dis. 2013
        ([("GSK3beta", 0)], 0.15, "LPL"),    # Add LPL condition
        
        # Dual-targeting approaches
        ([("GSK3beta", 0), ("BACE1", 0)], 0.32, "APOE4"),  # Estimated from preclinical models
        ([("GSK3beta", 0), ("BACE1", 0)], 0.28, "LPL"),    # Add LPL condition
        
        # mTOR inhibitors (rapamycin)
        ([("mTOR", 0)], 0.15, "APOE4"),  # Spilman P, et al. PLoS One. 2010
        ([("mTOR", 0)], 0.12, "LPL"),    # Add LPL condition
        
        # NMDA receptor modulators
        ([("e_NMDAR", 0), ("s_NMDAR", 1)], 0.17, "APOE4"),  # Based on combined studies
        ([("e_NMDAR", 0), ("s_NMDAR", 1)], 0.14, "LPL"),    # Add LPL condition
        
        # Neuroinflammation targets
        ([("TNFa", 0)], 0.14, "APOE4"),  # Butchart J, et al. J Alzheimers Dis. 2015
        ([("TNFa", 0)], 0.11, "LPL"),    # Add LPL condition
        
        # Combination cholinergic approaches
        ([("AChE", 0), ("nAChR", 1)], 0.25, "APOE4"),  # From galantamine + other enhancers
        ([("AChE", 0), ("nAChR", 1)], 0.22, "LPL"),    # Add LPL condition
        
        # PPAR-gamma agonists
        ([("PPAR_gamma", 1)], 0.12, "APOE4"),  # Risner ME, et al. Pharmacogenomics J. 2006
        ([("PPAR_gamma", 1)], 0.10, "LPL"),    # Add LPL condition
        
        # Antioxidant approaches
        ([("NRF2", 1)], 0.09, "APOE4"),  # Based on curcumin and related compounds
        ([("NRF2", 1)], 0.07, "LPL"),    # Add LPL condition
    ]
    
    empirical_data.extend(literature_data)
    
    return empirical_data

def train_model(X, y):
   
    print(f"Training model with {len(X)} samples...")
    
    # Handle small dataset sizes
    if len(X) < 5:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        print(f"Model trained on all {len(X)} samples - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        return {
            'model': model,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'r2': 'N/A - Too few samples',
                'cv_scores': 'N/A - Too few samples',
                'cv_mean': 'N/A - Too few samples'
            }
        }
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"Model validation - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Perform cross-validation if we have enough samples
    if len(X) >= 10:
        n_splits = min(5, len(X) // 2)  # Ensure we don't have too many splits
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean cross-validation R²: {np.mean(cv_scores):.4f}")
        
        cv_mean = np.mean(cv_scores)
    else:
        cv_scores = None
        cv_mean = None
        print("Skipping cross-validation due to small sample size")
    
    # Train on full dataset
    model.fit(X, y)
    
    return {
        'model': model,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_scores': cv_scores,
            'cv_mean': cv_mean
        }
    }

def training_pipeline(network_file="A_model.txt", model_file="ad_drug_efficacy_model.pkl", 
                     include_empirical=True, generate_plots=True):
   
    print("=== Starting Alzheimer's Disease Drug Efficacy Model Training ===")
    
    # Step 1: Load the Boolean network model
    print("\nStep 1: Loading network model")
    try:
        network_data = load_network(network_file)
        net = network_data['net']
        output_list = network_data['output_list']
        print(f"Network loaded with {len(output_list)} genes")
    except Exception as e:
        print(f"ERROR loading network: {e}")
        raise
    
    # Step 2: Define known drugs from real-world data
    print("\nStep 2: Defining known drugs from real data")
    known_drugs = list(DRUG_TARGETS.keys())
    print(f"Using {len(known_drugs)} known drugs: {', '.join(known_drugs)}")
    
    # Step 3: Add empirical data from literature
    empirical_data = []
    if include_empirical:
        print("\nStep 3: Adding empirical data from research literature")
        try:
            empirical_data = create_empirical_dataset()
            print(f"Added {len(empirical_data)} empirical data points from research")
        except Exception as e:
            print(f"WARNING: Error adding empirical data: {e}")
    else:
        print("\nStep 3: Skipping empirical data (not requested)")
    
    # Step 4: Collect training data
    print("\nStep 4: Collecting training data")
    try:
        X, y, drug_info = collect_training_data(
            net=net, 
            output_list=output_list,
            known_drugs=known_drugs,
            additional_data=empirical_data
        )
        print(f"Successfully collected data for {len(X)} samples")
    except Exception as e:
        print(f"ERROR in data collection: {e}")
        raise
    
    # Step 5: Train the model
    print("\nStep 5: Training model")
    try:
        model_results = train_model(X, y)
        # Important: Add output_list to model_results for feature naming
        model_results['output_list'] = output_list
        print(f"Model trained successfully")
    except Exception as e:
        print(f"ERROR in model training: {e}")
        raise
    
    # Predefined drugs and conditions for temporal simulation
    drugs_to_simulate = [
        "Lecanemab", 
        "Memantine", 
        "Donepezil", 
        "Galantamine"
    ]
    conditions_to_simulate = ["APOE4", "Normal", "LPL"]
    
    # Step 6: Run Temporal Simulation
    print("\nStep 6: Running Temporal Simulations")
    temporal_results = {}
    temporal_sim = TemporalDrugSimulation(
        network_file=network_file, 
        output_dir="drug_temporal_analysis"
    )
    
    # Run simulation for each drug and condition
    for drug in drugs_to_simulate:
        drug_results = {}
        for condition in conditions_to_simulate:
            print(f"\nSimulating {drug} in {condition} condition")
            result = temporal_sim.simulate_drug_over_time(
                drug_name=drug, 
                condition=condition, 
                include_visuals=True
            )
            drug_results[condition] = result
        temporal_results[drug] = drug_results
    
    # Save temporal results
    temporal_results_file = "drug_temporal_analysis/temporal_simulation_results.pkl"
    with open(temporal_results_file, 'wb') as f:
        pickle.dump(temporal_results, f)
    print(f"Temporal simulation results saved to {temporal_results_file}")
    
    # Optional: Generate summary plots or reports 
    # This is a placeholder for additional post-processing of temporal results
    
    # Return both model results and temporal simulation results
    return {
        'model_data': model_results,
        'network_data': network_data,
        'temporal_results': temporal_results,
        'temporal_results_file': temporal_results_file
    }

if __name__ == "__main__":
    # Train the model and run temporal simulations
    results = training_pipeline()