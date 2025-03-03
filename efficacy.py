import pandas as pd
import numpy as np
import boolnet
import time
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from helper import calc_attr_score, pert_single, pert_double  

# Define known drug targets for common AD drugs
DRUG_TARGETS = {
    "Lecanemab": ["APP", "BACE1"],   
    "Memantine": ["e_NMDAR"],        
    "Donepezil": ["AChE"],           
    "Galantamine": ["AChE", "nAChR"]   
}

# Define key output nodes associated with AD pathology
AD_OUTPUT_NODES = [
    "APP", "BACE1", "MAPT", "GSK3beta", "Ca_ion", "p53", 
    "CASP3", "LC3", "PTEN", "Bcl2", "mTOR", "Cholesterol"
]

# Define pathway groupings for analysis
PATHWAYS = {
    'Amyloid': ['APP', 'BACE1'],
    'Tau': ['MAPT', 'GSK3beta'],
    'Apoptosis': ['CASP3', 'p53', 'Bcl2', 'BAX'],
    'Autophagy': ['LC3', 'mTOR', 'beclin1'],
    'Lipid': ['Cholesterol', 'LPL', 'SREBP2'],
    'Synaptic': ['Ca_ion', 'e_NMDAR', 's_NMDAR']
}

def load_network(network_file="A_model.txt"):
    print("Loading network...")
    net = boolnet.load_network(network_file)
    output_list = net['genes'] 
    print(f"Loaded network with {len(net['genes'])} genes")
    print(f"First gene {net['genes'][0]} has inputs: {net['interactions'][0]['input']}")
    
    return {
        'net': net,
        'output_list': output_list
    }

def get_baseline_state(net, output_list, condition="Normal"):
   
    print(f"\nGetting baseline for {condition} condition...")
    start_time = time.time()
    
    # Set up condition-specific parameters
    genes_on = []
    genes_off = []
    
    if condition == "Normal":
        genes_off = ["APOE4"]
    elif condition == "APOE4":
        genes_on = ["APOE4"]
    elif condition == "LPL":
        genes_off = ["APOE4", "LPL"]
    
    # Run attractor analysis
    attractors = boolnet.get_attractors(
        net,
        type="synchronous",
        method="random",
        start_states=100000,   
        genes_on=genes_on,
        genes_off=genes_off
    )
    
    print(f"{condition} analysis time: {time.time() - start_time:.2f}s")
    
    if not attractors.get('attractors'):
        raise ValueError(f"No attractors found in {condition} analysis")
    
    # Calculate phenotype scores using the original helper function
    pheno_scores = calc_attr_score(attractors, output_list)
    print(f"{condition} phenotype scores calculated")
    
    return attractors

def simulate_drug_effect(net, output_list, drug_name=None, drug_targets=None, condition="APOE4"):
   
    # Set up simulation parameters
    genes_on = []
    genes_off = []
    
    # Configure disease condition
    if condition == "Normal":
        genes_off = ["APOE4"]
    elif condition == "APOE4":
        genes_on = ["APOE4"]
    elif condition == "LPL":
        genes_off = ["APOE4", "LPL"]
    
    # Add drug targets
    if drug_name and drug_name in DRUG_TARGETS:
        # For known drugs, use predefined targets
        targets = DRUG_TARGETS[drug_name]
        for target in targets:
            # Default to inhibition for most AD drugs
            genes_off.append(target)
    elif drug_targets:
        # For custom drug targets
        for target, effect in drug_targets:
            if effect == 1:
                genes_on.append(target)
            else:
                genes_off.append(target)
    else:
        raise ValueError("Either drug_name or drug_targets must be provided")
    
    # Run simulation with drug effects
    print(f"\nSimulating {drug_name if drug_name else 'custom drug'} on {condition} condition...")
    start_time = time.time()
    
    drug_attractors = boolnet.get_attractors(
        net,
        type="synchronous",
        method="random",
        start_states=100000,
        genes_on=genes_on,
        genes_off=genes_off
    )
    
    print(f"Drug simulation time: {time.time() - start_time:.2f}s")
    
    if not drug_attractors.get('attractors'):
        raise ValueError(f"No attractors found in drug simulation")
            
    return drug_attractors

def calculate_efficacy(baseline_attractors, drug_attractors, output_list):
   
    # Extract state matrices
    baseline_states = baseline_attractors['attractors']
    drug_states = drug_attractors['attractors']
    
    # Get relevant output nodes
    output_indices = [output_list.index(node) for node in AD_OUTPUT_NODES if node in output_list]
    
    # Calculate state changes in disease-relevant nodes
    baseline_scores = []
    for attractor in baseline_states:
        # Average state of each attractor
        avg_state = np.mean(attractor['involvedStates'], axis=0)
        # Focus on AD-relevant nodes
        baseline_scores.append([avg_state[i] for i in output_indices])
    
    drug_scores = []
    for attractor in drug_states:
        avg_state = np.mean(attractor['involvedStates'], axis=0)
        drug_scores.append([avg_state[i] for i in output_indices])
    
    # Average scores across all attractors
    avg_baseline = np.mean(baseline_scores, axis=0)
    avg_drug = np.mean(drug_scores, axis=0)
    
    # Calculate Euclidean distance between states
    # Normalize to [0,1] range where 1 means maximum effect
    state_change = np.linalg.norm(avg_drug - avg_baseline)
    max_possible_change = np.sqrt(len(output_indices))  # Maximum possible change
    
    # Normalize state change to get efficacy score
    efficacy_score = min(state_change / max_possible_change, 1.0)
    
    # Calculate node-specific changes
    node_changes = {}
    for i, node in enumerate(AD_OUTPUT_NODES):
        if node in output_list:
            idx = output_list.index(node)
            if idx in output_indices:
                pos = output_indices.index(idx)
                node_changes[node] = avg_drug[pos] - avg_baseline[pos]
    
    # Calculate pathway-level changes
    pathway_changes = {}
    for pathway, genes in PATHWAYS.items():
        # Get changes for genes that exist in our results
        pathway_genes = [gene for gene in genes if gene in node_changes]
        if pathway_genes:
            avg_change = np.mean([node_changes[gene] for gene in pathway_genes])
            pathway_changes[pathway] = avg_change
    
    return {
        'efficacy_score': efficacy_score,
        'node_changes': node_changes,
        'pathway_changes': pathway_changes,
        'avg_baseline': avg_baseline,
        'avg_drug': avg_drug,
        'output_indices': output_indices
    }

def extract_features(drug_targets, output_list, condition="APOE4"):
   
    # Create a binary vector representing which nodes are targeted
    feature_vector = [0] * len(output_list)
    
    # Mark targets in the feature vector
    for target, effect in drug_targets:
        if target in output_list:
            idx = output_list.index(target)
            # Use +1 for activation, -1 for inhibition
            feature_vector[idx] = 1 if effect == 1 else -1
    
    # Add condition as a feature
    condition_features = [0, 0, 0]  # [Normal, APOE4, LPL]
    if condition == "Normal":
        condition_features[0] = 1
    elif condition == "APOE4":
        condition_features[1] = 1
    elif condition == "LPL":
        condition_features[2] = 1
    
    # Combine all features
    return feature_vector + condition_features

def collect_training_data(net, output_list, known_drugs=None, additional_data=None):
   
    X = []
    y = []
    drug_info = []  # Store additional information about each data point
    
    # Process known drugs
    if known_drugs:
        for drug in known_drugs:
            if drug in DRUG_TARGETS:
                # Convert drug targets to (target, effect) format
                targets = [(target, 0) for target in DRUG_TARGETS[drug]]  # Assume inhibition
                
                # Get baseline for AD condition (APOE4)
                baseline = get_baseline_state(net, output_list, "APOE4")
                
                # Simulate drug effect
                drug_state = simulate_drug_effect(net, output_list, drug_name=drug, condition="APOE4")
                
                # Calculate efficacy
                efficacy = calculate_efficacy(baseline, drug_state, output_list)
                
                # Extract features and add to training data
                features = extract_features(targets, output_list, "APOE4")
                X.append(features)
                y.append(efficacy['efficacy_score'])
                drug_info.append({
                    'name': drug,
                    'targets': targets,
                    'condition': "APOE4",
                    'node_changes': efficacy['node_changes'],
                    'pathway_changes': efficacy.get('pathway_changes', {})
                })
    
    # Add additional training data if provided
    if additional_data:
        for targets, efficacy_score, condition in additional_data:
            features = extract_features(targets, output_list, condition)
            X.append(features)
            y.append(efficacy_score)
            drug_info.append({
                'name': f"Custom_{len(drug_info)}",
                'targets': targets,
                'condition': condition,
                'efficacy': efficacy_score
            })
    
    return np.array(X), np.array(y), drug_info

def train_model(X, y):
   
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
    
    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean cross-validation R²: {np.mean(cv_scores):.4f}")
    
    # Train on full dataset
    model.fit(X, y)
    
    return {
        'model': model,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores)
        }
    }

def save_model(model_data, filename="drug_efficacy_model.pkl"):
   
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filename}")

def load_model(filename="drug_efficacy_model.pkl"):
   
    try:
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Loaded model from {filename}")
        return model_data
    except:
        print(f"Could not load model from {filename}")
        return None

def plot_training_results(X, y, model, drug_info, output_dir="model_analysis"):
   
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Predicted vs Actual Efficacy
    y_pred = model['model'].predict(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.7)
    
    # Add drug names as annotations
    for i, drug in enumerate(drug_info):
        plt.annotate(drug['name'], (y[i], y_pred[i]), fontsize=9)
    
    # Add perfect prediction line
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Efficacy')
    plt.ylabel('Predicted Efficacy')
    plt.title('Model Predictions vs Actual Efficacy')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/predicted_vs_actual.png", dpi=300, bbox_inches='tight')
    
    # 2. Feature Importance
    if hasattr(model['model'], 'feature_importances_'):
        importances = model['model'].feature_importances_
        
        # Get top 20 features
        n_features = min(20, len(importances))
        indices = np.argsort(importances)[-n_features:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(n_features), importances[indices])
        
        # Create feature names
        feature_names = []
        for i in range(len(X[0]) - 3):  # Exclude condition features
            feature_names.append(f"Target_{i}")
        feature_names.extend(['Normal', 'APOE4', 'LPL'])
        
        plt.yticks(range(n_features), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Features by Importance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
    
    # 3. Residual Analysis
    residuals = y - y_pred
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Efficacy')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches='tight')
    
    print(f"Model analysis plots saved to {output_dir}/")

def predict_efficacy(model, drug_targets, output_list, condition="APOE4"):
   
    if isinstance(model, dict) and 'model' in model:
        # Extract the actual model if we have a dictionary with metadata
        ml_model = model['model']
    else:
        # Use as is if it's just the model
        ml_model = model
        
      
    # Validate model type
    if not hasattr(ml_model, 'predict'):
        raise ValueError("Invalid model: The provided model does not have a predict method")
    
    # Extract features for prediction
    features = extract_features(drug_targets, output_list, condition)
    
    # Validate feature length if model has feature_importances_ attribute
    if hasattr(ml_model, 'feature_importances_'):
        expected_features = len(ml_model.feature_importances_)
        if len(features) != expected_features:
            print(f"Warning: Feature length mismatch - model expects {expected_features} features but received {len(features)}")
            # Try to match feature length, either by padding or truncating
            if len(features) < expected_features:
                print(f"Padding feature vector with zeros to match expected length")
                features = features + [0] * (expected_features - len(features))
            else:
                print(f"Truncating feature vector to match expected length")
                features = features[:expected_features]
    
    # Make prediction
    try:
        prediction = ml_model.predict([features])[0]
        return prediction
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
    
def plot_drug_effects(node_changes, output_dir="drug_analysis"):
   
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort nodes by absolute change magnitude
    sorted_nodes = sorted(node_changes.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Extract names and values
    nodes = [item[0] for item in sorted_nodes]
    changes = [item[1] for item in sorted_nodes]
    
    # Create color map (red for negative/beneficial, blue for positive/harmful)
    colors = ['#d73027' if x < 0 else '#4575b4' for x in changes]
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(nodes)), changes, color=colors)
    
    # Add node names
    plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add title and labels
    plt.title('Drug Effects on AD-Related Pathways', fontsize=14)
    plt.ylabel('Change in Node State (Negative = Beneficial)', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/pathway_effects.png", dpi=300, bbox_inches='tight')
    
    return f"{output_dir}/pathway_effects.png"

def run_perturbation_analysis(net, output_list, drug_targets, condition="APOE4"):
   
    print("\nStarting perturbation analysis...")
    
    # Extract single targets for pert_single
    single_targets = [target for target, _ in drug_targets]
    
    # Create pairs for pert_double
    pairs = []
    if len(drug_targets) >= 2:
        for i in range(len(drug_targets)):
            for j in range(i+1, len(drug_targets)):
                pairs.append((drug_targets[i][0], drug_targets[j][0]))
    
    # Set up genes_on and genes_off based on condition and effects
    genes_on = []
    genes_off = []
    
    if condition == "Normal":
        genes_off = ["APOE4"]
    elif condition == "APOE4":
        genes_on = ["APOE4"]
    elif condition == "LPL":
        genes_off = ["APOE4", "LPL"]
    
    # Add drug targets to genes_on or genes_off
    for target, effect in drug_targets:
        if effect == 1:
            if target not in genes_on:
                genes_on.append(target)
        else:
            if target not in genes_off:
                genes_off.append(target)
    
    # Run single target perturbation analysis
    print("Running single perturbation analysis...")
    if single_targets:
        try:
            pert_single_results = pert_single(single_targets, net, output_list, 
                                              on_node=genes_on, off_node=genes_off)
            
            # Check if results are empty or malformed
            if isinstance(pert_single_results, dict) and 'attractors' in pert_single_results:
                if not pert_single_results['attractors'] or len(pert_single_results['attractors']) == 0:
                    print("Warning: No attractors found in single perturbation analysis")
                    single_results_t = pd.DataFrame()
                else:
                    single_results_t = pert_single_results.iloc[2:].T if pert_single_results.shape[0] > 2 else pd.DataFrame()
            else:
                # Handle DataFrame return type
                if isinstance(pert_single_results, pd.DataFrame) and pert_single_results.shape[0] > 2:
                    single_results_t = pert_single_results.iloc[2:].T
                else:
                    print("Warning: Single perturbation analysis returned unexpected result format")
                    single_results_t = pd.DataFrame()
        except Exception as e:
            print(f"Error in single perturbation analysis: {str(e)}")
            single_results_t = pd.DataFrame()
    
    # Run double target perturbation analysis
    print("Running double perturbation analysis...")
    if pairs:
        pert_double_results = pert_double(pairs, net, output_list, 
                                           on_node=genes_on, off_node=genes_off)
        double_results_t = pert_double_results.iloc[2:].T if pert_double_results.shape[0] > 2 else pd.DataFrame()
    else:
        double_results_t = pd.DataFrame()
    
    # Combine results if available
    if not single_results_t.empty or not double_results_t.empty:
        if single_results_t.empty:
            combined_results = double_results_t
        elif double_results_t.empty:
            combined_results = single_results_t
        else:
            combined_results = pd.concat([single_results_t, double_results_t], axis=1)
        
        # Create appropriate column names
        column_names = []
        for target, _ in drug_targets:
            column_names.append(target)
        
        # Add double perturbation column names
        if pairs:
            for target1, target2 in pairs:
                column_names.append(f"{target1}/{target2}")
        
        # Set column names if we have the right number
        if combined_results.shape[1] == len(column_names):
            combined_results.columns = column_names
    else:
        combined_results = pd.DataFrame()
    
    print("Perturbation analysis complete")
    
    return {
        'single_results': single_results_t,
        'double_results': double_results_t,
        'combined_results': combined_results
    }

def compare_drug_simulation_vs_prediction(net, output_list, model, drug_targets, condition="APOE4"):
   
    # Get ML prediction
    ml_prediction = predict_efficacy(model, drug_targets, output_list, condition)
    
    # Get simulation-based prediction
    baseline = get_baseline_state(net, output_list, condition)
    drug_state = simulate_drug_effect(net, output_list, drug_targets=drug_targets, condition=condition)
    sim_results = calculate_efficacy(baseline, drug_state, output_list)
    
    # Run perturbation analysis
    pert_results = run_perturbation_analysis(net, output_list, drug_targets, condition)
    
    return {
        'ml_prediction': ml_prediction,
        'simulation_efficacy': sim_results['efficacy_score'],
        'difference': abs(ml_prediction - sim_results['efficacy_score']),
        'node_changes': sim_results['node_changes'],
        'pathway_changes': sim_results.get('pathway_changes', {}),
        'perturbation_results': pert_results
    }

def training_pipeline(network_file="A_model.txt", model_file="drug_efficacy_model.pkl", 
                    known_drugs=None, empirical_data=None, generate_plots=True):
   
    print("=== Starting Drug Efficacy Model Training Pipeline ===")
    
    # Step 1: Load the Boolean network model
    print("\nStep 1: Loading network model")
    network_data = load_network(network_file)
    net = network_data['net']
    output_list = network_data['output_list']
    print(f"Network loaded with {len(output_list)} genes")
    
    # Step 2: Set default known drugs if not provided
    if known_drugs is None:
        known_drugs = ["Lecanemab", "Memantine", "Donepezil", "Galantamine"]
        print(f"\nStep 2: Using default known drugs: {', '.join(known_drugs)}")
    else:
        print(f"\nStep 2: Using provided known drugs: {', '.join(known_drugs)}")
    
    # Step 3: Collect training data
    print("\nStep 3: Collecting training data")
    X, y, drug_info = collect_training_data(
        net=net, 
        output_list=output_list,
        known_drugs=known_drugs,
        additional_data=empirical_data
    )
    
    print(f"Collected {len(X)} training samples")
    
    # Step 4: Train and evaluate model
    print("\nStep 4: Training and evaluating model")
    model_results = train_model(X, y)
    
    # Step 5: Generate analysis plots
    if generate_plots:
        print("\nStep 5: Generating model analysis plots")
        plot_training_results(X, y, model_results, drug_info)
    
    # Step 6: Save the model
    print("\nStep 6: Saving trained model")
    results_to_save = {
        'model': model_results['model'],
        'metrics': model_results['metrics'],
        'training_features': X,
        'training_labels': y,
        'drug_info': drug_info,
        'output_list': output_list
    }
    save_model(results_to_save, model_file)
    
    print("\n=== Training Pipeline Complete ===")
    print(f"Model saved to {model_file}")
    print(f"Model performance: R² = {model_results['metrics']['r2']:.4f}, " 
          f"Cross-validation R² = {model_results['metrics']['cv_mean']:.4f}")
    
    return results_to_save

def test_drug_efficacy(network_file="A_model.txt", model_file="drug_efficacy_model.pkl", 
                      drug_targets=None, drug_name="Test Drug", conditions=None, 
                      output_dir="drug_analysis"):
    
    # Validate inputs
    if drug_targets is None:
        raise ValueError("Drug targets must be provided")
    
    if conditions is None:
        conditions = ["Normal", "APOE4", "LPL"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load network model
    print("Step 1: Loading network model")
    network_data = load_network(network_file)
    net = network_data['net']
    output_list = network_data['output_list']
    
    # Step 2: Load trained model
    print("Step 2: Loading trained model")
    model_data = load_model(model_file)
    if model_data is None:
        raise ValueError(f"Could not load model from {model_file}")
    
    # Extract the model from model_data
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        output_list = model_data.get('output_list', output_list)
    else:
        model = model_data
    
    # Test drug across conditions
    print("Testing drug across conditions")
    results = {}
    
    for condition in conditions:
        print(f"\nEvaluating {drug_name} in {condition} condition")
        
        # Get baseline state
        baseline = get_baseline_state(net, output_list, condition)
        
        # Simulate drug effect
        drug_state = simulate_drug_effect(net, output_list, drug_targets=drug_targets, condition=condition)
        
        # Calculate efficacy from simulation
        sim_results = calculate_efficacy(baseline, drug_state, output_list)
        
        # Get ML model prediction
        ml_prediction = predict_efficacy(model, drug_targets, output_list, condition)
        
        # Run perturbation analysis
        pert_results = run_perturbation_analysis(net, output_list, drug_targets, condition)
        
        # Store results
        results[condition] = {
            'simulation_efficacy': sim_results['efficacy_score'],
            'ml_prediction': ml_prediction,
            'difference': abs(ml_prediction - sim_results['efficacy_score']),
            'node_changes': sim_results['node_changes'],
            'pathway_changes': sim_results.get('pathway_changes', {}),
            'perturbation_results': pert_results
        }
        
        # Generate plots for this condition
        plot_path = plot_drug_effects(sim_results['node_changes'], 
                                      f"{output_dir}/{condition}")
        results[condition]['plot_path'] = plot_path
        
        # Save perturbation results
        if not pert_results['combined_results'].empty:
            pert_file = f"{output_dir}/{condition}_perturbation_results.csv"
            pert_results['combined_results'].to_csv(pert_file)
            results[condition]['perturbation_file'] = pert_file
    
    # Generate summary report
    print("Generating summary report")
    
    # Create summary dataframe
    summary_data = []
    for condition in conditions:
        summary_data.append({
            'Condition': condition,
            'Simulation Efficacy': results[condition]['simulation_efficacy'],
            'ML Prediction': results[condition]['ml_prediction'],
            'Difference': results[condition]['difference']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_path = f"{output_dir}/efficacy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Generate summary plot
    plt.figure(figsize=(10, 6))
    x = range(len(conditions))
    width = 0.35
    
    # Plot bars for simulation and ML prediction
    sim_values = summary_df['Simulation Efficacy'].values
    ml_values = summary_df['ML Prediction'].values
    
    plt.bar(np.array(x) - width/2, sim_values, width, label='Simulation')
    plt.bar(np.array(x) + width/2, ml_values, width, label='ML Prediction')
    
    # Add labels and title
    plt.xlabel('Condition')
    plt.ylabel('Efficacy Score')
    plt.title(f'{drug_name} Efficacy Across Conditions')
    plt.xticks(x, conditions)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    summary_plot_path = f"{output_dir}/efficacy_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    
    # Print summary report
    print("\n=== Drug Efficacy Summary ===")
    print(f"Drug: {drug_name}")
    print(f"Targets: {drug_targets}")
    print("\nEfficacy by condition:")
    print(summary_df.to_string(index=False))
    
    # Calculate average efficacy across conditions
    avg_sim_efficacy = np.mean(summary_df['Simulation Efficacy'])
    avg_ml_efficacy = np.mean(summary_df['ML Prediction'])
    
    print(f"\nAverage simulation efficacy: {avg_sim_efficacy:.4f}")
    print(f"Average ML prediction: {avg_ml_efficacy:.4f}")
    
    # Return all results
    return {
        'drug_name': drug_name,
        'drug_targets': drug_targets,
        'results_by_condition': results,
        'summary_df': summary_df,
        'average_efficacy': avg_sim_efficacy,
        'summary_path': summary_path,
        'summary_plot_path': summary_plot_path
    }
