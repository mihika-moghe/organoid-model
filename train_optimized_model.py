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
from scipy.stats import pearsonr
from math import sqrt

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

from temporal_simulation import TemporalDrugSimulation

def create_empirical_dataset():
    """
    Create an empirical dataset from known drugs in clinical trials and additional literature data
    
    Returns:
        list: List of tuples containing drug targets, efficacy, and condition
    """
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
        # GSK3beta inhibitors (tideglusib, lithium, etc.)
        ([("GSK3beta", 0)], 0.18, "APOE4"),  # del Ser T, et al. J Alzheimers Dis. 2013 
        ([("GSK3beta", 0)], 0.15, "LPL"),    # Estimated from transgenic models
        
        # Tau aggregation inhibitors (LMTX, TRx0237)
        ([("Tau_aggregation", 0), ("Tau", 0)], 0.14, "APOE4"),  # Gauthier S, et al. Lancet. 2016
        ([("Tau_aggregation", 0), ("Tau", 0)], 0.12, "LPL"), 
        
        # Dual-targeting approaches
        ([("GSK3beta", 0), ("BACE1", 0)], 0.32, "APOE4"),  # Combination therapy models
        ([("GSK3beta", 0), ("BACE1", 0)], 0.28, "LPL"),
        
        # mTOR inhibitors (rapamycin)
        ([("mTOR", 0)], 0.15, "APOE4"),  # Spilman P, et al. PLoS One. 2010
        ([("mTOR", 0)], 0.12, "LPL"),
        
        # NMDAR modulators
        ([("e_NMDAR", 0), ("s_NMDAR", 1)], 0.17, "APOE4"),  # Based on memantine studies
        ([("e_NMDAR", 0), ("s_NMDAR", 1)], 0.14, "LPL"),
        
        # Neuroinflammation targets
        ([("TNFa", 0)], 0.14, "APOE4"),  # Butchart J, et al. J Alzheimers Dis. 2015 (etanercept)
        ([("TNFa", 0)], 0.11, "LPL"),
        
        # IL-1β inhibitors
        ([("IL1b", 0)], 0.13, "APOE4"),  # Based on anakinra studies
        ([("IL1b", 0)], 0.11, "LPL"),
        
        # Combination cholinergic approaches
        ([("AChE", 0), ("nAChR", 1)], 0.25, "APOE4"),  # Galantamine-like dual mechanisms
        ([("AChE", 0), ("nAChR", 1)], 0.22, "LPL"),
        
        # PPAR-gamma agonists (rosiglitazone)
        ([("PPAR_gamma", 1)], 0.12, "APOE4"),  # Risner ME, et al. Pharmacogenomics J. 2006
        ([("PPAR_gamma", 1)], 0.10, "LPL"),
        
        # Antioxidant approaches (curcumin)
        ([("NRF2", 1)], 0.09, "APOE4"),  # Small GW, et al. Am J Geriatr Psychiatry. 2018
        ([("NRF2", 1)], 0.07, "LPL"),
        
        # PDE inhibitors (PDE4)
        ([("PDE4", 0)], 0.15, "APOE4"),  # Based on rolipram studies
        ([("PDE4", 0)], 0.13, "LPL"),
        
        # Serotonergic compounds
        ([("5HT6", 0)], 0.11, "APOE4"),  # Based on idalopirdine trials
        ([("5HT6", 0)], 0.09, "LPL"),
        
        # GLP-1 receptor agonists (liraglutide)
        ([("GLP1R", 1)], 0.17, "APOE4"),  # Gejl M, et al. Front Aging Neurosci. 2016
        ([("GLP1R", 1)], 0.15, "LPL"),
        
        # Dual cholinergic and glutamatergic approaches (combined donepezil + memantine)
        ([("AChE", 0), ("e_NMDAR", 0)], 0.34, "APOE4"),  # Matsunaga S, et al. J Alzheimers Dis. 2015
        ([("AChE", 0), ("e_NMDAR", 0)], 0.30, "LPL"),
        
        # BDNF-enhancing approaches
        ([("BDNF", 1)], 0.14, "APOE4"),  # From various BDNF-enhancing compounds
        ([("BDNF", 1)], 0.12, "LPL"),
        
        # Anti-amyloid antibodies (various epitopes)
        ([("Abeta_oligomers", 0)], 0.28, "APOE4"),  # Oligomer-specific antibodies
        ([("Abeta_oligomers", 0)], 0.31, "LPL"),
        ([("Abeta_fibrils", 0)], 0.21, "APOE4"),  # Fibril-specific antibodies
        ([("Abeta_fibrils", 0)], 0.24, "LPL"),
        
        # Multi-target compounds
        ([("AChE", 0), ("MAO", 0), ("Abeta_aggregation", 0)], 0.22, "APOE4"),  # Ladostigil-like
        ([("AChE", 0), ("MAO", 0), ("Abeta_aggregation", 0)], 0.20, "LPL")
    ]
    
    empirical_data.extend(literature_data)
    
    return empirical_data

def train_model_with_optimized_params(X, y, metrics_file="optimized_model_metrics.txt"):
    """
    Trains a RandomForestRegressor model with optimized hyperparameters and outputs
    comprehensive metrics to a file for later access.
    
    Parameters:
    X (numpy.ndarray): Feature matrix
    y (numpy.ndarray): Target vector
    metrics_file (str): Path to text file where metrics will be saved
    
    Returns:
    dict: Contains the trained model and performance metrics
    """
    # Standard libraries
    import csv
    import os
    import time
    from math import sqrt
    
    # Scientific computing
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    
    # Machine learning
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    
    # Get timestamp for model ID
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_id = f"RF_optimized_model_{timestamp}"
    
    print(f"Training model with {len(X)} samples using optimized parameters...")
    
    # Handle very small dataset sizes with simplified approach
    if len(X) < 5:
        print("WARNING: Very small dataset, using simplified model")
        model = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate correlation coefficient
        r, p_value = pearsonr(y, y_pred)
        
        print(f"Model trained on all {len(X)} samples - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, r: {r:.4f}")
        
        # Save metrics to file
        metrics_dict = {
            'model_id': model_id,
            'timestamp': timestamp,
            'n_samples': len(X),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'r': r,
            'p_value': p_value,
            'cv_r2_mean': 'N/A',
            'cv_rmse_mean': 'N/A',
            'cv_mae_mean': 'N/A',
            'hyperparameter_tuning': 'No',
            'best_params': 'N/A'
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_file) if os.path.dirname(metrics_file) else '.', exist_ok=True)
        _save_metrics_to_file(metrics_dict, metrics_file)
        
        return {
            'model': model,
            'model_id': model_id,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'r': r,
                'p_value': p_value,
                'cv_scores': 'N/A - Too few samples',
                'cv_mean': 'N/A - Too few samples'
            }
        }
    
    # For larger datasets, use optimized parameters directly
    # Create efficacy bins for stratification
    try:
        y_bins = pd.qcut(y, min(4, len(y) // 2), labels=False, duplicates='drop')
    except ValueError:
        # If qcut fails, use regular bins
        y_bins = np.digitize(y, bins=np.linspace(min(y), max(y), min(4, len(y) // 2)))
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y_bins if len(set(y_bins)) > 1 else None
    )
    
    # Use the optimized hyperparameters
    optimized_params = {
        'n_estimators': 1000,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.3,
        'ccp_alpha': 0.0
    }
    
    print("Using optimized hyperparameters:")
    for param, value in optimized_params.items():
        print(f"  {param}: {value}")
    
    # Initialize model with optimized parameters
    model = RandomForestRegressor(
        **optimized_params,
        random_state=42,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1
    )
    
    # Fit model on training data
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Calculate correlation coefficient
    r, p_value = pearsonr(y_val, y_pred)
    
    print(f"Model validation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, r: {r:.4f}")
    
    # Perform cross-validation on the entire dataset
    # Determine appropriate number of folds based on data size
    n_splits = min(5, max(3, len(X) // 10))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"Performing {n_splits}-fold cross-validation with optimized model")
    
    # Calculate cross-validation scores for multiple metrics
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    
    # Calculate RMSE scores from MSE scores
    rmse_scores = np.sqrt(-mse_scores)
    
    print(f"Cross-validation R² scores: {r2_scores}")
    print(f"Mean cross-validation R²: {np.mean(r2_scores):.4f}")
    print(f"Mean cross-validation RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Mean cross-validation MAE: {-np.mean(mae_scores):.4f}")
    
    cv_r2_mean = np.mean(r2_scores)
    cv_rmse_mean = np.mean(rmse_scores)
    cv_mae_mean = -np.mean(mae_scores)
    
    # Analyze feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nFeature importance analysis:")
        feature_importance = model.feature_importances_
        
        # If we have explicit feature names, use them
        feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        
        # Print top 10 most important features
        sorted_idx = np.argsort(feature_importance)[::-1]
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
    else:
        feature_importance = None
        feature_names = None
    
    # Train final model on full dataset
    print("Training final model on full dataset with optimized parameters")
    final_model = RandomForestRegressor(**optimized_params, random_state=42)
    final_model.fit(X, y)
    
    # Get final metrics on full dataset
    y_pred_final = final_model.predict(X)
    final_mse = mean_squared_error(y, y_pred_final)
    final_rmse = sqrt(final_mse)
    final_mae = mean_absolute_error(y, y_pred_final)
    final_r2 = r2_score(y, y_pred_final)
    
    # Calculate final correlation coefficient
    final_r, final_p_value = pearsonr(y, y_pred_final)
    
    print(f"Final model on full dataset - MSE: {final_mse:.4f}, RMSE: {final_rmse:.4f}, "
          f"MAE: {final_mae:.4f}, R²: {final_r2:.4f}, r: {final_r:.4f}")
    
    # Save metrics to file
    metrics_dict = {
        'model_id': model_id,
        'timestamp': timestamp,
        'n_samples': len(X),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'r': r,
        'p_value': p_value,
        'cv_r2_mean': cv_r2_mean,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_mae_mean': cv_mae_mean,
        'final_mse': final_mse,
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'final_r2': final_r2,
        'final_r': final_r,
        'final_p_value': final_p_value,
        'hyperparameter_tuning': 'No (Using pre-optimized parameters)',
        'best_params': str(optimized_params),
        'n_estimators': optimized_params.get('n_estimators', 'N/A'),
        'max_depth': optimized_params.get('max_depth', 'N/A'),
        'min_samples_split': optimized_params.get('min_samples_split', 'N/A'),
        'min_samples_leaf': optimized_params.get('min_samples_leaf', 'N/A'),
        'max_features': optimized_params.get('max_features', 'N/A'),
        'ccp_alpha': optimized_params.get('ccp_alpha', 'N/A')
    }
    
    # Save metrics to file
    _save_metrics_to_file(metrics_dict, metrics_file)
    
    # Return comprehensive model information
    return {
        'model': final_model,
        'model_id': model_id,
        'optimized_params': optimized_params,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'r': r,
            'p_value': p_value,
            'cv_r2_scores': r2_scores,
            'cv_rmse_scores': rmse_scores,
            'cv_mae_scores': mae_scores,
            'cv_r2_mean': cv_r2_mean,
            'cv_rmse_mean': cv_rmse_mean,
            'cv_mae_mean': cv_mae_mean,
            'final_mse': final_mse,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'final_r2': final_r2,
            'final_r': final_r,
            'final_p_value': final_p_value,
            'feature_importance': list(zip(feature_names, feature_importance)) if feature_names and feature_importance is not None else None
        }
    }

def _save_metrics_to_file(metrics_dict, filename="optimized_model_metrics.txt"):
    """
    Helper function to save metrics to a nicely formatted text file
    
    Parameters:
    metrics_dict (dict): Dictionary containing metrics to save
    filename (str): Path to the text file
    """
    import os
    import time
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Format numeric values to be more readable (4 decimal places)
    formatted_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            # Format floating point numbers to 4 decimal places
            formatted_metrics[key] = f"{value:.4f}"
        else:
            formatted_metrics[key] = value
    
    # Open the file in append mode - we'll add a separator between runs
    with open(filename, 'a') as f:
        # Add a separator if the file already exists and has content
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            f.write("\n\n" + "="*80 + "\n\n")
        
        # Top level header
        f.write("="*80 + "\n")
        f.write(f"OPTIMIZED MODEL PERFORMANCE METRICS - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Basic model info section
        f.write("-"*80 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model ID:       {formatted_metrics.get('model_id', 'N/A')}\n")
        f.write(f"Training time:  {formatted_metrics.get('timestamp', 'N/A')}\n")
        f.write(f"Sample size:    {formatted_metrics.get('n_samples', 'N/A')}\n")
        f.write(f"Grid search:    {formatted_metrics.get('hyperparameter_tuning', 'N/A')}\n\n")
        
        # Validation metrics section
        f.write("-"*80 + "\n")
        f.write("VALIDATION METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Squared Error (MSE):        {formatted_metrics.get('mse', 'N/A')}\n")
        f.write(f"Root Mean Squared Error (RMSE):  {formatted_metrics.get('rmse', 'N/A')}\n")
        f.write(f"Mean Absolute Error (MAE):       {formatted_metrics.get('mae', 'N/A')}\n")
        f.write(f"R-squared (R²):                  {formatted_metrics.get('r2', 'N/A')}\n")
        f.write(f"Pearson Correlation (r):         {formatted_metrics.get('r', 'N/A')}\n")
        f.write(f"Correlation p-value:             {formatted_metrics.get('p_value', 'N/A')}\n\n")
        
        # Cross-validation metrics section
        f.write("-"*80 + "\n")
        f.write("CROSS-VALIDATION METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean R-squared (R²):             {formatted_metrics.get('cv_r2_mean', 'N/A')}\n")
        f.write(f"Mean RMSE:                       {formatted_metrics.get('cv_rmse_mean', 'N/A')}\n")
        f.write(f"Mean MAE:                        {formatted_metrics.get('cv_mae_mean', 'N/A')}\n\n")
        
        # Full dataset metrics section
        f.write("-"*80 + "\n")
        f.write("FULL DATASET METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Squared Error (MSE):        {formatted_metrics.get('final_mse', 'N/A')}\n")
        f.write(f"Root Mean Squared Error (RMSE):  {formatted_metrics.get('final_rmse', 'N/A')}\n")
        f.write(f"Mean Absolute Error (MAE):       {formatted_metrics.get('final_mae', 'N/A')}\n")
        f.write(f"R-squared (R²):                  {formatted_metrics.get('final_r2', 'N/A')}\n")
        f.write(f"Pearson Correlation (r):         {formatted_metrics.get('final_r', 'N/A')}\n")
        f.write(f"Correlation p-value:             {formatted_metrics.get('final_p_value', 'N/A')}\n\n")
        
        # Hyperparameter section
        f.write("-"*80 + "\n")
        f.write("MODEL HYPERPARAMETERS (OPTIMIZED)\n")
        f.write("-"*80 + "\n")
        f.write(f"n_estimators:                    {formatted_metrics.get('n_estimators', 'N/A')}\n")
        f.write(f"max_depth:                       {formatted_metrics.get('max_depth', 'N/A')}\n")
        f.write(f"min_samples_split:               {formatted_metrics.get('min_samples_split', 'N/A')}\n")
        f.write(f"min_samples_leaf:                {formatted_metrics.get('min_samples_leaf', 'N/A')}\n")
        f.write(f"max_features:                    {formatted_metrics.get('max_features', 'N/A')}\n")
        f.write(f"ccp_alpha:                       {formatted_metrics.get('ccp_alpha', 'N/A')}\n\n")
        
        # Also include the full parameter string for completeness
        f.write("Full parameter settings:\n")
        f.write(f"{formatted_metrics.get('best_params', 'N/A')}\n")
        
    print(f"Metrics saved to {filename}")
    
    # Also save as CSV for backward compatibility
    if filename.endswith('.txt'):
        csv_filename = filename.replace('.txt', '.csv')
        _save_metrics_to_csv(metrics_dict, csv_filename)

def _save_metrics_to_csv(metrics_dict, filename):
    """CSV metrics saving function for backward compatibility"""
    import os
    import csv
    
    # Format numeric values
    formatted_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            formatted_metrics[key] = f"{value:.4f}"
        else:
            formatted_metrics[key] = value
    
    # Use the same ordering as before
    ordered_fields = [
        'model_id', 'timestamp', 'n_samples', 
        'mse', 'rmse', 'mae', 'r2', 'r',
        'cv_r2_mean', 'cv_rmse_mean', 'cv_mae_mean',
        'final_mse', 'final_rmse', 'final_mae', 'final_r2', 'final_r', 'final_p_value',
        'hyperparameter_tuning', 'n_estimators', 'max_depth', 'min_samples_split', 
        'min_samples_leaf', 'max_features', 'ccp_alpha',
        'best_params'
    ]
    
    file_exists = os.path.isfile(filename)
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'a', newline='') as csvfile:
        available_fields = [field for field in ordered_fields if field in formatted_metrics]
        missing_fields = [field for field in formatted_metrics if field not in ordered_fields]
        all_fields = available_fields + missing_fields
        
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({k: formatted_metrics.get(k, metrics_dict.get(k)) for k in all_fields})

def _generate_drug_comparison_report(temporal_results, output_dir):
    """
    Generate comparative analysis between Memantine, Donepezil, and Galantamine
    
    Parameters:
    temporal_results (dict): Dictionary containing temporal simulation results
    output_dir (str): Directory to save the comparison report
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract drugs and conditions
    drugs = list(temporal_results.keys())
    conditions = list(temporal_results[drugs[0]].keys())
    
    # Initialize data structures for comparison
    efficacy_comparison = {}
    pathway_comparison = {}
    
    # For each condition, compare drugs
    for condition in conditions:
        efficacy_comparison[condition] = {}
        pathway_comparison[condition] = {}
        
        # Extract key timepoints for comparison
        timepoints = [0, 1, 6, 12, 36]  # Initial, 1 month, 6 months, 1 year, 3 years
        
        # For each drug, extract efficacy scores at key timepoints
        for drug in drugs:
            drug_data = temporal_results[drug][condition]
            
            # Initialize arrays for this drug
            efficacy_scores = []
            pathway_scores = {
                'Amyloid': [], 'Tau': [], 'Synaptic': [],
                'Neuroinflammation': [], 'Cholinergic': [], 'NMDA': []
            }
            
            # Extract data for each timepoint
            for i, month in enumerate(timepoints):
                if month == 0:
                    timepoint_key = "initial"
                else:
                    timepoint_key = f"month_{month}"
                
                # Skip if this timepoint doesn't exist
                if timepoint_key not in drug_data["results"]:
                    efficacy_scores.append(np.nan)
                    for pathway in pathway_scores:
                        pathway_scores[pathway].append(np.nan)
                    continue
                
                # Get results for this timepoint
                result = drug_data["results"][timepoint_key]
                
                # Extract efficacy score
                if "efficacy_score" in result:
                    efficacy_scores.append(result["efficacy_score"])
                else:
                    efficacy_scores.append(np.nan)
                
                # Extract pathway scores
                if "pathway_scores" in result:
                    for pathway in pathway_scores:
                        if pathway in result["pathway_scores"]:
                            pathway_scores[pathway].append(result["pathway_scores"][pathway])
                        else:
                            pathway_scores[pathway].append(np.nan)
                else:
                    for pathway in pathway_scores:
                        pathway_scores[pathway].append(np.nan)
            
            # Store results
            efficacy_comparison[condition][drug] = efficacy_scores
            pathway_comparison[condition][drug] = pathway_scores
        
        # Generate efficacy comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot efficacy scores for each drug
        for drug in drugs:
            efficacy_scores = efficacy_comparison[condition][drug]
            
            # Choose color based on drug mechanism
            if drug == "Memantine":
                color = "#4CAF50"  # Green for NMDA antagonist
                marker = "o"
                linestyle = "-"
            elif drug == "Donepezil":
                color = "#2196F3"  # Blue for AChEI
                marker = "s"
                linestyle = "--"
            elif drug == "Galantamine":
                color = "#9C27B0"  # Purple for dual mechanism
                marker = "^"
                linestyle = "-."
            else:
                color = "#F44336"  # Red for other
                marker = "x"
                linestyle = ":"
            
            plt.plot(timepoints, efficacy_scores, label=drug, color=color, 
                     marker=marker, linestyle=linestyle, linewidth=2, markersize=8)
        
        # Add title and labels
        plt.title(f"Drug Efficacy Comparison - {condition} Condition", fontsize=16)
        plt.xlabel("Months", fontsize=14)
        plt.ylabel("Efficacy Score", fontsize=14)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficacy_comparison_{condition.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate pathway comparison plots
        for pathway in ['Amyloid', 'Tau', 'Cholinergic', 'NMDA']:
            plt.figure(figsize=(12, 8))
            
            # Plot pathway scores for each drug
            for drug in drugs:
                pathway_data = pathway_comparison[condition][drug][pathway]
                
                # Choose color based on drug mechanism (same as before)
                if drug == "Memantine":
                    color = "#4CAF50"  # Green for NMDA antagonist
                    marker = "o"
                    linestyle = "-"
                elif drug == "Donepezil":
                    color = "#2196F3"  # Blue for AChEI
                    marker = "s"
                    linestyle = "--"
                elif drug == "Galantamine":
                    color = "#9C27B0"  # Purple for dual mechanism
                    marker = "^"
                    linestyle = "-."
                else:
                    color = "#F44336"  # Red for other
                    marker = "x"
                    linestyle = ":"
                
                plt.plot(timepoints, pathway_data, label=drug, color=color, 
                         marker=marker, linestyle=linestyle, linewidth=2, markersize=8)
            
            # Add title and labels
            plt.title(f"{pathway} Pathway Comparison - {condition} Condition", fontsize=16)
            plt.xlabel("Months", fontsize=14)
            plt.ylabel(f"{pathway} Pathway Score", fontsize=14)
            plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{pathway.lower()}_comparison_{condition.lower()}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Generate comprehensive report
    with open(f"{output_dir}/drug_comparison_report.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("ALZHEIMER'S DISEASE DRUG COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("This report compares the efficacy of Memantine, Donepezil, and Galantamine\n")
        f.write("across different conditions and timepoints using the optimized model.\n\n")
        
        for condition in conditions:
            f.write("-"*80 + "\n")
            f.write(f"CONDITION: {condition}\n")
            f.write("-"*80 + "\n\n")
            
            # Compare efficacy at different timepoints
            f.write("Efficacy Comparison:\n")
            f.write("-------------------\n")
            
            # Create a table header for timepoints
            f.write(f"{'Drug':<15}")
            for month in timepoints:
                if month == 0:
                    f.write(f"{'Initial':<12}")
                else:
                    f.write(f"{'Month '+str(month):<12}")
            f.write("\n")
            
            # Add drug data rows
            for drug in drugs:
                efficacy_scores = efficacy_comparison[condition][drug]
                f.write(f"{drug:<15}")
                for score in efficacy_scores:
                    if np.isnan(score):
                        f.write(f"{'N/A':<12}")
                    else:
                        f.write(f"{score:.4f}    ")
                f.write("\n")
            
            f.write("\n")
            
            # Add key pathway comparisons
            for pathway in ['Amyloid', 'Tau', 'Cholinergic', 'NMDA']:
                f.write(f"{pathway} Pathway Comparison:\n")
                f.write("-"*30 + "\n")
                
                # Table header
                f.write(f"{'Drug':<15}")
                for month in timepoints:
                    if month == 0:
                        f.write(f"{'Initial':<12}")
                    else:
                        f.write(f"{'Month '+str(month):<12}")
                f.write("\n")
                
                # Add drug data rows
                for drug in drugs:
                    pathway_data = pathway_comparison[condition][drug][pathway]
                    f.write(f"{drug:<15}")
                    for score in pathway_data:
                        if np.isnan(score):
                            f.write(f"{'N/A':<12}")
                        else:
                            f.write(f"{score:.4f}    ")
                    f.write("\n")
                
                f.write("\n")
            
            # Add summary comparison for this condition
            f.write("Summary for this condition:\n")
            
            # Calculate average efficacy across timepoints for ranking
            drug_avg_efficacy = {}
            for drug in drugs:
                scores = efficacy_comparison[condition][drug]
                valid_scores = [s for s in scores if not np.isnan(s)]
                if valid_scores:
                    drug_avg_efficacy[drug] = sum(valid_scores) / len(valid_scores)
                else:
                    drug_avg_efficacy[drug] = 0
            
            # Rank drugs by average efficacy
            ranked_drugs = sorted(drug_avg_efficacy.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"Rank by average efficacy:\n")
            for i, (drug, avg) in enumerate(ranked_drugs, 1):
                f.write(f"{i}. {drug}: {avg:.4f}\n")
            
            f.write("\n")
        
        # Add overall summary
        f.write("="*80 + "\n")
        f.write("OVERALL COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Calculate global average efficacy across all conditions
        global_efficacy = {}
        for drug in drugs:
            scores = []
            for condition in conditions:
                condition_scores = efficacy_comparison[condition][drug]
                valid_scores = [s for s in condition_scores if not np.isnan(s)]
                scores.extend(valid_scores)
            
            if scores:
                global_efficacy[drug] = sum(scores) / len(scores)
            else:
                global_efficacy[drug] = 0
        
        # Rank drugs by global average efficacy
        global_ranked_drugs = sorted(global_efficacy.items(), key=lambda x: x[1], reverse=True)
        
        f.write(f"Global rank by average efficacy:\n")
        for i, (drug, avg) in enumerate(global_ranked_drugs, 1):
            f.write(f"{i}. {drug}: {avg:.4f}\n")
        
        f.write("\n")
        
        # Add drug mechanism summary
        f.write("Drug Mechanism Summary:\n")
        f.write("-"*25 + "\n")
        f.write("Memantine: NMDA receptor antagonist, primarily affects glutamatergic transmission\n")
        f.write("Donepezil: Acetylcholinesterase inhibitor (AChEI), increases acetylcholine levels\n")
        f.write("Galantamine: Dual mechanism - AChEI and allosteric modulator of nicotinic receptors\n")
    
    print(f"Drug comparison report generated in {output_dir}")
    return True

def display_metrics_from_file(filename="optimized_model_metrics.txt"):
    """
    Read metrics from a text file and display them in a readable format
    
    Parameters:
    filename (str): Path to the text file with metrics
    """
    import os
    
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: Metrics file '{filename}' not found.")
            return None
        
        # Read the file content
        with open(filename, 'r') as f:
            content = f.read()
        
        # Print the content directly - it's already formatted
        print(content)
        
        # If this is the last model run in a file with multiple runs,
        # extract just the last run
        sections = content.split('='*80)
        if len(sections) > 2:  # More than one model run in the file
            last_run = sections[-2] + '='*80 + sections[-1]
            return last_run
        else:
            return content
            
    except Exception as e:
        print(f"Error reading metrics file: {e}")
        
        # Try reading as CSV if text file fails (backward compatibility)
        if filename.endswith('.txt'):
            csv_filename = filename.replace('.txt', '.csv')
            if os.path.exists(csv_filename):
                print(f"Attempting to read metrics from CSV file: {csv_filename}")
                try:
                    import pandas as pd
                    metrics_df = pd.read_csv(csv_filename)
                    print("\n===== OPTIMIZED MODEL METRICS SUMMARY =====")
                    print(metrics_df.iloc[-1].to_string())
                    return metrics_df.iloc[-1]
                except Exception as csv_err:
                    print(f"Error reading CSV metrics file: {csv_err}")
        
        return None

def optimized_training_pipeline(network_file="A_model.txt", model_file="optimized_ad_drug_efficacy_model.pkl", 
                     include_empirical=True, generate_plots=True, metrics_file="optimized_ad_drug_efficacy_metrics_2.txt"):
    """
    Pipeline to train an AD drug efficacy model using optimized hyperparameters
    specifically focusing on Memantine, Donepezil, and Galantamine drugs.
    
    Parameters:
    network_file (str): Path to the network model file
    model_file (str): Path to save the trained model
    include_empirical (bool): Whether to include empirical data
    generate_plots (bool): Whether to generate plots
    metrics_file (str): Path to save the metrics
    
    Returns:
    dict: Dictionary containing model results and analysis data
    """
    print("=== Starting Optimized Alzheimer's Disease Drug Efficacy Model Training ===")
    
    # Step 1: Import main file results
    print("\nStep 1: Importing main file results")
    try:
        # Import the main file to access its variables and results
        import main
        
        # Access all the results we need
        net = main.net
        output_list = main.output_list
        
        # Get baseline attractors from each condition
        attractors_Normal = main.attractors_Normal
        attractors_APOE4 = main.attractors_APOE4
        attractors_LPL = main.attractors_LPL
        
        # Get perturbation results
        APOE4_pert_res = main.APOE4_pert_res
        LPL_pert_res = main.LPL_pert_res
        
        # Store baseline attractors
        baseline_attractors = {
            "Normal": attractors_Normal,
            "APOE4": attractors_APOE4,
            "LPL": attractors_LPL
        }
        
        # Store perturbation results
        perturbation_results = {
            "APOE4": APOE4_pert_res,
            "LPL": LPL_pert_res,
            "Normal": None  # No perturbation for Normal in main file, using None as placeholder
        }
        
        print("Successfully imported main file results")
    except Exception as e:
        print(f"ERROR importing main file results: {e}")
        # Load network as fallback
        network_data = load_network(network_file)
        net = network_data['net']
        output_list = network_data['output_list']
        baseline_attractors = {}
        perturbation_results = {}
    
    # Step 2: Define known drugs, explicitly focusing on the three main drugs
    print("\nStep 2: Defining specific drugs to focus on")
    known_drugs = ["Memantine", "Donepezil", "Galantamine"]
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
    try:
        X, y, drug_info = collect_training_data(
            net=net, 
            output_list=output_list,
            known_drugs=known_drugs,
            additional_data=empirical_data,
            baseline_attractors=baseline_attractors  # Pass baseline attractors here
        )
        print(f"Successfully collected data for {len(X)} samples")
        
        # Print information about the collected data
        drugs_in_dataset = set([info['name'] for info in drug_info if 'name' in info])
        print(f"Dataset includes data from drugs: {', '.join(drugs_in_dataset)}")
        
        conditions_in_dataset = set([info['condition'] for info in drug_info if 'condition' in info])
        print(f"Dataset includes conditions: {', '.join(conditions_in_dataset)}")
        
    except Exception as e:
        print(f"ERROR in data collection: {e}")
        raise
    
    # Step 5: Train the model with optimized parameters and metrics output
    print("\nStep 5: Training model with optimized parameters")
    try:
        model_results = train_model_with_optimized_params(X, y, metrics_file=metrics_file)
        # Important: Add output_list to model_results for feature naming
        model_results['output_list'] = output_list
        print(f"Model trained successfully")
    except Exception as e:
        print(f"ERROR in model training: {e}")
        raise
    
    # Step 6: Run Temporal Simulation
    print("\nStep 6: Running Temporal Simulations for Key Drugs")
    temporal_results = {}
    temporal_sim = TemporalDrugSimulation(
        network_file=network_file, 
        output_dir="optimized_drug_temporal_analysis 2",
        baseline_attractors=baseline_attractors,
        perturbation_results=perturbation_results
    )
    
    # Run simulation for each drug and condition
    for drug in known_drugs:
        drug_results = {}
        for condition in ["APOE4", "Normal", "LPL"]:
            print(f"\nSimulating {drug} in {condition} condition")
            result = temporal_sim.simulate_drug_over_time(
                drug_name=drug, 
                condition=condition, 
                include_visuals=generate_plots
            )
            drug_results[condition] = result
        temporal_results[drug] = drug_results
    
    # Save temporal results
    temporal_results_file = "optimized_drug_temporal_analysis/temporal_simulation_results.pkl"
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(temporal_results_file), exist_ok=True)
    with open(temporal_results_file, 'wb') as f:
        pickle.dump(temporal_results, f)
    print(f"Temporal simulation results saved to {temporal_results_file}")
    
    # Create directory for model file if needed
    os.makedirs(os.path.dirname(model_file) if os.path.dirname(model_file) else '.', exist_ok=True)
    
    # Save the trained model
    save_model({
        'model': model_results['model'],
        'output_list': output_list,
        'metrics': model_results.get('metrics', {}),
        'baseline_attractors': baseline_attractors,
        'perturbation_results': perturbation_results,
        'optimized_parameters': model_results.get('optimized_params', {})
    }, model_file)
    print(f"Optimized model saved to {model_file}")
    
    # Generate comparative analysis between the three drugs
    if generate_plots:
        print("\nStep 7: Generating comparative analysis between Memantine, Donepezil, and Galantamine")
        try:
            _generate_drug_comparison_report(temporal_results, "optimized_drug_temporal_analysis/drug_comparison")
        except Exception as e:
            print(f"WARNING: Error generating drug comparison: {e}")
    
    # Return both model results and temporal simulation results
    return {
        'model_data': model_results,
        'network_data': {'net': net, 'output_list': output_list},
        'temporal_results': temporal_results,
        'temporal_results_file': temporal_results_file,
        'baseline_attractors': baseline_attractors,
        'perturbation_results': perturbation_results
    }

if __name__ == "__main__":
    # Train the model and run temporal simulations
    results = optimized_training_pipeline()