import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import time
from PET import PETGenerator, generate_universal_pet_scans


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
    
    # Additional empirical data from literature  
    literature_data = [
        # GSK3beta inhibitors (tideglusib, etc.)
        ([("GSK3beta", 0)], 0.18, "APOE4"),  # del Ser T, et al. J Alzheimers Dis. 2013
        
        # Dual-targeting approaches
        ([("GSK3beta", 0), ("BACE1", 0)], 0.32, "APOE4"),   
        
        # mTOR inhibitors (rapamycin)
        ([("mTOR", 0)], 0.15, "APOE4"),  # Spilman P, et al. PLoS One. 2010
        
        # NMDA receptor modulators
        ([("e_NMDAR", 0), ("s_NMDAR", 1)], 0.17, "APOE4"),  # Based on combined studies
        
        # Neuroinflammation targets
        ([("TNFa", 0)], 0.14, "APOE4"),  # Butchart J, et al. J Alzheimers Dis. 2015
        
        # Combination cholinergic approaches
        ([("AChE", 0), ("nAChR", 1)], 0.25, "APOE4"),  # From galantamine + other enhancers
        
        # PPAR-gamma agonists
        ([("PPAR_gamma", 1)], 0.12, "APOE4"),  # Risner ME, et al. Pharmacogenomics J. 2006
        
        # Antioxidant approaches
        ([("NRF2", 1)], 0.09, "APOE4"),  # Based on curcumin and related compounds
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

def plot_training_results(X, y, model_data, drug_info, output_dir="model_analysis"):
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model = model_data['model']
    
    # 1. Predicted vs Actual Efficacy
    y_pred = model.predict(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.7)
    
    # Add drug names as annotations
    for i, drug in enumerate(drug_info):
        if 'name' in drug:
            plt.annotate(f"{drug['name']} ({drug['condition']})", (y[i], y_pred[i]), fontsize=8)
    
    # Add perfect prediction line
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Efficacy')
    plt.ylabel('Predicted Efficacy')
    plt.title('Model Predictions vs Actual Efficacy')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/predicted_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get top features
        n_features = min(20, len(importances))
        indices = np.argsort(importances)[-n_features:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(n_features), importances[indices])
        
        # Create feature names if possible
        feature_names = []
        if 'output_list' in model_data:
            output_list = model_data['output_list']
            feature_names = output_list + ['Normal', 'APOE4', 'LPL']
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        plt.yticks(range(n_features), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Features by Importance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
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
    plt.close()
    
    # 4. Data Source Analysis
    data_sources = {}
    for drug in drug_info:
        source = drug.get('data_source', 'unknown')
        data_sources[source] = data_sources.get(source, 0) + 1
    
    plt.figure(figsize=(8, 6))
    plt.bar(data_sources.keys(), data_sources.values())
    plt.xlabel('Data Source')
    plt.ylabel('Count')
    plt.title('Training Data Sources')
    plt.savefig(f"{output_dir}/data_sources.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model analysis plots saved to {output_dir}/")
    
    return {
        'predicted_vs_actual': f"{output_dir}/predicted_vs_actual.png",
        'feature_importance': f"{output_dir}/feature_importance.png" if hasattr(model, 'feature_importances_') else None,
        'residuals': f"{output_dir}/residuals.png",
        'data_sources': f"{output_dir}/data_sources.png"
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
    
    # Step 6: Generate analysis plots
    plot_paths = {}
    if generate_plots:
        print("\nStep 6: Generating model analysis plots")
        try:
            plot_paths = plot_training_results(X, y, model_results, drug_info)
        except Exception as e:
            print(f"WARNING: Error generating plots: {e}")
    else:
        print("\nStep 6: Skipping plot generation (not requested)")
    
    # Step 7: Save the model
    print("\nStep 7: Saving trained model")
    results_to_save = {
        'model': model_results['model'],
        'metrics': model_results['metrics'],
        'training_features': X,
        'training_labels': y,
        'drug_info': drug_info,
        'output_list': output_list,
        'plot_paths': plot_paths,
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        save_model(results_to_save, model_file)
        print(f"Model saved to {model_file}")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        raise
    
    print("\n=== Training Pipeline Complete ===")
    print(f"Model performance: ")
    if isinstance(model_results['metrics']['r2'], (int, float)):
        print(f"  R² = {model_results['metrics']['r2']:.4f}")
    else:
        print(f"  R² = {model_results['metrics']['r2']}")
        
    if isinstance(model_results['metrics']['cv_mean'], (int, float)):
        print(f"  Cross-validation R² = {model_results['metrics']['cv_mean']:.4f}")
    else:
        print(f"  Cross-validation R² = {model_results['metrics']['cv_mean']}")
    
    print(f"  MSE = {model_results['metrics']['mse']:.4f}")
    print(f"  MAE = {model_results['metrics']['mae']:.4f}")
    
    return results_to_save, network_data

def generate_detailed_pet_visualizations(baseline_pet, post_treatment_pet, output_dir, drug_name):
  
    os.makedirs(output_dir, exist_ok=True)
    figure_paths = {}
    
    # Create comparison visualizations for different modalities
    for modality in ['amyloid_suvr', 'tau_suvr']:
        # Create a more detailed figure with before/after comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle(f"{drug_name}: {modality.upper()} PET Comparison", fontsize=16)
        
        # Sort regions by baseline value for better visualization
        regions = list(baseline_pet.keys())
        values = [baseline_pet[region][modality] for region in regions]
        sorted_indices = np.argsort(values)
        sorted_regions = [regions[i] for i in sorted_indices]
        
        # Get values
        baseline_values = [baseline_pet[region][modality] for region in sorted_regions]
        post_values = [post_treatment_pet[region][modality] for region in sorted_regions]
        
        # Calculate changes
        changes = [post - base for post, base in zip(post_values, baseline_values)]
        percent_changes = [(post - base) / base * 100 if base > 0 else 0 
                          for post, base in zip(post_values, baseline_values)]
        
        # Create color map
        cmap = 'YlOrRd' if modality == 'amyloid_suvr' else 'YlGnBu'
        norm = plt.Normalize(min(baseline_values), max(baseline_values))
        
        # Plot baseline (pre-treatment)
        axes[0].barh(sorted_regions, baseline_values, color=plt.cm.get_cmap(cmap)(norm(baseline_values)))
        axes[0].set_title('Pre-Treatment', fontsize=14)
        axes[0].set_xlabel(f'{modality.split("_")[0].capitalize()} SUVr', fontsize=12)
        axes[0].grid(alpha=0.3)
        
        # Add values as text
        for i, v in enumerate(baseline_values):
            axes[0].text(v + 0.05, i, f"{v:.2f}", va='center')
        
        # Plot post-treatment
        bars = axes[1].barh(sorted_regions, post_values, color=plt.cm.get_cmap(cmap)(norm(post_values)))
        axes[1].set_title('Post-Treatment', fontsize=14)
        axes[1].set_xlabel(f'{modality.split("_")[0].capitalize()} SUVr', fontsize=12)
        axes[1].grid(alpha=0.3)
        
        # Add values and changes as text
        for i, (v, change, pct) in enumerate(zip(post_values, changes, percent_changes)):
            # Red for increases (bad), green for decreases (good) in amyloid/tau
            color = 'green' if change < 0 else 'red'
            axes[1].text(v + 0.05, i, f"{v:.2f} ({change:+.2f}, {pct:+.1f}%)", va='center', color=color)
        
        # Set consistent limits for both plots
        max_val = max(max(baseline_values), max(post_values)) * 1.1
        axes[0].set_xlim(0, max_val)
        axes[1].set_xlim(0, max_val)
        
        # Add a legend showing clinical significance
        if modality == 'amyloid_suvr':
            fig.text(0.15, 0.02, "Clinical Reference: Amyloid SUVr > 1.4 considered positive", 
                    ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
        else:  # tau
            fig.text(0.15, 0.02, "Clinical Reference: Tau SUVr > 1.3 considered positive", 
                    ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save the figure
        output_file = os.path.join(output_dir, f"{modality}_comparison.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        figure_paths[modality] = output_file
        
    # Create summary figure showing percent changes across all regions
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"{drug_name}: Regional Biomarker Changes", fontsize=16)
    
    regions = list(baseline_pet.keys())
    
    # Calculate percent changes for both modalities
    amyloid_changes = [(post_treatment_pet[r]['amyloid_suvr'] - baseline_pet[r]['amyloid_suvr']) / 
                      baseline_pet[r]['amyloid_suvr'] * 100 for r in regions]
    tau_changes = [(post_treatment_pet[r]['tau_suvr'] - baseline_pet[r]['tau_suvr']) / 
                   baseline_pet[r]['tau_suvr'] * 100 for r in regions]
    
    # Set width of bars
    width = 0.35
    x = np.arange(len(regions))
    
    # Create bars
    ax.bar(x - width/2, amyloid_changes, width, label='Amyloid SUVr Change (%)', color='orange')
    ax.bar(x + width/2, tau_changes, width, label='Tau SUVr Change (%)', color='blue')
    
    # Add a horizontal line at 0%
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and legend
    ax.set_ylabel('Percent Change (%)')
    ax.set_xlabel('Brain Regions')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add clinical significance text
    fig.text(0.5, 0.02, "Negative values indicate improvement (reduction in pathological proteins)", 
            ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the figure
    output_file = os.path.join(output_dir, f"regional_changes_summary.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    figure_paths['summary'] = output_file
    
    return figure_paths

def test_model_with_pet_scans(model_file="ad_drug_efficacy_model.pkl", network_data=None, network_file="A_model.txt"):
   
    print("\n=== Testing Model and Generating Universal PET Scans ===")
    
    # Load network if not provided
    if network_data is None:
        network_data = load_network(network_file)
    
    net = network_data['net']
    output_list = network_data['output_list']
    
    # Load model
    model_data = load_model(model_file)
    
    if model_data is None:
        print("ERROR: Model not found. Please train the model first.")
        return None
    
    # Test on ALL FDA-approved drugs (explicitly list all 4)
    test_drugs = ["Lecanemab", "Memantine", "Donepezil", "Galantamine"]
    drug_results = {}
    
    for drug_name in test_drugs:
        print(f"\nEvaluating {drug_name} with universal PET scan generation...")
        
        # Include ALL conditions including LPL
        results = evaluate_drug_efficacy(
            net=net,
            output_list=output_list,
            drug_name=drug_name,
            conditions=["APOE4", "Normal", "LPL"],
            output_dir=f"{drug_name.lower()}_analysis"
        )
        
        # Report summary results
        print(f"Overall efficacy score for {drug_name}:")
        for condition, data in results.items():
            if condition != "summary":
                print(f"  - {condition}: {data['efficacy_score']:.4f}")
        
        print(f"Universal PET scan visualizations generated in: {drug_name.lower()}_analysis/")
        drug_results[drug_name] = results
    
    print("\nTesting custom multi-target drug...")
    custom_drug_targets = [
        ("APP", 0),       # Amyloid pathway
        ("GSK3beta", 0),  # Tau pathway
        ("TNFa", 0)       # Neuroinflammation
    ]
    
    custom_results = evaluate_drug_efficacy(
        net=net,
        output_list=output_list,
        drug_targets=custom_drug_targets,
        conditions=["APOE4", "Normal", "LPL"],  # Include LPL
        output_dir="custom_drug_analysis"
    )
    
    print("Custom drug evaluation complete. Universal PET scans generated in: custom_drug_analysis/")
    
    # Test a novel combination drug
    print("\nTesting novel combination drug...")
    combination_drug_targets = [
        ("GSK3beta", 0),  # Tau pathway
        ("MAPT", 0),      # Another tau target
        ("TNFa", 0),      # Neuroinflammation
        ("IL1b", 0)       # More neuroinflammation
    ]
    
    combination_results = evaluate_drug_efficacy(
        net=net,
        output_list=output_list,
        drug_targets=combination_drug_targets,
        conditions=["APOE4", "Normal", "LPL"],  # Include LPL
        output_dir="combination_drug_analysis"
    )
    
    print("Combination drug evaluation complete. Universal PET scans generated in: combination_drug_analysis/")
    
    return {
        "known_drugs": drug_results,
        "custom_drug": custom_results,
        "combination_drug": combination_results
    }
def test_custom_drug_example():
    
    print("\n=== Testing Custom Drug Example ===")
    
    # Step 1: Load the network and model
    network_data = load_network("A_model.txt")
    net = network_data['net']
    output_list = network_data['output_list']
    
    model_data = load_model("ad_drug_efficacy_model.pkl")
    
    if model_data is None:
        print("ERROR: Model not found. Please train the model first.")
        return
    
    # Step 2: Define a new experimental drug with multiple targets
    # This example targets both amyloid and tau pathways, plus neuroinflammation
    custom_drug_targets = [
        ("APP", 0),       # Amyloid pathway - inhibit APP processing
        ("GSK3beta", 0),  # Tau pathway - inhibit tau phosphorylation
        ("NLRP3", 0)      # Neuroinflammation - inhibit inflammasome
    ]
    
    # Step 3: Predict efficacy using the ML model
    prediction = predict_efficacy(model_data, custom_drug_targets, "APOE4")
    print(f"ML model prediction for custom drug: {prediction:.4f}")
    
    # Step 4: Run full efficacy evaluation with PET scan generation
    results = evaluate_drug_efficacy(
        net=net,
        output_list=output_list,
        drug_targets=custom_drug_targets,
        conditions=["APOE4", "Normal"],
        output_dir="custom_drug_results"
    )
    
    # Step 5: Generate detailed PET visualizations
    for condition in results:
        if condition != "summary":
            detailed_vis_dir = os.path.join("custom_drug_results", condition.lower(), "detailed_pet_viz")
            if 'pet_data' in results[condition]:
                try:
                    detailed_pet_viz = generate_detailed_pet_visualizations(
                        results[condition]['pet_data']['baseline'],
                        results[condition]['pet_data']['post_treatment'],
                        detailed_vis_dir,
                        f"Custom Drug ({condition})"
                    )
                    results[condition]['detailed_pet_viz'] = detailed_pet_viz
                    print(f"  Generated detailed PET visualizations in {detailed_vis_dir}")
                except Exception as e:
                    print(f"  ERROR generating detailed PET visualizations: {e}")
    
    # Step 6: Summarize results
    print("\nEfficacy scores by condition:")
    for condition, data in results.items():
        if condition != "summary":
            print(f"  {condition}: {data['efficacy_score']:.4f}")
    
    print(f"\nOverall average efficacy: {results['summary']['average_efficacy']:.4f}")
    print(f"Overall composite score: {results['summary']['average_composite']:.4f}")
    
    # Step 7: Compare to existing drugs
    # Load known drugs for comparison
    known_drugs = list(DRUG_TARGETS.keys())
    known_drug_results = {}
    
    # Just test one for example (Lecanemab)
    test_known_drug = "Lecanemab"
    if test_known_drug in known_drugs:
        known_results = evaluate_drug_efficacy(
            net=net,
            output_list=output_list,
            drug_name=test_known_drug,
            conditions=["APOE4"],
            output_dir=f"{test_known_drug.lower()}_results"
        )
        known_drug_results[test_known_drug] = known_results
        
        # Generate detailed PET visualizations for comparison
        for condition in known_results:
            if condition != "summary":
                detailed_vis_dir = os.path.join(f"{test_known_drug.lower()}_results", condition.lower(), "detailed_pet_viz")
                if 'pet_data' in known_results[condition]:
                    try:
                        detailed_pet_viz = generate_detailed_pet_visualizations(
                            known_results[condition]['pet_data']['baseline'],
                            known_results[condition]['pet_data']['post_treatment'],
                            detailed_vis_dir,
                            f"{test_known_drug} ({condition})"
                        )
                        known_results[condition]['detailed_pet_viz'] = detailed_pet_viz
                        print(f"  Generated detailed PET visualizations in {detailed_vis_dir}")
                    except Exception as e:
                        print(f"  ERROR generating detailed PET visualizations: {e}")
        
        # Compare with custom drug
        custom_efficacy = results["APOE4"]["efficacy_score"]
        known_efficacy = known_results["APOE4"]["efficacy_score"]
        
        print(f"\nComparison in APOE4 condition:")
        print(f"  Custom drug: {custom_efficacy:.4f}")
        print(f"  {test_known_drug}: {known_efficacy:.4f}")
        
        if custom_efficacy > known_efficacy:
            improvement = ((custom_efficacy / known_efficacy) - 1) * 100
            print(f"  Custom drug shows {improvement:.1f}% improvement over {test_known_drug}")
        else:
            difference = ((known_efficacy / custom_efficacy) - 1) * 100
            print(f"  {test_known_drug} is {difference:.1f}% more effective than custom drug")
    
    print("\nEvaluation complete. PET scans and detailed results saved to respective directories.")
    return results

if __name__ == "__main__":
    # Train the model
    model_data, network_data = training_pipeline()
    
    # Test the model and generate PET scans 
    test_results = test_model_with_pet_scans(
        model_file="ad_drug_efficacy_model.pkl", 
        network_data=network_data
    )
    
