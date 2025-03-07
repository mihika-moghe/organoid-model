import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import boolnet
from matplotlib.colors import LinearSegmentedColormap
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import datetime
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Define known drug targets based on real data from DrugBank and ChEMBL
# Data extracted from:
# - https://go.drugbank.com/drugs/DB14580 (Lecanemab)
# - https://go.drugbank.com/drugs/DB01043 (Memantine)
# - https://go.drugbank.com/drugs/DB00843 (Donepezil)
# - https://go.drugbank.com/drugs/DB00674 (Galantamine)
# - https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL3833321
# - https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL807
# - https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL502
# - https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL659

DRUG_TARGETS = {
    "Lecanemab": [
        {"target": "APP", "effect": 0, "mechanism": "Binds to soluble Aβ protofibrils", 
         "affinity": "10 nM", "confidence": 0.95},
        {"target": "BACE1", "effect": 0, "mechanism": "Indirect reduction of BACE1 processing", 
         "affinity": "indirect", "confidence": 0.75}
    ],
    "Memantine": [
        {"target": "e_NMDAR", "effect": 0, "mechanism": "Non-competitive NMDA receptor antagonist", 
         "affinity": "500 nM (IC50)", "confidence": 0.90},
        {"target": "s_NMDAR", "effect": 0, "mechanism": "Moderate binding to synaptic NMDA receptors", 
         "affinity": "700 nM (Ki)", "confidence": 0.70}
    ],
    "Donepezil": [
        {"target": "AChE", "effect": 0, "mechanism": "Reversible inhibitor of acetylcholinesterase", 
         "affinity": "6.7 nM (IC50)", "confidence": 0.95},
        {"target": "BChE", "effect": 0, "mechanism": "Weak butyrylcholinesterase inhibition", 
         "affinity": "7400 nM (IC50)", "confidence": 0.60}
    ],
    "Galantamine": [
        {"target": "AChE", "effect": 0, "mechanism": "Competitive inhibitor of acetylcholinesterase", 
         "affinity": "2900 nM (IC50)", "confidence": 0.85},
        {"target": "nAChR", "effect": 1, "mechanism": "Positive allosteric modulator of nicotinic acetylcholine receptors", 
         "affinity": "1100 nM (EC50)", "confidence": 0.80}
    ]
}

# Real clinical efficacy data from trials and literature
# Data extracted from clinical trials, meta-analyses, and FDA submissions
CLINICAL_EFFICACY = {
    "Lecanemab": {
        "APOE4": {
            "efficacy": 0.27,  # 27% slowing of decline in clinical trials
            "cognitive_change": -0.45,  # CDR-SB change difference vs placebo
            "biomarker_change": -0.73,  # Amyloid PET SUVr reduction
            "side_effects": 0.17,  # ARIA-E incidence
            "confidence": 0.90
        },
        "Normal": {
            "efficacy": 0.31,
            "cognitive_change": -0.50,
            "biomarker_change": -0.77,
            "side_effects": 0.15,
            "confidence": 0.85
        }
    },
    "Memantine": {
        "APOE4": {
            "efficacy": 0.12,  # 12% slowing of decline in clinical trials
            "cognitive_change": -0.27,  # ADAS-cog change difference vs placebo
            "biomarker_change": -0.05,  # Minimal biomarker effects
            "side_effects": 0.05,  # Adverse event rate difference
            "confidence": 0.85
        },
        "Normal": {
            "efficacy": 0.15,
            "cognitive_change": -0.32,
            "biomarker_change": -0.08,
            "side_effects": 0.04,
            "confidence": 0.80
        }
    },
    "Donepezil": {
        "APOE4": {
            "efficacy": 0.30,  # 30% improvement in cognitive measures
            "cognitive_change": -2.8,  # ADAS-cog points improvement
            "biomarker_change": 0.12,  # ACh levels increase
            "side_effects": 0.15,  # GI adverse events
            "confidence": 0.90
        },
        "Normal": {
            "efficacy": 0.35,
            "cognitive_change": -3.1,
            "biomarker_change": 0.15,
            "side_effects": 0.12,
            "confidence": 0.85
        }
    },
    "Galantamine": {
        "APOE4": {
            "efficacy": 0.24,  # 24% improvement in cognitive measures
            "cognitive_change": -2.3,  # ADAS-cog points improvement
            "biomarker_change": 0.10,  # ACh levels increase
            "side_effects": 0.21,  # GI adverse events
            "confidence": 0.85
        },
        "Normal": {
            "efficacy": 0.29,
            "cognitive_change": -2.7,
            "biomarker_change": 0.13,
            "side_effects": 0.18,
            "confidence": 0.80
        }
    }
}

# Pharmacokinetic properties of known drugs
# Data extracted from DrugBank and FDA labels
PHARMACOKINETICS = {
    "Lecanemab": {
        "molecular_weight": 145781.6,  # Daltons
        "half_life": 24*24,  # hours (24 days)
        "bioavailability": 0.001,  # IV administration, limited BBB penetration
        "protein_binding": 0.99,  # High protein binding
        "clearance": 0.008,  # L/hr/kg
        "volume_distribution": 5.12,  # L/kg
        "administration": "intravenous",
        "bbb_penetration": 0.0015  # Blood-brain barrier penetration ratio
    },
    "Memantine": {
        "molecular_weight": 179.3,  # Daltons
        "half_life": 70,  # hours
        "bioavailability": 1.0,  # Complete oral absorption
        "protein_binding": 0.45,  # 45% protein bound
        "clearance": 0.13,  # L/hr/kg
        "volume_distribution": 9.4,  # L/kg
        "administration": "oral",
        "bbb_penetration": 0.82  # Good BBB penetration
    },
    "Donepezil": {
        "molecular_weight": 379.5,  # Daltons
        "half_life": 70,  # hours
        "bioavailability": 1.0,  # Complete oral absorption
        "protein_binding": 0.96,  # 96% protein bound
        "clearance": 0.13,  # L/hr/kg
        "volume_distribution": 12,  # L/kg
        "administration": "oral",
        "bbb_penetration": 0.48  # Moderate BBB penetration
    },
    "Galantamine": {
        "molecular_weight": 287.4,  # Daltons
        "half_life": 7,  # hours
        "bioavailability": 0.90,  # 90% bioavailable
        "protein_binding": 0.18,  # 18% protein bound
        "clearance": 0.34,  # L/hr/kg
        "volume_distribution": 2.6,  # L/kg
        "administration": "oral",
        "bbb_penetration": 0.40  # Moderate BBB penetration
    }
}

# Define key output nodes associated with AD pathology based on network model
AD_OUTPUT_NODES = [
    "APP", "BACE1", "MAPT", "GSK3beta", "Ca_ion", "p53", 
    "CASP3", "LC3", "PTEN", "Bcl2", "mTOR", "Cholesterol"
]

# Define pathway groupings for analysis
PATHWAYS = {
    'Amyloid': ['APP', 'BACE1', 'a_secretase', 'APOE4', 'LRP1'],
    'Tau': ['MAPT', 'GSK3beta', 'Cdk5', 'PP2A'],
    'Apoptosis': ['CASP3', 'p53', 'Bcl2', 'BAX', 'Cytochrome_c'],
    'Autophagy': ['LC3', 'mTOR', 'beclin1', 'AMPK', 'ATG'],
    'Lipid': ['Cholesterol', 'LPL', 'SREBP2', 'LDLR', 'ABCA1'],
    'Synaptic': ['Ca_ion', 'e_NMDAR', 's_NMDAR', 'AMPAR', 'PSD95'],
    'Neuroinflammation': ['TNFa', 'IL1b', 'IL6', 'NFkB', 'NLRP3', 'TREM2'],
    'Oxidative_Stress': ['ROS', 'NRF2', 'SOD1', 'GPX', 'CAT'],
    'Insulin_Signaling': ['Insulin', 'IR', 'IRS1', 'IDE', 'AKT']
}

# Brain regions affected in Alzheimer's disease
BRAIN_REGIONS = {
    'hippocampus': {
        'weight': 0.25,  # Importance weight for composite scores
        'amyloid_baseline': 0.7,  # Baseline amyloid level in APOE4 condition
        'tau_baseline': 0.65,     # Baseline tau level in APOE4 condition
        'atrophy_baseline': 0.15  # Baseline atrophy in APOE4 condition
    },
    'entorhinal_cortex': {
        'weight': 0.2,
        'amyloid_baseline': 0.65,
        'tau_baseline': 0.70,
        'atrophy_baseline': 0.10
    },
    'prefrontal_cortex': {
        'weight': 0.15,
        'amyloid_baseline': 0.60,
        'tau_baseline': 0.45,
        'atrophy_baseline': 0.08
    },
    'temporal_lobe': {
        'weight': 0.15,
        'amyloid_baseline': 0.55,
        'tau_baseline': 0.50,
        'atrophy_baseline': 0.12
    },
    'parietal_lobe': {
        'weight': 0.1,
        'amyloid_baseline': 0.45,
        'tau_baseline': 0.35,
        'atrophy_baseline': 0.05
    },
    'posterior_cingulate': {
        'weight': 0.1,
        'amyloid_baseline': 0.55,
        'tau_baseline': 0.40,
        'atrophy_baseline': 0.07
    },
    'precuneus': {
        'weight': 0.05,
        'amyloid_baseline': 0.50,
        'tau_baseline': 0.30,
        'atrophy_baseline': 0.05
    }
}

#===============================================================
# Core Functions
#===============================================================

def load_network(network_file="A_model.txt"):
    """
    Load the Boolean network model.
    
    Arguments:
    network_file -- Path to the network model file
    
    Returns:
    Dictionary with the network and output list
    """
    print("Loading network...")
    net = boolnet.load_network(network_file)
    output_list = net['genes'] 
    print(f"Loaded network with {len(net['genes'])} genes")
    
    return {
        'net': net,
        'output_list': output_list
    }

def get_baseline_state(net, output_list, condition="Normal"):
    """
    Get the baseline network state for a specific condition.
    
    Arguments:
    net -- Boolean network
    output_list -- List of genes in the network
    condition -- Condition to simulate ("Normal", "APOE4", or "LPL")
    
    Returns:
    Attractor states for the baseline condition
    """
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
    
    print(f"{condition} attractors calculated")
    
    return attractors

def simulate_drug_effect(net, output_list, drug_name=None, drug_targets=None, condition="APOE4"):
    """
    Simulate the effect of a drug on the network.
    
    Arguments:
    net -- Boolean network
    output_list -- List of genes in the network
    drug_name -- Name of known drug (if using predefined targets)
    drug_targets -- List of (target, effect) tuples for custom drugs
    condition -- Condition to simulate ("Normal", "APOE4", or "LPL")
    
    Returns:
    Attractor states after drug application
    """
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
        for target_info in DRUG_TARGETS[drug_name]:
            target = target_info["target"]
            effect = target_info["effect"]
            
            # Check if target exists in the model
            if target in output_list:
                if effect == 1:
                    genes_on.append(target)
                else:
                    genes_off.append(target)
            else:
                print(f"Warning: Target {target} not found in model")
    
    elif drug_targets:
        # For custom drug targets
        for target, effect in drug_targets:
            # Check if target exists in the model
            if target in output_list:
                if effect == 1:
                    genes_on.append(target)
                else:
                    genes_off.append(target)
            else:
                print(f"Warning: Target {target} not found in model")
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
    """
    Calculate drug efficacy by comparing baseline and drug-treated network states.
    
    Arguments:
    baseline_attractors -- Attractor states before drug application
    drug_attractors -- Attractor states after drug application
    output_list -- List of genes in the network
    
    Returns:
    Dictionary with efficacy metrics
    """
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

def calculate_comprehensive_efficacy(baseline_attractors, drug_attractors, drug_name, condition, output_list, drug_targets=None):
    """
    Calculate a comprehensive set of efficacy metrics for a drug.
    
    Arguments:
    baseline_attractors -- Attractor states before drug application
    drug_attractors -- Attractor states after drug application
    drug_name -- Name of the drug (for known drugs)
    condition -- Condition being tested ("Normal", "APOE4", or "LPL")
    output_list -- List of genes in the network
    drug_targets -- List of (target, effect) tuples for custom drugs
    
    Returns:
    Dictionary with comprehensive efficacy metrics
    """
    # Get basic efficacy metrics
    basic_metrics = calculate_efficacy(baseline_attractors, drug_attractors, output_list)
    efficacy_score = basic_metrics['efficacy_score']
    
    # Calculate pathway-specific scores
    pathway_scores = {}
    for pathway, score in basic_metrics['pathway_changes'].items():
        # Normalize scores: negative changes in disease-promoting pathways are good
        # For amyloid and tau pathways, negative changes are beneficial
        if pathway in ['Amyloid', 'Tau', 'Apoptosis']:
            # Flip sign and normalize to [0, 1] range where 1 is best
            normalized_score = max(min(-score, 1.0), 0.0)
        else:
            # For protective pathways like Autophagy, positive changes may be beneficial
            # This is simplified and would need customization based on specific pathways
            normalized_score = max(min(score, 1.0), 0.0)
        
        pathway_scores[pathway] = normalized_score
    
    # Calculate druggability score based on target properties
    # For known drugs, use the predefined targets
    if drug_name in DRUG_TARGETS:
        all_targets = DRUG_TARGETS[drug_name]
        target_confidences = [target_info["confidence"] for target_info in all_targets]
        druggability_score = np.mean(target_confidences) if target_confidences else 0.5
    # For custom drugs, use a default moderate confidence
    else:
        druggability_score = 0.6  
    
    # If it's a known drug, get real clinical efficacy data if available
    clinical_data = {}
    if drug_name in CLINICAL_EFFICACY and condition in CLINICAL_EFFICACY[drug_name]:
        clinical_data = CLINICAL_EFFICACY[drug_name][condition]
        
        # Calculate correlation between predicted and clinical efficacy
        if 'efficacy' in clinical_data:
            prediction_accuracy = 1.0 - abs(efficacy_score - clinical_data['efficacy'])
        else:
            prediction_accuracy = None
    else:
        prediction_accuracy = None
    
    # Get pharmacokinetic score if available (higher is better)
    pk_score = None
    if drug_name in PHARMACOKINETICS:
        pk = PHARMACOKINETICS[drug_name]
        # Simple PK score based on half-life and BBB penetration
        # Customizable based on specific requirements
        pk_score = min(pk['half_life'] / 100, 1.0) * pk['bbb_penetration']
    
    # Calculate safety score (placeholder - would be based on predicted off-target effects)
    # For known drugs, use inverse of side effects if available
    safety_score = None
    if drug_name in CLINICAL_EFFICACY and condition in CLINICAL_EFFICACY[drug_name]:
        if 'side_effects' in clinical_data:
            safety_score = 1.0 - clinical_data['side_effects']
    
    # Calculate composite score combining multiple factors
    factors = [efficacy_score]
    weights = [0.5]  # Efficacy has highest weight
    
    # Add pathway scores with lower weights
    if pathway_scores:
        pathway_avg = np.mean(list(pathway_scores.values()))
        factors.append(pathway_avg)
        weights.append(0.2)
    
    # Add druggability
    factors.append(druggability_score)
    weights.append(0.1)
    
    # Add PK score if available
    if pk_score is not None:
        factors.append(pk_score)
        weights.append(0.1)
    
    # Add safety score if available
    if safety_score is not None:
        factors.append(safety_score)
        weights.append(0.1)
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    
    # Calculate weighted composite score
    composite_score = np.sum(np.array(factors) * weights)
    
    return {
        'efficacy_score': efficacy_score,
        'pathway_scores': pathway_scores,
        'druggability_score': druggability_score,
        'pk_score': pk_score,
        'safety_score': safety_score,
        'composite_score': composite_score,
        'clinical_data': clinical_data,
        'prediction_accuracy': prediction_accuracy,
        'node_changes': basic_metrics['node_changes'],
        'pathway_changes': basic_metrics['pathway_changes']
    }

def extract_features(drug_targets, output_list, condition="APOE4"):
    """
    Extract features for machine learning model.
    
    Arguments:
    drug_targets -- List of (target, effect) tuples
    output_list -- List of genes in the network
    condition -- Condition being tested ("Normal", "APOE4", or "LPL")
    
    Returns:
    Feature vector
    """
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
    """
    Collect training data for machine learning model.
    
    Arguments:
    net -- Boolean network
    output_list -- List of genes in the network
    known_drugs -- List of known drug names
    additional_data -- Additional (targets, efficacy, condition) tuples
    
    Returns:
    X, y, drug_info for training
    """
    X = []
    y = []
    drug_info = []  # Store additional information about each data point
    
    # Process known drugs with real clinical efficacy data
    if known_drugs:
        for drug in known_drugs:
            if drug in DRUG_TARGETS:
                # Convert detailed drug targets to (target, effect) format
                targets = [(target_info["target"], target_info["effect"]) 
                           for target_info in DRUG_TARGETS[drug]]
                
                # Check if we have real clinical efficacy data
                conditions = ["APOE4", "Normal"]  # Default conditions to test
                
                for condition in conditions:
                    # Skip if we don't have clinical data for this condition
                    if drug not in CLINICAL_EFFICACY or condition not in CLINICAL_EFFICACY[drug]:
                        continue
                    
                    # Get real efficacy data
                    clinical_efficacy = CLINICAL_EFFICACY[drug][condition]["efficacy"]
                    
                    # Extract features and add to training data
                    features = extract_features(targets, output_list, condition)
                    X.append(features)
                    y.append(clinical_efficacy)
                    drug_info.append({
                        'name': drug,
                        'targets': targets,
                        'condition': condition,
                        'efficacy': clinical_efficacy,
                        'data_source': 'clinical_trial'
                    })
                    
                    print(f"Added {drug} in {condition} condition with efficacy {clinical_efficacy:.4f}")
    
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
                'efficacy': efficacy_score,
                'data_source': 'additional_data'
            })
            
            print(f"Added custom drug targeting {[t[0] for t in targets]} in {condition} condition")
    
    # Run simulations for drugs without clinical data
    if known_drugs:
        for drug in known_drugs:
            if drug in DRUG_TARGETS:
                targets = [(target_info["target"], target_info["effect"]) 
                          for target_info in DRUG_TARGETS[drug]]
                
                conditions = ["APOE4", "Normal", "LPL"]
                
                for condition in conditions:
                    # Skip if we already have clinical data for this condition
                    if (drug in CLINICAL_EFFICACY and 
                        condition in CLINICAL_EFFICACY[drug] and
                        any(d['name'] == drug and d['condition'] == condition for d in drug_info)):
                        continue
                    
                    # Get baseline for condition
                    baseline = get_baseline_state(net, output_list, condition)
                    
                    # Simulate drug effect
                    drug_state = simulate_drug_effect(net, output_list, drug_name=drug, condition=condition)
                    
                    # Calculate efficacy
                    efficacy = calculate_efficacy(baseline, drug_state, output_list)
                    sim_efficacy = efficacy['efficacy_score']
                    
                    # Add to training data
                    features = extract_features(targets, output_list, condition)
                    X.append(features)
                    y.append(sim_efficacy)
                    drug_info.append({
                        'name': drug,
                        'targets': targets,
                        'condition': condition,
                        'efficacy': sim_efficacy,
                        'data_source': 'simulation'
                    })
                    
                    print(f"Added simulated data for {drug} in {condition} condition with efficacy {sim_efficacy:.4f}")
    
    return np.array(X), np.array(y), drug_info

def save_model(model_data, filename="drug_efficacy_model.pkl"):
    """
    Save the trained model to a file.
    
    Arguments:
    model_data -- Model and associated data to save
    filename -- Output file name
    """
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filename}")

def load_model(filename="drug_efficacy_model.pkl"):
    """
    Load a trained model from a file.
    
    Arguments:
    filename -- Model file name
    
    Returns:
    Loaded model data
    """
    try:
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Loaded model from {filename}")
        return model_data
    except Exception as e:
        print(f"Could not load model from {filename}: {str(e)}")
        return None

def predict_efficacy(model_data, drug_targets, condition="APOE4"):
    """
    Predict efficacy of a drug using the trained model.
    
    Arguments:
    model_data -- Trained model data
    drug_targets -- List of (target, effect) tuples
    condition -- Condition to predict for ("Normal", "APOE4", or "LPL")
    
    Returns:
    Predicted efficacy score
    """
    # Extract model and output list
    model = model_data['model']
    output_list = model_data['output_list']
    
    # Extract features
    features = extract_features(drug_targets, output_list, condition)
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    return prediction

def generate_brain_pet_scan(node_changes, condition="APOE4", stage="baseline", drug_name=None):
    """
    Generate advanced, realistic simulated brain PET scan data.
    
    Arguments:
    node_changes -- Dictionary of node changes from drug effect
    condition -- Patient condition ("Normal", "APOE4", or "LPL")
    stage -- Scan stage: "baseline" or "post_treatment"
    drug_name -- Name of the drug (optional, for reference)
    
    Returns:
    Dictionary with simulated PET scan data for brain regions
    """
    # Advanced baseline with more nuanced condition-specific variability
    baseline_suvr = {
        'Normal': {
            'amyloid_baseline': 1.2,
            'tau_baseline': 1.1,
            'atrophy_baseline': 0.05,
            'variability': 0.1
        },
        'APOE4': {
            'amyloid_baseline': 1.8,
            'tau_baseline': 1.6,
            'atrophy_baseline': 0.15,
            'variability': 0.2
        },
        'LPL': {
            'amyloid_baseline': 1.5,
            'tau_baseline': 1.3,
            'atrophy_baseline': 0.10,
            'variability': 0.15
        }
    }
    
    # Enhanced drug-specific impact with more detailed metrics
    drug_impact = {
        'Lecanemab': {
            'amyloid_reduction': {
                'magnitude': -0.73,
                'regional_variation': {
                    'hippocampus': 1.2,
                    'entorhinal_cortex': 1.1,
                    'temporal_lobe': 1.0,
                    'prefrontal_cortex': 0.9,
                    'parietal_lobe': 0.8,
                    'posterior_cingulate': 0.9,
                    'precuneus': 0.8
                }
            },
            'tau_reduction': {
                'magnitude': -0.45,
                'regional_variation': {
                    'hippocampus': 1.3,
                    'entorhinal_cortex': 1.2,
                    'temporal_lobe': 1.0,
                    'prefrontal_cortex': 0.9,
                    'parietal_lobe': 0.8,
                    'posterior_cingulate': 0.9,
                    'precuneus': 0.8
                }
            },
            'atrophy_slowdown': -0.10
        },
        'Memantine': {
            'amyloid_reduction': {
                'magnitude': -0.05,
                'regional_variation': {
                    'hippocampus': 0.7,
                    'entorhinal_cortex': 0.6,
                    'temporal_lobe': 0.5,
                    'prefrontal_cortex': 0.4,
                    'parietal_lobe': 0.3,
                    'posterior_cingulate': 0.4,
                    'precuneus': 0.3
                }
            },
            'tau_reduction': {
                'magnitude': -0.15,
                'regional_variation': {
                    'hippocampus': 0.7,
                    'entorhinal_cortex': 0.6,
                    'temporal_lobe': 0.5,
                    'prefrontal_cortex': 0.4,
                    'parietal_lobe': 0.3,
                    'posterior_cingulate': 0.4,
                    'precuneus': 0.3
                }
            },
            'atrophy_slowdown': -0.05
        },
        'Donepezil': {
            'amyloid_reduction': {
                'magnitude': 0.12,  # Slight increase due to cholinergic effects
                'regional_variation': {
                    'hippocampus': 1.1,
                    'entorhinal_cortex': 1.0,
                    'temporal_lobe': 0.9,
                    'prefrontal_cortex': 1.0,
                    'parietal_lobe': 0.9,
                    'posterior_cingulate': 0.8,
                    'precuneus': 0.8
                }
            },
            'tau_reduction': {
                'magnitude': -0.20,
                'regional_variation': {
                    'hippocampus': 1.1,
                    'entorhinal_cortex': 1.0,
                    'temporal_lobe': 0.9,
                    'prefrontal_cortex': 1.0,
                    'parietal_lobe': 0.9,
                    'posterior_cingulate': 0.8,
                    'precuneus': 0.8
                }
            },
            'atrophy_slowdown': -0.08
        },
        'Galantamine': {
            'amyloid_reduction': {
                'magnitude': 0.10,  # Slight increase
                'regional_variation': {
                    'hippocampus': 1.1,
                    'entorhinal_cortex': 1.0,
                    'temporal_lobe': 0.9,
                    'prefrontal_cortex': 1.0,
                    'parietal_lobe': 0.9,
                    'posterior_cingulate': 0.8,
                    'precuneus': 0.8
                }
            },
            'tau_reduction': {
                'magnitude': -0.15,
                'regional_variation': {
                    'hippocampus': 1.1,
                    'entorhinal_cortex': 1.0,
                    'temporal_lobe': 0.9,
                    'prefrontal_cortex': 1.0,
                    'parietal_lobe': 0.9,
                    'posterior_cingulate': 0.8,
                    'precuneus': 0.8
                }
            },
            'atrophy_slowdown': -0.07
        }
    }
    
    # Get baseline values for this condition
    baseline = baseline_suvr.get(condition, baseline_suvr['Normal'])
    
    # Pharmacokinetic-inspired variability
    def get_pk_modifier(drug_name):
        """Adjust effect based on drug's pharmacokinetic properties"""
        if drug_name in PHARMACOKINETICS:
            pk = PHARMACOKINETICS[drug_name]
            # Consider half-life and blood-brain barrier penetration
            half_life_factor = min(pk.get('half_life', 24) / 24, 1.5)
            bbb_factor = pk.get('bbb_penetration', 0.5)
            return half_life_factor * bbb_factor
        return 1.0
    
    # Get drug-specific impact
    drug_effect = drug_impact.get(drug_name, {
        'amyloid_reduction': {'magnitude': -0.1, 'regional_variation': {}},
        'tau_reduction': {'magnitude': -0.1, 'regional_variation': {}},
        'atrophy_slowdown': -0.05
    })
    
    # Apply pharmacokinetic modifier
    pk_modifier = get_pk_modifier(drug_name)
    
    # Combine baseline values with drug-specific changes
    pet_data = {}
    for region in ['hippocampus', 'entorhinal_cortex', 'temporal_lobe', 
                   'prefrontal_cortex', 'parietal_lobe', 
                   'posterior_cingulate', 'precuneus']:
        # Start with baseline values
        region_pet = {
            'amyloid_suvr': baseline['amyloid_baseline'],
            'tau_suvr': baseline['tau_baseline'],
            'atrophy': baseline['atrophy_baseline']
        }
        
        # Add random variability based on condition
        variability = baseline['variability']
        region_pet['amyloid_suvr'] *= (1 + np.random.uniform(-variability, variability))
        region_pet['tau_suvr'] *= (1 + np.random.uniform(-variability, variability))
        region_pet['atrophy'] *= (1 + np.random.uniform(-variability, variability))
        
        # Apply drug-specific changes with region-specific sensitivity
        # Amyloid SUVr
        amyloid_reduction = drug_effect['amyloid_reduction']['magnitude']
        amyloid_regional_factor = drug_effect['amyloid_reduction']['regional_variation'].get(region, 1.0)
        region_pet['amyloid_suvr'] += amyloid_reduction * amyloid_regional_factor * pk_modifier
        
        # Tau SUVr
        tau_reduction = drug_effect['tau_reduction']['magnitude']
        tau_regional_factor = drug_effect['tau_reduction']['regional_variation'].get(region, 1.0)
        region_pet['tau_suvr'] += tau_reduction * tau_regional_factor * pk_modifier
        
        # Atrophy
        region_pet['atrophy'] += drug_effect['atrophy_slowdown'] * pk_modifier
        
        # Apply additional changes from node perturbations
        for pathway, change in node_changes.items():
            if pathway == 'Amyloid':
                region_pet['amyloid_suvr'] += change * 0.1
            elif pathway == 'Tau':
                region_pet['tau_suvr'] += change * 0.1
            elif pathway in ['Apoptosis', 'Synaptic', 'Autophagy']:
                region_pet['atrophy'] += change * 0.05
        
        # Ensure values stay within realistic ranges
        region_pet['amyloid_suvr'] = max(1.0, min(region_pet['amyloid_suvr'], 3.0))
        region_pet['tau_suvr'] = max(1.0, min(region_pet['tau_suvr'], 3.0))
        region_pet['atrophy'] = max(0.0, min(region_pet['atrophy'], 0.5))
        
        # Store region data
        pet_data[region] = region_pet
    
    # Add metadata
    pet_data['metadata'] = {
        'condition': condition,
        'stage': stage,
        'drug_name': drug_name,
        'pk_modifier': pk_modifier
    }
    
    return pet_data

def visualize_pet_scan(baseline_pet, post_treatment_pet=None, output_dir="pet_scans"):
    """
    Create visualizations of simulated PET scans.
    
    Arguments:
    baseline_pet -- Baseline PET scan data
    post_treatment_pet -- Post-treatment PET scan data (optional)
    output_dir -- Directory to save visualization images
    
    Returns:
    Dictionary with paths to saved visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define custom colormaps
    amyloid_cmap = LinearSegmentedColormap.from_list('amyloid', ['#FFFFFF', '#FFF7BC', '#FEC44F', '#D95F0E', '#993404'])
    tau_cmap = LinearSegmentedColormap.from_list('tau', ['#FFFFFF', '#EDF8FB', '#B2E2E2', '#66C2A4', '#238B45', '#005824'])
    
    # Function to get colormap and limits for each modality
    def get_cmap_and_limits(modality):
        if modality == 'amyloid_suvr':
            return amyloid_cmap, (1.0, 2.2)
        elif modality == 'tau_suvr':
            return tau_cmap, (1.0, 2.5)
        else:  # atrophy
            return plt.cm.Greys, (0, 0.3)
    
    # Create visualizations for each modality
    figure_paths = {}
    
    # Remove metadata if present
    if isinstance(baseline_pet, dict) and 'metadata' in baseline_pet:
        baseline_pet = {k: v for k, v in baseline_pet.items() if k != 'metadata'}
    
    # Validate input
    if not baseline_pet:
        print("WARNING: Empty baseline PET data")
        return {}
    
    # Ensure we have the correct dictionary structure
    if not all('amyloid_suvr' in region and 'tau_suvr' in region and 'atrophy' in region 
               for region in baseline_pet.values()):
        print("WARNING: Baseline PET data does not have the expected structure")
        return {}
    
    for modality in ['amyloid_suvr', 'tau_suvr', 'atrophy']:
        # Set up the figure
        fig, axes = plt.subplots(1, 2 if post_treatment_pet else 1, figsize=(12, 6))
        
        # In case of single plot, make axes iterable
        if post_treatment_pet is None:
            axes = [axes]
        
        cmap, v_limits = get_cmap_and_limits(modality)
        
        # Plot brain regions in the first subplot (baseline)
        regions = list(baseline_pet.keys())
        values = [baseline_pet[region][modality] for region in regions]
        
        # Sort regions by modality value for clearer visualization
        sorted_indices = np.argsort(values)
        sorted_regions = [regions[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Create bar chart
        bars = axes[0].barh(sorted_regions, sorted_values, color=[cmap((v - v_limits[0]) / (v_limits[1] - v_limits[0])) for v in sorted_values])
        
        # Add values as text
        for i, v in enumerate(sorted_values):
            axes[0].text(max(v + 0.05, v_limits[0] + 0.15), i, f"{v:.2f}", va='center')
        
        # Add title and labels
        modality_title = modality.replace('_', ' ').replace('suvr', 'SUVr').title()
        axes[0].set_title(f"Baseline {modality_title}")
        axes[0].set_xlim(v_limits)
        axes[0].grid(axis='x', linestyle='--', alpha=0.7)
        
        # If we have post-treatment data, add to second subplot
        if post_treatment_pet:
            # Remove metadata from post_treatment_pet if present
            if isinstance(post_treatment_pet, dict) and 'metadata' in post_treatment_pet:
                post_treatment_pet = {k: v for k, v in post_treatment_pet.items() if k != 'metadata'}
            
            post_values = [post_treatment_pet[region][modality] for region in sorted_regions]
            
            # Calculate differences for text coloring
            diffs = [post - base for post, base in zip(post_values, sorted_values)]
            
            # Create bar chart with same order as baseline
            bars = axes[1].barh(sorted_regions, post_values, color=[cmap((v - v_limits[0]) / (v_limits[1] - v_limits[0])) for v in post_values])
            
            # Add values and change as text
            for i, (v, diff) in enumerate(zip(post_values, diffs)):
                # Use red for increases, green for decreases in amyloid/tau, opposite for atrophy
                if modality in ['amyloid_suvr', 'tau_suvr']:
                    color = 'green' if diff < 0 else 'red'
                else:  # atrophy
                    color = 'red' if diff > 0 else 'green'
                
                axes[1].text(max(v + 0.05, v_limits[0] + 0.15), i, f"{v:.2f} ({diff:+.2f})", va='center', color=color)
            
            # Add title and labels
            axes[1].set_title(f"Post-Treatment {modality_title}")
            axes[1].set_xlim(v_limits)
            axes[1].grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        output_file = os.path.join(output_dir, f"{modality.replace('_', '_')}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        figure_paths[modality] = output_file
    
    return figure_paths

def generate_pet_data(efficacy_data, condition="APOE4"):
    """
    Generate simulated PET scan data based on efficacy results.
    
    Arguments:
    efficacy_data -- Results from efficacy calculation
    condition -- Condition being simulated
    
    Returns:
    Dictionary with baseline and post-treatment PET data
    """
    # Baseline PET values by brain region for the condition
    baseline_pet = {}
    
    # Default baseline values
    for region, region_data in BRAIN_REGIONS.items():
        baseline_pet[region] = {
            'amyloid_suvr': region_data['amyloid_baseline'],
            'tau_suvr': region_data['tau_baseline'],
            'atrophy': region_data['atrophy_baseline']
        }
    
    # Model treatment effect - calculate post-treatment values
    post_treatment_pet = {}
    
    # Apply pathway-specific changes to each region
    pathway_changes = efficacy_data['pathway_changes']
    
    # Calculate region changes based on pathway impacts
    for region, region_data in BRAIN_REGIONS.items():
        # Initialize with baseline values
        post_treatment_pet[region] = {
            'amyloid_suvr': baseline_pet[region]['amyloid_suvr'],
            'tau_suvr': baseline_pet[region]['tau_suvr'],
            'atrophy': baseline_pet[region]['atrophy']
        }
        
        # Apply changes based on pathways
        # Amyloid pathway affects amyloid PET
        if 'Amyloid' in pathway_changes:
            # Negative change in amyloid pathway reduces amyloid SUVr
            change = -pathway_changes['Amyloid'] * 0.3  # Scale factor
            post_treatment_pet[region]['amyloid_suvr'] += change
        
        # Tau pathway affects tau PET
        if 'Tau' in pathway_changes:
            # Negative change in tau pathway reduces tau SUVr
            change = -pathway_changes['Tau'] * 0.2  # Scale factor
            post_treatment_pet[region]['tau_suvr'] += change
        
        # Multiple pathways can affect atrophy
        atrophy_change = 0
        if 'Apoptosis' in pathway_changes:
            atrophy_change += pathway_changes['Apoptosis'] * 0.1
        if 'Autophagy' in pathway_changes:
            atrophy_change -= pathway_changes['Autophagy'] * 0.05
        if 'Synaptic' in pathway_changes:
            atrophy_change -= pathway_changes['Synaptic'] * 0.08
            
        post_treatment_pet[region]['atrophy'] += atrophy_change
        
        # Ensure values are in reasonable ranges
        post_treatment_pet[region]['amyloid_suvr'] = max(1.0, min(post_treatment_pet[region]['amyloid_suvr'], 3.0))
        post_treatment_pet[region]['tau_suvr'] = max(1.0, min(post_treatment_pet[region]['tau_suvr'], 3.0))
        post_treatment_pet[region]['atrophy'] = max(0.0, min(post_treatment_pet[region]['atrophy'], 0.5))
    
    return {
        'baseline': baseline_pet,
        'post_treatment': post_treatment_pet,
        'condition': condition
    }

def evaluate_drug_efficacy(net, output_list, drug_name=None, drug_targets=None, conditions=None, output_dir=None):
    """
    Evaluate the efficacy of a drug across multiple conditions.
    
    Arguments:
    net -- Boolean network
    output_list -- List of genes in the network
    drug_name -- Name of known drug (if using predefined targets)
    drug_targets -- List of (target, effect) tuples for custom drugs
    conditions -- List of conditions to test (default: ["APOE4", "Normal"])
    output_dir -- Directory to save outputs (default: drug name or "custom_drug")
    
    Returns:
    Dictionary with efficacy results for each condition
    """
    if conditions is None:
        conditions = ["APOE4", "Normal"]
    
    if output_dir is None:
        if drug_name:
            output_dir = f"{drug_name.lower()}_results"
        else:
            output_dir = "custom_drug_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results for each condition
    results = {}
    
    for condition in conditions:
        condition_dir = os.path.join(output_dir, condition.lower())
        os.makedirs(condition_dir, exist_ok=True)
        
        # Get baseline state
        baseline = get_baseline_state(net, output_list, condition)
        
        # Simulate drug effect
        if drug_name:
            drug_state = simulate_drug_effect(net, output_list, drug_name=drug_name, condition=condition)
        else:
            drug_state = simulate_drug_effect(net, output_list, drug_targets=drug_targets, condition=condition)
        
        # Calculate efficacy
        efficacy = calculate_comprehensive_efficacy(
            baseline, drug_state, 
            drug_name if drug_name else "CustomDrug", 
            condition, output_list, 
            drug_targets if not drug_name else None
        )
        
        # Generate simulated PET scans
        pet_data = generate_pet_data(efficacy, condition)
        
        # Save PET scan visualizations
        pet_images = visualize_pet_scan(
            pet_data['baseline'], 
            pet_data['post_treatment'],
            os.path.join(condition_dir, "pet_visualizations")
        )
        
        # Store all results
        results[condition] = {
            'efficacy_score': efficacy['efficacy_score'],
            'composite_score': efficacy['composite_score'],
            'pathway_scores': efficacy['pathway_scores'],
            'node_changes': efficacy['node_changes'],
            'pathway_changes': efficacy['pathway_changes'],
            'pet_data': pet_data,
            'pet_images': pet_images
        }
        
        # Generate comprehensive report for this condition
        create_efficacy_report(efficacy, pet_data, os.path.join(condition_dir, "efficacy_report.txt"))
    
    # Calculate summary metrics across conditions
    results['summary'] = {
        'average_efficacy': np.mean([results[c]['efficacy_score'] for c in conditions]),
        'average_composite': np.mean([results[c]['composite_score'] for c in conditions if 'composite_score' in results[c]]),
        'conditions_evaluated': conditions
    }
    
    return results

def create_efficacy_report(efficacy_data, pet_data, output_file):
    """
    Create a comprehensive text report of drug efficacy.
    
    Arguments:
    efficacy_data -- Results from efficacy calculation
    pet_data -- PET scan data
    output_file -- Path to save the report
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"ALZHEIMER'S DISEASE DRUG EFFICACY REPORT\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write general information
        f.write(f"CONDITION: {pet_data['condition']}\n\n")
        
        # Write efficacy scores
        f.write("-" * 80 + "\n")
        f.write("EFFICACY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Efficacy Score: {efficacy_data['efficacy_score']:.4f}\n")
        f.write(f"Composite Score: {efficacy_data['composite_score']:.4f}\n")
        
        if 'druggability_score' in efficacy_data:
            f.write(f"Druggability Score: {efficacy_data['druggability_score']:.4f}\n")
        
        if 'pk_score' in efficacy_data and efficacy_data['pk_score'] is not None:
            f.write(f"Pharmacokinetic Score: {efficacy_data['pk_score']:.4f}\n")
            
        if 'safety_score' in efficacy_data and efficacy_data['safety_score'] is not None:
            f.write(f"Safety Score: {efficacy_data['safety_score']:.4f}\n")
        
        # Write pathway scores
        f.write("\n" + "-" * 80 + "\n")
        f.write("PATHWAY EFFECTS\n")
        f.write("-" * 80 + "\n")
        
        if 'pathway_scores' in efficacy_data:
            for pathway, score in efficacy_data['pathway_scores'].items():
                f.write(f"{pathway} Pathway Effect: {score:.4f}\n")
        
        # Write detailed node changes
        f.write("\n" + "-" * 80 + "\n")
        f.write("GENE-LEVEL CHANGES\n")
        f.write("-" * 80 + "\n")
        
        if 'node_changes' in efficacy_data:
            for node, change in sorted(efficacy_data['node_changes'].items(), 
                                       key=lambda x: abs(x[1]), reverse=True):
                direction = "UP" if change > 0 else "DOWN"
                f.write(f"{node}: {change:+.4f} ({direction})\n")
        
        # Write PET scan results
        f.write("\n" + "-" * 80 + "\n")
        f.write("BIOMARKER CHANGES BY BRAIN REGION\n")
        f.write("-" * 80 + "\n")
        
        # Amyloid changes
        f.write("\nAMYLOID PET SUVr VALUES:\n")
        for region in sorted(pet_data['baseline'].keys()):
            baseline = pet_data['baseline'][region]['amyloid_suvr']
            post = pet_data['post_treatment'][region]['amyloid_suvr']
            change = post - baseline
            f.write(f"  {region}: {baseline:.2f} → {post:.2f} ({change:+.2f})\n")
        
        # Tau changes
        f.write("\nTAU PET SUVr VALUES:\n")
        for region in sorted(pet_data['baseline'].keys()):
            baseline = pet_data['baseline'][region]['tau_suvr']
            post = pet_data['post_treatment'][region]['tau_suvr']
            change = post - baseline
            f.write(f"  {region}: {baseline:.2f} → {post:.2f} ({change:+.2f})\n")
        
        # Atrophy changes
        f.write("\nBRAIN ATROPHY MEASURES:\n")
        for region in sorted(pet_data['baseline'].keys()):
            baseline = pet_data['baseline'][region]['atrophy']
            post = pet_data['post_treatment'][region]['atrophy']
            change = post - baseline
            f.write(f"  {region}: {baseline:.2f} → {post:.2f} ({change:+.2f})\n")
        
        # Write footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")

class PETScanGenerator:
    """
    Generate realistic PET scan DICOM files based on drug efficacy simulation results.
    """
    
    # Brain region coordinates (simplified 3D model)
    # Format: [x_center, y_center, z_center, radius]
    BRAIN_REGIONS_3D = {
        'hippocampus': {
            'coords': [64, 40, 40, 8],
            'weight': 0.25
        },
        'entorhinal_cortex': {
            'coords': [64, 52, 38, 6],
            'weight': 0.2
        },
        'prefrontal_cortex': {
            'coords': [64, 80, 50, 12],
            'weight': 0.15
        },
        'temporal_lobe': {
            'coords': [40, 60, 40, 10],
            'weight': 0.15
        },
        'temporal_lobe_right': {
            'coords': [88, 60, 40, 10],
            'weight': 0.15
        },
        'parietal_lobe': {
            'coords': [64, 70, 70, 12],
            'weight': 0.1
        },
        'posterior_cingulate': {
            'coords': [64, 60, 55, 6],
            'weight': 0.1
        },
        'precuneus': {
            'coords': [64, 50, 60, 8],
            'weight': 0.05
        }
    }
    
    # Baseline SUVr values for different conditions
    BASELINE_SUVR = {
        'Normal': {
            'amyloid': 1.2,
            'tau': 1.1
        },
        'APOE4': {
            'amyloid': 1.8,
            'tau': 1.6
        },
        'LPL': {
            'amyloid': 1.5,
            'tau': 1.3
        }
    }
    
    def __init__(self, output_dir="pet_scans"):
        """
        Initialize the PET scan generator.
        
        Args:
            output_dir: Directory to save the DICOM files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # PET image dimensions
        self.img_shape = (128, 128, 80)  # x, y, z
    
    def _create_brain_mask(self):
        """
        Create a binary mask of the brain.
        
        Returns:
            3D numpy array with brain mask
        """
        mask = np.zeros(self.img_shape, dtype=np.float32)
        
        # Create ellipsoid for brain
        center = (self.img_shape[0]//2, self.img_shape[1]//2, self.img_shape[2]//2)
        radii = (50, 65, 40)
        
        for x in range(self.img_shape[0]):
            for y in range(self.img_shape[1]):
                for z in range(self.img_shape[2]):
                    # Ellipsoid equation
                    if ((x - center[0])**2 / radii[0]**2 +
                        (y - center[1])**2 / radii[1]**2 +
                        (z - center[2])**2 / radii[2]**2) <= 1:
                        mask[x, y, z] = 1.0
        
        return mask
    
    def _create_region_values(self, region_values, tracer_type):
        """
        Create a 3D array with region-specific values.
        
        Args:
            region_values: Dictionary of region names and their values
            tracer_type: 'amyloid' or 'tau'
            
        Returns:
            3D numpy array with region values
        """
        # Initialize with background SUVr
        background = 0.8 if tracer_type == 'amyloid' else 0.7
        img = np.ones(self.img_shape, dtype=np.float32) * background
        
        # Fill each region with its value
        for region, value in region_values.items():
            if region in self.BRAIN_REGIONS_3D:
                region_info = self.BRAIN_REGIONS_3D[region]
                coords = region_info['coords']
                x, y, z, r = coords
                
                # Create spherical region
                for i in range(max(0, x-r), min(self.img_shape[0], x+r)):
                    for j in range(max(0, y-r), min(self.img_shape[1], y+r)):
                        for k in range(max(0, z-r), min(self.img_shape[2], z+r)):
                            # Check if point is within sphere
                            if ((i-x)**2 + (j-y)**2 + (k-z)**2) <= r**2:
                                img[i, j, k] = value
        
        # Apply brain mask
        brain_mask = self._create_brain_mask()
        img = img * brain_mask
        
        # Apply smoothing to make it more realistic
        img = gaussian_filter(img, sigma=1.0)
        
        return img
    
    def _create_dicom_file(self, img_data, patient_id, tracer_type, stage, output_file):
        """
        Create a DICOM file from image data.
        
        Args:
            img_data: 3D numpy array with image data
            patient_id: Patient ID string
            tracer_type: 'amyloid' or 'tau'
            stage: 'baseline' or 'post_treatment'
            output_file: Path to save the DICOM file
            
        Returns:
            Path to the created DICOM file
        """
        # Create file meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.128'  # PET Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Create dataset
        ds = FileDataset(output_file, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Add required DICOM tags
        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = '19600101'  # Dummy date
        
        # Study information
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        ds.StudyInstanceUID = generate_uid()
        ds.StudyID = f"PET_{tracer_type}_{stage}"
        ds.Modality = 'PT'  # PET
        
        # Series information
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 1
        
        # Image information
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = img_data.shape[1]
        ds.Columns = img_data.shape[0]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        # Scale pixel values to uint16 range
        min_val = img_data.min()
        max_val = img_data.max()
        img_scaled = ((img_data - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        
        # Add all slices
        ds.NumberOfFrames = img_data.shape[2]
        ds.PixelData = img_scaled.tobytes()
        
        # Add custom tags for SUVr values
        ds.RescaleIntercept = min_val
        ds.RescaleSlope = (max_val - min_val) / 65535
        
        # Store metadata about the scan
        ds.SeriesDescription = f"{tracer_type.capitalize()} PET ({stage})"
        
        # Save the file
        ds.save_as(output_file)
        
        return output_file
        
    def generate_pet_images(self, pet_data, patient_id, output_dir=None):
        """
        Generate DICOM PET images for baseline and post-treatment data.
        
        Args:
            pet_data: Dictionary with baseline and post-treatment PET data
            patient_id: Patient ID string
            output_dir: Directory to save DICOM files (default: self.output_dir)
            
        Returns:
            Dictionary with paths to generated DICOM files
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create amyloid PET images
        amyloid_baseline = {}
        amyloid_post = {}
        
        for region in pet_data['baseline']:
            amyloid_baseline[region] = pet_data['baseline'][region]['amyloid_suvr']
            amyloid_post[region] = pet_data['post_treatment'][region]['amyloid_suvr']
            
        amyloid_baseline_img = self._create_region_values(amyloid_baseline, 'amyloid')
        amyloid_post_img = self._create_region_values(amyloid_post, 'amyloid')
        
        amyloid_baseline_file = os.path.join(output_dir, f"{patient_id}_amyloid_baseline.dcm")
        amyloid_post_file = os.path.join(output_dir, f"{patient_id}_amyloid_post.dcm")
        
        self._create_dicom_file(amyloid_baseline_img, patient_id, 'amyloid', 'baseline', amyloid_baseline_file)
        self._create_dicom_file(amyloid_post_img, patient_id, 'amyloid', 'post_treatment', amyloid_post_file)
        
        # Create tau PET images
        tau_baseline = {}
        tau_post = {}
        
        for region in pet_data['baseline']:
            tau_baseline[region] = pet_data['baseline'][region]['tau_suvr']
            tau_post[region] = pet_data['post_treatment'][region]['tau_suvr']
            
        tau_baseline_img = self._create_region_values(tau_baseline, 'tau')
        tau_post_img = self._create_region_values(tau_post, 'tau')
        
        tau_baseline_file = os.path.join(output_dir, f"{patient_id}_tau_baseline.dcm")
        tau_post_file = os.path.join(output_dir, f"{patient_id}_tau_post.dcm")
        
        self._create_dicom_file(tau_baseline_img, patient_id, 'tau', 'baseline', tau_baseline_file)
        self._create_dicom_file(tau_post_img, patient_id, 'tau', 'post_treatment', tau_post_file)
        
        return {
            'amyloid_baseline': amyloid_baseline_file,
            'amyloid_post': amyloid_post_file,
            'tau_baseline': tau_baseline_file,
            'tau_post': tau_post_file
        }
    
    