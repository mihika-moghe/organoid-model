import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import pickle
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import datetime
from efficacy import (
    load_network,
    get_baseline_state,
    simulate_drug_effect,
    calculate_efficacy,
    calculate_comprehensive_efficacy,
    generate_brain_pet_scan,
    visualize_pet_scan,
    PETScanGenerator,
    DRUG_TARGETS,
    CLINICAL_EFFICACY,
    PHARMACOKINETICS,
    AD_OUTPUT_NODES,
    PATHWAYS,
    BRAIN_REGIONS
)

class TemporalDrugSimulation:
    
    def __init__(self, network_file="A_model.txt", output_dir="temporal_simulation"):
       
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load network
        print("Loading network model...")
        network_data = load_network(network_file)
        self.net = network_data['net']
        self.output_list = network_data['output_list']
        
        # Define timepoints in months
        self.timepoints = [1, 6, 12, 36]  # 1 month, 6 months, 1 year, 3 years
        
        # Natural disease progression rates (based on literature)
        # Values indicate monthly worsening rate for each pathway
        self.progression_rates = {
            "Normal": {
                "Amyloid": 0.0045,  # 0.45% increase per month - based on PET longitudinal data
                "Tau": 0.0035,      # 0.35% increase per month - based on CSF tau studies
                "Apoptosis": 0.0025,
                "Autophagy": -0.0018,
                "Lipid": 0.0020,
                "Synaptic": -0.0028,
                "Neuroinflammation": 0.0037,
                "Oxidative_Stress": 0.0032,
                "Insulin_Signaling": -0.0019,
                "Cholinergic": -0.0031,  # Added for drugs like donepezil, galantamine
                "NMDA": 0.0029,         # Added for memantine
                "Neurotransmitter": -0.0027,  # General neurotransmitter pathway
                "Mitochondrial": 0.0024,  # Added for metabolic dysfunction
                "Vascular": 0.0022,      # Added for vascular component
                "Proteasome": -0.0016    # Added for protein clearance pathway
            },
            "APOE4": {
                "Amyloid": 0.0085,  # 0.85% increase per month (faster in APOE4 carriers) 
                "Tau": 0.0072,      # 0.72% increase per month
                "Apoptosis": 0.0056,
                "Autophagy": -0.0038,
                "Lipid": 0.0063,    # More lipid dysregulation in APOE4
                "Synaptic": -0.0059,
                "Neuroinflammation": 0.0068,
                "Oxidative_Stress": 0.0061,
                "Insulin_Signaling": -0.0034,
                "Cholinergic": -0.0058,
                "NMDA": 0.0053,
                "Neurotransmitter": -0.0049,
                "Mitochondrial": 0.0057,  # Greater mitochondrial dysfunction
                "Vascular": 0.0051,       # Greater vascular component
                "Proteasome": -0.0035    # Worse protein clearance
            },
            "LPL": {
                "Amyloid": 0.0067,  # Between Normal and APOE4
                "Tau": 0.0054,      
                "Apoptosis": 0.0047,
                "Autophagy": -0.0029,
                "Lipid": 0.0079,    # Higher lipid dysregulation in LPL
                "Synaptic": -0.0049,
                "Neuroinflammation": 0.0051,
                "Oxidative_Stress": 0.0042,
                "Insulin_Signaling": -0.0031,
                "Cholinergic": -0.0043,
                "NMDA": 0.0041,
                "Neurotransmitter": -0.0039,
                "Mitochondrial": 0.0046,
                "Vascular": 0.0042,
                "Proteasome": -0.0027
            }
        }
        
        # Regions affected over time (different progression by region)
        # Based on Braak staging and literature
        self.region_progression = {
            "hippocampus": 1.25,          # Fastest progression
            "entorhinal_cortex": 1.18,
            "temporal_lobe": 1.05,
            "posterior_cingulate": 0.92,
            "prefrontal_cortex": 0.78,
            "parietal_lobe": 0.72,
            "precuneus": 0.68
        }
        
        # Updated drug mechanism definitions based on pharmacology literature
        self.drug_mechanisms = {
            "Ritzaganine": {
                "primary_target": "BACE1",
                "primary_effect": "BACE1 inhibitor with additional tau and anti-inflammatory effects",
                "effect_on_clearance": 0.78,    # Strong clearance of amyloid
                "effect_on_tau": 0.25,          # Moderate downstream effect on tau
                "effect_on_cognitive": 0.27,    # Modest cognitive effect (based on clinical data)
                "effect_on_synaptic": 0.31,     # Modest secondary synaptic effect
                "effect_on_neuroinflammation": 0.42,  # Significant anti-inflammatory component
                "tolerance_rate": 0.015,        # Low tolerance development
                "persistence_factor": 0.92,     # Excellent persistence
                "onset_delay": 14,              # Days until clinical effect
                "efficacy_peak": 90,            # Days until peak efficacy
                "side_effects_rate": 0.12,      # Rate of side effects
                "dual_mechanism_benefit": 0.18  # Additional benefit from dual mechanism
            },
            "Memantine": {
                "primary_target": "NMDA",
                "primary_effect": "NMDA receptor antagonism, preferential blockade of extrasynaptic receptors",
                "effect_on_synaptic": 0.32,     # Moderate effect on synaptic function
                "effect_on_excitotoxicity": 0.68, # Strong effect on excitotoxicity
                "effect_on_cognitive": 0.12,    # Modest effect on cognitive function (trials)
                "effect_on_tau": 0.08,          # Minimal effect on tau
                "effect_on_neuroinflammation": 0.22, # Moderate anti-inflammatory effect
                "tolerance_rate": 0.025,        # Low tolerance development
                "persistence_factor": 0.88,     # Good persistence
                "onset_delay": 14,              # Days until clinical effect
                "efficacy_peak": 60,            # Days until peak efficacy
                "side_effects_rate": 0.05,      # Rate of side effects
                "calcium_reduction": 0.45       # Reduction in excess calcium
            },
            "Donepezil": {
                "primary_target": "Cholinergic",
                "primary_effect": "Acetylcholinesterase inhibition",
                "effect_on_synaptic": 0.58,     # Strong effect on synaptic function
                "effect_on_cognitive": 0.25,    # Moderate effect on cognitive function
                "effect_on_attention": 0.65,    # Strong effect on attention
                "effect_on_neuroinflammation": 0.15, # Weak anti-inflammatory effect
                "effect_on_amyloid": 0.10,      # Very small effect on amyloid
                "tolerance_rate": 0.037,        # Moderate tolerance development
                "persistence_factor": 0.82,     # Good persistence
                "onset_delay": 7,               # Days until clinical effect
                "efficacy_peak": 42,            # Days until peak efficacy
                "side_effects_rate": 0.17,      # Rate of side effects (mainly GI)
                "ach_increase": 0.72            # Increase in acetylcholine levels
            },
            "Galantamine": {
                "primary_target": "Cholinergic",
                "primary_effect": "Acetylcholinesterase inhibition and nicotinic receptor modulation",
                "effect_on_synaptic": 0.53,     # Slightly less effect on synaptic vs donepezil
                "effect_on_cognitive": 0.21,    # Slightly less effect on cognitive function
                "effect_on_attention": 0.57,    # Good effect on attention
                "effect_on_nicotinic": 0.63,    # Strong additional nicotinic receptor effect
                "effect_on_neuroinflammation": 0.12, # Weak anti-inflammatory
                "tolerance_rate": 0.045,        # Slightly faster tolerance
                "persistence_factor": 0.75,     # Slightly less persistence
                "onset_delay": 5,               # Days until clinical effect
                "efficacy_peak": 35,            # Days until peak efficacy
                "side_effects_rate": 0.24,      # Higher rate of side effects than donepezil
                "ach_increase": 0.65,           # Increase in acetylcholine levels
                "dual_mechanism_benefit": 0.12  # Additional benefit from dual mechanism
            }
        }
    
    def simulate_drug_over_time(self, drug_name=None, drug_targets=None, 
                  condition="APOE4", include_visuals=True):
    
        print(f"\nSimulating {'known drug: ' + drug_name if drug_name else 'custom drug'} "
            f"over time in {condition} condition...")
        
        # Get drug information
        drug_info = {}
        if drug_name and drug_name in DRUG_TARGETS:
            drug_info["name"] = drug_name
            drug_info["targets"] = [(t["target"], t["effect"]) for t in DRUG_TARGETS[drug_name]]
            
            # Get pharmacokinetic properties if available
            if drug_name in PHARMACOKINETICS:
                drug_info["pk"] = PHARMACOKINETICS[drug_name]
            else:
                # Default PK parameters for known drugs without explicit PK data
                # Differentiate between drug classes
                if drug_name in ["Donepezil", "Galantamine"]:  # Cholinesterase inhibitors
                    drug_info["pk"] = {
                        "half_life": 70,           # Hours (longer for Donepezil)
                        "bioavailability": 0.9,    # High oral bioavailability
                        "bbb_penetration": 0.7,    # Good blood-brain barrier penetration
                        "volume_distribution": 12.0
                    }
                elif drug_name == "Memantine":     # NMDA receptor antagonist
                    drug_info["pk"] = {
                        "half_life": 60,           # Hours
                        "bioavailability": 0.8,    # Good oral bioavailability
                        "bbb_penetration": 0.6,    # Moderate blood-brain barrier penetration
                        "volume_distribution": 9.0
                    }
                elif drug_name == "Ritzaganine":   # BACE1 inhibitor
                    drug_info["pk"] = {
                        "half_life": 36,           # Hours
                        "bioavailability": 0.85,   # Good oral bioavailability
                        "bbb_penetration": 0.75,   # Good blood-brain barrier penetration
                        "volume_distribution": 8.5
                    }
                else:  # Default values for other drugs
                    drug_info["pk"] = {
                        "half_life": 24,           # Hours
                        "bioavailability": 0.7,
                        "bbb_penetration": 0.4,    # Blood-brain barrier penetration
                        "volume_distribution": 7.0
                    }
            
            # Add drug mechanism info if available
            if drug_name in self.drug_mechanisms:
                drug_info["mechanism"] = self.drug_mechanisms[drug_name]
        else:
            drug_info["name"] = "Custom Drug"
            drug_info["targets"] = drug_targets if drug_targets else []
            
            # Use default PK parameters for custom drugs
            drug_info["pk"] = {
                "half_life": 24,          # Hours
                "bioavailability": 0.7,
                "bbb_penetration": 0.4,   # Blood-brain barrier penetration
                "volume_distribution": 7.0
            }
        
        # Create directory for this simulation
        sim_name = drug_info["name"].replace(" ", "_").lower()
        sim_dir = f"{self.output_dir}/{sim_name}_{condition.lower()}"
        os.makedirs(sim_dir, exist_ok=True)
        
        # Get baseline attractors for untreated condition
        baseline_attractors = get_baseline_state(self.net, self.output_list, condition)
        
        # Simulate initial drug effect (time = 0)
        # Add special handling for drugs not fully modeled in the original system
        if drug_name in ["Donepezil", "Galantamine", "Memantine", "Ritzaganine"] and not self._is_drug_in_targets(drug_name):
            # For drugs not properly modeled in DRUG_TARGETS, use custom targets
            custom_drug_targets = self._get_custom_drug_targets(drug_name)
            initial_drug_attractors = simulate_drug_effect(
                self.net, self.output_list, drug_targets=custom_drug_targets, condition=condition
            )
        elif drug_name:
            initial_drug_attractors = simulate_drug_effect(
                self.net, self.output_list, drug_name=drug_name, condition=condition
            )
        else:
            initial_drug_attractors = simulate_drug_effect(
                self.net, self.output_list, drug_targets=drug_targets, condition=condition
            )
        
        # Calculate initial efficacy
        initial_efficacy = calculate_comprehensive_efficacy(
            baseline_attractors, initial_drug_attractors, 
            drug_info["name"], condition, self.output_list
        )
        
        # Adjust initial efficacy for drugs that need special handling
        if drug_name in ["Donepezil", "Galantamine", "Memantine", "Ritzaganine"]:
            initial_efficacy = self._adjust_efficacy_for_special_drugs(
                initial_efficacy, drug_name, condition
            )
        
        # Store results for each timepoint
        temporal_results = {
            "drug_info": drug_info,
            "condition": condition,
            "timepoints": self.timepoints,
            "results": {}
        }
        
        # Generate baseline PET scan
        baseline_pet = generate_brain_pet_scan(
            {node: 0 for node in AD_OUTPUT_NODES if node in self.output_list},
            condition=condition,
            stage="baseline",
            drug_name=drug_name  # Pass drug name even for baseline for consistency
        )
        
        # Initial (time = 0) PET scan with drug effect
        initial_pet = generate_brain_pet_scan(
            initial_efficacy['node_changes'],
            condition=condition,
            stage="post_treatment",
            drug_name=drug_name  # Pass drug name
        )
        
        if include_visuals:
            # Visualize baseline vs initial - now with timepoint info
            visualize_pet_scan(baseline_pet, initial_pet, output_dir=f"{sim_dir}/initial", timepoint=0)
        
        # Store initial results
        temporal_results["results"]["initial"] = {
            "efficacy_score": initial_efficacy['efficacy_score'],
            "composite_score": initial_efficacy['composite_score'],
            "pathway_scores": initial_efficacy['pathway_scores'],
            "node_changes": initial_efficacy['node_changes'],
            "pathway_changes": initial_efficacy['pathway_changes'],
            "pet_data": initial_pet
        }
        
        # Simulate each timepoint with unique, time-dependent effects
        for months in self.timepoints:
            print(f"Simulating {months} month{'s' if months > 1 else ''} timepoint...")
            
            # Get adjusted pathway changes based on time
            time_adjusted_changes = self._calculate_time_adjusted_changes(
                initial_efficacy['pathway_changes'],
                drug_info,
                condition,
                months
            )
            
            # Special time-dependent adjustments for specific drugs
            if drug_name in ["Donepezil", "Galantamine", "Memantine", "Ritzaganine"]:
                time_adjusted_changes = self._adjust_drug_specific_temporal_effects(
                    time_adjusted_changes, drug_name, months
                )
            
            # Generate PET scan for this timepoint - using updated function
            # that ensures time-specific results
            timepoint_pet = self._generate_timepoint_pet_scan(
                baseline_pet,
                initial_pet,
                time_adjusted_changes,
                condition,
                months,
                drug_name=drug_name,
                initial_node_changes=initial_efficacy['node_changes']
            )
            
            # Calculate efficacy metrics for this timepoint
            timepoint_efficacy = {
                "efficacy_score": self._adjust_score_for_time(
                    initial_efficacy['efficacy_score'], drug_info, condition, months
                ),
                "pathway_scores": self._adjust_pathway_scores_for_time(
                    initial_efficacy['pathway_scores'], drug_info, condition, months
                ),
                "node_changes": self._calculate_node_changes_at_timepoint(
                    initial_efficacy['node_changes'], time_adjusted_changes, months
                ),
                "pathway_changes": time_adjusted_changes,
                "pet_data": timepoint_pet
            }
            
            # Calculate composite score
            timepoint_efficacy["composite_score"] = self._calculate_composite_score(
                timepoint_efficacy
            )
            
            # Visualize if requested - now with timepoint info
            if include_visuals:
                # Compare to baseline
                visualize_pet_scan(
                    baseline_pet, timepoint_pet,
                    output_dir=f"{sim_dir}/month_{months}",
                    timepoint=months
                )
            
            # Store results
            temporal_results["results"][f"month_{months}"] = timepoint_efficacy
        
        # Generate comparative visualizations across all timepoints
        if include_visuals:
            self._generate_temporal_comparison_visuals(temporal_results, sim_dir)
            # Generate the specific timepoint comparison matrix with proper titles
            self._create_timepoint_comparison_matrix(temporal_results, sim_dir)
            
            # Add drug-specific visualization based on mechanism
            if drug_name == "Ritzaganine":
                # For BACE1 inhibitor, show amyloid reduction and tau effects
                self._create_bace1_inhibitor_plot(temporal_results, sim_dir)
            elif drug_name in ["Donepezil", "Galantamine"]:
                # For cholinergic drugs, show acetylcholine levels and cognitive metrics
                self._create_cholinergic_metrics_plot(temporal_results, sim_dir)
            elif drug_name == "Memantine":
                # For memantine, show NMDA receptor effects
                self._create_excitotoxicity_plot(temporal_results, sim_dir)
        
        # Save results
        results_file = f"{sim_dir}/temporal_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(temporal_results, f)
        
        return temporal_results
    
    def _is_drug_in_targets(self, drug_name):
        """Check if the drug is properly modeled in DRUG_TARGETS"""
        return drug_name in DRUG_TARGETS and len(DRUG_TARGETS[drug_name]) > 0
    
    def _get_custom_drug_targets(self, drug_name):
        """Get custom targets for drugs not fully modeled in the network"""
        if drug_name == "Donepezil":
            return [
                ("AChE", 0),           # Strong inhibition of acetylcholinesterase
                ("ACh", 1),            # Increased acetylcholine levels
                ("Cholinergic", 1),    # Increase in cholinergic signaling
                ("Sigma1", 1),         # Sigma-1 receptor agonism
                ("nAChR", 1),          # Weak positive modulation of nicotinic receptors
                ("mAChR", 1),          # Indirect activation of muscarinic receptors via ACh increase
                ("LTP", 1),            # Enhanced long-term potentiation
                ("BDNF", 1)            # Increased BDNF expression
            ]
        elif drug_name == "Galantamine":
            return [
                ("AChE", 0),           # Inhibition of acetylcholinesterase
                ("ACh", 1),            # Increased acetylcholine levels
                ("Cholinergic", 1),    # Increase in cholinergic signaling
                ("nAChR", 1),          # Direct positive allosteric modulation
                ("nAChR_alpha7", 1),   # Strong effect on alpha7 subtype
                ("nAChR_alpha4beta2", 1), # Effect on alpha4beta2 subtype
                ("BACE1", 0),          # Mild inhibition of BACE1 (secondary effect)
                ("APP_processing", 0)  # Small effect on APP processing
            ]
        elif drug_name == "Memantine":
            return [
                ("NMDAR", 0),          # NMDA receptor antagonism
                ("e_NMDAR", 0),        # Strong blockade of extrasynaptic NMDA receptors
                ("s_NMDAR", 0),        # Weaker blockade of synaptic NMDA receptors
                ("Glutamate_toxicity", 0), # Reduced glutamate excitotoxicity
                ("Ca_ion", 0),         # Reduced calcium influx
                ("ROS", 0),            # Reduced reactive oxygen species
                ("Apoptosis", 0),      # Decreased apoptotic signaling
                ("Calpain", 0),        # Decreased calpain activation
                ("TNFa", 0)            # Mild reduction in TNFa levels
            ]
        elif drug_name == "Ritzaganine":
            return [
                ("BACE1", 0),          # Direct BACE1 inhibition
                ("APP_processing", 0), # Reduced amyloidogenic processing of APP
                ("Abeta_oligomers", 0), # Prevention of oligomer formation
                ("GSK3beta", 0),       # GSK3β inhibition
                ("Tau_phosphorylation", 0), # Reduced tau phosphorylation
                ("Neuroinflammation", 0), # Anti-inflammatory effects
                ("NF-kB", 0),          # Reduced NF-kB signaling
                ("Microglia", 1)       # Enhanced microglial phagocytosis
            ]
        else:
            # For other drugs, return an empty list
            print(f"Warning: No custom targets defined for {drug_name}")
            return []
    
    def _adjust_efficacy_for_special_drugs(self, efficacy, drug_name, condition):
        """Adjust efficacy calculations for drugs requiring special handling"""
        # Make a deep copy of the efficacy dict to avoid modifying the original
        adjusted_efficacy = {
            key: (value.copy() if isinstance(value, dict) else value) 
            for key, value in efficacy.items()
        }
        
        # Adjust pathway changes based on drug mechanism and established clinical data
        if drug_name == "Donepezil":
            # Donepezil primarily affects cholinergic pathways with some effect on amyloid
            if "pathway_changes" in adjusted_efficacy:
                adjusted_efficacy["pathway_changes"].update({
                    "Cholinergic": -0.6,        # Substantial improvement in cholinergic function
                    "Synaptic": -0.5,           # Moderate improvement in synaptic function
                    "Amyloid": -0.15,           # Small improvement in amyloid pathology
                    "Tau": -0.1                 # Minimal effect on tau pathology
                })
            # Adjust efficacy score - Donepezil has moderate efficacy
            adjusted_efficacy["efficacy_score"] = 0.45
            
        elif drug_name == "Galantamine":
            # Galantamine has similar but slightly different effects than Donepezil
            if "pathway_changes" in adjusted_efficacy:
                adjusted_efficacy["pathway_changes"].update({
                    "Cholinergic": -0.55,       # Good improvement in cholinergic function
                    "Synaptic": -0.45,          # Moderate improvement in synaptic function
                    "Amyloid": -0.1,            # Minimal effect on amyloid
                    "Tau": -0.05                # Very minimal effect on tau
                })
            # Adjust efficacy score - slightly less than Donepezil
            adjusted_efficacy["efficacy_score"] = 0.4
            
        elif drug_name == "Memantine":
            # Memantine primarily affects NMDA/glutamate pathways
            if "pathway_changes" in adjusted_efficacy:
                adjusted_efficacy["pathway_changes"].update({
                    "NMDA": -0.65,              # Strong effect on NMDA pathway
                    "Synaptic": -0.3,           # Modest effect on synaptic function
                    "Neuroinflammation": -0.25, # Some anti-inflammatory effect
                    "Oxidative_Stress": -0.3    # Moderate effect on oxidative stress
                })
            # Adjust efficacy score - modest efficacy, especially in moderate-severe AD
            adjusted_efficacy["efficacy_score"] = 0.35
        
        # Adjust node changes to align with pathway changes - no random values
        if "node_changes" in adjusted_efficacy and "pathway_changes" in adjusted_efficacy:
            # Map pathways to related nodes based on established biological relationships
            pathway_to_nodes = {
                "Cholinergic": ["ACh", "AChE", "nAChR", "mAChR"],
                "NMDA": ["NMDAR", "Glutamate", "Calcium"],
                "Synaptic": ["Synapse", "LTP", "LTD", "PSD95"],
                "Amyloid": ["Abeta", "APP", "BACE1"],
                "Tau": ["Tau", "p-Tau"],
                "Neuroinflammation": ["IL1B", "TNFa", "Microglia"],
                "Oxidative_Stress": ["ROS", "SOD", "NRF2"]
            }
            
            # Only update nodes that are both in our mapping and already present in node_changes
            for pathway, change in adjusted_efficacy["pathway_changes"].items():
                if pathway in pathway_to_nodes:
                    for node in pathway_to_nodes[pathway]:
                        if node in adjusted_efficacy["node_changes"]:
                            # Use a fixed proportion of the pathway change
                            adjusted_efficacy["node_changes"][node] = change * 0.8
        
        # Adjust pathway scores to align with efficacy score - based on fixed formula
        if "pathway_scores" in adjusted_efficacy and "pathway_changes" in adjusted_efficacy:
            for pathway, change in adjusted_efficacy["pathway_changes"].items():
                if change < 0:  # Beneficial effect (negative change)
                    score = 0.5 - change  # Convert negative change to positive score
                    adjusted_efficacy["pathway_scores"][pathway] = min(1.0, max(0.0, score))
        
        # Calculate/update composite score
        adjusted_efficacy["composite_score"] = self._calculate_composite_score(adjusted_efficacy)
        
        return adjusted_efficacy
    
    def _adjust_drug_specific_temporal_effects(self, pathway_changes, drug_name, months):
        """Adjusts pathway changes based on specific drug mechanisms over time"""
        
        adjusted_changes = pathway_changes.copy()

        if drug_name == "Ritzaganine":
            # BACE1 inhibitor with additional effects
            # Based on clinical trial data
            
            # Amyloid production inhibition shows rapid effect
            if "Amyloid" in adjusted_changes:
                # Phase 1: Rapid initial reduction (0-3 months)
                if months <= 3:
                    # Quick onset of amyloid reduction
                    amyloid_factor = 1.0 + (0.2 * (months / 3))
                # Phase 2: Continued effect (3-12 months)
                elif months <= 12:
                    amyloid_factor = 1.2 + (0.2 * ((months - 3) / 9))
                # Phase 3: Sustained phase (>12 months)
                else:
                    amyloid_factor = 1.4
                    
                # Apply amyloid reduction factor
                adjusted_changes["Amyloid"] = adjusted_changes["Amyloid"] * amyloid_factor
            
            # Tau effects develop more gradually
            if "Tau" in adjusted_changes:
                # Gradual onset of tau effects
                tau_effect_delay = 1.0 - np.exp(-months / 6)  # Exponential approach to 1.0
                
                # Calculate tau effect
                initial_tau_effect = adjusted_changes["Tau"]
                delayed_tau_effect = initial_tau_effect * tau_effect_delay
                
                # Apply tau effect
                adjusted_changes["Tau"] = delayed_tau_effect
                
            # Anti-inflammatory effects build steadily
            if "Neuroinflammation" in adjusted_changes:
                if months <= 2:
                    # Initial anti-inflammatory effect
                    infl_factor = 1.0 + (0.15 * (months / 2))
                elif months <= 6:
                    # Building effect
                    infl_factor = 1.15 + (0.25 * ((months - 2) / 4))
                else:
                    # Sustained effect
                    infl_factor = 1.4
                    
                # Apply inflammation reduction
                adjusted_changes["Neuroinflammation"] = adjusted_changes["Neuroinflammation"] * infl_factor
                
            # Synaptic effects emerge with delay
            if "Synaptic" in adjusted_changes:
                # Modest improvement in synaptic function follows amyloid reduction
                synapse_lag = 1.0 - np.exp(-months / 4)
                adjusted_changes["Synaptic"] = adjusted_changes["Synaptic"] * synapse_lag
                    
        elif drug_name == "Memantine":
            # NMDA receptor antagonist effects
            # Based on clinical trials and receptor pharmacology
            
            # NMDA receptor antagonism - rapid effect that maintains
            if "NMDA" in adjusted_changes:
                # Quick onset of receptor blockade
                if months <= 1:
                    # Rapid achievement of steady-state blockade
                    nmda_factor = 0.7 + (0.3 * months)
                else:
                    # Maintain effect with slight tolerance
                    tolerance = min(0.15, 0.03 * (months - 1))  # Max 15% tolerance
                    nmda_factor = 1.0 - tolerance
                    
                adjusted_changes["NMDA"] = adjusted_changes["NMDA"] * nmda_factor
                
            # Excitotoxicity protection builds up
            if "Excitotoxicity" in adjusted_changes:
                # Protection increases over early months
                exci_factor = min(1.2, 1.0 + (0.04 * months))
                adjusted_changes["Excitotoxicity"] = adjusted_changes["Excitotoxicity"] * exci_factor
                
            # Cognitive effects emerge gradually
            if "Cognitive" in adjusted_changes:
                # Clinical effect lags behind receptor occupancy
                cog_factor = min(1.0, 0.7 + (0.3 * (1.0 - np.exp(-months / 2))))
                adjusted_changes["Cognitive"] = adjusted_changes["Cognitive"] * cog_factor
                
            # Calcium homeostasis improves over time
            if "Calcium" in adjusted_changes:
                # Gradual normalization of calcium regulation
                ca_factor = min(1.1, 1.0 + (0.02 * months))
                adjusted_changes["Calcium"] = adjusted_changes["Calcium"] * ca_factor
                
            # Neuroinflammation decreases gradually
            if "Neuroinflammation" in adjusted_changes:
                # Anti-inflammatory effect increases with continued treatment
                if "Neuroinflammation" in adjusted_changes:
                    infl_factor = min(1.15, 1.0 + (0.025 * months))
                    adjusted_changes["Neuroinflammation"] = adjusted_changes["Neuroinflammation"] * infl_factor
                    
        elif drug_name == "Donepezil":
            # Acetylcholinesterase inhibitor effects
            # Based on clinical pharmacology and long-term trials
            
            # Cholinergic effect - initial strong effect with tolerance development
            if "Cholinergic" in adjusted_changes:
                # Strong initial effect followed by tolerance
                if months <= 1:
                    # Quick onset to full effect
                    chol_factor = 0.8 + (0.2 * months)
                else:
                    # Gradual tolerance development
                    # More realistic modeling of enzyme adaptation
                    tolerance = min(0.25, 0.04 * np.log(months + 1))  # Logarithmic tolerance development
                    chol_factor = 1.0 - tolerance
                    
                adjusted_changes["Cholinergic"] = adjusted_changes["Cholinergic"] * chol_factor
                
            # Synaptic effects peak early then partially decline
            if "Synaptic" in adjusted_changes:
                if months <= 3:
                    # Initial enhancement
                    syn_factor = 1.0 + (0.1 * (months / 3))
                else:
                    # Partial accommodation
                    syn_factor = 1.1 - (0.15 * (1.0 - np.exp(-(months - 3) / 6)))
                    syn_factor = max(0.95, syn_factor)  # Maintain most benefit
                    
                adjusted_changes["Synaptic"] = adjusted_changes["Synaptic"] * syn_factor
                
            # Subtle influence on amyloid processing emerges slowly
            if "Amyloid" in adjusted_changes:
                # Very mild effect that takes time to develop
                amy_factor = min(1.2, 1.0 + (0.03 * np.log(months + 1)))
                adjusted_changes["Amyloid"] = adjusted_changes["Amyloid"] * amy_factor
                
            # Attention effects robust but subject to some accommodation
            if "Attention" in adjusted_changes:
                if months <= 2:
                    # Quick improvement
                    att_factor = 1.0
                else:
                    # Slight diminishing of attention enhancement
                    att_factor = 1.0 - (0.1 * (1.0 - np.exp(-(months - 2) / 12)))
                    att_factor = max(0.9, att_factor)  # Maintain most benefit
                    
                adjusted_changes["Attention"] = adjusted_changes["Attention"] * att_factor
                
            # Side effects (mainly GI) tend to diminish over time
            if "GI_side_effects" in adjusted_changes:
                # Adaptation to GI effects
                gi_factor = max(0.6, 1.0 - (0.4 * (1.0 - np.exp(-months / 2))))
                adjusted_changes["GI_side_effects"] = adjusted_changes["GI_side_effects"] * gi_factor
                
        elif drug_name == "Galantamine":
            # Combined AChE inhibition and nicotinic modulation
            # Based on clinical pharmacology studies
            
            # Cholinergic effect - similar to donepezil but with faster tolerance
            if "Cholinergic" in adjusted_changes:
                if months <= 1:
                    # Quick onset to full effect
                    chol_factor = 0.8 + (0.2 * months)
                else:
                    # Slightly faster tolerance development than donepezil
                    tolerance = min(0.3, 0.05 * np.log(months + 1))
                    chol_factor = 1.0 - tolerance
                    
                adjusted_changes["Cholinergic"] = adjusted_changes["Cholinergic"] * chol_factor
                
            # Nicotinic receptor modulation - slower tolerance
            if "Nicotinic" in adjusted_changes:
                if months <= 1:
                    # Rapid onset
                    nic_factor = 0.85 + (0.15 * months)
                else:
                    # Slower tolerance development for nicotinic effects
                    tolerance = min(0.2, 0.035 * np.log(months + 1))
                    nic_factor = 1.0 - tolerance
                    
                adjusted_changes["Nicotinic"] = adjusted_changes["Nicotinic"] * nic_factor
                
            # Unique benefit from dual mechanism persists better
            if "Dual_mechanism" in adjusted_changes:
                dual_factor = min(1.1, 1.0 + (0.02 * np.log(months + 1)))
                adjusted_changes["Dual_mechanism"] = adjusted_changes["Dual_mechanism"] * dual_factor
                
            # Side effects more persistent than donepezil
            if "GI_side_effects" in adjusted_changes:
                # Less adaptation to GI effects
                gi_factor = max(0.75, 1.0 - (0.25 * (1.0 - np.exp(-months / 3))))
                adjusted_changes["GI_side_effects"] = adjusted_changes["GI_side_effects"] * gi_factor

        # Handle any custom drugs - using a more general approach
        else:
            # Check if we have any information in drug_mechanisms
            if drug_name in self.drug_mechanisms:
                mechanism = self.drug_mechanisms[drug_name]
                
                # General tolerance model based on mechanism properties
                if "tolerance_rate" in mechanism and "persistence_factor" in mechanism:
                    tolerance_rate = mechanism["tolerance_rate"]
                    persistence = mechanism["persistence_factor"]
                    
                    # Apply to all pathways with drug-specific parameters
                    for pathway in adjusted_changes:
                        if months <= 1:
                            # Initial effect buildup
                            factor = 0.8 + (0.2 * months)
                        else:
                            # Custom tolerance model
                            tolerance = min(0.4, tolerance_rate * np.log(months + 1))
                            # Adjust by persistence factor
                            tolerance = tolerance * (1.0 - persistence)
                            factor = 1.0 - tolerance
                            
                        adjusted_changes[pathway] = adjusted_changes[pathway] * factor
            else:
                # For completely unknown drugs, use a generic model
                # Moderate onset, moderate tolerance development
                for pathway in adjusted_changes:
                    if months <= 1:
                        factor = 0.8 + (0.2 * months)
                    else:
                        # Generic tolerance model
                        tolerance = min(0.3, 0.04 * np.log(months + 1))
                        factor = 1.0 - tolerance
                        
                    adjusted_changes[pathway] = adjusted_changes[pathway] * factor
                        
        return adjusted_changes
    
    def _create_bace1_inhibitor_plot(self, temporal_results, output_dir):
        """Create a special plot for BACE1 inhibitor effects (Ritzaganine)"""
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        timepoints = [0] + temporal_results["timepoints"]
        
        # Extract relevant data
        amyloid_reduction = []
        tau_effects = []
        cognitive_effects = []
        
        for i, months in enumerate(timepoints):
            t = "initial" if i == 0 else f"month_{months}"
            
            # Extract data points
            if i == 0:
                # Initial values (baseline)
                amyloid_reduction.append(0)
                tau_effects.append(0)
            else:
                # Amyloid inhibition - models BACE1 inhibition dynamics
                # Rapid onset with sustained effect
                if months <= 3:
                    amyloid_effect = 0.67 * (1 - np.exp(-1.5 * months / 3))
                else:
                    amyloid_effect = 0.67 * (1 - 0.1 * np.exp(-(months - 3) / 6))
                
                amyloid_reduction.append(amyloid_effect)
                
                # Tau effects - delayed but significant
                if months <= 2:
                    tau_effect = 0.38 * 0.3 * (months / 2)  # Initial slow onset
                else:
                    tau_effect = 0.38 * (0.3 + 0.7 * (1 - np.exp(-(months - 2) / 6)))
                
                tau_effects.append(tau_effect)
            
            # Cognitive effect from efficacy scores
            if t in temporal_results["results"]:
                result = temporal_results["results"][t]
                if "efficacy_score" in result:
                    cognitive_effects.append(result["efficacy_score"])
                else:
                    cognitive_effects.append(float('nan'))
            else:
                cognitive_effects.append(float('nan'))
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        
        # Multiple y-axes for the different metrics
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot amyloid inhibition effects
        line1 = ax1.plot(timepoints, amyloid_reduction, 'o-', color='#2E7D32',  # Darker green
                        linewidth=2, markersize=8, label='Amyloid Reduction')
        ax1.set_ylabel('Amyloid Reduction Effect', color='#2E7D32', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#2E7D32')
        
        # Plot tau effects on the same axis
        line2 = ax1.plot(timepoints, tau_effects, 's--', color='#4CAF50',  # Medium green
                        linewidth=2, markersize=8, label='Tau Effect')
        
        # Plot cognitive effect on second axis
        line3 = ax2.plot(timepoints, cognitive_effects, '^-', color='#1976D2',  # Medium blue
                        linewidth=2, markersize=8, label='Cognitive Effect')
        ax2.set_ylabel('Cognitive Efficacy', color='#1976D2', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#1976D2')
        
        # Add title and labels
        plt.title(f"{drug_name}: Mechanism Effects Over Time ({condition})", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bace1_inhibitor_effects.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_time_adjusted_changes(self, initial_pathway_changes, drug_info, condition, months):
       
        # Get progression rates for this condition
        progression = self.progression_rates[condition]
        
        # Calculate drug persistence factor based on half-life
        # For longer-acting drugs, effects persist longer
        if "pk" in drug_info and "half_life" in drug_info["pk"]:
            half_life_months = drug_info["pk"]["half_life"] / (24 * 30)  # Convert hours to months
            
            # For drugs with very long half-lives (e.g., antibodies), effects persist longer
            if half_life_months > 1:
                persistence = np.exp(-0.5 * months / half_life_months)
            else:
                # For typical small molecules, persistence declines faster but plateaus
                # (accounting for continued dosing in clinical practice)
                persistence = 0.3 + 0.7 * np.exp(-0.5 * months / half_life_months)
        else:
            # Default persistence model if PK data not available
            persistence = 0.3 + 0.7 * np.exp(-0.1 * months)
        
        # Apply drug-specific persistence factor if available
        if "mechanism" in drug_info and "persistence_factor" in drug_info["mechanism"]:
            persistence = persistence * drug_info["mechanism"]["persistence_factor"]
        
        # Calculate time-adjusted changes for each pathway
        adjusted_changes = {}
        for pathway, initial_change in initial_pathway_changes.items():
            # Get natural progression rate for this pathway
            monthly_progression = progression.get(pathway, 0.004)  # Default if not specified
            
            # Calculate cumulative disease progression
            # This is positive for worsening pathways, negative for diminishing protective pathways
            cumulative_progression = monthly_progression * months
            
            # Beneficial changes (negative values) are reduced over time due to disease progression
            # Harmful changes (positive values) are amplified over time
            if initial_change < 0:  # Beneficial change (reduction in pathology)
                # Drug effect diminishes over time but still helps counter progression
                drug_effect = initial_change * persistence
                
                # Net change is drug effect plus disease progression (which works against drug)
                adjusted_changes[pathway] = drug_effect + cumulative_progression
            else:  # Harmful or neutral change
                # Both drug effect and disease progression worsen the pathway
                adjusted_changes[pathway] = (initial_change * persistence) + cumulative_progression
        
        return adjusted_changes

    def _generate_timepoint_pet_scan(self, baseline_pet, initial_pet, 
                      adjusted_pathway_changes, condition, months, 
                      drug_name=None, initial_node_changes=None):
   
        # Ensure we have valid initial_node_changes to work with
        if initial_node_changes is None or not initial_node_changes:
            # If no initial node changes are provided, derive them from available data
            # This uses pathway information to create realistic node changes
            derived_node_changes = {}
            for pathway, change in adjusted_pathway_changes.items():
                if pathway in PATHWAYS:
                    # Apply changes to all nodes in the pathway
                    for node in PATHWAYS[pathway]:
                        # Scale the effect based on the node's role in the pathway
                        # Use a deterministic approach based on node name
                        node_importance = 0.8 + 0.2 * (hash(node) % 10) / 10.0  # Between 0.8 and 1.0
                        derived_node_changes[node] = change * node_importance
            
            initial_node_changes = derived_node_changes
        
        # Calculate node changes at this timepoint using the initial values
        node_changes_at_timepoint = self._calculate_node_changes_at_timepoint(
            initial_node_changes,
            adjusted_pathway_changes,
            months
        )
        
        # Generate a unique random seed for this timepoint to ensure different but reproducible results
        time_seed = hash(f"{drug_name}_{condition}_{months}") % 10000
        np.random.seed(time_seed)
        
        # Generate the timepoint PET scan with time-specific parameters
        timepoint_pet = generate_brain_pet_scan(
            node_changes_at_timepoint,
            condition=condition,
            stage=f"month_{months}",  # Use the actual month in the stage
            drug_name=drug_name
        )
        
        # Add special time-dependent effects
        if drug_name == "Lecanemab":
            # Time-dependent ARIA risk affects some regions
            if months >= 1 and months <= 6:
                # ARIA risk peaks in the first 6 months
                aria_factor = min(0.3, 0.05 * months)
                
                # Apply ARIA effect to regions (may counteract amyloid clearance in some regions)
                for region in ['temporal_lobe', 'entorhinal_cortex', 'posterior_cingulate']:
                    if region in timepoint_pet:
                        # ARIA can increase apparent amyloid signal in affected regions
                        inflammation_effect = aria_factor * np.random.uniform(0.1, 0.5)
                        timepoint_pet[region]['amyloid_suvr'] += inflammation_effect
                        
                        # Ensure values stay within realistic ranges
                        timepoint_pet[region]['amyloid_suvr'] = max(1.0, min(timepoint_pet[region]['amyloid_suvr'], 3.0))
        
        elif drug_name in ["Donepezil", "Galantamine"]:
            # Cholinesterase inhibitors have diminishing effects over time in some regions
            if months > 3:
                tolerance_factor = min(0.3, 0.04 * np.log(months))
                
                # Apply tolerance effect to regions
                for region in timepoint_pet:
                    if region != 'metadata':
                        # Metabolic improvement diminishes with tolerance
                        if 'hypometabolism' in timepoint_pet[region]:
                            timepoint_pet[region]['hypometabolism'] *= (1.0 - tolerance_factor)
        
        elif drug_name == "Memantine":
            # Memantine works better in more advanced disease and has time-dependent effects
            if months > 1:
                # Greater effect on reducing excitotoxicity over time
                for region in ['hippocampus', 'entorhinal_cortex']:
                    if region in timepoint_pet:
                        # Protection effect increases gradually
                        protection_factor = min(0.2, 0.02 * months)
                        if 'tau_suvr' in timepoint_pet[region]:
                            # Slight tau reduction from neuroprotection
                            timepoint_pet[region]['tau_suvr'] -= protection_factor
                            # Ensure values stay within realistic ranges
                            timepoint_pet[region]['tau_suvr'] = max(1.0, timepoint_pet[region]['tau_suvr'])
        
        # Update metadata to include timepoint information
        if 'metadata' in timepoint_pet:
            timepoint_pet['metadata']['timepoint'] = months
            timepoint_pet['metadata']['stage'] = f"month_{months}"
        
        return timepoint_pet
        
    def _adjust_score_for_time(self, initial_score, drug_info, condition, months):
        
        # Get progression rate based on condition
        progression_rate = 0.004 if condition == "Normal" else 0.007  # Monthly rate
        
        # Calculate drug persistence
        if "pk" in drug_info and "half_life" in drug_info["pk"]:
            half_life_months = drug_info["pk"]["half_life"] / (24 * 30)
            persistence = 0.3 + 0.7 * np.exp(-0.5 * months / half_life_months)
        else:
            persistence = 0.3 + 0.7 * np.exp(-0.1 * months)
        
        # Apply drug-specific persistence or tolerance factors
        if "mechanism" in drug_info:
            if "persistence_factor" in drug_info["mechanism"]:
                persistence = persistence * drug_info["mechanism"]["persistence_factor"]
            
            if "tolerance_rate" in drug_info["mechanism"]:
                # Apply tolerance effect (reduced efficacy over time)
                tolerance_factor = 1.0 - (drug_info["mechanism"]["tolerance_rate"] * months)
                tolerance_factor = max(0.4, tolerance_factor)  # Don't go below 40% efficacy
                persistence = persistence * tolerance_factor
        
        # Calculate adjusted score (typically decreases over time)
        # Initial benefit gradually diminishes, disease continues to progress
        adjusted_score = initial_score * persistence - (progression_rate * months / 10)
        
        # Ensure score remains within valid range
        return max(0, min(adjusted_score, 1.0))

    def _adjust_pathway_scores_for_time(self, initial_scores, drug_info, condition, months):
        
        adjusted_scores = {}
        progression = self.progression_rates[condition]
        
        # Calculate drug persistence (same as in adjust_score_for_time)
        if "pk" in drug_info and "half_life" in drug_info["pk"]:
            half_life_months = drug_info["pk"]["half_life"] / (24 * 30)
            persistence = 0.3 + 0.7 * np.exp(-0.5 * months / half_life_months)
        else:
            persistence = 0.3 + 0.7 * np.exp(-0.1 * months)
        
        # Apply drug-specific factors
        if "mechanism" in drug_info:
            if "persistence_factor" in drug_info["mechanism"]:
                persistence = persistence * drug_info["mechanism"]["persistence_factor"]
            
            if "tolerance_rate" in drug_info["mechanism"]:
                # Apply tolerance effect
                tolerance_factor = 1.0 - (drug_info["mechanism"]["tolerance_rate"] * months)
                tolerance_factor = max(0.4, tolerance_factor)
                persistence = persistence * tolerance_factor
        
        for pathway, score in initial_scores.items():
            # Get pathway-specific progression rate
            monthly_rate = progression.get(pathway, 0.004)
            
            # Calculate adjusted score - only use the progression rates and persistence
            # without introducing random or artificial adjustments
            adjusted = score * persistence - (monthly_rate * months / 10)
            adjusted_scores[pathway] = max(0, min(adjusted, 1.0))
        
        return adjusted_scores

    def _calculate_node_changes_at_timepoint(self, initial_node_changes, 
                                        adjusted_pathway_changes, months):

        # Group nodes by pathway for reference
        pathway_nodes = {}
        for pathway, nodes in PATHWAYS.items():
            pathway_nodes.update({node: pathway for node in nodes})
        
        # Calculate node changes based on pathway changes
        node_changes = {}
        for node, initial_change in initial_node_changes.items():
            # Find which pathway this node belongs to
            pathway = pathway_nodes.get(node)
            
            if pathway and pathway in adjusted_pathway_changes:
                # Scale the node change based on pathway change in a deterministic way
                # Avoid ratio calculations that could lead to extreme values
                pathway_change = adjusted_pathway_changes[pathway]
                
                if abs(initial_change) < 0.001:  # Effectively zero
                    # If initial change was effectively zero, use a small fraction of pathway change
                    node_changes[node] = pathway_change * 0.5
                else:
                    # Preserve direction of effect when the initial direction is meaningful
                    same_direction = (initial_change < 0) == (pathway_change < 0)
                    if same_direction:
                        # Same direction - scale proportionally but avoid division by very small numbers
                        scaling_factor = min(2.0, abs(initial_change) / max(0.05, abs(pathway_change)))
                        node_changes[node] = pathway_change * scaling_factor
                    else:
                        # Direction has changed due to disease progression or drug effects
                        # Apply a dampened version of the pathway change
                        node_changes[node] = pathway_change * 0.8
            else:
                # If no pathway mapping, apply time-based decay to the initial effect
                # This models the gradual return to disease state
                exponential_decay = np.exp(-0.05 * months)  # More gradual and realistic decay
                linear_component = max(0, 1 - (months / 36))  # Linear decline over 3 years
                
                # Combine exponential and linear components for realistic decay
                combined_decay = 0.7 * exponential_decay + 0.3 * linear_component
                node_changes[node] = initial_change * combined_decay
        
        return node_changes

    def _calculate_composite_score(self, timepoint_efficacy):
        
        # Simplified composite score calculation
        factors = [timepoint_efficacy['efficacy_score']]
        weights = [0.6]  # Higher weight for overall efficacy
        
        # Add pathway scores
        if timepoint_efficacy['pathway_scores']:
            pathway_values = list(timepoint_efficacy['pathway_scores'].values())
            factors.append(np.mean(pathway_values))
            weights.append(0.4)
        
        # Calculate weighted average
        weights = np.array(weights) / np.sum(weights)
        composite = np.sum(np.array(factors) * weights)
        
        return max(0, min(composite, 1.0))
        
    def _predict_mmse_scores(self, temporal_results):
   
        drug_info = temporal_results["drug_info"]
        condition = temporal_results["condition"]
        drug_name = drug_info["name"]
        
        # Realistic starting MMSE based on condition and disease stage
        # Using clinical trial baselines from actual studies
        if condition == "APOE4":
            # APOE4 carriers typically start at lower baseline
            starting_mmse = 24.8  # Slightly lower for APOE4 carriers
            disease_severity = "moderate"  # Assumed disease stage
        else:
            starting_mmse = 26.5  # Higher for non-carriers
            disease_severity = "mild"  # Assumed disease stage
            
        # Adjust for LPL condition
        if condition == "LPL":
            starting_mmse = 25.6  # Between APOE4 and Normal
            
        # Get timepoints and efficacy scores
        timepoints = [0] + temporal_results["timepoints"]
        efficacy_scores = [temporal_results["results"]["initial"]["efficacy_score"]]
        
        for t in temporal_results["timepoints"]:
            efficacy_scores.append(temporal_results["results"][f"month_{t}"]["efficacy_score"])
        
        # Calculate realistic MMSE decline without treatment
        # More sophisticated model based on clinical cohort studies
        # MMSE decline is non-linear and depends on starting score and APOE status

        # Define baseline decline parameters (points per month)
        if disease_severity == "mild" and starting_mmse > 24:
            # Mild AD decline rates from longitudinal studies
            if condition == "APOE4":
                base_decline_rate = 0.33  # Points per month for APOE4 carriers
                acceleration_factor = 0.008  # Non-linear acceleration of decline
            else:
                base_decline_rate = 0.24  # Points per month for non-carriers
                acceleration_factor = 0.006
        elif disease_severity == "moderate" and starting_mmse <= 24:
            # Moderate AD decline rates
            if condition == "APOE4":
                base_decline_rate = 0.42  # Faster decline in moderate stage
                acceleration_factor = 0.012
            else:
                base_decline_rate = 0.32
                acceleration_factor = 0.009
        else:
            # Default values
            base_decline_rate = 0.28
            acceleration_factor = 0.007
        
        # LPL-specific rates (intermediate between Normal and APOE4)
        if condition == "LPL":
            base_decline_rate = 0.29  # Between normal and APOE4
            acceleration_factor = 0.007
            
        # Get drug-specific parameters
        if drug_name in self.drug_mechanisms:
            drug_mech = self.drug_mechanisms[drug_name]
            cognitive_effect = drug_mech.get("effect_on_cognitive", 0.1)
            onset_delay = drug_mech.get("onset_delay", 14) / 30  # Convert days to months
            efficacy_peak = drug_mech.get("efficacy_peak", 60) / 30  # Convert days to months
            tolerance_rate = drug_mech.get("tolerance_rate", 0.02)
        else:
            # Default values for unknown drugs
            cognitive_effect = 0.15
            onset_delay = 0.5  # 0.5 months
            efficacy_peak = 2.0  # 2 months
            tolerance_rate = 0.03
            
        # Disease-modifying vs symptomatic effect
        is_disease_modifying = False
        if drug_name == "Ritzaganine":
            is_disease_modifying = True
            dm_factor = 0.75  # How much of effect is disease-modifying vs symptomatic
        elif drug_name in ["Memantine", "Donepezil", "Galantamine"]:
            is_disease_modifying = False
            dm_factor = 0.0  # Primarily symptomatic
        else:
            # For unknown drugs, assume mixed effect
            is_disease_modifying = True
            dm_factor = 0.4  # Moderate disease-modifying component

        # Calculate MMSE with treatment effect
        mmse_scores = []
        
        for i, months in enumerate(timepoints):
            # Natural non-linear decline - more accurate model
            natural_decline = base_decline_rate * months + acceleration_factor * (months ** 2)
            
            # Apply efficacy with realistic onset and peak timing
            if months < onset_delay:
                # Limited effect during onset phase
                onset_factor = months / onset_delay
                current_efficacy = efficacy_scores[i] * onset_factor * 0.5
            elif months < efficacy_peak:
                # Building to peak effect
                ramp_factor = onset_delay / efficacy_peak + ((months - onset_delay) / (efficacy_peak - onset_delay)) * 0.8
                current_efficacy = efficacy_scores[i] * min(1.0, ramp_factor)
            else:
                # Full effect followed by potential tolerance
                tolerance = min(0.4, tolerance_rate * np.log(months - efficacy_peak + 2))
                current_efficacy = efficacy_scores[i] * (1.0 - tolerance)
                
            # Different effect model for disease-modifying vs symptomatic drugs
            if is_disease_modifying:
                # Disease-modifying drugs affect the rate of decline
                # They have greater long-term but slower initial effect
                
                # Split effect into disease-modifying and symptomatic components
                dm_effect = current_efficacy * dm_factor
                symp_effect = current_efficacy * (1.0 - dm_factor)
                
                # Disease-modifying component reduces actual decline
                reduced_decline = natural_decline * (1.0 - dm_effect)
                
                # Symptomatic component directly improves score
                symptomatic_improvement = min(3.0, symp_effect * cognitive_effect * 4.0)
                
                # Combine effects
                actual_decline = reduced_decline - symptomatic_improvement
            else:
                # Purely symptomatic drugs directly improve score but don't affect disease progression
                symptomatic_improvement = min(4.0, current_efficacy * cognitive_effect * 5.0)
                actual_decline = natural_decline - symptomatic_improvement
                
            # Drug-specific adjustments for realistic effects
            if drug_name == "Ritzaganine":
                # Ritzaganine has moderate cognitive benefit with BACE1 inhibition
                # Effect increases over longer time periods
                if months > 6:
                    longer_term_bonus = min(0.7, 0.05 * (months - 6))
                    actual_decline -= longer_term_bonus
                    
            elif drug_name == "Memantine":
                # Memantine works better in moderate-severe AD
                if starting_mmse < 20:
                    # Greater benefit in more advanced disease
                    severity_bonus = min(0.8, 0.1 * (20 - starting_mmse))
                    actual_decline -= severity_bonus
                    
            elif drug_name == "Donepezil":
                # Donepezil has early strong effect that partially diminishes
                if months < 6:
                    initial_bonus = 0.3 * (1.0 - (months / 6))
                    actual_decline -= initial_bonus
                    
            elif drug_name == "Galantamine":
                # Galantamine has dual mechanism with different temporal profile
                if months > 3 and months < 12:
                    # Nicotinic modulation provides additional midterm benefit
                    nicotinic_bonus = 0.2 * np.sin(np.pi * (months - 3) / 9)
                    actual_decline -= nicotinic_bonus
                
            # Calculate MMSE
            mmse = starting_mmse - actual_decline
                
            # Ensure MMSE stays within valid range
            mmse = max(0, min(mmse, 30))
            mmse_scores.append(mmse)
        
        return mmse_scores


    def _predict_baseline_mmse_decline(self, condition, max_months):
    
        # Starting MMSE score
        if condition == "APOE4":
            starting_mmse = 25
        else:
            starting_mmse = 27
        
        # Monthly decline rate
        monthly_decline = 0.25 if condition == "APOE4" else 0.17  # points per month
        
        # Calculate scores
        mmse_scores = []
        for month in range(int(max_months) + 1):
            mmse = starting_mmse - (monthly_decline * month)
            mmse_scores.append(max(0, min(mmse, 30)))
        
        return mmse_scores

    # Updated visualization for MMSE scores with green and blue colors
    def _generate_mmse_plot(self, temporal_results, output_dir):
        """Generate a plot showing predicted MMSE scores over time"""
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        timepoints = [0] + temporal_results["timepoints"]
        
        # Predict MMSE scores with treatment
        mmse_scores = self._predict_mmse_scores(temporal_results)
        
        # Predict baseline MMSE decline (no treatment)
        max_month = max(temporal_results["timepoints"])
        baseline_mmse = self._predict_baseline_mmse_decline(condition, max_month)
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Plot with the new color scheme
        plt.plot(timepoints, mmse_scores, 'o-', linewidth=2, markersize=8, 
                label=f'With {drug_name} Treatment', color='#2E7D32')  # Dark green
        plt.plot(range(len(baseline_mmse)), baseline_mmse, 's--', linewidth=2, markersize=6,
                label='Without Treatment', color='#1976D2')  # Medium blue
        
        # Add title and labels
        plt.title(f"Predicted MMSE Scores: {drug_name} vs. No Treatment ({condition})", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.ylabel("MMSE Score", fontsize=12)
        plt.ylim(0, 30)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mmse_prediction.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_temporal_comparison_visuals(self, temporal_results, output_dir):
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        
        # Define timepoints
        timepoints = ["initial"] + [f"month_{t}" for t in temporal_results["timepoints"]]
        x_values = [0] + temporal_results["timepoints"]
        
        # Safely extract efficacy scores with null handling
        efficacy_scores = []
        composite_scores = []
        
        for t in timepoints:
            if t in temporal_results["results"]:
                result = temporal_results["results"][t]
                efficacy_scores.append(result.get("efficacy_score", float('nan')))
                composite_scores.append(result.get("composite_score", float('nan')))
            else:
                # Handle missing timepoints
                efficacy_scores.append(float('nan'))
                composite_scores.append(float('nan'))
        
        # 1. Efficacy over time plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, efficacy_scores, 'o-', label='Efficacy Score', linewidth=2, markersize=8)
        plt.plot(x_values, composite_scores, 's-', label='Composite Score', linewidth=2, markersize=8)
        
        # Add title and labels
        plt.title(f"{drug_name} Efficacy Over Time ({condition} condition)", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.ylabel("Efficacy Score", fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficacy_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Pathway changes over time with proper null handling
        # Select key pathways based on drug mechanism and common AD pathways
        key_pathways = ["Amyloid", "Tau", "Synaptic", "Neuroinflammation"]
        
        # Add drug-specific pathways to key_pathways
        if drug_name in ["Donepezil", "Galantamine"]:
            if "Cholinergic" not in key_pathways:
                key_pathways.append("Cholinergic")
        if drug_name == "Memantine":
            if "NMDA" not in key_pathways:
                key_pathways.append("NMDA")
        if drug_name == "Ritzaganine":
            if "Amyloid" not in key_pathways:
                key_pathways.append("Amyloid")
            
        plt.figure(figsize=(12, 7))
        
        # Plot each pathway's change over time with null handling
        for pathway in key_pathways:
            pathway_values = []
            
            for t in timepoints:
                if t in temporal_results["results"]:
                    # Get value or default to NaN - safely handle missing pathways
                    if "pathway_changes" in temporal_results["results"][t]:
                        value = temporal_results["results"][t]["pathway_changes"].get(pathway, float('nan'))
                    else:
                        value = float('nan')
                else:
                    value = float('nan')
                    
                pathway_values.append(value)
            
            # Plot with nan handling - skips missing values
            plt.plot(x_values, pathway_values, 'o-', linewidth=2, markersize=8, label=pathway)
        
        # Add a zero line
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add title and labels
        plt.title(f"{drug_name} Pathway Changes Over Time ({condition} condition)", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.ylabel("Pathway Change (Negative = Beneficial)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pathway_changes_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Regional PET changes over time - handle null values properly
        # Create a separate plot for each modality (amyloid, tau, atrophy, hypometabolism)
        for modality in ["amyloid_suvr", "tau_suvr", "atrophy", "hypometabolism"]:
            plt.figure(figsize=(12, 8))
            
            # Select key regions to display
            key_regions = ["hippocampus", "entorhinal_cortex", "temporal_lobe", "prefrontal_cortex"]
            
            # Plot each region's values over time with proper null handling
            for region in key_regions:
                region_values = []
                has_data = False  # Flag to check if we have any data for this region
                
                for t in timepoints:
                    # Careful extraction with null handling at each level
                    if t in temporal_results["results"]:
                        result = temporal_results["results"][t]
                        if "pet_data" in result and result["pet_data"] is not None:
                            pet_data = result["pet_data"]
                            if region in pet_data and modality in pet_data[region]:
                                value = pet_data[region][modality]
                                region_values.append(value)
                                has_data = True
                            else:
                                region_values.append(float('nan'))
                        else:
                            region_values.append(float('nan'))
                    else:
                        region_values.append(float('nan'))
                
                # Only plot if we have some data
                if has_data:
                    # Plot with NaN handling (will create line breaks where data is missing)
                    plt.plot(x_values, region_values, 'o-', linewidth=2, markersize=8, label=region)
            
            # Format title and labels based on modality
            if modality == "amyloid_suvr":
                title = f"{drug_name} Amyloid PET SUVr Over Time ({condition} condition)"
                ylabel = "Amyloid PET SUVr"
            elif modality == "tau_suvr":
                title = f"{drug_name} Tau PET SUVr Over Time ({condition} condition)"
                ylabel = "Tau PET SUVr"
            elif modality == "hypometabolism":
                title = f"{drug_name} Brain Metabolism Over Time ({condition} condition)"
                ylabel = "Hypometabolism Index"
            else:  # atrophy
                title = f"{drug_name} Brain Atrophy Over Time ({condition} condition)"
                ylabel = "Atrophy Index"
            
            plt.title(title, fontsize=14)
            plt.xlabel("Months Since Treatment Initiation", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{modality}_over_time.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Create a comprehensive summary figure with proper titles and null handling
        plt.figure(figsize=(15, 10))
        
        # Efficacy scores
        plt.subplot(2, 2, 1)
        plt.plot(x_values, efficacy_scores, 'o-', label='Efficacy Score', linewidth=2, markersize=8)
        plt.plot(x_values, composite_scores, 's-', label='Composite Score', linewidth=2, markersize=8)
        plt.title(f"{drug_name} Overall Efficacy ({condition})", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Score", fontsize=10)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        # Amyloid changes by region
        plt.subplot(2, 2, 2)
        key_regions = ["hippocampus", "entorhinal_cortex", "temporal_lobe", "prefrontal_cortex"]
        for region in key_regions:
            values = []
            has_data = False
            for t in timepoints:
                if t in temporal_results["results"]:
                    result = temporal_results["results"][t]
                    if "pet_data" in result and result["pet_data"] is not None:
                        pet_data = result["pet_data"]
                        if region in pet_data and "amyloid_suvr" in pet_data[region]:
                            values.append(pet_data[region]["amyloid_suvr"])
                            has_data = True
                        else:
                            values.append(float('nan'))
                    else:
                        values.append(float('nan'))
                else:
                    values.append(float('nan'))
                    
            if has_data:
                plt.plot(x_values, values, 'o-', linewidth=2, markersize=6, label=region)
                
        plt.title(f"{drug_name} Amyloid PET SUVr by Region ({condition})", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Amyloid SUVr", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # Tau changes by region
        plt.subplot(2, 2, 3)
        for region in key_regions:
            values = []
            has_data = False
            for t in timepoints:
                if t in temporal_results["results"]:
                    result = temporal_results["results"][t]
                    if "pet_data" in result and result["pet_data"] is not None:
                        pet_data = result["pet_data"]
                        if region in pet_data and "tau_suvr" in pet_data[region]:
                            values.append(pet_data[region]["tau_suvr"])
                            has_data = True
                        else:
                            values.append(float('nan'))
                    else:
                        values.append(float('nan'))
                else:
                    values.append(float('nan'))
                    
            if has_data:
                plt.plot(x_values, values, 'o-', linewidth=2, markersize=6, label=region)
                
        plt.title(f"{drug_name} Tau PET SUVr by Region ({condition})", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Tau SUVr", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # Pathway changes
        plt.subplot(2, 2, 4)
        for pathway in key_pathways:
            values = []
            for t in timepoints:
                if t in temporal_results["results"]:
                    result = temporal_results["results"][t]
                    if "pathway_changes" in result:
                        values.append(result["pathway_changes"].get(pathway, float('nan')))
                    else:
                        values.append(float('nan'))
                else:
                    values.append(float('nan'))
                    
            plt.plot(x_values, values, 'o-', linewidth=2, markersize=6, label=pathway)
            
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title(f"{drug_name} Pathway Changes ({condition})", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Change (Negative = Beneficial)", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save the comprehensive figure
        plt.savefig(f"{output_dir}/comprehensive_temporal_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. MMSE prediction plot with improved handling and updated colors
        plt.figure(figsize=(10, 6))
        
        # Predict MMSE scores with treatment
        mmse_scores = self._predict_mmse_scores(temporal_results)
        
        # Predict baseline MMSE decline (no treatment)
        max_month = max(temporal_results["timepoints"])
        baseline_mmse = self._predict_baseline_mmse_decline(condition, max_month)
        
        # Plot both lines with the requested green and blue colors
        plt.plot(x_values, mmse_scores, 'o-', linewidth=2, markersize=8, 
                label=f'With {drug_name} Treatment', color='#2E7D32')  # Dark green
        plt.plot(range(len(baseline_mmse)), baseline_mmse, 's--', linewidth=2, markersize=6,
                label='Without Treatment', color='#1976D2')  # Medium blue
        
        # Add title and labels
        plt.title(f"Predicted MMSE Scores: {drug_name} vs. No Treatment ({condition})", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.ylabel("MMSE Score", fontsize=12)
        plt.ylim(0, 30)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mmse_prediction.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Regional atrophy comparison at final timepoint with improved handling
        plt.figure(figsize=(12, 8))
        
        # Extract final timepoint atrophy data with proper null handling
        final_timepoint = f"month_{max(temporal_results['timepoints'])}"
        
        if final_timepoint in temporal_results["results"]:
            final_result = temporal_results["results"][final_timepoint]
            if "pet_data" in final_result and final_result["pet_data"] is not None:
                final_pet_data = final_result["pet_data"]
                
                # Collect atrophy values by region
                regions = []
                atrophy_values = []
                
                for region in BRAIN_REGIONS:
                    if region in final_pet_data and "atrophy" in final_pet_data[region]:
                        regions.append(region)
                        atrophy_values.append(final_pet_data[region]["atrophy"])
                
                # Only create plot if we have data
                if regions and atrophy_values:
                    # Create bar chart
                    plt.barh(regions, atrophy_values, color='#9C27B0')  # Purple color
                    plt.title(f"{drug_name}: Regional Atrophy at {max(temporal_results['timepoints'])} Months ({condition})", fontsize=14)
                    plt.xlabel("Atrophy Index", fontsize=12)
                    plt.grid(True, alpha=0.3, axis='x')
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/regional_atrophy.png", dpi=300, bbox_inches='tight')
            else:
                print(f"Warning: No PET data available for {drug_name} at final timepoint")
        else:
            print(f"Warning: Missing final timepoint data for {drug_name}")
            
        plt.close()
        
        # 7. Drug-specific biomarker visualization based on mechanism
        if drug_name == "Ritzaganine":
            # For BACE1 inhibitor, show amyloid reduction and tau effects
            self._create_bace1_inhibitor_plot(temporal_results, output_dir)
        elif drug_name in ["Donepezil", "Galantamine"]:
            # For cholinergic drugs, show acetylcholine levels and cognitive metrics
            self._create_cholinergic_metrics_plot(temporal_results, output_dir)
        elif drug_name == "Memantine":
            # For memantine, show NMDA receptor effects
            self._create_excitotoxicity_plot(temporal_results, output_dir)
        
        # 8. Create time point comparison matrix for each biomarker
        # This produces a grid of subplots showing each time point for amyloid, tau, and atrophy
        self._create_timepoint_comparison_matrix(temporal_results, output_dir)

            
    def _create_amyloid_clearance_plot(self, temporal_results, output_dir):
        """Create a special plot for amyloid clearance and ARIA risk for anti-amyloid drugs"""
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        timepoints = [0] + temporal_results["timepoints"]
        
        # Extract amyloid clearance data
        amyloid_changes = []
        aria_risk = []
        
        for i, months in enumerate(timepoints):
            t = "initial" if i == 0 else f"month_{months}"
            
            if t in temporal_results["results"]:
                result = temporal_results["results"][t]
                
                # Extract amyloid pathway change
                if "pathway_changes" in result and "Amyloid" in result["pathway_changes"]:
                    # Convert to positive for "clearance" (negative is good for amyloid)
                    amyloid_changes.append(-1 * result["pathway_changes"]["Amyloid"])
                else:
                    amyloid_changes.append(float('nan'))
                    
                # Extract ARIA risk - grows then plateaus
                if months == 0:
                    aria_risk.append(0)  # No ARIA at baseline
                elif months <= 3:
                    # ARIA risk increases in first 3 months
                    aria_risk.append(0.08 + (0.07 * months / 3))
                elif months <= 6:
                    # Risk plateaus
                    aria_risk.append(0.15)
                else:
                    # Risk gradually decreases after initial treatment phase
                    aria_risk.append(0.15 * np.exp(-(months - 6) / 18))
            else:
                amyloid_changes.append(float('nan'))
                aria_risk.append(float('nan'))
        
        # Create the plot
        plt.figure(figsize=(10, 6), constrained_layout=True)
        
        # Two y-axes for different metrics
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot amyloid clearance
        line1 = ax1.plot(timepoints, amyloid_changes, 'o-', color='blue', 
                        linewidth=2, markersize=8, label='Amyloid Clearance')
        ax1.set_ylabel('Amyloid Clearance (higher is better)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot ARIA risk
        line2 = ax2.plot(timepoints, aria_risk, 's--', color='red',
                        linewidth=2, markersize=8, label='ARIA-E Risk')
        ax2.set_ylabel('ARIA-E Risk', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add title and labels
        plt.title(f"{drug_name}: Amyloid Clearance vs. ARIA-E Risk ({condition})", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # Save figure
        plt.savefig(f"{output_dir}/amyloid_clearance_aria_risk.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_cholinergic_metrics_plot(self, temporal_results, output_dir):
        """Create a special plot for cholinergic metrics for AChE inhibitors"""
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        timepoints = [0] + temporal_results["timepoints"]
        
        # Extract relevant data
        ach_levels = []
        cognitive_effect = []
        side_effects = []
        
        for i, months in enumerate(timepoints):
            t = "initial" if i == 0 else f"month_{months}"
            
            # ACh levels increase rapidly then show tolerance
            if i == 0:
                ach_levels.append(0)
            else:
                if drug_name == "Donepezil":
                    ach_factor = 0.72  # Higher ACh increase
                else:  # Galantamine
                    ach_factor = 0.65  # Slightly lower increase
                    
                # Non-linear response with tolerance
                if months <= 1:
                    ach_level = ach_factor * 0.9  # 90% of effect by 1 month
                else:
                    # Tolerance development
                    tolerance = min(0.3, 0.05 * np.log(months + 1))
                    ach_level = ach_factor * (1.0 - tolerance)
                
                ach_levels.append(ach_level)
            
            # Cognitive effect (from efficacy scores)
            if t in temporal_results["results"]:
                result = temporal_results["results"][t]
                if "efficacy_score" in result:
                    cognitive_effect.append(result["efficacy_score"])
                else:
                    cognitive_effect.append(float('nan'))
                    
                # Side effects (higher initially, then adaptation)
                if i == 0:
                    side_effects.append(0)
                else:
                    if drug_name == "Donepezil":
                        base_se = 0.17  # Base side effect rate
                    else:  # Galantamine
                        base_se = 0.24  # Higher side effects
                        
                    # Side effects diminish somewhat over time
                    if months <= 1:
                        se_level = base_se  # Full side effects initially
                    else:
                        # Adaptation to side effects
                        adaptation = min(0.4, 0.1 * np.log(months + 1))
                        se_level = base_se * (1.0 - adaptation)
                    
                    side_effects.append(se_level)
            else:
                cognitive_effect.append(float('nan'))
                side_effects.append(float('nan'))
        
        # Create the plot
        plt.figure(figsize=(12, 6), constrained_layout=True)
        
        # Three y-axes for different metrics
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Offset the right spine of ax3
        ax3.spines["right"].set_position(("axes", 1.15))
        
        # Plot ACh levels
        line1 = ax1.plot(timepoints, ach_levels, 'o-', color='blue', 
                        linewidth=2, markersize=8, label='ACh Levels')
        ax1.set_ylabel('Acetylcholine Increase', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot cognitive effect
        line2 = ax2.plot(timepoints, cognitive_effect, 's-', color='green',
                        linewidth=2, markersize=8, label='Cognitive Effect')
        ax2.set_ylabel('Cognitive Efficacy', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Plot side effects
        line3 = ax3.plot(timepoints, side_effects, '^--', color='red',
                        linewidth=2, markersize=8, label='Side Effects')
        ax3.set_ylabel('Side Effect Severity', color='red', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='red')
        
        # Add title and labels
        plt.title(f"{drug_name}: Cholinergic Effects Over Time ({condition})", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        # Save figure
        plt.savefig(f"{output_dir}/cholinergic_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_excitotoxicity_plot(self, temporal_results, output_dir):
        """Create a special plot for NMDA receptor effects for memantine"""
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        timepoints = [0] + temporal_results["timepoints"]
        
        # Extract relevant data
        nmda_blockade = []
        calcium_levels = []
        cognitive_effect = []
        
        for i, months in enumerate(timepoints):
            t = "initial" if i == 0 else f"month_{months}"
            
            # NMDA receptor blockade - rapid onset
            if i == 0:
                nmda_blockade.append(0)
                calcium_levels.append(0)
            else:
                # NMDA blockade reaches maximum quickly then has mild tolerance
                if months <= 0.5:
                    blockade = 0.65 * (months / 0.5)  # 65% max blockade
                else:
                    # Slight tolerance over time
                    tolerance = min(0.15, 0.03 * months)
                    blockade = 0.65 * (1.0 - tolerance)
                
                nmda_blockade.append(blockade)
                
                # Calcium levels decrease as blockade increases
                # But with some delay in effect
                if months <= 1:
                    calcium_reduction = 0.45 * (months / 1.0)  # 45% reduction at steady state
                else:
                    # Sustained effect
                    calcium_reduction = 0.45 * (1.0 - 0.1 * np.exp(-(months - 1) / 2))
                
                calcium_levels.append(-calcium_reduction)  # Negative for reduction
            
            # Cognitive effect (from efficacy scores)
            if t in temporal_results["results"]:
                result = temporal_results["results"][t]
                if "efficacy_score" in result:
                    cognitive_effect.append(result["efficacy_score"])
                else:
                    cognitive_effect.append(float('nan'))
            else:
                cognitive_effect.append(float('nan'))
        
        # Create the plot
        plt.figure(figsize=(10, 6), constrained_layout=True)
        
        # Multiple y-axes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot NMDA blockade
        line1 = ax1.plot(timepoints, nmda_blockade, 'o-', color='purple', 
                        linewidth=2, markersize=8, label='NMDA Receptor Blockade')
        ax1.set_ylabel('NMDA Receptor Blockade', color='purple', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='purple')
        
        # Plot calcium levels
        line2 = ax2.plot(timepoints, calcium_levels, 's-', color='blue',
                        linewidth=2, markersize=8, label='Calcium Level Change')
        ax2.set_ylabel('Calcium Level Reduction', color='blue', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Add cognitive effect on the same axis as calcium
        line3 = ax2.plot(timepoints, cognitive_effect, '^--', color='green',
                        linewidth=2, markersize=8, label='Cognitive Effect')
        
        # Add title and labels
        plt.title(f"{drug_name}: NMDA Receptor Effects Over Time ({condition})", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
        # Save figure
        plt.savefig(f"{output_dir}/nmda_receptor_effects.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Create time point comparison matrix for each biomarker
        # This produces a grid of subplots showing each time point for amyloid, tau, and atrophy
        self._create_timepoint_comparison_matrix(temporal_results, output_dir)
    
    def _create_timepoint_comparison_matrix(self, temporal_results, output_dir):
        """
        Create a comparison matrix by combining existing individual timepoint plots.
        This implementation ensures the combined plot exactly matches the individual plots.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.gridspec import GridSpec
        
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        
        # Define key timepoints to include
        key_timepoints = ["initial"]
        key_months = [0]
        
        for m in [1, 6, 12, 36]:
            if m in temporal_results["timepoints"]:
                key_timepoints.append(f"month_{m}")
                key_months.append(m)
        
        # Define the biomarkers
        biomarkers = ["amyloid_suvr", "tau_suvr", "atrophy"]
        biomarker_titles = {
            "amyloid_suvr": "Amyloid PET SUVr",
            "tau_suvr": "Tau PET SUVr",
            "atrophy": "Brain Atrophy"
        }
        
        # Create a comprehensive grid figure by loading and placing the existing images
        fig = plt.figure(figsize=(20, 15))
        
        # Create a GridSpec with 3 rows (one for each biomarker) and len(key_timepoints) columns
        gs = GridSpec(3, len(key_timepoints), figure=fig, hspace=0.3, wspace=0.1)
        
        # For each biomarker and timepoint, load and place the individual plot images
        for i, biomarker in enumerate(biomarkers):
            # Create a separate row of individual plots for each biomarker
            fig_row, axes_row = plt.subplots(1, len(key_timepoints), figsize=(20, 5), sharey=True)
            
            # Set main title for this row
            fig_row.suptitle(f"{drug_name}: {biomarker_titles[biomarker]} Changes Over Time ({condition})", 
                            fontsize=16, y=1.05)
            
            for j, (timepoint, month) in enumerate(zip(key_timepoints, key_months)):
                # Determine the plot file path
                if timepoint == "initial":
                    timepoint_label = ""  # Initial timepoint
                else:
                    timepoint_label = f"_Month_{month}"
                    
                # Look for the post-treatment plot from that timepoint's directory
                plot_file = f"{output_dir}/{timepoint}/{biomarker}{timepoint_label}.png"
                
                # Check if the file exists
                if os.path.exists(plot_file):
                    # Load the existing image and place it in the subplot
                    try:
                        # For the individual biomarker row figure
                        img = mpimg.imread(plot_file)
                        axes_row[j].imshow(img)
                        axes_row[j].axis('off')  # Turn off axes
                        
                        # Set title only for top row in the combined plot
                        if timepoint == "initial":
                            axes_row[j].set_title(f"Initial (Month 0)", fontsize=12)
                        else:
                            axes_row[j].set_title(f"Month {month}", fontsize=12)
                        
                        # Also place the same image in the comprehensive grid
                        ax = fig.add_subplot(gs[i, j])
                        ax.imshow(img)
                        ax.axis('off')
                        
                        # Add title only on top row of comprehensive grid
                        if i == 0:
                            if timepoint == "initial":
                                ax.set_title(f"Initial (Month 0)", fontsize=12)
                            else:
                                ax.set_title(f"Month {month}", fontsize=12)
                        
                        # Add biomarker label only on first column
                        if j == 0:
                            ax.set_ylabel(biomarker_titles[biomarker], fontsize=12)
                            
                    except Exception as e:
                        print(f"Error loading image {plot_file}: {e}")
                        # Create an empty plot with error message
                        ax = fig.add_subplot(gs[i, j])
                        ax.text(0.5, 0.5, f"Image not found", 
                                horizontalalignment='center', verticalalignment='center')
                        ax.axis('off')
                else:
                    print(f"Warning: Plot file not found: {plot_file}")
                    # Create empty plot for the comprehensive grid
                    ax = fig.add_subplot(gs[i, j])
                    ax.text(0.5, 0.5, f"No data for\n{timepoint}", 
                            horizontalalignment='center', verticalalignment='center')
                    ax.axis('off')
                    
                    # Also for the individual biomarker row
                    axes_row[j].text(0.5, 0.5, f"No data for\n{timepoint}", 
                                    horizontalalignment='center', verticalalignment='center')
                    axes_row[j].axis('off')
            
            # Save the individual biomarker row figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            row_output_file = f"{output_dir}/{biomarker}_timepoint_comparison.png"
            fig_row.savefig(row_output_file, dpi=300, bbox_inches='tight')
            plt.close(fig_row)
        
        # Add main title to the comprehensive figure
        fig.suptitle(f"{drug_name}: Biomarker Changes Across Key Timepoints ({condition})", 
                    fontsize=16, y=0.98)
        
        # Save the comprehensive figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(f"{output_dir}/comprehensive_timepoint_grid.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created comparison matrices for {drug_name} in {condition} condition")
        
    def _create_combined_drug_comparison(self, temporal_results, output_dir):
        """
        Create a comprehensive grid comparing biomarkers at key timepoints
        by combining existing plot images rather than recreating them.
        """
        import matplotlib.image as mpimg
        from matplotlib.gridspec import GridSpec
        
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        
        # Define key timepoints to use
        key_timepoints = ["initial"]
        key_months = [0]
        
        for m in [1, 6, 12, 36]:
            if m in temporal_results["timepoints"]:
                key_timepoints.append(f"month_{m}")
                key_months.append(m)
        
        # Define biomarkers to display
        biomarkers = ["amyloid_suvr", "tau_suvr", "atrophy"]
        
        # Create a figure to hold all the subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Create a grid layout - 3 rows (biomarkers) × 1 column
        gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        
        # For each biomarker, find and add the existing comparison image
        for i, biomarker in enumerate(biomarkers):
            ax = fig.add_subplot(gs[i, 0])
            
            # Build the path to the existing comparison image
            img_path = f"{output_dir}/{biomarker}_timepoint_comparison.png"
            
            # Check if the image file exists
            if os.path.exists(img_path):
                # Read and display the image
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.axis('off')  # Hide axes
            else:
                # If image doesn't exist, show a placeholder message
                ax.text(0.5, 0.5, f"Image for {biomarker} not found", 
                        ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
        
        # Add main title
        plt.suptitle(f"{drug_name}: Biomarker Changes Across Key Timepoints ({condition})", 
                    fontsize=16, y=0.98)
        
        # Save the combined figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(f"{output_dir}/comprehensive_timepoint_grid.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_combined_biomarker_grid(self, temporal_results, sim_dir):
        """
        Create a comprehensive grid using the exact same data processing as individual plots
        """
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        
        # Define key timepoints to include
        key_timepoints = ["initial"]
        key_months = [0]
        
        for m in [1, 6, 12, 36]:
            if m in temporal_results["timepoints"]:
                key_timepoints.append(f"month_{m}")
                key_months.append(m)
        
        # Define the biomarkers
        biomarkers = ["amyloid_suvr", "tau_suvr", "atrophy"]
        biomarker_titles = {
            "amyloid_suvr": "Amyloid PET SUVr",
            "tau_suvr": "Tau PET SUVr",
            "atrophy": "Brain Atrophy"
        }
        
        # Create a shared data lookup to ensure consistency
        # Store exactly once and reuse for all plots
        biomarker_data = {}
        
        # First extract all data to ensure consistency
        for timepoint in key_timepoints:
            if timepoint in temporal_results["results"] and "pet_data" in temporal_results["results"][timepoint]:
                pet_data = temporal_results["results"][timepoint]["pet_data"]
                if timepoint not in biomarker_data:
                    biomarker_data[timepoint] = {}
                    
                # Extract each biomarker's data
                for biomarker in biomarkers:
                    if biomarker not in biomarker_data[timepoint]:
                        biomarker_data[timepoint][biomarker] = {}
                        
                    # Get all regions for this biomarker
                    regions = []
                    values = []
                    for region in pet_data:
                        if region != 'metadata' and biomarker in pet_data[region]:
                            regions.append(region)
                            values.append(pet_data[region][biomarker])
                    
                    # Sort regions if we have data
                    if regions and values:
                        # Sorting direction depends on the biomarker
                        bad_direction = 1  # Higher is worse for all these biomarkers
                        
                        # Sort the regions by value
                        sorted_data = sorted(zip(regions, values), key=lambda x: x[1] * bad_direction)
                        sorted_regions, sorted_values = zip(*sorted_data)
                        
                        # Store the sorted data
                        biomarker_data[timepoint][biomarker]['regions'] = sorted_regions
                        biomarker_data[timepoint][biomarker]['values'] = sorted_values
        
        # Define custom colormaps
        amyloid_cmap = LinearSegmentedColormap.from_list('amyloid_green', 
                                                        ['#FFFFFF', '#E5F5E0', '#C7E9C0', '#A1D99B', '#74C476', '#41AB5D', '#238B45', '#006D2C'])
        tau_cmap = LinearSegmentedColormap.from_list('tau_blue', 
                                                    ['#FFFFFF', '#EDF8FB', '#B2E2E2', '#66C2A4', '#41B6C4', '#2C7FB8', '#253494'])
        atrophy_cmap = LinearSegmentedColormap.from_list('atrophy_purple', 
                                                        ['#FFFFFF', '#F2F0F7', '#DADAEB', '#BCBDDC', '#9E9AC8', '#807DBA', '#6A51A3', '#54278F'])
        
        biomarker_cmaps = {
            "amyloid_suvr": amyloid_cmap,
            "tau_suvr": tau_cmap,
            "atrophy": atrophy_cmap
        }
        
        # Value limits for each biomarker
        vmin_vmax = {
            "amyloid_suvr": (1.0, 2.2),
            "tau_suvr": (1.0, 2.0),
            "atrophy": (0.0, 0.3)
        }
        
        # Create a multi-panel figure
        fig = plt.figure(figsize=(20, 15), constrained_layout=True)
        gs = gridspec.GridSpec(nrows=3, ncols=len(key_timepoints), hspace=0.4, wspace=0.3)
        
        # Now create the visualization grid using the exact same data
        for i, biomarker in enumerate(biomarkers):
            # Get colormap and limits
            cmap = biomarker_cmaps[biomarker]
            vmin, vmax = vmin_vmax[biomarker]
            
            for j, (timepoint, month) in enumerate(zip(key_timepoints, key_months)):
                ax = fig.add_subplot(gs[i, j])
                
                # If we have data for this timepoint and biomarker, plot it
                if (timepoint in biomarker_data and 
                    biomarker in biomarker_data[timepoint] and 
                    'regions' in biomarker_data[timepoint][biomarker]):
                    
                    # Use the exact same data as the individual plots
                    regions = biomarker_data[timepoint][biomarker]['regions']
                    values = biomarker_data[timepoint][biomarker]['values']
                    
                    # Create horizontal bar chart
                    bars = ax.barh(regions, values, 
                                color=[cmap((v - vmin) / (vmax - vmin)) for v in values])
                    
                    # Add value labels
                    for k, v in enumerate(values):
                        ax.text(max(v + 0.02, vmin + 0.02), k, f"{v:.2f}", va='center', fontsize=8)
                    
                    # Set consistent x-axis limits
                    ax.set_xlim(vmin, vmax)
                else:
                    ax.text(0.5, 0.5, "No data available", ha='center', va='center')
                
                # Add title for this specific subplot
                if i == 0:  # Only on top row
                    if timepoint == "initial":
                        ax.set_title(f"Initial (Month 0)", fontsize=12)
                    else:
                        ax.set_title(f"Month {month}", fontsize=12)
                
                # Add biomarker label on first column
                if j == 0:
                    ax.set_ylabel(biomarker_titles[biomarker], fontsize=12)
                    
                # Hide y-tick labels on all but first column to save space
                if j > 0:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    
                # Add gridlines
                ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Main title for the entire figure
        fig.suptitle(f"{drug_name}: Biomarker Changes Across Key Timepoints ({condition})", 
                    fontsize=16, y=0.98)
        
        # Adjust layout and save
        plt.subplots_adjust(top=0.93)
        plt.savefig(f"{sim_dir}/comprehensive_biomarker_grid.png", dpi=300, bbox_inches='tight')
        plt.close()