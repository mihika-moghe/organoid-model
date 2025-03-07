import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    DRUG_TARGETS,
    CLINICAL_EFFICACY,
    PHARMACOKINETICS,
    AD_OUTPUT_NODES,
    PATHWAYS,
    BRAIN_REGIONS
)

# Ensure the right imports for PETScanGenerator
try:
    from efficacy import PETScanGenerator
except ImportError:
    # If the class isn't directly importable, define a simplified version
    class PETScanGenerator:
        """Simplified PET scan generator for temporal simulations"""
        
        def __init__(self, output_dir="pet_scans"):
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

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
                "Amyloid": 0.005,  # 0.5% increase per month
                "Tau": 0.004,      # 0.4% increase per month
                "Apoptosis": 0.003,
                "Autophagy": -0.002,
                "Lipid": 0.002,
                "Synaptic": -0.003,
                "Neuroinflammation": 0.004,
                "Oxidative_Stress": 0.003,
                "Insulin_Signaling": -0.002
            },
            "APOE4": {
                "Amyloid": 0.009,  # 0.9% increase per month (faster in APOE4 carriers)
                "Tau": 0.008,      # 0.8% increase per month
                "Apoptosis": 0.006,
                "Autophagy": -0.004,
                "Lipid": 0.005,
                "Synaptic": -0.006,
                "Neuroinflammation": 0.007,
                "Oxidative_Stress": 0.006,
                "Insulin_Signaling": -0.003
            },
             
            "LPL": {
                "Amyloid": 0.007,  # Between Normal and APOE4
                "Tau": 0.006,      
                "Apoptosis": 0.005,
                "Autophagy": -0.003,
                "Lipid": 0.008,    # Higher lipid dysregulation in LPL
                "Synaptic": -0.005,
                "Neuroinflammation": 0.005,
                "Oxidative_Stress": 0.004,
                "Insulin_Signaling": -0.003
            }
        }
        
        # Regions affected over time (different progression by region)
        # Based on Braak staging and literature
        self.region_progression = {
            "hippocampus": 1.2,           # Fastest progression
            "entorhinal_cortex": 1.1,
            "temporal_lobe": 1.0,
            "posterior_cingulate": 0.9,
            "prefrontal_cortex": 0.8,
            "parietal_lobe": 0.7,
            "precuneus": 0.7
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
        if drug_name:
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
            # Visualize baseline vs initial
            visualize_pet_scan(baseline_pet, initial_pet, output_dir=f"{sim_dir}/initial")
        
        # Store initial results
        temporal_results["results"]["initial"] = {
            "efficacy_score": initial_efficacy['efficacy_score'],
            "composite_score": initial_efficacy['composite_score'],
            "pathway_scores": initial_efficacy['pathway_scores'],
            "node_changes": initial_efficacy['node_changes'],
            "pathway_changes": initial_efficacy['pathway_changes'],
            "pet_data": initial_pet
        }
        
        # Simulate each timepoint
        for months in self.timepoints:
            print(f"Simulating {months} month{'s' if months > 1 else ''} timepoint...")
            
        
            # Get adjusted pathway changes based on time
            time_adjusted_changes = self._calculate_time_adjusted_changes(
                initial_efficacy['pathway_changes'],
                drug_info,
                condition,
                months
            )
            
            # Generate PET scan for this timepoint
            timepoint_pet = self._generate_timepoint_pet_scan(
                baseline_pet,
                initial_pet,
                time_adjusted_changes,
                condition,
                months,
                drug_name=drug_name  # Pass drug name here too
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
            
            # Visualize if requested
            if include_visuals:
                # Compare to baseline
                visualize_pet_scan(
                    baseline_pet, timepoint_pet,
                    output_dir=f"{sim_dir}/month_{months}"
                )
            
            # Store results
            temporal_results["results"][f"month_{months}"] = timepoint_efficacy
        
        # Generate comparative visualizations across all timepoints
        if include_visuals:
            self._generate_temporal_comparison_visuals(temporal_results, sim_dir)
        
        # Save results
        results_file = f"{sim_dir}/temporal_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(temporal_results, f)
        
        return temporal_results
    
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
                              adjusted_pathway_changes, condition, months, drug_name=None):
   
        # Extract the relevant pathway changes for PET imaging (amyloid and tau)
        amyloid_change = adjusted_pathway_changes.get("Amyloid", 0)
        tau_change = adjusted_pathway_changes.get("Tau", 0)
        
        # Calculate node changes at this timepoint
        node_changes_at_timepoint = self._calculate_node_changes_at_timepoint(
            {node: 0 for node in AD_OUTPUT_NODES if node in self.output_list},  # Initialize with zeros
            adjusted_pathway_changes,
            months
        )
        
        # Use generate_brain_pet_scan with the drug_name parameter
        timepoint_pet = generate_brain_pet_scan(
            node_changes_at_timepoint,
            condition=condition,
            stage="post_treatment",
            drug_name=drug_name
        )
        
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
        
        for pathway, score in initial_scores.items():
            # Get pathway-specific progression rate
            monthly_rate = progression.get(pathway, 0.004)
            
            # Calculate adjusted score
            adjusted = score * persistence - (monthly_rate * months / 10)
            adjusted_scores[pathway] = max(0, min(adjusted, 1.0))
        
        return adjusted_scores
    
    def _calculate_node_changes_at_timepoint(self, initial_node_changes, 
                                          adjusted_pathway_changes, months):
       
        # Group nodes by pathway
        pathway_nodes = {}
        for pathway, nodes in PATHWAYS.items():
            pathway_nodes.update({node: pathway for node in nodes})
        
        # Calculate node changes based on pathway changes
        node_changes = {}
        for node, initial_change in initial_node_changes.items():
            # Find which pathway this node belongs to
            pathway = pathway_nodes.get(node)
            
            if pathway and pathway in adjusted_pathway_changes:
                # Scale the node change based on pathway change
                if initial_change != 0:
                    # Ratio of current pathway change to initial pathway change
                    ratio = adjusted_pathway_changes[pathway] / initial_change
                    
                    # Apply ratio to get adjusted node change
                    node_changes[node] = initial_change * ratio
                else:
                    # If initial change was 0, use a fraction of pathway change
                    node_changes[node] = adjusted_pathway_changes[pathway] * 0.5
            else:
                # If node doesn't belong to a specific pathway, apply time decay
                node_changes[node] = initial_change * np.exp(-0.1 * months)
        
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
        
        # Starting MMSE score - typically 26-28 for early AD, lower for APOE4 carriers
        if condition == "APOE4":
            starting_mmse = 25  # Slightly lower for APOE4 carriers
        else:
            starting_mmse = 27  # Higher for non-carriers
        
        # Get timepoints and efficacy scores
        timepoints = [0] + temporal_results["timepoints"]
        efficacy_scores = [temporal_results["results"]["initial"]["efficacy_score"]]
        
        for t in temporal_results["timepoints"]:
            efficacy_scores.append(temporal_results["results"][f"month_{t}"]["efficacy_score"])
        
        # Calculate MMSE decline without treatment
        # MMSE typically declines 2-4 points/year in AD, faster in APOE4 carriers
        monthly_decline = 0.25 if condition == "APOE4" else 0.17  # points per month
        
        # Calculate MMSE with treatment effect
        mmse_scores = []
        for i, months in enumerate(timepoints):
            # Natural decline
            natural_decline = monthly_decline * months
            
            # Treatment effect reduces decline
            treatment_effect = natural_decline * efficacy_scores[i]
            
            # Actual decline is natural minus treatment effect
            actual_decline = natural_decline - treatment_effect
            
            # Calculate MMSE
            mmse = starting_mmse - actual_decline
            
            # MMSE can't go above 30 (perfect score) or below 0
            mmse_scores.append(max(0, min(mmse, 30)))
        
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
    
    def _generate_temporal_comparison_visuals(self, temporal_results, output_dir):
      
        drug_name = temporal_results["drug_info"]["name"]
        condition = temporal_results["condition"]
        
        # 1. Efficacy over time
        plt.figure(figsize=(10, 6))
        
        # Extract timepoints and scores
        timepoints = ["initial"] + [f"month_{t}" for t in temporal_results["timepoints"]]
        efficacy_scores = [temporal_results["results"][t]["efficacy_score"] for t in timepoints]
        composite_scores = [temporal_results["results"][t]["composite_score"] for t in timepoints]
        
        # Convert timepoints to numeric values for x-axis
        x_values = [0] + temporal_results["timepoints"]
        
        # Plot scores
        plt.plot(x_values, efficacy_scores, 'o-', label='Efficacy Score', linewidth=2, markersize=8)
        plt.plot(x_values, composite_scores, 's-', label='Composite Score', linewidth=2, markersize=8)
        
        # Add title and labels
        plt.title(f"{drug_name} Efficacy Over Time ({condition} condition)", fontsize=14)
        plt.xlabel("Months Since Treatment Initiation", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.savefig(f"{output_dir}/efficacy_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Pathway changes over time
        # Select key pathways
        key_pathways = ["Amyloid", "Tau", "Synaptic", "Neuroinflammation"]
        
        plt.figure(figsize=(12, 7))
        
        # Plot each pathway's change over time
        for pathway in key_pathways:
            pathway_values = []
            for t in timepoints:
                # Get value or default to 0
                value = temporal_results["results"][t]["pathway_changes"].get(pathway, 0)
                pathway_values.append(value)
            
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
        plt.savefig(f"{output_dir}/pathway_changes_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Regional PET changes over time
        # Create a separate plot for each modality (amyloid, tau, atrophy)
        for modality in ["amyloid_suvr", "tau_suvr", "atrophy"]:
            plt.figure(figsize=(12, 8))
            
            # Select key regions to display
            key_regions = ["hippocampus", "entorhinal_cortex", "temporal_lobe", "prefrontal_cortex"]
            
            # Plot each region's values over time
            for region in key_regions:
                region_values = []
                for t in timepoints:
                    # Get value from PET data
                    if region in temporal_results["results"][t]["pet_data"]:
                        value = temporal_results["results"][t]["pet_data"][region][modality]
                        region_values.append(value)
                    else:
                        region_values.append(None)  # Handle missing data
                
                plt.plot(x_values, region_values, 'o-', linewidth=2, markersize=8, label=region)
            
            # Format title and labels based on modality
            if modality == "amyloid_suvr":
                title = f"{drug_name} Amyloid PET SUVr Over Time ({condition} condition)"
                ylabel = "Amyloid PET SUVr"
            elif modality == "tau_suvr":
                title = f"{drug_name} Tau PET SUVr Over Time ({condition} condition)"
                ylabel = "Tau PET SUVr"
            else:  # atrophy
                title = f"{drug_name} Brain Atrophy Over Time ({condition} condition)"
                ylabel = "Atrophy Index"
            
            plt.title(title, fontsize=14)
            plt.xlabel("Months Since Treatment Initiation", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save figure
            plt.savefig(f"{output_dir}/{modality}_over_time.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Create a comprehensive summary figure
        plt.figure(figsize=(15, 10))
        
        # Efficacy scores
        plt.subplot(2, 2, 1)
        plt.plot(x_values, efficacy_scores, 'o-', label='Efficacy Score', linewidth=2, markersize=8)
        plt.plot(x_values, composite_scores, 's-', label='Composite Score', linewidth=2, markersize=8)
        plt.title("Overall Efficacy", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Score", fontsize=10)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        # Amyloid changes by region
        plt.subplot(2, 2, 2)
        for region in key_regions:
            values = []
            for t in timepoints:
                if region in temporal_results["results"][t]["pet_data"]:
                    values.append(temporal_results["results"][t]["pet_data"][region]["amyloid_suvr"])
                else:
                    values.append(None)
            plt.plot(x_values, values, 'o-', linewidth=2, markersize=6, label=region)
        plt.title("Amyloid PET SUVr by Region", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Amyloid SUVr", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # Tau changes by region
        plt.subplot(2, 2, 3)
        for region in key_regions:
            values = []
            for t in timepoints:
                if region in temporal_results["results"][t]["pet_data"]:
                    values.append(temporal_results["results"][t]["pet_data"][region]["tau_suvr"])
                else:
                    values.append(None)
            plt.plot(x_values, values, 'o-', linewidth=2, markersize=6, label=region)
        plt.title("Tau PET SUVr by Region", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Tau SUVr", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        
        # Pathway changes
        plt.subplot(2, 2, 4)
        for pathway in key_pathways:
            values = []
            for t in timepoints:
                values.append(temporal_results["results"][t]["pathway_changes"].get(pathway, 0))
            plt.plot(x_values, values, 'o-', linewidth=2, markersize=6, label=pathway)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title("Pathway Changes", fontsize=12)
        plt.xlabel("Months", fontsize=10)
        plt.ylabel("Change (Negative = Beneficial)", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)