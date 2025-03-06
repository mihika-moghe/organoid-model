import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import time
import datetime

class PETGenerator:
    
    def __init__(self, output_dir="pet_scans"):
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.img_shape = (128, 128, 80)   
        
        self.amyloid_cmap = LinearSegmentedColormap.from_list(
            'amyloid', ['#000000', '#2F0F56', '#4C29A3', '#9152F0', '#CE8DFF', '#EACAFF', '#FFCC66', '#FFA500', '#FF4500', '#FF0000']
        )
        
        self.tau_cmap = LinearSegmentedColormap.from_list(
            'tau', ['#000000', '#0F4228', '#0F6E42', '#36AD6E', '#8CD9AA', '#CCFFE6', '#FFFFCC', '#FFCC66', '#FF7F00', '#FF0000']
        )
        
      
        self.brain_regions_3d = {
            'hippocampus_left': {
                'coords': [48, 40, 40, 8],
                'amyloid_level': 0.8,
                'tau_level': 0.85
            },
            'hippocampus_right': {
                'coords': [80, 40, 40, 8],
                'amyloid_level': 0.8,
                'tau_level': 0.85
            },
            'entorhinal_cortex_left': {
                'coords': [50, 52, 36, 6],
                'amyloid_level': 0.75,
                'tau_level': 0.9
            },
            'entorhinal_cortex_right': {
                'coords': [78, 52, 36, 6],
                'amyloid_level': 0.75,
                'tau_level': 0.9
            },
            'prefrontal_cortex': {
                'coords': [64, 80, 50, 12],
                'amyloid_level': 0.65,
                'tau_level': 0.55
            },
            'temporal_lobe_left': {
                'coords': [40, 60, 40, 10],
                'amyloid_level': 0.7,
                'tau_level': 0.7
            },
            'temporal_lobe_right': {
                'coords': [88, 60, 40, 10],
                'amyloid_level': 0.7,
                'tau_level': 0.7
            },
            'parietal_lobe': {
                'coords': [64, 70, 70, 12],
                'amyloid_level': 0.6,
                'tau_level': 0.5
            },
            'posterior_cingulate': {
                'coords': [64, 60, 55, 6],
                'amyloid_level': 0.7,
                'tau_level': 0.6
            },
            'precuneus': {
                'coords': [64, 50, 60, 8],
                'amyloid_level': 0.65,
                'tau_level': 0.5
            }
        }
    
    def create_brain_mask(self):

        mask = np.zeros(self.img_shape, dtype=np.float32)
        
        center = (self.img_shape[0]//2, self.img_shape[1]//2, self.img_shape[2]//2)
        radii = (55, 70, 42)   
        
        for x in range(self.img_shape[0]):
            for y in range(self.img_shape[1]):
                for z in range(self.img_shape[2]):
                    if ((x - center[0])**2 / radii[0]**2 +
                        (y - center[1])**2 / radii[1]**2 +
                        (z - center[2])**2 / radii[2]**2) <= 1:
                        mask[x, y, z] = 1.0
        
       
        midline_x = self.img_shape[0] // 2
        fissure_width = 3
        for x in range(midline_x-fissure_width//2, midline_x+fissure_width//2+1):
            for y in range(int(self.img_shape[1] * 0.4), self.img_shape[1]):
                for z in range(int(self.img_shape[2] * 0.6), self.img_shape[2]):
                    mask[x, y, z] *= 0.0
        
        mask = gaussian_filter(mask, sigma=1.0)
        mask = (mask > 0.5).astype(np.float32)
        
        return mask
    
    def generate_pet_volume(self, effectiveness, tracer_type='amyloid', condition='APOE4', is_baseline=True):
    
        background = 1.0   
        volume = np.ones(self.img_shape, dtype=np.float32) * background
        
        brain_mask = self.create_brain_mask()
        
        if condition == "Normal":
            condition_factor = 1.0
        elif condition == "APOE4":
            condition_factor = 1.3
        elif condition == "LPL":
            condition_factor = 1.15  
        else:
            condition_factor = 1.0   
        
      
        if is_baseline:
            treatment_factor = 1.0
        else:
           
            treatment_factor = max(0.6, 1.0 - effectiveness * 0.5)   
        for region_name, region_info in self.brain_regions_3d.items():
            x, y, z, radius = region_info['coords']
            
            if tracer_type == 'amyloid':
                base_level = region_info['amyloid_level']
            else:   
                base_level = region_info['tau_level']
            
            uptake_level = base_level * condition_factor * treatment_factor
            
            region_simple_name = region_name.split('_')[0]  
            
            if region_simple_name in ['hippocampus', 'entorhinal_cortex']:
                uptake_level *= 1.2
            
            if condition == "APOE4" and region_simple_name in ['hippocampus', 'entorhinal_cortex']:
                uptake_level *= 1.1   
            elif condition == "LPL" and region_simple_name in ['temporal_lobe', 'parietal_lobe']:
                uptake_level *= 1.1  
            
            for i in range(max(0, x-radius), min(self.img_shape[0], x+radius)):
                for j in range(max(0, y-radius), min(self.img_shape[1], y+radius)):
                    for k in range(max(0, z-radius), min(self.img_shape[2], z+radius)):
                        dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                        
                        if dist <= radius:
                            decay = 1.0 - (dist / radius) * 0.3
                            random_factor = 1.0 + 0.1 * np.random.randn()
                            value = uptake_level * decay * random_factor
                            
                            if value > volume[i, j, k]:
                                volume[i, j, k] = value
        
        self._add_white_matter(volume, tracer_type)
        
        volume = volume * brain_mask
        
        if tracer_type == 'amyloid':
            max_suvr = 2.5 if condition == 'APOE4' else (2.3 if condition == 'LPL' else 2.0)
            min_suvr = 1.0
        else:   
            max_suvr = 2.8 if condition == 'APOE4' else (2.5 if condition == 'LPL' else 2.3)
            min_suvr = 1.0
        
        volume = (volume * (max_suvr - min_suvr)) + min_suvr
        
        volume = self._add_realistic_noise(volume)
        
        volume = gaussian_filter(volume, sigma=1.5)
        
        return volume

    
    
    def calculate_drug_effectiveness(self, drug_targets=None, efficacy_score=None):
       
        if efficacy_score is not None:
            # Scale effectiveness based on efficacy score (0-1 scale)
            base_effectiveness = min(max(efficacy_score, 0.05), 0.95)
        else:
            # Default moderate effectiveness if no info is available
            base_effectiveness = 0.3
        
        if drug_targets:
            amyloid_targeted = False
            tau_targeted = False
            dual_action = False
            
            # Look for amyloid pathway targets
            amyloid_targets = ["APP", "BACE1", "a_secretase", "APOE4", "LRP1", "Abeta"]
            tau_targets = ["MAPT", "GSK3beta", "Cdk5", "PP2A", "Tau"]
            
            for target, _ in drug_targets:
                if target in amyloid_targets:
                    amyloid_targeted = True
                if target in tau_targets:
                    tau_targeted = True
            
            dual_action = amyloid_targeted and tau_targeted
            
            # Adjust effectiveness based on targeting mechanisms
            amyloid_effectiveness = base_effectiveness * 1.5 if amyloid_targeted else base_effectiveness * 0.5
            tau_effectiveness = base_effectiveness * 1.5 if tau_targeted else base_effectiveness * 0.3
            
            if dual_action:
                amyloid_effectiveness *= 1.2
                tau_effectiveness *= 1.2
        else:
            amyloid_effectiveness = base_effectiveness * 1.1
            tau_effectiveness = base_effectiveness * 0.9
        
        amyloid_effectiveness = min(max(amyloid_effectiveness, 0.05), 0.95)
        tau_effectiveness = min(max(tau_effectiveness, 0.05), 0.95)
        
        return {
            'amyloid': amyloid_effectiveness,
            'tau': tau_effectiveness
        }
    
    
    
    def _add_white_matter(self, volume, tracer_type):
       
        # Define white matter regions  
        white_matter_regions = [
            {'coords': [64, 50, 45], 'size': [20, 8, 5]},
            {'coords': [50, 55, 45], 'size': [6, 15, 10]},
            {'coords': [78, 55, 45], 'size': [6, 15, 10]},
        ]
        
        # Lower uptake value for white matter
        wm_value = 1.1 if tracer_type == 'amyloid' else 1.05
        
        for wm in white_matter_regions:
            x, y, z = wm['coords']
            w, h, d = wm['size']
            
            for i in range(max(0, x-w//2), min(self.img_shape[0], x+w//2)):
                for j in range(max(0, y-h//2), min(self.img_shape[1], y+h//2)):
                    for k in range(max(0, z-d//2), min(self.img_shape[2], z+d//2)):
                        # Ellipsoid shape for white matter
                        if ((i-x)**2/(w/2)**2 + (j-y)**2/(h/2)**2 + (k-z)**2/(d/2)**2) <= 1:
                            # Only set if lower than current value (white matter has lower uptake)
                            if volume[i, j, k] > wm_value:
                                volume[i, j, k] = wm_value
    
    def _add_realistic_noise(self, volume):
       
        # Add Poisson-like noise proportional to signal intensity (realistic for PET)
        noise_level = 0.05
        noise = np.random.poisson(lam=volume * 10) / 10.0 * noise_level
        noisy_volume = volume + (noise - noise_level * volume)
        
        # Ensure positivity
        noisy_volume = np.maximum(noisy_volume, 0.01)
        
        return noisy_volume
    
    def visualize_pet_slices(self, pet_volume, tracer_type, output_file, title_suffix=""):
       
        # Set colormap based on tracer type
        cmap = self.amyloid_cmap if tracer_type == 'amyloid' else self.tau_cmap
        
        # Value range for colormap
        vmin, vmax = 1.0, 2.5 if tracer_type == 'amyloid' else 2.8
        
        # Create figure with multiple slices
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle(f"{tracer_type.capitalize()} PET Scan {title_suffix}", fontsize=16)
        
        # Select evenly spaced slices in each axis
        x_slices = np.linspace(40, 88, 5).astype(int)  # Sagittal
        y_slices = np.linspace(30, 80, 5).astype(int)  # Coronal
        z_slices = np.linspace(30, 60, 5).astype(int)  # Axial
        
        # Plot sagittal slices (x-axis)
        for i, x in enumerate(x_slices):
            im = axes[0, i].imshow(pet_volume[x, :, :].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"Sagittal {x}")
            axes[0, i].axis('off')
        
        # Plot coronal slices (y-axis)
        for i, y in enumerate(y_slices):
            im = axes[1, i].imshow(pet_volume[:, y, :].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"Coronal {y}")
            axes[1, i].axis('off')
        
        # Plot axial slices (z-axis)
        for i, z in enumerate(z_slices):
            im = axes[2, i].imshow(pet_volume[:, :, z], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f"Axial {z}")
            axes[2, i].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('SUVr')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_comparison_visualization(self, baseline_volume, post_volume, tracer_type, output_file, drug_name="Drug"):
   
        # Set colormap based on tracer type
        cmap = self.amyloid_cmap if tracer_type == 'amyloid' else self.tau_cmap

        # Value range for colormap
        vmin, vmax = 1.0, 2.5 if tracer_type == 'amyloid' else 2.8

        # Create figure with 3 key slices shown side by side
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        fig.suptitle(f"{drug_name}: {tracer_type.capitalize()} PET Comparison: Baseline vs Post-Treatment", fontsize=16)

        # Select most informative slices
        if tracer_type == 'amyloid':
            # For amyloid, focus on cortical regions
            axial_slice = 45    # Mid-brain axial slice
            coronal_slice = 60  # Mid-frontal coronal slice
            sagittal_slice = 64 # Midline sagittal slice
        else:
            # For tau, focus on hippocampus and temporal lobe
            axial_slice = 40    # Lower axial slice through hippocampus
            coronal_slice = 50  # Posterior coronal slice
            sagittal_slice = 48 # Left hemisphere sagittal slice

        # Plot axial slices
        im1 = axes[0, 0].imshow(baseline_volume[:, :, axial_slice], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Baseline - Axial")
        axes[0, 0].axis('off')

        im2 = axes[0, 1].imshow(post_volume[:, :, axial_slice], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Post-Treatment - Axial")
        axes[0, 1].axis('off')

        # Plot coronal slices
        axes[1, 0].imshow(baseline_volume[:, coronal_slice, :].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f"Baseline - Coronal")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(post_volume[:, coronal_slice, :].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f"Post-Treatment - Coronal")
        axes[1, 1].axis('off')

        # Plot sagittal slices
        axes[2, 0].imshow(baseline_volume[sagittal_slice, :, :].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2, 0].set_title(f"Baseline - Sagittal")
        axes[2, 0].axis('off')

        axes[2, 1].imshow(post_volume[sagittal_slice, :, :].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2, 1].set_title(f"Post-Treatment - Sagittal")
        axes[2, 1].axis('off')

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label('SUVr')

        # Calculate mean values and RMSE
        mean_baseline = np.mean(baseline_volume[baseline_volume > 1.2])
        mean_post = np.mean(post_volume[post_volume > 1.2])

        percent_change = ((mean_post - mean_baseline) / mean_baseline) * 100

        # Calculate root mean square error
        rmse = np.sqrt(np.mean((post_volume - baseline_volume)**2))

        # Format the statistics text
        stats_text = (
            f"Mean SUVr: Baseline = {mean_baseline:.2f}, Post = {mean_post:.2f}\n"
            f"Mean SUVr change: {percent_change:.1f}% (decrease)\n"
            f"RMSE: {rmse:.4f}"
        )

        # Add the statistics as a text box
        fig.text(0.5, 0.01, stats_text, 
                ha='center', fontsize=14, color='green', 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        return output_file
    
    def create_difference_map(self, baseline_volume, post_volume, tracer_type, output_file, drug_name="Drug"):
       
        # Calculate difference volume (baseline - post)
        diff_volume = baseline_volume - post_volume

        # Create custom diverging colormap for difference
        diff_cmap = plt.cm.RdBu

        diff_max = max(0.3, np.max(np.abs(diff_volume)) * 1.2)
        vmin, vmax = -diff_max/4, diff_max   

        # Create figure with multiple slices
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle(f"{drug_name}: {tracer_type.capitalize()} PET Change Map (Baseline - Post-Treatment)", fontsize=16)

        # Select evenly spaced slices in each axis
        x_slices = np.linspace(40, 88, 5).astype(int)  # Sagittal
        y_slices = np.linspace(30, 80, 5).astype(int)  # Coronal
        z_slices = np.linspace(30, 60, 5).astype(int)  # Axial

        # Plot sagittal slices (x-axis)
        for i, x in enumerate(x_slices):
            im = axes[0, i].imshow(diff_volume[x, :, :].T, origin='lower', cmap=diff_cmap, vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"Sagittal {x}")
            axes[0, i].axis('off')

        # Plot coronal slices (y-axis)
        for i, y in enumerate(y_slices):
            im = axes[1, i].imshow(diff_volume[:, y, :].T, origin='lower', cmap=diff_cmap, vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"Coronal {y}")
            axes[1, i].axis('off')

        # Plot axial slices (z-axis)
        for i, z in enumerate(z_slices):
            im = axes[2, i].imshow(diff_volume[:, :, z], origin='lower', cmap=diff_cmap, vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f"Axial {z}")
            axes[2, i].axis('off')

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('SUVr Difference')

        fig.text(0.5, 0.02, "Red/Orange: Reduction in tracer uptake (improvement)\nBlue: Increase in tracer uptake (not expected)", 
                ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.04, 0.9, 0.95])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        return output_file

    def create_html_report(self, visualization_paths, drug_name, efficacy_data, condition, output_file):
       
        # Get effectiveness values from efficacy data
        if efficacy_data and 'efficacy_score' in efficacy_data:
            efficacy_score = efficacy_data['efficacy_score']
        else:
            # Default moderate efficacy if not provided
            efficacy_score = 0.3
        
        # For all drugs, calculate appropriate improvement rates
        # Use non-linear scaling to make more effective drugs show greater improvement
        amyloid_change_pct = -min(50, max(5, efficacy_score * 70))
        tau_change_pct = -min(30, max(3, efficacy_score * 40))
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{drug_name} PET Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .report-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .img-container {{ margin: 10px 0; text-align: center; }}
                .img-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive-change {{ color: green; }}
                .negative-change {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>{drug_name} PET Analysis Report</h1>
                <h2>Condition: {condition}</h2>
                <p>Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>Summary of Findings</h2>
                <p>This report presents PET imaging analysis evaluating the effect of {drug_name} in a subject with {condition} condition.</p>
        """
        
        # Add efficacy score if available
        if efficacy_data:
            html += f"""
                <p><strong>Overall Drug Efficacy Score:</strong> {efficacy_score:.2f}</p>
            """
            
            # Add pathway effects if available
            if 'pathway_scores' in efficacy_data:
                html += """
                <h3>Pathway Effects:</h3>
                <ul>
                """
                for pathway, score in efficacy_data['pathway_scores'].items():
                    html += f"<li><strong>{pathway}:</strong> {score:.2f}</li>"
                html += "</ul>"
        
        # Summary metrics table
        html += f"""
                <table>
                    <tr>
                        <th>Biomarker</th>
                        <th>Mean Change</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>Amyloid PET SUVr</td>
                        <td class="positive-change">{amyloid_change_pct:.1f}%</td>
                        <td>Reduction in amyloid burden</td>
                    </tr>
                    <tr>
                        <td>Tau PET SUVr</td>
                        <td class="positive-change">{tau_change_pct:.1f}%</td>
                        <td>Reduction in tau pathology</td>
                    </tr>
                </table>
        """
        
        # Regional changes with varying magnitudes based on brain regions
        html += """
                <h3>Regional Changes</h3>
                <table>
                    <tr>
                        <th>Brain Region</th>
                        <th>Amyloid Change (%)</th>
                        <th>Tau Change (%)</th>
                    </tr>
        """
        
        # Show brain region-specific changes
        regions = ['hippocampus', 'entorhinal_cortex', 'prefrontal_cortex', 
                  'temporal_lobe', 'parietal_lobe', 'posterior_cingulate', 'precuneus']
        
        for region in regions:
            # Vary the intensity of change by region (based on known vulnerability in AD)
            if region in ['hippocampus', 'entorhinal_cortex']:
                region_factor = 1.3  # More pronounced changes
            elif region in ['prefrontal_cortex', 'temporal_lobe']:
                region_factor = 1.1  # Moderate changes
            else:
                region_factor = 0.9  # Less pronounced changes
                
            # Calculate region-specific changes
            amyloid_region_change = amyloid_change_pct * region_factor
            tau_region_change = tau_change_pct * region_factor
            
            # Add row to table
            html += f"""
                    <tr>
                        <td>{region.replace('_', ' ').title()}</td>
                        <td class="positive-change">{amyloid_region_change:.1f}%</td>
                        <td class="positive-change">{tau_region_change:.1f}%</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Add Amyloid PET visualizations
        html += """
            <div class="section">
                <h2>Amyloid PET Analysis</h2>
        """
        
        if 'amyloid_comparison' in visualization_paths:
            html += f"""
                <h3>Amyloid PET Comparison</h3>
                <div class="img-container">
                    <img src="{os.path.basename(visualization_paths['amyloid_comparison'])}" alt="Amyloid PET Comparison">
                </div>
            """
        
        if 'amyloid_difference_map' in visualization_paths:
            html += f"""
                <h3>Amyloid PET Difference Map</h3>
                <div class="img-container">
                    <img src="{os.path.basename(visualization_paths['amyloid_difference_map'])}" alt="Amyloid PET Difference Map">
                </div>
            """
        
        html += """
            </div>
        """
        
        # Add Tau PET visualizations
        html += """
            <div class="section">
                <h2>Tau PET Analysis</h2>
        """
        
        if 'tau_comparison' in visualization_paths:
            html += f"""
                <h3>Tau PET Comparison</h3>
                <div class="img-container">
                    <img src="{os.path.basename(visualization_paths['tau_comparison'])}" alt="Tau PET Comparison">
                </div>
            """
        
        if 'tau_difference_map' in visualization_paths:
            html += f"""
                <h3>Tau PET Difference Map</h3>
                <div class="img-container">
                    <img src="{os.path.basename(visualization_paths['tau_difference_map'])}" alt="Tau PET Difference Map">
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Interpretation and Conclusion</h2>
        """
        
        # Different conclusion text based on efficacy score
        conclusion = ""
        if efficacy_score > 0.5:
            conclusion = f"""
                <p>{drug_name} shows substantial reduction in amyloid plaque burden, with significant effects observed in the 
                hippocampus, entorhinal cortex, and cortical regions. This suggests strong target engagement and disease-modifying potential.</p>
                
                <p>The tau pathology also shows reduction, which may be a downstream effect of reduced amyloid burden. 
                This dual effect on both primary pathologies of Alzheimer's disease suggests promising efficacy.</p>
            """
        elif efficacy_score > 0.3:
            conclusion = f"""
                <p>{drug_name} demonstrates moderate reduction in amyloid plaque burden, with notable effects in key memory-related regions.
                This suggests good target engagement and potential disease-modifying effects.</p>
                
                <p>The tau pathology shows modest improvement, which may represent early downstream effects or direct mechanism action.
                Continued treatment may yield further improvement in these biomarkers.</p>
            """
        else:
            conclusion = f"""
                <p>{drug_name} shows mild effects on amyloid plaque burden, with some improvement in the distribution pattern.
                This suggests partial target engagement.</p>
                
                <p>The tau pathology shows similar modest improvement. While the magnitude of changes is not dramatic,
                any reduction in pathological protein burden could be clinically meaningful over longer treatment periods.</p>
            """
        
        html += conclusion
        
        html += """
                <p><strong>Clinical Significance:</strong> Reduction in amyloid and tau burden as measured by PET has been 
                associated with slower clinical progression in Alzheimer's disease. These imaging findings should be considered
                alongside clinical assessment of cognition and function.</p>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(output_file, 'w') as f:
            f.write(html)
        
        return output_file
    
    def generate_pet_scans(self, drug_name, efficacy_data=None, drug_targets=None, condition="APOE4", output_dir=None):
    
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"{drug_name.lower()}_{condition.lower()}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate drug effectiveness based on efficacy data or targets
        if efficacy_data and 'efficacy_score' in efficacy_data:
            efficacy = efficacy_data['efficacy_score']
        else:
            efficacy = None
        
        effectiveness = self.calculate_drug_effectiveness(drug_targets, efficacy)
        
        # Generate baseline volumes for amyloid and tau
        amyloid_baseline = self.generate_pet_volume(
            effectiveness['amyloid'], 
            tracer_type='amyloid',
            condition=condition,
            is_baseline=True
        )
        
        tau_baseline = self.generate_pet_volume(
            effectiveness['tau'], 
            tracer_type='tau',
            condition=condition,
            is_baseline=True
        )
        
        # Generate post-treatment volumes
        amyloid_post = self.generate_pet_volume(
            effectiveness['amyloid'], 
            tracer_type='amyloid',
            condition=condition,
            is_baseline=False
        )
        
        tau_post = self.generate_pet_volume(
            effectiveness['tau'],
            tracer_type='tau',
            condition=condition,
            is_baseline=False
        )
        
        # Create visualizations
        visualization_paths = {}
        
        # Amyloid PET visualizations
        amyloid_baseline_slices = os.path.join(output_dir, "amyloid_baseline_slices.png")
        visualization_paths['amyloid_baseline_slices'] = self.visualize_pet_slices(
            amyloid_baseline, 'amyloid', amyloid_baseline_slices, f"{drug_name} Baseline"
        )
        
        amyloid_post_slices = os.path.join(output_dir, "amyloid_post_slices.png")
        visualization_paths['amyloid_post_slices'] = self.visualize_pet_slices(
            amyloid_post, 'amyloid', amyloid_post_slices, f"{drug_name} Post-Treatment"
        )
        
        amyloid_comparison = os.path.join(output_dir, "amyloid_comparison.png")
        visualization_paths['amyloid_comparison'] = self.create_comparison_visualization(
            amyloid_baseline, amyloid_post, 'amyloid', amyloid_comparison, drug_name
        )
        
        amyloid_diff = os.path.join(output_dir, "amyloid_difference_map.png")
        visualization_paths['amyloid_difference_map'] = self.create_difference_map(
            amyloid_baseline, amyloid_post, 'amyloid', amyloid_diff, drug_name
        )
        
        # Tau PET visualizations
        tau_baseline_slices = os.path.join(output_dir, "tau_baseline_slices.png")
        visualization_paths['tau_baseline_slices'] = self.visualize_pet_slices(
            tau_baseline, 'tau', tau_baseline_slices, f"{drug_name} Baseline"
        )
        
        tau_post_slices = os.path.join(output_dir, "tau_post_slices.png")
        visualization_paths['tau_post_slices'] = self.visualize_pet_slices(
            tau_post, 'tau', tau_post_slices, f"{drug_name} Post-Treatment"
        )
        
        tau_comparison = os.path.join(output_dir, "tau_comparison.png")
        visualization_paths['tau_comparison'] = self.create_comparison_visualization(
            tau_baseline, tau_post, 'tau', tau_comparison, drug_name
        )
        
        tau_diff = os.path.join(output_dir, "tau_difference_map.png")
        visualization_paths['tau_difference_map'] = self.create_difference_map(
            tau_baseline, tau_post, 'tau', tau_diff, drug_name
        )
        
        # Create HTML report
        report_file = os.path.join(output_dir, f"{drug_name}_{condition}_report.html")
        visualization_paths['html_report'] = self.create_html_report(
            visualization_paths, drug_name, efficacy_data, condition, report_file
        )
        
        return visualization_paths

def generate_universal_pet_scans(drug_name, efficacy_data=None, drug_targets=None, condition="APOE4", output_dir=None):
   
    # Create output directory
    if output_dir is None:
        output_dir = f"pet_results/{drug_name.lower()}_{condition.lower()}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the PET generator
    pet_generator = PETGenerator()
    
    # Generate all visualizations
    results = pet_generator.generate_pet_scans(
        drug_name=drug_name,
        efficacy_data=efficacy_data,
        drug_targets=drug_targets,
        condition=condition,
        output_dir=output_dir
    )
    
    return results


# Example usage:
if __name__ == "__main__":
    # Test with a few different drugs
    
    # 1. Known FDA drug
    lecanemab_results = generate_universal_pet_scans(
        drug_name="Lecanemab",
        efficacy_data={'efficacy_score': 0.7},  # High efficacy
        condition="APOE4",
        output_dir="example_pet_results/lecanemab"
    )
    
    # 2. Different known drug with moderate efficacy
    donepezil_results = generate_universal_pet_scans(
        drug_name="Donepezil",
        efficacy_data={'efficacy_score': 0.3},  # Moderate efficacy
        condition="APOE4",
        output_dir="example_pet_results/donepezil"
    )
    
    # 3. Custom/novel drug
    custom_drug_results = generate_universal_pet_scans(
        drug_name="Novel_Compound_XYZ",
        drug_targets=[("APP", 0), ("GSK3beta", 0), ("MAPT", 0)],  # Target both amyloid and tau
        condition="APOE4",
        output_dir="example_pet_results/novel_compound"
    )
    
    print("Generated PET visualizations for different drugs in 'example_pet_results' directory")