#!/usr/bin/env python3
"""
TDA Reality Check - Honest Diagnostic Analysis
==============================================

This script performs a brutally honest assessment of whether TDA can actually
produce meaningful topological features for network flow data, and if so, HOW.

CRITICAL QUESTIONS TO ANSWER:
1. Do network flow samples actually form meaningful point clouds?
2. Are the resulting persistence diagrams non-trivial?
3. What topological structure (if any) exists in the data?
4. How should we properly construct point clouds from tabular network data?
5. What does "topology" even mean for network flow features?

NO BULLSHIT APPROACH:
- Show actual persistence diagrams
- Visualize point clouds
- Count non-zero topological features
- Compare attack vs benign topology visually
- Document what fails and why
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from pathlib import Path

# Import TDA infrastructure
try:
    from src.core.persistent_homology import PersistentHomologyAnalyzer
    from src.core.mapper import MapperAnalyzer
    print("✓ TDA infrastructure imported")
except ImportError as e:
    print(f"❌ Cannot import TDA infrastructure: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TDAReality:
    """
    Honest assessment of TDA applicability to network flow data
    """
    
    def __init__(self):
        self.data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
        self.scaler = StandardScaler()
        
    def load_sample_data(self, attack_type="SSH-Bruteforce", n_samples=100):
        """Load small sample for analysis"""
        logger.info(f"Loading {n_samples} samples of {attack_type} attacks...")
        
        attack_samples = []
        benign_samples = []
        chunk_size = 5000
        numeric_cols = []
        
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            # Convert numeric columns
            if not numeric_cols:  # Set on first chunk
                numeric_cols = [col for col in chunk.columns 
                               if col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack']]
            
            for col in numeric_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            chunk = chunk.fillna(0)
            
            # Filter attack samples
            attack_chunk = chunk[chunk['Attack'] == attack_type]
            if len(attack_chunk) > 0:
                attack_samples.extend(attack_chunk.to_dict('records'))
            
            # Filter benign samples
            benign_chunk = chunk[chunk['Attack'] == 'Benign']
            if len(benign_chunk) > 0:
                benign_samples.extend(benign_chunk.to_dict('records'))
            
            if len(attack_samples) >= n_samples and len(benign_samples) >= n_samples:
                break
        
        if len(attack_samples) < n_samples:
            logger.warning(f"Only found {len(attack_samples)} {attack_type} samples")
        
        attack_df = pd.DataFrame(attack_samples[:n_samples])
        benign_df = pd.DataFrame(benign_samples[:n_samples])
        
        return attack_df, benign_df, numeric_cols
    
    def visualize_data_structure(self, attack_df, benign_df, numeric_cols):
        """Visualize the actual data structure before TDA"""
        logger.info("Analyzing data structure for TDA suitability...")
        
        # Get feature matrices
        attack_features = attack_df[numeric_cols].values
        benign_features = benign_df[numeric_cols].values
        
        print(f"\nDATA STRUCTURE ANALYSIS:")
        print(f"Attack samples shape: {attack_features.shape}")
        print(f"Benign samples shape: {benign_features.shape}")
        print(f"Feature range - Attack: [{attack_features.min():.3f}, {attack_features.max():.3f}]")
        print(f"Feature range - Benign: [{benign_features.min():.3f}, {benign_features.max():.3f}]")
        
        # Check for zero/constant features
        attack_nonzero = np.count_nonzero(attack_features, axis=0)
        benign_nonzero = np.count_nonzero(benign_features, axis=0)
        
        print(f"\nFEATURE ACTIVITY:")
        print(f"Attack - Non-zero features: {np.sum(attack_nonzero > 0)}/{len(numeric_cols)}")
        print(f"Benign - Non-zero features: {np.sum(benign_nonzero > 0)}/{len(numeric_cols)}")
        
        # Visualize distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature value distributions
        axes[0,0].hist(attack_features.flatten(), bins=50, alpha=0.7, label='Attack', density=True)
        axes[0,0].hist(benign_features.flatten(), bins=50, alpha=0.7, label='Benign', density=True)
        axes[0,0].set_xlabel('Feature Value')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Feature Value Distributions')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')
        
        # PCA visualization
        combined_features = np.vstack([attack_features, benign_features])
        labels = ['Attack'] * len(attack_features) + ['Benign'] * len(benign_features)
        
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined_features)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_scaled)
        
        attack_pca = pca_result[:len(attack_features)]
        benign_pca = pca_result[len(attack_features):]
        
        axes[0,1].scatter(attack_pca[:, 0], attack_pca[:, 1], alpha=0.6, label='Attack', s=20)
        axes[0,1].scatter(benign_pca[:, 0], benign_pca[:, 1], alpha=0.6, label='Benign', s=20)
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        axes[0,1].set_title('PCA Projection (Potential Point Cloud)')
        axes[0,1].legend()
        
        # Feature importance (variance)
        feature_vars = np.var(combined_scaled, axis=0)
        top_features = np.argsort(feature_vars)[-20:]  # Top 20 most variable
        
        axes[1,0].barh(range(len(top_features)), feature_vars[top_features])
        axes[1,0].set_xlabel('Variance')
        axes[1,0].set_ylabel('Feature Index')
        axes[1,0].set_title('Top 20 Most Variable Features')
        
        # Correlation structure
        corr_matrix = np.corrcoef(combined_scaled.T)
        im = axes[1,1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1,1].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('data_structure_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return combined_scaled, labels, top_features
    
    def test_point_cloud_construction_methods(self, data, labels, top_features):
        """Test different methods of constructing point clouds from tabular data"""
        logger.info("Testing point cloud construction methods...")
        
        attack_mask = np.array(labels) == 'Attack'
        attack_data = data[attack_mask]
        benign_data = data[~attack_mask]
        
        methods = {}
        
        # Method 1: Direct use of top 2 features
        methods['Top2Features'] = {
            'attack': attack_data[:, top_features[-2:]],
            'benign': benign_data[:, top_features[-2:]]
        }
        
        # Method 2: PCA to 2D
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)
        methods['PCA2D'] = {
            'attack': pca_result[attack_mask],
            'benign': pca_result[~attack_mask]
        }
        
        # Method 3: First 2 features
        methods['First2Features'] = {
            'attack': attack_data[:, :2],
            'benign': benign_data[:, :2]
        }
        
        # Method 4: Random 2 features
        np.random.seed(42)
        random_features = np.random.choice(data.shape[1], 2, replace=False)
        methods['Random2Features'] = {
            'attack': attack_data[:, random_features],
            'benign': benign_data[:, random_features]
        }
        
        # Visualize point clouds
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (method_name, point_clouds) in enumerate(methods.items()):
            ax = axes[i]
            
            attack_pc = point_clouds['attack']
            benign_pc = point_clouds['benign']
            
            ax.scatter(attack_pc[:, 0], attack_pc[:, 1], alpha=0.6, label='Attack', s=20, c='red')
            ax.scatter(benign_pc[:, 0], benign_pc[:, 1], alpha=0.6, label='Benign', s=20, c='blue')
            ax.set_title(f'Point Cloud: {method_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('point_cloud_methods.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return methods
    
    def analyze_persistence_diagrams(self, point_cloud_methods):
        """Analyze actual persistence diagrams from different point cloud methods"""
        logger.info("Computing persistence diagrams...")
        
        ph_analyzer = PersistentHomologyAnalyzer(maxdim=2, backend='ripser')
        
        results = {}
        
        for method_name, point_clouds in point_cloud_methods.items():
            print(f"\n{'='*60}")
            print(f"PERSISTENCE ANALYSIS: {method_name}")
            print(f"{'='*60}")
            
            method_results = {}
            
            for class_name, pc in point_clouds.items():
                print(f"\n{class_name.upper()} CLASS:")
                print(f"Point cloud shape: {pc.shape}")
                
                try:
                    # Compute persistence - ensure proper format
                    pc_array = np.array(pc)
                    ph_analyzer.fit(pc_array)
                    ph_result = ph_analyzer.persistence_diagrams_
                    
                    if ph_result and len(ph_result) > 0:
                        # Convert to expected format - list of intervals per dimension
                        intervals = []
                        for dim, diagram in enumerate(ph_result):
                            for birth, death in diagram:
                                intervals.append([dim, birth, death])
                        
                        if intervals:
                            # Analyze by dimension
                            h0_intervals = [interval for interval in intervals if interval[0] == 0]
                            h1_intervals = [interval for interval in intervals if interval[0] == 1]
                            h2_intervals = [interval for interval in intervals if interval[0] == 2]
                            
                            print(f"  H0 (components): {len(h0_intervals)} features")
                            print(f"  H1 (loops): {len(h1_intervals)} features")
                            print(f"  H2 (voids): {len(h2_intervals)} features")
                            
                            # Calculate persistence lifetimes
                            if h0_intervals:
                                h0_lifetimes = [interval[2] - interval[1] for interval in h0_intervals 
                                              if interval[2] != np.inf and not np.isnan(interval[2])]
                                print(f"  H0 max persistence: {max(h0_lifetimes) if h0_lifetimes else 0:.4f}")
                            
                            if h1_intervals:
                                h1_lifetimes = [interval[2] - interval[1] for interval in h1_intervals 
                                              if interval[2] != np.inf and not np.isnan(interval[2])]
                                print(f"  H1 max persistence: {max(h1_lifetimes) if h1_lifetimes else 0:.4f}")
                            
                            method_results[class_name] = {
                                'h0_count': len(h0_intervals),
                                'h1_count': len(h1_intervals),
                                'h2_count': len(h2_intervals),
                                'intervals': intervals
                            }
                        else:
                            print("  ❌ No persistence intervals found")
                            method_results[class_name] = {
                                'h0_count': 0, 'h1_count': 0, 'h2_count': 0,
                                'intervals': []
                            }
                        
                    else:
                        print("  ❌ No persistence intervals computed")
                        method_results[class_name] = {
                            'h0_count': 0, 'h1_count': 0, 'h2_count': 0,
                            'intervals': []
                        }
                        
                except Exception as e:
                    print(f"  ❌ Persistence computation failed: {e}")
                    method_results[class_name] = {
                        'h0_count': 0, 'h1_count': 0, 'h2_count': 0,
                        'intervals': []
                    }
            
            results[method_name] = method_results
        
        return results
    
    def visualize_persistence_diagrams(self, persistence_results, point_cloud_methods):
        """Visualize persistence diagrams for comparison"""
        logger.info("Visualizing persistence diagrams...")
        
        n_methods = len(persistence_results)
        fig, axes = plt.subplots(n_methods, 2, figsize=(12, 4*n_methods))
        
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        for i, (method_name, results) in enumerate(persistence_results.items()):
            
            # Attack persistence diagram
            ax_attack = axes[i, 0]
            if 'attack' in results and results['attack']['intervals']:
                intervals = results['attack']['intervals']
                
                for interval in intervals:
                    dim, birth, death = interval[0], interval[1], interval[2]
                    if death != np.inf:
                        color = ['red', 'blue', 'green'][int(dim)]
                        ax_attack.scatter(birth, death, c=color, alpha=0.7, s=50)
                
                # Diagonal line
                max_val = max([interval[2] for interval in intervals if interval[2] != np.inf] + [0])
                ax_attack.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                
            ax_attack.set_title(f'{method_name} - Attack')
            ax_attack.set_xlabel('Birth')
            ax_attack.set_ylabel('Death')
            
            # Benign persistence diagram
            ax_benign = axes[i, 1]
            if 'benign' in results and results['benign']['intervals']:
                intervals = results['benign']['intervals']
                
                for interval in intervals:
                    dim, birth, death = interval[0], interval[1], interval[2]
                    if death != np.inf:
                        color = ['red', 'blue', 'green'][int(dim)]
                        ax_benign.scatter(birth, death, c=color, alpha=0.7, s=50)
                
                # Diagonal line
                max_val = max([interval[2] for interval in intervals if interval[2] != np.inf] + [0])
                ax_benign.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                
            ax_benign.set_title(f'{method_name} - Benign')
            ax_benign.set_xlabel('Birth')
            ax_benign.set_ylabel('Death')
        
        plt.tight_layout()
        plt.savefig('persistence_diagrams.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def honest_assessment(self, persistence_results):
        """Provide honest assessment of TDA applicability"""
        print(f"\n{'='*80}")
        print("HONEST TDA REALITY CHECK")
        print(f"{'='*80}")
        
        total_topological_features = 0
        meaningful_features = 0
        
        for method_name, results in persistence_results.items():
            print(f"\nMETHOD: {method_name}")
            print("-" * 40)
            
            for class_name, class_results in results.items():
                h0 = class_results['h0_count']
                h1 = class_results['h1_count'] 
                h2 = class_results['h2_count']
                
                total_features = h0 + h1 + h2
                total_topological_features += total_features
                
                print(f"  {class_name}: H0={h0}, H1={h1}, H2={h2} (Total: {total_features})")
                
                # Consider features "meaningful" if we have non-trivial topology
                if h1 > 0 or h2 > 0:  # Loops or voids indicate interesting topology
                    meaningful_features += h1 + h2
        
        print(f"\nOVERALL ASSESSMENT:")
        print(f"Total topological features: {total_topological_features}")
        print(f"Meaningful features (H1+H2): {meaningful_features}")
        print(f"Percentage meaningful: {100*meaningful_features/max(total_topological_features,1):.1f}%")
        
        if meaningful_features == 0:
            print(f"\n❌ VERDICT: TDA produces NO meaningful topological structure")
            print(f"   - All topology is trivial (only H0 connected components)")
            print(f"   - Network flow data may not naturally form topological structures")
            print(f"   - Point cloud construction may be inappropriate for this data type")
        elif meaningful_features < total_topological_features * 0.1:
            print(f"\n⚠️ VERDICT: TDA produces MINIMAL meaningful topological structure")
            print(f"   - Most features are trivial connected components")
            print(f"   - Limited discriminative power expected")
        else:
            print(f"\n✓ VERDICT: TDA produces some meaningful topological structure")
            print(f"   - Non-trivial loops and voids detected")
            print(f"   - May have discriminative power for classification")
        
        print(f"\nRECOMMENDATIONS:")
        if meaningful_features == 0:
            print(f"1. Consider whether TDA is appropriate for this problem")
            print(f"2. Explore alternative point cloud construction methods")
            print(f"3. Consider using graph-based TDA instead of point cloud TDA")
            print(f"4. Evaluate simpler statistical methods first")
        else:
            print(f"1. Focus on methods that produce non-trivial topology")
            print(f"2. Extract features from H1 and H2 persistence")
            print(f"3. Compare persistence landscapes between classes")
            print(f"4. Validate discriminative power properly")

def main():
    """Run complete TDA reality check"""
    print("STARTING TDA REALITY CHECK...")
    print("This will honestly assess whether TDA can work for network flow data")
    
    tda_reality = TDAReality()
    
    # Load sample data
    attack_df, benign_df, numeric_cols = tda_reality.load_sample_data("SSH-Bruteforce", 50)
    
    if len(attack_df) == 0:
        print("❌ No attack samples found - cannot proceed")
        return
    
    # Analyze data structure
    combined_data, labels, top_features = tda_reality.visualize_data_structure(
        attack_df, benign_df, numeric_cols
    )
    
    # Test point cloud construction
    point_cloud_methods = tda_reality.test_point_cloud_construction_methods(
        combined_data, labels, top_features
    )
    
    # Analyze persistence
    persistence_results = tda_reality.analyze_persistence_diagrams(point_cloud_methods)
    
    # Visualize persistence diagrams
    tda_reality.visualize_persistence_diagrams(persistence_results, point_cloud_methods)
    
    # Honest assessment
    tda_reality.honest_assessment(persistence_results)
    
    print(f"\n🔍 DIAGNOSTIC COMPLETE")
    print(f"📊 Visualizations saved: data_structure_analysis.png, point_cloud_methods.png, persistence_diagrams.png")

if __name__ == "__main__":
    main()
