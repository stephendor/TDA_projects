#!/usr/bin/env python3
"""
UNSW-NB15 TDA Validation with Granular Attack-Type Analysis
Addresses the issue of hiding failure states behind aggregate statistics
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GranularAttackAnalyzer:
    """UNSW-NB15 TDA validation with detailed attack-type analysis"""
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"validation/unsw_nb15_granular_analysis/{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectories following project structure
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        self.attack_types = []
        self.results = {}
    
    def load_and_preprocess_data(self, sample_size=15000):
        """Load UNSW-NB15 data with attack category preservation"""
        print("📂 Loading UNSW-NB15 data with attack categories...")
        
        train_path = "data/apt_datasets/UNSW-NB15/UNSW_NB15_training-set.parquet"
        test_path = "data/apt_datasets/UNSW-NB15/UNSW_NB15_testing-set.parquet"
        
        # Load and combine
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"   Total samples: {len(df):,}")
        print(f"   Attack types: {df['attack_cat'].nunique()}")
        
        # Show attack distribution
        attack_dist = df['attack_cat'].value_counts()
        print("   Attack distribution:")
        for attack_type, count in attack_dist.items():
            print(f"     {attack_type}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Stratified sampling to maintain attack type distribution
        if len(df) > sample_size:
            # Sample proportionally from each attack type
            sampled_dfs = []
            for attack_type in df['attack_cat'].unique():
                attack_df = df[df['attack_cat'] == attack_type]
                attack_proportion = len(attack_df) / len(df)
                attack_sample_size = max(1, int(sample_size * attack_proportion))
                
                if len(attack_df) > attack_sample_size:
                    attack_sample = attack_df.sample(n=attack_sample_size, random_state=42)
                else:
                    attack_sample = attack_df
                
                sampled_dfs.append(attack_sample)
                print(f"     Sampled {attack_type}: {len(attack_sample):,}")
            
            df = pd.concat(sampled_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   Final dataset: {len(df):,} samples")
        
        # Store attack types for later analysis
        self.attack_types = sorted([t for t in df['attack_cat'].unique() if t != 'Normal'])
        print(f"   Attack types for analysis: {self.attack_types}")
        
        # Select numeric features for TDA processing
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in ['label', 'attack_cat']][:25]  # Limit to 25 features
        
        X = df[numeric_cols].copy()
        y_binary = df['label'].values  # Binary classification
        y_multiclass = df['attack_cat'].values  # Multi-class for granular analysis
        
        # Clean data
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"   Features: {X.shape[1]}")
        print(f"   Clean samples: {len(X):,}")
        
        return X.values, y_binary, y_multiclass, list(X.columns)
    
    def extract_optimized_features(self, X):
        """Extract TDA features efficiently using vectorized operations"""
        print("🔮 Extracting optimized TDA features...")
        start_time = time.time()
        
        n_samples, n_features = X.shape
        
        # Scale data once
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        all_features = []
        
        # 1. Statistical features (vectorized)
        print("   Statistical features...")
        stat_features = self.extract_fast_statistical_features(X_scaled)
        all_features.append(stat_features)
        
        # 2. Geometric features (optimized)
        print("   Geometric features...")
        geom_features = self.extract_fast_geometric_features(X_scaled)
        all_features.append(geom_features)
        
        # 3. Local density features (sampled)
        print("   Local density features...")
        density_features = self.extract_fast_density_features(X_scaled)
        all_features.append(density_features)
        
        # Combine
        combined_features = np.concatenate(all_features, axis=1)
        
        extraction_time = time.time() - start_time
        print(f"   ✅ Feature extraction: {extraction_time:.1f}s")
        print(f"   Total features: {combined_features.shape[1]}")
        
        return combined_features
    
    def extract_fast_statistical_features(self, X):
        """Fast statistical features using numpy vectorization"""
        features = np.column_stack([
            np.mean(X, axis=1),          # Mean
            np.std(X, axis=1),           # Std
            np.median(X, axis=1),        # Median
            np.min(X, axis=1),           # Min
            np.max(X, axis=1),           # Max
            np.percentile(X, 25, axis=1), # Q1
            np.percentile(X, 75, axis=1), # Q3
            np.var(X, axis=1),           # Variance
            np.ptp(X, axis=1),           # Range
            np.sum(X**2, axis=1),        # Energy
        ])
        return features
    
    def extract_fast_geometric_features(self, X):
        """Fast geometric features using broadcasting"""
        n_samples = X.shape[0]
        
        # Global centroid
        global_centroid = np.mean(X, axis=0)
        
        # Distance to global centroid (vectorized)
        centroid_distances = np.linalg.norm(X - global_centroid, axis=1)
        
        # Local features using sliding window
        window_size = 10
        local_features = np.zeros((n_samples, 5))
        
        for i in range(n_samples):
            start_idx = max(0, i - window_size)
            end_idx = min(n_samples, i + window_size + 1)
            local_data = X[start_idx:end_idx]
            
            if len(local_data) > 1:
                local_centroid = np.mean(local_data, axis=0)
                local_dist = np.linalg.norm(X[i] - local_centroid)
                
                # Fast local statistics
                distances_to_local = np.linalg.norm(local_data - X[i], axis=1)
                distances_to_local = distances_to_local[distances_to_local > 0]  # Exclude self
                
                if len(distances_to_local) > 0:
                    local_features[i] = [
                        local_dist,
                        np.mean(distances_to_local),
                        np.min(distances_to_local),
                        np.std(distances_to_local),
                        len(distances_to_local)
                    ]
        
        # Combine all geometric features
        geom_features = np.column_stack([
            centroid_distances,
            local_features
        ])
        
        return geom_features
    
    def extract_fast_density_features(self, X):
        """Fast density features using sampling"""
        n_samples = X.shape[0]
        
        # Sample subset for density estimation (for efficiency)
        sample_size = min(5000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_indices]
        
        # For each point, compute density features
        density_features = np.zeros((n_samples, 4))
        
        # Batch process in chunks
        chunk_size = 1000
        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            chunk = X[i:end_i]
            
            # Distance to sample points
            for j, point in enumerate(chunk):
                actual_idx = i + j
                
                # Distance to sampled points
                distances = np.linalg.norm(X_sample - point, axis=1)
                
                # Density features
                density_features[actual_idx] = [
                    np.mean(distances),      # Average distance
                    np.std(distances),       # Distance variance
                    np.min(distances),       # Nearest neighbor
                    np.percentile(distances, 10)  # 10th percentile
                ]
        
        return density_features
    
    def train_and_evaluate_models(self, X, y_binary, y_multiclass):
        """Train models and perform comprehensive evaluation"""
        print("\n🎯 Training models with granular analysis...")
        
        # Split data preserving attack type distribution
        X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
            X, y_binary, y_multiclass, test_size=0.3, random_state=42, stratify=y_binary
        )
        
        print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
        print(f"   Train attack rate: {y_train_bin.mean()*100:.1f}%")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test multiple models
        models = {
            'Logistic Regression': LogisticRegression(
                C=0.1, max_iter=1000, class_weight='balanced', random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced', 
                random_state=42, n_jobs=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            start_time = time.time()
            
            # Train on binary classification
            model.fit(X_train_scaled, y_train_bin)
            training_time = time.time() - start_time
            
            # Binary predictions
            y_pred_bin = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Overall binary classification metrics
            f1_overall = f1_score(y_test_bin, y_pred_bin)
            accuracy_overall = (y_pred_bin == y_test_bin).mean()
            
            # Calculate precision and recall manually for safety
            tp = ((y_pred_bin == 1) & (y_test_bin == 1)).sum()
            fp = ((y_pred_bin == 1) & (y_test_bin == 0)).sum()
            fn = ((y_pred_bin == 0) & (y_test_bin == 1)).sum()
            tn = ((y_pred_bin == 0) & (y_test_bin == 0)).sum()
            
            precision_overall = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_overall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_test_bin, y_pred_proba)
            except Exception:
                roc_auc = None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train_bin, cv=3, scoring='f1')
            
            # GRANULAR ATTACK-TYPE ANALYSIS
            attack_type_metrics = self.analyze_by_attack_type(
                y_test_multi, y_pred_bin, y_test_bin
            )
            
            results[name] = {
                # Overall metrics
                'f1_score': f1_overall,
                'accuracy': accuracy_overall,
                'precision': precision_overall,
                'recall': recall_overall,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)},
                
                # Granular attack-type analysis
                'attack_type_metrics': attack_type_metrics,
                
                # For plotting
                'y_test_binary': y_test_bin,
                'y_test_multiclass': y_test_multi,
                'y_pred_binary': y_pred_bin
            }
            
            print(f"      Overall: F1={f1_overall:.3f}, Acc={accuracy_overall:.3f}, Prec={precision_overall:.3f}, Rec={recall_overall:.3f}")
            
            # Print granular results summary
            print("      Attack-type performance:")
            for attack_type, metrics in attack_type_metrics.items():
                if attack_type != 'Normal':  # Skip normal since it's handled by overall metrics
                    print(f"        {attack_type}: F1={metrics['f1']:.3f}, Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}, Count={metrics['support']}")
        
        return results
    
    def analyze_by_attack_type(self, y_test_multi, y_pred_bin, y_test_bin):
        """Perform detailed analysis by attack type"""
        attack_metrics = {}
        
        # Get unique attack types in test set
        unique_attacks = np.unique(y_test_multi)
        
        for attack_type in unique_attacks:
            # Get indices for this attack type
            attack_indices = y_test_multi == attack_type
            
            if attack_type == 'Normal':
                # For normal traffic, we want to see how well we predict normal (label=0)
                true_labels = y_test_bin[attack_indices]
                pred_labels = y_pred_bin[attack_indices]
                
                # For normal, positive class is actually the normal class (0)
                # But sklearn expects 1 as positive, so we'll invert for Normal
                if len(true_labels) > 0:
                    tp_norm = np.sum((true_labels == 0) & (pred_labels == 0))
                    fp_norm = np.sum((true_labels == 1) & (pred_labels == 0))
                    fn_norm = np.sum((true_labels == 0) & (pred_labels == 1))
                    tn_norm = np.sum((true_labels == 1) & (pred_labels == 1))
                    
                    precision = tp_norm / (tp_norm + fp_norm) if (tp_norm + fp_norm) > 0 else 0.0
                    recall = tp_norm / (tp_norm + fn_norm) if (tp_norm + fn_norm) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    attack_metrics[attack_type] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'support': len(true_labels),
                        'tp': tp_norm,
                        'fp': fp_norm,
                        'fn': fn_norm,
                        'tn': tn_norm
                    }
            else:
                # For attack types, positive class is attack (label=1)
                true_labels = y_test_bin[attack_indices]
                pred_labels = y_pred_bin[attack_indices]
                
                if len(true_labels) > 0:
                    tp = np.sum((true_labels == 1) & (pred_labels == 1))
                    fp = np.sum((true_labels == 0) & (pred_labels == 1))
                    fn = np.sum((true_labels == 1) & (pred_labels == 0))
                    tn = np.sum((true_labels == 0) & (pred_labels == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    attack_metrics[attack_type] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'support': len(true_labels),
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'tn': tn
                    }
        
        return attack_metrics
    
    def generate_granular_outputs(self, results):
        """Generate comprehensive outputs including attack-type analysis"""
        print("\n📊 Generating granular analysis outputs...")
        
        # Select best model
        best_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_result = results[best_name]
        best_f1 = best_result['f1_score']
        
        # 1. Attack-type performance plot
        self.plot_attack_type_performance(results)
        
        # 2. Attack-type confusion matrices
        self.plot_attack_type_confusion_matrices(best_result)
        
        # 3. Overall confusion matrix
        self.plot_overall_confusion_matrix(best_result, best_name)
        
        # 4. Model comparison with granular details
        self.plot_model_comparison_detailed(results)
        
        # 5. Save comprehensive results
        self.save_granular_results(results, best_name, best_f1)
        
        # 6. Generate detailed report
        self.generate_detailed_report(results, best_name, best_f1)
        
        print(f"   ✅ All outputs saved to: {self.output_dir}")
        
        return best_name, best_f1
    
    def plot_attack_type_performance(self, results):
        """Plot performance metrics by attack type"""
        best_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        attack_metrics = results[best_name]['attack_type_metrics']
        
        attack_types = list(attack_metrics.keys())
        f1_scores = [attack_metrics[att]['f1'] for att in attack_types]
        precisions = [attack_metrics[att]['precision'] for att in attack_types]
        recalls = [attack_metrics[att]['recall'] for att in attack_types]
        supports = [attack_metrics[att]['support'] for att in attack_types]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1 Scores by attack type
        bars1 = ax1.bar(attack_types, f1_scores)
        ax1.set_title(f'F1-Score by Attack Type - {best_name}')
        ax1.set_ylabel('F1-Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Target (0.75)')
        ax1.legend()
        
        # Color code bars
        for bar, f1 in zip(bars1, f1_scores):
            if f1 >= 0.75:
                bar.set_color('green')
            elif f1 >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Precision by attack type
        bars2 = ax2.bar(attack_types, precisions, color='lightblue')
        ax2.set_title('Precision by Attack Type')
        ax2.set_ylabel('Precision')
        ax2.tick_params(axis='x', rotation=45)
        
        # Recall by attack type
        bars3 = ax3.bar(attack_types, recalls, color='lightcoral')
        ax3.set_title('Recall by Attack Type')
        ax3.set_ylabel('Recall')
        ax3.tick_params(axis='x', rotation=45)
        
        # Support (sample count) by attack type
        bars4 = ax4.bar(attack_types, supports, color='lightgreen')
        ax4.set_title('Sample Count by Attack Type')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / f"attack_type_performance_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Attack-type performance plot: {plot_path}")
    
    def plot_attack_type_confusion_matrices(self, best_result):
        """Plot individual confusion matrices for each attack type"""
        attack_metrics = best_result['attack_type_metrics']
        
        n_attacks = len(attack_metrics)
        cols = 3
        rows = (n_attacks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (attack_type, metrics) in enumerate(attack_metrics.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Create confusion matrix for this attack type
            cm = np.array([[metrics['tn'], metrics['fp']], 
                          [metrics['fn'], metrics['tp']]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Predicted Normal', 'Predicted Attack'],
                       yticklabels=['True Normal', 'True Attack'])
            
            ax.set_title(f'{attack_type}\nF1: {metrics["f1"]:.3f} | Prec: {metrics["precision"]:.3f} | Rec: {metrics["recall"]:.3f}')
        
        # Hide empty subplots
        for idx in range(n_attacks, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / f"attack_type_confusion_matrices_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Attack-type confusion matrices: {plot_path}")
    
    def plot_overall_confusion_matrix(self, best_result, best_name):
        """Plot overall binary classification confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(best_result['y_test_binary'], best_result['y_pred_binary'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        
        plt.title(f'Overall Confusion Matrix - {best_name}\\nF1-Score: {best_result["f1_score"]:.3f}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        
        plot_path = self.output_dir / "plots" / f"overall_confusion_matrix_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Overall confusion matrix: {plot_path}")
    
    def plot_model_comparison_detailed(self, results):
        """Plot detailed model comparison"""
        models = list(results.keys())
        overall_f1s = [results[m]['f1_score'] for m in models]
        
        plt.figure(figsize=(12, 8))
        
        # Overall F1 comparison
        bars = plt.bar(models, overall_f1s, alpha=0.8)
        plt.axhline(y=0.75, color='red', linestyle='--', label='Target (0.75)')
        plt.axhline(y=0.567, color='orange', linestyle='--', label='Baseline (0.567)')
        plt.ylabel('F1-Score')
        plt.title('UNSW-NB15 TDA Enhanced Results - Model Comparison')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Color bars based on performance
        for bar, f1 in zip(bars, overall_f1s):
            if f1 >= 0.75:
                bar.set_color('green')
            elif f1 >= 0.65:
                bar.set_color('orange') 
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / f"model_comparison_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Model comparison plot: {plot_path}")
    
    def save_granular_results(self, results, best_name, best_f1):
        """Save comprehensive results with granular analysis"""
        results_data = {
            'best_model': best_name,
            'best_f1_score': float(best_f1),
            'target_f1': 0.75,
            'baseline_f1': 0.567,
            'improvement_vs_baseline': float(best_f1 - 0.567),
            'improvement_vs_target': float(best_f1 - 0.75),
            'timestamp': self.timestamp,
            'attack_types_analyzed': self.attack_types,
            'models': {}
        }
        
        for name, result in results.items():
            # Clean result for JSON serialization
            model_data = {
                'f1_score': float(result['f1_score']),
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'roc_auc': float(result['roc_auc']) if result['roc_auc'] is not None else None,
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std']),
                'training_time': float(result['training_time']),
                'confusion_matrix': result['confusion_matrix'],
                'attack_type_metrics': {}
            }
            
            # Add attack-type metrics
            for attack_type, metrics in result['attack_type_metrics'].items():
                model_data['attack_type_metrics'][attack_type] = {
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1']),
                    'support': int(metrics['support']),
                    'confusion_matrix': {
                        'tp': int(metrics['tp']),
                        'fp': int(metrics['fp']),
                        'fn': int(metrics['fn']),
                        'tn': int(metrics['tn'])
                    }
                }
            
            results_data['models'][name] = model_data
        
        # Save main results
        results_path = self.output_dir / "results" / "metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"   ✅ Results saved: {results_path}")
        
        return results_data
    
    def generate_detailed_report(self, results, best_name, best_f1):
        """Generate comprehensive markdown report with granular analysis"""
        best_result = results[best_name]
        attack_metrics = best_result['attack_type_metrics']
        
        # Format ROC-AUC separately to avoid f-string issues
        roc_auc_str = f"{best_result['roc_auc']:.3f}" if best_result['roc_auc'] is not None else 'N/A'
        
        report_content = f"""# UNSW-NB15 TDA Enhanced Validation - Granular Attack Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Experiment ID:** {self.timestamp}

## Executive Summary

- **Best Model:** {best_name}
- **Overall F1-Score:** {best_f1:.3f}
- **vs Target (0.75):** {best_f1 - 0.75:+.3f}
- **vs Baseline (0.567):** {best_f1 - 0.567:+.3f} ({((best_f1 - 0.567)/0.567)*100:+.1f}%)

## Overall Model Performance

### {best_name}
- **F1-Score**: {best_result['f1_score']:.3f}
- **Accuracy**: {best_result['accuracy']:.3f}
- **Precision**: {best_result['precision']:.3f}
- **Recall**: {best_result['recall']:.3f}
- **ROC-AUC**: {roc_auc_str}
- **Cross-Validation**: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}
- **Training Time**: {best_result['training_time']:.1f}s

## 🎯 Granular Attack-Type Analysis

This section addresses the concern of hiding failure states behind aggregate statistics.

"""
        
        # Add detailed attack-type analysis
        report_content += "### Performance by Attack Type\n\n"
        report_content += "| Attack Type | F1-Score | Precision | Recall | Support | Status |\n"
        report_content += "|-------------|----------|-----------|--------|---------|--------|\n"
        
        failure_types = []
        success_types = []
        
        for attack_type, metrics in attack_metrics.items():
            f1 = metrics['f1']
            precision = metrics['precision']
            recall = metrics['recall']
            support = metrics['support']
            
            if f1 >= 0.75:
                status = "✅ Excellent"
                success_types.append(attack_type)
            elif f1 >= 0.5:
                status = "⚠️ Acceptable"
            else:
                status = "❌ Poor"
                failure_types.append(attack_type)
            
            report_content += f"| {attack_type} | {f1:.3f} | {precision:.3f} | {recall:.3f} | {support} | {status} |\n"
        
        # Identify failure patterns
        report_content += f"\n### 🚨 Failure Analysis\n\n"
        if failure_types:
            report_content += f"**Attack types with poor performance (F1 < 0.5):**\n"
            for attack_type in failure_types:
                metrics = attack_metrics[attack_type]
                report_content += f"- **{attack_type}**: F1={metrics['f1']:.3f}, likely due to "
                if metrics['precision'] < 0.3:
                    report_content += "high false positive rate (low precision)"
                elif metrics['recall'] < 0.3:
                    report_content += "high false negative rate (low recall)"
                else:
                    report_content += "balanced but overall poor performance"
                report_content += f" with {metrics['support']} samples\n"
        else:
            report_content += "✅ No attack types with poor performance identified.\n"
        
        # Success patterns
        report_content += f"\n### 🎉 Success Analysis\n\n"
        if success_types:
            report_content += f"**Attack types with excellent performance (F1 ≥ 0.75):**\n"
            for attack_type in success_types:
                metrics = attack_metrics[attack_type]
                report_content += f"- **{attack_type}**: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f} ({metrics['support']} samples)\n"
        
        # Improvement opportunities
        report_content += f"\n### 📈 Improvement Opportunities\n\n"
        low_precision_types = [att for att, metrics in attack_metrics.items() 
                              if metrics['precision'] < 0.7 and att != 'Normal']
        low_recall_types = [att for att, metrics in attack_metrics.items() 
                           if metrics['recall'] < 0.7 and att != 'Normal']
        
        if low_precision_types:
            report_content += f"**High false positive rates (low precision):** {', '.join(low_precision_types)}\n"
            report_content += "- *Recommendation*: Add more discriminative features or adjust decision threshold\n\n"
        
        if low_recall_types:
            report_content += f"**High false negative rates (low recall):** {', '.join(low_recall_types)}\n"
            report_content += "- *Recommendation*: Investigate feature adequacy for these attack patterns\n\n"
        
        # Add comparison with other models if available
        if len(results) > 1:
            report_content += "## Model Comparison\n\n"
            for name, result in results.items():
                if name != best_name:
                    report_content += f"### {name}\n"
                    report_content += f"- **Overall F1-Score**: {result['f1_score']:.3f}\n"
                    report_content += f"- **Training Time**: {result['training_time']:.1f}s\n\n"
        
        # Files generated
        report_content += f"""
## Files Generated

- **Attack-Type Performance**: `plots/attack_type_performance_{self.timestamp}.png`
- **Attack-Type Confusion Matrices**: `plots/attack_type_confusion_matrices_{self.timestamp}.png`
- **Overall Confusion Matrix**: `plots/overall_confusion_matrix_{self.timestamp}.png`
- **Model Comparison**: `plots/model_comparison_{self.timestamp}.png`
- **Detailed Results**: `results/metrics.json`

---
*Generated by UNSW-NB15 Granular Attack Analysis Pipeline*
*This report provides attack-type-specific insights to avoid hiding failure states behind aggregate statistics*
"""
        
        report_path = self.output_dir / "VALIDATION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"   ✅ Detailed report: {report_path}")

def run_granular_attack_analysis():
    """Run UNSW-NB15 validation with granular attack-type analysis"""
    print("🔬 UNSW-NB15 GRANULAR ATTACK-TYPE ANALYSIS")
    print("=" * 80)
    
    analyzer = GranularAttackAnalyzer()
    
    start_time = time.time()
    
    # Load data with attack categories preserved
    X, y_binary, y_multiclass, feature_names = analyzer.load_and_preprocess_data()
    
    # Extract TDA features
    X_enhanced = analyzer.extract_optimized_features(X)
    
    # Train and evaluate with granular analysis
    results = analyzer.train_and_evaluate_models(X_enhanced, y_binary, y_multiclass)
    
    # Generate comprehensive outputs
    best_model, best_f1 = analyzer.generate_granular_outputs(results)
    
    total_time = time.time() - start_time
    
    print(f"\n🎯 GRANULAR ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Best Model: {best_model}")
    print(f"Overall F1-Score: {best_f1:.3f}")
    print(f"Target (0.75): {best_f1 - 0.75:+.3f}")
    print(f"Baseline (0.567): {best_f1 - 0.567:+.3f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Output: {analyzer.output_dir}")
    
    # Show attack-type summary
    best_result = results[best_model]
    attack_metrics = best_result['attack_type_metrics']
    
    print(f"\n📊 ATTACK-TYPE PERFORMANCE SUMMARY:")
    failure_count = 0
    for attack_type, metrics in attack_metrics.items():
        if attack_type != 'Normal':
            f1 = metrics['f1']
            status = "✅" if f1 >= 0.75 else "⚠️" if f1 >= 0.5 else "❌"
            if f1 < 0.5:
                failure_count += 1
            print(f"  {status} {attack_type}: F1={f1:.3f} (n={metrics['support']})")
    
    if failure_count > 0:
        print(f"\n⚠️  {failure_count} attack types show poor performance - see detailed report for analysis")
    else:
        print(f"\n✅ All attack types show acceptable performance")
    
    return best_f1, analyzer.output_dir

if __name__ == "__main__":
    f1_score, output_dir = run_granular_attack_analysis()