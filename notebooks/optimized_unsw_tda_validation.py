#!/usr/bin/env python3
"""
Optimized UNSW-NB15 TDA Validation - Fast Feature Extraction
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
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class OptimizedUNSWValidator:
    """Optimized UNSW-NB15 TDA validation with fast feature extraction"""
    
    def __init__(self, output_dir="optimized_unsw_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
    
    def load_and_preprocess_data(self, sample_size=15000):
        """Load and preprocess UNSW-NB15 data efficiently"""
        print("📂 Loading UNSW-NB15 data...")
        
        train_path = "data/apt_datasets/UNSW-NB15/UNSW_NB15_training-set.parquet"
        test_path = "data/apt_datasets/UNSW-NB15/UNSW_NB15_testing-set.parquet"
        
        # Load and combine
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"   Total samples: {len(df):,}")
        print(f"   Attack rate: {(df['label'] == 1).mean()*100:.1f}%")
        
        # Stratified sample for efficiency
        if len(df) > sample_size:
            df_normal = df[df['label'] == 0]
            df_attack = df[df['label'] == 1]
            
            normal_size = min(len(df_normal), sample_size // 3)  # 33% normal
            attack_size = min(len(df_attack), sample_size - normal_size)  # 67% attacks
            
            normal_sample = df_normal.sample(n=normal_size, random_state=42)
            attack_sample = df_attack.sample(n=attack_size, random_state=42)
            
            df = pd.concat([normal_sample, attack_sample], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   Sampled: {len(df):,} samples")
        print(f"   Final attack rate: {(df['label'] == 1).mean()*100:.1f}%")
        
        # Select numeric features, exclude label
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'label'][:25]  # Limit to 25 features
        
        X = df[numeric_cols].copy()
        y = df['label'].values
        
        # Clean data efficiently
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"   Features: {X.shape[1]}")
        print(f"   Clean samples: {len(X):,}")
        
        return X.values, y, list(X.columns)
    
    def extract_optimized_features(self, X):
        """Extract features efficiently using vectorized operations"""
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
        # All computed in vectorized fashion
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
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate models efficiently"""
        print("\n🎯 Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
        print(f"   Train attack rate: {y_train.mean()*100:.1f}%")
        
        # Scale
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
            
            # Train
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate all standard classification metrics
            f1 = f1_score(y_test, y_pred)
            accuracy = (y_pred == y_test).mean()
            
            # Calculate precision and recall manually for safety
            tp = ((y_pred == 1) & (y_test == 1)).sum()
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            fn = ((y_pred == 0) & (y_test == 1)).sum()
            tn = ((y_pred == 0) & (y_test == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except Exception:
                roc_auc = None
            
            # Cross-validation (quick)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='f1')
            
            results[name] = {
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'y_test': y_test,
                'y_pred': y_pred,
                'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}
            }
            
            print(f"      F1: {f1:.3f}, Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}")
            print(f"      CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, Time: {training_time:.1f}s")
        
        return results
    
    def generate_outputs(self, results):
        """Generate validation outputs"""
        print("\n📊 Generating outputs...")
        
        # Best model
        best_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_f1 = results[best_name]['f1_score']
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(results[best_name]['y_test'], results[best_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix - {best_name}\nF1-Score: {best_f1:.3f}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        
        cm_path = self.output_dir / "plots" / f"confusion_matrix_{self.timestamp}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model comparison
        plt.figure(figsize=(10, 6))
        models = list(results.keys())
        f1_scores = [results[m]['f1_score'] for m in models]
        
        bars = plt.bar(models, f1_scores, alpha=0.8)
        plt.axhline(y=0.75, color='red', linestyle='--', label='Target (0.75)')
        plt.axhline(y=0.567, color='orange', linestyle='--', label='Baseline (0.567)')
        plt.ylabel('F1-Score')
        plt.title('UNSW-NB15 TDA Enhanced Results')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Color bars based on performance
        for bar, f1 in zip(bars, f1_scores):
            if f1 >= 0.75:
                bar.set_color('green')
            elif f1 >= 0.65:
                bar.set_color('orange') 
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        comp_path = self.output_dir / "plots" / f"model_comparison_{self.timestamp}.png"
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results
        results_data = {
            'best_model': best_name,
            'best_f1_score': float(best_f1),
            'target_f1': 0.75,
            'baseline_f1': 0.567,
            'improvement_vs_baseline': float(best_f1 - 0.567),
            'timestamp': self.timestamp,
            'models': {}
        }
        
        for name, result in results.items():
            results_data['models'][name] = {
                'f1_score': float(result['f1_score']),
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'roc_auc': float(result['roc_auc']) if result['roc_auc'] is not None else None,
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std']),
                'training_time': float(result['training_time']),
                'confusion_matrix': result['confusion_matrix']
            }
        
        results_path = self.output_dir / "data" / f"results_{self.timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Generate report
        report_content = f"""# Optimized UNSW-NB15 TDA Validation Results

**Timestamp:** {self.timestamp}
**Target:** 0.75 F1-Score
**Baseline:** 0.567 F1-Score

## Results

**Best Model:** {best_name}
**Best F1-Score:** {best_f1:.3f}
**vs Target:** {best_f1 - 0.75:+.3f}
**vs Baseline:** {best_f1 - 0.567:+.3f} ({((best_f1 - 0.567)/0.567)*100:+.1f}%)

## Model Performance

"""
        for name, result in results.items():
            roc_auc_str = f"{result['roc_auc']:.3f}" if result['roc_auc'] is not None else "N/A"
            report_content += f"""### {name}
- **F1-Score**: {result['f1_score']:.3f}
- **Accuracy**: {result['accuracy']:.3f}
- **Precision**: {result['precision']:.3f}
- **Recall**: {result['recall']:.3f}
- **ROC-AUC**: {roc_auc_str}
- **Cross-Validation**: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}
- **Training Time**: {result['training_time']:.1f}s
- **Confusion Matrix**: TP={result['confusion_matrix']['tp']}, FP={result['confusion_matrix']['fp']}, FN={result['confusion_matrix']['fn']}, TN={result['confusion_matrix']['tn']}

"""
        
        report_content += f"""
## Files Generated
- Confusion Matrix: `plots/confusion_matrix_{self.timestamp}.png`
- Model Comparison: `plots/model_comparison_{self.timestamp}.png`  
- Results Data: `data/results_{self.timestamp}.json`
"""
        
        report_path = self.output_dir / "reports" / f"report_{self.timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"   ✅ Outputs saved to: {self.output_dir}")
        
        return best_name, best_f1

def run_optimized_validation():
    """Run optimized UNSW-NB15 validation"""
    print("🔬 OPTIMIZED UNSW-NB15 TDA VALIDATION")
    print("=" * 80)
    
    validator = OptimizedUNSWValidator()
    
    start_time = time.time()
    
    # Load data
    X, y, feature_names = validator.load_and_preprocess_data()
    
    # Extract features 
    X_enhanced = validator.extract_optimized_features(X)
    
    # Train and evaluate
    results = validator.train_and_evaluate(X_enhanced, y)
    
    # Generate outputs
    best_model, best_f1 = validator.generate_outputs(results)
    
    total_time = time.time() - start_time
    
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 50)
    print(f"Best Model: {best_model}")
    print(f"F1-Score: {best_f1:.3f}")
    print(f"Target (0.75): {best_f1 - 0.75:+.3f}")
    print(f"Baseline (0.567): {best_f1 - 0.567:+.3f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Output: {validator.output_dir}")
    
    if best_f1 >= 0.75:
        status = "🎉 TARGET ACHIEVED"
    elif best_f1 >= 0.65:
        status = "📈 GOOD PROGRESS"
    else:
        status = "🔧 NEEDS WORK"
    
    print(f"Status: {status}")
    
    return best_f1

if __name__ == "__main__":
    f1_score = run_optimized_validation()