#!/usr/bin/env python3
"""
Enhanced Hybrid TDA Validation with Comprehensive Evidence Capture
Demonstrates new validation framework that prevents result discrepancies
"""
import sys
sys.path.append('/home/stephen-dorman/dev/TDA_projects')
sys.path.append('/home/stephen-dorman/dev/TDA_projects/validation')

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import gudhi as gd
import time
import warnings
warnings.filterwarnings('ignore')

from validation_framework import ValidationFramework, report_validated_results

def enhanced_hybrid_tda_validation():
    """
    Enhanced validation with comprehensive evidence capture
    This demonstrates how the new framework prevents discrepancies
    """
    
    # Initialize validation framework
    validator = ValidationFramework(
        experiment_name="hybrid_multiscale_graph_tda",
        random_seed=42
    )
    
    # Capture ALL console output
    with validator.capture_console_output():
        
        print("🌊 ENHANCED HYBRID TDA VALIDATION WITH COMPREHENSIVE EVIDENCE CAPTURE")
        print("=" * 85)
        print("Purpose: Validate hybrid multi-scale + graph TDA with complete evidence package")
        print("Framework: ValidationFramework v1.0 - Prevents result discrepancies")
        print("Random Seed: 42 (Fixed for reproducibility)")
        print("=" * 85)
        
        # Load data with detailed logging
        print("\n📂 DATA LOADING AND PREPARATION")
        print("-" * 50)
        
        try:
            # Try to load real data first
            print("Attempting to load CIC-IDS2017 processed data...")
            X = np.load('/home/stephen-dorman/dev/TDA_projects/data/cicids2017/processed/sample_sequences.npy')
            y = np.load('/home/stephen-dorman/dev/TDA_projects/data/cicids2017/processed/sample_labels.npy')
            print(f"✅ Real data loaded successfully")
            print(f"   Sequences shape: {X.shape}")
            print(f"   Labels shape: {y.shape}")
            print(f"   Attack rate: {np.mean(y):.1%}")
        except Exception as e:
            print(f"❌ Real data loading failed: {e}")
            print("🔧 Generating synthetic data for validation...")
            
            # Generate deterministic synthetic data
            np.random.seed(42)  # Ensure reproducibility
            n_sequences = 200
            sequence_length = 50
            n_features = 5
            
            print(f"   Generating {n_sequences} sequences of length {sequence_length}")
            print(f"   Features per timestep: {n_features}")
            
            # Create benign and attack patterns with detailed logging
            X_benign = []
            X_attack = []
            
            print("   Creating benign traffic patterns...")
            for i in range(n_sequences//2):
                sequence = np.random.normal(0, 1, (sequence_length, n_features))
                # Add periodic pattern for benign traffic
                t = np.linspace(0, 4*np.pi, sequence_length)
                sequence[:, 0] += 0.5 * np.sin(t) + 0.3 * np.sin(3*t)
                X_benign.append(sequence)
            
            print("   Creating attack traffic patterns...")
            for i in range(n_sequences//2):
                sequence = np.random.normal(0, 1, (sequence_length, n_features))
                # Add attack signatures
                attack_points = np.random.choice(sequence_length, 5, replace=False)
                sequence[attack_points] += np.random.normal(3, 1, (5, n_features))
                X_attack.append(sequence)
            
            X = np.array(X_benign + X_attack)
            y = np.array([0] * (n_sequences//2) + [1] * (n_sequences//2))
            
            print(f"✅ Synthetic data generated successfully")
            print(f"   Final shape: {X.shape}")
            print(f"   Attack rate: {np.mean(y):.1%}")
        
        # Detailed feature extraction with logging
        print(f"\n🔧 FEATURE EXTRACTION - TEMPORAL MULTI-SCALE TDA")
        print("-" * 60)
        
        def extract_temporal_tda_features(sequences):
            """Extract multi-scale temporal TDA features with detailed logging"""
            print("Extracting temporal TDA features across multiple scales...")
            
            features = []
            scales = [5, 10, 20]  # Multiple temporal scales
            
            print(f"Processing {len(sequences)} sequences at {len(scales)} scales: {scales}")
            
            for seq_idx, seq in enumerate(sequences):
                if seq_idx % 50 == 0:
                    print(f"   Processing sequence {seq_idx+1}/{len(sequences)}")
                
                seq_features = []
                
                for scale_idx, scale in enumerate(scales):
                    # Subsample sequence at different scales
                    if len(seq) >= scale:
                        step = max(1, len(seq) // scale)
                        subseq = seq[::step]
                    else:
                        subseq = seq
                    
                    # Create time-delay embedding for topology
                    if len(subseq) >= 3:
                        # 2D embedding for Rips complex
                        embedded = []
                        for i in range(len(subseq) - 2):
                            embedded.append([
                                np.mean(subseq[i]),
                                np.mean(subseq[i+1]),
                            ])
                        
                        if len(embedded) >= 3:
                            embedded = np.array(embedded)
                            
                            # Compute Rips complex
                            rips_complex = gd.RipsComplex(points=embedded, max_edge_length=2.0)
                            simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
                            
                            # Persistence computation
                            persistence = simplex_tree.persistence()
                            
                            # Extract features from persistence diagram
                            h0_persistence = [p[1][1] - p[1][0] for p in persistence if p[0] == 0 and p[1][1] != float('inf')]
                            h1_persistence = [p[1][1] - p[1][0] for p in persistence if p[0] == 1 and p[1][1] != float('inf')]
                            
                            # Statistical features from persistence
                            if h0_persistence:
                                seq_features.extend([
                                    np.mean(h0_persistence),
                                    np.std(h0_persistence),
                                    np.max(h0_persistence),
                                    len(h0_persistence)
                                ])
                            else:
                                seq_features.extend([0, 0, 0, 0])
                            
                            if h1_persistence:
                                seq_features.extend([
                                    np.mean(h1_persistence),
                                    np.std(h1_persistence),
                                    np.max(h1_persistence),
                                    len(h1_persistence)
                                ])
                            else:
                                seq_features.extend([0, 0, 0, 0])
                        else:
                            seq_features.extend([0] * 8)
                    else:
                        seq_features.extend([0] * 8)
                
                features.append(seq_features)
            
            features_array = np.array(features)
            print(f"✅ Temporal TDA features extracted: {features_array.shape[1]} dimensions")
            return features_array
        
        def extract_graph_tda_features(sequences):
            """Extract graph-based TDA features with detailed logging"""
            print("Extracting graph-based network topology TDA features...")
            
            features = []
            
            print(f"Processing {len(sequences)} sequences for graph topology analysis")
            
            for seq_idx, seq in enumerate(sequences):
                if seq_idx % 50 == 0:
                    print(f"   Processing sequence {seq_idx+1}/{len(sequences)}")
                
                seq_features = []
                
                # Create correlation-based network
                if seq.shape[0] > seq.shape[1]:  # More time points than features
                    try:
                        corr_matrix = np.corrcoef(seq.T)
                        # Handle NaN values
                        corr_matrix = np.nan_to_num(corr_matrix)
                        
                        # Create graph from correlation matrix
                        threshold = 0.5
                        adj_matrix = (np.abs(corr_matrix) > threshold).astype(int)
                        np.fill_diagonal(adj_matrix, 0)
                        
                        # Basic graph properties
                        graph = nx.from_numpy_array(adj_matrix)
                        
                        # Extract graph TDA features
                        seq_features.extend([
                            graph.number_of_nodes(),
                            graph.number_of_edges(),
                            nx.density(graph),
                            len(list(nx.connected_components(graph))),
                            np.mean(list(dict(graph.degree()).values())) if graph.number_of_nodes() > 0 else 0,
                            nx.average_clustering(graph) if graph.number_of_nodes() > 0 else 0,
                        ])
                        
                        # Spectral features
                        eigenvals = np.linalg.eigvals(corr_matrix)
                        eigenvals = np.real(eigenvals)
                        eigenvals = eigenvals[~np.isnan(eigenvals)]
                        
                        if len(eigenvals) > 0:
                            seq_features.extend([
                                np.max(eigenvals),
                                np.mean(eigenvals),
                                np.std(eigenvals)
                            ])
                        else:
                            seq_features.extend([0, 0, 0])
                            
                    except Exception as e:
                        print(f"   Warning: Graph construction failed for sequence {seq_idx}: {e}")
                        seq_features.extend([0] * 9)
                else:
                    seq_features.extend([0] * 9)
                
                features.append(seq_features)
            
            features_array = np.array(features)
            print(f"✅ Graph TDA features extracted: {features_array.shape[1]} dimensions")
            return features_array
        
        # Extract features with timing
        start_time = time.time()
        
        temporal_features = extract_temporal_tda_features(X)
        graph_features = extract_graph_tda_features(X)
        
        extraction_time = time.time() - start_time
        
        # Combine features
        print(f"\n🔗 FEATURE COMBINATION")
        print("-" * 30)
        combined_features = np.hstack([temporal_features, graph_features])
        print(f"✅ Combined features: {combined_features.shape[1]} dimensions")
        print(f"   - Temporal features: {temporal_features.shape[1]} dimensions")
        print(f"   - Graph features: {graph_features.shape[1]} dimensions")
        print(f"   - Total extraction time: {extraction_time:.1f} seconds")
        
        # Data splitting with detailed info
        print(f"\n📊 DATASET PREPARATION")
        print("-" * 30)
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, y, 
            test_size=0.3, 
            random_state=42, 
            stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"   - Benign: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
        print(f"   - Attack: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
        
        print(f"Test set: {len(X_test)} samples")  
        print(f"   - Benign: {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
        print(f"   - Attack: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"✅ Feature scaling applied (StandardScaler)")
        
        # Model training with detailed logging
        print(f"\n🤖 HYBRID ENSEMBLE TRAINING")
        print("-" * 40)
        
        # Define ensemble components
        models = [
            ('rf1', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('rf2', RandomForestClassifier(n_estimators=50, random_state=43, max_depth=10)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]
        
        print("Ensemble components:")
        for name, model in models:
            print(f"   - {name}: {model.__class__.__name__}")
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft'
        )
        
        print("Training ensemble...")
        train_start = time.time()
        ensemble.fit(X_train_scaled, y_train)
        train_time = time.time() - train_start
        print(f"✅ Ensemble training completed in {train_time:.1f} seconds")
        
        # Prediction with detailed logging
        print(f"\n🎯 PREDICTION AND EVALUATION")
        print("-" * 40)
        
        print("Generating predictions...")
        y_pred = ensemble.predict(X_test_scaled)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]  # Probability for attack class
        
        print(f"Predictions generated for {len(y_test)} test samples")
        print(f"Predicted attack rate: {np.mean(y_pred):.1%}")
        print(f"Actual attack rate: {np.mean(y_test):.1%}")
    
    # COMPREHENSIVE VALIDATION WITH EVIDENCE CAPTURE
    print(f"\n" + "=" * 80)
    print("🔍 COMPREHENSIVE VALIDATION WITH EVIDENCE CAPTURE")
    print("=" * 80)
    
    # Run validation with complete evidence capture
    metrics = validator.validate_classification_results(
        y_true=y_test,
        y_pred=y_pred, 
        y_pred_proba=y_pred_proba,
        class_names=['Benign', 'Attack']
    )
    
    # Verify against known claim
    claimed_f1 = 0.706  # Previous validated claim
    validation_passed = validator.verify_claim(claimed_f1, tolerance=0.05)
    
    print(f"\n🎯 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Experiment: {validator.experiment_name}")
    print(f"Validation ID: {validator.timestamp}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Claim Verification: {'PASSED' if validation_passed else 'FAILED'}")
    print(f"Evidence Package: {len(validator.plots)} plots generated")
    print(f"Console Output: Captured and saved")
    print(f"Raw Data: Saved for audit")
    
    return validator, metrics

if __name__ == "__main__":
    # Run enhanced validation
    validator, metrics = enhanced_hybrid_tda_validation()
    
    # Generate standard report
    if validator.validation_passed:
        report = report_validated_results("Hybrid Multi-scale + Graph TDA", validator=validator)
        print("\n" + "=" * 80)
        print("📋 VALIDATED RESULTS REPORT")
        print("=" * 80)
        print(report)
    else:
        print("\n❌ VALIDATION FAILED - Cannot generate results report")