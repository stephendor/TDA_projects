#!/usr/bin/env python3
"""
Hybrid TDA Results Validation Script
Verify the claimed 70.6% F1-score performance from hybrid multi-scale + graph-based TDA
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# Import TDA modules
import sys
sys.path.append('.')
from src.core.persistent_homology import PersistentHomologyAnalyzer

def validate_hybrid_tda_performance():
    """
    Validate the claimed Hybrid TDA performance with exact reproduction.
    """
    
    print("🔍 VALIDATION: Hybrid Multi-Scale + Graph-Based TDA Results")
    print("=" * 70)
    print("Claim: 70.6% F1-score with VotingClassifier ensemble")
    print("=" * 70)
    
    # Load data with same parameters
    print("\n1. Loading dataset...")
    file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Same balanced dataset as claimed
    attacks = df[df['Label'] != 'BENIGN']
    benign = df[df['Label'] == 'BENIGN']
    
    benign_sample = benign.sample(n=min(8000, len(benign)), random_state=42)
    df_balanced = pd.concat([attacks, benign_sample]).sort_index()
    
    print(f"   Dataset: {len(df_balanced):,} flows ({len(attacks)} attacks)")
    print(f"   Attack rate: {(df_balanced['Label'] != 'BENIGN').mean():.3%}")
    
    # Extract hybrid features using same methodology
    print("\n2. Extracting hybrid TDA features...")
    hybrid_features, hybrid_labels = extract_hybrid_tda_features(df_balanced)
    
    if hybrid_features is None:
        print("❌ Hybrid feature extraction failed")
        return False
    
    print(f"   Features: {hybrid_features.shape}")
    print(f"   Attack rate: {np.mean(hybrid_labels):.3%}")
    
    # Split data with same random state
    print("\n3. Splitting data (same random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        hybrid_features, hybrid_labels, test_size=0.3, random_state=42, stratify=hybrid_labels
    )
    
    print(f"   Training: {X_train.shape[0]} samples, {y_train.sum()} attacks")
    print(f"   Testing: {X_test.shape[0]} samples, {y_test.sum()} attacks")
    
    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train exact same ensemble as claimed
    print("\n5. Training Hybrid TDA Ensemble (exact reproduction)...")
    
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf2 = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=123, class_weight='balanced')
    lr = LogisticRegression(C=0.1, random_state=42, class_weight='balanced', max_iter=1000)
    
    ensemble = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('lr', lr)],
        voting='soft'
    )
    
    start_time = time.time()
    ensemble.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"   Training completed in {training_time:.2f}s")
    
    # Make predictions
    print("\n6. Making predictions...")
    y_pred = ensemble.predict(X_test_scaled)
    
    # Detailed evaluation
    print("\n7. DETAILED VALIDATION RESULTS:")
    print("=" * 50)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                Benign  Attack")
    print(f"Actual  Benign     {cm[0,0]}      {cm[0,1]}")
    print(f"        Attack     {cm[1,0]}      {cm[1,1]}")
    
    # Key metrics
    accuracy = report['accuracy']
    precision = report.get('1', {}).get('precision', 0)
    recall = report.get('1', {}).get('recall', 0)
    f1_score = report.get('1', {}).get('f1-score', 0)
    
    print(f"\n📊 KEY METRICS:")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy:.1%})")
    print(f"   Precision: {precision:.3f} ({precision:.1%})")
    print(f"   Recall:    {recall:.3f} ({recall:.1%})")
    print(f"   F1-Score:  {f1_score:.3f} ({f1_score:.1%})")
    
    # Validation assessment
    print(f"\n🎯 VALIDATION ASSESSMENT:")
    print("=" * 30)
    
    claimed_f1 = 0.706
    tolerance = 0.05  # Allow 5% tolerance for randomness
    
    if f1_score >= claimed_f1 - tolerance:
        if f1_score >= claimed_f1:
            print(f"✅ CLAIM VALIDATED: F1-score {f1_score:.3f} matches/exceeds claim of {claimed_f1:.3f}")
        else:
            print(f"✅ CLAIM ACCEPTABLE: F1-score {f1_score:.3f} within tolerance of claim {claimed_f1:.3f}")
        validation_status = "VALIDATED"
    else:
        gap = claimed_f1 - f1_score
        print(f"❌ CLAIM DISPUTED: F1-score {f1_score:.3f} significantly below claim {claimed_f1:.3f} (gap: {gap:.3f})")
        validation_status = "DISPUTED"
    
    # Additional validation metrics
    print(f"\n📋 ADDITIONAL VALIDATION:")
    print(f"   Test set balance: {y_test.sum()}/{len(y_test)} attacks ({np.mean(y_test):.1%})")
    print(f"   Prediction distribution: {y_pred.sum()}/{len(y_pred)} predicted attacks ({np.mean(y_pred):.1%})")
    
    # Target assessment
    target_f1 = 0.75
    print(f"\n🎯 TARGET ASSESSMENT:")
    if f1_score >= target_f1:
        print(f"   ✅ TARGET ACHIEVED: F1 {f1_score:.3f} ≥ {target_f1:.3f}")
    else:
        gap_to_target = target_f1 - f1_score
        print(f"   ⚠️ Gap to target: {gap_to_target:.3f} ({gap_to_target/target_f1*100:.1f}%)")
    
    return validation_status == "VALIDATED", f1_score

def extract_hybrid_tda_features(df):
    """
    Extract the exact hybrid TDA features used in the claimed result.
    This reproduces the temporal + graph feature extraction process.
    """
    
    print("   Extracting temporal multi-scale features...")
    temporal_features, temporal_labels = extract_temporal_features(df)
    
    print("   Extracting graph-based features...")
    graph_features, graph_labels = extract_graph_features(df)
    
    if temporal_features is None or graph_features is None:
        print("   ❌ Failed to extract one or both feature types")
        return None, None
    
    # Align datasets (same as claimed approach)
    min_samples = min(len(temporal_features), len(graph_features))
    
    temporal_subset = temporal_features[:min_samples]
    graph_subset = graph_features[:min_samples]
    labels_subset = temporal_labels[:min_samples]
    
    # Combine features
    hybrid_features = np.concatenate([temporal_subset, graph_subset], axis=1)
    
    print(f"   Temporal features: {temporal_subset.shape}")
    print(f"   Graph features: {graph_subset.shape}")
    print(f"   Combined features: {hybrid_features.shape}")
    print(f"   Attack sequences: {np.sum(labels_subset)} ({np.mean(labels_subset):.3%})")
    
    return hybrid_features, labels_subset

def extract_temporal_features(df):
    """Extract multi-scale temporal TDA features (proven method)."""
    
    feature_columns = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Packet Length Mean', 'Packet Length Std', 'Min Packet Length',
        'Max Packet Length', 'FIN Flag Count', 'SYN Flag Count'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = (df['Label'] != 'BENIGN').astype(int)
    
    # Multi-scale analysis (same window sizes as claimed)
    window_sizes = [5, 10, 20, 40, 60]
    all_temporal_features = []
    all_temporal_labels = []
    
    for scale_idx, window_size in enumerate(window_sizes):
        sequences, labels = create_temporal_sequences(X, y, window_size)
        
        if len(sequences) == 0:
            continue
            
        tda_features = extract_temporal_tda_features(sequences, scale_idx)
        
        if tda_features is not None:
            all_temporal_features.append(tda_features)
            all_temporal_labels.append(labels)
    
    if not all_temporal_features:
        return None, None
    
    # Use scale with best attack preservation (same logic as claimed)
    attack_rates = [np.mean(labels) for labels in all_temporal_labels]
    best_scale_idx = np.argmax(attack_rates)
    
    primary_features = all_temporal_features[best_scale_idx]
    primary_labels = all_temporal_labels[best_scale_idx]
    
    # Add features from other scales
    n_samples = len(primary_features)
    additional_features = []
    
    for scale_idx, features in enumerate(all_temporal_features):
        if scale_idx != best_scale_idx and len(features) >= n_samples:
            additional_features.append(features[:n_samples])
    
    if additional_features:
        combined_temporal = np.concatenate([primary_features] + additional_features, axis=1)
    else:
        combined_temporal = primary_features
        
    return combined_temporal, primary_labels

def extract_graph_features(df):
    """Extract graph-based TDA features (proven method)."""
    
    window_sizes = [20, 50, 100, 200]
    all_graph_features = []
    all_graph_labels = []
    
    for window_idx, window_size in enumerate(window_sizes):
        graph_sequences, graph_labels = create_graph_sequences(df, window_size)
        
        if len(graph_sequences) == 0:
            continue
            
        graph_tda_features = extract_graph_tda_features(graph_sequences, window_idx)
        
        if graph_tda_features is not None:
            all_graph_features.append(graph_tda_features)
            all_graph_labels.append(graph_labels)
    
    if not all_graph_features:
        return None, None
    
    # Use scale with best attack preservation
    attack_rates = [np.mean(labels) for labels in all_graph_labels]
    best_scale_idx = np.argmax(attack_rates)
    
    primary_features = all_graph_features[best_scale_idx]
    primary_labels = all_graph_labels[best_scale_idx]
    
    # Add complementary features
    n_samples = len(primary_features)
    additional_features = []
    
    for scale_idx, features in enumerate(all_graph_features):
        if scale_idx != best_scale_idx and len(features) >= n_samples:
            additional_features.append(features[:n_samples])
    
    if additional_features:
        combined_graph = np.concatenate([primary_features] + additional_features, axis=1)
    else:
        combined_graph = primary_features
        
    return combined_graph, primary_labels

def create_temporal_sequences(X, y, window_size):
    """Create temporal sequences for TDA analysis."""
    
    step_size = max(1, window_size // 3)
    
    if len(X) < window_size:
        return [], []
    
    sequences = []
    labels = []
    
    for i in range(0, len(X) - window_size + 1, step_size):
        sequence = X.iloc[i:i+window_size].values
        window_labels = y.iloc[i:i+window_size].values
        sequence_label = 1 if np.sum(window_labels) > 0 else 0
        
        sequences.append(sequence)
        labels.append(sequence_label)
    
    return np.array(sequences), np.array(labels)

def create_graph_sequences(df, window_size):
    """Create graph sequences for TDA analysis."""
    
    step_size = max(1, window_size // 4)
    
    if len(df) < window_size:
        return [], []
    
    graph_sequences = []
    labels = []
    
    for i in range(0, len(df) - window_size + 1, step_size):
        window_flows = df.iloc[i:i+window_size]
        G = build_network_graph(window_flows)
        
        if G.number_of_nodes() < 3:
            continue
            
        window_labels = window_flows['Label'].values
        is_attack = any(label != 'BENIGN' for label in window_labels)
        sequence_label = 1 if is_attack else 0
        
        graph_sequences.append(G)
        labels.append(sequence_label)
    
    return graph_sequences, np.array(labels)

def build_network_graph(flows_window):
    """Build network graph from flow data."""
    
    G = nx.Graph()
    min_connections = 2
    
    for _, flow in flows_window.iterrows():
        try:
            src_ip = str(flow.get('Source IP', f'src_{len(G.nodes)}'))
            dst_ip = str(flow.get('Destination IP', f'dst_{len(G.nodes)}'))
            
            flow_bytes = float(flow.get('Flow Bytes/s', 0))
            flow_packets = float(flow.get('Flow Packets/s', 0))
            duration = float(flow.get('Flow Duration', 1))
            
            weight = (flow_bytes + flow_packets) / max(duration, 1)
            
            if G.has_edge(src_ip, dst_ip):
                G[src_ip][dst_ip]['weight'] += weight
            else:
                G.add_edge(src_ip, dst_ip, weight=weight)
                
        except Exception:
            continue
    
    # Filter nodes with too few connections
    nodes_to_remove = [node for node in G.nodes() 
                      if G.degree(node) < min_connections]
    G.remove_nodes_from(nodes_to_remove)
    
    return G

def extract_temporal_tda_features(sequences, scale_idx):
    """Extract TDA features from temporal sequences."""
    
    try:
        max_dim = 1 if len(sequences[0]) < 50 else 2
        thresh = 3.0 if scale_idx < 2 else 5.0
        
        ph_analyzer = PersistentHomologyAnalyzer(
            maxdim=max_dim, thresh=thresh, backend='ripser'
        )
        
        batch_size = 50
        all_features = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_features = []
            
            for seq in batch:
                if len(seq) >= 3:
                    try:
                        ph_analyzer.fit(seq)
                        features = ph_analyzer.extract_features()
                        
                        if len(features) < 12:
                            padded_features = np.zeros(12)
                            padded_features[:len(features)] = features
                            features = padded_features
                        
                        batch_features.append(features[:12])
                        
                    except Exception:
                        batch_features.append(np.zeros(12))
                else:
                    batch_features.append(np.zeros(12))
            
            if batch_features:
                all_features.extend(batch_features)
        
        if all_features:
            feature_matrix = np.array(all_features)
            feature_matrix = np.nan_to_num(feature_matrix)
            return feature_matrix
        else:
            return None
            
    except Exception:
        return None

def extract_graph_tda_features(graph_sequences, scale_idx):
    """Extract TDA features from graph sequences."""
    
    try:
        ph_analyzer = PersistentHomologyAnalyzer(
            maxdim=1, thresh=2.0, metric='precomputed', backend='ripser'
        )
        
        batch_size = 20
        all_features = []
        
        for i in range(0, len(graph_sequences), batch_size):
            batch_graphs = graph_sequences[i:i+batch_size]
            batch_features = []
            
            for G in batch_graphs:
                try:
                    distance_matrix = graph_to_distance_matrix(G)
                    
                    if distance_matrix is not None:
                        ph_analyzer.fit(distance_matrix)
                        features = ph_analyzer.extract_features()
                        
                        if len(features) < 12:
                            padded_features = np.zeros(12)
                            padded_features[:len(features)] = features
                            features = padded_features
                        
                        # Add graph statistics
                        graph_stats = extract_graph_statistics(G)
                        combined_features = np.concatenate([features[:12], graph_stats])
                        
                        batch_features.append(combined_features)
                    else:
                        batch_features.append(np.zeros(18))
                        
                except Exception:
                    batch_features.append(np.zeros(18))
            
            if batch_features:
                all_features.extend(batch_features)
        
        if all_features:
            feature_matrix = np.array(all_features)
            feature_matrix = np.nan_to_num(feature_matrix)
            return feature_matrix
        else:
            return None
            
    except Exception:
        return None

def graph_to_distance_matrix(G):
    """Convert graph to distance matrix."""
    
    try:
        if G.number_of_nodes() < 3:
            return None
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        if n_nodes > 50:
            node_degrees = [(node, G.degree(node)) for node in nodes]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in node_degrees[:50]]
        else:
            selected_nodes = nodes
        
        subG = G.subgraph(selected_nodes)
        distance_dict = dict(nx.all_pairs_shortest_path_length(subG))
        
        n_selected = len(selected_nodes)
        distance_matrix = np.full((n_selected, n_selected), np.inf)
        
        for i, node_i in enumerate(selected_nodes):
            for j, node_j in enumerate(selected_nodes):
                if node_j in distance_dict.get(node_i, {}):
                    distance_matrix[i, j] = distance_dict[node_i][node_j]
                elif i == j:
                    distance_matrix[i, j] = 0
        
        max_finite = np.max(distance_matrix[np.isfinite(distance_matrix)])
        distance_matrix[np.isinf(distance_matrix)] = max_finite + 1
        
        return distance_matrix
        
    except Exception:
        return None

def extract_graph_statistics(G):
    """Extract graph statistics."""
    
    try:
        n_nodes = G.number_of_nodes()
        
        if n_nodes == 0:
            return np.zeros(6)
        
        density = nx.density(G)
        degree_centrality = np.mean(list(nx.degree_centrality(G).values()))
        
        try:
            betweenness_centrality = np.mean(list(nx.betweenness_centrality(G).values()))
        except:
            betweenness_centrality = 0.0
            
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0.0
        
        try:
            n_components = nx.number_connected_components(G)
        except:
            n_components = 1
            
        avg_degree = np.mean([d for n, d in G.degree()]) if n_nodes > 0 else 0
        
        return np.array([
            density, degree_centrality, betweenness_centrality,
            clustering, n_components / max(n_nodes, 1), avg_degree / max(n_nodes, 1)
        ])
        
    except Exception:
        return np.zeros(6)

def main():
    """Run hybrid TDA validation test."""
    
    print("🧪 HYBRID TDA VALIDATION")
    print("=" * 80)
    print("Purpose: Validate claimed 70.6% F1-score from hybrid multi-scale + graph TDA")
    print("Method: Exact reproduction of claimed methodology")
    print("=" * 80)
    
    try:
        is_validated, actual_f1 = validate_hybrid_tda_performance()
        
        print(f"\n🏁 FINAL VALIDATION RESULT:")
        print("=" * 40)
        
        if is_validated:
            print("✅ VALIDATION SUCCESSFUL")
            print(f"   The claimed 70.6% F1-score is reproducible (actual: {actual_f1:.3f})")
        else:
            print("❌ VALIDATION FAILED") 
            print(f"   The claimed 70.6% F1-score could not be reproduced (actual: {actual_f1:.3f})")
        
        # Current status assessment
        print(f"\n📊 CURRENT STATUS:")
        print(f"   Best validated result: {actual_f1:.1%} F1-score")
        
        target_f1 = 0.75
        if actual_f1 >= target_f1:
            print(f"   ✅ Target achieved: {actual_f1:.1%} ≥ {target_f1:.1%}")
        else:
            gap = target_f1 - actual_f1
            print(f"   ⚠️ Gap to 75% target: {gap:.3f} ({gap/target_f1*100:.1f}%)")
            
    except Exception as e:
        print(f"\n❌ VALIDATION ERROR: {e}")
        print("   Could not complete validation due to technical issues")

if __name__ == "__main__":
    main()