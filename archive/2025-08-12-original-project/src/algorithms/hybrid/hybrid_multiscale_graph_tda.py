#!/usr/bin/env python3
"""
Hybrid Multi-Scale + Graph-Based TDA Implementation
Phase 2A+ Advanced Enhancement

Combines the best of both approaches:
- Multi-Scale Temporal TDA (65.4% F1-score)  
- Graph-Based Network TDA (70.8% F1-score)
Target: Push beyond 75% F1-score by intelligent ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Import TDA modules - Updated for new structure
from ...core.persistent_homology import PersistentHomologyAnalyzer

class HybridTDAAnalyzer:
    """
    Advanced hybrid analyzer combining multi-scale temporal TDA
    with graph-based network topology TDA for maximum performance.
    """
    
    def __init__(self):
        # Multi-scale temporal parameters (proven to work)
        self.temporal_window_sizes = [5, 10, 20, 40, 60]
        
        # Graph-based parameters (proven to work)
        self.graph_window_sizes = [20, 50, 100, 200]
        self.min_connections = 2
        
        # TDA analyzers
        self.temporal_tda_analyzers = {}
        self.graph_tda_analyzers = {}
        
        print(f"🔬 Hybrid Multi-Scale + Graph TDA Analyzer initialized")
        print(f"   Temporal scales: {self.temporal_window_sizes}")
        print(f"   Graph scales: {self.graph_window_sizes}")

    def extract_hybrid_features(self, df):
        """Extract both temporal and graph-based TDA features."""
        
        print(f"\n🔄 EXTRACTING HYBRID TDA FEATURES")
        print("=" * 60)
        
        # Extract temporal multi-scale features (from previous breakthrough)
        temporal_features, temporal_labels = self.extract_temporal_features(df)
        
        # Extract graph-based features (from current implementation)
        graph_features, graph_labels = self.extract_graph_features(df)
        
        if temporal_features is None or graph_features is None:
            print("❌ Failed to extract one or both feature types")
            return None, None
        
        # Align the datasets (use intersection of samples)
        min_samples = min(len(temporal_features), len(graph_features))
        
        # Take samples from both feature sets
        temporal_subset = temporal_features[:min_samples]
        graph_subset = graph_features[:min_samples]
        
        # Use labels from temporal features (more granular)
        labels_subset = temporal_labels[:min_samples]
        
        # Combine features
        hybrid_features = np.concatenate([temporal_subset, graph_subset], axis=1)
        
        print(f"\n📊 HYBRID FEATURE SUMMARY:")
        print(f"   Temporal features: {temporal_subset.shape}")
        print(f"   Graph features: {graph_subset.shape}")  
        print(f"   Combined features: {hybrid_features.shape}")
        print(f"   Total attack sequences: {np.sum(labels_subset)}")
        print(f"   Total benign sequences: {np.sum(labels_subset == 0)}")
        print(f"   Attack rate: {np.mean(labels_subset):.3%}")
        
        return hybrid_features, labels_subset

    def extract_temporal_features(self, df):
        """Extract multi-scale temporal TDA features (from previous success)."""
        
        print(f"   🕐 Extracting temporal multi-scale features...")
        
        # Prepare data for temporal analysis
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
        
        # Extract multi-scale temporal features
        all_temporal_features = []
        all_temporal_labels = []
        
        for scale_idx, window_size in enumerate(self.temporal_window_sizes):
            sequences, labels = self.create_temporal_sequences(X, y, window_size)
            
            if len(sequences) == 0:
                continue
                
            tda_features = self.extract_temporal_tda_features(sequences, scale_idx)
            
            if tda_features is not None:
                all_temporal_features.append(tda_features)
                all_temporal_labels.append(labels)
        
        if not all_temporal_features:
            return None, None
        
        # Use best scale as primary (largest windows preserve more attacks)
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
            
        print(f"      ✅ Temporal features: {combined_temporal.shape}")
        
        return combined_temporal, primary_labels

    def extract_graph_features(self, df):
        """Extract graph-based TDA features (from current implementation)."""
        
        print(f"   🕸️ Extracting graph-based network features...")
        
        all_graph_features = []
        all_graph_labels = []
        
        for window_idx, window_size in enumerate(self.graph_window_sizes):
            graph_sequences, graph_labels = self.create_graph_sequences(df, window_size)
            
            if len(graph_sequences) == 0:
                continue
                
            graph_tda_features = self.extract_graph_tda_features(graph_sequences, window_idx)
            
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
            
        print(f"      ✅ Graph features: {combined_graph.shape}")
        
        return combined_graph, primary_labels

    def create_temporal_sequences(self, X, y, window_size, step_size=None):
        """Create temporal sequences for TDA analysis."""
        
        if step_size is None:
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

    def create_graph_sequences(self, df, window_size, step_size=None):
        """Create graph sequences for TDA analysis."""
        
        if step_size is None:
            step_size = max(1, window_size // 4)
        
        if len(df) < window_size:
            return [], []
        
        graph_sequences = []
        labels = []
        
        for i in range(0, len(df) - window_size + 1, step_size):
            window_flows = df.iloc[i:i+window_size]
            G = self.build_network_graph(window_flows)
            
            if G.number_of_nodes() < 3:
                continue
                
            window_labels = window_flows['Label'].values
            is_attack = any(label != 'BENIGN' for label in window_labels)
            sequence_label = 1 if is_attack else 0
            
            graph_sequences.append(G)
            labels.append(sequence_label)
        
        return graph_sequences, np.array(labels)

    def build_network_graph(self, flows_window):
        """Build network graph from flow data."""
        
        G = nx.Graph()
        
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
                          if G.degree(node) < self.min_connections]
        G.remove_nodes_from(nodes_to_remove)
        
        return G

    def extract_temporal_tda_features(self, sequences, scale_idx):
        """Extract TDA features from temporal sequences."""
        
        try:
            if scale_idx not in self.temporal_tda_analyzers:
                max_dim = 1 if len(sequences[0]) < 50 else 2
                thresh = 3.0 if scale_idx < 2 else 5.0
                
                self.temporal_tda_analyzers[scale_idx] = PersistentHomologyAnalyzer(
                    maxdim=max_dim, thresh=thresh, backend='ripser'
                )
            
            ph_analyzer = self.temporal_tda_analyzers[scale_idx]
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

    def extract_graph_tda_features(self, graph_sequences, scale_idx):
        """Extract TDA features from graph sequences."""
        
        try:
            if scale_idx not in self.graph_tda_analyzers:
                self.graph_tda_analyzers[scale_idx] = PersistentHomologyAnalyzer(
                    maxdim=1, thresh=2.0, metric='precomputed', backend='ripser'
                )
            
            ph_analyzer = self.graph_tda_analyzers[scale_idx]
            batch_size = 20
            all_features = []
            
            for i in range(0, len(graph_sequences), batch_size):
                batch_graphs = graph_sequences[i:i+batch_size]
                batch_features = []
                
                for G in batch_graphs:
                    try:
                        distance_matrix = self.graph_to_distance_matrix(G)
                        
                        if distance_matrix is not None:
                            ph_analyzer.fit(distance_matrix)
                            features = ph_analyzer.extract_features()
                            
                            if len(features) < 12:
                                padded_features = np.zeros(12)
                                padded_features[:len(features)] = features
                                features = padded_features
                            
                            # Add graph statistics
                            graph_stats = self.extract_graph_statistics(G)
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

    def graph_to_distance_matrix(self, G):
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

    def extract_graph_statistics(self, G):
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

def load_and_prepare_data():
    """Load and prepare dataset for hybrid analysis."""
    
    print("🔍 LOADING DATA FOR HYBRID TDA ANALYSIS")
    print("=" * 50)
    
    file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Get balanced dataset
    attacks = df[df['Label'] != 'BENIGN']
    benign = df[df['Label'] == 'BENIGN']
    
    print(f"   Total attacks: {len(attacks)}")
    print(f"   Total benign: {len(benign):,}")
    
    # Use moderate sample size for hybrid analysis
    benign_sample = benign.sample(n=min(8000, len(benign)), random_state=42)
    df_balanced = pd.concat([attacks, benign_sample]).sort_index()
    
    print(f"   Using {len(attacks)} attacks + {len(benign_sample):,} benign = {len(df_balanced):,} total")
    print(f"   Attack rate: {(df_balanced['Label'] != 'BENIGN').mean():.3%}")
    
    return df_balanced

def evaluate_hybrid_tda():
    """Main evaluation function for hybrid TDA."""
    
    # Load data
    df = load_and_prepare_data()
    
    # Initialize hybrid analyzer
    analyzer = HybridTDAAnalyzer()
    
    # Extract hybrid features
    start_time = time.time()
    hybrid_features, hybrid_labels = analyzer.extract_hybrid_features(df)
    extraction_time = time.time() - start_time
    
    if hybrid_features is None:
        print("❌ Hybrid feature extraction failed")
        return None
    
    print(f"\n⏱️ Hybrid feature extraction completed in {extraction_time:.1f}s")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        hybrid_features, hybrid_labels, test_size=0.3, random_state=42, stratify=hybrid_labels
    )
    
    print(f"\n📊 EVALUATION SETUP:")
    print(f"   Training: {X_train.shape}, attacks: {y_train.sum()}")
    print(f"   Testing: {X_test.shape}, attacks: {y_test.sum()}")
    
    # Train advanced ensemble on hybrid features
    print(f"\n🚀 TRAINING HYBRID TDA ENSEMBLE")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create ensemble of classifiers optimized for different aspects
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf2 = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=123, class_weight='balanced')
    lr = LogisticRegression(C=0.1, random_state=42, class_weight='balanced', max_iter=1000)
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('lr', lr)],
        voting='soft'  # Use probability averaging
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = ensemble.predict(X_test_scaled)
    
    # Evaluation
    print(f"\n📈 HYBRID TDA ENSEMBLE RESULTS:")
    print("=" * 50)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = report['accuracy']
    precision = report.get('1', {}).get('precision', 0)
    recall = report.get('1', {}).get('recall', 0)
    f1_score = report.get('1', {}).get('f1-score', 0)
    
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1_score:.3f}")
    
    if cm.shape == (2, 2):
        print(f"\n   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
    else:
        print(f"\n   Confusion Matrix: {cm}")
    
    # Compare with previous approaches
    print(f"\n📊 PERFORMANCE EVOLUTION:")
    print("=" * 50)
    
    single_scale_f1 = 0.182
    multi_scale_f1 = 0.654
    graph_based_f1 = 0.708
    
    print(f"   Single-Scale TDA: F1 = {single_scale_f1:.3f}")
    print(f"   Multi-Scale TDA: F1 = {multi_scale_f1:.3f} (+{(multi_scale_f1-single_scale_f1)*100:.1f}%)")
    print(f"   Graph-Based TDA: F1 = {graph_based_f1:.3f} (+{(graph_based_f1-multi_scale_f1)*100:.1f}%)")
    print(f"   🚀 Hybrid TDA: F1 = {f1_score:.3f} (+{(f1_score-graph_based_f1)*100:.1f}%)")
    
    total_improvement = f1_score - single_scale_f1
    print(f"\n   📈 Total improvement: +{total_improvement:.3f} ({(total_improvement/single_scale_f1)*100:.1f}%)")
    
    # Target assessment
    if f1_score > 0.75:
        print(f"   ✅ SUCCESS: Exceeded Phase 2A target (F1 > 75%)!")
        status = "SUCCESS"
    elif f1_score > 0.70:
        print(f"   ✅ STRONG PROGRESS: Approaching target (F1 = {f1_score:.1%})")
        status = "PROGRESS"
    else:
        print(f"   ⚠️ MODERATE PROGRESS: Further optimization needed")
        status = "MODERATE"
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_improvement': total_improvement,
        'extraction_time': extraction_time,
        'status': status
    }

def main():
    """Main execution function."""
    
    print("🔬 HYBRID MULTI-SCALE + GRAPH-BASED TDA")
    print("=" * 60)
    print("Phase 2A+ Advanced Enhancement Strategy")
    print("Combines: Multi-Scale (65.4%) + Graph-Based (70.8%)")
    print("Target: F1-Score >75%")
    print("=" * 60)
    
    # Run evaluation
    results = evaluate_hybrid_tda()
    
    if results:
        print(f"\n🎯 HYBRID TDA EVALUATION COMPLETE")
        print("=" * 60)
        
        if results['status'] == 'SUCCESS':
            print(f"✅ BREAKTHROUGH: Hybrid TDA exceeded all targets!")
            print(f"   Recommended: Document success and prepare for deployment")
        elif results['status'] == 'PROGRESS':
            print(f"✅ STRONG PROGRESS: Close to target performance")
            print(f"   Recommended: Fine-tune ensemble and proceed to Phase 2B")
        else:
            print(f"⚠️ MODERATE PROGRESS: Continue optimization")
            print(f"   Recommended: Try temporal evolution tracking (Phase 2B)")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Document hybrid approach results")
        print(f"   2. Analyze feature importance across both TDA types")
        print(f"   3. Consider Phase 2B: Temporal Persistence Evolution")

if __name__ == "__main__":
    main()